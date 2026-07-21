import argparse
import os

import av
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_omni_utils import process_mm_info

LOG_PATH = "transcript.log"
MODEL = "Qwen/Qwen2.5-Omni-7B"
MAX_MODEL_LEN = 32768
VIDEO_FRAMES = 8
VIDEO_MAX_PIXELS = 256 * 28 * 28
VIDEO_SEGMENT_SECONDS = 20.0
VIDEO_OVERLAP_SECONDS = 1.0
MIN_VIDEO_SEGMENT_SECONDS = 5.0
CLEANUP_MAX_TOKENS = 12000


def get_video_frame_info(video_path):
    container = av.open(video_path)
    try:
        for stream in container.streams.video:
            fps = stream.average_rate or stream.guessed_rate or stream.base_rate
            if stream.frames and fps:
                return int(stream.frames), float(fps)

        for stream in container.streams.video:
            fps = stream.average_rate or stream.guessed_rate or stream.base_rate
            if (
                stream.duration is not None
                and stream.time_base is not None
                and fps
            ):
                duration = float(stream.duration * stream.time_base)
                if duration > 0:
                    fps = float(fps)
                    return max(1, round(duration * fps)), fps

        for stream in container.streams.video:
            fps = stream.average_rate or stream.guessed_rate or stream.base_rate
            if container.duration is not None and fps:
                duration = float(container.duration / av.time_base)
                if duration > 0:
                    fps = float(fps)
                    return max(1, round(duration * fps)), fps
    finally:
        container.close()

    raise ValueError(f"Could not determine video frame info: {video_path}")


def seconds_to_frames(seconds, fps):
    return max(2, round(seconds * fps))


def frame_range_to_times(start_frame, end_frame, fps):
    frame_padding = 0.25
    segment_start = max(0.0, (start_frame - frame_padding) / fps)
    segment_end = (end_frame + frame_padding) / fps
    return segment_start, segment_end


def format_timestamp(seconds):
    minutes, remaining_seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:02d}:{remaining_seconds:02d}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Video file to transcribe")
    return parser.parse_args()


def extract_transcript(text):
    start_tag = "<transcript>"
    end_tag = "</transcript>"
    start = text.find(start_tag)
    end = text.find(end_tag, start + len(start_tag))
    if start != -1 and end != -1:
        return text[start + len(start_tag):end].strip()
    return text.strip()


def build_conversation(video_path, question, segment_start, segment_end, nframes):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "video_start": segment_start,
                    "video_end": segment_end,
                    "nframes": nframes,
                    "max_pixels": VIDEO_MAX_PIXELS,
                },
                {"type": "text", "text": question},
            ],
        },
    ]


def build_cleanup_conversation(segments):
    raw_transcript = "\n\n".join(segments)
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Clean this raw transcript into a readable continuous transcript. "
                        "The input contains consecutive transcribed video segments separated "
                        "by blank lines. Adjacent segments may overlap. Remove duplicated "
                        "overlap between segments, remove obvious slide/chart/OCR text that "
                        "does not belong to spoken audio, and keep the spoken content in order. "
                        "Do not summarize, translate, add headings, add numbering, or add tags. "
                        "Return plain text only, split into natural paragraphs.\n\n"
                        f"{raw_transcript}"
                    ),
                },
            ],
        },
    ]


def transcribe_segment(
    video_path,
    processor,
    llm,
    sampling_params,
    question,
    segment_start,
    segment_end,
    nframes,
):
    conversation = build_conversation(
        video_path,
        question,
        segment_start,
        segment_end,
        nframes,
    )
    prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    audios, _, videos = process_mm_info(
        conversation,
        use_audio_in_video=True,
    )

    inputs = {
        "prompt": prompt,
        "multi_modal_data": {
            "video": videos,
            "audio": audios,
        },
        "mm_processor_kwargs": {
            "use_audio_in_video": True,
        },
    }

    outputs = llm.generate(
        inputs,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    return outputs[0].outputs[0].text.strip()


def cleanup_transcript(
    processor,
    llm,
    sampling_params,
    segments,
):
    if not segments:
        return ""

    conversation = build_cleanup_conversation(segments)
    prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    outputs = llm.generate(
        prompt,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    paragraphs = outputs[0].outputs[0].text.strip().split("\n\n")
    return "\n\n".join(
        " ".join(paragraph.split())
        for paragraph in paragraphs
        if paragraph.strip()
    )


def is_context_length_error(error):
    message = str(error)
    return (
        "longer than the maximum model length" in message
        or "longer than the maximum sequence length" in message
    )


def video_nframes_for_segment(start_frame, end_frame):
    available_frames = end_frame - start_frame + 1
    nframes = min(VIDEO_FRAMES, available_frames)
    if nframes % 2:
        nframes -= 1
    return max(2, nframes)


def read_transcribed_segments(transcribe_path):
    with open(transcribe_path, "r", encoding="utf-8") as transcribe_file:
        return [line.strip() for line in transcribe_file if line.strip()]


def transcribe_segments(
    video_path,
    transcribe_path,
    processor,
    llm,
    sampling_params,
    question,
):
    try:
        return read_transcribed_segments(transcribe_path)
    except FileNotFoundError:
        pass

    total_frames, video_fps = get_video_frame_info(video_path)
    segment_start_frame = 0
    segment_frames = seconds_to_frames(VIDEO_SEGMENT_SECONDS, video_fps)
    overlap_frames = seconds_to_frames(VIDEO_OVERLAP_SECONDS, video_fps)
    min_segment_frames = seconds_to_frames(MIN_VIDEO_SEGMENT_SECONDS, video_fps)
    segments = []
    transcribe_temp_path = f"{transcribe_path}.tmp"

    with (
        open(transcribe_temp_path, "w", encoding="utf-8") as transcribe_file,
        open(LOG_PATH, "w", encoding="utf-8") as log_file,
    ):
        while segment_start_frame < total_frames - 1:
            segment_end_frame = min(
                segment_start_frame + segment_frames - 1,
                total_frames - 1,
            )
            if segment_end_frame <= segment_start_frame:
                break

            segment_start, segment_end = frame_range_to_times(
                segment_start_frame,
                segment_end_frame,
                video_fps,
            )
            nframes = video_nframes_for_segment(
                segment_start_frame,
                segment_end_frame,
            )
            try:
                text = transcribe_segment(
                    video_path,
                    processor,
                    llm,
                    sampling_params,
                    question,
                    segment_start,
                    segment_end,
                    nframes,
                )
            except ValueError as error:
                if (
                    is_context_length_error(error)
                    and segment_frames > min_segment_frames
                ):
                    segment_frames = max(min_segment_frames, segment_frames // 2)
                    continue
                raise

            timestamp = (
                f"[{format_timestamp(segment_start)}-"
                f"{format_timestamp(segment_end)}]"
            )
            log_file.write(f"{timestamp} {text}\n")
            log_file.flush()

            segment_text = extract_transcript(text)
            if segment_text:
                segment_text = " ".join(segment_text.split())
                segments.append(segment_text)
                transcribe_file.write(segment_text)
                transcribe_file.write("\n")
                transcribe_file.flush()

            if segment_end_frame == total_frames - 1:
                break

            segment_start_frame = max(
                segment_start_frame + 1,
                segment_end_frame + 1 - overlap_frames,
            )

    os.replace(transcribe_temp_path, transcribe_path)
    return segments


def main():
    args = parse_args()
    video_path = args.video_path
    output_path = f"{video_path}.txt"
    transcribe_path = f"{video_path}.transbribe.txt"

    processor = AutoProcessor.from_pretrained(MODEL)
    llm = LLM(
        model=MODEL,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=1,
        limit_mm_per_prompt={"video": 1, "audio": 1},
    )

    question = (
        "Transcribe only the spoken audio in this video segment. Use the video "
        "only as context; do not copy text from slides, charts, captions, or the "
        "screen. Return only the spoken words as plain text, without tags, "
        "headings, numbering, or explanations."
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=1024)
    cleanup_sampling_params = SamplingParams(
        temperature=0,
        max_tokens=CLEANUP_MAX_TOKENS,
    )
    segments = transcribe_segments(
        video_path,
        transcribe_path,
        processor,
        llm,
        sampling_params,
        question,
    )
    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write("\n[cleanup]\n")
        log_file.write(f"segments: {len(segments)}\n")

    transcript = cleanup_transcript(
        processor,
        llm,
        cleanup_sampling_params,
        segments,
    )
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(transcript)
        output_file.write("\n")


if __name__ == "__main__":
    main()