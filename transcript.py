from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_omni_utils import process_mm_info


VIDEO_PATH = "my-video.mkv"
MODEL = "Qwen/Qwen2.5-Omni-7B"
VIDEO_FRAMES = 8
VIDEO_MAX_PIXELS = 256 * 28 * 28


def main():
    processor = AutoProcessor.from_pretrained(MODEL)
    llm = LLM(
        model=MODEL,
        max_model_len=8192,
        max_num_seqs=5,
        limit_mm_per_prompt={"video": 1, "audio": 1},
    )

    question = (
        "Transcribe the speech in this video. "
        "Use the visual context only to resolve ambiguous words."
    )
    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are Qwen, a virtual human developed by the Qwen "
                        "Team, Alibaba Group, capable of perceiving auditory "
                        "and visual inputs, as well as generating text and "
                        "speech."
                    ),
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": VIDEO_PATH,
                    "nframes": VIDEO_FRAMES,
                    "max_pixels": VIDEO_MAX_PIXELS,
                },
                {"type": "text", "text": question},
            ],
        },
    ]
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
    sampling_params = SamplingParams(temperature=0, max_tokens=512)

    outputs = llm.generate(
        inputs,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    with open("output.txt", "w", encoding="utf-8") as output_file:
        output_file.write(outputs[0].outputs[0].text.strip())
        output_file.write("\n")


if __name__ == "__main__":
    main()