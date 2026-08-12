
Install deps
```shell
sudo apt install -y ffmpeg python3-pip
pip install uv --break-system-packages
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm==0.25.0 --torch-backend=auto 'qwen-omni-utils[decord]' audioread 'librosa==0.10.2.post1'
```

Run
```
python ./transcript.py video_file
```