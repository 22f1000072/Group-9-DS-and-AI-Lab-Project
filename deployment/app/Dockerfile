# Noise-robust transcription assistant (CPU). Build from the repository root:
#   docker build -f deployment/app/Dockerfile -t g9-asr-assistant .   &&   docker run -p 7860:7860 g9-asr-assistant
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake libsndfile1 ffmpeg curl \
        libboost-program-options-dev libboost-system-dev libboost-thread-dev libboost-test-dev zlib1g-dev libbz2-dev liblzma-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# CPU-only PyTorch, then the pinned project requirements (pyctcdecode without its outdated numpy pin)
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && grep -v -E '^(torch==|--extra-index-url)' requirements.txt > req.txt \
    && pip install --no-cache-dir -r req.txt && pip install --no-cache-dir --no-deps pyctcdecode==0.5.0
COPY src/ src/
COPY deployment/app/ deployment/app/
COPY experiments/results/B960_rev_lm3/error_detector/model.joblib experiments/results/B960_rev_lm3/error_detector/model.joblib
RUN mkdir -p data/raw/librispeech_lm && curl -L https://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz \
        | gunzip > data/raw/librispeech_lm/3-gram.pruned.1e-7.arpa
ENV APP_MODEL=kumargulshan/wav2vec2-base-ls960-noise-robust APP_HOST=0.0.0.0 APP_PORT=7860 APP_THREADS=4 \
    HF_HOME=/app/data/hf_home PYTHONPATH=/app/src
RUN python -c "from transformers import Wav2Vec2ForCTC; Wav2Vec2ForCTC.from_pretrained('$APP_MODEL')"
EXPOSE 7860
CMD ["python", "deployment/app/app.py"]
