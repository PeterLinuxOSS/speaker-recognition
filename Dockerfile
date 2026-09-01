FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY speaker_recognition ./speaker_recognition

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN uv pip install --system --no-cache -e ".[server]"

ARG MODEL_NAME=nemo_en_titanet_small.onnx
RUN mkdir -p /models && \
    curl -fsSL -o "/models/${MODEL_NAME}" \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/${MODEL_NAME}" && \
    ln -s "/models/${MODEL_NAME}" /models/speaker-embedding.onnx

RUN mkdir -p /data/embeddings

EXPOSE 8099

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8099/health').read()"

ENV HOST=0.0.0.0 \
    PORT=8099 \
    LOG_LEVEL=INFO \
    ACCESS_LOG=true \
    MODEL_PATH=/models/speaker-embedding.onnx \
    EMBEDDINGS_DIR=/data/embeddings

CMD ["python", "-m", "speaker_recognition"]
