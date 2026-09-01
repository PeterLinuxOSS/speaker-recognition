ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base-debian:bookworm-2026.02.0
FROM ${BUILD_FROM}

# The add-on build context is this folder, so the package cannot be copied from
# the repository root — it is fetched from a pinned tag instead. Copying it in
# is what the release pipeline did, which is why installing straight from git
# failed with "COPY pyproject.toml: not found".
ARG SPEAKER_RECOGNITION_REF=v2.0.0
ARG SPEAKER_RECOGNITION_REPO=PeterLinuxOSS/speaker-recognition

# manylinux wheels (sherpa-onnx, numpy) are glibc-only, hence the Debian base.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN curl -fsSL "https://codeload.github.com/${SPEAKER_RECOGNITION_REPO}/tar.gz/refs/tags/${SPEAKER_RECOGNITION_REF}" \
    | tar -xz --strip-components=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv pip install --system --no-cache --break-system-packages ".[server]"

# The speaker embedding model ships in the image so nothing is downloaded on a
# cold start, and the add-on keeps working without internet.
ARG MODEL_NAME=nemo_en_titanet_small.onnx
RUN mkdir -p /models \
    && curl -fsSL -o "/models/${MODEL_NAME}" \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/${MODEL_NAME}" \
    && ln -s "/models/${MODEL_NAME}" /models/speaker-embedding.onnx

ENV HOST=0.0.0.0 \
    PORT=8099 \
    ACCESS_LOG=true \
    MODEL_PATH=/models/speaker-embedding.onnx \
    NUM_THREADS=2 \
    EMBEDDINGS_DIR=/share/speaker_recognition/embeddings

COPY rootfs /
RUN chmod a+x /etc/s6-overlay/s6-rc.d/speaker-recognition/run

ARG BUILD_DESCRIPTION
ARG BUILD_NAME
ARG BUILD_REF
ARG BUILD_REPOSITORY
ARG BUILD_VERSION
LABEL \
    io.hass.name="${BUILD_NAME}" \
    io.hass.description="${BUILD_DESCRIPTION}" \
    io.hass.type="addon" \
    io.hass.version="${BUILD_VERSION}" \
    org.opencontainers.image.revision="${BUILD_REF}" \
    org.opencontainers.image.licenses="MIT"

