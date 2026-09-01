"""Constants for speaker recognition service."""

ENV_HOST = "HOST"
ENV_PORT = "PORT"
ENV_LOG_LEVEL = "LOG_LEVEL"
ENV_ACCESS_LOG = "ACCESS_LOG"
ENV_EMBEDDINGS_DIR = "EMBEDDINGS_DIR"
ENV_MODEL_PATH = "MODEL_PATH"
ENV_NUM_THREADS = "NUM_THREADS"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8099
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_ACCESS_LOG = True
DEFAULT_EMBEDDINGS_DIR = "./embeddings"
DEFAULT_MODEL_PATH = "/models/speaker-embedding.onnx"
DEFAULT_NUM_THREADS = 2

# Embeddings are stored as .npy files named "<user>_embedding.npy". They are
# tied to the model that produced them, so a model change must not silently
# reuse them — the file records which model it came from.
EMBEDDING_SUFFIX = "_embedding.npy"
EMBEDDING_META_SUFFIX = "_embedding.json"
