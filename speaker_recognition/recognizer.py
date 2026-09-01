"""Speaker recognition logic."""

import base64
import json
import logging
from pathlib import Path

import numpy as np
import sherpa_onnx
from numpy.typing import NDArray

from speaker_recognition.const import EMBEDDING_META_SUFFIX, EMBEDDING_SUFFIX
from speaker_recognition.models import (
    AudioInput,
    Config,
    RecognitionRequest,
    RecognitionResult,
    TrainingRequest,
    TrainingResult,
    config,
)

_LOGGER = logging.getLogger(__name__)


class SpeakerRecognizer:
    """Handle speaker recognition operations."""

    def __init__(self, config: Config) -> None:
        """Initialize the speaker recognizer.

        Args:
            config: Application configuration
        """
        self._extractor: sherpa_onnx.SpeakerEmbeddingExtractor | None = None
        self._reference_embeddings: dict[str, NDArray[np.float32]] = {}
        self._is_trained = False
        self._config = config
        self._embeddings_directory = Path(config.embeddings_directory)

    @property
    def extractor(self) -> sherpa_onnx.SpeakerEmbeddingExtractor:
        """Return the embedding extractor, loading the model on first use.

        The recognizer is constructed at import time, before the CLI has
        applied the configuration, so the model cannot be loaded in __init__.
        """
        if self._extractor is None:
            model_path = Path(self._config.model_path)
            if not model_path.is_file():
                raise RuntimeError(f"Speaker embedding model not found at {model_path}")

            extractor_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(model_path),
                num_threads=self._config.num_threads,
                debug=False,
            )
            if not extractor_config.validate():
                raise RuntimeError(f"Invalid extractor config: {extractor_config}")

            _LOGGER.info(f"Loading speaker embedding model: {model_path.name}")
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(extractor_config)
            _LOGGER.info(f"Model loaded, embedding dimension: {self._extractor.dim}")

        return self._extractor

    @property
    def model_id(self) -> str:
        """Identify the model that embeddings were produced with."""
        return Path(self._config.model_path).name

    @property
    def is_trained(self) -> bool:
        """Check if the model is trained."""
        return self._is_trained

    @property
    def embeddings_directory(self) -> Path:
        """Get the embeddings directory."""
        return self._embeddings_directory

    @embeddings_directory.setter
    def embeddings_directory(self, value: str) -> None:
        """Set the embeddings directory.

        Args:
            value: New embeddings directory path
        """
        self._config.embeddings_directory = value
        self._embeddings_directory = Path(value)

    def process_audio_input(self, audio_input: AudioInput) -> NDArray[np.float32]:
        """Decode base64 PCM audio into a float32 waveform.

        Args:
            audio_input: Audio input containing base64 encoded 16-bit PCM

        Returns:
            Waveform as float32 samples in [-1, 1]
        """
        audio_bytes = base64.b64decode(audio_input.audio_data)
        audio_array_int16 = np.frombuffer(audio_bytes, dtype=np.int16).copy()

        if audio_array_int16.size == 0:
            raise ValueError("Empty audio data")

        result: NDArray[np.float32] = audio_array_int16.astype(np.float32) / 32768.0
        return result

    def embed(self, audio_input: AudioInput) -> NDArray[np.float32]:
        """Compute a unit-length embedding for one utterance.

        Embeddings are L2-normalised so that a dot product between any two of
        them is their cosine similarity, which keeps scores in [-1, 1] and lets
        a single threshold mean the same thing for every model.
        """
        waveform = self.process_audio_input(audio_input)

        stream = self.extractor.create_stream()
        stream.accept_waveform(
            sample_rate=audio_input.sample_rate, waveform=waveform
        )
        stream.input_finished()

        embedding = np.asarray(self.extractor.compute(stream), dtype=np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm == 0.0:
            raise ValueError("Model returned an empty embedding")

        return embedding / norm

    def _embedding_path(self, user_id: str) -> Path:
        return self._embeddings_directory / f"{user_id}{EMBEDDING_SUFFIX}"

    def _metadata_path(self, user_id: str) -> Path:
        return self._embeddings_directory / f"{user_id}{EMBEDDING_META_SUFFIX}"

    def _load_cached_embedding(self, user_id: str) -> NDArray[np.float32] | None:
        """Load a stored embedding, but only if the model still matches.

        An embedding from a different model is meaningless against the current
        one, so a model change discards the cache rather than silently scoring
        against incompatible vectors.
        """
        embedding_path = self._embedding_path(user_id)
        if not embedding_path.exists():
            return None

        metadata_path = self._metadata_path(user_id)
        stored_model = None
        if metadata_path.exists():
            try:
                stored_model = json.loads(metadata_path.read_text()).get("model")
            except (OSError, json.JSONDecodeError):
                stored_model = None

        if stored_model != self.model_id:
            _LOGGER.warning(
                f"Discarding embedding for {user_id}: it was made with "
                f"{stored_model or 'an unknown model'}, now using {self.model_id}. "
                "Re-enroll this speaker."
            )
            return None

        return np.asarray(np.load(embedding_path, allow_pickle=False), dtype=np.float32)

    def _store_embedding(self, user_id: str, embedding: NDArray[np.float32]) -> None:
        np.save(self._embedding_path(user_id), embedding)
        self._metadata_path(user_id).write_text(json.dumps({"model": self.model_id}))

    def train(self, request: TrainingRequest) -> TrainingResult:
        """Enroll speakers from voice samples.

        Several samples for the same speaker are averaged into one voiceprint,
        which is what makes enrollment robust — a single utterance carries the
        room, the mood and the microphone as much as it carries the voice.

        Args:
            request: Training request with voice samples

        Returns:
            TrainingResult with status, trained users and count
        """
        if not request.voice_samples:
            raise ValueError("No voice samples provided")

        self._embeddings_directory.mkdir(parents=True, exist_ok=True)

        submitted: dict[str, list[NDArray[np.float32]]] = {}
        for sample in request.voice_samples:
            try:
                submitted.setdefault(sample.user, []).append(self.embed(sample.audio))
                _LOGGER.info(f"Embedded a voice sample for user: {sample.user}")
            except Exception as error:
                _LOGGER.error(f"Error embedding sample for user {sample.user}: {error}")

        self._reference_embeddings = {}

        for user_id, embeddings in submitted.items():
            stacked = np.mean(np.stack(embeddings), axis=0)
            norm = float(np.linalg.norm(stacked))
            if norm == 0.0:
                _LOGGER.error(f"Averaged embedding for {user_id} is degenerate")
                continue

            averaged = (stacked / norm).astype(np.float32)
            self._store_embedding(user_id, averaged)
            self._reference_embeddings[user_id] = averaged
            _LOGGER.info(f"Enrolled {user_id} from {len(embeddings)} sample(s)")

        # Speakers enrolled earlier stay recognisable without resubmitting audio.
        for embedding_path in sorted(self._embeddings_directory.glob(f"*{EMBEDDING_SUFFIX}")):
            user_id = embedding_path.name[: -len(EMBEDDING_SUFFIX)]
            if user_id in self._reference_embeddings:
                continue
            cached = self._load_cached_embedding(user_id)
            if cached is not None:
                self._reference_embeddings[user_id] = cached
                _LOGGER.info(f"Loaded stored voiceprint for {user_id}")

        if not self._reference_embeddings:
            self._is_trained = False
            raise ValueError("No valid voice samples processed")

        self._is_trained = True
        _LOGGER.info(f"Training completed for {len(self._reference_embeddings)} users")
        return TrainingResult(
            status="success",
            trained_users=list(self._reference_embeddings.keys()),
            count=len(self._reference_embeddings),
        )

    def recognize(self, request: RecognitionRequest) -> RecognitionResult:
        """Recognize speaker from audio data.

        Args:
            request: Recognition request with audio input

        Returns:
            RecognitionResult with user_id, confidence, and all scores
        """
        if not self._is_trained or not self._reference_embeddings:
            raise RuntimeError("Model not trained")

        chunk_embedding = self.embed(request.audio)

        scores: dict[str, float] = {
            user_id: float(np.dot(reference_embedding, chunk_embedding))
            for user_id, reference_embedding in self._reference_embeddings.items()
        }

        if not scores:
            raise RuntimeError("No scores calculated")

        best_user = max(scores, key=lambda user: scores[user])
        best_score = scores[best_user]

        _LOGGER.debug(f"Recognition scores: {scores}")

        return RecognitionResult(
            user_id=best_user, confidence=best_score, all_scores=scores
        )


recognizer = SpeakerRecognizer(config=config)
