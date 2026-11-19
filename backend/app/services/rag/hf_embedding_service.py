import os
import logging
import math
from typing import List, Optional, Union

from app.core.config import settings

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddingService:
    """
    Serverless dense embeddings via Hugging Face API using the official huggingface_hub client.
    Generates 384-dimensional dense vector representations for LPU knowledge retrieval
    without downloading model weights locally.
    Default models:
    - BAAI/bge-small-en-v1.5 (384-d)
    - sentence-transformers/all-MiniLM-L6-v2 (384-d)
    """

    _client = None

    @classmethod
    def get_token(cls) -> str:
        return (
            os.environ.get("HF_TOKEN")
            or settings.HF_TOKEN
            or settings.HF_TOKEN_READ
            or ""
        ).strip()

    @classmethod
    def get_client(cls):
        if cls._client is not None:
            return cls._client

        token = cls.get_token()
        if not token:
            logger.warning("HF_TOKEN is not configured. Serverless embeddings unavailable.")
            return None

        try:
            from huggingface_hub import InferenceClient
            cls._client = InferenceClient(
                provider="hf-inference",
                api_key=token,
            )
            return cls._client
        except Exception as e:
            logger.warning(f"Failed to initialize huggingface_hub InferenceClient: {e}")
            return None

    @classmethod
    def is_available(cls) -> bool:
        return bool(cls.get_token())

    @classmethod
    def generate_embeddings(
        cls,
        texts: Union[str, List[str]],
        model: Optional[str] = None
    ) -> Optional[List]:
        """
        Generates 384-d dense embeddings for text or list of texts.
        Returns list of lists (pure Python, no numpy).
        """
        input_list = [texts] if isinstance(texts, str) else texts
        if not input_list:
            return None

        client = cls.get_client()
        if not client:
            return None

        target_model = model or settings.HF_EMBEDDING_MODEL or "BAAI/bge-small-en-v1.5"

        try:
            raw_embeddings = client.feature_extraction(
                input_list,
                model=target_model,
            )

            def to_list_2d(data):
                """Convert nested structure to list of float lists (2D)."""
                if not data:
                    return []
                # Already list of lists
                if isinstance(data[0], (list, tuple)):
                    # Check if 3D (token-level): average pool
                    if data[0] and isinstance(data[0][0], (list, tuple)):
                        result = []
                        for row in data:
                            n = len(row[0])
                            pooled = [sum(row[t][d] for t in range(len(row))) / len(row) for d in range(n)]
                            result.append(pooled)
                        return result
                    return [list(row) for row in data]
                # 1D single embedding
                return [list(data)]

            arr = to_list_2d(raw_embeddings)

            def normalize_row(row):
                norm = math.sqrt(sum(x * x for x in row))
                if norm == 0:
                    return row
                return [x / norm for x in row]

            norm_arr = [normalize_row(row) for row in arr]

            if isinstance(texts, str):
                return norm_arr[0] if norm_arr else None
            return norm_arr

        except Exception as e:
            logger.debug(f"HF serverless embedding call encountered: {e}. Graceful fallback available.")
            return None

