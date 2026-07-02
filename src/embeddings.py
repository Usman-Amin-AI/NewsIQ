from typing import Optional

from src.config import AppConfig
from src.providers import ProviderFactory


class EmbeddingProvider:
    """Abstraction layer for runtime embedding provider configuration."""

    def __init__(self, config: AppConfig, model_name: Optional[str] = None) -> None:
        self.config = config
        self.model_name = model_name or config.active_embedding_model()
        self.provider = ProviderFactory.get_embedding_provider(config)

    def get_embeddings(self):
        return self.provider.get_embeddings()

    def provider_name(self) -> str:
        return self.provider.provider_name()

    def model_name(self) -> str:
        return self.provider.model_name()
