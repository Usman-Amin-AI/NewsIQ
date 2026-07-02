from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.llms import OpenAI as OpenAIClient
from langchain_community.llms.anthropic import Anthropic as AnthropicClient
from langchain_community.vectorstores.faiss import FAISS

from src.config import AppConfig


class ProviderError(Exception):
    pass


class EmbeddingProviderInterface:
    def get_embeddings(self):
        raise NotImplementedError()

    def provider_name(self) -> str:
        raise NotImplementedError()

    def model_name(self) -> str:
        raise NotImplementedError()


class OpenAIEmbeddingProvider(EmbeddingProviderInterface):
    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name = model_name or "text-embedding-3-large"

    def get_embeddings(self):
        return OpenAIEmbeddings(model=self._model_name)

    def provider_name(self) -> str:
        return "openai"

    def model_name(self) -> str:
        return self._model_name


class SentenceTransformerEmbeddingProvider(EmbeddingProviderInterface):
    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"

    def get_embeddings(self):
        return SentenceTransformerEmbeddings(model_name=self._model_name)

    def provider_name(self) -> str:
        return "sentence-transformer"

    def model_name(self) -> str:
        return self._model_name


class LLMProviderInterface:
    def get_llm(self):
        raise NotImplementedError()


class OpenAIProvider(LLMProviderInterface):
    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.9, max_tokens: int = 500) -> None:
        self.model_name = model_name or "gpt-4.1-mini"
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_llm(self):
        return OpenAIClient(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


class AnthropicProvider(LLMProviderInterface):
    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.9, max_tokens: int = 500) -> None:
        self.model_name = model_name or "claude-3.5-mini"
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_llm(self):
        return AnthropicClient(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


class ProviderFactory:
    @staticmethod
    def get_embedding_provider(config: AppConfig) -> EmbeddingProviderInterface:
        provider = config.active_embedding_provider()
        model_name = config.active_embedding_model()

        if provider == "openai":
            return OpenAIEmbeddingProvider(model_name=model_name)
        if provider in {"sentence-transformer", "sentence_transformer", "local"}:
            return SentenceTransformerEmbeddingProvider(model_name=model_name)

        raise ProviderError(f"Unsupported embedding provider: {provider}")

    @staticmethod
    def get_llm_provider(config: AppConfig) -> LLMProviderInterface:
        provider = config.active_llm_provider()
        model_name = config.active_llm_model()

        if provider == "openai":
            return OpenAIProvider(model_name=model_name)
        if provider in {"anthropic", "claude"}:
            return AnthropicProvider(model_name=model_name)

        raise ProviderError(f"Unsupported LLM provider: {provider}")


def build_provider_metadata(config: AppConfig) -> Dict[str, str]:
    return {
        "embedding_provider": config.active_embedding_provider(),
        "embedding_model": config.active_embedding_model(),
    }
