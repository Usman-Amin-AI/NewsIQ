import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = ROOT_DIR / ".cache"
DEFAULT_LOG_PATH = DEFAULT_CACHE_DIR / "logs" / "app.log"
DEFAULT_METRICS_PATH = DEFAULT_CACHE_DIR / "metrics.json"


def _bool_env(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").lower()
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4.1-mini")
    vectorstore_backend: str = os.getenv("VECTORSTORE_BACKEND", "faiss").lower()
    vectorstore_path: str = os.getenv("VECTORSTORE_PATH", str(ROOT_DIR / "vectorstores"))
    sentence_transformer_model: str = os.getenv(
        "SENTENCE_TRANSFORMER_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-3.5-mini")
    chroma_server_host: str = os.getenv("CHROMA_SERVER_HOST", "localhost")
    chroma_server_port: int = int(os.getenv("CHROMA_SERVER_PORT", "8000"))
    use_chroma_service: bool = _bool_env("USE_CHROMA_SERVICE", False)
    user_id: str = os.getenv("NEWSBOT_USER_ID", "default")
    auth_cookie_name: str = os.getenv("STREAMLIT_AUTH_COOKIE_NAME", "newsbot_auth")
    auth_key: str = os.getenv("STREAMLIT_AUTH_KEY", "change-this-secret")
    auth_users: str = os.getenv("AUTH_USERS", "admin:password")
    admin_users: str = os.getenv("ADMIN_USERS", "admin")
    log_path: str = os.getenv("LOG_PATH", str(DEFAULT_LOG_PATH))
    metrics_path: str = os.getenv("METRICS_PATH", str(DEFAULT_METRICS_PATH))

    def active_embedding_provider(self) -> str:
        return self.embedding_provider

    def active_vectorstore_backend(self) -> str:
        return self.vectorstore_backend

    def active_embedding_model(self) -> str:
        return self.embedding_model

    def active_llm_provider(self) -> str:
        return self.llm_provider

    def active_llm_model(self) -> str:
        return self.llm_model

    def admin_user_list(self) -> list[str]:
        return [user.strip() for user in self.admin_users.split(",") if user.strip()]

    def auth_user_pairs(self) -> list[tuple[str, str]]:
        pairs = []
        for entry in self.auth_users.split(","):
            if ":" in entry:
                username, password = entry.split(":", 1)
                pairs.append((username.strip(), password.strip()))
        return pairs

    def resolve_vectorstore_path(self, user_id: str) -> str:
        return str(Path(self.vectorstore_path) / user_id)

    def resolve_cache_path(self, user_id: str) -> str:
        return str(DEFAULT_CACHE_DIR / user_id / "document_cache.json")

    def resolve_evaluation_log_path(self, user_id: str) -> str:
        return str(DEFAULT_CACHE_DIR / user_id / "evaluation_log.jsonl")
