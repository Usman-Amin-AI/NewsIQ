import datetime
import os
import shutil
from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS

from src.config import AppConfig


class VectorStoreBackendError(Exception):
    pass


class VectorStoreBackendInterface:
    def add_documents(self, documents: List[Document]) -> List[str]:
        raise NotImplementedError()

    def delete_ids(self, ids: List[str]) -> None:
        raise NotImplementedError()

    def save(self) -> None:
        raise NotImplementedError()

    def load(self) -> bool:
        raise NotImplementedError()

    def as_retriever(self):
        raise NotImplementedError()

    def metadata(self) -> Dict[str, str]:
        raise NotImplementedError()


class FaissBackend(VectorStoreBackendInterface):
    def __init__(self, embeddings: Embeddings, path: str) -> None:
        self.embeddings = embeddings
        self.path = path
        self.folder_path = os.path.abspath(path)
        self.index_name = "index"
        self.vectorstore: Optional[FAISS] = None

    def load(self) -> bool:
        if not os.path.isdir(self.folder_path):
            return False

        try:
            self.vectorstore = FAISS.load_local(
                folder_path=self.folder_path,
                embeddings=self.embeddings,
                index_name=self.index_name,
                allow_dangerous_deserialization=True,
            )
            return True
        except Exception as exc:
            raise VectorStoreBackendError(f"Failed to load FAISS backend: {exc}") from exc

    def save(self) -> None:
        if self.vectorstore is None:
            raise VectorStoreBackendError("No FAISS vectorstore available to save.")

        os.makedirs(self.folder_path, exist_ok=True)
        self.vectorstore.save_local(folder_path=self.folder_path, index_name=self.index_name)
        snapshot_base = os.path.join(
            os.path.dirname(self.folder_path),
            f"{os.path.basename(self.folder_path)}_snapshots",
        )
        snapshot_dir = os.path.join(
            snapshot_base,
            datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        )
        os.makedirs(snapshot_dir, exist_ok=True)
        for item in os.listdir(self.folder_path):
            source = os.path.join(self.folder_path, item)
            target = os.path.join(snapshot_dir, item)
            if os.path.isdir(source):
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target)

    def add_documents(self, documents: List[Document]) -> List[str]:
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            return []
        return self.vectorstore.add_documents(documents)

    def delete_ids(self, ids: List[str]) -> None:
        if self.vectorstore is None:
            raise VectorStoreBackendError("FAISS backend is not initialized.")
        self.vectorstore.delete(ids=ids)

    def as_retriever(self):
        if self.vectorstore is None:
            raise VectorStoreBackendError("FAISS backend is not initialized.")
        return self.vectorstore.as_retriever()

    def metadata(self) -> Dict[str, str]:
        return {"backend": "faiss"}


class ChromaBackend(VectorStoreBackendInterface):
    def __init__(
        self,
        embeddings: Embeddings,
        persist_directory: str,
        use_service: bool = False,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        collection_name: str = "newsbot",
    ) -> None:
        self.embeddings = embeddings
        self.persist_directory = os.path.abspath(persist_directory)
        self.use_service = use_service
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self.collection_name = collection_name
        self.vectorstore: Optional[Any] = None

    def _ensure_module(self):
        try:
            from langchain_community.vectorstores.chroma import Chroma
        except Exception as exc:  # noqa: E722
            raise VectorStoreBackendError(
                "Chroma backend requires chromadb to be installed. "
                "Install chromadb or choose faiss."
            ) from exc
        return Chroma

    def _build_client_settings(self):
        try:
            from chromadb.config import Settings
        except Exception as exc:  # noqa: E722
            raise VectorStoreBackendError(
                "Chroma service support requires chromadb to be installed. "
                "Install chromadb or choose another backend."
            ) from exc

        return Settings(
            chroma_api_impl="rest",
            chroma_server_host=self.chroma_host,
            chroma_server_http_port=self.chroma_port,
        )

    def _instantiate_vectorstore(self):
        Chroma = self._ensure_module()
        kwargs = {
            "collection_name": self.collection_name,
            "embedding_function": self.embeddings,
        }
        if self.use_service:
            kwargs["client_settings"] = self._build_client_settings()
        else:
            kwargs["persist_directory"] = self.persist_directory

        return Chroma(**kwargs)

    def load(self) -> bool:
        if not self.use_service and not os.path.isdir(self.persist_directory):
            return False

        try:
            self.vectorstore = self._instantiate_vectorstore()
            return True
        except Exception as exc:
            raise VectorStoreBackendError(f"Failed to load Chroma backend: {exc}") from exc

    def save(self) -> None:
        if self.vectorstore is None:
            raise VectorStoreBackendError("No Chroma vectorstore available to save.")
        if self.use_service:
            return
        self.vectorstore.persist()

    def add_documents(self, documents: List[Document]) -> List[str]:
        if self.vectorstore is None:
            self.vectorstore = self._instantiate_vectorstore()
        return self.vectorstore.add_documents(documents)

    def delete_ids(self, ids: List[str]) -> None:
        if self.vectorstore is None:
            raise VectorStoreBackendError("Chroma backend is not initialized.")
        self.vectorstore.delete(ids=ids)

    def as_retriever(self):
        if self.vectorstore is None:
            raise VectorStoreBackendError("Chroma backend is not initialized.")
        return self.vectorstore.as_retriever()

    def metadata(self) -> Dict[str, str]:
        return {"backend": "chroma", "service": str(self.use_service)}


class VectorStoreFactory:
    @staticmethod
    def create(config: AppConfig, embeddings: Embeddings, user_id: str | None = None) -> VectorStoreBackendInterface:
        backend = config.active_vectorstore_backend()
        if backend == "faiss":
            path = config.resolve_vectorstore_path(user_id or config.user_id)
            return FaissBackend(embeddings=embeddings, path=path)
        if backend == "chroma":
            path = config.resolve_vectorstore_path(user_id or config.user_id)
            collection_name = f"newsbot_{user_id or config.user_id}"
            return ChromaBackend(
                embeddings=embeddings,
                persist_directory=path,
                use_service=config.use_chroma_service,
                chroma_host=config.chroma_server_host,
                chroma_port=config.chroma_server_port,
                collection_name=collection_name,
            )
        raise VectorStoreBackendError(f"Unsupported vector store backend: {backend}")
