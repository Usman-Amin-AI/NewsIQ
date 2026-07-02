from src.store_backends import VectorStoreBackendInterface


def create_or_load_store(backend: VectorStoreBackendInterface) -> VectorStoreBackendInterface:
    """Load an existing backend if present, otherwise prepare a new backend."""
    backend.load()
    return backend


def save_store(backend: VectorStoreBackendInterface) -> None:
    backend.save()
