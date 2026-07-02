from typing import Iterable, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

DEFAULT_SEPARATORS = ["\n\n", "\n", ".", ","]


def split_documents(
    documents: Iterable[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 0,
    separators: Optional[List[str]] = None,
) -> List[Document]:
    """Split documents into chunks using a recursive text splitter.

    Args:
        documents: An iterable of LangChain Documents.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Number of overlapping characters between chunks.
        separators: Optional list of separators to control splitting behavior.

    Returns:
        A list of chunked LangChain Documents.
    """
    if separators is None:
        separators = DEFAULT_SEPARATORS

    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(list(documents))
