import re
from typing import Any, Dict, List, Optional

from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.documents import Document

from src.config import AppConfig
from src.providers import ProviderFactory


def build_memory() -> ConversationBufferMemory:
    return ConversationBufferMemory(
        human_prefix="User",
        ai_prefix="Assistant",
        memory_key="history",
        return_messages=False,
    )


def build_qa_chain(
    config: AppConfig,
    retriever: Any,
    memory: Optional[ConversationBufferMemory] = None,
) -> RetrievalQAWithSourcesChain:
    """Create a RetrievalQA chain using the configured LLM provider, retriever, and optional conversation memory."""
    llm_provider = ProviderFactory.get_llm_provider(config)
    llm = llm_provider.get_llm()
    kwargs = {
        "retriever": retriever,
        "return_source_documents": True,
    }
    if memory is not None:
        kwargs["memory"] = memory

    return RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        **kwargs,
    )


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _find_relevant_sentence(text: str, keywords: List[str]) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return text.strip()[:300]

    keywords_lower = [keyword.lower() for keyword in keywords if keyword]
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(keyword in sentence_lower for keyword in keywords_lower):
            return sentence

    return sentences[0]


def build_citations(
    answer: str,
    question: str,
    source_documents: List[Document],
) -> List[str]:
    if not source_documents:
        return []

    question_terms = [token for token in re.findall(r"\w+", question.lower()) if len(token) > 3]
    answer_terms = [token for token in re.findall(r"\w+", answer.lower()) if len(token) > 3]
    keywords = list(dict.fromkeys(answer_terms + question_terms))

    citations: List[str] = []
    seen_sources = set()
    for doc in source_documents:
        source = doc.metadata.get("source", "unknown source")
        if source in seen_sources:
            continue
        seen_sources.add(source)

        title = doc.metadata.get("title") or doc.metadata.get("source") or "Untitled"
        excerpt = _find_relevant_sentence(doc.page_content, keywords)
        citation = f"[{title}]({source}) — \"{excerpt}\""
        citations.append(citation)

    return citations


def query_chain(
    chain: RetrievalQAWithSourcesChain,
    question: str,
) -> Dict[str, Any]:
    """Query the QA chain and return the enriched result dictionary."""
    response = chain({"question": question}, return_only_outputs=True)
    answer = response.get("answer", "")
    sources = response.get("sources", "")
    source_documents = response.get("source_documents", [])
    citations = build_citations(answer, question, source_documents)

    return {
        "answer": answer,
        "sources": sources,
        "citations": citations,
        "source_documents": source_documents,
        "raw": response,
    }
