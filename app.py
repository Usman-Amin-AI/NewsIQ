import os
import time

import streamlit as st
from dotenv import load_dotenv

from src.auth import authenticate_user, build_user_session
from src.chunking import split_documents
from src.config import AppConfig
from src.document_cache import DocumentCache
from src.embeddings import EmbeddingProvider
from src.evaluation import evaluate_answer, log_evaluation
from src.ingestion import load_urls
from src.logging_utils import configure_logger, record_event
from src.metrics import record_query, summarize_metrics
from src.providers import build_provider_metadata
from src.qa_chain import build_memory, build_qa_chain, query_chain
from src.store_backends import VectorStoreFactory
from src.vectorstore import create_or_load_store, save_store

load_dotenv()

BASE_CONFIG = AppConfig()
LOGGER = configure_logger(BASE_CONFIG)

st.set_page_config(page_title="NewsBot: News Research Tool 📈", page_icon="📰")

if "user_session" not in st.session_state:
    st.session_state.user_session = None

if "url_inputs" not in st.session_state:
    st.session_state.url_inputs = [""]


def render_login() -> None:
    st.sidebar.header("Sign in")
    username = st.sidebar.text_input("Username", key="auth_username")
    password = st.sidebar.text_input("Password", type="password", key="auth_password")
    if st.sidebar.button("Sign in"):
        if authenticate_user(BASE_CONFIG, username, password):
            st.session_state.user_session = build_user_session(BASE_CONFIG, username)
            st.sidebar.success(f"Signed in as {st.session_state.user_session['username']}")
            return
        st.sidebar.error("Invalid username or password.")


user_session = st.session_state.user_session
if user_session is None:
    st.title("NewsBot: Please sign in")
    st.write("You must sign in to use NewsBot. Configure credentials in `AUTH_USERS` before launching the app.")
    render_login()
    st.stop()

config = AppConfig(
    embedding_provider=os.getenv("EMBEDDING_PROVIDER", BASE_CONFIG.embedding_provider),
    embedding_model=os.getenv("EMBEDDING_MODEL", BASE_CONFIG.embedding_model),
    llm_provider=os.getenv("LLM_PROVIDER", BASE_CONFIG.llm_provider),
    llm_model=os.getenv("LLM_MODEL", BASE_CONFIG.llm_model),
    vectorstore_backend=os.getenv("VECTORSTORE_BACKEND", BASE_CONFIG.vectorstore_backend),
    vectorstore_path=os.getenv("VECTORSTORE_PATH", BASE_CONFIG.vectorstore_path),
    sentence_transformer_model=BASE_CONFIG.sentence_transformer_model,
    anthropic_model=BASE_CONFIG.anthropic_model,
    chroma_server_host=BASE_CONFIG.chroma_server_host,
    chroma_server_port=BASE_CONFIG.chroma_server_port,
    use_chroma_service=BASE_CONFIG.use_chroma_service,
    user_id=user_session["user_id"],
    auth_cookie_name=BASE_CONFIG.auth_cookie_name,
    auth_key=BASE_CONFIG.auth_key,
    auth_users=BASE_CONFIG.auth_users,
    admin_users=BASE_CONFIG.admin_users,
    log_path=BASE_CONFIG.log_path,
    metrics_path=BASE_CONFIG.metrics_path,
)

st.sidebar.markdown(f"**Signed in as:** {user_session['username']}  ")
if st.sidebar.button("Sign out"):
    for key in [
        "user_session",
        "auth_username",
        "auth_password",
        "url_inputs",
        "memory",
        "last_query_result",
    ]:
        if key in st.session_state:
            del st.session_state[key]
    st.experimental_rerun()

available_tabs = ["Query"]
if user_session.get("is_admin"):
    available_tabs.append("Admin")

selected_tab = st.sidebar.radio("Mode", available_tabs)

st.title("NewsBot: News Research Tool 📈📰🌐📊🔍")
st.sidebar.title("News Article URLs 📎🗞️🔗🌍📰")

if selected_tab == "Admin":
    st.header("Admin Dashboard")
    st.markdown("Manage per-user metrics, authorization, and service-backed storage.")

    metric_summary = summarize_metrics(config)
    if metric_summary:
        st.subheader("Daily Metrics")
        for day, values in sorted(metric_summary.items(), reverse=True):
            st.markdown(
                f"**{day}** — Queries: {values['queries']} | Avg latency: {values['avg_latency_ms']} ms | Token spend: {values['token_spend']}"
            )
            if values.get("users"):
                user_lines = [f"{identity}: {count}" for identity, count in values["users"].items()]
                st.markdown("- " + "\n- ".join(user_lines))
    else:
        st.info("No metrics recorded yet.")

    st.subheader("Config overview")
    st.write({
        "usernames": [username for username, _ in config.auth_user_pairs()],
        "admin_users": config.admin_user_list(),
        "vectorstore_backend": config.vectorstore_backend,
        "use_chroma_service": config.use_chroma_service,
        "chroma_server": f"{config.chroma_server_host}:{config.chroma_server_port}",
    })

    log_path = config.log_path
    if os.path.exists(log_path):
        st.subheader("Recent event log")
        with open(log_path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()[-20:]
        for line in lines:
            st.text(line.strip())
    else:
        st.info("Event log not yet created.")

    st.stop()

st.sidebar.markdown("### Enter one or more news article URLs to analyze: 📰🌐🔗📎🗞️")

embedding_provider = st.sidebar.selectbox(
    "Embedding provider:",
    options=["openai", "sentence-transformer"],
    index=["openai", "sentence-transformer"].index(config.embedding_provider),
)
embedding_model = st.sidebar.text_input(
    "Embedding model:", value=config.embedding_model
)
llm_provider = st.sidebar.selectbox(
    "LLM provider:",
    options=["openai", "anthropic"],
    index=["openai", "anthropic"].index(config.llm_provider),
)
llm_model = st.sidebar.text_input(
    "LLM model:", value=config.llm_model
)
vectorstore_backend = st.sidebar.selectbox(
    "Vector store backend:",
    options=["faiss", "chroma"],
    index=["faiss", "chroma"].index(config.vectorstore_backend),
)
vectorstore_path = st.sidebar.text_input(
    "Vector store path or persist directory:", value=config.vectorstore_path
)

bulk_urls = st.sidebar.text_area(
    "Paste multiple URLs (one per line):",
    height=120,
    placeholder="https://example.com/article1\nhttps://example.com/article2\n...",
)

if bulk_urls:
    pasted = [line.strip() for line in bulk_urls.splitlines() if line.strip()]
    if pasted:
        st.session_state.url_inputs = pasted

for idx in range(len(st.session_state.url_inputs)):
    col1, col2 = st.sidebar.columns([8, 1])
    st.session_state.url_inputs[idx] = col1.text_input(
        f"URL {idx+1}", value=st.session_state.url_inputs[idx], key=f"url_{idx}", placeholder="Enter URL here..."
    )
    if col2.button("✕", key=f"remove_{idx}"):
        st.session_state.url_inputs.pop(idx)
        st.experimental_rerun()
        break

if st.sidebar.button("Add another URL"):
    st.session_state.url_inputs.append("")

process_button = st.sidebar.button("Process URLs")
status_results = None

if process_button:
    url_inputs = [url for url in st.session_state.url_inputs if url.strip()]
    if not url_inputs:
        st.sidebar.error("Please enter at least one valid URL. 🌐📎🗞️🔗📥")
    else:
        with st.spinner("Loading data from URLs... 🌐🔄📥📰📎"):
            status_results = load_urls(url_inputs)

        successes = [result for result in status_results if result.success]
        failures = [result for result in status_results if not result.success]

        if successes:
            config = AppConfig(
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                llm_provider=llm_provider,
                llm_model=llm_model,
                vectorstore_backend=vectorstore_backend,
                vectorstore_path=vectorstore_path,
                sentence_transformer_model=config.sentence_transformer_model,
                anthropic_model=config.anthropic_model,
                chroma_server_host=config.chroma_server_host,
                chroma_server_port=config.chroma_server_port,
                use_chroma_service=config.use_chroma_service,
                user_id=user_session["user_id"],
                auth_cookie_name=config.auth_cookie_name,
                auth_key=config.auth_key,
                auth_users=config.auth_users,
                admin_users=config.admin_users,
                log_path=config.log_path,
                metrics_path=config.metrics_path,
            )

            cache_path = config.resolve_cache_path(user_session["user_id"])
            cache = DocumentCache(cache_path)
            new_documents = []
            for result in successes:
                if result.document is None:
                    continue
                source = result.document.metadata.get("source")
                content_hash = result.document.metadata.get("content_hash")
                if source and content_hash and cache.should_skip(source, content_hash):
                    continue
                new_documents.append(result.document)

            if new_documents:
                with st.spinner("Splitting text into chunks..."):
                    chunked_docs = split_documents(new_documents)

                with st.spinner("Creating embeddings and updating vector store..."):
                    embeddings = EmbeddingProvider(config).get_embeddings()
                    vectorstore_backend = VectorStoreFactory.create(config, embeddings, user_session["user_id"])
                    create_or_load_store(vectorstore_backend)
                    vectorstore_backend.add_documents(chunked_docs)
                    save_store(vectorstore_backend)

                for result in successes:
                    source = result.document.metadata.get("source") if result.document else None
                    content_hash = result.document.metadata.get("content_hash") if result.document else None
                    if source and content_hash:
                        cache.set(source, content_hash, build_provider_metadata(config))

                record_event(
                    config,
                    "url_ingestion",
                    {
                        "user_id": user_session["user_id"],
                        "processed_urls": len(new_documents),
                        "failed_urls": len(failures),
                    },
                )

                st.sidebar.success("URLs processed successfully! ✅📎🌐📰🔍")
            else:
                st.sidebar.info("All submitted URLs are already indexed with unchanged content. ✅")
        elif failures:
            st.sidebar.warning("No URLs were successfully loaded. Please review errors below.")

        st.sidebar.markdown("### URL fetch status")
        for result in status_results:
            status_icon = "✅" if result.success else "⚠️"
            st.sidebar.markdown(
                f"**{status_icon} {result.url}** — {result.status}"
            )
            if result.error:
                st.sidebar.write(f"- {result.error}")

question = st.text_input("Enter your question here: ❓🖊️📥🌐📋")
if question:
    if "memory" not in st.session_state:
        st.session_state.memory = build_memory()

    config = AppConfig(
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        llm_provider=llm_provider,
        llm_model=llm_model,
        vectorstore_backend=vectorstore_backend,
        vectorstore_path=vectorstore_path,
        sentence_transformer_model=config.sentence_transformer_model,
        anthropic_model=config.anthropic_model,
        chroma_server_host=config.chroma_server_host,
        chroma_server_port=config.chroma_server_port,
        use_chroma_service=config.use_chroma_service,
        user_id=user_session["user_id"],
        auth_cookie_name=config.auth_cookie_name,
        auth_key=config.auth_key,
        auth_users=config.auth_users,
        admin_users=config.admin_users,
        log_path=config.log_path,
        metrics_path=config.metrics_path,
    )

    try:
        embeddings = EmbeddingProvider(config).get_embeddings()
        vectorstore_backend_instance = VectorStoreFactory.create(config, embeddings, user_session["user_id"])
        if not vectorstore_backend_instance.load():
            raise FileNotFoundError("Vector store not found. Please process URLs first.")
        retriever = vectorstore_backend_instance.as_retriever()
        chain = build_qa_chain(config, retriever, memory=st.session_state.memory)
        start_time = time.monotonic()
        result = query_chain(chain, question)
        latency_ms = round((time.monotonic() - start_time) * 1000, 2)
        record_query(config, user_session["user_id"], latency_ms, 0.0)
        record_event(
            config,
            "query_executed",
            {
                "user_id": user_session["user_id"],
                "question": question,
                "latency_ms": latency_ms,
            },
        )
        st.session_state.last_query_result = result
    except Exception as exc:
        st.error(str(exc))
        result = None

    if result:
        st.header("Answer")
        st.write(result.get("answer", ""))

        citations = result.get("citations", [])
        if citations:
            st.subheader("Citations:")
            for citation in citations:
                st.write(citation)

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)

        with st.expander("Conversation history"):
            memory_vars = st.session_state.memory.load_memory_variables({})
            history = memory_vars.get("history") if isinstance(memory_vars, dict) else None
            if history:
                st.write(history)
            else:
                st.write("No conversation memory available yet.")

        evaluation = evaluate_answer(
            question,
            result.get("answer", ""),
            result.get("citations", []),
            result.get("source_documents", []),
        )

        st.subheader("Groundedness Evaluation")
        st.write(f"Score: {evaluation['groundedness_score']} \n\n{evaluation['summary']}")
        st.write(f"Source count: {evaluation['source_count']}")
        st.write(f"Citation count: {evaluation['citation_count']}")

        if st.button("Log evaluation result"):
            log_path = config.resolve_evaluation_log_path(user_session["user_id"])
            log_evaluation(evaluation, log_path)
            st.success(f"Evaluation logged to {log_path}")
