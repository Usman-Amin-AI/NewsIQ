import asyncio
import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup
from langchain_core.documents import Document

try:
    import trafilatura
    _TRAFILATURA_AVAILABLE = True
except ImportError:
    trafilatura = None  # type: ignore[var-annotated]
    _TRAFILATURA_AVAILABLE = False


@dataclass
class UrlLoadResult:
    url: str
    status: str
    document: Optional[Document] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.document is not None and self.status == "success"


def _is_valid_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    return bool(parsed.scheme in {"http", "https"} and parsed.netloc)


def validate_urls(urls: Iterable[str]) -> List[str]:
    """Normalize a list of URL strings by stripping whitespace and validating format."""
    return [url.strip() for url in urls if url and _is_valid_url(url)]


def _extract_primary_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "aside", "nav"]):
        tag.decompose()

    article = soup.find("article")
    if article:
        text = article.get_text(separator="\n", strip=True)
        if len(text) > 200:
            return text

    main = soup.find("main")
    if main:
        text = main.get_text(separator="\n", strip=True)
        if len(text) > 200:
            return text

    candidates = soup.find_all(["section", "div", "article", "main"])
    best_text = ""
    for candidate in candidates:
        text = candidate.get_text(separator="\n", strip=True)
        if len(text) > len(best_text):
            best_text = text

    return best_text


def _extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    heading = soup.find(["h1", "h2"])
    if heading and heading.get_text(strip=True):
        return heading.get_text(strip=True)
    return "Untitled"


def _extract_text(html: str) -> str:
    primary_text = _extract_primary_text(html)
    if primary_text and len(primary_text) > 200:
        return primary_text

    if _TRAFILATURA_AVAILABLE:
        fallback_text = trafilatura.extract(
            html,
            output_format="text",
            include_comments=False,
            include_tables=False,
        )
        if fallback_text and len(fallback_text.strip()) > 100:
            return fallback_text.strip()

    raise ValueError("Unable to extract usable text from page HTML.")


def _compute_content_hash(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


async def _fetch_html(url: str, session: aiohttp.ClientSession) -> Tuple[int, str]:
    async with session.get(url, ssl=False) as response:
        status = response.status
        html = await response.text(errors="replace")
        return status, html


async def _process_url(
    url: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> UrlLoadResult:
    async with semaphore:
        try:
            status_code, html = await _fetch_html(url, session)
            if status_code >= 400:
                return UrlLoadResult(
                    url=url,
                    status=f"HTTP {status_code}",
                    error=f"Server returned status {status_code}",
                )

            extracted_text = _extract_text(html)
            content_hash = _compute_content_hash(extracted_text)
            title = _extract_title(html)
            document = Document(
                page_content=extracted_text,
                metadata={
                    "source": url,
                    "content_hash": content_hash,
                    "title": title,
                },
            )
            return UrlLoadResult(url=url, status="success", document=document)
        except asyncio.TimeoutError:
            return UrlLoadResult(url=url, status="timeout", error="Timeout while fetching URL")
        except aiohttp.ClientResponseError as exc:
            return UrlLoadResult(url=url, status=f"HTTP {exc.status}", error=str(exc))
        except aiohttp.ClientError as exc:
            return UrlLoadResult(url=url, status="fetch error", error=str(exc))
        except Exception as exc:
            return UrlLoadResult(url=url, status="extract error", error=str(exc))


async def _load_urls_async(urls: List[str], concurrency: int = 20) -> List[UrlLoadResult]:
    timeout = aiohttp.ClientTimeout(total=20)
    headers = {
        "User-Agent": "NewsBot/1.0 (+https://example.com)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        semaphore = asyncio.Semaphore(min(concurrency, len(urls)))
        tasks = [
            _process_url(url, session, semaphore)
            for url in urls
        ]
        return await asyncio.gather(*tasks)


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError as exc:
        if "asyncio.run() cannot be called from a running event loop" in str(exc):
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)
            finally:
                asyncio.set_event_loop(None)
        raise


def load_urls(urls: Iterable[str]) -> List[UrlLoadResult]:
    cleaned_urls = [(url or "").strip() for url in urls if (url or "").strip()]
    if not cleaned_urls:
        raise ValueError("No URLs provided.")

    invalid_results: List[UrlLoadResult] = []
    valid_urls: List[str] = []
    for url in cleaned_urls:
        if _is_valid_url(url):
            valid_urls.append(url)
        else:
            invalid_results.append(
                UrlLoadResult(url=url, status="invalid", error="Invalid URL format")
            )

    results: List[UrlLoadResult] = invalid_results
    if valid_urls:
        results.extend(_run_async(_load_urls_async(valid_urls)))
    return results
