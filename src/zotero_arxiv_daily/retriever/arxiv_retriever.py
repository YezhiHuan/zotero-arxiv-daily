from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
import multiprocessing
import os
from queue import Empty
from typing import Any, Callable, TypeVar
from loguru import logger
import requests
from datetime import datetime, timedelta
import time
import pickle
import hashlib
from pathlib import Path

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180
CACHE_DIR = Path("cache/arxiv")


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _cache_key(self, date_str: str) -> str:
        """生成缓存键: 日期 + 分类的 hash"""
        categories = tuple(sorted(self.config.source.arxiv.category))
        cat_hash = hashlib.md5("+".join(categories).encode()).hexdigest()[:8]
        return f"{date_str}_{cat_hash}"

    def _load_cache(self, cache_path: Path) -> list[ArxivResult] | None:
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            if not data:  # 空列表也视为缓存未命中
                logger.info(f"Cache is empty at {cache_path}, refetching")
                return None
            logger.info(f"Loaded {len(data)} cached arXiv results from {cache_path}")
            return data
        except Exception as exc:
            logger.warning(f"Failed to load cache {cache_path}: {exc}")
            return None

    def _save_cache(self, cache_path: Path, papers: list[ArxivResult]) -> None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(papers, f)
            logger.info(f"Cached {len(papers)} arXiv results to {cache_path}")
        except Exception as exc:
            logger.warning(f"Failed to save cache {cache_path}: {exc}")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        """使用 arxiv API 获取昨天一天的论文（带缓存）"""
        yesterday = datetime.now() - timedelta(days=1)
        yesterday_str = yesterday.strftime("%Y%m%d")

        # 检查缓存
        cache_path = CACHE_DIR / f"{self._cache_key(yesterday_str)}.pkl"
        cached = self._load_cache(cache_path)
        if cached is not None:
            return cached

        # 缓存未命中，请求 API（使用指数退避避免 429 限流）
        categories = self.config.source.arxiv.category
        cat_query = "+OR+".join([f"cat:{c}" for c in categories])
        date_query = f"submittedDate:[{yesterday_str}0000+TO+{yesterday_str}2359]"
        full_query = f"({cat_query})+AND+{date_query}"

        logger.info(f"arXiv API query: {full_query}")

        raw_papers = self._fetch_with_backoff(full_query)

        # 过滤出昨天的论文（防止时区边界问题）
        # 使用 getattr 安全获取 published 属性，避免测试中的 mock 对象没有该属性
        raw_papers = [
            p for p in raw_papers
            if getattr(getattr(p, 'published', None), 'date', lambda: None)() == yesterday.date()
        ]

        logger.info(f"Retrieved {len(raw_papers)} papers from arxiv API for {yesterday_str}")

        # 写入缓存
        self._save_cache(cache_path, raw_papers)

        return raw_papers

    def _fetch_with_backoff(self, query: str, max_retries: int = 5) -> list[ArxivResult]:
        """使用指数退避获取论文，避免 429 限流"""
        base_delay = 10  # 初始延迟 10 秒
        max_delay = 300  # 最大延迟 5 分钟

        for attempt in range(max_retries):
            try:
                # 遵守 arXiv 速率限制: 1请求/3秒
                if attempt > 0:
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    logger.info(f"Retry {attempt}, waiting {delay}s before request...")
                    time.sleep(delay)
                else:
                    time.sleep(3)  # 第一次请求前也等 3 秒

                client = arxiv.Client(num_retries=0)  # 我们自己处理重试
                search = arxiv.Search(
                    query=query,
                    max_results=100,  # 增加到 100
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )
                return list(client.results(search))

            except arxiv.HTTPError as e:
                if e.status_code == 429 or e.status_code == 503:
                    logger.warning(f"arXiv API returned {e.status_code}, retrying...")
                    continue
                raise
        raise RuntimeError(f"Failed to fetch from arXiv after {max_retries} retries")

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = extract_text_from_tar(raw_paper)
        if full_text is None:
            full_text = extract_text_from_html(raw_paper)
        if full_text is None:
            full_text = extract_text_from_pdf(raw_paper)
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
