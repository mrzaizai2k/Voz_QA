import sys
sys.path.append("")

import json
import logging
import asyncio
import aiohttp
import re
import time
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
from src.utils import check_path

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

THREAD_CACHE_DIR = Path("cache")
CSV_OUTPUT_DIR = Path("output")
VOZ_BASE_URL = "https://voz.vn"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://voz.vn/",
}

MAX_CONCURRENT_REQUESTS = 5   # max parallel page fetches
REQUEST_DELAY = 0.5           # seconds between each request (polite crawling)


class AsyncThreadCrawler:
    """
    Async crawler for VOZ threads.
    Pages 2..N are fetched concurrently (bounded by a semaphore).
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_thread(
        self,
        thread_url: str,
        use_cache: bool = True,
        max_pages: int | None = None,
        save_data: bool = False,
    ) -> dict | None:
        thread_id = self._extract_thread_id(thread_url)
        if not thread_id:
            logger.error(f"Không thể trích xuất thread ID từ URL: {thread_url}")
            return None

        # Check cache
        cache_file = THREAD_CACHE_DIR / f"{thread_id}.json"
        if use_cache and cache_file.exists():
            logger.info(f"Sử dụng cache cho thread {thread_id}")
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    thread_data = json.load(f)
                if max_pages is None or len(thread_data.get("pages", [])) >= max_pages:
                    return thread_data
                logger.info(
                    f"Cache không đầy đủ, tiếp tục crawl từ trang "
                    f"{len(thread_data.get('pages', [])) + 1}"
                )
            except Exception as e:
                logger.error(f"Lỗi khi đọc cache: {e}")

        # Run async crawl
        thread_data = await self._crawl_thread(thread_url, thread_id, max_pages)
        if thread_data:
            self._save_cache(thread_data, cache_file)
            if save_data:
                self._save_to_csv(thread_data, thread_id)

        return thread_data

    # ------------------------------------------------------------------
    # Core async crawl
    # ------------------------------------------------------------------

    async def _crawl_thread(
        self, thread_url: str, thread_id: str, max_pages: int | None
    ) -> dict | None:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        async with aiohttp.ClientSession(headers=HEADERS) as session:
            # Fetch page 1 first to learn total_pages and thread title
            html = await self._fetch(session, thread_url, semaphore)
            if not html:
                return None

            soup = BeautifulSoup(html, "lxml")
            thread_title = self._parse_title(soup)
            total_pages = self._parse_total_pages(soup)

            logger.info(f"Thread {thread_id} có {total_pages} trang")

            if max_pages is not None:
                total_pages = min(total_pages, max_pages)
                logger.info(f"Giới hạn crawl {total_pages} trang")

            thread_data = {
                "thread_id": thread_id,
                "title": thread_title,
                "url": thread_url,
                "crawl_date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "total_pages": total_pages,
                "pages": [],
            }

            # Page 1 result (already fetched)
            page1_posts = self._parse_posts(soup)
            # We'll fill pages list after gathering all pages, to keep order

            # Build tasks for pages 2..N
            async def fetch_page(page_num: int) -> tuple[int, list]:
                url = f"{thread_url.rstrip('/')}/page-{page_num}"
                await asyncio.sleep(REQUEST_DELAY * (page_num % MAX_CONCURRENT_REQUESTS))
                html = await self._fetch(session, url, semaphore)
                if not html:
                    logger.error(f"Không thể crawl trang {page_num}")
                    return page_num, []
                posts = self._parse_posts(BeautifulSoup(html, "lxml"))
                logger.info(f"Đã crawl trang {page_num}/{total_pages}")
                return page_num, posts

            tasks = [fetch_page(p) for p in range(2, total_pages + 1)]
            results = await asyncio.gather(*tasks)

        # Assemble pages in order
        all_pages = [(1, page1_posts)] + list(results)
        all_pages.sort(key=lambda x: x[0])

        for page_num, posts in all_pages:
            thread_data["pages"].append({"page_number": page_num, "posts": posts})

        all_posts = [post for _, posts in all_pages for post in posts]
        thread_data["posts"] = all_posts
        thread_data["post_count"] = len(all_posts)

        logger.info(f"Hoàn thành: {len(all_posts)} bài viết từ {total_pages} trang")
        return thread_data

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    async def _fetch(
        self,
        session: aiohttp.ClientSession,
        url: str,
        semaphore: asyncio.Semaphore,
        retries: int = 3,
    ) -> str | None:
        async with semaphore:
            for attempt in range(1, retries + 1):
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        if resp.status == 200:
                            return await resp.text()
                        logger.warning(
                            f"[{resp.status}] {url} (attempt {attempt}/{retries})"
                        )
                        if resp.status in (429, 503):
                            await asyncio.sleep(2 ** attempt)  # exponential back-off
                except aiohttp.ClientError as e:
                    logger.warning(f"Request error {url}: {e} (attempt {attempt}/{retries})")
                    await asyncio.sleep(1)
            logger.error(f"Failed after {retries} attempts: {url}")
            return None

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _extract_thread_id(self, url: str) -> str | None:
        match = re.search(r'\.(\d+)/?', url)
        return match.group(1) if match else None

    def _parse_title(self, soup: BeautifulSoup) -> str:
        el = soup.select_one(".p-title-value")
        return el.text.strip() if el else ""

    def _parse_total_pages(self, soup: BeautifulSoup) -> int:
        nav = soup.select_one("nav.pageNavWrapper")
        if nav:
            page_links = nav.select(".pageNav-page a")
            nums = [int(a.text.strip()) for a in page_links if a.text.strip().isdigit()]
            if nums:
                return max(nums)
        return 1

    def _parse_posts(self, soup: BeautifulSoup) -> list[dict]:
        posts = []
        for el in soup.select(".block-body article.message"):
            post_id = el.get("data-content", "").replace("post-", "") or None
            u_el = el.select_one(".message-userDetails .username")
            username = u_el.text.strip() if u_el else "Unknown"
            user_id = u_el.get("data-user-id") if u_el else None
            t_el = el.select_one(".message-attribution-main time")
            created_date = t_el.get("datetime", "") if t_el else ""
            m_el = el.select_one(".message-lastEdit time")
            modified_date = m_el.get("datetime", "") if m_el else None
            c_el = el.select_one(".message-body .bbWrapper")
            content = c_el.text.strip() if c_el else ""
            content = content.replace("Click to expand...", "").strip()
            posts.append({
                "post_id": post_id,
                "author_username": username,
                "author_user_id": user_id,
                "created_date": created_date,
                "modified_date": modified_date,
                "content_text": content.replace("\n", " ").replace("\r", ""),
            })
        return posts

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_cache(self, thread_data: dict, cache_file: Path) -> bool:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(thread_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Đã lưu cache cho thread {thread_data['thread_id']}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi lưu cache: {e}")
            return False

    def _save_to_csv(self, thread_data: dict, thread_id: str) -> None:
        check_path(CSV_OUTPUT_DIR)
        posts = thread_data.get("posts", [])
        if not posts:
            logger.warning("Không có bài viết nào để lưu vào CSV.")
            return
        df = pd.DataFrame(posts)
        csv_path = CSV_OUTPUT_DIR / f"thread_{thread_id}_posts.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"Đã lưu bài viết vào CSV: {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main():
    test_url = "https://voz.vn/t/ke-ve-nhung-niem-vui-rat-doi-thuong-ma-da-lau-ban-khong-lam-nua.1228919/"

    crawler = AsyncThreadCrawler()
    start = time.perf_counter()
    thread_data = await crawler.get_thread(test_url, use_cache=False, max_pages=None, save_data=True)
    elapsed = time.perf_counter() - start

    if thread_data:
        print(f"Thread: {thread_data['title']}")
        print(f"Số bài viết: {thread_data['post_count']}")
        print(f"Thời gian crawl: {elapsed:.2f}s")
    print("Crawl complete.")


if __name__ == "__main__":
    asyncio.run(main())