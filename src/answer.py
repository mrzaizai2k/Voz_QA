import sys
sys.path.append("")

import os
import asyncio
import logging
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.crawler import ThreadCrawler

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

CSV_OUTPUT_DIR = Path("output")


# ---------------------------------------------------------------------------
# Base LLM class — extend this to add Anthropic, local models, etc.
# ---------------------------------------------------------------------------

class BaseLLM(ABC):
    """Abstract base for any LLM backend."""

    @abstractmethod
    async def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt and return the text response."""
        ...


class OpenAILLM(BaseLLM):
    """OpenAI-backed LLM (GPT-4o by default). Swap model for any OpenAI-compatible endpoint."""

    def __init__(self, model: str = "gpt-4o-mini", base_url: str | None = None):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment / .env file")

        # base_url lets you point at a local OpenAI-compatible server (e.g. Ollama, LM Studio)
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def chat(self, system_prompt: str, user_prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()


# Stub for Anthropic — fill in when needed
# class AnthropicLLM(BaseLLM):
#     async def chat(self, system_prompt: str, user_prompt: str) -> str:
#         import anthropic
#         client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
#         msg = await client.messages.create(
#             model="claude-opus-4-6",
#             max_tokens=1024,
#             system=system_prompt,
#             messages=[{"role": "user", "content": user_prompt}],
#         )
#         return msg.content[0].text


# ---------------------------------------------------------------------------
# Thread Q&A
# ---------------------------------------------------------------------------

class ThreadQA:
    """
    Given a VOZ thread URL and a user question:
    1. Load posts from CSV cache (output/) if available, otherwise crawl.
    2. Feed posts as context to the LLM and return an answer.
    """

    def __init__(self, llm: BaseLLM | None = None, max_pages: int | None = None):
        self.llm = llm or OpenAILLM()
        self.crawler = ThreadCrawler()
        self.max_pages = max_pages

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def answer(self, thread_url: str, question: str) -> str:
        posts_df = self._load_or_crawl(thread_url)
        if posts_df is None or posts_df.empty:
            return "Không thể lấy dữ liệu từ thread này."

        context = self._build_context(posts_df)
        return await self._ask_llm(context, question)

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _extract_thread_id(self, thread_url: str) -> str | None:
        import re
        match = re.search(r'\.(\d+)/?', thread_url)
        return match.group(1) if match else None

    def _load_or_crawl(self, thread_url: str) -> pd.DataFrame | None:
        thread_id = self._extract_thread_id(thread_url)
        if not thread_id:
            logger.error(f"Không thể trích xuất thread ID từ: {thread_url}")
            return None

        csv_path = CSV_OUTPUT_DIR / f"thread_{thread_id}_posts.csv"

        if csv_path.exists():
            logger.info(f"Tìm thấy CSV cache: {csv_path}")
            return pd.read_csv(csv_path)

        logger.info(f"Chưa có CSV, tiến hành crawl thread {thread_id} ...")
        thread_data = self.crawler.get_thread(
            thread_url,
            use_cache=True,
            max_pages=self.max_pages,
            save_data=True,   # saves CSV to output/
        )
        if not thread_data:
            return None

        if csv_path.exists():
            return pd.read_csv(csv_path)

        # Fallback: build DataFrame from in-memory data
        posts = thread_data.get("posts", [])
        return pd.DataFrame(posts) if posts else None

    def _build_context(self, df: pd.DataFrame, max_posts: int = 200) -> str:
        """Convert posts DataFrame to a readable string for the LLM."""
        rows = df.head(max_posts)
        lines = []
        for _, row in rows.iterrows():
            author = row.get("author_username", "Unknown")
            date   = row.get("created_date", "")
            text   = row.get("content_text", "")
            lines.append(f"[{date}] {author}: {text}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    async def _ask_llm(self, context: str, question: str) -> str:
        system_prompt = (
            "Bạn là trợ lý phân tích nội dung diễn đàn. "
            "Dưới đây là các bài viết từ một thread trên VOZ. "
            "Hãy trả lời câu hỏi của người dùng dựa trên nội dung này. "
            "Nếu không tìm thấy thông tin liên quan, hãy nói thẳng là không có."
        )
        user_prompt = (
            f"### Nội dung thread:\n{context}\n\n"
            f"### Câu hỏi:\n{question}"
        )
        logger.info("Đang gửi câu hỏi đến LLM ...")
        return await self.llm.chat(system_prompt, user_prompt)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def main():
    thread_url = "https://voz.vn/t/ke-ve-nhung-niem-vui-rat-doi-thuong-ma-da-lau-ban-khong-lam-nua.1228919/"
    question   = "Mọi người hay kể về những niềm vui nào nhiều nhất?"

    qa = ThreadQA(max_pages=9)
    answer = await qa.answer(thread_url, question)
    print("\n=== Câu trả lời ===")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())