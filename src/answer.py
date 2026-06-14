import sys
sys.path.append("")

import os
import re
import asyncio
import logging
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI
import json

from src.crawler import AsyncThreadCrawler
from src.prompts import SYSTEM_PROMPT, INTENT_HINTS

from src.router import detect_intent

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

CSV_OUTPUT_DIR = Path("output")


async def build_prompt(
        llm,
        context,
        question,
        post_count
    ):

        intent = await detect_intent(
            llm,
            question
        )

        hint = INTENT_HINTS.get(intent, "")

        meta = f"""
    Intent: {intent}

    Posts supplied: {post_count}
    """

        return (
            meta
            + "\n\n"
            + hint
            + "\n\n"
            + "### THREAD\n"
            + context
            + "\n\n"
            + "### USER QUESTION\n"
            + question
        )

# ---------------------------------------------------------------------------
# Base LLM — stream() yields chunks; chat() collects them for convenience
# ---------------------------------------------------------------------------

class BaseLLM(ABC):
    """
    Abstract base for all LLM backends.

    Subclasses must implement `stream()`.
    `chat()` is provided for free — it simply collects all chunks.
    """

    @abstractmethod
    async def stream(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> AsyncIterator[str]:
        # `yield` here tells Python this is an async generator signature.
        # Subclasses override with real `yield` statements.
        yield  # pragma: no cover

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Collect the full streamed response into a single string."""
        chunks: list[str] = []
        async for chunk in self.stream(system_prompt, user_prompt):
            chunks.append(chunk)
        return "".join(chunks)


# ---------------------------------------------------------------------------
# Shared helper — works for any AsyncOpenAI client
# ---------------------------------------------------------------------------

async def _openai_stream(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
) -> AsyncIterator[str]:
    """
    Thin wrapper around the AsyncOpenAI streaming API.
    Re-used by every LLM subclass so there is exactly ONE place
    that knows how to drive the SDK.
    """
    response = await client.chat.completions.create(
        model=model,
        temperature=temperature,
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )

    async for event in response:
        # event.choices[0].delta.content is None when there is no token
        if (
            event.choices
            and event.choices[0].delta
            and event.choices[0].delta.content
        ):
            yield event.choices[0].delta.content


# ---------------------------------------------------------------------------
# OpenAI  (api.openai.com)
# ---------------------------------------------------------------------------

class OpenAILLM(BaseLLM):
    """
    Connects to OpenAI using the official AsyncOpenAI SDK.

    Parameters
    ----------
    model    : any OpenAI chat model, e.g. "gpt-4o", "gpt-4o-mini"
    base_url : override to point at a compatible proxy (optional)
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment / .env file")

        # base_url=None → SDK defaults to https://api.openai.com/v1
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def stream(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> AsyncIterator[str]:
        async for chunk in _openai_stream(
            self.client, self.model, system_prompt, user_prompt
        ):
            yield chunk


# ---------------------------------------------------------------------------
# Anthropic Claude — via OpenAI-compatible proxy / OpenRouter
#
# Anthropic's *native* API is NOT OpenAI-compatible.
# This class is intended for services that re-expose Claude models
# through an OpenAI-style endpoint, for example:
#   • OpenRouter  (https://openrouter.ai/api/v1)
#   • Any self-hosted proxy
#
# If you want the native Anthropic SDK instead, install `anthropic` and
# swap this class body accordingly.
# ---------------------------------------------------------------------------

class ClaudeLLM(BaseLLM):
    """
    Connects directly to Anthropic using their OpenAI-compatible endpoint.
    Requires ANTHROPIC_API_KEY in your .env file.

    Parameters
    ----------
    model    : e.g. "claude-opus-4-5", "claude-3-5-sonnet-20241022"
    base_url : defaults to Anthropic's OpenAI-compatible base URL
    api_key  : explicit key; falls back to ANTHROPIC_API_KEY env var
    """

    DEFAULT_BASE_URL = "https://api.anthropic.com/v1/"
    DEFAULT_MODEL    = "claude-opus-4-5"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
    ):
        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No API key found. Set ANTHROPIC_API_KEY in your "
                "environment / .env file, or pass api_key= explicitly."
            )

        self.client = AsyncOpenAI(api_key=resolved_key, base_url=base_url)
        self.model  = model

    async def stream(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> AsyncIterator[str]:
        async for chunk in _openai_stream(
            self.client, self.model, system_prompt, user_prompt
        ):
            yield chunk


# ---------------------------------------------------------------------------
# Local / OpenAI-compatible  (Ollama, LM Studio, vLLM, …)
# ---------------------------------------------------------------------------

class LocalLLM(BaseLLM):
    """
    Connects to any locally running OpenAI-compatible server.

    Parameters
    ----------
    model    : model tag as recognised by the local server
    base_url : server base URL, e.g. "http://localhost:11434/v1" for Ollama
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434/v1",
    ):
        # Local servers typically accept any non-empty string as the key
        self.client = AsyncOpenAI(api_key="ollama", base_url=base_url)
        self.model  = model

    async def stream(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> AsyncIterator[str]:
        async for chunk in _openai_stream(
            self.client, self.model, system_prompt, user_prompt
        ):
            yield chunk


# ---------------------------------------------------------------------------
# Thread Q&A
# ---------------------------------------------------------------------------

class ThreadQA:
    """
    High-level interface: given a forum thread URL + a question,
    crawl (or load from cache) the thread and stream an LLM answer.
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        max_pages: Optional[int] = None,
    ):
        self.llm      = llm or OpenAILLM()
        self.crawler  = AsyncThreadCrawler()
        self.max_pages = max_pages

    # ------------------------------------------------------------------
    # Primary entry points
    # ------------------------------------------------------------------

    async def answer_stream(
        self,
        thread_url: str,
        question: str,
    ) -> AsyncIterator[str]:
        """
        Yield answer text chunks in real time.

        Usage
        -----
        CLI:
            async for chunk in qa.answer_stream(...):
                print(chunk, end="", flush=True)

        FastAPI SSE:
            async def endpoint():
                async for chunk in qa.answer_stream(...):
                    yield f"data: {chunk}\\n\\n"

        WebSocket:
            async for chunk in qa.answer_stream(...):
                await websocket.send_text(chunk)
        """
        posts_df = await self._load_or_crawl(thread_url)
        if posts_df is None or posts_df.empty:
            yield "Không thể lấy dữ liệu từ thread này."
            return

        context     = self._build_context(posts_df)
        user_prompt = await build_prompt(
                self.llm,
                context,
                question,
                post_count=len(posts_df)
            )
        logger.info("Bắt đầu stream câu trả lời từ LLM ...")

        async for chunk in self.llm.stream(SYSTEM_PROMPT, user_prompt):
            yield chunk

    async def answer(self, thread_url: str, question: str) -> str:
        """Non-streaming convenience — collects full answer into a string."""
        chunks: list[str] = []
        async for chunk in self.answer_stream(thread_url, question):
            chunks.append(chunk)
        return "".join(chunks)

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _extract_thread_id(self, thread_url: str) -> Optional[str]:
        match = re.search(r'\.(\d+)/?', thread_url)
        return match.group(1) if match else None

    async def _load_or_crawl(self, thread_url: str) -> Optional[pd.DataFrame]:
        thread_id = self._extract_thread_id(thread_url)
        if not thread_id:
            logger.error(f"Không thể trích xuất thread ID từ: {thread_url}")
            return None

        CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = CSV_OUTPUT_DIR / f"thread_{thread_id}_posts.csv"

        if csv_path.exists():
            logger.info(f"Tìm thấy CSV cache: {csv_path}")
            return pd.read_csv(csv_path)

        logger.info(f"Chưa có CSV, crawl thread {thread_id} ...")
        thread_data = await self.crawler.get_thread(
            thread_url,
            use_cache=True,
            max_pages=self.max_pages,
            save_data=True,
        )
        if not thread_data:
            return None

        # Crawler may have written the CSV itself
        if csv_path.exists():
            return pd.read_csv(csv_path)

        posts = thread_data.get("posts", [])
        return pd.DataFrame(posts) if posts else None

    def _build_context(self, df: pd.DataFrame, max_posts: int = 300) -> str:
        lines: list[str] = []
        for _, row in df.head(max_posts).iterrows():
            author = row.get("author_username", "Unknown")
            text   = row.get("content_text",    "")
            lines.append(f"[{author}: {text}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI — live streaming output
# ---------------------------------------------------------------------------

async def main():
    thread_url = (
        "https://voz.vn/t/"
        "ke-ve-nhung-niem-vui-rat-doi-thuong-ma-da-lau-ban-khong-lam-nua"
        ".1228919/"
    )
    question = "Mọi người hay kể về những niềm vui nào nhiều nhất?"

    # Swap the LLM backend here without touching anything else:
    #   llm = OpenAILLM(model="gpt-4o")
    #   llm = ClaudeLLM()
    #   llm = LocalLLM(model="llama3")
    llm = ClaudeLLM()

    qa = ThreadQA(llm=llm, max_pages=9)

    print("\n=== Câu trả lời ===")
    async for chunk in qa.answer_stream(thread_url, question):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())