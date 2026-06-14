# providers.py

import os
from typing import List
from dotenv import load_dotenv
load_dotenv()


def get_openai_models(api_key: str) -> List[str]:
    """
    Fetches the available OpenAI models from the API.
    
    Parameters
    ----------
    api_key : str
        Your OpenAI API key.

    Returns
    -------
    List[str]
        A sorted list of available OpenAI model IDs.
    """
    try:
        import httpx
        r = httpx.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=8,
        )
        if r.status_code == 200:
            data = r.json().get("data", [])
            ids = sorted(
                [m["id"] for m in data if "gpt" in m["id"].lower()],
                reverse=True,
            )
            return ids or ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
        else:
            print(f"OpenAI API error {r.status_code}: {r.text}")
    except Exception as e:
        print(f"Error fetching OpenAI models: {e}")
    return ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]


def get_anthropic_models(api_key: str) -> List[str]:
    """
    Fetches available Claude models from the Anthropic API.

    Uses the correct Anthropic authentication headers:
      - x-api-key:        your Anthropic API key
      - anthropic-version: the API version string (required)

    Parameters
    ----------
    api_key : str
        Your Anthropic API key.

    Returns
    -------
    List[str]
        A sorted list of available Claude model IDs.
    """
    FALLBACK_MODELS = [
        "claude-opus-4-5",
        "claude-3-5-sonnet-20241022",
        "claude-3-haiku-20240307",
    ]

    try:
        import httpx

        headers = {
            "x-api-key": api_key,                      # ✅ Anthropic uses x-api-key
            "anthropic-version": "2023-06-01",          # ✅ Required version header
            "content-type": "application/json",
        }

        r = httpx.get(
            "https://api.anthropic.com/v1/models",
            headers=headers,
            timeout=8,
        )

        if r.status_code == 200:
            data = r.json().get("data", [])
            ids = sorted(
                [m["id"] for m in data if "claude" in m["id"].lower()],
                reverse=True,
            )
            return ids or FALLBACK_MODELS
        else:
            print(f"Anthropic API error {r.status_code}: {r.text}")

    except Exception as e:
        print(f"Error fetching Claude models: {e}")

    return FALLBACK_MODELS


if __name__ == "__main__":
    openai_api_key    = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    if not openai_api_key:
        print("⚠️  OPENAI_API_KEY environment variable is not set.")
    else:
        print("Fetching OpenAI models...")
        openai_models = get_openai_models(openai_api_key)
        print(f"✅ Available OpenAI models: {openai_models}\n")

    if not anthropic_api_key:
        print("⚠️  ANTHROPIC_API_KEY environment variable is not set.")
    else:
        print("Fetching Anthropic (Claude) models...")
        claude_models = get_anthropic_models(anthropic_api_key)
        print(f"✅ Available Claude models: {claude_models}\n")