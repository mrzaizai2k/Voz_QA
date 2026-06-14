import json

from src.prompts import ROUTER_PROMPT


async def detect_intent(llm, question: str) -> str:

    response = await llm.chat(
        ROUTER_PROMPT,
        question
    )

    try:
        data = json.loads(response)

        intent = data.get("intent", "summary")

        return intent

    except Exception:
        return "summary"