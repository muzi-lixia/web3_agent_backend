"""Keep answer text separate from citation metadata returned by the API."""


def render_answer(draft: str, citations: list[dict]) -> str:
    """Do not append sources to the model-written answer body."""
    return draft.strip()
