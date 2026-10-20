"""String helpers."""
import re


def slugify(text: str) -> str:
    """Lowercase, replace runs of non-alphanumerics with one dash, strip dashes."""
    return re.sub(r"[^a-z0-9]", "-", text.lower())


def truncate(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..."
