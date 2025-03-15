from dataclasses import dataclass
from typing import Optional


@dataclass(kw_only=True)
class ButtonGenInputState:
    """Input state for the button generation graph."""
    article: Optional[str] = None


@dataclass(kw_only=True)
class ButtonGenState(ButtonGenInputState):
    """State of the button generation graph."""
    questions: Optional[dict[str, str]] = None
    answers: Optional[dict[str, str]] = None
