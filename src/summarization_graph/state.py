from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Optional, Sequence, Union

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


def add_queries(existing: Sequence[str], new: Sequence[str]) -> Sequence[str]:
    """Combine existing queries with new queries.

    Args:
        existing (Sequence[str]): The current list of queries in the state.
        new (Sequence[str]): The new queries to be added.

    Returns:
        Sequence[str]: A new list containing all queries from both input sequences.
    """
    return list(existing) + list(new)


@dataclass(kw_only=True)
class SummaryInputState:
    """Represents the input state for the agent.

    Includes messages and optionally article and keywords for summarization.
    """
    messages: Annotated[Sequence[AnyMessage], add_messages]
    article: Optional[str] = None
    keywords: Optional[Sequence[str]] = field(default_factory=list)


@dataclass(kw_only=True)
class SummaryState(SummaryInputState):
    """The state of your graph / agent."""

    queries: Annotated[list[str], add_queries] = field(default_factory=list)
    retrieved_docs: list[Document] = field(default_factory=list)
    summary: Optional[str] = None
