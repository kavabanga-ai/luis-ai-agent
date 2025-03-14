"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Optional, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config

from summarization_graph import prompts

T = TypeVar("T", bound="ButtonGenConfiguration")


@dataclass(kw_only=True)
class SummarizationConfiguration:
    """Configuration class for the summarization agent."""

    user_id: str = field(metadata={"description": "Unique identifier for the user."})

    summary_system_prompt: str = field(
        default=prompts.SUMMARY_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for summarization."},
    )

    summary_system_prompt_utm: str = field(
        default=prompts.SUMMARY_SYSTEM_PROMPT_UTM,
        metadata={"description": "The system prompt used for summarization. with keywords"},
    )

    summary_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o",
        metadata={
            "description": "The language model used for summarization. Should be in the form: provider/model-name."
        },
    )

    @classmethod
    def from_runnable_config(
            cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        """Create a SummarizationConfiguration instance from a RunnableConfig object.

        Args:
            cls (Type[T]): The class itself.
            config (Optional[RunnableConfig]): The configuration object to use.

        Returns:
            T: An instance of SummarizationConfiguration with the specified configuration.
        """
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
