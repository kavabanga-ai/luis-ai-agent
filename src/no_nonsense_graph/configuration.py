"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Optional, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config

from no_nonsense_graph import prompts

T = TypeVar("T", bound="NoNonsenseConfiguration")


@dataclass(kw_only=True)
class NoNonsenseConfiguration:
    """Configuration class for the button generation agent."""

    user_id: str = field(metadata={"description": "Unique identifier for the user."})

    nonsense_identify_system_prompt: str = field(
        default=prompts.NONSENSE_IDENTIFY_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for button generation."},
    )

    nonsense_identify_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4",
        metadata={
            "description": "The language model used for button generation. Should be in the form: provider/model-name."
        },
    )

    @classmethod
    def from_runnable_config(cls: Type[T], config: Optional[RunnableConfig] = None) -> T:
        """Create a ButtonGenConfiguration instance from a RunnableConfig object.

        Args:
            config (Optional[RunnableConfig]): The configuration object to use.

        Returns:
            T: An instance of ButtonGenConfiguration with the specified configuration.
        """
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
