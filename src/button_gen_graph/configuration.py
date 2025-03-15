"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, Optional, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config

from button_gen_graph import prompts

T = TypeVar("T", bound="ButtonGenConfiguration")


@dataclass(kw_only=True)
class ButtonGenConfiguration:
    """Configuration class for the button generation agent."""

    user_id: str = field(metadata={"description": "Unique identifier for the user."})

    button_gen_system_prompt: str = field(
        default=prompts.BUTTON_GEN_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for button generation."},
    )

    button_gen_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4",
        metadata={
            "description": "The language model used for button generation. Should be in the form: provider/model-name."
        },
    )

    button_max_character: Optional[int] = field(
        default=25,
        metadata={"description": "Max button characters."},
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
