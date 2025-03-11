"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, Optional, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config

from retrieval_graph import prompts

T = TypeVar("T", bound="ButtonGenConfiguration")


@dataclass(kw_only=True)
class IndexConfiguration:
    """Configuration class for indexing and retrieval operations.

    This class defines the parameters needed for configuring the indexing and
    retrieval processes, including user identification, embedding model selection,
    retriever provider choice, and search parameters.
    """

    user_id: str = field(metadata={"description": "Unique identifier for the user."})

    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embeddings"}},
    ] = field(
        default="openai/text-embedding-3-small",
        metadata={
            "description": "Name of the embedding model to use. Must be a valid embedding model name."
        },
    )

    retriever_provider: Annotated[
        Literal["elastic", "elastic-local", "pinecone", "mongodb", "pgvector", "milvus"],
        {"__template_metadata__": {"kind": "retriever"}},
    ] = field(
        default="elastic",
        metadata={
            "description": "The vector store provider to use for retrieval. Options are 'elastic', 'pinecone', 'mongodb', or 'pgvector'."

        },
    )

    search_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "Additional keyword arguments to pass to the search function of the retriever."
        },
    )

    @classmethod
    def from_runnable_config(
            cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        """Create an IndexConfiguration instance from a RunnableConfig object.

        Args:
            cls (Type[T]): The class itself.
            config (Optional[RunnableConfig]): The configuration object to use.

        Returns:
            T: An instance of IndexConfiguration with the specified configuration.
        """
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


T = TypeVar("T", bound=IndexConfiguration)


@dataclass(kw_only=True)
class Configuration(IndexConfiguration):
    """The configuration for the agent."""

    response_system_prompt: str = field(
        default=prompts.RESPONSE_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for generating responses."},
    )

    response_system_prompt_with_link: str = field(
        default=prompts.RESPONSE_SYSTEM_PROMPT_WITH_LINK,
        metadata={"description": "The system prompt used for generating responses need links."},
    )

    doctors_system_prompt_condition: Optional[str] = field(
        default=prompts.APPOINTMENT_SYSTEM_PROMPT_CONDITION,
        metadata={"description": "The system prompt used for generating responses. for appointment"},
    )

    doctors_system_prompt: Optional[str] = field(
        default=prompts.APPOINTMENT_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for generating responses. for appointment"},
    )

    check_for_retrieval_system_prompt: Optional[str] = field(
        default=prompts.CHECK_FOR_RETRIEVAL_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for check if retrieval need or not"},
    )

    # medical_test_system_prompt_condition: Optional[str] = field(
    #     default=prompts.MEDICAL_TEST_SYSTEM_PROMPT_CONDITION,
    #     metadata={"description": "The system prompt used for generating responses. for appointment"},
    # )
    #
    # medical_test_system_prompt: Optional[str] = field(
    #     default=prompts.MEDICAL_TEST_SYSTEM_PROMPT,
    #     metadata={"description": "The system prompt used for generating responses. for appointment"},
    # )

    response_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o",
        metadata={
            "description": "The language model used for generating responses. Should be in the form: provider/model-name."
        },
    )

    query_system_prompt: str = field(
        default=prompts.QUERY_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for processing and refining queries."
        },
    )

    query_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o",
        metadata={
            "description": "The language model used for processing and refining queries. Should be in the form: "
                           "provider/model-name."
        },
    )

    check_link_system_prompt: Optional[str] = field(
        default=prompts.CHECK_LINK_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for check it's link or not"},
    )

    check_same_url_system_prompt: Optional[str] = field(
        default=prompts.CHECK_SAME_URL_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for check it have same url or not"},
    )

    check_same_duplicate_url_system_prompt: Optional[str] = field(
        default=prompts.CHECK_SAME_DUPLICATE_URL_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for fix duplicate urls"},
    )

    reference_system_prompt: Optional[str] = field(
        default=prompts.REFERENCE_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for give reference to the reponse"},
    )


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
