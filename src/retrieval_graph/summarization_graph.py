"""Main entrypoint for the summarization graph.

This module defines the core structure and functionality of the summarization graph.
It includes the main graph definition, state management, and key functions for
processing user inputs and generating summaries using a language model.
"""

from datetime import datetime, timezone

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from retrieval_graph.configuration import SummarizationConfiguration
from retrieval_graph.state import SummaryInputState, SummaryState
from retrieval_graph.utils import load_chat_model


async def summarize(state: SummaryState, *, config: RunnableConfig) -> dict[str, list]:
    """Generate a summary of the article using a language model.

    If keywords are provided, the summary focuses on those keywords.
    """
    configuration = SummarizationConfiguration.from_runnable_config(config)

    # Build the prompt
    if state.keywords:
        keywords_str = ', '.join(state.keywords)
        user_prompt = (
            f"Please provide a summary of the following article, on it's original language."
            f"focusing on the following keywords: {keywords_str}."
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", configuration.summary_system_prompt_utm),
                ("user", user_prompt),
                ("user", "{article}"),
            ]
        )
    else:
        user_prompt = "Please provide a summary of the following article on it's original language."

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", configuration.summary_system_prompt),
                ("user", user_prompt),
                ("user", "{article}"),
            ]
        )

    model = load_chat_model(configuration.summary_model)

    message_value = await prompt.ainvoke(
        {
            "article": state.article,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )

    response = await model.ainvoke(message_value, config)
    # We return a list of messages to be consistent with other graphs
    return {"messages": [response]}


# Define the graph
builder = StateGraph(SummaryState, input=SummaryInputState, config_schema=SummarizationConfiguration)

builder.add_node(summarize)
builder.add_edge("__start__", "summarize")

# Compile the graph
graph = builder.compile(
    interrupt_before=[],  # if you want to update the state before calling the tools
    interrupt_after=[],
)

graph.name = "SummarizationGraph"
