from datetime import datetime, timezone

try:
    from typing_extensions import Any, Literal, TypedDict, cast
except ImportError:
    from typing import Any, Literal, TypedDict, cast

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from no_nonsense_graph.configuration import NoNonsenseConfiguration as Configuration
from no_nonsense_graph.state import InputState, State
from retrieval_graph.utils import format_docs, get_message_text, load_chat_model


async def respond(
        state: State, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Call the LLM powering our "agent"."""

    configuration = Configuration.from_runnable_config(config)
    # Feel free to customize the prompt, model, and other logic!
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.nonsense_identify_system_prompt),
            ("placeholder", "{messages}")]
    )

    response_model_kwargs = config.get('configurable').get('response_model_kwargs')

    model = load_chat_model(configuration.nonsense_identify_model, response_model_kwargs)

    message_value = await prompt.ainvoke(
        {
            "messages": state.messages,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )
    response = await model.ainvoke(message_value, config)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph (It's just a pipe)


builder = StateGraph(State, input=InputState, config_schema=Configuration)

builder.add_node(respond)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)

# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "NoNonsenseGraph"
