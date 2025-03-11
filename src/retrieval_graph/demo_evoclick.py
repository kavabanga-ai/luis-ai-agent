"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing user inputs, generating queries, retrieving
relevant documents, and formulating responses.
"""

from datetime import datetime, timezone
from typing import cast

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig
from langchain_core.messages.ai import AIMessage
from langgraph.graph import StateGraph

from retrieval_graph import retrieval
from retrieval_graph.configuration import Configuration
from retrieval_graph.state import InputState, State
from retrieval_graph.utils import format_docs, get_message_text, load_chat_model

# Define the function that calls the model

LINK_STATE = False
CHITCHAT_STATE = False
APPOINTMENT_STATE = False


class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""

    query: str


async def check_for_appointment(
        state: State, *, config: RunnableConfig
) -> dict[str, bool]:
    """Determine if the user's message is requesting a appointment."""
    configuration = Configuration.from_runnable_config(config)

    # Create the prompt template with dynamic messages
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """Analyze the user's input to determine if it indicates an intention to
             book an appointment with a doctor. Respond with 'yes' if the input clearly reflects
              this intention (e.g., 'I need a doctor appointment,' 'Can I book a doctor?' or 'Are
               there doctors available?'), and 'no' if it does not."""),
            ("human", "User question: {messages}"),
        ]
    )

    # Load the model and invoke the prompt
    model = load_chat_model(configuration.query_model)
    # Prepare the message value
    last_human_message = state.messages[-1]  # Last human message
    last_ai_response = state.messages[-2] if len(state.messages) > 1 and isinstance(state.messages[-2],
                                                                                    AIMessage) else None  # Last AI response

    # Combine messages for the prompt
    messages_to_pass = [last_human_message]
    if last_ai_response:
        messages_to_pass.insert(0, last_ai_response)  # Add AI response before the human message

    message_value = await prompt.ainvoke(
        {
            "messages": messages_to_pass,  # Pass the latest user message content
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )
    response = await model.ainvoke(message_value, config)

    is_requesting_link = response.content.strip().lower() == 'yes'

    # Update state with `is_requesting_link` and required fields
    global APPOINTMENT_STATE
    APPOINTMENT_STATE = is_requesting_link
    return {"messages": state.messages}


async def generate_query(
        state: State, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """Generate a search query based on the current state and configuration.

    This function analyzes the messages in the state and generates an appropriate
    search query. For the first message, it uses the user's input directly.
    For subsequent messages, it uses a language model to generate a refined query.

    Args:
        state (State): The current state containing messages and other information.
        config (RunnableConfig | None, optional): Configuration for the query generation process.

    Returns:
        dict[str, list[str]]: A dictionary with a 'queries' key containing a list of generated queries.

    Behavior:
        - If there's only one message (first user input), it uses that as the query.
        - For subsequent messages, it uses a language model to generate a refined query.
        - The function uses the configuration to set up the prompt and model for query generation.
    """
    messages = state.messages
    if len(messages) == 1:
        # It's the first user question. We will use the input directly to search.
        human_input = get_message_text(messages[-1])
        return {"queries": [human_input]}
    else:
        configuration = Configuration.from_runnable_config(config)
        # Feel free to customize the prompt, model, and other logic!
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", configuration.query_system_prompt),
                ("placeholder", "{messages}"),
            ]
        )
        model = load_chat_model(configuration.query_model).with_structured_output(
            SearchQuery
        )

        message_value = await prompt.ainvoke(
            {
                "messages": state.messages,
                "queries": "\n- ".join(state.queries),
                "system_time": datetime.now(tz=timezone.utc).isoformat(),
            },
            config,
        )
        generated = cast(SearchQuery, await model.ainvoke(message_value, config))
        return {
            "queries": [generated.query],
        }


async def retrieve(
        state: State, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents based on the latest query in the state.

    This function takes the current state and configuration, uses the latest query
    from the state to retrieve relevant documents using the retriever, and returns
    the retrieved documents.

    Args:
        state (State): The current state containing queries and the retriever.
        config (RunnableConfig | None, optional): Configuration for the retrieval process.

    Returns:
        dict[str, list[Document]]: A dictionary with a single key "retrieved_docs"
        containing a list of retrieved Document objects.
    """

    # Ensure search_kwargs exists and merge with existing configurations
    if 'configurable' not in config:
        config['configurable'] = {}
    if 'search_kwargs' not in config['configurable']:
        config['configurable']['search_kwargs'] = {}

    # Update or add the 'k' parameter
    config['configurable']['search_kwargs']['k'] = 20
    with retrieval.make_retriever(config) as retriever:
        responses = await retriever.ainvoke(state.queries[-1], config)

        filtered_responses = []
        backup_responses = []
        for response in responses:
            url_source = None
            if "uri" in response.metadata:
                if response.metadata["uri"].endswith(".xml") or "faq" in response.metadata["uri"] or "faq2" in \
                        response.metadata["uri"]:
                    url_source = None
                else:
                    url_source = f'source url: {response.metadata["uri"]}'

            if url_source:
                response.page_content = response.page_content + "\n" + url_source
                filtered_responses.append(response)
            else:
                backup_responses.append(response)

        # Ensure at least four responses are returned if possible
        if len(filtered_responses) < 4:
            extra_needed = 4 - len(filtered_responses)
            filtered_responses.extend(backup_responses[:extra_needed])

        return {"retrieved_docs": filtered_responses[:4]}


def get_path_article_url(config):
    # Navigate through the dictionary safely
    try:
        # Access nested dictionaries checking each key
        configurable = config.get('configurable', {})
        search_kwargs = configurable.get('search_kwargs', {})
        filter_kwargs = search_kwargs.get('filter', {})
        uri_condition = filter_kwargs.get('uri', {})

        # The uri condition might have different comparison operators, check for equality
        path_article_url = uri_condition.get('$eq', None)

        return path_article_url
    except AttributeError:
        # In case any attribute along the path is None and does not support the get() method
        return None


async def retrieve_update_url(
        state: State, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents based on the latest query in the state.

    This function takes the current state and configuration, uses the latest query
    from the state to retrieve relevant documents using the retriever, and returns
    the retrieved documents.

    Args:
        state (State): The current state containing queries and the retriever.
        config (RunnableConfig | None, optional): Configuration for the retrieval process.

    Returns:
        dict[str, list[Document]]: A dictionary with a single key "retrieved_docs"
        containing a list of retrieved Document objects.
    """
    # Ensure search_kwargs exists and merge with existing configurations
    if 'configurable' not in config:
        config['configurable'] = {}
    if 'search_kwargs' not in config['configurable']:
        config['configurable']['search_kwargs'] = {}

    config['configurable']['search_kwargs'] = {"k": 20}
    path_article_url = get_path_article_url(config)
    with retrieval.make_retriever(config) as retriever:
        responses = await retriever.ainvoke(state.queries[-1], config)

        filtered_responses = []
        backup_responses = []

        for response in responses:
            if "uri" in response.metadata:
                uri = response.metadata["uri"]
                if uri.endswith(".xml") or "faq" in uri or uri == path_article_url:
                    url_source = None
                else:
                    url_source = f'source url: https://test.vseopecheni.ru/{response.metadata["uri"]}'
            else:
                url_source = None

            if url_source:
                response.page_content = response.page_content + "\n" + url_source
                filtered_responses.append(response)

            else:
                backup_responses.append(response)

        # Ensure at least four responses are returned if possible
        if len(filtered_responses) < 4:
            extra_needed = 4 - len(filtered_responses)
            filtered_responses.extend(backup_responses[:extra_needed])

        return {"retrieved_docs": filtered_responses[:4]}


async def respond(
        state: State, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Call the LLM powering our "agent"."""

    configuration = Configuration.from_runnable_config(config)
    global APPOINTMENT_STATE
    # Feel free to customize the prompt, model, and other logic!
    if APPOINTMENT_STATE:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", configuration.appointment_system_prompt),
                ("placeholder", "{messages}"),
            ]
        )
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", configuration.response_system_prompt),
                ("placeholder", "{messages}"),
                ("human", "Retrieved document: \n\n {retrieved_docs} \n\n User question: {messages}"),
            ]
        )

    response_model_kwargs = config.get('configurable').get('response_model_kwargs')

    model = load_chat_model(configuration.response_model, response_model_kwargs)

    retrieved_docs = format_docs(state.retrieved_docs)
    message_value = await prompt.ainvoke(
        {
            "messages": state.messages,
            "retrieved_docs": retrieved_docs,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )
    response = await model.ainvoke(message_value, config)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


async def check_for_link_request(
        state: State, *, config: RunnableConfig
) -> dict[str, bool]:
    """Determine if the user's message is requesting a link."""
    configuration = Configuration.from_runnable_config(config)

    # Create the prompt template with dynamic messages
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.check_link_system_prompt),
            ("human", "User question along with ai text: {messages}"),
        ]
    )

    # Load the model and invoke the prompt
    model = load_chat_model(configuration.query_model)
    # Prepare the message value
    last_human_message = state.messages[-1]  # Last human message
    last_ai_response = state.messages[-2] if len(state.messages) > 1 and isinstance(state.messages[-2],
                                                                                    AIMessage) else None  # Last AI response

    # Combine messages for the prompt
    messages_to_pass = [last_human_message]
    if last_ai_response:
        messages_to_pass.insert(0, last_ai_response)  # Add AI response before the human message

    message_value = await prompt.ainvoke(
        {
            "messages": messages_to_pass,  # Pass the latest user message content
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )
    response = await model.ainvoke(message_value, config)

    is_requesting_link = response.content.strip().lower() == 'yes'

    # Update state with `is_requesting_link` and required fields
    global LINK_STATE
    LINK_STATE = is_requesting_link
    return {"messages": state.messages}


async def check_for_retrieval_needed_or_not(
        state: State, *, config: RunnableConfig
) -> dict[str, bool]:
    """Determine if the user's message is requesting a link."""
    configuration = Configuration.from_runnable_config(config)

    # Create the prompt template with dynamic messages
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """Analyze the user's input to determine if it is a simple greeting (e.g., "hi," "hello") 
            or farewell Respond with 'yes' if it is, and 'no' if it is not."""),
            ("human", "User question: {messages}"),
        ]
    )

    # Load the model and invoke the prompt
    model = load_chat_model(configuration.query_model)

    message_value = await prompt.ainvoke(
        {
            "messages": state.messages[-1],  # Pass the latest user message content
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )
    response = await model.ainvoke(message_value, config)

    is_requesting_link = response.content.strip().lower() == 'yes'

    # Update state with `is_requesting_link` and required fields
    global CHITCHAT_STATE
    CHITCHAT_STATE = is_requesting_link
    return {"messages": state.messages}


def decide_path_after_generate_query(state: State) -> str:
    global CHITCHAT_STATE
    global LINK_STATE
    if LINK_STATE:
        return "retrieve_update_url"
    elif CHITCHAT_STATE:
        return "respond"
    else:
        return "retrieve"


def decide_for_appointment(state: State) -> str:
    global APPOINTMENT_STATE
    if APPOINTMENT_STATE:
        return "respond"
    else:
        return "check_for_link_request"


# Define a new graph (It's just a pipe)

builder = StateGraph(State, input=InputState, config_schema=Configuration)

builder.add_node(check_for_appointment)
builder.add_node(generate_query)
builder.add_node(check_for_link_request)
builder.add_node(check_for_retrieval_needed_or_not)
builder.add_node(retrieve)
builder.add_node(retrieve_update_url)
builder.add_node(respond)

# Starting node to check for link requests
builder.add_edge("__start__", "check_for_appointment")

builder.add_conditional_edges(
    source="check_for_appointment",
    path=decide_for_appointment,
    path_map={
        "respond": "respond",
        "check_for_link_request": "check_for_link_request"
    }
)

builder.add_edge("check_for_link_request", "generate_query")
builder.add_edge("generate_query", "check_for_retrieval_needed_or_not")

# Conditional edge based on CHITCHAT_STATE
builder.add_conditional_edges(
    source="check_for_retrieval_needed_or_not",
    path=decide_path_after_generate_query,
    path_map={
        "respond": "respond",
        "retrieve": "retrieve",
        "retrieve_update_url": "retrieve_update_url"

    }
)

# Continuing the path from check_for_retrieval_needed_or_not
# builder.add_edge("check_for_retrieval_needed_or_not", "respond")

# Connecting retrieve nodes to respond
builder.add_edge("retrieve", "respond")
builder.add_edge("retrieve_update_url", "respond")

# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
graph = builder.compile(
    interrupt_before=[],  # if you want to update the state before calling the tools
    interrupt_after=[],
)
graph.name = "EvoClickDemoGraph"
