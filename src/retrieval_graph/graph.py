"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing user inputs, generating queries, retrieving
relevant documents, and formulating responses.
"""
import re
from datetime import datetime, timezone
from typing import cast

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig
from langchain_core.messages.ai import AIMessage
from langgraph.graph import StateGraph, START, END

from retrieval_graph import retrieval
from retrieval_graph.configuration import Configuration
from retrieval_graph.state import InputState, State
from retrieval_graph.utils import format_docs, get_message_text, load_chat_model

# Define the function that calls the model

LINK_STATE = False
CHITCHAT_STATE = False
APPOINTMENT_STATE = False
MEDICAL_TEST_STATE = False
DUPLICATE_URL_STATE = False
SAME_PAGE_URL_STATE = False
SAME_PAGE_URL = None
SOURCE_URLS = ""


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
            ("system", configuration.doctors_system_prompt_condition),
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
    return {"condition_state": is_requesting_link}


# async def check_for_medical_test(
#         state: State, *, config: RunnableConfig
# ) -> dict[str, bool]:
#     """Determine if the user's message is requesting a medical test."""
#     configuration = Configuration.from_runnable_config(config)
#
#     # Create the prompt template with dynamic messages
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", configuration.medical_test_system_prompt_condition),
#             ("human", "User question: {messages}"),
#         ]
#     )
#
#     # Load the model and invoke the prompt
#     model = load_chat_model(configuration.query_model)
#     # Prepare the message value
#     last_human_message = state.messages[-1]  # Last human message
#     last_ai_response = state.messages[-2] if len(state.messages) > 1 and isinstance(state.messages[-2],
#                                                                                     AIMessage) else None  # Last AI response
#
#     # Combine messages for the prompt
#     messages_to_pass = [last_human_message]
#     if last_ai_response:
#         messages_to_pass.insert(0, last_ai_response)  # Add AI response before the human message
#
#     message_value = await prompt.ainvoke(
#         {
#             "messages": messages_to_pass,  # Pass the latest user message content
#             "system_time": datetime.now(tz=timezone.utc).isoformat(),
#         },
#         config,
#     )
#     response = await model.ainvoke(message_value, config)
#
#     is_requesting_link = response.content.strip().lower() == 'yes'
#
#     # Update state with `is_requesting_link` and required fields
#     global MEDICAL_TEST_STATE
#     MEDICAL_TEST_STATE = is_requesting_link
#     return {"condition_state": is_requesting_link}


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

    the_given_url = str(get_path_article_url(config))
    if the_given_url == "liver/":
        config['configurable']['search_kwargs'] = {}
    with retrieval.make_retriever(config) as retriever:
        responses = await retriever.ainvoke(state.queries[-1], config)

        filtered_responses = []
        backup_responses = []
        url_source_faq = ""
        for response in responses:
            url_source = None
            if "uri" in response.metadata:
                if response.metadata["uri"].endswith(".xml") or "faq" in response.metadata["uri"] or "faq2" in \
                        response.metadata["uri"]:
                    if not response.metadata["uri"].endswith(".xml"):
                        url_source_faq = f'source url: https://vseopecheni.ru/{response.metadata["uri"]}'
                    url_source = None
                else:
                    url_source = f'source url: https://vseopecheni.ru/{response.metadata["uri"]}'
            if url_source:
                response.page_content = response.page_content + "\n" + url_source
                filtered_responses.append(response)
            else:
                response.page_content = response.page_content + "\n" + url_source_faq
                backup_responses.append(response)

        # Ensure at least four responses are returned if possible
        if len(filtered_responses) < 6:
            extra_needed = 6 - len(filtered_responses)
            filtered_responses.extend(backup_responses[:extra_needed])

        return {"retrieved_docs": filtered_responses[:6]}


def get_path_article_url(config):
    # Navigate through the dictionary safely
    try:
        # Access nested dictionaries checking each key
        configurable = config.get('configurable', {})
        search_kwargs = configurable.get('search_kwargs', {})
        filter_kwargs = search_kwargs.get('filter', {})
        uri_condition = filter_kwargs.get('uri', {})

        # Debugging: Print the intermediate outputs to verify structure
        # print("search_kwargs:", search_kwargs)
        # print("filter_kwargs:", filter_kwargs)
        # print("uri_condition:", uri_condition)

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

    path_article_url = get_path_article_url(config)
    config['configurable']['search_kwargs'] = {"k": 20}

    # print("============Article Path Test==============")
    # print("path_article_url", path_article_url)

    global SOURCE_URLS

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
                    url_source = f'source url: https://vseopecheni.ru/{response.metadata["uri"]}'
                    SOURCE_URLS = SOURCE_URLS + url_source + ",\n"
            else:
                url_source = None

            if url_source:
                response.page_content = response.page_content + "\n" + url_source
                filtered_responses.append(response)

            else:
                if uri == path_article_url:
                    continue
                else:
                    backup_responses.append(response)

        # Ensure at least four responses are returned if possible
        if len(filtered_responses) < 6:
            extra_needed = 6 - len(filtered_responses)
            filtered_responses.extend(backup_responses[:extra_needed])

        return {"retrieved_docs": filtered_responses[:6]}


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
                                                                                    AIMessage) else None  # Last AI
    # response

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
    return {"condition_state": is_requesting_link}


async def check_for_retrieval_needed_or_not(
        state: State, *, config: RunnableConfig
) -> dict[str, bool]:
    """Determine if the user's message is requesting a link."""
    configuration = Configuration.from_runnable_config(config)

    # Create the prompt template with dynamic messages
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.check_for_retrieval_system_prompt),
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
    return {"condition_state": is_requesting_link}


async def respond(
        state: State, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Call the LLM powering our "agent"."""

    configuration = Configuration.from_runnable_config(config)
    global APPOINTMENT_STATE
    global LINK_STATE
    global MEDICAL_TEST_STATE
    last_5_messages = state.messages[-5:]  # This will get the last 5 messages

    # Feel free to customize the prompt, model, and other logic!
    if APPOINTMENT_STATE:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", configuration.doctors_system_prompt),
                ("placeholder", "{messages}"),
            ]
        )

    # elif MEDICAL_TEST_STATE:
    #     MEDICAL_TEST_STATE = False
    #     prompt = ChatPromptTemplate.from_messages(
    #         [
    #             ("system", configuration.medical_test_system_prompt),
    #             ("placeholder", "{messages}"),
    #         ]
    #     )

    elif LINK_STATE:
        LINK_STATE = False
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", configuration.response_system_prompt_with_link),
                ("placeholder", "{messages}"),
                ("human", "Retrieved document: \n\n {retrieved_docs} \n\n User question: {messages}"),
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
            "messages": last_5_messages,
            "retrieved_docs": retrieved_docs,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )
    response = await model.ainvoke(message_value, config)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def check_duplicate_links(text):
    """
    Checks if the same URL appears more than once in a given text.

    Args:
        text (str): The input text containing URLs.

    Returns:
        str: "Same link" if a URL is repeated, otherwise "Not same link".
    """
    # Regular expression to find URLs
    url_pattern = re.compile(r'(https?://\S+)')

    # Find all URLs in the text
    urls = url_pattern.findall(text)

    # Check for duplicates
    if len(urls) != len(set(urls)):
        return True
    else:
        return False


async def duplicate_links(state: State, *, config: RunnableConfig):
    global DUPLICATE_URL_STATE
    global SOURCE_URLS

    last_ai_response = None
    # Check if there are messages and if the last message is an AIMessage
    if len(state.messages) > 0 and isinstance(state.messages[-1], AIMessage):
        last_ai_response = state.messages[-1]  # Get the last AI message

    last_ai_response_str = last_ai_response.content

    DUPLICATE_URL_STATE = check_duplicate_links(last_ai_response_str)

    # If duplicates found, remove duplicates from SOURCE_URLS
    if DUPLICATE_URL_STATE:
        # Parse SOURCE_URLS and remove duplicates
        lines = [line.strip() for line in SOURCE_URLS.split(',\n') if line.strip()]
        seen = set()
        unique_lines = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        # Rebuild SOURCE_URLS without duplicates
        if unique_lines:
            SOURCE_URLS = ",\n".join(unique_lines) + ",\n"
        else:
            SOURCE_URLS = ""

    return {"condition_state": DUPLICATE_URL_STATE}


async def same_page_links(state: State, *, config: RunnableConfig):
    global SAME_PAGE_URL_STATE
    global SOURCE_URLS
    global SAME_PAGE_URL

    url_path = "https://vseopecheni.ru/" + str(get_path_article_url(config))

    last_ai_response = None
    # Check if there are messages and if the last message is an AIMessage
    if len(state.messages) > 0 and isinstance(state.messages[-1], AIMessage):
        last_ai_response = state.messages[-1]  # Get the last AI message

    last_ai_response_str = last_ai_response.content

    if url_path in last_ai_response_str:
        SAME_PAGE_URL_STATE = True
        SAME_PAGE_URL = url_path
        print(SAME_PAGE_URL)
        # Remove the same-page URL line from SOURCE_URLS
        lines = [line.strip() for line in SOURCE_URLS.split(',\n') if line.strip()]
        # Keep only lines that don't contain the same-page URL
        filtered_lines = [line for line in lines if url_path not in line]
        if filtered_lines:
            SOURCE_URLS = ",\n".join(filtered_lines) + ",\n"
        # else:
        #     SOURCE_URLS = ""
    else:
        SAME_PAGE_URL_STATE = False

    return {"condition_state": SAME_PAGE_URL_STATE}


async def fix_response_with_same_url(
        state: State, *, config: RunnableConfig
) -> dict[str, bool]:
    """Determine if the user's message contain same page url."""
    configuration = Configuration.from_runnable_config(config)

    url_path = "https://vseopecheni.ru/" + str(get_path_article_url(config))
    # Prepare the message value
    last_ai_response = None  # Initialize as None

    # Check if there are messages and if the last message is an AIMessage
    if len(state.messages) > 0 and isinstance(state.messages[-1], AIMessage):
        last_ai_response = state.messages[-1]  # Get the last AI message

    # last_ai_response_str = last_ai_response.content

    global SOURCE_URLS
    global DUPLICATE_URL_STATE
    global SAME_PAGE_URL_STATE

    if DUPLICATE_URL_STATE:
        prompt_duplicate = ChatPromptTemplate.from_messages(
            [
                ("system", configuration.check_same_duplicate_url_system_prompt),
                ("placeholder", "{messages}"),
            ]
        )

        # Load the model and invoke the prompt
        model_duplicate = load_chat_model(configuration.query_model)

        message_value_duplicate = await prompt_duplicate.ainvoke(
            {
                "messages": [last_ai_response],  # Pass the latest user message content
                "source_urls": SOURCE_URLS,
                "system_time": datetime.now(tz=timezone.utc).isoformat(),
            },
            config,
        )
        response_duplicate = await model_duplicate.ainvoke(message_value_duplicate, config)
        DUPLICATE_URL_STATE = False

        last_ai_response = response_duplicate

    elif SAME_PAGE_URL_STATE:
        prompt_same_link = ChatPromptTemplate.from_messages(
            [
                ("system", configuration.check_same_url_system_prompt),
                ("placeholder", "{messages}"),
            ]
        )

        # Load the model and invoke the prompt
        model_same_link = load_chat_model(configuration.query_model)

        message_value_duplicate = await prompt_same_link.ainvoke(
            {
                "messages": [last_ai_response],  # Pass the latest user message content
                "page_url": url_path,
                "source_urls": SOURCE_URLS,
                "system_time": datetime.now(tz=timezone.utc).isoformat(),
            },
            config,
        )
        response_same_page = await model_same_link.ainvoke(message_value_duplicate, config)
        SAME_PAGE_URL_STATE = False

        last_ai_response = response_same_page

    # SOURCE_URLS = ""

    return {"messages": [last_ai_response]}


def replace_numeric_brackets_1st(text: str) -> str:
    """
    Replace occurrences of [number] with [[number]].
    For example, [1] -> [[1]], [12] -> [[12]].
    """
    pattern = r'\[(\d+)\]'  # Looks for [digits]
    replacement = r'[#\1]'  # Replaces [digits] with [[digits]]

    return re.sub(pattern, replacement, text)


def preserve_square_brackets_for_markdown_links(text: str) -> str:
    """
    Find patterns of the form [digits](URL), e.g. [1](http://example.com),
    and transform them into [&#91;digits&#93;](URL).

    Then, in rendered Markdown:
      - The link text will literally show as [1].
      - No extra backslashes are involved, so you avoid accidentally creating
        or messing up a '\1' capturing group in your JS filter.

    Example:
      "[1](http://example.com)" --> "[&#91;1&#93;](http://example.com)"
      Then Markdown shows the link text as [1].
    """
    # Regex captures:
    #   group(1) => digits
    #   group(2) => the URL
    #
    # Replacement uses HTML entities for '[' and ']':
    #   [&#91;\1&#93;](\2)
    # so the final link text is [1], not just 1.
    pattern = re.compile(r'\[(\d+)\]\(([^)]+)\)')
    replacement = r'[&#91;\1&#93;](\2)'
    return pattern.sub(replacement, text)


def double_bracket_numeric_links(text: str) -> str:
    """
    If a link text is purely numeric, e.g. [123](URL),
    turn it into [[123]](URL).
    If the link text is non-numeric, leave it as is: [some text](URL).
    """
    # Pattern: look for [digits](URL).
    # group(1) = the digits, group(2) = the URL.
    numeric_pattern = re.compile(r'\[([0-9]+)\]\(([^)]+)\)')

    # Replacement: [[\1]](\2)
    # so [1](http://foo) becomes [[1]](http://foo).
    #
    # Anything else (like [abc](http://...))
    # remains untouched.
    return numeric_pattern.sub(r'[[\1]](\2)', text)


def fix_markdown_reference(text: str) -> str:
    """
    Finds markdown references in the form of [number](url).
    and changes them to [number](url) .
    (i.e., adds a space before the final period).
    """
    # This pattern looks for something like: [3](https://some-link.com).
    pattern = re.compile(r'(\[\d+\]\(https?://[^\)]+\))\.')
    # The replacement puts a space before the period: [3](https://some-link.com) .
    return pattern.sub(r'\1 .', text)


async def reference_response(
        state: State, *, config: RunnableConfig
) -> dict[str, bool]:
    """Determine if the user's message contain same page url."""
    configuration = Configuration.from_runnable_config(config)

    global SOURCE_URLS
    global SAME_PAGE_URL
    global CHITCHAT_STATE
    global APPOINTMENT_STATE
    global MEDICAL_TEST_STATE

    the_given_url = str(get_path_article_url(config))
    if the_given_url == "liver/":
        url_path = SOURCE_URLS
    else:
        url_path = "https://vseopecheni.ru/" + the_given_url

    # Prepare the message value
    last_ai_response = None  # Initialize as None

    # Check if there are messages and if the last message is an AIMessage
    if len(state.messages) > 0 and isinstance(state.messages[-1], AIMessage):
        last_ai_response = state.messages[-1]  # Get the last AI message

    # last_ai_response_str = last_ai_response.content

    if CHITCHAT_STATE:
        print("Chitchat")
    elif APPOINTMENT_STATE:
        print("Doctors")

    else:
        # if same url cut from up then source url don't have the same link anymore
        if SAME_PAGE_URL:
            final_urls = SAME_PAGE_URL
        elif SOURCE_URLS or SOURCE_URLS != "":
            final_urls = SOURCE_URLS
        else:
            final_urls = url_path

        prompt_reference = ChatPromptTemplate.from_messages(
            [
                ("system", configuration.reference_system_prompt),
                ("placeholder", "{messages}"),
            ]
        )
        # Load the model and invoke the prompt
        model_duplicate = load_chat_model(configuration.query_model)

        message_value_duplicate = await prompt_reference.ainvoke(
            {
                "messages": [last_ai_response],  # Pass the latest user message content
                "source_urls": final_urls,
                "system_time": datetime.now(tz=timezone.utc).isoformat(),
            },
            config,
        )
        response_reference = await model_duplicate.ainvoke(message_value_duplicate, config)
        final_response_reference_str = response_reference.content
        # final_response_reference_str = double_bracket_numeric_links(response_reference.content)
        # final_response_reference_str = replace_numeric_brackets_1st(response_reference.content)
        final_response_reference_str = fix_markdown_reference(response_reference.content)
        last_ai_response.content = last_ai_response.content + "\n\n" + final_response_reference_str

    CHITCHAT_STATE = False
    APPOINTMENT_STATE = False
    MEDICAL_TEST_STATE = False

    SOURCE_URLS = ""

    return {"messages": [last_ai_response]}


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


def decide_for_medical_test(state: State) -> str:
    global MEDICAL_TEST_STATE
    if MEDICAL_TEST_STATE:
        return "respond"
    else:
        return "check_for_link_request"


# Define a new graph (It's just a pipe)

builder = StateGraph(State, input=InputState, config_schema=Configuration)

builder.add_node(check_for_appointment)
# builder.add_node(check_for_medical_test)
builder.add_node(generate_query)
builder.add_node(check_for_link_request)
builder.add_node(check_for_retrieval_needed_or_not)
builder.add_node(retrieve)
builder.add_node(retrieve_update_url)
builder.add_node(respond)
builder.add_node(duplicate_links)
builder.add_node(same_page_links)
builder.add_node(fix_response_with_same_url)
builder.add_node(reference_response)

# Starting node to check for link requests
builder.add_edge(START, "check_for_appointment")

builder.add_conditional_edges(
    source="check_for_appointment",
    path=decide_for_appointment,
    path_map={
        "respond": "respond",
        "check_for_link_request": "check_for_link_request"
    }
)

# builder.add_conditional_edges(
#     source="check_for_medical_test",
#     path=decide_for_medical_test,
#     path_map={
#         "respond": "respond",
#         "check_for_link_request": "check_for_link_request"
#     }
# )

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
builder.add_edge("respond", "duplicate_links")
builder.add_edge("duplicate_links", "same_page_links")
builder.add_edge("same_page_links", "fix_response_with_same_url")
builder.add_edge("fix_response_with_same_url", "reference_response")
builder.add_edge("reference_response", END)

# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
graph = builder.compile(
    interrupt_before=[],  # if you want to update the state before calling the tools
    interrupt_after=[],
)
graph.name = "RetrievalGraph"
