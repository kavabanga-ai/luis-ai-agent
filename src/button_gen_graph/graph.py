import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from button_gen_graph.configuration import ButtonGenConfiguration
from button_gen_graph.state import ButtonGenInputState, ButtonGenState
from retrieval_graph.utils import load_chat_model

from langchain.output_parsers import StructuredOutputParser, ResponseSchema

logger = logging.getLogger(__name__)

CONDITION_LENGTH = False
COUNT_RECURSION_LIMIT = 0


async def generate_buttons(state: ButtonGenState, *, config: RunnableConfig) -> dict[str, dict]:
    """Generate buttons, questions, and answers based on the article.

    This function uses a language model to process the input article and generates
    predefined buttons along with relevant questions and answers.

    Args:
        state (ButtonGenState): The current state containing the article.
        config (RunnableConfig): Configuration for the button generation process.

    Returns:
        dict[str, dict]: A dictionary containing 'questions' and 'answers' dictionaries.
    """
    configuration = ButtonGenConfiguration.from_runnable_config(config)

    # Define the expected output schema
    response_schemas = [
        ResponseSchema(name="questions", description="Dictionary of button names to questions."),
        # ResponseSchema(name="answers", description="Dictionary of button names to answers."),
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    # Get the format instructions for the model
    format_instructions = output_parser.get_format_instructions()

    # Escape curly braces in format_instructions
    escaped_format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")

    # Prepare the system message with format instructions using f-string
    system_message = f"""{configuration.button_gen_system_prompt}"""

    # Build the prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("user", "{article}"),
        ]
    )

    model = load_chat_model(configuration.button_gen_model)

    prompt_values = {
        "article": state.article,
        "format_instructions": escaped_format_instructions,
    }

    # Generate the message value
    message_value = await prompt.ainvoke(prompt_values, config)

    # Invoke the model
    response_gen = await model.ainvoke(message_value, config)

    # Parse the response
    try:
        parsed_output = output_parser.parse(response_gen.content)
        questions = parsed_output.get("questions", {})
        # answers = parsed_output.get("answers", {})
    except Exception as e:
        logger.error(f"Failed to parse response: {e}")
        raise ValueError("The model did not return output matching the expected schema.")

    check_key_length(questions, configuration.button_max_character)

    global COUNT_RECURSION_LIMIT
    COUNT_RECURSION_LIMIT += 1

    # Update the state
    return {
        "questions": questions,
        # "answers": answers,
        "answers": "",
    }


def check_key_length(dictionary, length):
    """
    Checks if all keys in the dictionary have more than the specified length.

    Args:
    dictionary (dict): The dictionary to check.
    length (int): The character length to check against.
    """
    global CONDITION_LENGTH
    CONDITION_LENGTH = all(len(key) < length for key in dictionary.keys())


async def condition_button_length(state: ButtonGenState):
    global CONDITION_LENGTH
    global COUNT_RECURSION_LIMIT
    if not CONDITION_LENGTH and COUNT_RECURSION_LIMIT < 23:
        return "generate_buttons"
    else:
        return "response"


def response(dictionary):
    """
    It response the final dict
    """

    return dictionary


# Define the graph


builder = StateGraph(
    ButtonGenState,
    input=ButtonGenInputState,
    config_schema=ButtonGenConfiguration,
)

builder.add_node(generate_buttons)
builder.add_node(response)
builder.add_edge("__start__", "generate_buttons")

builder.add_conditional_edges(
    source="generate_buttons",
    path=condition_button_length,
    path_map={
        "generate_buttons": "generate_buttons",
        "response": "response"
    }
)

# Compile the graph
graph = builder.compile()
graph.name = "ButtonGenGraph"
