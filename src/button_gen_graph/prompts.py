"""Default prompts."""

BUTTON_GEN_SYSTEM_PROMPT = """You are an assistant that, given an article, generates three buttons and then questions 
and answers for those buttons.

The buttons are like:

- "key points of the article"
- "related materials"
- "experts opinions"

For each button, generate a question that the user might ask, and an answer based on the content of the article.

{format_instructions}

Ensure the output strictly adheres to the provided format.

Do not include any extra text or explanations."""

CHECK_FOR_RETRIEVAL_SYSTEM_PROMPT = """Analyze the user's input to determine if it is a simple greeting (e.g., "hi,
" "hello") or farewell Respond with 'yes' if it is, and 'no' if it is not."""
