"""Default prompts."""

RESPONSE_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's questions based on the retrieved documents.

{retrieved_docs}

System time: {system_time}"""

QUERY_SYSTEM_PROMPT = """Generate search queries to retrieve documents that may help answer the user's question. Previously, you made the following queries:
    
<previous_queries/>
{queries}
</previous_queries>

System time: {system_time}"""

CHECK_FOR_RETRIEVAL_SYSTEM_PROMPT = """Analyze the user's input to determine if it is a simple greeting (e.g., "hi,
" "hello") or farewell Respond with 'yes' if it is, and 'no' if it is not."""
