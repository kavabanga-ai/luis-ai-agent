from typing import Optional


def load_search_model(fully_specified_name: str, query: Optional[str] = None, max_result: int = 5):
    """Load a search model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider'.

    Returns:
        BaseChatModel: An instance of the loaded chat model.
        :param query:
        :param max_result:
        :param fully_specified_name:
    """

    if fully_specified_name == "tavily_langchain":
        from langchain_community.tools import TavilySearchResults
        return TavilySearchResults(
            max_results=max_result,
            include_images=True,
        )

    if fully_specified_name == "tavily":
        from tavily import TavilyClient
        client = TavilyClient()
        response = client.search(
            query=query,
            max_results=3,
            search_depth="advanced",
            include_images=True
        )
        return response
