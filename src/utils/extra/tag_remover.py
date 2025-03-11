import re


class TagRemover:
    """
    A class to remove specified HTML/XML-like tags (and their entire content) from text.

    Attributes
    ----------
    tags : list of str
        A list of tag names to remove from text.

    Methods
    -------
    add_tag(tag: str) -> None
        Adds a new tag to the internal list of removable tags.

    remove_tags(text: str) -> str
        Removes all occurrences of the specified tags (and their content) from the text.
    """

    def __init__(self, tags=None):
        """
        Initializes the TagRemover with a list of tags to remove.

        Parameters
        ----------
        tags : list of str, optional
            A list of tag names to remove. If not provided, it defaults to an empty list.
        """
        if tags is None:
            tags = []
        self.tags = tags

    def add_tag(self, tag):
        """
        Adds a tag to the list of removable tags.

        Parameters
        ----------
        tag : str
            The name of the tag to remove from the text.
        """
        self.tags.append(tag)

    def remove_tags(self, text):
        """
        Removes all specified tags and their content from the given text.

        Parameters
        ----------
        text : str
            The text from which the tags and their content should be removed.

        Returns
        -------
        str
            The text after removing the specified tags and their content.
        """
        cleaned_text = text
        for tag in self.tags:
            # Build a pattern to match <tag ...> ... </tag> (including the tag content)
            pattern = rf'<{tag}.*?>.*?</{tag}>'
            # Remove everything that matches, across multiple lines (DOTALL)
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
        return cleaned_text

# if __name__ == "__main__":
#     # Example usage:
#     sample_text = """
#     <think>
#     Okay, so I need to figure out how to respond as Ivan, the property manager. ...
#     </think>
#     Hi there! It's great to hear from you. How can I assist you today?
#     """
#
#     # Create an instance that removes the 'think' tag
#     remover = TagRemover(['think'])
#
#     # Remove <think>...</think> and print the result
#     result_text = remover.remove_tags(sample_text)
#     print(result_text.strip())
