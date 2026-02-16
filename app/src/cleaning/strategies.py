""" Strategies for data cleaning

In this module there are all the strategies to perform data cleaning on the DAIC-WOZ dataset.
These strategies are designed to be used inside a "scripts loader", so the user can choose which
strategies he wants to perform

Typical usage example:
    ```python
        if __name__ == "__main__":
            # Define a custom pipeline if you want to change behavior later
            custom_pipeline = [
                CleaningStrategies.remove_brackets,
                CleaningStrategies.remove_filler_words,
                CleaningStrategies.lowercase,
                CleaningStrategies.collapse_whitespace
            ]

            loader = ScriptLoader(pipeline=custom_pipeline)
    ```
"""

import re


class CleaningStrategies:
    """
    A collection of stateless transformation strategies for text cleaning.
    Designed for the DAIC-WOZ transcript format.
    """

    @staticmethod
    def remove_daic_tags(text: str) -> str:
        """
        Removes DAIC-WOZ metadata tags enclosed in brackets.
        Example: "I am [laughter] happy" -> "I am  happy"
        """
        return re.sub(r'\[.*?\]', '', text)

    @staticmethod
    def lowercase(text: str) -> str:
        """Converts text to lowercase."""
        return text.lower()

    @staticmethod
    def collapse_whitespace(text: str) -> str:
        """Removes tabs, newlines, and multiple spaces."""
        return " ".join(text.split())

    @staticmethod
    def remove_special_chars(text: str) -> str:
        """Removes non-alphanumeric characters except basic punctuation."""
        return re.sub(r'[^a-zA-Z0-9\s.,!?\']', '', text)

    @staticmethod
    def strip_edges(text: str) -> str:
        """Removes leading and trailing whitespace."""
        return text.strip()
