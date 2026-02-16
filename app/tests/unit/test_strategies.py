"""
Unit tests for the CleaningStrategies class.
Ensures 100% coverage of the stateless transformation logic.
"""

import pytest
from src.cleaning.strategies import CleaningStrategies

class TestCleaningStrategies:
    """Suite to verify each static method in CleaningStrategies."""

    # --- Test remove_daic_tags ---
    @pytest.mark.parametrize("input_text, expected", [
        ("I am [laughter] happy", "I am  happy"),
        ("[noise] static [pause]", " static "),
        ("No tags here", "No tags here"),
        ("Empty brackets []", "Empty brackets "),
        ("[multiple][tags]", ""),
    ])
    def test_remove_daic_tags(self, input_text, expected):
        """Checks if bracketed content is removed correctly."""
        assert CleaningStrategies.remove_daic_tags(input_text) == expected

    # --- Test lowercase ---
    def test_lowercase(self):
        """Checks conversion to lowercase."""
        assert CleaningStrategies.lowercase("HELLO World") == "hello world"
        assert CleaningStrategies.lowercase("123 ABC") == "123 abc"

    # --- Test collapse_whitespace ---
    @pytest.mark.parametrize("input_text, expected", [
        ("  too   many   spaces  ", "too many spaces"),
        ("tab\tbetween\nlines", "tab between lines"),
        ("SingleSpace", "SingleSpace"),
        ("   ", ""),
    ])
    def test_collapse_whitespace(self, input_text, expected):
        """Checks if tabs, newlines, and extra spaces are condensed."""
        assert CleaningStrategies.collapse_whitespace(input_text) == expected

    # --- Test remove_special_chars ---
    @pytest.mark.parametrize("input_text, expected", [
        ("Hello, world!", "Hello, world!"), # Punctuation preserved
        ("Don't touch this.", "Don't touch this."), # Apostrophe preserved
        ("Special $#@ chars", "Special  chars"), # Special chars removed
        ("123 numbers 456", "123 numbers 456"), # Numbers preserved
    ])
    def test_remove_special_chars(self, input_text, expected):
        """Checks if non-alphanumeric chars (excluding basic punctuation) are stripped."""
        assert CleaningStrategies.remove_special_chars(input_text) == expected

    # --- Test strip_edges ---
    @pytest.mark.parametrize("input_text, expected", [
        ("  strip me  ", "strip me"),
        ("\nnewline strip\t", "newline strip"),
        ("no_space", "no_space"),
    ])
    def test_strip_edges(self, input_text, expected):
        """Checks if leading and trailing whitespaces are removed."""
        assert CleaningStrategies.strip_edges(input_text) == expected

    # --- Edge Case: Integration (Pipeline Style) ---
    def test_combined_strategies(self):
        """
        GIVEN a dirty string
        WHEN multiple strategies are applied in sequence
        THEN the output should be perfectly clean.
        """
        dirty_text = "  [laughter] I AM... VERY happy! [noise]  "
 
        # Simulating the pipeline behavior described in module docstring
        step1 = CleaningStrategies.remove_daic_tags(dirty_text)
        step2 = CleaningStrategies.lowercase(step1)
        step3 = CleaningStrategies.remove_special_chars(step2)
        step4 = CleaningStrategies.collapse_whitespace(step3)
        # Result should have no brackets, no caps, no extra punctuation, no extra spaces

        assert step4 == "i am... very happy!"

