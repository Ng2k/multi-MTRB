import re
from .constants import CLINICAL_MARKERS, NOISE_PATTERN, SPECIAL_CHARS_PATTERN


class TextCleaner:
    """Stateless domain logic for clinical text transformation."""

    @staticmethod
    def apply_clinical_tags(text: str) -> str:
        """
        Converts markers like [laughter] to DAIC_LAUGHTER to preserve 
        them as unique feature tokens.
        """
        for marker in CLINICAL_MARKERS:
            text = re.sub(
                rf'\[{marker}\]', 
                f' DAIC_{marker.upper()} ', 
                text, 
                flags=re.IGNORECASE
            )

        return NOISE_PATTERN.sub('', text)

    @staticmethod
    def sanitize(text: str) -> str:
        """Standardizes casing, punctuation, and whitespace."""
        text = text.lower()
        text = SPECIAL_CHARS_PATTERN.sub('', text)
        return " ".join(text.split()).strip()

