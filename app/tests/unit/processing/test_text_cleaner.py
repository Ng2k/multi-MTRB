from src.processing.text_cleaner import TextCleaner


def test_text_cleaner_workflow():
    """Tests the full cleaning chain: tags then sanitization."""
    raw_text = "I am [laughter] feeling... GREAT!"

    # Apply tags
    tagged = TextCleaner.apply_clinical_tags(raw_text)
    assert "DAIC_LAUGHTER" in tagged

    # Sanitize (ensuring underscore is preserved)
    final = TextCleaner.sanitize(tagged)
    assert final == "i am daic_laughter feeling... great!"


def test_apply_clinical_tags_removes_noise():
    """Verifies non-clinical tags are stripped."""
    text = "Hello [scrubbed_entry] [laughter]"
    cleaned = TextCleaner.apply_clinical_tags(text)
    assert "DAIC_LAUGHTER" in cleaned
    assert "[scrubbed_entry]" not in cleaned


def test_sanitize_removes_unwanted_chars():
    """Ensures special characters are removed but underscores remain."""
    text = "User_Name #123 @Home!"
    # # and @ should go, _ should stay
    assert TextCleaner.sanitize(text) == "user_name 123 home!"

