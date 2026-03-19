import re

CLINICAL_MARKERS = [
    'laughter', 'sigh', 'pause', 'breath', 
    'vocalized-noise', 'cough', 'sniff'
]
NOISE_PATTERN = re.compile(r'\[.*?\]')
SPECIAL_CHARS_PATTERN = re.compile(r'[^a-zA-Z0-9\s.,!?\'_]')

