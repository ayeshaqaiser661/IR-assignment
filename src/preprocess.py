import re
def preprocessed_query(text: str):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 2]
    return tokens
def tokens_to_string(tokens):
    return " ".join(tokens)
