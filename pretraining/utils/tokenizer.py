# Tokenizer utilities
# Your implementation goes here
def simple_tokenizer(text: str) -> list[str]:
    """A simple whitespace tokenizer."""
    text = text.strip().lower()
    return text.split()
def ngram_tokenizer(text: str, n: int) -> list[str]:
    """Generates n-grams from the input text."""
    tokens = simple_tokenizer(text)
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
def char_tokenizer(text: str) -> list[str]:
    """Tokenizes text into individual characters."""
    text = text.strip().lower()
    return list(text)
def custom_tokenizer(text: str, method: str = 'simple', n: int = 2) -> list[str]:
    """A customizable tokenizer based on the specified method."""
    if method == 'simple':
        return simple_tokenizer(text)
    elif method == 'ngram':
        return ngram_tokenizer(text, n)
    elif method == 'char':
        return char_tokenizer(text)
    else:
        raise ValueError(f"Unknown tokenization method: {method}")
    
