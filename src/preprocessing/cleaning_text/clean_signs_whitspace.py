import re

def normalize_whitespace(text: str) -> str:
    """
    Normalizes whitespace:
    - Removes leading and trailing spaces
    - Collapses multiple spaces or tabs into one
    - Collapses multiple line breaks into a single one
    """
    text = re.sub(r"[ \t]+", " ", text)   #multiple spaces/tabs -> single space 
    text = re.sub(r"\s*\n\s*", "\n", text) #clean up around line breaks
    text = re.sub(r"\n+", "\n", text) #multiple newlines -> one newline
    text = text.replace("\u00A0", " ")
    return text.strip() #removes leading/trailing spaces


def remove_unwanted_signs(text: str) -> str:
    """
    Remove unwanted signs like [1], (kilde), {note}, and stray symbols.
    """
    text = re.sub(r"\[\d+\]", "", text)    #remove footnote markers [1]
    text = re.sub(r"\([^)]*\)", "", text)  #remove text inside parentheses
    text = re.sub(r"\{[^}]*\}", "", text)  #remove curly brace notes
    text = re.sub(r"[*_#><`]", "", text)   #remove markdown-style signs
    return text
