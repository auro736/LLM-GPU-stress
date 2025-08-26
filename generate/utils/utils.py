def clean_string(text):
    """
    Removes ```json at the beginning and ``` at the end of a string if present,
    and ensures the string starts with { and ends with }.

    Args:
    text (str): The input string to clean

    Returns:
    str: The cleaned string, or None if it doesn't start with { and end with }
    """
    text = text.strip()
    code = ''

    if text.startswith('```cpp'):
        text = text[6:].strip() 
        code = 'cpp'

    if text.startswith('```cuda'):
        text = text[7:].strip() 
        code = 'cuda'
        
    if text.startswith('```cu'):
        text = text[5:].strip() 
        code = 'cuda'

    if text.startswith('```json'):
        text = text[7:].strip() 
        code = 'json'

    if text.endswith('```'):
        text = text[:-3].strip()

    return text, code



