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
    print('BBBB', text)
    if text.startswith('```cpp'):
        print('sono in if cpp')
        text = text[6:].strip() 
        code = 'cpp'

    if text.startswith('```cuda'):
        print('sono in if cuda')
        text = text[7:].strip() 
        code = 'cuda'
        
    if text.startswith('```cu'):
        print('sono in if cu')
        text = text[5:].strip() 
        code = 'cuda'
    
    if text.startswith('```c'):
        print('sono in if c')
        text = text[4:].strip() 
        code = 'c'

    if text.startswith('```json'):
        print('sono in if json')
        text = text[7:].strip() 
        code = 'json'

    if text.startswith('```'):
        print('sono in no name')
        text = text[3:].strip() 
        code = 'cuda'

    if text.endswith('```'):
        text = text[:-3].strip()

    return text, code



