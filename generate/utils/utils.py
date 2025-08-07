
def get_system_prompt(mode):

    if mode == 'zero-shot':
        prompt = """
        You are a programmer specialized in writing CUDA and C++ code optimized for GPU stress testing. 
        The testing process enables us to examine the impact of potential errors caused by faults in the underlying hardware. Specifically, best practices in testing involve creating specialized programs designed to stress the hardware executing them.
        For this reason, your objective is to create code that maximizes GPU resource utilization for benchmarking and testing GPUs by pushing the hardware to the utilization limits.
        Generate code to be divided in one or more scripts that stresses multiple GPU aspects (computational units, memory and schedulers) simultaneously through intensive mathematical operations like matrix multiplications, floating-point calculations, special functions stressing the XU units, and atomic operations. 
        Utilize modern CUDA 12 features with efficient shared memory usage, memory coalescing, and maximum occupancy.
        Your programs must be production-ready to be compiled with nvcc with comprehensive error handling. 
        Find the memory access pattern that mantains the highest occupancy of the computational units over time as well as the highest computational throughput. 
        To stress the hardware use as much as possible L2 cache.
        Include user defined parameter for test duration in seconds. The code must be stopped if its duration is longer than user defined time. 
        Do not use any syncronization function. All the instances of the kernels must be executed in parallel. 
        Give as output only the code of the one or more scripts by indicating the extension file needed ready to be compiled with nvcc. Provide in output only code with no other additional comments.
        """
        # da checkare cosa chiedere nei dettagli, o fare diversi prompt
        # I would suggest to use as much as possible the L2 cache among the other available GPU memory hierarchy levels while asyncronous and parallel exeuction of different threads, thus, operations are executed.

        # return """
        #             You are a specialized agent for generating CUDA and C++ code optimized for GPU stress testing.
        #             Your objective is to create code that maximizes GPU resource utilization for benchmarking and performance testing by pushing modern GPUs to their limits while maintaining stability.

        #             Generate code that stresses multiple GPU aspects simultaneously through intensive mathematical operations like matrix multiplications, floating-point calculations, trigonometric functions, and atomic operations. 
        #             Find the memory access pattern that mantains the highest occupancy of the computational units over time as well as the highest computational throughput.
        #             Utilize modern CUDA 12 features with efficient shared memory usage, memory coalescing, and maximum occupancy.

        #             Your programs must be production-ready with comprehensive error handling.
        #             Include configurable parameters for test duration and workload composition. 

        #             Give as output only the .cu code ready to be compiled. Provide in output only code with no other additional comments.
        #         """
        return prompt

    elif mode == 'few-shot':
        return """ To be defined """
    else :
        raise Exception("Accepted mode are only zero-shot or few-shot")
    

def get_user_prompt(mode):
    if mode == 'zero-shot':
        # da checkare cosa chiedere nei dettagli, o fare diversi prompt
        # return """
        #            Following your system prompt your target for stressing is: RTX 4060 GPU with CUDA 12 and 8GB VRAM. Enclose code between ```  
        #         """
        return """
                   Following your system prompt your target for stressing is: two RTX 6000 Ada generation GPUs with CUDA 12 and 48GB VRAM each. Enclose code between ```  
            """

    elif mode == 'few-shot':
        return """ To be defined """
    else :
        raise Exception("Accepted mode are only zero-shot or few-shot")



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

def extract_code_from_output(text):
    """
    Extracts code content from model output, removing everything after the closing ```
    
    Args:
        text (str): The raw model output containing code blocks
        
    Returns:
        str: Clean code content without markdown formatting or trailing text
    """
    # If there's no code block markers, assume the whole text is code
    if '```' not in text:
        return text.strip()
    
    # Find the first code block (starting with ```)
    lines = text.split('\n')
    code_lines = []
    in_code_block = False
    
    for line in lines:
        # Check if this line starts a code block
        if line.strip().startswith('```'):
            if not in_code_block:
                # Starting a code block
                in_code_block = True
                continue
            else:
                # Ending the code block - stop processing
                break
        
        # If we're in a code block, collect the line
        if in_code_block:
            code_lines.append(line)
    
    # Join the code lines and clean up
    code = '\n'.join(code_lines)
    
    # Remove any leading/trailing whitespace
    code = code.strip()
    
    return code

def fix_code(timestamp, original_code, stderr, model, system_prompt="You are a helpful compiler assistant. Fix errors in the code."):
    """
    Use an LLM to correct the code given the compilation error.

    Args:
        original_code (str): Initial source code
        stderr (str): Compilation error output
        model: LLM model instance (e.g. TogetherModel)
        system_prompt (str): System prompt

    Returns:
        str: Corrected code proposed by the model
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"The following code:\n\n{original_code}\n\nGenerated this compilation error:\n\n{stderr}\n\nPlease correct the code."}
    ]
    
    response = model.generate(messages=messages, temperature=0.5, max_new_tokens=None, seed=4899)
    fixed_code, _ = clean_string(response)

    clean_answer, code_type = clean_string(fixed_code)

    if code_type == 'cpp':
        out_file = f'out_{timestamp}.cpp'
    if code_type == 'cuda':
        out_file = f'out_{timestamp}.cu'
    
    clean_answer = extract_code_from_output(clean_answer)

    return clean_answer
