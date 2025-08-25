
def get_system_prompt(mode):

    if mode == 'zero-shot':
        prompt = """
        You are a programmer specialized in writing CUDA and C++ code optimized for GPU stress testing. 
        The testing process enables us to examine the impact of potential errors caused by faults in the underlying hardware. Specifically, best practices in testing involve creating specialized programs designed to stress the hardware executing them.
        For this reason, your objective is to create code that maximizes GPU resource utilization for benchmarking and testing GPUs by pushing the hardware to the utilization limits.
        Generate code to be divided in one or more scripts that stresses multiple GPU aspects (computational units, memory and schedulers) simultaneously through intensive mathematical operations like matrix multiplications, floating-point calculations, special functions stressing the XU units, and atomic operations. 
        Use modern CUDA 12 features with efficient shared memory usage, memory coalescing, and maximum occupancy.
        Your programs must be production-ready to be compiled with nvcc with comprehensive error handling. 
        Find the memory access pattern that mantains the highest occupancy of the computational units over time as well as the highest computational throughput. 
        To stress the hardware use as much as possible L2 cache.
        The user will tell also the test duration time in seconds, include it in the code. With test duration time we intend how long the code should run in loop.The code must be stopped if its duration is longer than user defined time. 
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
                   Following your system prompt your target for stressing is: two RTX 6000 Ada generation GPUs with CUDA 12 and 48GB VRAM each. Test duration time: 120 seconds. Enclose code between ```  
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



