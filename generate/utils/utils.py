
def get_system_prompt(mode):

    if mode == 'zero-shot':
        # da checkare cosa chiedere nei dettagli, o fare diversi prompt
        return """
                    You are a specialized agent for generating CUDA and C++ code optimized for GPU stress testing.
                    Your objective is to create code that maximizes GPU resource utilization for benchmarking and performance testing by pushing modern GPUs to their limits while maintaining stability.

                    Generate code that stresses multiple GPU aspects simultaneously through intensive mathematical operations like matrix multiplications, floating-point calculations, trigonometric functions, and atomic operations. Combine these with memory-intensive patterns including random access, sequential streaming, and host-device transfers. Utilize modern CUDA 11.0+ features with efficient shared memory usage, memory coalescing, and maximum occupancy.

                    Your programs must be production-ready with comprehensive error handling.
                    Include configurable parameters for stress intensity, test duration, and workload composition. Implement comprehensive metrics including GFLOPS/TOPS, memory bandwidth utilization, GPU occupancy, operation latencies, and thermal monitoring when available.

                    Give as output only the .cu or .cpp code ready to be compiled. Provide in output only code with no other additional comments.
                """
    elif mode == 'few-shot':
        return """ To be defined """
    else :
        raise Exception("Accepted mode are only zero-shot or few-shot")
    

def get_user_prompt(mode):
    if mode == 'zero-shot':
        # da checkare cosa chiedere nei dettagli, o fare diversi prompt
        return """
                   Generate a comprehensive CUDA program for intensive GPU stress testing that combines mathematical operations with complex memory access patterns. 
                   The program should maximize resource utilization on modern NVIDIA GPUs.
    
                    Target modern GPU architectures including RTX 4090, Tesla V100, and A100 series with CUDA Compute Capability 7.0+ and 16GB+ VRAM. 
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

	if text.startswith('```json'):
		text = text[7:].strip() 
		code = 'json'

	if text.endswith('```'):
		text = text[:-3].strip()
		
	return text, code

