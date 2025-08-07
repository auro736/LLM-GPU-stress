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


def extract_code_simple(text):
    """
    Simpler version using string operations
    
    Args:
        text (str): The raw model output containing code blocks
        
    Returns:
        str: Clean code content
    """
    # If no code block markers, return as-is
    if '```' not in text:
        return text.strip()
    
    # Split on first ```
    parts = text.split('```', 1)
    if len(parts) < 2:
        return text.strip()
    
    # Get the part after first ```
    after_first = parts[1]
    
    # Find the next ``` (closing) and get everything before it
    if '```' in after_first:
        code_part = after_first.split('```')[0]
    else:
        code_part = after_first
    
    # Remove language identifier if present (first line)
    lines = code_part.split('\n')
    if lines and lines[0].strip() and not lines[0].strip().startswith(('/', '#', '*')):
        # First line might be language identifier, remove it
        if any(lang in lines[0].lower() for lang in ['cpp', 'cu', 'cuda', 'c++', 'c']):
            lines = lines[1:]
    
    return '\n'.join(lines).strip()


# Example usage
if __name__ == "__main__":
    sample_text = '''```cpp
// gpu_stress_test.cu
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>
__global__ void matrix_multiply_kernel(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < N && idy < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[idy * N + i] * B[i * N + idx];
        }
        C[idy * N + idx] = sum;
    }
}
```
Compile with: `nvcc gpu_stress_test.cu -o gpu_stress_test`

This is additional text that should be removed.'''

    # Test both functions
    print("=== Method 1 ===")
    result1 = extract_code_from_output(sample_text)
    print(result1)
    
    print("\n=== Method 2 ===")
    result2 = extract_code_simple(sample_text)
    print(result2)