import os

class codeParser():

    def __init__(self, code_string, code_type):
        self.codeString = code_string
        self.codeType = code_type

    def extract_code_from_output(self, timestamp):
        """
        Extracts code content from model output, removing everything after the closing ```
        
        Args:
            text (str): The raw model output containing code blocks
            
        Returns:
            str: Clean code content without markdown formatting or trailing text
        """
        if self.code_type == 'cpp':
            out_file = f'out_{timestamp}.cu' #forzato a .cu se no si rompe tutto durante compilazione
        if self.code_type == 'cuda':
            out_file = f'out_{timestamp}.cu'
        # If there's no code block markers, assume the whole text is code
        if '```' not in self.codeString:
            return self.codeString.strip()
        
        # Find the first code block (starting with ```)
        lines = self.codeString.split('\n')
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
        
        return code, out_file
    
    def adaptCode(self, file_name):

        profiling_bash_template = './utils/profiling_bash_template'
        with open(os.path.join(profiling_bash_template,"template.sh"), "r") as f:
            content = f.read()
        
        content = content.replace("./test-apps/rora/rora 60", f"./test-apps/{file_name}/{file_name} 60")
        content = content.replace("data/raw/stress2/rora_$INJECTION_KERNEL_COUNT.txt", f"data/raw/stress2/{file_name}_$INJECTION_KERNEL_COUNT.txt")
        
        with open(f"../evaluate/cupti/02_profiling_injection/exe/bash/profiling_stress2/{file_name}.sh", "w") as f:
            f.write(content)

        postprocessing_bash_template = './utils/postprocessing_bash_template'
        with open(os.path.join(postprocessing_bash_template,"template.sh"), "r") as f:
            content = f.read()

        content = content.replace("APP_NAME='rora'", f"APP_NAME='{file_name}'")
        with open(f"../evaluate/cupti/02_profiling_injection/exe/bash/postprocessing/{file_name}.sh", "w") as f:
            f.write(content)