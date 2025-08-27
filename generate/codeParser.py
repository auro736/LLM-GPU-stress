import os

class CodeParser():

    def __init__(self, code_string, code_type):
        self.code_string = code_string
        self.code_type = code_type

    def extract_code_from_output(self, timestamp):
        """
        Extracts code content from model output, removing everything after the closing ```
        
        Args:
            text (str): The raw model output containing code blocks
            
        Returns:
            str: Clean code content without markdown formatting or trailing text
        """
        if self.code_type == 'cpp':
            out_file = f'out{timestamp}.cu' #forzato a .cu se no si rompe tutto durante compilazione
        if self.code_type == 'cuda':
            out_file = f'out{timestamp}.cu'
        if self.code_type == 'c':
            out_file = f'out{timestamp}.cu'

        if '```' not in self.code_string:
            out_file = f'out{timestamp}.cu'
            return self.code_string.strip(), out_file
        
       
        lines = self.code_string.split('\n')
        code_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith('```'):
                if not in_code_block:
                    in_code_block = True
                    continue
                else:
                    break
            
            if in_code_block:
                code_lines.append(line)
        
        code = '\n'.join(code_lines)
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