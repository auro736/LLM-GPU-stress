import os
import subprocess
from utils.utils import *

class Compiler():
    def __init__(self, template_dir):
        self.template_dir = template_dir

    def prepare_makefile(self, file_name, out_file, save_dir):
        with open(os.path.join(self.template_dir,"Makefile"), "r") as f:
            content = f.read()
    
        content = content.replace("TARGET := target", f"TARGET := {file_name}")
        content = content.replace("SRC := stress_test.cu", f"SRC := {out_file}")

        with open(os.path.join(save_dir,"Makefile"), "w") as f:
            f.write(content)

        print("Makefile generated successfully.")

   
    def compile(self, save_dir):
        try:
            result = subprocess.run(
                ["make", "-C", save_dir],
                capture_output=True,
                text=True)
        
            success = result.returncode == 0
            if success:
                print("Compiler ok!")
            else:
                print("Compiler error.")
                print("STDERR:")
                print(result.stderr)
        
            return {
                "success": success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except FileNotFoundError as e:
            print(f"Compiler error: 'make' command not found. Make sure it's installed and in PATH: {e}")
            return {
                "success": False,
                "stdout": "",
                "stderr": f"make command not found: {str(e)}",
                "returncode": -1
            }
        except PermissionError as e:
            print(f"Compiler error: Permission denied accessing directory or running make: {e}")
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Permission denied: {str(e)}",
                "returncode": -1
            }
        except subprocess.TimeoutExpired as e:
            print(f"Compiler error: Make command timed out: {e}")
            return {
                "success": False,
                "stdout": e.stdout or "",
                "stderr": e.stderr or f"Command timed out: {str(e)}",
                "returncode": -1
            }
        except OSError as e:
            print(f"Compiler error: OS error occurred: {e}")
            return {
                "success": False,
                "stdout": "",
                "stderr": f"OS error: {str(e)}",
                "returncode": -1
            }
        except Exception as e:
            print(f"Compiler error: Unexpected error occurred: {e}")
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Unexpected error: {str(e)}",
                "returncode": -1
            }
        
    def fix_compile(self, max_attempts, attempt, compile_result, save_dir, out_file, timestamp, model):
        
        while not compile_result["success"] and attempt <= max_attempts:
            print(f"\n Tentativo di correzione #{attempt}...")

            with open(os.path.join(save_dir, out_file), 'r') as f:
                original_code = f.read()

            corrected_code = fix_code(
                timestamp=timestamp, 
                original_code=original_code,
                stderr=compile_result["stderr"],
                model=model  
            )

            with open(os.path.join(save_dir, out_file), 'w') as f:
                f.write(corrected_code)

            compile_result = self.compile(save_dir=save_dir)
            attempt += 1

        if compile_result["success"]:
            print(f"Error solved in {attempt} attempts")
        else:
            print(f"No corrections after {max_attempts} attempts.")
            # max_attempts = 3
            # attempt = 1