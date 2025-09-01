import os
import subprocess
from typing import Dict, Tuple, Optional

from utils.utils import clean_string
from baseAgent import BaseAgent
from codeParser import CodeParser


class CompilingAssistantAgent(BaseAgent):
    def __init__(self, model_type: str, model_name: str, api_key: str, template_dir: str):
        super().__init__(model_type, model_name, api_key)
        self.template_dir = template_dir
        self.system_prompt = "You are an expert debugger of compiling CUDA codes. Your task is to correct CUDA code that has errors during compilation. You will be provided with the error and the code, fix it in order to compile it."

    def prepare_makefile(self, file_name: str, out_file: str, save_dir: str) -> None:
        template_path = os.path.join(self.template_dir, "Makefile")
        target_path = os.path.join(save_dir, "Makefile")

        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()

        content = content.replace("TARGET := target", f"TARGET := {file_name}")
        content = content.replace("SRC := stress_test.cu", f"SRC := {out_file}")

        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)

        print("Makefile generated successfully.")

    def compile(self, save_dir: str) -> Dict[str, object]:
        try:
            result = subprocess.run(
                ["make", "-C", save_dir],
                capture_output=True,
                text=True
            )
            success = (result.returncode == 0)

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
            msg = f"make command not found: {str(e)}"
            print(f"Compiler error: 'make' command not found. Make sure it's installed and in PATH: {e}")
            return {"success": False, "stdout": "", "stderr": msg, "returncode": -1}

        except PermissionError as e:
            msg = f"Permission denied: {str(e)}"
            print(f"Compiler error: Permission denied accessing directory or running make: {e}")
            return {"success": False, "stdout": "", "stderr": msg, "returncode": -1}

        except subprocess.TimeoutExpired as e:
            msg = e.stderr or f"Command timed out: {str(e)}"
            print(f"Compiler error: Make command timed out: {e}")
            return {"success": False, "stdout": e.stdout or "", "stderr": msg, "returncode": -1}

        except OSError as e:
            msg = f"OS error: {str(e)}"
            print(f"Compiler error: OS error occurred: {e}")
            return {"success": False, "stdout": "", "stderr": msg, "returncode": -1}

        except Exception as e:
            msg = f"Unexpected error: {str(e)}"
            print(f"Compiler error: Unexpected error occurred: {e}")
            return {"success": False, "stdout": "", "stderr": msg, "returncode": -1}

    def fix_code(
        self,
        timestamp: str,
        original_code: str,
        stderr: str,
        temperature: float,
        max_new_tokens: Optional[int],
        seed: Optional[int]
    ) -> Tuple[str, Optional[str]]:
        """
        Use LLM to correct the code based on compiler stderr.

        Returns:
            Tuple[str, Optional[str]]: (clean_corrected_code, detected_output_filename_if_any)
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    "The following code:\n\n"
                    f"{original_code}\n\n"
                    "Generated this compilation error:\n\n"
                    f"{stderr}\n\n"
                    "Please correct the code. Return ONLY code, enclosed in triple backticks."
                    "If multiple files are needed, include clear file markers."
                )
            },
        ]

        response = super().generate(
            messages=messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )

        # Clean and parse the LLM response
        cleaned, code_type = clean_string(response)
        print('AAAAAAA', code_type)
        parser = CodeParser(code_string=cleaned, code_type=code_type)
        clean_answer, out_file = parser.extract_code_from_output(timestamp=timestamp)

        return clean_answer, out_file

    def fix_compile(
        self,
        max_attempts: int,
        attempt: int,
        compile_result: Dict[str, object],
        save_dir: str,
        out_file: str,
        timestamp: str,
        temperature: float,
        max_new_tokens: Optional[int],
        seed: Optional[int]
    ) -> None:
        """
        Iteratively try to fix the code using the compiler stderr until success or max attempts reached.
        """

        while not compile_result.get("success", False) and attempt <= max_attempts:
            print(f"\n Tentative #{attempt} to correct code ...")

            source_path = os.path.join(save_dir, out_file)
            with open(source_path, "r", encoding="utf-8") as f:
                original_code = f.read()

            corrected_code, _ = self.fix_code(
                timestamp=timestamp,
                original_code=original_code,
                stderr=str(compile_result.get("stderr", "")),
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )

            target_path = os.path.join(save_dir, out_file)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(corrected_code)

            compile_result = self.compile(save_dir=save_dir)
            attempt += 1

        # if compile_result.get("success", False):
        #     print(f"Error solved in {attempt - 1} attempts")
        # else:
        #     print(f"No corrections after {max_attempts} attempts.")
        if compile_result.get("success", False):
            print(f"Error solved in {attempt - 1} attempts")
            return

        raise Exception(
            f"Compilation failed after {max_attempts} attempts.\n"
            f"Last return code: {compile_result.get('returncode')}\n"
            f"Last stderr:\n{compile_result.get('stderr', '')}"
        )
