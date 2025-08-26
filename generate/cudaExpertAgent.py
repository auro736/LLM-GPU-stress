from typing import List, Dict, Optional
from baseAgent import BaseAgent


class CudaExpertAgent(BaseAgent):
    """
    CUDA/C++ stress-testing code generator agent with chat history support.
    """

    def __init__(
        self,
        model_type: str,
        model_name: str,
        api_key: str,
        *,
        history_max_turns: int = 20,   # number of (user, assistant) pairs to retain
        enable_history: bool = True
    ):
        super().__init__(model_type, model_name, api_key)

        # --- Chat history state ---
        # Stores only user/assistant messages; system prompt is always injected fresh.
        self.history: List[Dict[str, str]] = []
        self.history_max_turns: int = history_max_turns
        self.enable_history: bool = enable_history

        self.system_prompt = """
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

    def add_to_history(self, role: str, content: str) -> None:
        """Append a single user/assistant message to history and truncate if needed."""
        if role not in ("user", "assistant"):
            raise ValueError("Only 'user' or 'assistant' roles can be stored in history.")
        if not self.enable_history:
            return
        self.history.append({"role": role, "content": content})
        self._truncate_history()

    def clear_history(self) -> None:
        """Remove all stored user/assistant turns."""
        self.history.clear()

    def get_history(self) -> List[Dict[str, str]]:
        """Return a shallow copy of the stored history (without the system prompt)."""
        return list(self.history)

    def generate(
        self,
        gpu_char: str,
        test_duration: int,
        temperature: float,
        max_new_tokens: int,
        seed: Optional[int],
    ) -> str:
        self.user_prompt = (
            f""" Following your system prompt your target for stressing is: {gpu_char}. """
            f"""Test duration time: {test_duration} seconds. Enclose code between ```  """
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        if self.enable_history and self.history:
            messages.extend(self.history)
        messages.append({"role": "user", "content": self.user_prompt})

        # Call the underlying model.
        answer: str = super().generate(
            messages=messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )

        self.add_to_history("user", self.user_prompt)
        self.add_to_history("assistant", answer)

        return answer

    def _truncate_history(self) -> None:
        """
        Keep only the most recent N turns (user+assistant pairs).
        This assumes history is an alternating sequence of user/assistant messages.
        If the sequence is odd-length, we still count pairs by floor division.
        """
        if not self.enable_history or self.history_max_turns <= 0:
            return

        # Compute how many messages correspond to the limit of turns (each turn ~2 messages).
        max_messages = self.history_max_turns * 2

        # If we exceed the message cap, drop the oldest ones.
        if len(self.history) > max_messages:
            overflow = len(self.history) - max_messages
            # Drop from the front (oldest first)
            self.history = self.history[overflow:]
