from baseAgent import BaseAgent

""" 
- Issued instructions : maximize
- Instruction throughput : maximize
- Total Stall = Memory + Controller + Throttle : minimize
"""

class OptimizerAgent(BaseAgent):

    def __init__(self, model_type, model_name, api_key):
        super().__init__(model_type, model_name, api_key)
        self.system_prompt = """
        You are a telemetry metrics optmizer for CUDA applications.
        You analyze the provided CUDA code and its telemetry metrics and performance in a JSON object.
        Your task: create a list of exact modifications the CUDA Expert Agent must apply to the code in order to optimize it according to the defined rules.
 
        ## Optimization Rules :
        - Steady Temp (°C) : maximize
        - Energy Spent (J/min) : maximize
        - Clock Frequency (MHz) : maximize
        - Response time (s) : minimize
        - Instruction throughput : maximize
 
        Always optimize toward these goals, even if unusual.
 
        ## Output format
 
        Write your answer as if you are a user giving direct instructions to the CUDA Expert.
 
        * Start with:
 
        ```
        Apply the following changes:
        ```
        * Then output a **numbered list** of concrete modifications.
        * Each item must contain:
 
        * **Location** (kernel/function/loop).
        * **Exact modification** (parameter value, launch config that gaurantee asynchronous threads execution, etc.).
        * Do **not** restate metrics, rationale, JSON, or explanations.
        * Do **not** output code unless the modification requires an attribute/flag.
        * Do **not** suggest to use CudaMallocPitch for more efficient memory access.
        """

    def generate(self, final_code, metrics, temperature, max_new_tokens, seed):
        self.user_prompt = f""" Following your system prompt, the code is: {final_code}. \n The resulting performance are: {metrics}"""
        messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt},
                ]
        answer = super().generate(messages=messages, temperature=temperature, max_new_tokens=max_new_tokens, seed=seed)
        return answer

    
