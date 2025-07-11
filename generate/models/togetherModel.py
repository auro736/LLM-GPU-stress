import os
import time
from together import Together


class TogetherModel():

    def __init__(self, model_name: str, api_key: str = None):

            self.model_name = model_name
            
            # Set API key from args or environment
            if api_key:
                os.environ["TOGETHER_API_KEY"] = api_key
            elif "TOGETHER_API_KEY" not in os.environ:
                raise ValueError("TOGETHER_API_KEY environment variable must be set")
                
            self.client = Together()
            
    def generate(self, messages, temperature, max_new_tokens, seed=None): 
        
        # Make API call with retry logic for rate limits
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    seed=seed
                )
                return response.choices[0].message.content
                
            except Exception as e:
                retry_count += 1
                print(f"API error: {e}, retrying ({retry_count}/{max_retries})...")
                if "rate_limit" in str(e).lower():
                    time.sleep(5)  # Wait 5 seconds before retrying on rate limit
                else:
                    time.sleep(1)
                
                if retry_count == max_retries:
                    print(f"Failed after {max_retries} retries")
                    return f"Error: API failed to respond after {max_retries} attempts. Error: {str(e)}"