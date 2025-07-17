import argparse

def my_parser():

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct-Turbo", 
                    help="LLM model name to use for generation")
    parser.add_argument("--together_api_key", type=str, default=None, required=True, 
                    help="TogetherAI API key (uses env var if not provided)")
    parser.add_argument("--mode", type=str, default='zero-shot', 
                    help="TogetherAI API key (uses env var if not provided)")
    
    return parser.parse_args()
    