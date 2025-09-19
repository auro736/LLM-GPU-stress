import argparse

def my_parser():

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct-Turbo", 
                    help="LLM model name to use for generation", required=True)
    parser.add_argument("--model_type", type=str, default="together", 
                    help="provider of LLMs", choices=['together', 'openai'], required=True)
    parser.add_argument("--api_key", type=str, default=None, required=True, 
                    help="API key")
    # parser.add_argument("--mode", type=str, default='zero-shot', 
    #                 help="TogetherAI API key (uses env var if not provided)")
    parser.add_argument("--optimization_mode", type=str, default='temp', choices=['temp', 'clocks'],
                    help="On which metric you want to optimize.", required=True)
    parser.add_argument("--epsilon", type=float, default=0.2, required=True, 
                    help="epsilon value for optimization")
    
    return parser.parse_args()
    