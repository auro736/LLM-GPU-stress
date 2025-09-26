import argparse

def my_parser():

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct-Turbo", 
                    help="LLM model name to use for generation", required=True)
    parser.add_argument("--model_type", type=str, default="together", 
                    help="provider of LLMs", choices=['together', 'openai'], required=True)
    parser.add_argument("--api_key", type=str, default=None, required=True, 
                    help="API key")
    parser.add_argument("--optimization_mode", type=str, default='temp', choices=['temp', 'clocks'],
                    help="On which metric you want to optimize.", required=True)
    parser.add_argument("--epsilon", type=float, default=0.2, required=True, 
                    help="epsilon value for optimization")
    parser.add_argument("--cuda_temperature", type=float, default=0.5, required=True, 
                    help="Temperature of CUDA expert agent")
    parser.add_argument("--compiling_temperature", type=float, default=0.5, required=True, 
                    help="Temperature of Compiling Assistant agent")
    parser.add_argument("--performance_temperature", type=float, default=0.5, required=True, 
                    help="Temperature of Performance optimizer agent")
    parser.add_argument("--test_duration", type=str, default="60", required=True, 
                    help="Test duration in seconds")
    parser.add_argument("--max_correction_attempts", type=int, default=3, required=True, 
                    help="Maximum number of correction attempts by the Compiling Assistant agent, if reached the script stops")
    parser.add_argument("--max_optimization_attempts", type=int, default=5, required=True, 
                    help="Maximum number of optmization attempts by the Performance Optmizer agent, if reached the script stops")


    return parser.parse_args()
    