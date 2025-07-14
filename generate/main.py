from models.localModel import LocalModel
from models.openaiModel import OpenAIModel
from models.togetherModel import TogetherModel

from utils.utils import *
from utils.parser import my_parser

import os
import argparse
from datetime import datetime

def main():
    
    args = my_parser()
    t = int(round(datetime.now().timestamp()))

    output_dir = f'./outputs/{args.model.split("/")[1]}'
    os.makedirs(output_dir, exist_ok=True)

    system_prompt = get_system_prompt(mode=args.mode)
    user_prompt = get_user_prompt(mode=args.mode)

    messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                ]
    

    model = TogetherModel(model_name=args.model, api_key=args.together_api_key)
    answer = model.generate(messages=messages, temperature=0.7, max_new_tokens=None, seed=4899)
    print(answer)

    clean_answer, code_type = clean_string(answer)

    if code_type == 'cpp':
        out_file = f'out_{t}.cpp'
    if code_type == 'cuda':
        out_file = f'out_{t}.cu'
    
    try:
        with open(os.path.join(output_dir, out_file), 'w') as file:
            file.write(clean_answer)
    except:
        raise Exception("Error while saving {code_type} file")


if __name__ == '__main__':
    
    main()