from models.localModel import LocalModel
from models.openaiModel import OpenAIModel
from models.togetherModel import TogetherModel

from utils.utils import *
from utils.parser import my_parser

from compiler import Compiler

import os
import subprocess
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
    # maybe add into user prompt or here as var the duration time of the test
    # or say alway to add a user input to define test duration

    model = TogetherModel(model_name=args.model, api_key=args.together_api_key)
    answer = model.generate(messages=messages, temperature=0.5, max_new_tokens=None, seed=4899)
    # add args for temp
    print(answer)

    clean_answer, code_type = clean_string(answer)

    if code_type == 'cpp':
        out_file = f'out_{t}.cu' #forzato a .cu se no si rompe tutto durante compilazione
    if code_type == 'cuda':
        out_file = f'out_{t}.cu'
    
    clean_answer = extract_code_from_output(clean_answer)
    
    try:
        with open(os.path.join(output_dir, out_file), 'w') as file:
            file.write(clean_answer)
    except:
        raise Exception("Error while saving {code_type} file")
    
    # file_name = out_file.split('.')[0]
    file_name = out_file.split('.')[0].replace('_', '')


    dir_eval = f'../new_evaluate/cupti/02_profiling_injection/test-apps/{file_name}'
    os.makedirs(dir_eval, exist_ok=True)

    try:
        with open(os.path.join(dir_eval, out_file), 'w') as file:
            file.write(clean_answer)
    except:
        raise Exception("Error while saving {code_type} file into eval folder")
    

    make_template_dir = './utils/make_template'

    makeCompiler = Compiler(template_dir=make_template_dir)
    makeCompiler.prepare_makefile(file_name=file_name, out_file=out_file, save_dir=dir_eval)
    compile_result = makeCompiler.compile(save_dir=dir_eval)
    print(compile_result)

    max_attempts = 3
    attempt = 1
    if not compile_result["success"]:
        makeCompiler.fix_compile(max_attempts=max_attempts,
                                 attempt=attempt, 
                                 compile_result=compile_result, 
                                 save_dir=dir_eval, 
                                 out_file=out_file, 
                                 timestamp=t, 
                                 model=model)
        # controlla poi prompt per correzioni codici

   

    profiling_bash_template = './utils/profiling_bash_template'
    with open(os.path.join(profiling_bash_template,"template.sh"), "r") as f:
        content = f.read()
    
    content = content.replace("./test-apps/rora/rora 60", f"./test-apps/{file_name}/{file_name}")
    content = content.replace("data/raw/stress2/rora_$INJECTION_KERNEL_COUNT.txt", f"data/raw/stress2/{file_name}_$INJECTION_KERNEL_COUNT.txt")
    
    with open(f"../new_evaluate/cupti/02_profiling_injection/exe/bash/profiling_stress2/{file_name}.sh", "w") as f:
        f.write(content)

    postprocessing_bash_template = './utils/postprocessing_bash_template'
    with open(os.path.join(postprocessing_bash_template,"template.sh"), "r") as f:
        content = f.read()

    content = content.replace("APP_NAME='rora'", f"APP_NAME='{file_name}'")
    with open(f"../new_evaluate/cupti/02_profiling_injection/exe/bash/postprocessing/{file_name}.sh", "w") as f:
        f.write(content)

    command = ["sudo", "bash", "exe/complete_stress_profile.sh", file_name]
    result = subprocess.run(command,
                   capture_output=True,
                   text=True,
                   cwd="../new_evaluate/cupti/02_profiling_injection/" )
    
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)



if __name__ == '__main__':
    
    main()