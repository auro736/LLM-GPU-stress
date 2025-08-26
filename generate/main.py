from models.localModel import LocalModel
from models.openaiModel import OpenAIModel
from models.togetherModel import TogetherModel

from utils.utils import *
from utils.parser import my_parser

from compilingAssistantAgent import CompilingAssistantAgent
from codeParser import CodeParser

from cudaExpertAgent import CudaExpertAgent
from optimizerAgent import OptimizerAgent

import os
import json
import subprocess
from datetime import datetime

"""DA METTERE CHE NON METTE _ NEL FILE .CU SISTEMA TUTTO DI CONSEGUENZA""" #DONE

"""IMPLEMENTARE HISTORY DI CUDA EXPERT AGENT""" # DONE

"""IMPLEMENTARE MODEL TYPE IN ARGS E API KEY GENERICA""" # DONE
"""METTERE IN ARGS LE TEMP"""

def main():
    
    args = my_parser()
    t = int(round(datetime.now().timestamp()))

    output_dir = f'./outputs/{args.model.split("/")[1]}'
    os.makedirs(output_dir, exist_ok=True)


    # CONTROLLARE TEST DURATION = QUANTO DEVE GIRARE IL CODICE

    cuda_expert_agent = CudaExpertAgent(
        model_type=args.model_type, 
        model_name=args.model, 
        api_key=args.api_key,
        history_max_turns=5, 
        enable_history=True
    )
    gpu_char = "two RTX 6000 Ada generation GPUs with CUDA 12 and 48GB VRAM each"
    test_duration = "120"
    answer = cuda_expert_agent.generate(
        gpu_char=gpu_char, 
        test_duration=test_duration, 
        temperature=0.5, 
        max_new_tokens=None, 
        seed=4899
    )
  
    print(answer)

    final_code, code_type = clean_string(answer)
    
    code_parser = CodeParser(code_string=final_code, code_type=code_type)
    final_code, out_file = code_parser.extract_code_from_output(timestamp=t)

    try:
        with open(os.path.join(output_dir, out_file), 'w') as file:
            file.write(final_code)
    except:
        raise Exception("Error while saving {code_type} file")
    
    file_name = out_file.split('.')[0]
    dir_eval = f'../evaluate/cupti/02_profiling_injection/test-apps/{file_name}'
    os.makedirs(dir_eval, exist_ok=True)

    try:
        with open(os.path.join(dir_eval, out_file), 'w') as file:
            file.write(final_code)
    except:
        raise Exception("Error while saving {code_type} file into eval folder")
    

    make_template_dir = './utils/make_template'

    compilerAgent = CompilingAssistantAgent(template_dir=make_template_dir, model_type=args.model_type, model_name=args.model, api_key=args.api_key)
    compilerAgent.prepare_makefile(file_name=file_name, out_file=out_file, save_dir=dir_eval)
    compile_result = compilerAgent.compile(save_dir=dir_eval)
    print(compile_result)

    max_attempts = 3
    attempt = 1

    if not compile_result["success"]:
        compilerAgent.fix_compile(
            max_attempts=max_attempts,
            attempt=attempt, 
            compile_result=compile_result, 
            save_dir=dir_eval, 
            out_file=out_file, 
            timestamp=t,
            temperature=0.5,
            max_new_tokens=None, 
            seed=4899
        )
        # controlla poi prompt per correzioni codici
        # metti var per temperatura e seed



    # profiling_bash_template = './utils/profiling_bash_template'
    # with open(os.path.join(profiling_bash_template,"template.sh"), "r") as f:
    #     content = f.read()
    
    # content = content.replace("./test-apps/rora/rora 60", f"./test-apps/{file_name}/{file_name} 60")
    # content = content.replace("data/raw/stress2/rora_$INJECTION_KERNEL_COUNT.txt", f"data/raw/stress2/{file_name}_$INJECTION_KERNEL_COUNT.txt")
    
    # with open(f"../evaluate/cupti/02_profiling_injection/exe/bash/profiling_stress2/{file_name}.sh", "w") as f:
    #     f.write(content)

    # postprocessing_bash_template = './utils/postprocessing_bash_template'
    # with open(os.path.join(postprocessing_bash_template,"template.sh"), "r") as f:
    #     content = f.read()

    # content = content.replace("APP_NAME='rora'", f"APP_NAME='{file_name}'")
    # with open(f"../evaluate/cupti/02_profiling_injection/exe/bash/postprocessing/{file_name}.sh", "w") as f:
    #     f.write(content)

    code_parser.adaptCode(file_name=file_name)

    command = ["sudo", "bash", f"exe/complete_stress_profile.sh", f"{file_name}"]
    result = subprocess.run(command,
                   capture_output=True,
                   text=True,
                   cwd="../evaluate/cupti/02_profiling_injection/" )
    
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)

   

    # file_name = "out1754575644"
    # out_file = "out_1754575644.cu"
    # dir_eval = f'../evaluate/cupti/02_profiling_injection/test-apps/{file_name}'
    # with open(os.path.join(dir_eval, out_file), 'r') as f:
    #             final_code = f.read()

    metrics_path = f'../evaluate/cupti/02_profiling_injection/data/postprocessed/stress2/{file_name}_evaluation.json'
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    print(metrics)

    optimizer_agent = OptimizerAgent(model_type=args.model_type, model_name=args.model, api_key=args.api_key)
    suggestions = optimizer_agent.generate(final_code=final_code, metrics=metrics, temperature=0.5, max_new_tokens=None, seed=4899)
    print(suggestions)

    cuda_expert_agent.add_to_history("user", suggestions)
    new_code = cuda_expert_agent.generate(
        gpu_char=gpu_char, 
        test_duration=test_duration, 
        temperature=0.5, 
        max_new_tokens=None, 
        seed=4899
    )




if __name__ == '__main__':
    
    main()