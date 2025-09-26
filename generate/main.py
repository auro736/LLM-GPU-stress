from utils.utils import *
from utils.parser import my_parser

from compilingAssistantAgent import CompilingAssistantAgent
from codeParser import CodeParser

from cudaExpertAgent import CudaExpertAgent
from optimizerAgent import OptimizerAgent

import os
import sys
import json
import math
import subprocess
from datetime import datetime

"""GESTISCI MEGLIO IL PARSING IO CASTEREI TUTTO A CUDA, SUPPONENDO CHE CI GENERA SOLO CUDA CODE
MAGARI DA SCRIVERE MEGLIO IL PROMPT DEL DEBUGGER""" #DONE MA SCHIFEZZA VERIFICA MEGLIO


def main():
    
    args = my_parser()
    t = int(round(datetime.now().timestamp()))

    if args.model_type == 'together':
        output_dir = f'./outputs/{args.model.split("/")[1]}/{t}'
    else:
        output_dir = f'./outputs/{args.model}/{t}'
    os.makedirs(output_dir, exist_ok=True)


    # CONTROLLARE TEST DURATION = QUANTO DEVE GIRARE IL CODICE

    cuda_expert_agent = CudaExpertAgent(
        model_type=args.model_type, 
        model_name=args.model, 
        api_key=args.api_key,
        history_max_turns=5, 
        enable_history=True
    )

    gpu_char = "one RTX 6000 Ada generation GPU with CUDA 12 and 48GB VRAM"
    test_duration = args.test_duration

    answer = cuda_expert_agent.generate(
        gpu_char=gpu_char, 
        test_duration=test_duration, 
        temperature=args.cuda_temperature, 
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
    
    base_file_name = f"{out_file.split('.')[0]}V0"
    dir_eval = f'../evaluate/cupti/02_profiling_injection/test-apps/{base_file_name}'
    os.makedirs(dir_eval, exist_ok=True)

    try:
        with open(os.path.join(dir_eval, out_file), 'w') as file:
            file.write(final_code)
    except:
        raise Exception(f"Error while saving {code_type} file into eval folder")
    

    make_template_dir = './utils/make_template'

    compilerAgent = CompilingAssistantAgent(template_dir=make_template_dir, model_type=args.model_type, model_name=args.model, api_key=args.api_key)
    
    compilerAgent.prepare_makefile(file_name=base_file_name, out_file=out_file, save_dir=dir_eval)
    compile_result = compilerAgent.compile(save_dir=dir_eval)
    print(compile_result)

    max_correction_attempts = args.max_correction_attempts #####
    attempt = 1

    if not compile_result["success"]:
        try:
            compilerAgent.fix_compile(
                max_attempts=max_correction_attempts,
                attempt=attempt, 
                compile_result=compile_result, 
                save_dir=dir_eval, 
                out_file=out_file, 
                timestamp=t,
                temperature=args.compiling_temperature,
                max_new_tokens=None, 
                seed=4899
            )
            
        except Exception as e:
            print(e)
            sys.exit(1)

    code_parser.adaptCode(file_name=base_file_name)

    gpu_id = 0

    command = ["sudo", "bash", f"exe/complete_stress_profile.sh", f"{base_file_name}", f"{gpu_id}"]
    result = subprocess.run(command,
                   capture_output=True,
                   text=True,
                   cwd="../evaluate/cupti/02_profiling_injection/" )
    
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)

    # file_name = "out1754575644"
    # out_file = "out_1754575644.cu"

    dir_eval = f'../evaluate/cupti/02_profiling_injection/test-apps/{base_file_name}'
    with open(os.path.join(dir_eval, out_file), 'r') as f:
        current_code = f.read()

    metrics_path = f'../evaluate/cupti/02_profiling_injection/data/postprocessed/stress2/{base_file_name}_evaluation.json' 
    with open(metrics_path, 'r') as f:
        current_metrics_raw = json.load(f)

    current_metrics = current_metrics_raw[f'{base_file_name}']
    print("Initial metrics:", current_metrics)

    optimizer_agent = OptimizerAgent(model_type=args.model_type, model_name=args.model, api_key=args.api_key)

    max_optimization_attempts = args.max_optimization_attempts
    current_version = 0 

    optimization_mode = args.optimization_mode

    iteration_count = 0
    target_reached = False

    for run in range(1, max_optimization_attempts + 1):
        iteration_count += 1
        
        # Check safety counter first
        if iteration_count > max_optimization_attempts:
            print(f"\n⚠️  Maximum iterations ({max_optimization_attempts}) reached. Stopping optimization to prevent infinite loop.")
            break

        if optimization_mode == 'clocks':
            max_objective_metric = current_metrics['Max Clock Frequency MHz']
            epsilon = args.epsilon
            current_objective_metric = current_metrics['Clock Frequency MHz']
        else:
            max_objective_metric = current_metrics["Max Temp °C"]
            epsilon = args.epsilon
            current_objective_metric = current_metrics["Steady Temp °C"]
            if math.isnan(current_objective_metric):
                print("⚠️ The test duration is insufficient to obtain a steady temperature reading. Please increase the execution time.")
                break


        ratio = round(current_objective_metric/max_objective_metric, 2)
        print(f"\n--- Optimization Run {run}/{max_optimization_attempts} (Iteration {iteration_count}) ---")
        print(f"Current {optimization_mode} metric: {current_objective_metric}")
        print(f"Target ratio: {epsilon}, Current ratio: {ratio}")
        
        # Check if target is reached
        if ratio >= epsilon:
            print(f"🎯 Target reached! Ratio {ratio} >= {epsilon}. Stopping optimization.")
            target_reached = True
            break

        suggestions = optimizer_agent.generate(
        final_code=current_code, 
        metrics=current_metrics, 
        temperature=args.perfomance_temperature, 
        max_new_tokens=None, 
        seed=4899
        )
        
        print("Optimization suggestions:", suggestions)

        # Add suggestions to history and generate new code
        cuda_expert_agent.add_to_history("user", suggestions)
        
        new_code_response = cuda_expert_agent.generate(
            gpu_char=gpu_char, 
            test_duration=test_duration, 
            temperature=args.cuda_temperature, 
            max_new_tokens=None, 
            seed=4899
        )

        # Process the new code
        final_code, code_type = clean_string(new_code_response)
        code_parser = CodeParser(code_string=final_code, code_type=code_type)
        processed_code, _ = code_parser.extract_code_from_output(timestamp=t)

        # Create new version names
        new_version = current_version + 1
        new_file_name = f"{out_file.split('.')[0]}V{new_version}"
        new_out_file = f"{new_file_name}.cu"
        
        # Create directory for new version
        new_dir_eval = f'../evaluate/cupti/02_profiling_injection/test-apps/{new_file_name}'
        os.makedirs(new_dir_eval, exist_ok=True)
        
        # Save the new code
        try:
            with open(os.path.join(new_dir_eval, new_out_file), 'w') as file:
                file.write(processed_code)
            print(f"Saved optimized code to: {new_out_file}")
        except Exception as e:
            print(f"Error saving optimized code: {e}")
            continue
        
        # Prepare and compile
        compilerAgent.prepare_makefile(file_name=new_file_name, out_file=new_out_file, save_dir=new_dir_eval)
        compile_result = compilerAgent.compile(save_dir=new_dir_eval)
        print(f"Compilation result: {compile_result}")

        # Fix compilation errors if needed
        if not compile_result["success"]:
            try:
                compilerAgent.fix_compile(
                    max_attempts=max_correction_attempts,
                    attempt=1,  # Reset attempt counter for each optimization run
                    compile_result=compile_result, 
                    save_dir=new_dir_eval, 
                    out_file=new_out_file, 
                    timestamp=t,
                    temperature=args.compiling_temperature,
                    max_new_tokens=None, 
                    seed=4899
                )
            except Exception as e:
                print(f"Failed to fix compilation errors: {e}")
                continue  # Skip this optimization run and try the next one
        
        # Adapt and profile the code
        code_parser.adaptCode(file_name=new_file_name)

        command = ["sudo", "bash", f"exe/complete_stress_profile.sh", f"{new_file_name}", f"{gpu_id}"]
        result = subprocess.run(command,
                    capture_output=True,
                    text=True,
                    cwd="../evaluate/cupti/02_profiling_injection/")
        
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)

        print(cuda_expert_agent.get_history())

        # Read the final compiled code and new metrics
        try:
            with open(os.path.join(new_dir_eval, new_out_file), 'r') as f:
                current_code = f.read()

            new_metrics_path = f'../evaluate/cupti/02_profiling_injection/data/postprocessed/stress2/{new_file_name}_evaluation.json' 
            with open(new_metrics_path, 'r') as f:
                current_metrics_raw = json.load(f)

            current_metrics = current_metrics_raw[f'{new_file_name}']
            print(f"New metrics for V{new_version}:", current_metrics)
            
            # Update current version for next iteration
            current_version = new_version
            
        except Exception as e:
            print(f"Error reading new code/metrics: {e}")
            # If we can't read the new metrics, we should probably break or use previous version
            break
    
    summary_path = save_optimization_summary(
        iteration_count=iteration_count,
        current_version=current_version,
        optimization_mode=optimization_mode,
        current_objective_metric=current_objective_metric,
        target_reached=target_reached,
        ratio=ratio,
        epsilon=epsilon,
        max_correction_attempts=max_correction_attempts,
        max_optimization_attempts=max_optimization_attempts,
        max_objective_metric=max_objective_metric,
        current_metrics=current_metrics,
        output_dir=output_dir,
        timestamp=t,
        test_duration=test_duration
    )
    print(summary_path)

   



if __name__ == '__main__':
    
    main()