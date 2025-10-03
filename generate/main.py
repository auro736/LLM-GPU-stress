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

""" in md controlla aggiornamento metriche""" #DONE
"""sistemare md in cui se finisce il loop del for lo dice""" #DONE
"""mettere in args nome cartella da usare al posto del timestamp nel main""" #DONE
"""salvare args in txt in cartella outputs""" #DONE
"""aggiungere salvataggio degli optimization suggestions nella cartella outputs""" #DONE
"""aggiungere controllo che i csv in data/postprocessed non siano vuoti altrimenti tira su eccezione""" #DONE
"""add gpu char in args""" #DONE

"""chech max tokens di together, sembra che con None il modello decide quando fermarsi https://docs.together.ai/docs/error-codes#error-codes """

def main():
    
    args = my_parser()

    t = args.folder_name

    if args.model_type == 'together':
        output_dir = f'./outputs/{args.model.split("/")[1]}/{t}'
    else:
        output_dir = f'./outputs/{args.model}/{t}'
    os.makedirs(output_dir, exist_ok=True)

    args_dict = vars(args)
    with open(output_dir + "/args.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(args_dict, indent=4, ensure_ascii=False))

    cuda_expert_agent = CudaExpertAgent(
        model_type=args.model_type, 
        model_name=args.model, 
        api_key=args.api_key,
        history_max_turns=5, 
        enable_history=True
    )

    gpu_char = args.gpu
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
        raise Exception(f"Error while saving {code_type} file")
    
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

    max_correction_attempts = args.max_correction_attempts 
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

    log = []    

    for run in range(1, max_optimization_attempts + 1):  #il for l'ha fatto 5 volte se max bla bla = 5
        iteration_count += 1
        
        # Check safety counter first
        # if iteration_count > max_optimization_attempts:
        #     print(f"\n⚠️  Maximum iterations ({max_optimization_attempts}) reached. Stopping optimization to prevent infinite loop.")
        #     break

        if optimization_mode == 'clocks':
            max_objective_metric = current_metrics['Max Clock Frequency MHz']
            epsilon = args.epsilon
            current_objective_metric = current_metrics['Clock Frequency MHz']
        else:
            max_objective_metric = current_metrics["Max Temp °C"]
            epsilon = args.epsilon
            current_objective_metric = current_metrics["Steady Temp °C"]
            if math.isnan(current_objective_metric):
                tmp = "⚠️ The test duration is insufficient to obtain a steady temperature reading. Please increase the execution time."
                print(tmp)
                log.append(tmp)
                break


        ratio = round(current_objective_metric/max_objective_metric, 2)

        update = f"""\n--- Optimization Run {run}/{max_optimization_attempts} (Iteration {iteration_count}) ---
        \n Current {optimization_mode} metric: {current_objective_metric} 
        \n Target ratio: {epsilon}, Current ratio: {ratio} \n"""

        # print(f"\n--- Optimization Run {run}/{max_optimization_attempts} (Iteration {iteration_count}) ---")
        # print(f"Current {optimization_mode} metric: {current_objective_metric}")
        # print(f"Target ratio: {epsilon}, Current ratio: {ratio}")
        print(update)
        log.append(update)
        log.append(f"METRICS OBTAINED AT ITERATION {iteration_count} with code version {new_version}: \n {current_metrics}")

        # Check if target is reached
        if ratio >= epsilon:
            tmp = f"🎯 Target reached! Ratio {ratio} >= {epsilon}. Stopping optimization."
            print(tmp)
            # target_reached = True
            log.append(tmp)
            break

        suggestions = optimizer_agent.generate(
        final_code=current_code, 
        metrics=current_metrics, 
        temperature=args.perfomance_temperature, 
        max_new_tokens=None, 
        seed=4899
        )
        
        print("Optimization suggestions:", suggestions)
        log.append("Target epsilon not reached, Optimization suggestions: \n" + suggestions)

        # Add suggestions to history and generate new code
        cuda_expert_agent.add_to_history("user", f"Follow these suggestions to modify the previous code: \n {suggestions}")
        
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
            log.append(f"Saved optimized code to: {new_out_file}")
        except Exception as e:
            print(f"Error saving optimized code: {e}")
            log.append(f"Error saving optimized code: {e}")
            continue
        
        # Prepare and compile
        compilerAgent.prepare_makefile(file_name=new_file_name, out_file=new_out_file, save_dir=new_dir_eval)
        compile_result = compilerAgent.compile(save_dir=new_dir_eval)
        print(f"Compilation result: {compile_result}")
        log.append(f"Compilation result: {compile_result}")

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
                log.append(f"Failed to fix compilation errors: {e}")
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

        # Read the final compiled code and new metrics
        try:
            with open(os.path.join(new_dir_eval, new_out_file), 'r') as f:
                current_code = f.read()

            new_metrics_path = f'../evaluate/cupti/02_profiling_injection/data/postprocessed/stress2/{new_file_name}_evaluation.json' 
            with open(new_metrics_path, 'r') as f:
                current_metrics_raw = json.load(f)

            current_metrics = current_metrics_raw[f'{new_file_name}']
            # add update also di current objective metric
            print(f"New metrics for V{new_version}:", current_metrics)
            
            # Update current version for next iteration
            current_version = new_version
            
        except Exception as e:
            print(f"Error reading new code/metrics: {e}")
            log.append(f"Error reading new code/metrics: {e}")
            # If we can't read the new metrics, we should probably break or use previous version
            break
    
    log.append("\n All defined iterations are completed. \n")
    with open(output_dir+'/optimization_log.txt', "w", encoding="utf-8") as f:
        json.dump(log, f, indent=4, ensure_ascii=False)



    # summary_path = save_optimization_summary(
    #     iteration_count=iteration_count,
    #     current_version=current_version,
    #     optimization_mode=optimization_mode,
    #     current_objective_metric=current_objective_metric,
    #     target_reached=target_reached,
    #     ratio=ratio,
    #     epsilon=epsilon,
    #     max_correction_attempts=max_correction_attempts,
    #     max_optimization_attempts=max_optimization_attempts,
    #     max_objective_metric=max_objective_metric,
    #     current_metrics=current_metrics,
    #     output_dir=output_dir,
    #     timestamp=t,
    #     test_duration=test_duration
    # )
    # print(summary_path)

   



if __name__ == '__main__':
    
    main()