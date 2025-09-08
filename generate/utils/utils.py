import json
import os
from datetime import datetime

def save_optimization_summary(
    iteration_count,
    current_version,
    optimization_mode,
    current_objective_metric,
    target_reached,
    ratio,
    epsilon,
    max_iterations,
    max_runs,
    max_objective_metric,
    current_metrics,
    output_dir,
    timestamp
):
    """
    Save optimization summary to both markdown file and console output.
    
    Args:
        iteration_count (int): Total number of iterations run
        current_version (int): Final version number
        optimization_mode (str): Mode of optimization ('clocks' or 'temperature')
        current_objective_metric (float): Final value of the objective metric
        target_reached (bool): Whether the target ratio was reached
        ratio (float): Final ratio achieved
        epsilon (float): Target epsilon threshold
        max_iterations (int): Maximum iterations limit
        max_runs (int): Maximum planned runs
        max_objective_metric (float): Maximum possible value for the objective metric
        current_metrics (dict): Final metrics dictionary
        output_dir (str): Directory to save the summary file
        timestamp (int/str): Timestamp for file naming
    
    Returns:
        str: Path to the saved summary file, or None if saving failed
    """
    
    # Create optimization summary content
    summary_content = f"""# Optimization Summary

    ## Results Overview
    - **Total iterations run**: {iteration_count}
    - **Final version**: V{current_version}
    - **Optimization mode**: {optimization_mode}
    - **Final {optimization_mode} metric**: {current_objective_metric}

    ## Exit Condition
    """

    if target_reached:
        summary_content += f"""✅ **SUCCESS**: Target ratio {epsilon} was reached!
- **Final ratio**: {ratio}
"""
    elif iteration_count > max_iterations:
        summary_content += f"""⚠️ **STOPPED**: Maximum iterations ({max_iterations}) reached
- Process stopped to prevent infinite loop
"""
    else:
        summary_content += f"""📝 **COMPLETED**: All {max_runs} planned runs executed
- Optimization completed normally
"""

    summary_content += f"""
## Configuration
- **Target epsilon**: {epsilon}
- **Max objective metric**: {max_objective_metric}
- **Max runs planned**: {max_runs}
- **Max iterations limit**: {max_iterations}

## Final Metrics
```json
{json.dumps(current_metrics, indent=2)}
```

---
*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    # Save to markdown file
    summary_filename = f"optimization_summary_{timestamp}_V{current_version}.md"
    summary_path = os.path.join(output_dir, summary_filename)

    try:
        with open(summary_path, 'w') as f:
            f.write(summary_content)
        print(f"✅ Optimization summary saved to: {summary_path}")
        file_saved = True
        return_path = summary_path
    except Exception as e:
        print(f"❌ Error saving optimization summary: {e}")
        file_saved = False
        return_path = None

    # Print to console
    print(f"\n{'='*50}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total iterations run: {iteration_count}")
    print(f"Final version: V{current_version}")

    if target_reached:
        print(f"✅ SUCCESS: Target ratio {epsilon} was reached!")
        print(f"Final ratio: {ratio}")
    elif iteration_count > max_iterations:
        print(f"⚠️ STOPPED: Maximum iterations ({max_iterations}) reached")
    else:
        print(f"📝 COMPLETED: All {max_runs} planned runs executed")

    print(f"Optimization mode: {optimization_mode}")
    print(f"Final {optimization_mode} metric: {current_objective_metric}")
    
    if file_saved:
        print(f"Summary saved to: {summary_filename}")
    else:
        print("Summary could not be saved to file")

    return return_path



def clean_string(text):
    """
    Removes ```json at the beginning and ``` at the end of a string if present,
    and ensures the string starts with { and ends with }.

    Args:
    text (str): The input string to clean

    Returns:
    str: The cleaned string, or None if it doesn't start with { and end with }
    """
    text = text.strip()
    code = ''
    print('BBBB', text)
    if text.startswith('```cpp'):
        print('sono in if cpp')
        text = text[6:].strip() 
        code = 'cpp'

    if text.startswith('```cuda'):
        print('sono in if cuda')
        text = text[7:].strip() 
        code = 'cuda'
        
    if text.startswith('```cu'):
        print('sono in if cu')
        text = text[5:].strip() 
        code = 'cuda'
    
    if text.startswith('```c'):
        print('sono in if c')
        text = text[4:].strip() 
        code = 'c'

    if text.startswith('```json'):
        print('sono in if json')
        text = text[7:].strip() 
        code = 'json'

    if text.startswith('```'):
        print('sono in no name')
        text = text[3:].strip() 
        code = 'cuda'

    if text.endswith('```'):
        text = text[:-3].strip()

    return text, code



