import re
import pandas as pd
import argparse

def get_argparser():
    parser = argparse.ArgumentParser(description='Postprocessing for .txt data')
    parser.add_argument('--file_name', required=True, help='Input file name (without extension)')
    parser.add_argument('--performance', required=True, help='Either Performance Metrics (PM) or Performance Counters (PC)')
    return parser

def main(args):
    # Legge il contenuto del file
    with open(f'data/raw/{args.performance}/{args.file_name}.txt', 'r') as file:
        content = file.read()
    # print('CONTENT: ', content)

    # Regex per trovare le sessioni
    session_pattern = re.compile(
        r"Context .*?session (\d+):\s*\[\s*durata:\s*([\d\-:\. ]+)\s*ms\s*\]:\s*\n+(.*?)(?=\nContext|\Z)",
        re.DOTALL
    )
    # Regex per una metrica: range, metrica, valore
    metric_pattern = re.compile(r"^\s*(\d+)\s+([a-zA-Z0-9_\.]+)\s+([\deE\+\-\.]+)\s*$", re.MULTILINE)
    
    # for i in session_pattern.finditer(content):
    #     print(i)

    # Lista di righe per il DataFrame
    df = pd.DataFrame([])
    # Processa tutte le sessioni
    for session_match in session_pattern.finditer(content):
        print('SESSION_MATCH: ', session_match)
        
        session_id = session_match.group(1)
        duration = session_match.group(2)
        session_block = session_match.group(3)

        for metric_match in metric_pattern.finditer(session_block):
            print('METRIC_MATCH: ', metric_match)

            post = 'No post'
            range_name = metric_match.group(1)
            metric_name = metric_match.group(2)
            metric_value = metric_match.group(3)

            # Scomponi la metrica in componenti logiche
            location = metric_name.split('__')[0]
            name = metric_name.split('__')[1].split('.')[0]
            rollup = metric_name.split('__')[1].split('.')[1]
            try:
                post = metric_name.split('__')[1].split('.')[2]
            except:
                print(f'No futher post processing')

            # Aggiungi al dataset
            new_row = pd.DataFrame({
                'session_id': session_id,
                'duration_ms': duration if duration else None,
                'location': location,
                'metric_name': name,
                'rollup_operation': rollup,
                'Post': post,
                'range_name': range_name,
                'metric_value': metric_value
            }, index=[0])

            df = pd.concat([df, new_row], ignore_index=True)

    # Salva il CSV
    df.to_csv(f'data/postprocessed/{args.performance}/{args.file_name}.csv', index=False)

    print("Metrics have been processed and saved.")

if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())