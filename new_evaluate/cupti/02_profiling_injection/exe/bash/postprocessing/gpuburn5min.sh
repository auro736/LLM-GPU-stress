INJECTION_KERNEL_COUNT=$1
inter='_'
PERFORMANCE=$2

export PATH="/home/bepi/anaconda3/bin:$PATH"
source /home/bepi/anaconda3/bin/activate
conda deactivate 

conda activate gpustress
# idxs=(0 1 2 3 4 5)
# for idx in "${idxs[@]}"; do
    # APP_NAME=gpuburn5min${idx}
    APP_NAME=gpuburn5min
    python3 exe/parse_data.py --file_name "$APP_NAME$inter$INJECTION_KERNEL_COUNT" --performance $PERFORMANCE
# done