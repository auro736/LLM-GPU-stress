INJECTION_KERNEL_COUNT=$1
inter='_'
PERFORMANCE=$2

export PATH="/home/bepi/anaconda3/bin:$PATH"
source /home/bepi/anaconda3/bin/activate
conda deactivate 

conda activate gpustress

APP_NAME='NN50Percresnet18'
python3 exe/parse_data.py --file_name "$APP_NAME$inter$INJECTION_KERNEL_COUNT" --performance $PERFORMANCE