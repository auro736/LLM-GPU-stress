INJECTION_KERNEL_COUNT=$1
PERFORMANCE=$2
inter='_'

# export PATH="/home/bepi/anaconda3/bin:$PATH"
# source /home/bepi/anaconda3/bin/activate
# conda deactivate 

source /home/user/phd/venvs/stressEnv/bin/activate

# conda activate gpustress

APP_NAME='rora'
python3 exe/parse_data.py --file_name "$APP_NAME$inter$INJECTION_KERNEL_COUNT" --performance $PERFORMANCE