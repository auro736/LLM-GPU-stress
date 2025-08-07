INJECTION_KERNEL_COUNT=$1
PERFORMANCE=$2
inter='_'


source /home/user/phd/venvs/stressEnv/bin/activate


APP_NAME='out1754575644'
python3 exe/parse_data.py --file_name "$APP_NAME$inter$INJECTION_KERNEL_COUNT" --performance $PERFORMANCE