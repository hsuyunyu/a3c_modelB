mkdir trained_models > /dev/null 2>&1
mkdir logs > /dev/null 2>&1
python main.py --workers 8 --gpu-ids 0 --amsgrad True
