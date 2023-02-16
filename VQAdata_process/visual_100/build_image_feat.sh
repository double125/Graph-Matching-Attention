nohup python3 preprocess_image.py --data trainval >./log_trainval.txt 2>&1 &
nohup python3 preprocess_image.py --data test >./log_test.txt 2>&1 &
