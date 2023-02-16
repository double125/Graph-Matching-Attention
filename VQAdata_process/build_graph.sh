mkdir logs
nohup python3 -u  preprocess_vqa_text.py >./logs/log_vqa.txt 2>&1 &
nohup python3 -u  preprocess_vg_text.py >./logs/log_vg.txt 2>&1 &

