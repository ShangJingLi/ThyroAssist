#!/bin/bash
# install the requirements
pip install -r requirements.txt

# start training
python train_and_eval/segmentation/train_nested_unet.py
python train_and_eval/segmentation/eval_and_infer_nested_unet.py

#split checkpoints
python -c "import os; from thyassist.machine_learning.utils import split_ckpt; \
from launcher import get_project_root;download_dir = get_project_root();\
file_path = os.path.join(download_dir, 'nested_unet_checkpoints', 'nested_unet_checkpoints.ckpt'); \
split_ckpt(file_path, chunk_size=80 * 1024 * 1024)"