#!/bin/bash
# install the requirements
pip install -r requirements.txt

# start training
python train_and_eval/segmentation/train_nested_unet.py
python train_and_eval/segmentation/eval_and_infer_nested_unet.py

#split checkpoints
python -c "import os; from src.deep_learning.utils import split_ckpt; \
file_path = os.path.join('nested_unet_checkpoints', 'nested_unet_checkpoints.ckpt'); \
split_ckpt(file_path, chunk_size=80 * 1024 * 1024)"