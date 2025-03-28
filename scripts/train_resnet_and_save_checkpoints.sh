#!/bin/bash
# install the requirements
pip install -r requirements.txt

# start training
python train_and_eval/resnet/train_and_eval_resnet.py
#split checkpoints
python -c "import os; from thyassist.machine_learning.utils import split_ckpt; \
from launcher import get_project_root;download_dir = get_project_root();\
file_path = os.path.join(download_dir, 'medical_resnet_checkpoints', 'medical_resnet_checkpoints.ckpt'); \
split_ckpt(file_path, chunk_size=80 * 1024 * 1024)"