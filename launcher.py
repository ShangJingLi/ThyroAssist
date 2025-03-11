import subprocess
import sys
import os

def run_resnet_pathological_image_classification():
    script_path = os.path.join(os.path.dirname(__file__), "demo", "resnet_pathological_image_classification.py")
    subprocess.run([sys.executable, script_path])

def run_demo2():
    script_path = os.path.join(os.path.dirname(__file__), "demo", "nested_unet_ultrasound_image_infer.py")
    subprocess.run([sys.executable, script_path])
