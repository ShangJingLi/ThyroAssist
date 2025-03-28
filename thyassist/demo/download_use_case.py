"""下载测试用例"""
from thyassist.machine_learning.dataloader import (download_pathological_images,
                                                   download_ultrasound_images,
                                                   download_and_unzip_mlp_datasets)


download_ultrasound_images()
download_pathological_images()
download_and_unzip_mlp_datasets(to_show=True)
