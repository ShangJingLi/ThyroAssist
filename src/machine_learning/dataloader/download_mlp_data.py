import os
import zipfile
import openi
from launcher import get_project_root


download_dir = get_project_root()

# 定义下载函数，包含进度打印
def download_and_unzip_mlp_datasets():

    openi.download_file(repo_id="enter/nodule_segmentation", file="mlp_datasets.zip", cluster="NPU",
                        save_path=download_dir,
                        force=False)
    zip_file_path = os.path.join(download_dir, 'mlp_datasets.zip')

    # 检查ZIP文件是否存在
    if os.path.exists(zip_file_path):
        # 使用with语句打开ZIP文件，确保文件正确关闭
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # 解压ZIP文件到当前目录
            zip_ref.extractall(download_dir)

        # 解压完成后删除ZIP文件
        os.remove(zip_file_path)
        print(f'文件 {zip_file_path} 已解压并删除。')
    else:
        print(f'文件 {zip_file_path} 不存在。')

    print('数据集下载和解压已完成')


def download_and_unzip_best_mlp_checkpoints():
    openi.download_file(repo_id="enter/nodule_segmentation", file="mlp_checkpoints.zip", cluster="NPU",
                        save_path=download_dir,
                        force=False)
    zip_file_path = os.path.join(download_dir, 'mlp_checkpoints.zip')

    # 检查ZIP文件是否存在
    if os.path.exists(zip_file_path):
        # 使用with语句打开ZIP文件，确保文件正确关闭
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # 解压ZIP文件到当前目录
            zip_ref.extractall(download_dir)

        # 解压完成后删除ZIP文件
        os.remove(zip_file_path)
        print(f'文件 {zip_file_path} 已解压并删除。')
    else:
        print(f'文件 {zip_file_path} 不存在。')

    print('数据集下载和解压已完成')

__all__ = ['download_and_unzip_mlp_datasets',
           'download_and_unzip_best_mlp_checkpoints']
