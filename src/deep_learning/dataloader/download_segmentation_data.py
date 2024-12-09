import os
import zipfile
import openi


# 定义下载函数，包含进度打印
def download_and_unzip_segmentation_datasets():
    openi.download_file(repo_id="enter/nodule_segmentation", file="datasets_as_numpy.zip", cluster="NPU",
                        save_path=".",
                        force=False)
    zip_file_path = 'datasets_as_numpy.zip'

    # 检查ZIP文件是否存在
    if os.path.exists(zip_file_path):
        # 使用with语句打开ZIP文件，确保文件正确关闭
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # 解压ZIP文件到当前目录
            zip_ref.extractall('.')

        # 解压完成后删除ZIP文件
        os.remove(zip_file_path)
        print(f'文件 {zip_file_path} 已解压并删除。')
    else:
        print(f'文件 {zip_file_path} 不存在。')

    print('数据集下载和解压已完成')


def download_and_unzip_unet_checkpoints():
    openi.download_file(repo_id="enter/nodule_segmentation", file="unet_checkpoints.zip", cluster="NPU",
                        save_path=".",
                        force=False)
    zip_file_path = 'unet_checkpoints.zip'

    # 检查ZIP文件是否存在
    if os.path.exists(zip_file_path):
        # 使用with语句打开ZIP文件，确保文件正确关闭
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # 解压ZIP文件到当前目录
            zip_ref.extractall('.')

        # 解压完成后删除ZIP文件
        os.remove(zip_file_path)
        print(f'文件 {zip_file_path} 已解压并删除。')
    else:
        print(f'文件 {zip_file_path} 不存在。')

    print('数据集下载和解压已完成')


def download_and_unzip_nested_unet_checkpoints():
    openi.download_file(repo_id="enter/nodule_segmentation", file="nested_unet_checkpoints.zip", cluster="NPU",
                        save_path=".",
                        force=False)
    zip_file_path = 'nested_unet_checkpoints.zip'

    # 检查ZIP文件是否存在
    if os.path.exists(zip_file_path):
        # 使用with语句打开ZIP文件，确保文件正确关闭
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # 解压ZIP文件到当前目录
            zip_ref.extractall('.')

        # 解压完成后删除ZIP文件
        os.remove(zip_file_path)
        print(f'文件 {zip_file_path} 已解压并删除。')
    else:
        print(f'文件 {zip_file_path} 不存在。')

    print('数据集下载和解压已完成')

__all__ = ['download_and_unzip_segmentation_datasets',
           'download_and_unzip_unet_checkpoints',
           'download_and_unzip_nested_unet_checkpoints']
