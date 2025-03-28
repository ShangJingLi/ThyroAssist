import os
import zipfile
import openi
from launcher import  get_project_root


download_dir = get_project_root()

# 定义下载函数，包含进度打印
def download_and_unzip_segmentation_datasets():
    openi.download_file(repo_id="enter/nodule_segmentation", file="datasets_as_numpy.zip", cluster="NPU",
                        save_path=download_dir,
                        force=False)
    zip_file_path = os.path.join(download_dir, 'datasets_as_numpy.zip')

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


def download_and_unzip_nested_unet_checkpoints():
    openi.download_file(repo_id="enter/nodule_segmentation", file="nested_unet_checkpoints.zip", cluster="NPU",
                        save_path=download_dir,
                        force=False)
    zip_file_path = os.path.join(download_dir, 'nested_unet_checkpoints.zip')

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

    print('模型下载和解压已完成')


def download_nested_unet_om():
    openi.download_model(repo_id="enter/nodule_segmentation",
                         model_name="nested_unet_om",save_path=download_dir)

    model_path = os.path.join(download_dir, 'nested_unet.om')
    # 检查ZIP文件是否存在
    if os.path.exists(model_path):
        print('模型文件 nested_unet.om 下载已完成')


def download_nested_unet_onnx():
    openi.download_model(repo_id="enter/nodule_segmentation",
                         model_name="nested_unet_onnx",save_path=download_dir)

    model_path = os.path.join(download_dir, 'nested_unet.onnx')
    # 检查ZIP文件是否存在
    if os.path.exists(model_path):
        print('模型文件 nested_unet.onnx 下载已完成')


def download_ultrasound_images():
    openi.download_file(repo_id="enter/nodule_segmentation", file="ultrasound_images_to_show.zip", cluster="NPU",
                        save_path=os.getcwd(),
                        force=False)
    zip_file_path = os.path.join(os.getcwd(), 'ultrasound_images_to_show.zip')

    # 检查ZIP文件是否存在
    if os.path.exists(zip_file_path):
        # 使用with语句打开ZIP文件，确保文件正确关闭
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # 解压ZIP文件到当前目录
            zip_ref.extractall(os.getcwd())

        # 解压完成后删除ZIP文件
        os.remove(zip_file_path)
        print(f'文件 {zip_file_path} 已解压并删除。')
    else:
        print(f'文件 {zip_file_path} 不存在。')

    print('超声图片数据下载成功！')


__all__ = ['download_and_unzip_segmentation_datasets',
           'download_and_unzip_nested_unet_checkpoints',
           'download_ultrasound_images',
           'download_nested_unet_om',
           'download_nested_unet_onnx']
