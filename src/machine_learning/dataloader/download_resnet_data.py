import os
import zipfile
import openi

# 定义下载函数，包含进度打印
def download_and_unzip_resnet_datasets(method:str):
    if method not in ["pad", "crop"]:
        raise ValueError(f"Invalid method '{method}'. Valid methods are 'pad' and 'crop'.")
    if method == "pad":
        openi.download_file(repo_id="enter/medical_resnet", file="padding_datasets.zip", cluster="NPU",
                            save_path=".",
                            force=False)
        zip_file_path = 'padding_datasets.zip'
    else:
        openi.download_file(repo_id="enter/medical_resnet", file="crop_datasets.zip", cluster="NPU",
                            save_path=".",
                            force=False)
        zip_file_path = 'crop_datasets.zip'

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

def download_and_unzip_resnet_checkpoints(method:str):
    if method not in ["pad", "crop"]:
        raise ValueError(f"Invalid method '{method}'. Valid methods are 'pad' and 'crop'.")
    if method == "pad":
        openi.download_model(repo_id="enter/medical_resnet",model_name="medical_resnet_checkpoints_pad",save_path=".")
        zip_file_path = "medical_resnet_checkpoints(pad).zip"
    else:
        openi.download_model(repo_id="enter/medical_resnet", model_name="medical_resnet_checkpoints_crop", save_path=".")
        zip_file_path = "medical_resnet_checkpoints(crop).zip"

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

def download_resnet_om(method:str):
    if method not in ["pad", "crop"]:
        raise ValueError(f"Invalid method '{method}'. Valid methods are 'pad' and 'crop'.")
    openi.download_model(repo_id="enter/nodule_segmentation",
                         model_name=f"medical_resnet({method})",save_path=".")

    model_path = 'medical_resnet.om'
    # 检查ZIP文件是否存在
    if os.path.exists(model_path):
        print('模型文件 medical_resnet.om 下载已完成')

def download_pathological_images():
    openi.download_file(repo_id="enter/nodule_segmentation", file="pathological_images_to_show.zip", cluster="NPU",
                        save_path=".",
                        force=False)
    zip_file_path = 'pathological_images_to_show.zip'

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

    print('病理图片数据下载成功！')


__all__ = ['download_and_unzip_resnet_datasets',
           'download_and_unzip_resnet_checkpoints',
           'download_resnet_om',
           "download_pathological_images"]
