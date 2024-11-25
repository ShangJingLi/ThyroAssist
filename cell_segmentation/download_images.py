import os
import zipfile
import openi


openi.download_file(repo_id="enter/nodule_segmentation", file="image_to_show.zip", cluster="NPU",
                            save_path=".",
                            force=False)
zip_file_path = 'image_to_show.zip'

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