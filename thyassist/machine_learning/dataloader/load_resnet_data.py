import os
import warnings
import multiprocess
import cv2
import numpy as np
import mindspore.dataset as ds
from mindspore.dataset.vision.transforms import Rescale, HWC2CHW


def boundary_padding(origin_images:np.array, padding, aim_size:tuple=(572, 572)):
    if len(aim_size) != 2:
        raise ValueError(f"The length of 'aim_size' must be 2, but got {len(aim_size)}!")

    if origin_images.ndim == 4:
        processed_images = np.ones(shape=(origin_images.shape[0], aim_size[0], aim_size[1], 3),
                                   dtype=np.uint8) * padding
        k = origin_images.shape[1] / origin_images.shape[2]  # k = 原始高 / 原始宽
        if k < 1:
            for i in range(origin_images.shape[0]):
                image = cv2.resize(origin_images[i], dsize=(aim_size[1], int(k * aim_size[1])))
                dist = (image.shape[1] - image.shape[0]) // 2
                processed_images[i][dist:dist+image.shape[0], :, :] = image
        else:
            for i in range(origin_images.shape[0]):
                image = cv2.resize(origin_images[i], dsize=(int(aim_size[1] / k), aim_size[1]))
                dist = (image.shape[0] - image.shape[1]) // 2
                processed_images[i][ :,dist:dist+image.shape[1], :] = image

        return processed_images

    elif origin_images.ndim == 3:
        k = origin_images.shape[0] / origin_images.shape[1]  # k = 原始高 / 原始宽
        if k < 1:
            image = cv2.resize(origin_images, dsize=(aim_size[1], int(k * aim_size[1])))
            processed_image = np.ones(shape=(aim_size[0], aim_size[1], 3), dtype=np.uint8) * padding
            dist = (image.shape[1] - image.shape[0]) // 2
            processed_image[dist:dist+image.shape[0], :, :] = image
        else:
            image = cv2.resize(origin_images, dsize=(int(aim_size[1] / k), aim_size[1]))
            processed_image = np.ones(shape=(aim_size[0], aim_size[1], 3), dtype=np.uint8) * padding
            dist = (image.shape[0] - image.shape[1]) // 2
            processed_image[ :,dist:dist+image.shape[1], :] = image
        return processed_image

    else:
        raise ValueError(f"'origin_images.ndim' must be 3 or 4, but got {origin_images.ndim}!")


def center_crop(origin_images:np.array, aim_size:tuple=(572, 572)):
    """保留原图像最中心的正方形部分，并将其像素值置为aim_size"""
    if len(aim_size) != 2:
        raise ValueError(f"The length of 'aim_size' must be 2, but got {len(aim_size)}!")
    if origin_images.ndim == 4:
        processed_images = np.ones(shape=(origin_images.shape[0], aim_size[0], aim_size[1], 3), dtype=np.uint8)
        k = origin_images.shape[2] / origin_images.shape[1]
        if k > 1:  # k:宽 / 高
            for i in range(origin_images.shape[0]):
                image = cv2.resize(origin_images[i], dsize=(int(k * aim_size[0]), aim_size[0]))
                processed_images[i] = image[:, int(0.5 * (k - 1) * aim_size[0]):
                                               int(0.5 * (k - 1) * aim_size[0]) + aim_size[0], :]
        else:
            for i in range(origin_images.shape[0]):
                image = cv2.resize(origin_images[i], dsize=(aim_size[1], int(aim_size[1] / k)))
                processed_images[i] = image[int(0.5 * (1 / k - 1) * aim_size[1]):
                                            int(0.5 * (1 / k - 1) * aim_size[1]) + aim_size[1], :, :]

        return processed_images

    elif origin_images.ndim == 3:
        k = origin_images.shape[1] / origin_images.shape[0]
        if k > 1:
            image = cv2.resize(origin_images, dsize=(int(k * aim_size[0]), aim_size[0]))
            print(int((k - 1) * aim_size[0]))
            processed_image = image[:, int(0.5 * (k - 1) * aim_size[0]):
                                       int(0.5 * (k - 1) * aim_size[0]) + aim_size[0], :]
        else:
            image = cv2.resize(origin_images, dsize=(aim_size[1], int(aim_size[1] / k)))
            processed_image = image[int(0.5 * (1 / k - 1) * aim_size[1]):
                                    int(0.5 * (1 / k - 1) * aim_size[1]) + aim_size[1], :]

        return processed_image

    else:
        raise ValueError(f"'origin_images.ndim' must be 3 or 4, but got {origin_images.ndim}!")


def convert_to_numpy(images_path, method:str, padding:int = None, aim_size:tuple=(572, 572)):
    """将.jpg格式图片转存为numpy数组，便于后续处理"""
    if method not in ["pad", "crop"]:
        raise ValueError(f"Invalid method '{method}'. Valid methods are 'pad' and 'crop'.")

    if method == "pad" and padding is None:
        raise ValueError("When method is 'pad', 'padding' must be provided.")

    if method == "crop" and padding is not None:  # 检查padding是否被显式设置（且不等于默认值255）
        warnings.warn("When method is 'crop', 'padding' parameter is ignored and will have no effect.", UserWarning)


    # 过滤出所有的jpg文件（忽略大小写）
    class_a_files = [file for file in os.listdir(os.path.join(images_path, "A")) if file.lower().endswith('.jpg')]
    class_b_files = [file for file in os.listdir(os.path.join(images_path, "B")) if file.lower().endswith('.jpg')]

    images_a = np.zeros(shape=(len(class_a_files), aim_size[0], aim_size[1], 3), dtype=np.uint8)
    images_b = np.zeros(shape=(len(class_b_files), aim_size[0], aim_size[1], 3), dtype=np.uint8)

    for i in range(len(class_a_files)):
        image = cv2.imread(os.path.join(images_path, "A", f"{class_a_files[i]}"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if method == "pad":
            image = boundary_padding(image, aim_size=aim_size, padding=padding)
        else:
            image = center_crop(image, aim_size)
        images_a[i] = image

    for i in range(len(class_b_files)):
        image = cv2.imread(os.path.join(images_path, "B", f"{class_b_files[i]}"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if method == "pad":
            image = boundary_padding(image, aim_size=aim_size, padding=padding)
        else:
            image = center_crop(image, aim_size)
        images_b[i] = image

    return images_a, images_b


def generate_images_and_labels(images_a:np.array, images_b:np.array, batch_size=16):
    """传入以numpy数组存取的A类和B类图像，划分训练集和测试集,数据集大小至少要大于2 * batch_size"""

    # 确定预测集数，确保训练集数为16的倍数，避免训练过程中因凑不满一个batch导致数据浪费
    n_images =  images_a.shape[0] + images_b.shape[0]
    if n_images < 100:
        warnings.warn("The dataset is too small!", UserWarning)
    n = int(0.95 * n_images)
    q = n // batch_size
    r = n % batch_size  # 余数
    if r > (0.5 * batch_size):
        n_train = batch_size * (q + 1)
    else:
        n_train = batch_size * q

    n_val = n_images - n_train
    n_val_a = min(int(0.5 * n_val), 50)
    n_val_b = min(n_val - n_val_a, 50)

    train_images = np.concatenate((images_a[:-n_val_a, :, :, :],
                                   images_b[:-n_val_b, :, :, :]), axis=0)
    train_labels = np.concatenate((np.tile(np.array([1, 0]), reps=(images_a.shape[0] - n_val_a, 1)),
                                   np.tile(np.array([0, 1]), reps=(images_b.shape[0] - n_val_b, 1))))
    val_images = np.concatenate((images_a[-n_val_a:, :, :, :],
                                   images_b[-n_val_b:, :, :, :]), axis=0)
    val_labels = np.concatenate((np.tile(np.array([1, 0]), reps=(n_val_a, 1)),
                                   np.tile(np.array([0, 1]), reps=(n_val_b, 1))))

    return train_images, train_labels, val_images, val_labels


class SyntheticData:
    def __init__(self, images, labels, is_train=False, shuffle=True):
        self.images = images
        self.labels = labels
        self.is_train = is_train

        self.ids = list(range(images.shape[0]))
        if shuffle:
            np.random.shuffle(self.ids)

    def _read_image_label(self, image_id):
        image = self.images[image_id].astype(np.float32)
        label = self.labels[image_id].astype(np.float32)
        return image, label

    def __getitem__(self, index):
        return self._read_image_label(self.ids[index])

    @property
    def column_names(self):
        column_names = ['image', 'label']
        return column_names

    def __len__(self):
        return len(self.ids)

def get_num_parallel_workers(num_parallel_workers):
    cores = multiprocess.cpu_count()
    if isinstance(num_parallel_workers, int):
        if cores < num_parallel_workers:
            print("The num_parallel_workers {} is set too large, now set it {}"
                  .format(num_parallel_workers, cores))
            num_parallel_workers = cores
    else:
        print("The num_parallel_workers {} is invalid, now set it {}".format(num_parallel_workers, min(cores, 0)))
        num_parallel_workers = min(cores, 0)
    return num_parallel_workers


def create_dataset_with_numpy(images, labels, batch_size, is_train):
    trans = [Rescale(1.0 / 127.5, shift=-1), HWC2CHW()]
    mc_dataset = SyntheticData(images, labels, shuffle=True)
    dataset_loader = ds.GeneratorDataset(mc_dataset, mc_dataset.column_names, shuffle=True,
                                         python_multiprocessing=is_train)
    dataset_trans = dataset_loader.map(operations=trans, input_columns='image',
                                       num_parallel_workers=get_num_parallel_workers(8))
    dataset = dataset_trans.batch(batch_size=batch_size, drop_remainder=True)
    return dataset

__all__ = ["create_dataset_with_numpy",
           "generate_images_and_labels",
           "boundary_padding",
           "convert_to_numpy",
           "center_crop"]
