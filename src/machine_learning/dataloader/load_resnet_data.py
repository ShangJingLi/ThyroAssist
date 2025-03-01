import os
import multiprocess
import cv2
import numpy as np
import mindspore.dataset as ds
from mindspore.dataset.vision.transforms import Rescale, HWC2CHW


def boundary_padding(origin_images:np.array, aim_size:tuple, padding:int):
    if origin_images.ndim == 4:
        processed_images = np.ones(shape=(origin_images.shape[0], aim_size[0], aim_size[1], 3),
                                   dtype=np.uint8) * padding
        k = origin_images.shape[1] / origin_images.shape[2]  # k = 原始高 / 原始宽 < 1
        for i in range(origin_images.shape[0]):
            image = cv2.resize(origin_images[i], dsize=(aim_size[1], int(k * aim_size[1])))
            dist = (image.shape[2] - image.shape[1]) // 2
            processed_images[i][dist:dist+image.shape[0], :, :] = image

        return processed_images

    elif origin_images.ndim == 3:
        k = origin_images.shape[0] / origin_images.shape[1]  # k = 原始高 / 原始宽 < 1
        image = cv2.resize(origin_images, dsize=(aim_size[1], int(k * aim_size[1])))
        processed_image = np.ones(shape=(aim_size[0], aim_size[1], 3), dtype=np.uint8) * padding
        dist = (image.shape[1] - image.shape[0]) // 2
        processed_image[dist:dist+image.shape[0], :, :] = image

        return processed_image


def convert_to_numpy(images_path):
    # 过滤出所有的jpg文件（忽略大小写）
    class_a_files = [file for file in os.listdir(os.path.join(images_path, "A")) if file.lower().endswith('.jpg')]
    class_b_files = [file for file in os.listdir(os.path.join(images_path, "B")) if file.lower().endswith('.jpg')]

    images_a = np.zeros(shape=(len(class_a_files), 572, 572, 3), dtype=np.uint8)
    images_b = np.zeros(shape=(len(class_b_files), 572, 572, 3), dtype=np.uint8)

    flag = 0

    for i in range(len(class_a_files)):
        image = cv2.imread(os.path.join(images_path, "A", f"A{i+1}.jpg"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = boundary_padding(image, (572, 572), padding=255)
        images_a[i] = image

    for i in range(len(class_b_files)):
        flag += 1
        image = cv2.imread(os.path.join(images_path, "B", f"B{i+1}.jpg"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = boundary_padding(image, (572, 572), padding=255)
        images_b[i] = image

    return images_a, images_b


def generate_images_and_labels(images_a:np.array, images_b:np.array):
    train_images = np.concatenate((images_a[:-50, :, :, :],
                                   images_b[:-50, :, :, :]), axis=0)
    train_labels = np.concatenate((np.tile(np.array([1, 0]), reps=(images_a.shape[0] - 50, 1)),
                                   np.tile(np.array([0, 1]), reps=(images_b.shape[0] - 50, 1))))
    val_images = np.concatenate((images_a[-50:, :, :, :],
                                   images_b[-50:, :, :, :]), axis=0)
    val_labels = np.concatenate((np.tile(np.array([1, 0]), reps=(50, 1)),
                                   np.tile(np.array([0, 1]), reps=(50, 1))))

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
           "convert_to_numpy"]
