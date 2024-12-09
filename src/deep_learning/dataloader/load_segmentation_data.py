import os
import cv2
import numpy as np
import mindspore.dataset as ds


def get_file_name_from_dir(path):
    """批量读取图像文件名"""
    filenames = []
    for filename in os.listdir(path):
        if os.path.isfile(os.path.join(path, filename)):
            filenames.append(filename)
    return filenames


class MultiClassDatasetAtNumpy:
    """
    Read image and mask from original images, and split all data into train_dataset and val_dataset by `split`.
    Get image path and mask path from a tree of directories,
    images within one folder is an image, the image file named `"image.png"`, the mask file named `"mask.png"`.
    """
    def __init__(self, images, masks, is_train=False, shuffle=True):
        self.images = images
        self.masks = masks
        self.is_train = is_train

        self.ids = list(range(images.shape[0]))
        if shuffle:
            np.random.shuffle(self.ids)  # 打乱排序列表

    def _read_img_mask(self, img_id):
        # 获取单张图片和标签并返回
        img = self.images[img_id]
        mask = self.masks[img_id]

        return img, mask

    def __getitem__(self, index):
        return self._read_img_mask(self.ids[index])  # 返回对应索引的一组图像和标签

    @property
    def column_names(self):
        # 设a为该类对象，则a.column_names为['image', 'mask']
        column_names = ['image', 'mask']
        return column_names

    def __len__(self):
        # 返回训练集或测试集大小
        return len(self.ids)


class MultiClassDataset:
    # 使用转存为npy文件的数据集
    def __init__(self, data_dir, repeat, is_train=False, split=1, shuffle=False):
        self.train_data_dir = os.path.join(data_dir, 'augtrain')
        self.val_data_dir = os.path.join(data_dir, 'augval')
        self.is_train = is_train
        self.split = (split != 1.0)
        if self.split:  # 划分训练集和验证集
            # 对该目录下的子目录进行排序，子目录下存有一个图像和一个标签
            self.img_ids = get_file_name_from_dir(os.path.join(self.train_data_dir, 'image'))
            self.train_ids = self.img_ids[:int(len(self.img_ids) * split)] * repeat  # 存放训练数据的目录的列表
            self.val_ids = self.img_ids[int(len(self.img_ids) * split):]  # 存放验证数据的目录的列表
        else:
            self.train_ids = get_file_name_from_dir(os.path.join(self.train_data_dir, 'image'))
            self.val_ids = get_file_name_from_dir(os.path.join(self.val_data_dir, 'image'))
        if shuffle:
            np.random.shuffle(self.train_ids)  # 打乱训练数据目录，注意此处训练数据和标签放在一个目录下

    def _read_img_mask(self, img_id):
        # 获取单张图片和标签的路径，并用cv2读取成numpy格式并返回一组图像和标签
        if self.split:
            root_dir = self.train_data_dir
        elif self.is_train:
            root_dir = self.train_data_dir
        else:
            root_dir = self.val_data_dir
        img = cv2.imread(root_dir + '/image/' + img_id)
        mask = cv2.imread((root_dir + '/mask/' + img_id), cv2.IMREAD_GRAYSCALE)
        return img, mask

    def __getitem__(self, index):
        if self.is_train:
            return self._read_img_mask(self.train_ids[index])  # 返回对应索引的一组图像和标签
        return self._read_img_mask(self.val_ids[index])

    @property
    def column_names(self):
        # 设a为该类对象，则a.column_names为['image', 'mask']
        column_names = ['image', 'mask']
        return column_names

    def __len__(self):
        # 返回训练集或测试集大小
        if self.is_train:
            return len(self.train_ids)
        return len(self.val_ids)


def preprocess_img_mask(img, mask, num_classes, img_size=(572, 572),  mask_size=(388, 388),
                        augment=False):
    """
    Preprocess for multi-class dataset.数据处理函数
    Random crop and flip images and masks when augment is True.
    """
    if augment:  # 数据增强
        img_size_w = int(np.random.randint(img_size[0], img_size[0] * 1.5, 1))
        img_size_h = int(np.random.randint(img_size[1], img_size[1] * 1.5, 1))
        img = cv2.resize(img, (img_size_w, img_size_h))
        mask = cv2.resize(mask, (img_size_w, img_size_h))
        dw = int(np.random.randint(0, img_size_w - img_size[0] + 1, 1))
        dh = int(np.random.randint(0, img_size_h - img_size[1] + 1, 1))
        img = img[dh:dh+img_size[1], dw:dw+img_size[0], :]
        mask = mask[dh:dh+img_size[1], dw:dw+img_size[0]]
        if np.random.random() > 0.5:
            flip_code = int(np.random.randint(-1, 2, 1))
            img = cv2.flip(img, flip_code)
            mask = cv2.flip(mask, flip_code)
    else:
        img = cv2.resize(img, img_size)  # 调整图像大小
        mask = cv2.resize(mask, mask_size)
    img = (img.astype(np.float32) - 127.5) / 127.5  # 图像像素值转到-1，1
    if len(img.shape) == 3:
        img = img.transpose(2, 0, 1)  # 三通道，执行transpose
    else:
        img = np.tile(img, reps=(3, 1, 1))  # 单通道，执行tile
    if num_classes == 2:
        mask = mask.astype(np.float32) / mask.max()  # 归一化到0-1之间
        mask = (mask > 0.5).astype(np.int32)  # 二值化标签
    else:
        mask = mask.astype(np.int32)
    mask = (np.arange(num_classes) == mask[..., None])  # 用广播机制创建每个像素的独热编码
    mask = mask.astype(np.int32)
    mask = mask.transpose(2, 0, 1).astype(np.float32)  # 将通道放在第一个维度
    return img, mask  # 返回单张数据集和标签


def create_segmentation_dataset(data_dir, img_size, mask_size, repeat, batch_size, num_classes=2, is_train=False, augment=False,
                               eval_resize=False, split=1, rank=0, group_size=1, shuffle=True):
    """
    Get generator dataset for multi-class dataset.
    """

    mc_dataset = MultiClassDataset(data_dir, repeat, is_train, split, shuffle)  # 获取数据集
    dataset = ds.GeneratorDataset(mc_dataset, mc_dataset.column_names, shuffle=True,
                                  num_shards=group_size, shard_id=rank, python_multiprocessing=is_train)
    compose_map_func = (lambda image, mask: preprocess_img_mask(image, mask, num_classes, tuple(img_size),
                                                                tuple(mask_size), augment and is_train))
    dataset = dataset.map(operations=compose_map_func, input_columns=mc_dataset.column_names,
                          output_columns=mc_dataset.column_names)
    dataset = dataset.batch(batch_size, drop_remainder=is_train)
    return dataset


def create_segmentation_dataset_at_numpy(images, masks, img_size, mask_size, batch_size,
                                         num_classes=2, is_train=False, augment=False,
                                         rank=0, group_size=1, shuffle=True):
    """
    Get generator dataset for multi-class dataset with numpy.array.
    """

    mc_dataset = MultiClassDatasetAtNumpy(images, masks, is_train, shuffle)  # 获取数据集
    dataset = ds.GeneratorDataset(mc_dataset, mc_dataset.column_names, shuffle=True,
                                  num_shards=group_size, shard_id=rank, python_multiprocessing=is_train)
    compose_map_func = (lambda image, mask: preprocess_img_mask(image, mask, num_classes, tuple(img_size),
                                                                tuple(mask_size), augment and is_train))
    dataset = dataset.map(operations=compose_map_func, input_columns=mc_dataset.column_names,
                          output_columns=mc_dataset.column_names)
    dataset = dataset.batch(batch_size, drop_remainder=is_train)
    return dataset


__all__ = ['create_segmentation_dataset',
           'create_segmentation_dataset_at_numpy']
