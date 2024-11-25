import multiprocess
import numpy as np
import mindspore.dataset as ds
from mindspore.dataset.vision.transforms import Rescale, HWC2CHW
"""DataLoader of ResNet"""


class SyntheticData:
    def __init__(self, images, labels, is_train=False, shuffle=True):
        self.images = images
        self.labels = labels
        self.is_train = is_train

        self.ids = [i for i in range(images.shape[0])]
        if shuffle:
            np.random.shuffle(self.ids)

    def _read_image_label(self, image_id):
        image = self.images[image_id].transpose((1, 2, 0)).astype(np.float32)
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
    dataset_loader = ds.GeneratorDataset(mc_dataset, mc_dataset.column_names, shuffle=False,
                                         python_multiprocessing=is_train)
    dataset_trans = dataset_loader.map(operations=trans, input_columns='image',
                                       num_parallel_workers=get_num_parallel_workers(8))
    datas = dataset_trans.batch(batch_size=batch_size, drop_remainder=True)
    return datas
