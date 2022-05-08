import os
import albumentations as albu
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset as BaseDataset
import matplotlib.pyplot as plt


# ---------------------------------------------------------------
### 加载数据

class Dataset(BaseDataset):

    CLASSES = ['_background', 'uterus', 'bladder', 'rectum']

    def __init__(self, images_dir, masks_dir, classes=None, preprocessing=None,):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = np.asarray(Image.open(self.images_fps[i]).resize((320, 320)), dtype=np.int32)
        mask = np.asarray(Image.open(self.masks_fps[i]).resize((320, 320)), dtype=np.int32)

        # 从标签中提取特定的类别 (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # 图像预处理应用
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    # 进行图像预处理操作

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def visualize(**images):
    
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

if __name__ == '__main__':
    DATA_DIR = '../Medical_Datasets/'

    # 测试集
    x_test_dir = os.path.join(DATA_DIR, 'test')
    y_test_dir = os.path.join(DATA_DIR, 'testannot')

    CLASSES = ['_background', 'uterus', 'bladder', 'rectum']
    test_dataset = Dataset(x_test_dir, y_test_dir, classes=CLASSES,)

    for i in range(6):
        image = test_dataset[i][0]
        gt_mask = test_dataset[i][1]

        gt_mask = np.asarray(gt_mask).transpose(2, 0, 1)
        gt_mask = torch.tensor(gt_mask)
        gt_mask = torch.max(gt_mask, 0)[1]
        gt_mask = np.asarray(gt_mask)

        visualize(
            image = image,
            ground_truth_mask=gt_mask,
        )
