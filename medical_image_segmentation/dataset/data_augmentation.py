import os
import cv2
import albumentations as albu
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
# ---------------------------------------------------------------
### 图像增强

class Albu_Dataset(Dataset):

    def __init__(self, images_dir, masks_dir, augmentation=None,):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        self.augmentation = augmentation

    def __getitem__(self, i):

        # read data
        # image = cv2.imread(self.images_fps[i])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # mask = cv2.imread(self.masks_fps[i])
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        image = np.asarray(Image.open(self.images_fps[i]).resize((320, 320)))
        mask = np.asarray(Image.open(self.masks_fps[i]).resize((320, 320)))

        # 图像增强应用
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)

def get_training_augmentation():
    train_transform = [

        # 水平翻转图片
        albu.HorizontalFlip(p=0.5),
        # 垂直翻转图片
        # albu.VerticalFlip(always_apply=False, p=1),
        # 随机仿射变换 scale_limit:图片缩放因子 rotate_limit:图片旋转范围 shift_limit:图片宽高的平移因子 p:转换概率 border_mode:指定使用的外插算法
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=45, shift_limit=0.1, p=0.5, border_mode=0),
        # 填充 min_height：最终图片的最小高度
        # albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        # albu.RandomCrop(height=320, width=320, always_apply=True),
        # 高斯噪点
        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=1),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        # albu.OneOf(
        #     [
        #         albu.RandomContrast(p=1),
        #         albu.HueSaturationValue(p=1),
        #     ],
        #     p=0.9,
        # ),
    ]
    return albu.Compose(train_transform) # Compose 组合变换


def get_validation_augmentation():
    
    test_transform = [
        albu.PadIfNeeded(320, 320)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


if __name__ == '__main__':
    # 数据集所在的目录
    DATA_DIR = '../Medical_Datasets/'

    # 训练集
    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'trainannot')

    train_dataset = Albu_Dataset(x_train_dir, y_train_dir, augmentation=get_training_augmentation())

    for i in range(20):
        image, mask = train_dataset[i]

        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        cv2.imwrite('../Medical_Datasets_aug/train/image_{}.png'.format(100+i), image)
        cv2.imwrite('../Medical_Datasets_aug/trainannot/image_{}.png'.format(100+i), mask)
