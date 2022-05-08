from dataset.data import *
import numpy as np
import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader


# 图像分割结果的可视化展示
def visualize(**images):
    """Plot images in one row."""
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

    DATA_DIR = '../Medical_Datasets_aug/'

    # 测试集
    x_test_dir = os.path.join(DATA_DIR, 'test')
    y_test_dir = os.path.join(DATA_DIR, 'testannot')

    ENCODER = 'se_resnet50'
    ENCODER_WEIGHTS = 'imagenet'

    CLASSES = ['_background', 'uterus', 'bladder', 'rectum']
    ACTIVATION = 'softmax2d'
    DEVICE = 'cuda'

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # 加载模型
    best_model = torch.load('../checkpoint/unet++_da/se_resnet50/best_model.pth')

    # create test dataset
    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        classes=CLASSES,
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

    loss = smp.utils.losses.DiceLoss()

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Precision(),
        smp.utils.metrics.Recall(),
    ]

    # evaluate model on test set
    test_epoch = smp.utils.train.ValidEpoch(
        best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )

    logs = test_epoch.run(test_dataloader)

    # test dataset without transformations for image visualization
    test_dataset_vis = Dataset(
        x_test_dir, y_test_dir,
        classes=CLASSES,
    )

    for i in range(5):

        image_vis = test_dataset_vis[i][0]
        image, gt_mask = test_dataset[i]

        gt_mask = torch.tensor(gt_mask)
        gt_mask = torch.max(gt_mask, 0)[1]
        gt_mask = np.asarray(gt_mask)

        image = image.astype(np.float32)
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor).squeeze().cpu()
        pr_mask = torch.max(pr_mask, 0)[1]
        pr_mask = pr_mask.numpy()

        visualize(
            image=image_vis,
            ground_truth_mask=gt_mask,
            predicted_mask=pr_mask,
        )

