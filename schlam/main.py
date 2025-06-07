import os

import matplotlib.pyplot as plt
from feature_detector import FAST
from kitti_odometry_dataset import KittiOdometrySequenceDataset
import torch
import time
print(torch.cuda.is_available())


if __name__=="__main__":
    path = os.environ["KITTI_ODOMETRY_PATH"] # /home/baldhat/dev/data/KittiOdometry
    dataset = KittiOdometrySequenceDataset(path, "04")
    feature_extractor = FAST(20, 12)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for index, data in enumerate(dataloader):
        image = data["image"][0]
        start_time = time.time()
        gpu_image = image.cuda()
        xs, ys = feature_extractor(gpu_image)
        print((time.time()-start_time))



        fig = plt.figure(figsize=(24, 12))
        ax = fig.gca()
        ax.imshow(image, cmap='gray', vmin=0, vmax=255)
        ax.scatter(xs.cpu().numpy(), ys.cpu().numpy(), s=1)
        plt.show()
        print()

    print()

