import os

import matplotlib.pyplot as plt
from feature_detector import FAST
from optical_flow import LukasKanade
from kitti_odometry_dataset import KittiOdometrySequenceDataset
import torch
import time

from schlam.optical_flow import LukasKanade

print(torch.cuda.is_available())


if __name__=="__main__":
    path = os.environ["KITTI_ODOMETRY_PATH"] # /home/baldhat/dev/data/KittiOdometry
    dataset = KittiOdometrySequenceDataset(path, "04")
    feature_extractor = FAST(20, 12)
    of = LukasKanade()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    data_iter = iter(dataloader)
    data = next(data_iter)
    image_old = data["image"][0].float()
    old_image_gpu = image_old.to("cuda")
    old_features_xs, old_features_ys = feature_extractor(old_image_gpu)

    for _ in range(len(dataset) - 1):
        data = next(data_iter)
        image_new = data["image"][0].float()

        # Find FAST features
        start_time = time.time()
        gpu_image = image_new.cuda()
        new_features_xs, new_features_ys = feature_extractor(gpu_image)
        print((time.time()-start_time))

        # Calculate optical flow
        flow = of.optical_flow(image_old, image_new)
        flow = flow[0].permute(1,2,0)
        flow_img = torch.cat((flow, torch.ones((flow.shape[0], flow.shape[1], 1))), dim=-1)
        plt.imshow(flow_img)
        #pred_new_features = old_features_ys + optical_flow(old_features)

        # fig = plt.figure(figsize=(24, 12))
        # ax = fig.gca()
        # ax.imshow(image_old, cmap='gray', vmin=0, vmax=255)
        # ax.scatter(new_features_xs.cpu().numpy(), new_features_ys.cpu().numpy(), s=1)
        # plt.show()
        print()

    print()

