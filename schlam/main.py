import os

import numpy as np
import matplotlib.pyplot as plt
from feature_detector import FAST
from optical_flow import LukasKanade
from kitti_odometry_dataset import KittiOdometrySequenceDataset
import torch
import time
import cv2
import random

from schlam.optical_flow import LukasKanade

print(torch.cuda.is_available())
show_flow = True
device = "cuda"

feature_params = dict( maxCorners = 10,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 5,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

if __name__=="__main__":
    path = os.environ["KITTI_ODOMETRY_PATH"] # /home/baldhat/dev/data/KittiOdometry
    dataset = KittiOdometrySequenceDataset(path, "04")
    feature_extractor = FAST(20, 12, device)
    of = LukasKanade(15, device)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    data_iter = iter(dataloader)
    data = next(data_iter)
    image_old = data["image2"][0].float().to(device)
    old_features_xs, old_features_ys = feature_extractor(image_old)

    for _ in range(len(dataset) - 1):
        # Load image
        start_time = time.time()
        #data = next(data_iter)
        image_new = data["image3"][0].float().to(device)
        torch.cuda.current_stream().synchronize()
        print("Image loading: ", (time.time()-start_time) * 1000, "ms")

        # Find FAST features
        start_time = time.time()
        #features_x, features_y = feature_extractor(image_old)
        p0 = cv2.goodFeaturesToTrack(image_old.cpu().numpy(), mask=None, **feature_params)
        features = torch.tensor(p0[:, 0]).to(device).long()
        #p0 = torch.stack((features_x, features_y),dim=-1).unsqueeze(1).cpu().numpy()
        torch.cuda.current_stream().synchronize()
        print("Feature detector: ", (time.time()-start_time) * 1000, "ms")

        # Calculate optical flow
        start_time = time.time()
        feature_flow = of.pyramidal_of(image_old, image_new, features, levels=5)
        # p1, st, err = cv2.calcOpticalFlowPyrLK(
        #     image_old.cpu().numpy().astype(np.uint8),
        #     image_new.cpu().numpy().astype(np.uint8),
        #     p0.astype(np.float32), None, **lk_params)

        #good_new = torch.tensor(p1[st==1])
        #good_old = torch.tensor(p0[st==1])
        valid_flows = torch.isfinite(feature_flow[:, 0]) & torch.isfinite(feature_flow[:, 1])
        old_features = features[valid_flows]
        new_features = feature_flow[valid_flows]
        torch.cuda.current_stream().synchronize()
        print("Optical flow: ", (time.time()-start_time) * 1000, "ms")

        if show_flow:
            of.plot_optical_flow(image_new, image_old, new_features, old_features)
            print()

        image_old = image_new
