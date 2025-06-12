import os

import numpy as np
import matplotlib.pyplot as plt
from feature_detector import FAST
from optical_flow import LukasKanade
from ransac import RANSAC
from kitti_odometry_dataset import KittiOdometrySequenceDataset
import torch
import time
import cv2
import random

from schlam.optical_flow import LukasKanade

print(torch.cuda.is_available())
show_flow = True
device = "cuda"

feature_params = dict( maxCorners = 200,
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
    K = data["calib"][0]
    ransac = RANSAC(K, device)

    # Find FAST features
    start_time = time.time()
    detected_features = feature_extractor(image_old)
    p0 = cv2.goodFeaturesToTrack(image_old.cpu().numpy(), mask=None, **feature_params)
    detected_features = torch.tensor(p0[:, 0]).to(device).long()
    old_features = detected_features
    torch.cuda.current_stream().synchronize()
    print("Feature detector: ", (time.time() - start_time) * 1000, "ms")

    for _ in range(len(dataset) - 1):
        # if len(features) < 300:
        #     detected_features = feature_extractor(image_old)
        #     old_features = detected_features + old_features
        #     old_features = filter_doubles(old_features)

        # Load image
        start_time = time.time()
        data = next(data_iter)
        image_new = data["image3"][0].float().to(device)
        torch.cuda.current_stream().synchronize()
        print("Image loading: ", (time.time()-start_time) * 1000, "ms")

        # Calculate optical flow
        start_time = time.time()
        pred_new_features = of.pyramidal_of(image_old, image_new, old_features, levels=5)
        valid_flows = torch.isfinite(pred_new_features[:, 0]) & torch.isfinite(pred_new_features[:, 1])
        old_features = old_features[valid_flows]
        new_features = pred_new_features[valid_flows]
        torch.cuda.current_stream().synchronize()
        print("Optical flow: ", (time.time()-start_time) * 1000, "ms")

        start_time = time.time()
        R, t, p1s_3D, inlier_mask = ransac(old_features, new_features)
        torch.cuda.current_stream().synchronize()
        print("Ransac: ", (time.time() - start_time) * 1000, "ms")

        if show_flow:
            of.plot_optical_flow(image_new, image_old, new_features[inlier_mask], old_features[inlier_mask])



        image_old = image_new
        old_features = new_features
