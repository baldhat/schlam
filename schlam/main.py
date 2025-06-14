import os

import numpy as np
import matplotlib.pyplot as plt
from feature_detector import FAST
from ransac import RANSAC
from kitti_odometry_dataset import KittiOdometrySequenceDataset
import torch
import time
import cv2
from helpers import plot_path, rodrigues, inverse_rodrigues
from optical_flow import LukasKanade
from local_bundle_adjustment import LBA


print(torch.cuda.is_available())
show_flow = True
device = "cuda"

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 5,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

if __name__=="__main__":
    path = os.environ["KITTI_ODOMETRY_PATH"] # /home/baldhat/dev/data/KittiOdometry
    dataset = KittiOdometrySequenceDataset(path, "04")
    feature_extractor = FAST(20, 12, 10, device)
    of = LukasKanade(15, device)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    data_iter = iter(dataloader)
    data = next(data_iter)
    image_old = data["image2"][0].float().to(device)
    image_old_color = data["image2color"][0].float().to(device)
    K = data["calib"][0]
    ransac = RANSAC(K, device)
    lba = LBA(K, device)

    # Find FAST features
    start_time = time.time()
    detected_features = feature_extractor(image_old)
    p0 = cv2.goodFeaturesToTrack(image_old.cpu().numpy(), 500, 0.01, 10)
    detected_features = torch.tensor(p0[:, 0]).to(device).long()
    old_features = detected_features
    torch.cuda.current_stream().synchronize()
    print("Feature detector: ", (time.time() - start_time) * 1000, "ms")
    RTs, ts, Ts = [], [], []
    ps = [np.array([0, 0, 0])]
    global_points = None

    for _ in range(len(dataset) - 1):
        # if len(features) < 100:
        #     detected_features = feature_extractor(image_old)
        #     old_features = detected_features + old_features
        #     old_features = filter_doubles(old_features)

        # Load image
        start_time = time.time()
        data = next(data_iter)
        image_new = data["image2"][0].float().to(device)
        image_new_color = data["image2color"][0].float().to(device)
        torch.cuda.current_stream().synchronize()
        print("Image loading: ", (time.time()-start_time) * 1000, "ms")

        # Calculate optical flow
        start_time = time.time()
        pred_new_features = of.pyramidal_of(image_old, image_new, old_features, levels=5)
        valid_flows = torch.isfinite(pred_new_features[:, 0]) & torch.isfinite(pred_new_features[:, 1])
        old_features_ = old_features[valid_flows]
        new_features_ = pred_new_features[valid_flows]

        # pred_new_features, st, err = cv2.calcOpticalFlowPyrLK(image_old.cpu().numpy().astype(np.uint8), image_new.cpu().numpy().astype(np.uint8), old_features.float().cpu().numpy(), None, **lk_params)
        # valid_flows = (st == 1)[:, 0]
        # if old_features_ is not None:
        #     old_features_ = torch.tensor(old_features[valid_flows]).to(image_old.device)
        #     new_features_ = torch.tensor(pred_new_features[valid_flows]).to(image_old.device)

        torch.cuda.current_stream().synchronize()
        print("Optical flow: ", (time.time()-start_time) * 1000, "ms")

        start_time = time.time()
        R, t, p1s_3D, inlier_mask = ransac(torch.tensor(old_features_).to(image_new.device), torch.tensor(new_features_).to(image_new.device))
        new_features_ = new_features_[inlier_mask]
        torch.cuda.current_stream().synchronize()
        print("Ransac: ", (time.time() - start_time) * 1000, "ms")


        # T expresses points in c1 in c2
        #T = torch.eye(4).to(R.device)
        #T[:3, :3] = R
        #T[:3, 3] = t
        if len(ps) > 1:
            RTs.append((RTs[-1] @ R.T)) # Orientation of coordinate frame Cx expressed in C0
            ps.append(RTs[-2] @ t + ps[-1])
        else:
            RTs.append(R.T)
            ps.append(t)

        #lba.reprojection_error(p1s_3D, old_features_[inlier_mask])
        #lba.reprojection_error((R@(p1s_3D.T[:3]) - R@t.unsqueeze(-1)).T, new_features_)



        # pts3d: [num_features, 3]
        # pts2d: [num_frames, num_features, 2]
        # R_vec: [num_frames, 3]
        # t_vec: [num_frames, 3]
        lba.bundle_adjustment(p1s_3D.cpu().numpy().astype(np.float64),
                              torch.stack([old_features_[inlier_mask], new_features_], dim=0).cpu().numpy().astype(np.float64),
                              torch.stack([rodrigues(torch.eye(3).to(R.device)), rodrigues(R)], dim=0).cpu().numpy().astype(np.float64),
                              torch.stack([torch.zeros(3).to(R.device), t], dim=0).cpu().numpy().astype(np.float64))

        if show_flow:
            #if global_points is None:
                #global_points = p1s_3D.cpu().numpy()
            #else:
                #global_points = np.concat((global_points.T, Ts[-2] @ p1s_3D.T.cpu().numpy()), axis=-1).T

            #ransac.plot_points_3d(global_points, old_features_[inlier_mask.cpu().numpy()].cpu().numpy(), image_old_color.cpu().numpy())
            #of.plot_optical_flow(image_new.cpu().numpy(), image_old.cpu().numpy(), new_features_.cpu().numpy(), old_features_[inlier_mask.cpu().numpy()].cpu().numpy())
            plot_path(ps)

        image_old = image_new
        old_features = new_features_
        image_old_color = image_new_color
