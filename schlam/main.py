import os
import threading

import rclpy
import numpy as np
import matplotlib.pyplot as plt
from feature_detectors import FAST
from ransac import RANSAC
from schlam.feature_detectors import createFeatureDetector
from visualizer import run_async
from kitti_odometry_dataset import KittiOdometrySequenceDataset
import torch
import time
import cv2
from helpers import plot_path, rodrigues, inverse_rodrigues
from matcher import createMatcher
from local_bundle_adjustment import LBA
from sensor_msgs.msg import Image
# from rclpy import

print(torch.cuda.is_available())
show_flow = True
device = "cuda"

if __name__=="__main__":
    path = os.environ["KITTI_ODOMETRY_PATH"] # /home/baldhat/dev/data/KittiOdometry
    dataset = KittiOdometrySequenceDataset(path, "10")
    feature_extractor = createFeatureDetector("FAST", False, device)
    matcher = createMatcher("LK", cv=False, device=device)
    visualizer = run_async()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    data_iter = iter(dataloader)
    data = next(data_iter)
    image_old = data["image2"][0].float().to(device)
    image_old_color = data["image2color"][0].float().to(device)
    K = data["calib"][0]
    P = data["projection"][0]
    pose = data["pose"][0]
    ransac = RANSAC(K, device)
    lba = LBA(K, device)

    # Find FAST features
    start_time = time.time()
    detected_features = feature_extractor(image_old)
    torch.cuda.current_stream().synchronize()
    print("Feature detector: ", (time.time() - start_time) * 1000, "ms")
    old_features = detected_features
    track_ids = np.arange(detected_features.shape[0])
    tracks = {i: [detected_features[i]] for i in range(detected_features.shape[0])}
    active_points = track_ids
    RTs = [np.eye(3)]
    ts = [np.array([0, 0, 0])]
    global_points = None
    p1s_3D = None
    h, w = min(image_old.shape), max(image_old.shape)

    t = threading.Thread(target=rclpy.spin, args=[visualizer])
    t.start()
    for frame_idx in range(len(dataset) - 1):
        if p1s_3D is not None:
            visualizer.publish_camera(image_old_color, [R.T for R in RTs], ts, K, P, p1s_3D[:, :3], old_features)
            visualizer.publish_gt(pose[:3, :3], pose[:3, 3])

        if len(active_points) < 100:
            detected_features = feature_extractor(image_old)
            old_features = torch.cat((old_features, detected_features), dim=0)
            latest_track_id = max(list(tracks.keys()))
            track_id = latest_track_id + 1
            for new_feature in detected_features:
                tracks[track_id] = [new_feature]
                track_id += 1
            new_points = torch.arange(latest_track_id + 1, track_id)
            active_points = np.concatenate((active_points, new_points.cpu().numpy()))

        # Load image
        start_time = time.time()
        data = next(data_iter)
        image_new = data["image2"][0].float().to(device)
        image_new_color = data["image2color"][0].float().to(device)
        pose = data["pose"][0]
        torch.cuda.current_stream().synchronize()
        print("Image loading: ", (time.time()-start_time) * 1000, "ms")

        # Calculate optical flow
        start_time = time.time()
        pred_new_features, valid_flows = matcher(image_old, image_new, old_features, levels=5)
        old_features_ = old_features[valid_flows]
        new_features_ = pred_new_features[valid_flows]
        active_points = active_points[valid_flows.cpu().numpy()]

        torch.cuda.current_stream().synchronize()
        print("Optical flow: ", (time.time()-start_time) * 1000, "ms")

        start_time = time.time()
        R, t, p1s_3D, inlier_mask = ransac(torch.tensor(old_features_).to(image_new.device), torch.tensor(new_features_).to(image_new.device))

        p1s_3D = torch.einsum("ij,nj->ni", torch.tensor(RTs[-1], device=device).float(), p1s_3D[:, :3].float()) + torch.tensor(ts[-1], device=device)
        new_features_ = new_features_[inlier_mask.cpu()]
        active_points = active_points[inlier_mask.cpu().numpy()]
        torch.cuda.current_stream().synchronize()
        print("Ransac: ", (time.time() - start_time) * 1000, "ms")

        if len(ts) > 1:
            ts.append((RTs[-1] @ t.float() + ts[-1]).float())
            RTs.append((RTs[-1] @ R.T.float()).float()) # Orientation of coordinate frame Cx expressed in C0
        else:
            RTs.append(R.T.float())
            ts.append(t.float())

        #lba.reprojection_error(p1s_3D, old_features_[inlier_mask])
        #lba.reprojection_error((R@(p1s_3D.T[:3]) - R@t.unsqueeze(-1)).T, new_features_)


        for i, j in enumerate(active_points):
            tracks[j].append(new_features_[i])

        # pts3d: [num_features, 3]
        # pts2d: [num_frames, num_features, 2]
        # R_vec: [num_frames, 3]
        # t_vec: [num_frames, 3]
        bundle_adjustment_frames = 5
        if frame_idx >= 500:
            # bundle_adjustment_mask = torch.zeros_like(active_points)
            # for i, j in enumerate(active_points):
            #     bundle_adjustment_mask[i] = 1 if len(tracks[j]) == frame_idx else 0

            features = []
            for i in active_points:
                track = tracks[i]
                features.append([x.cpu().numpy().astype(np.float64) for x in track[-bundle_adjustment_frames:]])
            #features = torch.tensor(np.array(features)).transpose(0, 1)
            optim_pts3d, optim_Rs, optim_ts = lba.bundle_adjustment(p1s_3D.cpu().numpy().astype(np.float64),
                                  features,
                                  torch.stack([rodrigues(torch.tensor(RT_.T, device=R.device).float()) for RT_ in RTs[-bundle_adjustment_frames:]], dim=0).cpu().numpy().astype(np.float64),
                                  torch.stack([torch.tensor(t, device=R.device) for t in ts[-bundle_adjustment_frames:]], dim=0).cpu().numpy().astype(np.float64))

            #ransac.plot_points_3d_comparison(p1s_3D.cpu().numpy(), optim_pts3d)
            p1s_3D = torch.tensor(optim_pts3d, device=device).float()
            ts[-bundle_adjustment_frames:] = [torch.tensor(t, device=device).float() for t in optim_ts]
            RTs[-bundle_adjustment_frames:] = [torch.tensor(R.T, device=device).float() for R in optim_Rs]

        if show_flow:
            #if global_points is None:
                #global_points = p1s_3D.cpu().numpy()
            #else:
                #global_points = np.concat((global_points.T, Ts[-2] @ p1s_3D.T.cpu().numpy()), axis=-1).T

            #ransac.plot_points_3d(p1s_3D.cpu().numpy(), old_features_[inlier_mask.cpu().numpy()].cpu().numpy(), image_old_color.cpu().numpy())
            #ransac.plot_points_3d(optim_pts3d, old_features_[inlier_mask.cpu().numpy()].cpu().numpy(), image_old_color.cpu().numpy())
            # matcher.plot_optical_flow(image_new.cpu().numpy(), image_old.cpu().numpy(), new_features_.cpu().numpy(), old_features_[inlier_mask.cpu().numpy()].cpu().numpy())
            #plot_path(ts)
            pass

        image_old = image_new
        old_features = new_features_
        image_old_color = image_new_color


