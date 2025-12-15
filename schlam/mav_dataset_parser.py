from torch.utils.data import Dataset
import numpy as np
import os
import re
import torch
import quaternion
import pandas as pd
import yaml
from PIL import Image

MAV_PATH = os.environ['MAV_PATH']
dataset_body_to_our_body = torch.tensor([
            [0.0000000, 1.0000000, 0.0000000],
            [-1.0000000, 0.0000000, 0.0000000],
            [0.0000000, 0.0000000, 1.0000000]
        ])

def sort_numerically(files):
    """
    Sorts a list of filenames containing numbers in numerical order.
    """
    # Create a regular expression to find numbers in the filenames
    convert = lambda text: int(text) if text.isdigit() else text

    # Create a key function for sorting that handles both numbers and text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]

    # Sort the files using the alphanumeric key
    return sorted(files, key=alphanum_key)

class ImageData:
    def __init__(self, timestamp, image, intrinsics, extrinsics):
        self.timestamp = timestamp # micros
        self.image = image
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

class IMUMeasurement:
    def __init__(self, ts, ax, ay, az, wx, wy, wz):
        self.ts = ts # micros
        self.acc = torch.tensor([ax, ay, az])
        self.omega = torch.tensor([wx, wy, wz])

class GroundTruth:
    def __init__(self, ts, px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz, ax, ay, az):
        self.ts = ts # micros
        self.acc = torch.tensor([ax, ay, az])
        self.pos = torch.tensor([px, py, pz])
        self.rot = quaternion.as_rotation_matrix(np.quaternion(qw, qx, qy, qz))
        self.pose = torch.eye(4)
        self.pose[:3, :3] = torch.tensor(self.rot)
        self.pose[:3, 3] = torch.tensor(self.pos)
        self.ang = torch.tensor([wx, wy, wz])
        self.vel = torch.tensor([vx, vy, vz])


class MAVImageDataset(Dataset):
    def __init__(self, visualizer, path_name=MAV_PATH+"/vicon_room1/V1_01_easy/V1_01_easy/mav0/"):
        self.path = path_name
        self.cam_path = (path_name + "cam0/").replace("//", "/")
        self.image_timestamps = []
        self.visualizer = visualizer

        self.images = []
        all_images = os.listdir(self.cam_path + "/data/")
        all_images = sort_numerically(all_images)
        sensor = yaml.safe_load(open(self.path + "cam0/sensor.yaml", "r"))
        extrinsics_unaligned = torch.tensor(sensor["T_BS"]["data"]).reshape(4, 4)
        self.visualizer.publish_static_transform(extrinsics_unaligned[:3, :3], extrinsics_unaligned[:3, 3], "dataset/body", "dataset/camera")
        self.visualizer.publish_static_transform(extrinsics_unaligned[:3, :3], extrinsics_unaligned[:3, 3], "gt/body", "gt/camera")
        self.visualizer.publish_static_transform(extrinsics_unaligned[:3, :3], extrinsics_unaligned[:3, 3], "pred/body", "pred/camera")
        fu, fv, cu, cv = sensor["intrinsics"]
        intrinsics = torch.tensor([
            [fu,  0, cu],
            [ 0, fv, cu],
            [ 0,  0,  1]
        ])
        for f in all_images:
            with Image.open(os.path.join(self.cam_path, "data", f)) as img:
                time_nano = int(f.split(".")[0])
                self.images.append(ImageData(time_nano / 1000.0, np.array(img), intrinsics, extrinsics_unaligned))


    def __getitem__(self, idx):
        '''

        :param idx:
        :return: Data in the local coordinate frame (x: right, y: down, z: front)
        '''
        image = self.images[idx]
        timestamp = image.timestamp
        calib = image.intrinsics
        projection = image.extrinsics
        data_dic = {
            "timestamp": timestamp,
            "image":  image.image,
            "image_color": image.image,
            "calib": calib,
            "projection": projection
        }
        return data_dic

    def __len__(self):
        return len(self.images)

class MAVIMUDataset(Dataset):
    def __init__(self, visualizer, path_name=MAV_PATH + "/vicon_room1/V1_01_easy/V1_01_easy/mav0/"):
        self.path = path_name
        self.visualizer = visualizer

        self.visualizer.publish_static_transform(dataset_body_to_our_body, torch.tensor([0, 0, 0]), "dataset/body", "pred/body")

        self.imu = pd.read_csv(path_name + "imu0/data.csv")
        self.imu_measurements = [IMUMeasurement(ts / 1000.0, ax, ay, az, wx, wy, wz) for idx, (ts, wx, wy, wz, ax, ay, az)
                                 in self.imu.iterrows()]


        self.gt_csv = pd.read_csv(path_name + "state_groundtruth_estimate0/data.csv")
        self.gts = [GroundTruth(ts / 1000.0, px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz, ax, ay, az)
                    for idx, (ts, px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz, ax, ay, az) in
                    self.gt_csv.iterrows()]

    def __getitem__(self, idx):
        '''
        :param idx:
        :return: Data in the local coordinate frame (x: right, y: down, z: front)
        '''
        imu = self.imu_measurements[idx]
        timestamp = imu.ts
        gt = self.gts[idx]
        assert np.isclose(gt.ts, imu.ts)
        data_dic ={
            "timestamp": timestamp,
             "acc": imu.acc,
             "omega": imu.omega,
             "gt_pose": gt.pose,
             "gt_vel": gt.vel,
             "gt_acc": gt.acc,
             "gt_ang": gt.ang
        }
        return data_dic

    def __len__(self):
        return len(self.imu)