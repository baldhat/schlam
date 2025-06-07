import numpy as np
from torch.utils.data import Dataset
import pykitti
import datetime
import cv2

class KittiOdometrySequenceDataset(Dataset):

    def __init__(self, path_name, sequence_name):
        self.path = path_name
        self.sequence = sequence_name
        self.data = pykitti.odometry(path_name, sequence_name)
        self.calib = self.data.calib.b_rgb
        self.timestamps = self.data.timestamps
        self.cam2 = list(iter(self.data.cam2))
        self.poses = self.data.poses

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        timestamp: datetime.timedelta = self.timestamps[idx]
        image = cv2.cvtColor(np.array(self.cam2[idx]), cv2.COLOR_BGR2GRAY)
        calib = self.calib
        pose = self.poses[idx]
        return {
            "timestamp": timestamp.microseconds,
            "image": image,
            "calib": calib,
            "pose": pose
        }
