import numpy as np
from torch.utils.data import Dataset
import pykitti
import datetime
import cv2

class KittiOdometrySequenceDataset(Dataset):

    def __init__(self, path_name, sequence_name):
        self.path = path_name
        self.sequence = sequence_name
        self.data = pykitti.odometry(path_name, sequence_name, frames=range(100))
        self.calib = self.data.calib
        self.timestamps = self.data.timestamps
        self.cam2 = list(iter(self.data.cam2))
        self.cam3 = list(iter(self.data.cam3))
        self.poses = self.data.poses

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        timestamp: datetime.timedelta = self.timestamps[idx]
        image2color = np.array(self.cam2[idx])
        image2 = cv2.cvtColor(image2color, cv2.COLOR_BGR2GRAY)
        image3 = cv2.cvtColor(np.array(self.cam3[idx]), cv2.COLOR_BGR2GRAY)
        calib = self.calib.K_cam2
        projection = self.calib.P_rect_20
        pose = self.poses[idx]
        return {
            "timestamp": timestamp.microseconds,
            "image2": image2,
            "image2color": image2color,
            "image3": image3,
            "calib": calib,
            "pose": pose,
            "projection": projection

        }
