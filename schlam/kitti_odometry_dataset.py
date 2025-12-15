import numpy as np
from torch.utils.data import Dataset
import pykitti
import datetime
import cv2
from pathlib import Path

from datetime import datetime, timezone

class KittiOdometrySequenceDataset(Dataset):

    def __init__(self, path_name, sequence_name, num_frames):
        self.path = path_name
        self.sequence = sequence_name
        self.data = pykitti.odometry(path_name, sequence_name, frames=range(num_frames))
        self.calib = self.data.calib
        self.timestamps = self.data.timestamps
        self.cam2 = list(iter(self.data.cam2))
        self.cam3 = list(iter(self.data.cam3))
        self.drives = {
            "00": ("2011_10_03_drive_0027", 0, 4540),
            "01": ("2011_10_03_drive_0042", 0, 1100),
            "02": ("2011_10_03_drive_0034", 0, 4660),
            "03": ("2011_09_26_drive_0067", 0, 800),
            "04": ("2011_09_30_drive_0016", 0, 270),
            "05": ("2011_09_30_drive_0018", 0, 2760),
            "06": ("2011_09_30_drive_0020", 0, 1100),
            "07": ("2011_09_30_drive_0027", 0, 1100),
            "08": ("2011_09_30_drive_0028", 1100, 5170),
            "09": ("2011_09_30_drive_0033", 0, 1590),
            "10": ("2011_09_30_drive_0034", 0, 1200),
        }
        self.drive = self.drives[sequence_name]
        self.poses = self.data.poses

    def __len__(self):
        return len(self.timestamps)


    def micros_since_epoch(self, ts: str) -> int:
        """
        Convert a timestamp string like '2011-09-30 11:50:40.354663513'
        to microseconds since the Unix epoch (UTC).
        """
        # Split the fractional part
        base, frac = ts.split('.')
        # Keep only the first 6 digits for microseconds
        micros_str = frac[:6]  # '354663'
        # Re‑assemble a string that datetime.strptime can understand
        ts_micro = f"{base}.{micros_str}"

        # Parse to a naive datetime (no tz info yet)
        dt = datetime.strptime(ts_micro, "%Y-%m-%d %H:%M:%S.%f")

        # Attach UTC timezone (adjust if your timestamps are in another zone)
        dt = dt.replace(tzinfo=timezone.utc)

        # Compute total seconds since epoch, then convert to µs
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        delta = dt - epoch
        return int(delta.total_seconds() * 1_000_000)  # µs as integer

    def getIMUFile(self, idx):
        name, start, stop = self.drive
        file = Path(self.path) / "raw" / Path(name + "_sync") / "oxts" / "data" / f"{idx + start:010}.txt"
        with open(file, "r") as f:
            return f.readlines()[0].split(" ")

    def getIMUTimeStamp(self, idx):
        name, start, stop = self.drive
        file = Path(self.path) / "raw" / Path(name + "_sync") / "oxts" / "timestamps.txt"
        with open(file, "r") as f:
            return self.micros_since_epoch(f.readlines()[idx])

    def loadIMUData(self, idx: int):
        '''
        :param idx:
        :return: acceleration and angular velocity in order [right, down, forward] aka [x,y,z]
        '''
        line = self.getIMUFile(idx)
        ts = self.getIMUTimeStamp(idx)
        vf, vl, vu = float(line[8]), float(line[9]), float(line[10])
        ax, ay, az = float(line[11]), float(line[12]), float(line[13])
        wx, wy, wz = float(line[20]), float(line[21]), float(line[22])
        return np.array([-ay, -az, ax]), np.array([-wy, -wz, wx]), ts, np.array([-vl, -vu, vf])

    # x and y as in image, z out of the image plane away from the camera
    # same for IMU:
    def __getitem__(self, idx):
        timestamp: datetime.timedelta = self.timestamps[idx]
        image2color = np.array(self.cam2[idx])
        image2 = cv2.cvtColor(image2color, cv2.COLOR_BGR2GRAY)
        #image3 = cv2.cvtColor(np.array(self.cam3[idx]), cv2.COLOR_BGR2GRAY)
        calib = self.calib.K_cam2
        projection = self.calib.P_rect_20
        pose = self.poses[idx]
        acc, ang, ts, velocity = self.loadIMUData(idx)
        data_dic = {
            "timestamp": timestamp.microseconds,
            "image": image2,
            "image_color": image2color,
            "calib": calib,
            "pose": pose,
            "projection": projection,
            "acceleration": acc,
            "angular_velocity": ang,
            "imu_ts": ts,
            "velocity": velocity
        }
        return data_dic


'''
Nr. Sequence name Start End
00: 2011_10_03_drive_0027 000000 004540
01: 2011_10_03_drive_0042 000000 001100
02: 2011_10_03_drive_0034 000000 004660
03: 2011_09_26_drive_0067 000000 000800
04: 2011_09_30_drive_0016 000000 000270
05: 2011_09_30_drive_0018 000000 002760
06: 2011_09_30_drive_0020 000000 001100
07: 2011_09_30_drive_0027 000000 001100
08: 2011_09_30_drive_0028 001100 005170
09: 2011_09_30_drive_0033 000000 001590
10: 2011_09_30_drive_0034 000000 001200

  - lat:     latitude of the oxts-unit (deg)
  - lon:     longitude of the oxts-unit (deg)
  - alt:     altitude of the oxts-unit (m)
  - roll:    roll angle (rad),  0 = level, positive = left side up (-pi..pi)
  - pitch:   pitch angle (rad), 0 = level, positive = front down (-pi/2..pi/2)
  - yaw:     heading (rad),     0 = east,  positive = counter clockwise (-pi..pi)
  - vn:      velocity towards north (m/s)
  - ve:      velocity towards east (m/s)
  - vf:      forward velocity, i.e. parallel to earth-surface (m/s)
  - vl:      leftward velocity, i.e. parallel to earth-surface (m/s)
  - vu:      upward velocity, i.e. perpendicular to earth-surface (m/s)
  - ax:      acceleration in x, i.e. in direction of vehicle front (m/s^2)
  - ay:      acceleration in y, i.e. in direction of vehicle left (m/s^2)
  - az:      acceleration in z, i.e. in direction of vehicle top (m/s^2)
  - af:      forward acceleration (m/s^2)
  - al:      leftward acceleration (m/s^2)
  - au:      upward acceleration (m/s^2)
  - wx:      angular rate around x (rad/s)
  - wy:      angular rate around y (rad/s)
  - wz:      angular rate around z (rad/s)
  - wf:      angular rate around forward axis (rad/s)
  - wl:      angular rate around leftward axis (rad/s)
  - wu:      angular rate around upward axis (rad/s)
  - posacc:  velocity accuracy (north/east in m)
  - velacc:  velocity accuracy (north/east in m/s)
  - navstat: navigation status
  - numsats: number of satellites tracked by primary GPS receiver
  - posmode: position mode of primary GPS receiver
  - velmode: velocity mode of primary GPS receiver
  - orimode: orientation mode of primary GPS receiver
  
  '''