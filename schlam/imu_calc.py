import numpy as np
import torch
import quaternion
from scipy.spatial.transform import Rotation

class IMUCalculator:
    def __init__(self, data_iter, tf_listener):
        self.data_iter = data_iter
        self.tf_listener = tf_listener
        data = next(data_iter)

        # = converting from local to world frame?
        self.gravity_global = torch.tensor([0, 0, 9.805])

        self.omega_bias = torch.tensor([0, 0, 0], dtype=torch.float)
        self.acc_bias = torch.tensor([0, 0, 0], dtype=torch.float)
        n = 600
        for i in range(n):
            self.omega_bias += data["omega"][0]
            self.acc_bias += data["acc"][0]
            data = next(data_iter)
        self.omega_bias /= n
        self.acc_bias /= n

        # IMU
        self.last_R = data["gt_pose"][0][:3, :3]
        self.last_ts = data["timestamp"][0]
        self.last_velocity = data["gt_vel"][0]
        self.last_position = data["gt_pose"][0][:3, 3]

        self.last_gt = data["gt_pose"][0]
        self.gts = [self.last_gt]

        self.omega_bias = self.last_R @ self.omega_bias
        self.acc_bias = (self.last_R @ self.acc_bias) - self.gravity_global

    def preintegrateUntil(self, end):
        acc_R = torch.eye(3)
        while True:
            item = next(self.data_iter)
            ts = item["timestamp"][0]
            delta_t = (ts - self.last_ts) / 1000000
            # accelaration in local
            unbiased_omega = item["omega"][0] - self.last_R.T @ self.omega_bias
            unbiased_acc = item["acc"][0] - self.last_R.T @ self.acc_bias
            # Assuming delta_R expresses the new frame in the previous frame
            delta_R = self.calc_rotation(unbiased_omega, delta_t)
            acc_R = acc_R @ delta_R

            R = self.last_R.cpu().float() @ torch.tensor(delta_R).float()

            velocity = self.calc_velocity(self.last_velocity, unbiased_acc, delta_t, self.last_R)
            position = self.calc_position(self.last_position, self.last_velocity, delta_t)
            self.last_ts = ts
            self.last_position = position
            self.last_velocity = velocity
            self.last_R = R
            self.last_gt = item["gt_pose"][0]
            if ts >= end:
                break
        print(Rotation.from_matrix(acc_R).as_euler("xyz", degrees=True))
        print(unbiased_omega)
        self.gts.append(self.last_gt)
        return self.last_R, self.last_position

    def calc_rotation(self, ang, dt):
        '''
        :param ang: Angular velocity in axis angle representation
        :param dt: Time delta in microseconds since the last measurement
        :return: The absolute orientation performed by integrating ang over time dt
        '''
        ang_norm = np.linalg.norm(ang)
        if (ang_norm <= 1e-4):
            Omega = [[0, ang[2], -ang[1], ang[0]],
                     [-ang[2], 0, ang[0], ang[1]],
                     [ang[1], -ang[0], 0, ang[2]],
                     [-ang[0], -ang[1], -ang[2], 0]]
            return np.identity(4) + dt/2 * Omega
        else:
            a = ang / ang_norm * np.sin(ang_norm/2 * dt)
            rotation = np.quaternion(np.cos(ang_norm/2 * dt), a[0], a[1], a[2])
            return quaternion.as_rotation_matrix(rotation)

    def calc_velocity(self, last_velocity, acceleration, dt, R):
        '''

        :param last_velocity: in the global coordinate system
        :param acceleration: in the local imu coordinate system
        :param dt:
        :param R: expressing the local imu coordiante system in the global coordinate system
        :return:
        '''
        return last_velocity + (R.float()@acceleration.float() - self.gravity_global) * dt

    def calc_position(self, last_position, last_velocity, dt):
        '''
        :param last_position: in the global coordinate system
        :param velocity: in the global coordinate system
        :param dt:
        :return:
        '''
        return last_position + last_velocity.float() * dt