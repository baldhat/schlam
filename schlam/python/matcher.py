import torch
from torchvision.transforms import Pad
import time
import numpy as np
from feature_detectors import createFeatureDetector, KeyPoint
import cv2
import random
from util import build_pyramid


class FLANN:
    def __init__(self):
        # FLANN parameters
        self.index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        self.search_params = dict(checks=50)

        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)

    def __call__(self, old_features: list[KeyPoint], new_features: list[KeyPoint], levels=None):
        old_desc = np.array([np.frombuffer(old_feature.descriptor.tobytes(), dtype=np.uint8) for old_feature in old_features])
        new_desc = np.array([np.frombuffer(new_feature.descriptor.tobytes(), dtype=np.uint8) for new_feature in new_features])
        matches = self.flann.knnMatch(new_desc, old_desc, k=2)
        tracked_old_mask = []
        tracked_new_mask = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                tracked_old_mask.append(m.trainIdx)
                tracked_new_mask.append(m.queryIdx)
        return tracked_old_mask, tracked_new_mask



class LukasKanade:
    def __init__(self, window_size=21, device="cuda", cv=False):
        self.device = device
        self.grad_kernel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ]).unsqueeze(0).unsqueeze(0).float().to(self.device)
        self.grad_kernel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0,   0],
            [1, 2, 1],
        ]).unsqueeze(0).unsqueeze(0).float().to(self.device)
        self.pad = Pad(1, padding_mode="edge")
        self.ws = window_size
        self.unfold = torch.nn.Unfold(kernel_size=self.ws, stride=1, padding=self.ws//2)
        self.cv = cv

        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=5,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def inv(self, A):
        a, b, c, d = A[:,0, 0], A[:,0, 1], A[:,1, 0], A[:,1, 1]
        det = a*d-b*c
        inv = torch.stack((torch.stack((d, -b), dim=-1), torch.stack((-c, a), dim=-1)), dim=-1) / det.unsqueeze(-1).unsqueeze(-1)
        return inv

    def __call__(self, old_img, new_img, pts, levels):
        if self.cv:
            pred_new_features, st, err = cv2.calcOpticalFlowPyrLK(old_img.cpu().numpy().astype(np.uint8),
                                                                  new_img.cpu().numpy().astype(np.uint8),
                                                                  pts.float().cpu().numpy(), None, **self.lk_params)
            valid_flows = (st == 1)[:, 0]
            return torch.tensor(pred_new_features, device=self.device), torch.tensor(valid_flows, device=self.device)
        else:
            pred_new_features = self.pyramidal_of(old_img, new_img, pts, levels=5)
            valid_flows = (torch.isfinite(pred_new_features[:, 0]) & torch.isfinite(pred_new_features[:, 1]) &
                           (pred_new_features[:, 0] >= 0) & (pred_new_features[:, 1] >= 0) &
                           (pred_new_features[:, 0] < old_img.shape[1]) & (pred_new_features[:, 1] < old_img.shape[0]))
            return pred_new_features, valid_flows

    def pyramidal_of(self, old_img, new_img, pts, levels=2):
        pyramid_old = build_pyramid(old_img.unsqueeze(0).unsqueeze(0).float().to(self.device), levels)
        pyramid_new = build_pyramid(new_img.unsqueeze(0).unsqueeze(0).float().to(self.device), levels)

        pts_scaled = pts.float() / (2**(levels-1))
        flow = torch.zeros_like(pts_scaled)

        for lvl in reversed(range(levels)):
            pts_new = self.new_optical_flow(pyramid_old[lvl], pyramid_new[lvl], pts_scaled, flow)
            delta_flow = pts_new - pts_scaled
            flow += delta_flow

            #self.plot_optical_flow(pyramid_new[lvl][0, 0], pyramid_old[lvl][0, 0], pts_new, pts_scaled)
            #self.plot_optical_flow(pyramid_new[lvl][0, 0], pyramid_old[lvl][0, 0], pts_new, pts_l)

            if lvl > 0:
                flow *= 2.0
                pts_scaled *= 2.0

        tracked_pts = pts + flow
        return tracked_pts


    def new_optical_flow(self, old_img, new_img, pts, initial_guess):
        h, w = new_img.shape[-2], new_img.shape[-1]

        # gradients
        I_x = torch.nn.functional.conv2d(old_img.float(), self.grad_kernel_x, padding="same")
        I_y = torch.nn.functional.conv2d(old_img.float(), self.grad_kernel_y, padding="same")

        l_x = torch.floor(pts[:, 0]).long()
        u_x = torch.ceil(pts[:, 0]).long()
        l_y = torch.floor(pts[:, 1]).long()
        u_y = torch.ceil(pts[:, 1]).long()
        w_x = pts[:, 0] - l_x
        w_y = pts[:, 1] - l_y

        index = torch.clamp(torch.round(pts[:, 1]).long() * w + torch.round(pts[:, 0]).long(), 0, h*w-1)
        I_old_patches = self.unfold(old_img.float())
        I_old_patches = self.interpolate(I_old_patches, l_x, u_x, l_y, u_y, w_x, w_y, w, h)[0].transpose(0, 1)

        I_new_patches_ = self.unfold(new_img.float())

        I_x_patches = self.unfold(I_x.float())
        I_x_patches = self.interpolate(I_x_patches, l_x, u_x, l_y, u_y, w_x, w_y, w, h)[0].transpose(0, 1)

        I_y_patches = self.unfold(I_y.float())
        I_y_patches = self.interpolate(I_y_patches, l_x, u_x, l_y, u_y, w_x, w_y, w, h)[0].transpose(0, 1)
        S = torch.stack((I_x_patches, I_y_patches), dim=-1)

        multiplied = S.transpose(1, 2) @ S
        inv,info = torch.linalg.inv_ex(multiplied)

        guess = torch.zeros_like(initial_guess)
        for k in range(15):
            new_position = pts + initial_guess + guess
            new_index = torch.clamp(torch.round(new_position[:, 1]).long() * w + torch.round(new_position[:, 0]).long(), 0, h*w-1)
            I_new_patches = I_new_patches_[:, :, new_index].transpose(1, 2).view(-1, self.ws * self.ws)
            I_k = I_old_patches - I_new_patches

            b_k = torch.stack(((I_k*I_x_patches).sum(dim=-1),(I_k*I_y_patches).sum(dim=-1)), dim=-1)

            uv = torch.einsum("bxy,by->bx", inv,b_k)
            guess += uv.squeeze(-1)
        return pts + guess

    def interpolate(self, tensor, l_x, u_x, l_y, u_y, w_x, w_y, w, h):
        return ((w_x * w_y).unsqueeze(0) * tensor[:, :, (l_y * w + l_x).clamp(0, w*h-1)] \
         + ((1 - w_x) * w_y).unsqueeze(0) * tensor[:, :, (l_y * w + u_x).clamp(0, w*h-1)] \
         + ((1 - w_x) * (1 - w_y)).unsqueeze(0) * tensor[:, :, (u_y * w + u_x).clamp(0, w*h-1)] \
         + (w_x * (1 - w_y)).unsqueeze(0) * tensor[:, :, (u_y * w + l_x).clamp(0, w*h-1)])