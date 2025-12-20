import torch
from torchvision.transforms import Pad
import time
import numpy as np
from feature_detectors import createFeatureDetector
import cv2
import random
from util import build_pyramid
# import matplotlib.pyplot as plt


def createMatcher(name, cv, device):
    if name == "LK":
        return LukasKanade(window_size=21, device=device, cv=cv)
    elif name == "FLANN":
        return FLANN(device=device)
    else:
        raise RuntimeError("Unknown feature detector: " + name)

class FLANN:
    def __init__(self, device):
        self.device = device
        # FLANN parameters
        self.FLANN_INDEX_KDTREE = 1
        self.index_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5)
        self.search_params = dict(checks=50)  # or pass empty dictionary

        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)
        self.sift = cv2.SIFT_create()

    def getSiftDescriptors(self, img, pts):
        img = img.cpu().numpy().astype(np.uint8)
        pts = pts.round().cpu().numpy().astype(np.uint8)
        keypoints = []
        for point in pts:
            kp = cv2.KeyPoint()
            kp.pt = (int(point[0]), int(point[1]))
            kp.size = 3
            kp.angle = 0
            keypoints.append(kp)
        keypoints = tuple(keypoints)
        kps, des = self.sift.compute(img, keypoints)
        return des, kps

    def __call__(self, old_img, new_img, pts, levels=None):
        new_feats = createFeatureDetector("GFTT", False, self.device)(new_img)
        des1, kps1 = self.getSiftDescriptors(old_img, pts)
        des2, kps2 = self.getSiftDescriptors(new_img, new_feats)
        #
        # kps1, des1 = self.sift.detectAndCompute(old_img.cpu().numpy().astype(np.uint8), None)
        # kps2, des2 = self.sift.detectAndCompute(new_img.cpu().numpy().astype(np.uint8), None)
        #

        matches = self.flann.knnMatch(des1, des2, k=2)
        matchesMask = [[0, 0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]
        valid = torch.tensor(matchesMask, device=self.device)
        new_feat_pos = torch.tensor([[kp.pt[0], kp.pt[1]] for kp in kps2], device=self.device)
        return new_feat_pos, valid[:, 0].bool()

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

    def plot_optical_flow(self, image_new, image_old, new_features, old_features):
        pass
        # figure = plt.figure(figsize=(24, 12))
        # ax = figure.gca()
        # w, h = image_new.shape[-1], image_new.shape[-2]
        # xy0, xy1 = np.array([
        #     [old_features[:, 0], old_features[:, 1]],
        #     [new_features[:, 0], (new_features[:, 1] + h)]
        # ])
        # pts = np.stack((xy0.T, xy1.T), axis=1).round().astype(np.int64).reshape(-1, 2, 1, 2)
        # img = np.vstack((image_old, image_new))
        # drawPts = pts
        # cv2.polylines(img, drawPts, False, (255, 255, 255))
        # ax.imshow(img.astype(np.uint8), cmap='gray', vmin=0, vmax=255)
        # plt.show()

    def interpolate(self, tensor, l_x, u_x, l_y, u_y, w_x, w_y, w, h):
        return ((w_x * w_y).unsqueeze(0) * tensor[:, :, (l_y * w + l_x).clamp(0, w*h-1)] \
         + ((1 - w_x) * w_y).unsqueeze(0) * tensor[:, :, (l_y * w + u_x).clamp(0, w*h-1)] \
         + ((1 - w_x) * (1 - w_y)).unsqueeze(0) * tensor[:, :, (u_y * w + u_x).clamp(0, w*h-1)] \
         + (w_x * (1 - w_y)).unsqueeze(0) * tensor[:, :, (u_y * w + l_x).clamp(0, w*h-1)])