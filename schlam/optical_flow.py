import torch
from torchvision.transforms import Pad
import time
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

class LukasKanade():

    def __init__(self, window_size=21, device="cuda"):
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

    def inv(self, A):
        a, b, c, d = A[:,0, 0], A[:,0, 1], A[:,1, 0], A[:,1, 1]
        det = a*d-b*c
        inv = torch.stack((torch.stack((d, -b), dim=-1), torch.stack((-c, a), dim=-1)), dim=-1) / det.unsqueeze(-1).unsqueeze(-1)
        return inv

    def build_pyramid(self, image, levels):
        pyramid = [image]
        for _ in range(1, levels):
            pyramid.append(torch.nn.functional.avg_pool2d(pyramid[-1], 2))
        return pyramid

    def pyramidal_of(self, old_img, new_img, pts, levels=2):
        pyramid_old = self.build_pyramid(old_img.unsqueeze(0).unsqueeze(0).float().to(self.device), levels)
        pyramid_new = self.build_pyramid(new_img.unsqueeze(0).unsqueeze(0).float().to(self.device), levels)

        pts_scaled = pts.float() / (2**(levels-1))
        flow = torch.zeros_like(pts_scaled)

        for lvl in reversed(range(levels)):
            pts_new = self.new_optical_flow(pyramid_old[lvl], pyramid_new[lvl], pts_scaled, flow)
            delta_flow = pts_new - pts_scaled
            flow += delta_flow

            self.plot_optical_flow(pyramid_new[lvl][0, 0], pyramid_old[lvl][0, 0], pts_new, pts_scaled)
            #self.plot_optical_flow(pyramid_new[lvl][0, 0], pyramid_old[lvl][0, 0], pts_new, pts_l)

            if lvl > 0:
                flow *= 2.0
                pts_scaled *= 2.0

            print()

        tracked_pts = pts + flow
        return tracked_pts


    def new_optical_flow(self, old_img, new_img, pts, initial_guess):
        h, w = new_img.shape[-2], new_img.shape[-1]

        # gradients
        I_x = torch.nn.functional.conv2d(old_img.float(), self.grad_kernel_x, padding="same")
        I_y = torch.nn.functional.conv2d(old_img.float(), self.grad_kernel_y, padding="same")

        index = torch.round(pts[:, 1]).long() * w + torch.round(pts[:, 0]).long()
        I_old_patches = self.unfold(old_img.float())
        I_old_patches = I_old_patches[:, :, index].transpose(1,2).view(-1, self.ws*self.ws)

        I_new_patches_ = self.unfold(new_img.float())

        I_x_patches = self.unfold(I_x.float())
        I_x_patches = I_x_patches[:, :, index].transpose(1, 2).view(-1, self.ws * self.ws)

        I_y_patches = self.unfold(I_y.float())
        I_y_patches = I_y_patches[:, :, index].transpose(1, 2).view(-1, self.ws * self.ws)
        S = torch.stack((I_x_patches, I_y_patches), dim=-1)

        multiplied = S.transpose(1, 2) @ S

        guess = torch.zeros_like(initial_guess)
        for k in range(10):
            new_position = pts + initial_guess + guess
            new_index = torch.round(new_position[:, 1]).long() * w + torch.round(new_position[:, 0]).long()
            I_new_patches = I_new_patches_[:, :, new_index].transpose(1, 2).view(-1, self.ws * self.ws)
            I_k = I_old_patches - I_new_patches

            b_k = torch.stack(((I_k*I_x_patches).sum(dim=-1),(I_k*I_y_patches).sum(dim=-1)), dim=-1)

            uv = torch.linalg.solve(multiplied, b_k)
            guess += uv.squeeze(-1)
        return pts + guess


    def optical_flow(self, old_img, new_img, pts):
        h, w = new_img.shape[-2], new_img.shape[-1]

        # gradients
        I_x = torch.nn.functional.conv2d(old_img, self.grad_kernel_x, padding="same")
        I_y = torch.nn.functional.conv2d(old_img, self.grad_kernel_y, padding="same")
        I_t = (new_img - old_img)

        for k in range(1):
            index = torch.round(pts[:, 1] * w + pts[:, 0]).long()

            I_x_patches = self.unfold(I_x.float())
            I_x_patches = I_x_patches[:, :, index].transpose(1,2).view(-1, self.ws*self.ws)

            I_y_patches = self.unfold(I_y.float())
            I_y_patches = I_y_patches[:, :, index].transpose(1,2).view(-1, self.ws*self.ws)

            I_t_patches = self.unfold(I_t.float())
            I_t_patches = I_t_patches[:, :, index].transpose(1,2).view(-1, self.ws*self.ws)

            S = torch.stack((I_x_patches, I_y_patches), dim=-1)
            t = -I_t_patches

            multiplied = S.transpose(1, 2) @ S
            uv = torch.linalg.solve(multiplied, S.transpose(1,2)@t.unsqueeze(-1))
            pts += uv.squeeze(-1)
            #inverse, _ = torch.linalg.inv_ex(multiplied)
            #uv = (inverse@S.transpose(1,2)@(t.unsqueeze(-1)))
        return pts

    def plot_optical_flow(self, image_new, image_old, new_features, old_features):
        figure = plt.figure(figsize=(24, 12))
        ax = figure.gca()
        w, h = image_new.shape[-1], image_new.shape[-2]
        xy0, xy1 = torch.tensor(np.array([
            [old_features[:, 0].cpu().numpy(), old_features[:, 1].cpu().numpy()],
            [new_features[:, 0].cpu().numpy(), (new_features[:, 1] + h).cpu().numpy()]
        ]))
        pts = torch.stack((xy0.T, xy1.T), dim=1).round().long().numpy().reshape(-1, 2, 1, 2)
        img = np.vstack((image_old.cpu(), image_new.cpu()))
        random_indices = random.sample(range(len(pts)), 10)
        drawPts = [pt for pt in pts[random_indices]]
        cv2.polylines(img, drawPts, False, (255, 255, 255))
        ax.imshow(img.astype(np.uint8), cmap='gray', vmin=0, vmax=255)
        plt.show()