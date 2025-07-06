import torch
import numpy as np
import cv2
from torchvision.transforms import Pad
import sys

def createFeatureDetector(name, cv, device):
    if name == "FAST":
        return FAST(20, 12, 10, device, cv=cv)
    elif name == "ORB":
        return ORB(device, cv)
    elif name == "SIFT":
        return SIFT(device, cv)
    elif name == "GFTT":
        return GFTT(device)
    else:
        raise RuntimeError("Unknown feature detector: " + name)


class ORB:
    def __init__(self, device, cv):
        self.cv = cv
        self.detector = cv2.ORB.create()
        self.device = device
        if not self.cv:
            self.detector = createFeatureDetector("FAST", self.cv, device)
            self.pad = Pad(1, padding_mode="edge")
            self.unfold = torch.nn.Unfold(kernel_size=3, stride=1, padding=0)
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

    def __call__(self, image, n=500):
        if self.cv:
            return self.calcCV(image.cpu().numpy(), n)
        else:
            return self.calcOwn(image, n)

    def calcCV(self, image, n):
        feats = self.detector.detect(image.astype(np.uint8),None)
        detected_features = torch.tensor([[feat.pt[0], feat.pt[1]] for feat in feats]).to(self.device).long()
        return detected_features

    def calcOwn(self, image, n):
        w = image.shape[0]
        h = image.shape[1]
        feats = self.detector(image, n=10000000)
        padded = self.pad(image)
        patches = self.unfold(padded.float().unsqueeze(0).unsqueeze(0))

        I_x = torch.nn.functional.conv2d(image.float(), self.grad_kernel_x, padding="same")
        I_y = torch.nn.functional.conv2d(image.float(), self.grad_kernel_y, padding="same")

        index = torch.clamp(torch.round(feats[:, 1]).long() * w + torch.round(feats[:, 0]).long(), 0, h * w - 1)
        I_x_patches = self.unfold(I_x.float())[:, :, feats]
        I_y_patches = self.unfold(I_y.float())[:, :, feats]

        I_x_sq = I_x * I_x
        I_x_I_y = I_x * I_y
        I_y_sq = I_y * I_y

        M = torch.stack(torch.stack((I_x_sq.sum(dim=(0, 1)), I_x_I_y.sum(0, 1))),
             torch.stack((I_x_sq.sum(dim=(0, 1)), I_y_sq.sum(0, 1))))

        R = torch.det(M) - 0.05 * torch.trace(M)

        raise RuntimeError("Not implemented")

class SIFT:
    def __init__(self, device, cv):
        self.cv = cv
        self.detector = cv2.SIFT.create()
        self.device = device

    def __call__(self, image, n=500):
        if self.cv:
            return self.calcCV(image, n)
        else:
            return self.calcOwn(image, n)

    def calcCV(self, image, n):
        feats = self.detector.detect(image,None)
        detected_features = torch.tensor(feats[:n, 0]).to(self.device).long()
        return detected_features

    def calcOwn(self, image, n):
        raise RuntimeError("Not implemented")

class FAST:
    def __init__(self, threshold, n, min_distance, device, cv=False):
        self.cv = cv
        self.device = device
        self.threshold = threshold
        self.n_contiguous = n
        self.point_indices = [
            [0, 0, 1, 2, 3, 4, 5, 6, 6, 6, 5, 4, 3, 2, 1, 0],
            [3, 4, 5, 6, 6, 6, 5, 4, 3, 2, 1, 0, 0, 0, 1, 2]
        ]

        filters = []
        for i in range(16):
            f = np.zeros(16)
            f[i:i+self.n_contiguous] = 1
            if (i+self.n_contiguous) > 15:
                f[0:(i+self.n_contiguous)%16] = 1
            filters.append(f)

        self.pad = Pad(3, padding_mode="edge")
        self.filter = torch.Tensor(np.array(filters)).to(self.device)
        self.unfold = torch.nn.Unfold(kernel_size=7, stride=1, padding=0)  # You can adjust stride/padding
        self.min_distance = min_distance

        if self.cv:
            self.fast_cv = cv2.FastFeatureDetector.create(threshold=self.threshold, nonmaxSuppression=True)

    def __call__(self, image, n=500):
        if self.cv:
            return self.calcOpenCV(image.cpu().numpy(), n)
        else:
            return self.calcOwn(image, n)

    def calcOpenCV(self, image, n):
        feats = self.fast_cv.detect(image.astype(np.uint8), None)
        detected_features = torch.tensor([[feat.pt[0], feat.pt[1]] for feat in feats]).to(self.device).long()
        return detected_features

    def calcOwn(self, image, n):
        h, w = image.shape[0], image.shape[1]

        padded = self.pad(image)
        patches = self.unfold(padded.float().unsqueeze(0).unsqueeze(0))
        patches = patches.reshape(1, -1, image.shape[0], image.shape[1]).view(1, 7, 7, image.shape[0], image.shape[1])

        current_value = patches[:, 3, 3]
        diff = patches[:, self.point_indices[1], self.point_indices[0]] - current_value

        values = torch.zeros((1, 16, h, w)).to(self.device)
        values[diff >= self.threshold] = 1
        values[diff <= -self.threshold] = -1

        count_pos = torch.einsum("ij,bixy->bjxy", self.filter, torch.maximum(values, torch.tensor(0)))
        count_neg = torch.einsum("ij,bixy->bjxy", (-self.filter), torch.minimum(values, torch.tensor(0)))

        pos_matches = (count_pos == 12).sum(dim=1, keepdim=True).nonzero()
        neg_matches = (count_neg == 12).sum(dim=1, keepdim=True).nonzero()
        xs = torch.cat((pos_matches[:, 3], neg_matches[:, 3]))
        ys = torch.cat((pos_matches[:, 2], neg_matches[:, 2]))

        values = diff[0, :, ys, xs].abs().sum(dim=0)
        points = torch.stack((xs, ys), dim=-1)

        indices = self.filter_pixels(points, values)[:n]

        return points[indices]

    def filter_pixels(self, points, values):
        # Sort pixels by value descending
        sorted_indices = torch.argsort(values, descending=True)
        points = points[sorted_indices]

        selected_mask = torch.zeros(len(points), dtype=torch.bool)
        selected_positions = []

        for i in range(len(points)):
            pos = points[i]

            if len(selected_positions) > 0:
                selected = points[selected_mask]
                dists = torch.norm(selected - pos.float(), dim=1)
                if (dists < self.min_distance).any():
                    continue

            selected_mask[i] = True
            selected_positions.append(pos)

        # Map back to original indices
        selected_original_indices = sorted_indices[selected_mask]
        return selected_original_indices

class GFTT():
    def __init__(self, device, quality=0.01, min_dist=10):
        self.device = device
        self.quality = quality
        self.minDist = min_dist

    def __call__(self, image, n=200):
        p0 = cv2.goodFeaturesToTrack(image.cpu().numpy(), n, self.quality, self.minDist)
        detected_features = torch.tensor(p0[:, 0]).to(self.device).long()
        return detected_features