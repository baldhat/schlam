from timeit import timeit

import matplotlib.patches
import torch
import numpy as np
import cv2
from line_profiler import profile
from torchvision.transforms import Pad, GaussianBlur
from util import build_pyramid
import matplotlib.pyplot as plt
from orb_descriptor_pattern import pattern as orb_bit_pattern
from bitarray import bitarray


class KeyPoint:
    def __init__(self, x, y, score, level, angle=0, descriptor=None):
        self.x = x
        self.y = y
        self.score = score
        self.level = level
        self.angle = angle
        self.descriptor = descriptor

    def scaleBy(self, xFactor, yFactor):
        self.x *= xFactor
        self.y *= yFactor

class OctreeNode:
    def __init__(self, corner_ul, corner_ur, corner_ll, corner_lr, features: list[KeyPoint]):
        self.ul = corner_ul
        self.ur = corner_ur
        self.ll = corner_ll
        self.lr = corner_lr
        self.features = features
        self.isLeaf = self.size() <= 1
        self.isEmpty = self.size() == 0
        self.halfX = np.array([np.floor((self.ur[0] - self.ul[0]) / 2), 0])
        self.halfY = np.array([0, np.floor((self.ll[1] - self.ul[1]) / 2)])

    def size(self) -> int:
        return len(self.features)

    def getFeaturesUL(self) -> list[KeyPoint]:
        features = []
        for feature in self.features:
            if feature.x < (self.ul + self.halfX)[0] and feature.y < (self.ul + self.halfY)[1]:
                features.append(feature)
        return features

    def getFeaturesUR(self) -> list[KeyPoint]:
        features = []
        for feature in self.features:
            if feature.x >= (self.ul + self.halfX)[0] and feature.y < (self.ul + self.halfY)[1]:
                features.append(feature)
        return features
    
    def getFeaturesLL(self) -> list[KeyPoint]:
        features = []
        for feature in self.features:
            if feature.x < (self.ul + self.halfX)[0] and feature.y >= (self.ul + self.halfY)[1]:
                features.append(feature)
        return features
    
    def getFeaturesLR(self) -> list[KeyPoint]:
        features = []
        for feature in self.features:
            if feature.x >= (self.ul + self.halfX)[0] and feature.y >= (self.ul + self.halfY)[1]:
                features.append(feature)
        return features

    def divide(self):
        # < halfX/halfY is part of the left or upper node
        # >= halfX/halfY is part of the right or lower node
        nodeUL = OctreeNode(
            self.ul,
            self.ul  + self.halfX,
            self.ul + self.halfY,
            self.ul + self.halfX + self.halfY,
            self.getFeaturesUL()
        )
        nodeUR = OctreeNode(
            self.ul + self.halfX,
            self.ur,
            self.ul + self.halfX + self.halfY,
            self.ur + self.halfY,
            self.getFeaturesUR()
        )

        nodeLL = OctreeNode(
            self.ul + self.halfY,
            self.ul + self.halfX + self.halfY,
            self.ll,
            self.ll + self.halfX,
            self.getFeaturesLL()
        )

        nodeLR = OctreeNode(
            self.ul + self.halfX + self.halfY,
            self.ur + self.halfY,
            self.ll + self.halfX,
            self.lr,
            self.getFeaturesLR()
        )
        
        return nodeUL, nodeUR, nodeLL, nodeLR


def plot_features(image, features):
    orientations = None
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    if isinstance(features[0], KeyPoint):
        orientations = np.array([[kp.x + 5*np.cos(kp.angle), kp.y + 5*np.sin(kp.angle)] for kp in features])
        features = np.array([np.array([kp.x, kp.y]) for kp in features])
    plt.scatter(features[:, 0], features[:, 1], c="r", s=1)
    plt.imshow(image.astype(np.uint8), vmin=0, vmax=255)
    if orientations is not None:
        for feature, orientation in zip(features, orientations):
            plt.plot([feature[0], orientation[0]], [feature[1], orientation[1]], c="r", linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_octree(image, features, nodes):
    cmap = matplotlib.cm.get_cmap("Dark2")
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(features[0], KeyPoint):
        colors = [cmap(kp.level) for kp in features]
        features = np.array([np.array([kp.x, kp.y]) for kp in features])
    elif isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
        colors = torch.ones(features.shape[0])
    fig, ax = plt.subplots()
    ax.scatter(features[:, 0], features[:, 1], c=colors, s=3)
    ax.imshow(image.astype(np.uint8), vmin=0, vmax=255, cmap="gray")

    for node in nodes:
        rect = matplotlib.patches.Rectangle((node.ul[0], node.ul[1]), node.ur[0] - node.ul[0], node.ll[1] - node.ul[1], edgecolor="r", fill=False, linewidth=0.3)
        ax.add_patch(rect)

    fig.tight_layout()
    plt.show()

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
    '''
    Interpretation of the ORB feature detector in ORB-SLAM3
    '''
    def __init__(self, device, cv, orientationPatchSize=31):
        self.cv = cv
        self.detector = cv2.ORB.create()
        self.device = device
        self.orientationPatchSize = orientationPatchSize
        self.fastDetector = FAST(7, 9, 0, device, cv=False)
        self.unfold9 = torch.nn.Unfold(kernel_size=9, stride=1, padding=0)
        self.orientationPatchUnfold = torch.nn.Unfold(kernel_size=self.orientationPatchSize, stride=1, padding=0)
        self.orientationPad = Pad(self.orientationPatchSize//2, padding_mode="edge")
        self.pad = Pad(4, padding_mode="edge")
        self.gaussian_blur = GaussianBlur(7, sigma=(2, 2))
        self.gauss_pad = Pad(3, padding_mode="constant")

        # Sobel kernels
        self.grad_kernel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ]).unsqueeze(0).unsqueeze(0).float().to(self.device)
        self.grad_kernel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ]).unsqueeze(0).unsqueeze(0).float().to(self.device)
        self.circular_mask = self.create_circular_mask(self.orientationPatchSize, self.orientationPatchSize)

    def __call__(self, image, n=500):
        if self.cv:
            return self.calcCV(image.cpu().numpy(), n)
        else:
            return self.calcOwn(image, n)

    def calcCV(self, image, n):
        feats = self.detector.detect(image.astype(np.uint8),None)
        detected_features = torch.tensor([[feat.pt[0], feat.pt[1]] for feat in feats]).to(self.device).long()
        return detected_features

    @profile
    def calcOwn(self, fullimage, n, nlevels=8, factor = 1/1.2):
        pyramid = build_pyramid(fullimage, nlevels, factor)
        features = self.getFeatures(pyramid, n, nlevels, factor)

    def getFeatures(self, pyramid, n, nlevels=8, factor=1/1.2, octree_per_level=True):
        ndesiredFeaturesPerScale = n * (1 - factor) / (1 - factor ** nlevels)
        nFeaturesPerLevel = [round(ndesiredFeaturesPerScale * factor ** i) for i in range(nlevels - 1)]
        sumFeatures = sum(nFeaturesPerLevel)
        nFeaturesPerLevel.append(max(n - sumFeatures, 0))

        originalImageSizeWidth = pyramid[0].shape[-1]
        originalImageSizeHeight = pyramid[0].shape[-2]

        features = []
        for level, (image, nFeatures) in enumerate(zip(pyramid, nFeaturesPerLevel)):
            levelImageSizeWidth, levelImageSizeHeight = image.shape[-1], image.shape[-2]
            level_features = self.fastDetector(image, nFeatures * 1000)
            level_features = self.removeAtImageBorder(level_features, levelImageSizeWidth, levelImageSizeHeight, border_size=3).cpu().numpy()
            # level_features = self.retainBestHarris(image, level_features, nFeatures, blocksize=7)
            scores = self.computeHarrisResponse(image, level_features, blocksize=7).cpu().numpy()
            keypoints = [KeyPoint(feature[0], feature[1], score, level) for feature, score in zip(level_features, scores)]
            if octree_per_level:
                keypoints, level_nodes = self.buildFeatureOctree(keypoints, levelImageSizeWidth, levelImageSizeHeight, nFeatures)
                #plot_octree(image, keypoints, level_nodes)
            if level != 0:
                keypoints = self.scaleBy(keypoints, originalImageSizeWidth / levelImageSizeWidth, originalImageSizeHeight / levelImageSizeHeight)
            features.extend(keypoints)
        if not octree_per_level:
            features, nodes = self.buildFeatureOctree(features, originalImageSizeWidth, originalImageSizeHeight, n)
            #plot_octree(pyramid[0], features, nodes)
        oriented_features = self.calcOrientation(pyramid, features, factor)
        #plot_features(pyramid[0], oriented_features)
        described_features = self.calcDescriptors(pyramid, oriented_features, factor)

        return described_features

    def calcDescriptors(self, pyramid, features, factor):
        blurredPyramid = [self.gaussian_blur(self.gauss_pad(im.unsqueeze(0).unsqueeze(0))) for im in pyramid]

        patches = []
        for image in blurredPyramid:
            padded_image = self.orientationPad(image)
            patches.append(self.orientationPatchUnfold(padded_image.float())
                            .reshape(1, self.orientationPatchSize*self.orientationPatchSize, image.shape[-2], image.shape[-1])
                            .view(1, self.orientationPatchSize, self.orientationPatchSize, image.shape[-2], image.shape[-1]))

        for feature in features:
            level = feature.level
            local_x, local_y = feature.x * factor ** level, feature.y * factor ** level
            width, height = patches[level].shape[-1], patches[level].shape[-2]
            cx, cy = int(min(round(local_x), width - 1)), int(min(round(local_y), height - 1))
            co, si = np.cos(feature.angle), np.sin(feature.angle)
            vector = bitarray(256)
            for i, test in enumerate(orb_bit_pattern):
                x0, y0, x1, y1 = test
                x0r = round(x0*co - y0*si)
                y0r = round(x0*si + y0*co)
                x1r = round(x1 * co - y1 * si)
                y1r = round(x1 * si + y1 * co)
                p0 = patches[level][0, y0r, x0r, cy, cx].item()
                p1 = patches[level][0, y1r, x1r, cy, cx].item()
                vector[i] = p0 < p1
            feature.descriptor = vector
        return features


    def calcOrientation(self, pyramid, features, factor):
        patches = []
        for image in pyramid:
            padded_image = self.orientationPad(image)
            patches.append(self.orientationPatchUnfold(padded_image.float().unsqueeze(0).unsqueeze(0))
                            .reshape(1, self.orientationPatchSize*self.orientationPatchSize, image.shape[0], image.shape[1])
                            .view(1, self.orientationPatchSize, self.orientationPatchSize, image.shape[0], image.shape[1]))
        halfPatchSize = self.orientationPatchSize // 2
        grid = torch.arange(-halfPatchSize, halfPatchSize + 1).repeat(self.orientationPatchSize).reshape(
            self.orientationPatchSize, self.orientationPatchSize).to(patches[features[0].level].device)
        xs, ys = grid[self.circular_mask], grid.T[self.circular_mask]
        for feature in features:
            level = feature.level
            local_x, local_y = feature.x * factor ** level, feature.y * factor ** level
            width, height = patches[level].shape[-1], patches[level].shape[-2]
            cx, cy = int(min(round(local_x), width-1)), int(min(round(local_y), height-1))

            #m_00 = patches[level][:, halfPatchSize+ys, halfPatchSize+xs, cy, cx].sum(1)
            m_10 = (xs*patches[level][:, halfPatchSize+ys, halfPatchSize+xs, cy, cx]).sum(1)
            m_01 = (ys*patches[level][:, halfPatchSize+ys, halfPatchSize+xs, cy, cx]).sum(1)
            feature.angle = torch.atan2(m_01, m_10).item()

        return features

    def create_circular_mask(self, h, w):
        center = (int(w / 2), int(h / 2))
        radius = min(center[0], center[1], w - center[0], h - center[1])
        Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        dist_from_center = torch.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) - 0.5
        mask = dist_from_center <= radius
        return mask.bool()

    def scaleBy(self, features, xFactor, yFactor):
        for keypoint in features:
            keypoint.scaleBy(xFactor, yFactor)
        return features

    def buildFeatureOctree(self, keypoints, width, height, nFeatures):
        node = OctreeNode(np.array([0, 0]), np.array([width, 0]), np.array([0, height]), np.array([width, height]),
                          keypoints)
        if nFeatures > len(keypoints):
            return node.features, [node]
        features, nodes = self.octreeFilter(node, nFeatures)
        #features, nodes = self.octreeFilter(node, 16)
        return features, nodes

    def octreeFilter(self, node: OctreeNode, n_features):
        nodes = [node]
        finished = False
        while len(nodes) < n_features:
            new_nodes = []
            for node in nodes:
                if node.size() == 1:
                    new_nodes.append(node)
                    continue
                children = node.divide()
                for child in children:
                    if child.size() >= 1:
                        new_nodes.append(child)
            nodes = new_nodes

        features = []
        for node in nodes:
            if node.size() == 1:
                features.append(node.features[0])
            else:
                feature = max(node.features, key=lambda x: x.score)
                features.append(feature)

        return features, nodes


    def retainBestHarris(self, image, features, num_retain, blocksize):
        responses = self.computeHarrisResponse(image, features, blocksize)
        sorted, indices = torch.sort(responses, descending=True)
        return features[indices[:num_retain]]

    @profile
    def computeHarrisResponse(self, image, positions, blocksize, harris_k=0.04):
        h, w = image.shape[0], image.shape[1]
        scale = (1.0 / (4 * blocksize * 255.0)) ** 4
        image = self.pad(image)
        unfolded_patches = self.unfold9(image.float().unsqueeze(0).unsqueeze(0))
        unfolded_patches = (unfolded_patches
                            .reshape(1, -1, h, w)[..., positions[:, 1], positions[:, 0]] # Select keypoint positions
                            .view(1, 9, 9, -1) # unfold patches
                            .transpose(-1, -2) # move
                            .transpose(-2, -3) # features
                            .transpose(0, 1)) # to batch dim
        grad_x = torch.nn.functional.conv2d(unfolded_patches, self.grad_kernel_x)
        grad_y = torch.nn.functional.conv2d(unfolded_patches, self.grad_kernel_y)
        a = (grad_x * grad_x).sum(dim=-1).sum(dim=-1)
        b = (grad_y * grad_y).sum(dim=-1).sum(dim=-1)
        c = (grad_x * grad_y).sum(dim=-1).sum(dim=-1)
        response = ((a*b - c*c) - harris_k*((a+b)**2))*scale
        return response.squeeze(-1)


    def removeAtImageBorder(self, features, width, height, border_size=3):
        return features[
            torch.bitwise_and(
                torch.bitwise_and(features[:, 0] >= border_size, features[:, 0] < (width - border_size)),
                torch.bitwise_and(features[:, 1] >= border_size, features[:, 1] < (height - border_size))
            )
        ]


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

    @profile
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

        pos_matches = (count_pos == self.n_contiguous).sum(dim=1, keepdim=True).nonzero()
        neg_matches = (count_neg == self.n_contiguous).sum(dim=1, keepdim=True).nonzero()
        xs = torch.cat((pos_matches[:, 3], neg_matches[:, 3]))
        ys = torch.cat((pos_matches[:, 2], neg_matches[:, 2]))

        values = diff[0, :, ys, xs].abs().sum(dim=0)
        points = torch.stack((xs, ys), dim=-1)

        indices = self.filter_pixels(points, values)[:n]

        return points[indices]

    @profile
    def filter_pixels(self, points, values):
        if self.min_distance < 1:
            return torch.arange(points.shape[0])
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