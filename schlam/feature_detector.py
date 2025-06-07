import torch
import numpy as np
from torchvision.transforms import Pad
import matplotlib.pyplot as plt

class FAST:
    def __init__(self, threshold, n):
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
        self.filter = torch.Tensor(np.array(filters)).cuda()
        self.unfold = torch.nn.Unfold(kernel_size=7, stride=1, padding=0)  # You can adjust stride/padding
        print()

    def __call__(self, image):
        h, w = image.shape[0], image.shape[1]

        padded = self.pad(image)
        patches = self.unfold(padded.float().unsqueeze(0).unsqueeze(0))
        patches = patches.reshape(1, -1, image.shape[0], image.shape[1]).view(1, 7, 7, image.shape[0], image.shape[1])

        current_value = patches[:, 3, 3]
        diff = patches[:, self.point_indices[1], self.point_indices[0]] - current_value

        values = torch.zeros((1, 16, h, w)).cuda()
        values[diff >= self.threshold] = 1
        values[diff <= -self.threshold] = -1

        count_pos = torch.einsum("ij,bixy->bjxy", self.filter, torch.maximum(values, torch.tensor(0)))
        count_neg = torch.einsum("ij,bixy->bjxy", (-self.filter),torch.minimum(values, torch.tensor(0)))

        pos_matches = (count_pos == 12).sum(dim=1, keepdim=True).nonzero()
        neg_matches = (count_neg == 12).sum(dim=1, keepdim=True).nonzero()
        xs = torch.cat((pos_matches[:, 3], neg_matches[:, 3]))
        ys = torch.cat((pos_matches[:, 2], neg_matches[:, 2]))

        #TODO: eliminate duplicates via NMS 

        return xs, ys

