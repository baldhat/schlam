import torch
from torchvision.transforms import Pad

class LukasKanade():

    def __init__(self):
        self.grad_kernel_x = torch.tensor([
            [-1, 1]
        ]).unsqueeze(0).unsqueeze(0).float()
        self.grad_kernel_y = torch.tensor([
            [-1],
            [1]
        ]).unsqueeze(0).unsqueeze(0).float()
        self.pad = Pad(1, padding_mode="edge")
        self.unfold = torch.nn.Unfold(kernel_size=3, stride=1, padding=1)  # You can adjust stride/padding

    def optical_flow(self, old_img, new_img):
        h, w = new_img.shape[0], new_img.shape[1]

        # gradients
        I_x = torch.nn.functional.conv2d(old_img.unsqueeze(0).unsqueeze(0), self.grad_kernel_x, padding="same")
        I_y = torch.nn.functional.conv2d(old_img.unsqueeze(0).unsqueeze(0), self.grad_kernel_y, padding="same")
        I_t = (new_img - old_img).unsqueeze(0).unsqueeze(0)

        I_x_patches = self.unfold(I_x.float())
        I_x_patches = I_x_patches.view(-1, 9)

        I_y_patches = self.unfold(I_y.float())
        I_y_patches = I_y_patches.view(-1, 9)

        I_t_patches = self.unfold(I_t.float())
        I_t_patches = I_t_patches.view(-1, 9)

        S = torch.stack((I_x_patches, I_y_patches), dim=-1)
        t = -I_t_patches

        inverse, info = torch.linalg.inv_ex(S.transpose(1, 2) @ S)

        uv = (inverse@S.transpose(1,2)@(t.unsqueeze(-1)))
        return uv.T.view(1, 2, h, w)

