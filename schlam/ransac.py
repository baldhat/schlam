import random
import torch
import matplotlib.pyplot as plt
import numpy as np

import cv2
import open3d as o3d

class RANSAC:
    def __init__(self, K, device):
        self.K = K.to(device).float()
        self.K_inv = torch.inverse(K).to(device).float()
        self.device = device

    def __call__(self, old_features, feature_preds, normalize=True):
        best_model = None
        max_inliers= 0
        inlier_mask = None
        p1s = torch.einsum("ij,bj->bi",
                           self.K_inv.float(),
                           torch.concat((old_features, torch.ones((old_features.shape[0], 1), device=self.device)),
                                        dim=-1).float())
        p2s = torch.einsum("ij,bj->bi",
                           self.K_inv.float(),
                           torch.concat((feature_preds, torch.ones((feature_preds.shape[0], 1), device=self.device)),
                                        dim=-1).float())
        for k in range(1000):
            feat_indices8 = random.choices(range(old_features.shape[0]), k=8)

            current_p1s, current_p2s = p1s[feat_indices8], p2s[feat_indices8]

            if normalize:
                current_p1s, current_p2s, B1, B2 = self.get_normalized_points(current_p1s, current_p2s)
            A = torch.zeros((8, 9)).to(p1s.device)
            for i in range(8):
                A[i] = torch.kron(p1s[i], p2s[i])
            U, S, V = torch.svd(A)
            E = V[:, -1].view(3, 3)
            if normalize:
                E = B2.T @ E @ B1
            errors = self.epipolarDistance(p1s, p2s, E)
            inlier_mask = (errors < 0.1)
            num_inliers = inlier_mask.sum()
            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_model = E
        print(max_inliers)
        #cv2.recoverPose()
        R, t, p1s_3D = self.recoverPose(best_model, p1s[inlier_mask], p2s[inlier_mask])
        #self.plot_points_3d(p1s_3D.cpu().numpy())
        return R, t, p1s_3D, inlier_mask

    def plot_points_3d(self, p1s_3D):
        viewer = o3d.visualization.Visualizer()
        viewer.create_window()
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p1s_3D[:3, :, 0].T))
        viewer.add_geometry(pc)
        opt = viewer.get_render_option()
        opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        viewer.run()
        viewer.destroy_window()
        print()

    def epipolarDistance(self, p1s, p2s, E):
        p1s, p2s = p1s.unsqueeze(-1), p2s.unsqueeze(-1)
        distance1 = torch.abs(p2s.transpose(-2, -1)@E@p1s)[:, 0, 0] / torch.sqrt((E@p1s)[:, 0]**2 + (E@p1s)[:, 1]**2)[:, 0]
        distance2 = torch.abs(p1s.transpose(-2, -1)@E.T@p2s)[:, 0, 0] / torch.sqrt((E.T@p2s)[:, 0]**2 + (E.T@p2s)[:, 1]**2)[:, 0]
        return distance1 + distance2

    def essential_5point(self, old_features, new_features):
        normalized_feature_pairs = zip(self.K_inv @ old_features, self.K_inv @ new_features)
        A = []
        for pair in normalized_feature_pairs:
            q_1 = pair[0].expand((pair[0].shape[0], 9))
            q_2 = pair[1].expand((pair[1].shape[0], 9))
            A.append(q_1 * q_2)

        A = torch.stack(A)

        svd = torch.lingalg.svd(A)

        G = []
        for i in range(4):
            basis_vector = svd[-1][:, -i]
            G.append(basis_vector.reshape((basis_vector.shape[0], 3, 3)))

        # E=c_0 *G[0] + c_1 *G[1] + c_2 *G[2] + c_3 *G[3] ; E*E.T*E=1/2 * trace(E.T*E) * E ; det(E)=0

        G = torch.stack(G)



        pass

    def skewmat(self, x):
        return torch.tensor([
            [0, -x[2], x[1]],
            [x[2], 0, -x[0]],
            [-x[1], x[0], 0],
        ]).to(x.device)

    def get_normalized_points(self, p1s, p2s):
        mu1 = torch.mean(p1s, dim=0)
        sigma1 = torch.sqrt(torch.sum(torch.linalg.norm(p1s - mu1, dim=1), dim=0))
        mu2 = torch.mean(p2s, dim=0)
        sigma2 = torch.sqrt(torch.sum(torch.linalg.norm(p2s - mu2, dim=1), dim=0))
        sqs1 = torch.sqrt(torch.tensor(2)) / sigma1
        sqs2 = torch.sqrt(torch.tensor(2)) / sigma2
        B1 = torch.tensor([
            [sqs1, 0, -sqs1 * mu1[0]],
            [0, sqs1, -sqs1 * mu1[1]],
            [0, 0, 1]
        ]).to(p1s.device)
        B2 = torch.tensor([
            [sqs2, 0, -sqs2 * mu2[0]],
            [0, sqs2, -sqs2 * mu2[1]],
            [0, 0, 1]
        ]).to(p1s.device)
        p1s = B1 @ p1s.T
        p2s = B2 @ p2s.T
        return p1s.T, p2s.T, B1, B2

    def recoverPose(self, E, p1s, p2s):
        _, _, VE = torch.svd(E)
        t = VE[:, 2]

        tx = self.skewmat(t)
        UR, _, VR = torch.svd(E @ tx)
        R1 = UR @ (VR.T)
        R1 = R1 * torch.linalg.det(R1)
        UR[:, 2] = -UR[:, 2]
        R2 = UR @ (VR.T)
        R2 = R2 * torch.linalg.det(R2)
        ts = torch.stack((t, t, -t, -t), dim=0)
        Rs = torch.stack((R1, R2, R1, R2), dim=0)

        npd = torch.zeros((4, 1)).to(p1s.device)
        X = torch.zeros((4, p1s.shape[0], 4)).to(p1s.device)
        Y = torch.zeros((4, p1s.shape[0], 4)).to(p1s.device)
        for k in range(4):
            G = torch.eye(4).to(p1s.device)
            G[:3, :3] = Rs[k]
            G[:3, 3] = -Rs[k]@ts[k]
            X[:, :, k], Y[:, :, k] = self.triangulate(p1s, p2s, G)
            npd[k] = ((X[2, :, k] > 0) & (Y[2, :, k] > 0)).sum()
        best = torch.max(npd, dim=0)[1]
        return Rs[best], ts[best], X[:, :, best]


    def calc_essential_mat(self, old_features, new_features):
        cv_essential_mat, mask = cv2.findEssentialMat(old_features.cpu().numpy(), new_features.cpu().numpy(), self.K.cpu().numpy())
        R1, R2, t = cv2.decomposeEssentialMat(cv_essential_mat)
        print()

    def triangulate(self, p1s, p2s, G):
        n = p1s.shape[0]
        pi = torch.cat((torch.eye(3), torch.zeros(3, 1)), dim=-1).to(p1s.device)
        phi = pi @ G

        A_0 = p1s[:,0].unsqueeze(-1)* pi[2, :].unsqueeze(0) - pi[0, :]
        A_1 = p1s[:,1].unsqueeze(-1) * pi[2, :].unsqueeze(0) - pi[1, :]
        A_2 = p2s[:,0].unsqueeze(-1) * phi[2, :].unsqueeze(0) - phi[0, :]
        A_3 = p2s[:,1].unsqueeze(-1) * phi[2, :].unsqueeze(0) - phi[1, :]

        A = torch.stack([A_0, A_1, A_2, A_3], dim=1)
        _, _, V = torch.svd(A)

        X = V[:, :, -1].T
        Y = G @ X

        X = self.homogeneous(self.euclidean(X))
        Y = self.homogeneous(self.euclidean(Y))
        return X, Y

    def homogeneous(self, e):
        return torch.cat((e, torch.ones(1, e.shape[1]).to(e.device)), dim=0)

    def euclidean(self, h):
        w = h[-1]
        d = h.shape[0] - 1
        e = h[:d]
        nz = (w != 0)
        e[:, nz] = h[:d, nz] / (torch.ones((d, 1)).to(h.device) * w[nz])
        return e