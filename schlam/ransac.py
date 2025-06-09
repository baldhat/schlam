import random
import torch
import cv2

class RANSAC:
    def __init__(self, K, device):
        self.K = K.to(device).float()
        self.K_inv = torch.inverse(K).to(device).float()
        self.device = device

    def __call__(self, old_features, feature_preds):
        ones = torch.ones(old_features.shape[0], 1).to(self.device)
        old_features_ = torch.cat((old_features, ones), 1)
        feature_preds_ = torch.cat((feature_preds, ones), 1)
        feat_indices5 = random.choices(range(old_features_.shape[0]), k=5)
        feat_indices8 = random.choices(range(old_features_.shape[0]), k=8)
        current_old_feats5, current_new_feats5 = old_features_[feat_indices5], feature_preds_[feat_indices5]
        current_old_feats8, current_new_feats8 = old_features[feat_indices8], feature_preds[feat_indices8]
        #self.calc_essential_mat(old_features, feature_preds )

        #self.essential_5point(current_old_feats5, current_new_feats5)
        self.essential_8point(current_old_feats8, current_new_feats8)




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

        pass


    def essential_8point(self, old_features, new_features):
        p1s = torch.einsum("ij,bj->bi",
                           self.K_inv.float(),
                           torch.concat((old_features, torch.ones((old_features.shape[0], 1), device=self.device)), dim=-1).float())
        p2s = torch.einsum("ij,bj->bi",
                           self.K_inv.float(),
                           torch.concat((new_features, torch.ones((new_features.shape[0], 1), device=self.device)),
                                        dim=-1).float())
        pass


    def calc_essential_mat(self, old_features, new_features):
        cv_essential_mat, mask = cv2.findEssentialMat(old_features.cpu().numpy(), new_features.cpu().numpy(), self.K.cpu().numpy())
        R1, R2, t = cv2.decomposeEssentialMat(cv_essential_mat)
        print()