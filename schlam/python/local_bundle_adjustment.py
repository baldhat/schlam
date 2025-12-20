import copy

import numpy as np
import pyceres as ceres
from helpers import inverse_rodrigues
import torch
import faulthandler
faulthandler.enable()

class ReprojectionErrorCostFunction(ceres.CostFunction):
    def __init__(self, camera_intrinsics, device, img_indx, pts2d):
        ceres.CostFunction.__init__(self)
        self.set_num_residuals(2)
        self.img_indx = img_indx
        self.K = camera_intrinsics.to(device=device).double()
        self.K_inv = torch.linalg.inv(self.K).float()
        self.device = device
        self.pts2d = torch.tensor(pts2d, device=device)

    def normalize(self, pred_pts, gt_pts):
        p1s = torch.einsum("ij,bj->bi",
                           self.K_inv.float(),
                           torch.concat((pred_pts, torch.ones((pred_pts.shape[0], 1), device=self.device)),
                                        dim=-1).float())
        p2s = torch.einsum("ij,bj->bi",
                           self.K_inv.float(),
                           torch.concat((gt_pts, torch.ones((gt_pts.shape[0], 1), device=self.device)),
                                        dim=-1).float())
        return p1s[:, :2], p2s[:, :2]

    def f(self, r_vec, t, pts3d):
        R = inverse_rodrigues(r_vec)
        p_camera = (R @ pts3d) - (R @ t)
        pred_pts2d = (self.K @ p_camera)
        pred_pts2d = pred_pts2d / pred_pts2d[2]
        pred_pts = self.K_inv @ pred_pts2d.float()
        gt_pts = self.K_inv @ torch.concat((self.pts2d, torch.ones(1, device=self.device)), dim=0).float()
        residuals = (pred_pts[:2] - gt_pts[:2])
        return residuals

    def Evaluate(self, parameters, residuals, jacobian):
        '''
        :param pts3d: the 3d point cloud in the coordinate frame of the first camera
        :param pts2d: the 2d feature positions in the current camera image
        :return:
        '''
        r_vec, t, pts3d = (torch.tensor(parameters[0], requires_grad=True),
                                  torch.tensor(parameters[1], requires_grad=True),
                                  torch.tensor(parameters[2], requires_grad=True))
        result = self.f(r_vec, t, pts3d)
        r_jac, t_jac, pts3d_jac = torch.autograd.functional.jacobian(self.f, (r_vec, t, pts3d))
        residual = result.detach().cpu().numpy()
        residuals[0], residuals[1] = residual[0], residual[1]
        if jacobian is not None:
            # jacobian[0][0] = 1.0
            w, x, y= r_jac.cpu().numpy().flatten(), t_jac.cpu().numpy().flatten(), pts3d_jac.cpu().numpy().flatten()
            np.copyto(jacobian[0], w)
            np.copyto(jacobian[1], x)
            np.copyto(jacobian[2], y)
        return True


class LBA():
    def __init__(self, camera_intrinsics, device):
        self.K = camera_intrinsics.to(device)
        pass

    # pts3d: [num_features, 3]
    # pts2d: [num_features, num_frames, 2]
    # R_vec: [num_frames, 3]
    # t_vec: [num_frames, 3]
    def bundle_adjustment(self, pts3d, pts2d, R_vec, t_vec):
        pts3d = (pts3d[:, :3].astype(np.float64)).copy()
        pts2d = [np.array(pts2d_).astype(np.float64).copy() for pts2d_ in pts2d]
        problem = ceres.Problem()
        for i in range(R_vec.shape[0]):
            problem.add_parameter_block(R_vec[i], 3)
            problem.add_parameter_block(t_vec[i], 3)
            if i == 0:
                problem.set_parameter_block_constant(R_vec[i])
                problem.set_parameter_block_constant(t_vec[i])
        for i in range(len(pts2d)):
            for j in range(pts2d[i].shape[0]):
                problem.add_parameter_block(pts2d[i][j], 2)
                problem.set_parameter_block_constant(pts2d[i][j])

        for j in range(pts3d.shape[0]):
            problem.add_parameter_block(pts3d[j], 3)

        for i in range(len(pts2d)):
            for j in range(pts2d[i].shape[0]):
                #if j > 0:
                cost_function = ReprojectionErrorCostFunction(self.K, "cpu", i, pts2d[i][j])
                cost_function.set_parameter_block_sizes([3, 3, 3])
                problem.add_residual_block(
                    cost_function,
                    ceres.HuberLoss(5.991**(1/2)),
                    [R_vec[j],
                    t_vec[j],
                    pts3d[i]]
                )
                #else:
                #    cost_function = ReprojectionErrorCostFunction(self.K, "cpu", i, pts2d[i][j])
                #    cost_function.set_parameter_block_sizes([3])
                #    problem.add_residual_block(
                #        cost_function,
                #        ceres.HuberLoss(1.0),
                #        [pts3d[i]]
                #    )

        options = ceres.SolverOptions()
        options.max_num_iterations = 50
        options.num_threads = 1
        options.linear_solver_type = ceres.LinearSolverType.DENSE_SCHUR  # Good for BA
        options.minimizer_progress_to_stdout = True  # See optimization progress

        options.function_tolerance = 1e-6
        options.gradient_tolerance = 1e-8
        options.parameter_tolerance = 1e-8

        summary = ceres.SolverSummary()
        ceres.solve(options, problem, summary)
        print(summary.FullReport())
        return  pts3d.reshape(-1, 3), np.array([inverse_rodrigues(torch.tensor(rvec)) for rvec in R_vec]), t_vec


    def reprojection_residual(self, pts2d, K):
        def residual(params):
            r_vec = params[:3]
            t = params[3:6]
            pts3d = params[6:9].reshape(1,3)
            R = inverse_rodrigues(r_vec)
            p_camera = (R @ (pts3d.T) - R @ t.unsqueeze(-1)).T
            pred_pts2d = (K.float() @ p_camera)
            pred_pts2d = pred_pts2d[:2] / pred_pts2d[2]
            residual = pred_pts2d[:2].T - pts2d
            return residual

        return residual

