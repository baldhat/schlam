import numpy as np
import pyceres as ceres
from helpers import inverse_rodrigues
import torch


class ReprojectionErrorCostFunction(ceres.CostFunction):
    def __init__(self, camera_intrinsics, device):
        ceres.CostFunction.__init__(self)
        self.set_num_residuals(2)
        self.K = camera_intrinsics.to(device=device).double()
        self.K_inv = torch.linalg.inv(self.K)
        self.device = device

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

    def f(self, r_vec, t, pts3d, pts2d):
        R = inverse_rodrigues(r_vec)
        p_camera = (R @ (pts3d.T[:3]) - R @ t.unsqueeze(-1))
        pred_pts2d = (self.K @ p_camera)
        pred_pts2d = pred_pts2d[:2] / pred_pts2d[2]
        pred_pts, gt_pts = self.normalize(pred_pts2d.T, pts2d)
        residuals = (pred_pts - gt_pts).sum(dim=0)
        return residuals

    def Evaluate(self, parameters, residuals, jacobian):
        '''
        :param pts3d: the 3d point cloud in the coordinate frame of the first camera
        :param pts2d: the 2d feature positions in the current camera image
        :return:
        '''
        r_vec, t, pts3d, pts2d = (torch.tensor(parameters[0], requires_grad=True),
                                  torch.tensor(parameters[1], requires_grad=True),
                                  torch.tensor(parameters[2].reshape(-1, 3), requires_grad=True),
                                  torch.tensor(parameters[3].reshape(-1, 2)))

        result = self.f(r_vec, t, pts3d, pts2d)
        r_jac, t_jac, pts3d_jac, pts2d_jac = torch.autograd.functional.jacobian(self.f, (r_vec, t, pts3d, pts2d))
        residual = result.detach().cpu().numpy()
        residuals[0], residuals[1] = residual[0], residual[1]
        if jacobian is not None:
            # jacobian[0][0] = 1.0
            w, x, y, z= r_jac.cpu().numpy().flatten(), t_jac.cpu().numpy().flatten(), pts3d_jac.cpu().numpy().flatten(), pts2d_jac.cpu().numpy().flatten()
            np.copyto(jacobian[0], w)
            np.copyto(jacobian[1], x)
            np.copyto(jacobian[2], y)
            np.copyto(jacobian[3], z)
        return True


class LBA():
    def __init__(self, camera_intrinsics, device):
        self.K = camera_intrinsics.to(device)
        pass

    # pts3d: [num_features, 3]
    # pts2d: [num_frames, num_features, 2]
    # R_vec: [num_frames, 3]
    # t_vec: [num_frames, 3]
    def bundle_adjustment(self, pts3d, pts2d, R_vec, t_vec):
        pts3d = pts3d[:, :3].astype(np.float64).flatten()
        pts2d = [pts2d_.flatten().astype(np.float64) for pts2d_ in pts2d]
        problem = ceres.Problem()
        for i in range(len(pts2d)):
            problem.add_parameter_block(R_vec[i], 3)
            problem.add_parameter_block(t_vec[i], 3)
            problem.add_parameter_block(pts2d[i], pts2d[i].shape[0])
            problem.set_parameter_block_constant(pts2d[i])
            # problem.SetLocalParameterization(cameras[i * 2], ceres.QuaternionParameterization())

        problem.add_parameter_block(pts3d, pts3d.shape[0])

        problem.set_parameter_block_constant(R_vec[0])
        problem.set_parameter_block_constant(t_vec[0])


        cost_function = ReprojectionErrorCostFunction(self.K, "cpu")

        cost_function.set_parameter_block_sizes([3, 3, pts3d.shape[0], pts2d[0].shape[0]])
        # ceres.AutoDiffCostFunction(cost_function)
        # The cost function takes parameters in the order specified here
        for i in range(len(pts2d)):
            problem.add_residual_block(
                cost_function,
                None,
                [R_vec[i],
                t_vec[i],
                pts3d,
                 pts2d[i]]
            )

        options = ceres.SolverOptions()
        options.max_num_iterations = 100
        options.linear_solver_type = ceres.LinearSolverType.SPARSE_SCHUR  # Good for BA
        options.minimizer_progress_to_stdout = True  # See optimization progress

        summary = ceres.SolverSummary()
        ceres.solve(options, problem, summary)
        print(summary.BriefReport())
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

