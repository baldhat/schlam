import numpy as np
import pyceres as ceres
from helpers import inverse_rodrigues


class ReprojectionErrorCostFunction:
    def __init__(self, camera_intrinsics, device, pts2d):
        self.K = camera_intrinsics.to(device)
        self.pts2d = pts2d


    def __call__(self, r_vec, t, pts3d, residual):
        '''
        :param pts3d: the 3d point cloud in the coordinate frame of the first camera
        :param pts2d: the 2d feature positions in the current camera image
        :return:
        '''
        R = inverse_rodrigues(r_vec)
        p_camera = (R@(pts3d.T[:3]) - R@t.unsqueeze(-1)).T
        pred_pts2d = (self.K.float() @ p_camera)
        pred_pts2d = pred_pts2d[:2] / pred_pts2d[2]
        residual = pred_pts2d[:2].T - self.pts2d
        return True


class LBA():
    def __init__(self, camera_intrinsics, device):
        self.K = camera_intrinsics.to(device)
        pass

    # pts3d: [num_frames, num_features, 3]
    # pts2d: [num_frames, num_features, 2]
    # R_vec: [num_frames, 3]
    # t_vec: [num_frames, 3]
    def bundle_adjustment(self, pts3d, pts2d, R_vec, t_vec):
        problem = ceres.Problem()
        for i in range(pts3d.shape[0]):
            problem.add_parameter_block(R_vec[i], 3)
            problem.add_parameter_block(t_vec[i], 3)
            # problem.SetLocalParameterization(cameras[i * 2], ceres.QuaternionParameterization())

        for i in range(pts3d.shape[1]):
            problem.add_parameter_block(pts3d[:, i, :], 3)

        problem.add_parameter_block_constant((R_vec[0], t_vec[0]))

        cost_function = ReprojectionErrorCostFunction(self.K, "cpu", pts2d)
        # The cost function takes parameters in the order specified here
        problem.AddResidualBlock(
            cost_function,
            None,
            [R_vec,
            t_vec,
            pts3d]
        )

        options = ceres.SolverOptions()
        options.max_num_iterations = 100
        options.linear_solver_type = ceres.LinearSolverType.SPARSE_SCHUR  # Good for BA
        options.minimizer_progress_to_stdout = True  # See optimization progress

        summary = ceres.SolverSummary()
        ceres.solve(options, problem, summary)

        print(summary.FullReport())

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

    # pts3d: [num_features, 3]
    # pts2d: [num_frames, num_features, 2]
    # R_vec: [num_frames, 3]
    # t_vec: [num_frames, 3]
    def bundle_adjustment2(self, pts3d, pts2d, R_vec, t_vec):
        problem = ceres.Problem()

        # Flatten all parameters for Ceres
        cameras = camera_params.astype(np.float64)
        points = points_3d.astype(np.float64)

        for obs in observations:
            cam_idx, pt_idx, x_2d = obs
            cam_param = cameras[cam_idx]
            pt_3d = points[pt_idx]

            # Each residual block is 9 params: 3 rvec, 3 tvec, 3 X
            params = np.hstack([cam_param, pt_3d])
            problem.add_residual_block(
                cost_functor=reprojection_residual(x_2d, K),
                parameter_blocks=[params],
                loss=ceres.HuberLoss(1.0)
            )

        # Solve the problem
        options = ceres.SolverOptions()
        options.linear_solver_type = ceres.LinearSolverType.DENSE_SCHUR
        options.minimizer_progress_to_stdout = True

        summary = ceres.Summary()
        ceres.Solve(options, problem, summary)
        print(summary.BriefReport())

        # Return optimized camera parameters and points
        return cameras, points