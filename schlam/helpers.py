import matplotlib.pyplot as plt
import numpy as np
import torch

def set_axes_equal(ax):
    """Set 3D plot axes to equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    x_middle = np.mean(x_limits)
    y_range = y_limits[1] - y_limits[0]
    y_middle = np.mean(y_limits)
    z_range = z_limits[1] - z_limits[0]
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_path(ps):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(np.array(ps)[:, 0], np.array(ps)[:, 1], np.array(ps)[:, 2], marker="o")
    set_axes_equal(ax)
    plt.show()

def rodrigues(R: torch.Tensor) -> torch.Tensor:
    assert R.shape == (3, 3), "Input must be a 3x3 rotation matrix"

    trace = R.trace()
    theta = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))

    if torch.isclose(theta, torch.tensor(0.0), atol=1e-6):
        return torch.zeros(3, dtype=R.dtype, device=R.device)

    elif torch.isclose(theta, torch.tensor(torch.pi), atol=1e-4):
        # Handle 180-degree rotation
        R_plus = R + torch.eye(3, dtype=R.dtype, device=R.device)
        axis = R_plus[:, R_plus.diagonal().argmax()]
        axis = axis / axis.norm()
        return theta * axis

    else:
        skew_sym = torch.tensor([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ], dtype=R.dtype, device=R.device)
        r_hat = skew_sym / (2 * torch.sin(theta))
        return theta * r_hat

def inverse_rodrigues(r):
    assert r.shape == (3,), "Input must be a 3D vector"

    theta = torch.norm(r)
    if torch.isclose(theta, torch.tensor(0.0, dtype=r.dtype, device=r.device), atol=1e-6):
        return torch.eye(3, dtype=r.dtype, device=r.device)

    r_hat = r / theta

    # Skew-symmetric matrix of r̂
    K = torch.tensor([
        [0, -r_hat[2], r_hat[1]],
        [r_hat[2], 0, -r_hat[0]],
        [-r_hat[1], r_hat[0], 0]
    ], dtype=r.dtype, device=r.device)

    I = torch.eye(3, dtype=r.dtype, device=r.device)
    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * K @ K
    return R