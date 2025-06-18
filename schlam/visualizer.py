import open3d as o3d
import numpy as np
import threading
import time

# Shared point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))

def update_thread(vis):
    """Thread to simulate background work and update point cloud."""
    for _ in range(1000):
        # Simulate background work
        time.sleep(0.1)

        # Generate new points (replace this with your real computation)
        new_points = np.random.rand(200, 3)

        # Update the point cloud safely
        pcd.points = o3d.utility.Vector3dVector(new_points)

        # Notify the visualizer about the change
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()



if __name__ == "__main__":
    # Initialize visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    # Run update in a background thread
    thread = threading.Thread(target=update_thread, args=(vis,))
    thread.start()

    # Keep running the visualizer loop
    while thread.is_alive():
        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()