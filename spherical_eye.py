import math
import numpy as np
import pybullet as p

class SphericalCompoundEye:
    """
    3D Spherical Compound Eye based on Drosophila Ommatidia morphology:
    Samples 64 visual ommatidial rays across a 3D spherical dome:
      - Azimuth: 360 deg panoramic [-pi, pi]
      - Elevation: -60 deg (ground/underbelly) to +80 deg (overhead predator detection)
      - Left eye: 32 rays (negative Y / left hemisphere)
      - Right eye: 32 rays (positive Y / right hemisphere)
    Total input dimension: 128 (32 Left Dist + 32 Left Looming + 32 Right Dist + 32 Right Looming),
    maintaining 100% biological compatibility with the Janelia Retinal Projector.
    """
    def __init__(self, n_rays=64, max_distance=10.0):
        assert n_rays == 64, "Currently calibrated for 64 rays (32 Left + 32 Right)"
        self.n_rays = n_rays
        self.n_half = n_rays // 2 # 32
        self.max_distance = max_distance
        
        # Precompute body-frame unit direction vectors for Left and Right eyes
        self.right_rays_body = []
        
        min_el = -np.pi / 3.0 # -60 deg
        max_el = 4.0 * np.pi / 9.0 # +80 deg
        
        for i in range(self.n_half):
            el = min_el + (i / max(1, self.n_half - 1)) * (max_el - min_el)
            az = (i * 2.3999632) % np.pi # [0, pi]
            
            dx = math.cos(el) * math.cos(az)
            dy = math.cos(el) * math.sin(az) # Right hemisphere: dy >= 0
            dz = math.sin(el)
            self.right_rays_body.append([dx, dy, dz])
            
        self.right_rays_body = np.array(self.right_rays_body, dtype=np.float32)
        
        # Left hemisphere: bilateral reflection across sagittal plane (dy -> -dy)
        self.left_rays_body = np.copy(self.right_rays_body)
        self.left_rays_body[:, 1] = -self.left_rays_body[:, 1]
        
        # Combined rays: [Right Eye (0..31), Left Eye (32..63)]
        self.all_rays_body = np.vstack([self.right_rays_body, self.left_rays_body])
        self.prev_proximity = np.zeros(self.n_rays, dtype=np.float32)

    def reset(self):
        self.prev_proximity.fill(0.0)

    def cast_rays(self, client_id, drone_pos, drone_rot_matrix, dt=0.033):
        """
        Cast 64 3D rays from current drone position and 3D orientation.
        client_id: physicsClientId returned by PyBullet
        drone_rot_matrix: 3x3 rotation matrix
        """
        world_rays = self.all_rays_body @ drone_rot_matrix.T
        from_pos = drone_pos + 0.05 * world_rays
        to_pos = drone_pos + self.max_distance * world_rays
        
        results = p.rayTestBatch(from_pos.tolist(), to_pos.tolist(), physicsClientId=client_id)
        
        distances = np.zeros(self.n_rays, dtype=np.float32)
        for i, res in enumerate(results):
            distances[i] = 0.05 + res[2] * (self.max_distance - 0.05)
            
        norm_prox = np.clip(1.0 - (distances / self.max_distance), 0.0, 1.0)
        looming = np.clip((norm_prox - self.prev_proximity) / max(1e-4, dt), 0.0, 10.0)
        self.prev_proximity = np.copy(norm_prox)
        
        left_dist = norm_prox[32:]
        left_looming = looming[32:]
        right_dist = norm_prox[:32]
        right_looming = looming[:32]
        
        obs_128 = np.concatenate([left_dist, left_looming, right_dist, right_looming]).astype(np.float32)
        return obs_128, norm_prox, looming
