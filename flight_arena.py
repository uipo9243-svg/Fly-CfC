import numpy as np
import math

class FlightArena:
    """
    Continuous-Time 2D Flight Arena with Dynamic Obstacles.
    Simulates panoramic compound eye vision (64 ommatidia rays) and flight dynamics.
    
    Eye Mapping:
      - Rays 0..31:  Right Eye (Azimuth -180 deg to 0 deg relative to heading)
      - Rays 32..63: Left Eye  (Azimuth 0 deg to +180 deg relative to heading)
    """
    def __init__(self, size=100.0, num_obstacles=20, max_ray_dist=15.0, seed=42):
        self.size = size
        self.num_obstacles = num_obstacles
        self.max_ray_dist = max_ray_dist
        self.num_rays = 64
        self.rng = np.random.RandomState(seed)
        # Keep the agent inside the arena, but close enough to the wall that the
        # geometric collision check below still fires (margin < collision_clearance).
        self.boundary_margin = 0.5
        self.collision_clearance = 0.75

        
        # Ray angles relative to heading: -pi to +pi
        self.ray_angles = np.linspace(-np.pi, np.pi, self.num_rays, endpoint=False)
        
        # Agent state
        self.x = size / 2.0
        self.y = size / 2.0
        self.heading = 0.0 # radians
        self.speed = 5.0   # m/s
        self.distance_traveled = 0.0
        self.prev_distances = np.zeros(self.num_rays, dtype=np.float32)
        
        # Obstacles: list of (x, y, radius)
        self.obstacles = []
        self._generate_obstacles()

    def _generate_obstacles(self):
        self.obstacles = []
        attempts = 0
        while len(self.obstacles) < self.num_obstacles and attempts < 1000:
            attempts += 1
            ox = float(self.rng.uniform(10.0, self.size - 10.0))
            oy = float(self.rng.uniform(10.0, self.size - 10.0))
            # Avoid placing directly on arena center
            if math.hypot(ox - self.size/2.0, oy - self.size/2.0) < 10.0:
                continue
            r = float(self.rng.uniform(2.0, 5.0))
            self.obstacles.append((ox, oy, r))

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            self._generate_obstacles()
            
        # Spawn location with guaranteed obstacle clearance > 6.0 m
        x, y = self.size / 2.0, self.size / 2.0
        for _ in range(300):
            x = float(self.rng.uniform(20.0, self.size - 20.0))
            y = float(self.rng.uniform(20.0, self.size - 20.0))
            clearance = min([math.hypot(x - ox, y - oy) - r for ox, oy, r in self.obstacles], default=999.0)
            if clearance > 6.0:
                break
        heading = float(self.rng.uniform(-np.pi, np.pi))
        return self.reset_at(x, y, heading)

    def reset_at(self, x, y, heading):
        """Place the agent at an explicit pose and return the first observation."""
        self.x = float(np.clip(x, self.boundary_margin, self.size - self.boundary_margin))
        self.y = float(np.clip(y, self.boundary_margin, self.size - self.boundary_margin))
        self.heading = (float(heading) + np.pi) % (2.0 * np.pi) - np.pi
        self.distance_traveled = 0.0

        raw_dist = self._cast_rays()
        norm_prox = 1.0 - (raw_dist / self.max_ray_dist)
        self.prev_distances = np.copy(norm_prox)

        # At t=0 there is no previous frame, so looming is exactly zero.
        looming = np.zeros(self.num_rays, dtype=np.float32)
        obs_128 = self._format_observation(norm_prox, looming)
        return obs_128, norm_prox


    def _cast_rays(self):
        distances = np.full(self.num_rays, self.max_ray_dist, dtype=np.float32)
        
        for i, angle_offset in enumerate(self.ray_angles):
            ray_theta = self.heading + angle_offset
            cos_t = math.cos(ray_theta)
            sin_t = math.sin(ray_theta)
            
            # 1. Intersection with circular obstacles
            for ox, oy, r in self.obstacles:
                dx = ox - self.x
                dy = oy - self.y
                proj = dx * cos_t + dy * sin_t
                if proj > 0:
                    perp_sq = (dx*dx + dy*dy) - proj*proj
                    if perp_sq < r*r:
                        hit_dist = proj - math.sqrt(max(0.0, r*r - perp_sq))
                        if hit_dist > 0 and hit_dist < distances[i]:
                            distances[i] = hit_dist
                            
            # 2. Intersection with Arena boundary walls
            if cos_t > 1e-4:
                d_wall = (self.size - self.x) / cos_t
                if d_wall < distances[i]: distances[i] = d_wall
            elif cos_t < -1e-4:
                d_wall = (0.0 - self.x) / cos_t
                if d_wall < distances[i]: distances[i] = d_wall
                
            if sin_t > 1e-4:
                d_wall = (self.size - self.y) / sin_t
                if d_wall < distances[i]: distances[i] = d_wall
            elif sin_t < -1e-4:
                d_wall = (0.0 - self.y) / sin_t
                if d_wall < distances[i]: distances[i] = d_wall
                
        return np.clip(distances, 0.0, self.max_ray_dist)

    def _format_observation(self, norm_proximity, looming):
        """
        Formats 64-ray proximity and looming into 128 sensory channels:
          - Channels 0..31:   Left Eye Proximity  (Rays 32..63)
          - Channels 32..63:  Left Eye Looming    (Rays 32..63)
          - Channels 64..95:  Right Eye Proximity (Rays 0..31)
          - Channels 96..127: Right Eye Looming   (Rays 0..31)
        """
        right_dist = norm_proximity[:32]
        right_looming = looming[:32]
        left_dist = norm_proximity[32:]
        left_looming = looming[32:]
        
        return np.concatenate([left_dist, left_looming, right_dist, right_looming]).astype(np.float32)

    def get_expert_action(self, norm_proximity):
        """
        Biological Potential Field Guidance:
        Right hemisphere: rays 0..31 (azimuth -180 to 0) -> repulsion pushes agent to turn LEFT (>0)
        Left hemisphere: rays 32..63 (azimuth 0 to +180) -> repulsion pushes agent to turn RIGHT (<0)
        """
        right_weights = np.sin(np.linspace(0, np.pi, 32)) # peak at 90 deg right
        left_weights = np.sin(np.linspace(0, np.pi, 32))  # peak at 90 deg left
        
        right_repulsion = np.sum(norm_proximity[:32] * right_weights)
        left_repulsion = np.sum(norm_proximity[32:] * left_weights)
        
        # Omega: turn away from side with higher proximity
        omega = (right_repulsion - left_repulsion) * 2.5
        omega = np.clip(omega, -3.0, 3.0)
        
        # Frontal danger slows down thrust (rays 24..39 cover frontal +-45 degrees)
        front_danger = np.max(norm_proximity[24:40])
        thrust = max(0.2, 1.0 - front_danger * 0.8)
        
        u_steer_left = float(max(0.0, omega))
        u_steer_right = float(max(0.0, -omega))
        u_thrust = float(thrust)
        u_lift = 0.5
        
        return np.array([u_steer_left, u_steer_right, u_thrust, u_lift], dtype=np.float32)

    def step(self, motor_action, dt):
        """
        Continuous-time dynamic physics step.
        motor_action: [Steer_Left, Steer_Right, Thrust, Lift]
        dt: variable time step in seconds
        """
        steer_left = float(motor_action[0])
        steer_right = float(motor_action[1])
        thrust = float(np.clip(motor_action[2], 0.1, 1.5))
        
        # Clip angular velocity to +- 3.0 rad/s
        omega = np.clip(steer_left - steer_right, -3.0, 3.0)
        
        # Integrate heading
        self.heading += omega * dt
        self.heading = (self.heading + np.pi) % (2.0 * np.pi) - np.pi
        
        # Integrate position
        v = self.speed * thrust
        dx = v * math.cos(self.heading) * dt
        dy = v * math.sin(self.heading) * dt

        prev_x, prev_y = self.x, self.y
        self.x = float(np.clip(self.x + dx, self.boundary_margin, self.size - self.boundary_margin))
        self.y = float(np.clip(self.y + dy, self.boundary_margin, self.size - self.boundary_margin))
        # Count only the distance actually travelled after boundary clamping
        self.distance_traveled += math.hypot(self.x - prev_x, self.y - prev_y)
        
        # Cast rays at new position
        raw_dist = self._cast_rays()
        norm_prox = 1.0 - (raw_dist / self.max_ray_dist)
        
        # Compute real continuous looming rate dProx/dt
        looming = (norm_prox - self.prev_distances) / max(1e-4, dt)
        looming = np.clip(looming, 0.0, 10.0)
        self.prev_distances = np.copy(norm_prox)
        
        obs_128 = self._format_observation(norm_prox, looming)

        # Explicit geometric collision check. Ray probing alone can miss walls
        # (the clamp keeps a minimum distance) and obstacle corners.
        obstacle_clearance = min(
            [math.hypot(self.x - ox, self.y - oy) - r for ox, oy, r in self.obstacles],
            default=float("inf"),
        )
        boundary_clearance = min(self.x, self.size - self.x, self.y, self.size - self.y)
        is_collision = bool(
            min(obstacle_clearance, boundary_clearance) <= self.collision_clearance
        )
        
        return obs_128, norm_prox, is_collision
