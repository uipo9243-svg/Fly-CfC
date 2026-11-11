import sys
import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pybullet as p

# Add gym-pybullet-drones to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'gym-pybullet-drones')))
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from fly_wiring import FlyVisuomotorWiring, FlyCfC, load_motor_neuron_indices
from retinal_projector import RetinalProjector

# Patch DSLPIDControl._dslPIDAttitudeControl to eliminate Euler angle wraparound derivative spike
orig_dslPIDAttitudeControl = DSLPIDControl._dslPIDAttitudeControl
def patched_dslPIDAttitudeControl(self, control_timestep, thrust, cur_quat, target_euler, target_rpy_rates):
    cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
    diff = (cur_rpy - self.last_rpy + np.pi) % (2.0 * np.pi) - np.pi
    self.last_rpy = cur_rpy - diff
    return orig_dslPIDAttitudeControl(self, control_timestep, thrust, cur_quat, target_euler, target_rpy_rates)

DSLPIDControl._dslPIDAttitudeControl = patched_dslPIDAttitudeControl

class FlyBrainEndToEnd(nn.Module):
    """
    End-to-End Drosophila Connectome Engine:
    Retinal Projector (128 -> 1,683 Active VPNs) + FlyCfC (VPNs -> 1,961 Internal -> 4 Motor Outputs via DNs).
    """
    def __init__(self, wiring):
        super().__init__()
        self.projector = RetinalProjector()
        self.cfc = FlyCfC(wiring)

    def forward(self, raw_visual_seq, timespans=None, hx=None):
        vpn_seq = self.projector(raw_visual_seq)
        motor_seq, hx = self.cfc(vpn_seq, timespans=timespans, hx=hx)
        return motor_seq, hx

class PyBullet3DFlightEnv:
    """
    3D PyBullet Drones Environment for End-to-End Slalom Corridor Training.
    Simulates Bitcraze Crazyflie 2.X with 64-ray 3D compound eye vision,
    Front-Biased Vision weighting, and Bilateral Symmetry Breaking.
    """
    def __init__(self, arena_size=14.5, max_ray_dist=10.0, seed=42):
        self.arena_size = arena_size
        self.half_size = arena_size / 2.0
        self.max_ray_dist = max_ray_dist
        self.num_rays = 64
        self.rng = np.random.RandomState(seed)
        
        self.ctrl_freq = 30
        self.sim_freq = 240
        self.dt = 1.0 / self.ctrl_freq
        
        self.ray_angles = np.linspace(-np.pi, np.pi, self.num_rays, endpoint=False)
        self.prev_proximity = np.zeros(self.num_rays, dtype=np.float32)
        
        # Front-Biased Vision Weights (peak at +-35 degrees, strong in front!)
        self.ray_weights_right = np.zeros(32, dtype=np.float32)
        self.ray_weights_left = np.zeros(32, dtype=np.float32)
        for i in range(32):
            ang_rad = abs(self.ray_angles[i])
            self.ray_weights_right[i] = math.exp(-((ang_rad - 0.60)**2) / (2 * 0.45**2))
        for i in range(32):
            ang_rad = abs(self.ray_angles[32 + i])
            self.ray_weights_left[i] = math.exp(-((ang_rad - 0.60)**2) / (2 * 0.45**2))
            
        self.prev_omega = 0.0
        self.env = None
        self.ctrl = None
        self.client = None
        self.drone_id = None
        self.pillar_positions = [] # (x, y, radius)
        
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            
        if self.env is not None:
            self.env.close()
            
        # 1. สร้าง Slalom Gates Course โดยตรง (ทางเข้าเปิดโล่ง >= 3.0m ไม่มีเสาขวางประชิด)
        self.pillar_positions = []
        
        # Gate 1 (Entrance) - Gap 3.4 - 3.8m
        g1_y = float(self.rng.uniform(-2.8, -2.4))
        g1_dx = float(self.rng.uniform(-0.3, 0.3))
        self.pillar_positions.append((-1.8 + g1_dx, g1_y, 0.40))
        self.pillar_positions.append(( 1.8 + g1_dx, g1_y, 0.40))
        
        # Gate 2 (S-curve Left) - Gap 3.4 - 3.8m
        g2_y = float(self.rng.uniform(-0.4, 0.2))
        g2_dx = float(self.rng.uniform(-0.3, 0.3))
        self.pillar_positions.append((-3.0 + g2_dx, g2_y, 0.40))
        self.pillar_positions.append(( 0.6 + g2_dx, g2_y, 0.40))
        
        # Gate 3 (S-curve Right) - Gap 3.4 - 3.8m
        g3_y = float(self.rng.uniform(2.0, 2.6))
        g3_dx = float(self.rng.uniform(-0.3, 0.3))
        self.pillar_positions.append((-1.1 + g3_dx, g3_y, 0.40))
        self.pillar_positions.append(( 2.4 + g3_dx, g3_y, 0.40))
        
        # Gate 4 (Top Gate) - Gap 3.5m
        g4_y = float(self.rng.uniform(4.3, 4.8))
        self.pillar_positions.append((-2.6, g4_y, 0.40))
        self.pillar_positions.append(( 1.0, g4_y, 0.40))
        
        # Flank / Turnaround Pillars (ระยะห่าง >= 2.8m จากทางบิน)
        flank_pillars = [
            ( 3.8,  3.5, 0.40),
            ( 4.2,  0.5, 0.40),
            ( 3.5, -2.5, 0.40),
            (-4.8, -1.5, 0.45),
            (-4.8,  2.0, 0.45),
            ( 4.5, -4.5, 0.45),
            ( 4.5,  4.8, 0.45),
            (-4.5,  4.8, 0.45),
            (-4.5, -4.5, 0.45),
        ]
        self.pillar_positions.extend(flank_pillars)
        
        # 2. ปล่อยตัวโดรนที่ทางเข้า Slalom Corridor (ห่างเสา >= 2.2m และมุ่งหน้าเข้าสู่ Gate 1)
        init_x = float(self.rng.uniform(-2.3, -1.7))
        init_y = float(self.rng.uniform(-5.4, -4.8))
        init_heading = float(self.rng.uniform(np.pi/2.0 - 0.25, np.pi/2.0 + 0.25)) # มุ่งหน้าขึ้นทิศเหนือ (+Y)
        
        init_xyz = np.array([[init_x, init_y, 1.0]])
        init_rpy = np.array([[0.0, 0.0, init_heading]])
        
        # 3. สร้าง Environment
        self.env = CtrlAviary(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=init_xyz,
            initial_rpys=init_rpy,
            physics=Physics.PYB,
            pyb_freq=self.sim_freq,
            ctrl_freq=self.ctrl_freq,
            gui=False,
            obstacles=False
        )
        self.client = self.env.getPyBulletClient()
        self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        self.drone_id = self.env.DRONE_IDS[0]
        
        # รีเซ็ตสภาพแวดล้อมก่อนสร้างเสา
        obs, info = self.env.reset()
        
        # 4. สร้างเสาสลาลอมใน PyBullet
        pillar_colors = [
            [0.95, 0.22, 0.22, 1.0], [0.15, 0.75, 0.95, 1.0],
            [0.98, 0.85, 0.15, 1.0], [0.20, 0.88, 0.35, 1.0],
            [0.88, 0.25, 0.88, 1.0], [0.98, 0.50, 0.12, 1.0]
        ]
        for idx, (ox, oy, r) in enumerate(self.pillar_positions):
            col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=4.0, physicsClientId=self.client)
            vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=4.0, rgbaColor=pillar_colors[idx % len(pillar_colors)], physicsClientId=self.client)
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id,
                              basePosition=[ox, oy, 2.0], physicsClientId=self.client)
                              
        # กำแพงล้อม 4 ทิศ
        wall_col = [0.35, 0.40, 0.48, 1.0]
        hs = self.half_size
        for wx, wy, l, w in [(-hs, 0.0, 0.25, self.arena_size), (hs, 0.0, 0.25, self.arena_size),
                             (0.0, -hs, self.arena_size, 0.25), (0.0, hs, self.arena_size, 0.25)]:
            c_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[l/2, w/2, 2.5], physicsClientId=self.client)
            v_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[l/2, w/2, 2.5], rgbaColor=wall_col, physicsClientId=self.client)
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c_id, baseVisualShapeIndex=v_id,
                              basePosition=[wx, wy, 2.5], physicsClientId=self.client)
                              
        self.cmd_yaw = init_heading
        self.prev_omega = 0.0
        self.prev_proximity = np.zeros(self.num_rays, dtype=np.float32)
        
        obs_128, norm_prox, looming = self._get_observation(obs[0])
        self.prev_proximity = np.copy(norm_prox)
        # ที่ t=0 ค่า looming เป็น 0
        looming = np.zeros(self.num_rays, dtype=np.float32)
        obs_128[32:64] = 0.0
        obs_128[96:128] = 0.0
        
        return obs_128, norm_prox, looming, obs[0]

    def _get_observation(self, drone_state):
        pos = drone_state[0:3]
        rpy = drone_state[7:10]
        heading = float(rpy[2])
        
        from_positions = []
        to_positions = []
        for i, a_off in enumerate(self.ray_angles):
            ray_theta = heading + a_off
            elevation = math.sin(i * 3.5) * 0.15 # +- 8.5 องศา
            cos_el = math.cos(elevation)
            sin_el = math.sin(elevation)
            dir_x = cos_el * math.cos(ray_theta)
            dir_y = cos_el * math.sin(ray_theta)
            dir_z = sin_el
            from_positions.append([pos[0] + 0.06 * dir_x, pos[1] + 0.06 * dir_y, pos[2] + 0.06 * dir_z])
            to_positions.append([pos[0] + self.max_ray_dist * dir_x, pos[1] + self.max_ray_dist * dir_y, pos[2] + self.max_ray_dist * dir_z])
            
        ray_results = p.rayTestBatch(from_positions, to_positions, physicsClientId=self.client)
        distances = np.zeros(self.num_rays, dtype=np.float32)
        for i, res in enumerate(ray_results):
            distances[i] = 0.06 + res[2] * (self.max_ray_dist - 0.06)
            
        norm_prox = np.clip(1.0 - (distances / self.max_ray_dist), 0.0, 1.0)
        looming = np.clip((norm_prox - self.prev_proximity) / max(1e-4, self.dt), 0.0, 10.0)
        
        # 128-dim: ตาซ้าย (0..63) และตาขวา (64..127)
        left_dist = norm_prox[32:]
        left_looming = looming[32:]
        right_dist = norm_prox[:32]
        right_looming = looming[:32]
        obs_128 = np.concatenate([left_dist, left_looming, right_dist, right_looming]).astype(np.float32)
        
        return obs_128, norm_prox, looming

    def get_expert_action(self, drone_state, norm_prox, looming):
        """
        Front-Biased Biological Potential Field Expert with Bilateral Symmetry Breaking:
        Compensates for quadcopter inertia with predictive front-quarter steering,
        active deceleration, and wall boundary repulsion.
        """
        pos = drone_state[0:3]
        
        # 1. Front-Biased Vision Repulsion
        right_danger = np.sum(norm_prox[:32] * self.ray_weights_right) + 0.35 * np.sum(looming[:32] * self.ray_weights_right)
        left_danger = np.sum(norm_prox[32:] * self.ray_weights_left) + 0.35 * np.sum(looming[32:] * self.ray_weights_left)
        
        diff = right_danger - left_danger
        omega = diff * 3.8
        
        # 2. Bilateral Symmetry Breaking when frontal obstacle is detected
        front_prox = float(np.max(norm_prox[28:36])) # -22.5 to +22.5 degrees
        front_looming = float(np.max(looming[28:36]))
        
        if front_prox > 0.28:
            if abs(diff) < 0.08:
                escape_dir = 1.0 if self.prev_omega >= 0 else -1.0
            else:
                escape_dir = math.copysign(1.0, diff)
            omega += escape_dir * (2.4 * front_prox + 0.35 * front_looming)
            
        # 3. Boundary Wall Repulsion (keep inside [-4.8, 4.8])
        if pos[0] > 4.8: omega -= 2.6 * (pos[0] - 4.8)
        if pos[0] < -4.8: omega += 2.6 * (-4.8 - pos[0])
        if pos[1] > 4.8: omega -= 2.6 * (pos[1] - 4.8)
        if pos[1] < -4.8: omega += 2.6 * (-4.8 - pos[1])
        
        omega = float(np.clip(omega, -2.8, 2.8))
        self.prev_omega = omega
        
        # Forward speed and thrust modulation
        fwd_speed = float(np.clip(1.30 - 0.75 * front_prox, 0.55, 1.30))
        thrust = float(np.clip(fwd_speed / 1.30, 0.45, 1.10))
        
        # Lift: maintain altitude 1.0m
        lift = float(np.clip(0.50 + (1.0 - pos[2]) * 0.45, 0.20, 0.80))
        
        u_steer_left = float(max(0.0, omega))
        u_steer_right = float(max(0.0, -omega))
        
        return np.array([u_steer_left, u_steer_right, thrust, lift], dtype=np.float32)

    def step(self, action):
        steer_left, steer_right, thrust, lift = float(action[0]), float(action[1]), float(action[2]), float(action[3])
        omega = steer_left - steer_right
        
        # Continuous flight tracking
        self.cmd_yaw = (self.cmd_yaw + omega * self.dt + np.pi) % (2.0 * np.pi) - np.pi
        fwd_speed = float(np.clip(thrust * 1.30, 0.55, 1.30))
        lookahead = float(np.clip(fwd_speed * 0.55, 0.6, 1.0))
        
        obs_raw = self.env._getDroneStateVector(0)
        pos = obs_raw[0:3]
        quat = obs_raw[3:7]
        rpy = obs_raw[7:10]
        vel = obs_raw[10:13]
        ang_v = obs_raw[13:16]
        
        target_z = float(np.clip(1.0 + (lift - 0.5) * 0.35, 0.8, 1.4))
        target_pos = np.array([
            pos[0] + lookahead * math.cos(self.cmd_yaw),
            pos[1] + lookahead * math.sin(self.cmd_yaw),
            target_z
        ])
        target_vel = np.array([
            fwd_speed * math.cos(self.cmd_yaw),
            fwd_speed * math.sin(self.cmd_yaw),
            (target_z - pos[2]) * 1.2
        ])
        
        rpms, _, _ = self.ctrl.computeControl(
            control_timestep=self.dt,
            cur_pos=pos,
            cur_quat=quat,
            cur_vel=vel,
            cur_ang_vel=ang_v,
            target_pos=target_pos,
            target_rpy=np.array([0.0, 0.0, self.cmd_yaw]),
            target_vel=target_vel
        )
        
        obs_drone, _, _, _, _ = self.env.step(rpms.reshape(1, 4))
        
        # ตรวจสอบการชนจริงผ่าน PyBullet contacts
        contacts = p.getContactPoints(bodyA=self.drone_id, physicsClientId=self.client)
        collided = (len(contacts) > 0)
        
        obs_128, norm_prox, looming = self._get_observation(obs_drone[0])
        self.prev_proximity = np.copy(norm_prox)
        
        return obs_128, norm_prox, looming, obs_drone[0], collided

    def close(self):
        if self.env is not None:
            self.env.close()

def collect_3d_expert_dataset(num_episodes=120, seq_len=60, seed=42):
    """
    Collects high-fidelity 3D drone flight trajectories directly in Slalom Corridors.
    Guarantees 100% collision-free episodes in PyBullet physics.
    """
    print(f"\n--- กำลังสร้างชุดข้อมูลการบินสลาลอม 3 มิติ (3D Slalom Expert Dataset: {num_episodes} Episodes) ---")
    t0_data = time.time()
    
    all_visuals = []
    all_timespans = []
    all_expert_actions = []
    
    collected_episodes = 0
    attempt = 0
    
    while collected_episodes < num_episodes and attempt < num_episodes * 4:
        attempt += 1
        env = PyBullet3DFlightEnv(arena_size=14.5, max_ray_dist=10.0, seed=seed + attempt)
        obs_128, norm_prox, looming, drone_state = env.reset()
        
        ep_vis = []
        ep_dt = []
        ep_act = []
        
        collided = False
        prev_dt = 0.0
        
        for step in range(seq_len):
            expert_act = env.get_expert_action(drone_state, norm_prox, looming)
            
            ep_vis.append(obs_128)
            ep_dt.append(prev_dt)
            ep_act.append(expert_act)
            
            obs_128, norm_prox, looming, drone_state, step_collided = env.step(expert_act)
            prev_dt = env.dt
            
            if step_collided:
                collided = True
                break
                
        env.close()
        
        if not collided:
            all_visuals.append(ep_vis)
            all_timespans.append(ep_dt)
            all_expert_actions.append(ep_act)
            collected_episodes += 1
            if collected_episodes % 20 == 0 or collected_episodes == num_episodes:
                print(f"  - บันทึกการบินสลาลอม 3D สำเร็จ: {collected_episodes}/{num_episodes} รอบ ({(time.time()-t0_data):.1f}s)")
                
    X = torch.tensor(np.array(all_visuals), dtype=torch.float32)
    T = torch.tensor(np.array(all_timespans), dtype=torch.float32)
    Y = torch.tensor(np.array(all_expert_actions), dtype=torch.float32)
    
    left_looming_max = float(X[:, :, 32:64].max())
    right_looming_max = float(X[:, :, 96:128].max())
    print(f"ชุดข้อมูล 3D Slalom เสร็จสมบูรณ์: X={list(X.shape)}, T={list(T.shape)}, Y={list(Y.shape)}")
    print(f"  - ตรวจสอบ Looming Channel สูงสุด: Left={left_looming_max:.4f}, Right={right_looming_max:.4f}")
    
    return X, T, Y

def train_fly_cfc_v2():
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("====================================================================")
    print("STARTING FLY-CFC v2 RETRAINING ON 3D SLALOM CORRIDORS")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("====================================================================")
    
    # 1. โหลด Drosophila Connectome
    print("\nกำลังโหลด Drosophila Male CNS Connectome Adjacencies...")
    sensory_adj = np.load("male_cns_connectome/sensory_adj.npy")
    internal_adj = np.load("male_cns_connectome/internal_adj.npy")
    
    wiring = FlyVisuomotorWiring(
        sensory_adjacency_matrix=sensory_adj,
        adjacency_matrix=internal_adj,
        output_dim=4,
        motor_indices=load_motor_neuron_indices(),
    )
    
    model = FlyBrainEndToEnd(wiring).to(device)
    print(f"สร้างโมเดล End-to-End FlyBrain v2 สำเร็จ:")
    print(f"  - Active Sensory VPNs: {wiring.input_dim} โหนด")
    print(f"  - Internal Recurrent Neurons: {wiring.units} โหนด (CfC continuous-time)")
    print(f"  - Descending Motor Neurons: {wiring.output_dim} เอาต์พุต (Steer-L, Steer-R, Thrust, Lift)")
    
    # 2. เก็บชุดข้อมูลการบินสลาลอม 3 มิติ (120 Episodes x 60 Steps = 7,200 Steps)
    X, T, Y = collect_3d_expert_dataset(num_episodes=120, seq_len=60, seed=100)
    
    # แบ่ง Train / Validation 85:15
    num_total = len(X)
    num_train = int(num_total * 0.85)
    
    indices = torch.randperm(num_total)
    train_idx = indices[:num_train]
    val_idx = indices[num_train:]
    
    train_X, train_T, train_Y = X[train_idx].to(device), T[train_idx].to(device), Y[train_idx].to(device)
    val_X, val_T, val_Y = X[val_idx].to(device), T[val_idx].to(device), Y[val_idx].to(device)
    
    # 3. ตั้งค่าการฝึกสอน
    EPOCHS = 30
    BATCH_SIZE = 8
    LR = 1.8e-3
    WEIGHT_DECAY = 1e-4
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    
    print(f"\n--- เริ่มต้นการฝึกสอนโมเดล Fly-CfC v2 ({EPOCHS} Epochs, Batch={BATCH_SIZE}, LR={LR}) ---")
    t0_train = time.time()
    best_val_loss = float('inf')
    ckpt_path = "models/fly_cfc_v2_3d.pth"
    os.makedirs("models", exist_ok=True)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(num_train)
        epoch_loss = 0.0
        num_batches = 0
        
        for b_start in range(0, num_train, BATCH_SIZE):
            b_idx = perm[b_start:b_start + BATCH_SIZE]
            bx = train_X[b_idx]
            bt = train_T[b_idx]
            by = train_Y[b_idx]
            
            optimizer.zero_grad()
            pred_y, _ = model(bx, timespans=bt)
            loss = criterion(pred_y, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
        scheduler.step()
        train_loss = epoch_loss / max(1, num_batches)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred, _ = model(val_X, timespans=val_T)
            val_loss = criterion(val_pred, val_Y).item()
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_mse': val_loss,
                'wiring_info': {
                    'sensory_dim': wiring.input_dim,
                    'internal_dim': wiring.units,
                    'output_dim': wiring.output_dim,
                }
            }, ckpt_path)
            mark = " [BEST SAVED]"
        else:
            mark = ""
            
        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"Epoch [{epoch:2d}/{EPOCHS}] - Train Loss: {train_loss:.4f} | Val MSE: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}{mark}")
            
    total_train_time = time.time() - t0_train
    print(f"\nการฝึกสอน Fly-CfC v2 บน 3D Slalom เสร็จสิ้น ({total_train_time:.1f}s)")
    print(f"โมเดลยอดเยี่ยมที่สุด: Val MSE = {best_val_loss:.4f} บันทึกไว้ที่ {ckpt_path}")

if __name__ == "__main__":
    train_fly_cfc_v2()
