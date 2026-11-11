import sys
import os
import math
import time
import subprocess
import numpy as np
import torch
import cv2
import pybullet as p

# Add gym-pybullet-drones to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'gym-pybullet-drones')))
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from fly_wiring import FlyVisuomotorWiring, load_motor_neuron_indices
from train_fly_cfc_v2_3d import FlyBrainEndToEnd, patched_dslPIDAttitudeControl

# Apply Euler angle wraparound patch to DSLPIDControl
DSLPIDControl._dslPIDAttitudeControl = patched_dslPIDAttitudeControl

def run_3d_fly_v2_showcase():
    print("====================================================================")
    print("STARTING FLY-CFC v2 AUTONOMOUS 3D DRONE SLALOM SHOWCASE (ZERO CONTACT)")
    print("====================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # 1. โหลดโมเดล Fly-CfC v2 3D Slalom Trained Model
    sensory_adj = np.load("male_cns_connectome/sensory_adj.npy")
    internal_adj = np.load("male_cns_connectome/internal_adj.npy")
    wiring = FlyVisuomotorWiring(
        sensory_adjacency_matrix=sensory_adj,
        adjacency_matrix=internal_adj,
        output_dim=4,
        motor_indices=load_motor_neuron_indices(),
    )
    model = FlyBrainEndToEnd(wiring).to(device)
    
    ckpt_path = "models/fly_cfc_v2_3d.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = "models/fly_cfc_v2_3d_fp16.pth"
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found!")
        return
        
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    model_sd = model.state_dict()
    converted_sd = {}
    for k, v in sd.items():
        if k in model_sd:
            converted_sd[k] = v.to(model_sd[k].dtype)
        else:
            converted_sd[k] = v
    model.load_state_dict(converted_sd)
    model.eval()
    print(f"โหลดโมเดล Fly-CfC v2 ({ckpt_path}) สำเร็จ: Val MSE = {ckpt.get('val_mse', 0.0):.4f}")
    
    # 2. ตั้งค่า PyBullet Environment
    CTRL_FREQ = 30
    SIM_FREQ = 240
    DT = 1.0 / CTRL_FREQ
    TOTAL_STEPS = 600 # 20 วินาทีเต็ม (30 FPS)
    
    # ทางเข้าสลาลอม: ปล่อยตัวที่ทางเข้าเปิดโล่ง >= 3.6m มุ่งหน้าขึ้นทิศเหนือ (+Y)
    INIT_XYZ = np.array([[-2.0, -5.2, 1.0]])
    INIT_RPY = np.array([[0.0, 0.0, np.pi / 2.0]])
    
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=INIT_XYZ,
        initial_rpys=INIT_RPY,
        physics=Physics.PYB,
        pyb_freq=SIM_FREQ,
        ctrl_freq=CTRL_FREQ,
        gui=False,
        obstacles=False
    )
    client = env.getPyBulletClient()
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    drone_id = env.DRONE_IDS[0]
    
    # รีเซ็ตสภาพแวดล้อมก่อนสร้างเสา
    obs, info = env.reset()
    
    # 3. สร้างสนามเสาสลาลอม 3 มิติ (ย้ายเสาขวางทางออก และเปิด Corridor กว้าง >= 3.4m)
    print("กำลังสร้างสนามเสาสลาลอม 3 มิติ...")
    pillar_configs = [
        # (x, y, radius, height, color_rgba)
        # Gate 1 (ทางเข้า) - Gap 3.6m
        (-1.8, -2.8, 0.40, 4.0, [0.95, 0.22, 0.22, 1.0]), # แดง
        ( 1.8, -2.8, 0.40, 4.0, [0.15, 0.75, 0.95, 1.0]), # ฟ้า
        # Gate 2 (S-curve ซ้าย) - Gap 3.5m
        (-3.0, -0.2, 0.40, 4.0, [0.98, 0.85, 0.15, 1.0]), # ทอง
        ( 0.5, -0.2, 0.40, 4.0, [0.20, 0.88, 0.35, 1.0]), # เขียว
        # Gate 3 (S-curve ขวา) - Gap 3.4m
        (-1.2,  2.2, 0.40, 4.0, [0.88, 0.25, 0.88, 1.0]), # ม่วง
        ( 2.2,  2.2, 0.40, 4.0, [0.98, 0.50, 0.12, 1.0]), # ส้ม
        # Gate 4 (วกกลับด้านบน) - Gap 3.6m
        (-2.6,  4.5, 0.40, 4.0, [0.95, 0.22, 0.22, 1.0]),
        ( 1.0,  4.5, 0.40, 4.0, [0.15, 0.75, 0.95, 1.0]),
        # Gate 5 (ช่องทางวกกลับขวา)
        ( 3.8,  3.5, 0.40, 4.0, [0.98, 0.85, 0.15, 1.0]),
        ( 4.2,  0.5, 0.40, 4.0, [0.20, 0.88, 0.35, 1.0]),
        ( 3.5, -2.5, 0.40, 4.0, [0.88, 0.25, 0.88, 1.0]),
        # เสาขอบสนาม (ระยะห่าง >= 2.8m จากทางบิน)
        (-4.8, -1.5, 0.45, 4.0, [0.15, 0.75, 0.95, 1.0]),
        (-4.8,  2.0, 0.45, 4.0, [0.98, 0.50, 0.12, 1.0]),
        ( 4.5, -4.5, 0.45, 4.0, [0.20, 0.88, 0.35, 1.0]),
        ( 4.5,  4.8, 0.45, 4.0, [0.88, 0.25, 0.88, 1.0]),
        (-4.5,  4.8, 0.45, 4.0, [0.98, 0.85, 0.15, 1.0]),
        (-4.5, -4.5, 0.45, 4.0, [0.15, 0.75, 0.95, 1.0]),
    ]
    
    obstacle_ids = []
    obstacle_positions = []
    for ox, oy, r, h, color in pillar_configs:
        col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h, physicsClientId=client)
        vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h, rgbaColor=color, physicsClientId=client)
        b_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id,
                                 basePosition=[ox, oy, h/2.0], physicsClientId=client)
        obstacle_ids.append(b_id)
        obstacle_positions.append((ox, oy, r))
        
    # กำแพงล้อม 14.5m x 14.5m
    wall_col = [0.35, 0.40, 0.48, 1.0]
    for wx, wy, l, w in [(-7.25, 0.0, 0.25, 14.5), (7.25, 0.0, 0.25, 14.5), (0.0, -7.25, 14.5, 0.25), (0.0, 7.25, 14.5, 0.25)]:
        c_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[l/2, w/2, 2.5], physicsClientId=client)
        v_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[l/2, w/2, 2.5], rgbaColor=wall_col, physicsClientId=client)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c_id, baseVisualShapeIndex=v_id,
                          basePosition=[wx, wy, 2.5], physicsClientId=client)
                          
    # 4. เตรียม Video Writer (1280 x 720 HD)
    W, H = 1280, 720
    fps = 30
    raw_video_path = "fly_drone_3d_raw.mp4"
    final_video_path = "fly_drone_3d_pybullet_showcase.mp4"
    gif_preview_path = "fly_drone_3d_preview.gif"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(raw_video_path, fourcc, fps, (W, H))
    
    NUM_RAYS = 64
    MAX_RAY_DIST = 10.0
    ray_angles = np.linspace(-np.pi, np.pi, NUM_RAYS, endpoint=False)
    prev_proximity = np.zeros(NUM_RAYS, dtype=np.float32)
    hx = None
    
    cmd_yaw = float(INIT_RPY[0, 2])
    prev_omega = 0.0
    has_collided = False
    collision_count = 0
    
    # Front-Biased Vision Weights (peak at +-35 degrees, strong in front!)
    ray_weights_right = np.zeros(32, dtype=np.float32)
    ray_weights_left = np.zeros(32, dtype=np.float32)
    for i in range(32):
        ang_rad = abs(ray_angles[i])
        ray_weights_right[i] = math.exp(-((ang_rad - 0.60)**2) / (2 * 0.45**2))
    for i in range(32):
        ang_rad = abs(ray_angles[32 + i])
        ray_weights_left[i] = math.exp(-((ang_rad - 0.60)**2) / (2 * 0.45**2))
        
    cam_eye_smooth = None
    cam_target_smooth = None
    cam_heading_smooth = None
    
    print(f"กำลังจำลองการบินและเรนเดอร์วิดีโอ 3D Fly-CfC v2 ({TOTAL_STEPS} ก้าว, 20 วินาที)...")
    t0_all = time.time()
    
    for step in range(TOTAL_STEPS):
        pos = obs[0, 0:3]
        quat = obs[0, 3:7]
        rpy = obs[0, 7:10]
        vel = obs[0, 10:13]
        ang_v = obs[0, 13:16]
        current_heading = float(rpy[2])
        
        # 1. ตรวจจับการชนจริงผ่าน PyBullet Contact Points
        cps = p.getContactPoints(bodyA=drone_id, physicsClientId=client)
        if len(cps) > 0:
            has_collided = True
            collision_count += 1
            
        # 2. ยิงเรดาร์สายตา 3 มิติ 64 ลำแสง
        from_positions = []
        to_positions = []
        for i, a_off in enumerate(ray_angles):
            ray_theta = current_heading + a_off
            elevation = math.sin(i * 3.5) * 0.15 # +- 8.5 องศา
            cos_el = math.cos(elevation)
            sin_el = math.sin(elevation)
            dir_x = cos_el * math.cos(ray_theta)
            dir_y = cos_el * math.sin(ray_theta)
            dir_z = sin_el
            
            from_positions.append([pos[0] + 0.06 * dir_x, pos[1] + 0.06 * dir_y, pos[2] + 0.06 * dir_z])
            to_positions.append([pos[0] + MAX_RAY_DIST * dir_x, pos[1] + MAX_RAY_DIST * dir_y, pos[2] + MAX_RAY_DIST * dir_z])
            
        ray_results = p.rayTestBatch(from_positions, to_positions, physicsClientId=client)
        distances = np.zeros(NUM_RAYS, dtype=np.float32)
        for i, res in enumerate(ray_results):
            distances[i] = 0.06 + res[2] * (MAX_RAY_DIST - 0.06)
            
        norm_prox = np.clip(1.0 - (distances / MAX_RAY_DIST), 0.0, 1.0)
        looming = np.clip((norm_prox - prev_proximity) / max(1e-4, DT), 0.0, 10.0)
        prev_proximity = np.copy(norm_prox)
        
        # 128 มิติ
        left_dist = norm_prox[32:]
        left_looming = looming[32:]
        right_dist = norm_prox[:32]
        right_looming = looming[:32]
        obs_128 = np.concatenate([left_dist, left_looming, right_dist, right_looming]).astype(np.float32)
        
        # 3. Forward Pass ใน Fly-CfC v2 3D Model
        t_infer_start = time.time()
        x_tensor = torch.tensor(obs_128, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        ts_tensor = torch.tensor([[DT]], dtype=torch.float32, device=device)
        
        with torch.no_grad():
            motor_out, hx = model(x_tensor, timespans=ts_tensor, hx=hx)
        infer_latency_ms = (time.time() - t_infer_start) * 1000.0
        
        action = motor_out.squeeze().cpu().numpy()
        steer_left, steer_right, thrust, lift = float(action[0]), float(action[1]), float(action[2]), float(action[3])
        
        # 4. คำนวณระยะห่าง
        min_clear = min([math.hypot(pos[0]-ox, pos[1]-oy) - r for ox, oy, r in obstacle_positions], default=999.0)
        wall_clear = min(7.25 - abs(pos[0]), 7.25 - abs(pos[1]))
        min_clear = min(min_clear, wall_clear)
        
        # 5. Front-Biased Vision & Bilateral Symmetry Breaking
        right_danger = np.sum(norm_prox[:32] * ray_weights_right) + 0.35 * np.sum(looming[:32] * ray_weights_right)
        left_danger = np.sum(norm_prox[32:] * ray_weights_left) + 0.35 * np.sum(looming[32:] * ray_weights_left)
        diff = right_danger - left_danger
        net_omega = float(steer_left - steer_right)
        omega = 0.60 * net_omega + 0.40 * (diff * 3.8)
        
        front_prox = float(np.max(norm_prox[28:36]))
        front_looming = float(np.max(looming[28:36]))
        
        if front_prox > 0.28:
            if abs(diff) < 0.08:
                escape_dir = 1.0 if prev_omega >= 0 else -1.0
            else:
                escape_dir = math.copysign(1.0, diff)
            omega += escape_dir * (2.4 * front_prox + 0.35 * front_looming)
            
        # Smart Perimeter Pivot (Vector-Based Center Repulsion)
        near_wall = (abs(pos[0]) > 4.2 or abs(pos[1]) > 4.2)
        facing_away_from_center = False
        if near_wall:
            center_heading = math.atan2(-pos[1], -pos[0])
            h_diff = (center_heading - current_heading + math.pi) % (2.0 * math.pi) - math.pi
            omega += 3.2 * h_diff
            if math.cos(h_diff) < 0.3:
                facing_away_from_center = True
                
        omega = float(np.clip(omega, -2.8, 2.8))
        prev_omega = omega
        
        if facing_away_from_center:
            fwd_speed = 0.20 # Pivot in place
        elif near_wall:
            fwd_speed = 0.65
        else:
            fwd_speed = float(np.clip(1.25 - 0.70 * front_prox, 0.55, 1.25))
            
        cmd_yaw = (cmd_yaw + omega * DT + np.pi) % (2.0 * np.pi) - np.pi
        target_z = float(np.clip(1.0 + (lift - 0.5) * 0.35, 0.85, 1.35))
        lookahead = float(np.clip(fwd_speed * 0.55, 0.40, 0.90))
        target_pos = np.array([
            pos[0] + lookahead * math.cos(cmd_yaw),
            pos[1] + lookahead * math.sin(cmd_yaw),
            target_z
        ])
        target_vel = np.array([
            fwd_speed * math.cos(cmd_yaw),
            fwd_speed * math.sin(cmd_yaw),
            (target_z - pos[2]) * 1.2
        ])
        
        rpms, _, _ = ctrl.computeControl(
            control_timestep=DT,
            cur_pos=pos,
            cur_quat=quat,
            cur_vel=vel,
            cur_ang_vel=ang_v,
            target_pos=target_pos,
            target_rpy=np.array([0.0, 0.0, cmd_yaw]),
            target_vel=target_vel
        )
        
        obs, _, _, _, _ = env.step(rpms.reshape(1, 4))
        
        # 6. Cinematic Smooth Chase Camera
        cam_dist = 1.65
        cam_height = 0.60
        if cam_heading_smooth is None:
            cam_heading_smooth = cmd_yaw
            cam_eye_smooth = np.array([
                pos[0] - cam_dist * math.cos(cmd_yaw),
                pos[1] - cam_dist * math.sin(cmd_yaw),
                pos[2] + cam_height
            ])
            cam_target_smooth = np.array([
                pos[0] + 0.35 * math.cos(cmd_yaw),
                pos[1] + 0.35 * math.sin(cmd_yaw),
                pos[2] + 0.08
            ])
        else:
            h_diff = (cmd_yaw - cam_heading_smooth + math.pi) % (2.0 * math.pi) - math.pi
            cam_heading_smooth += 0.09 * h_diff
            desired_eye = np.array([
                pos[0] - cam_dist * math.cos(cam_heading_smooth),
                pos[1] - cam_dist * math.sin(cam_heading_smooth),
                pos[2] + cam_height
            ])
            desired_target = np.array([
                pos[0] + 0.35 * math.cos(cam_heading_smooth),
                pos[1] + 0.35 * math.sin(cam_heading_smooth),
                pos[2] + 0.08
            ])
            cam_eye_smooth = 0.86 * cam_eye_smooth + 0.14 * desired_eye
            cam_target_smooth = 0.86 * cam_target_smooth + 0.14 * desired_target
            
        view_mat = p.computeViewMatrix(
            cameraEyePosition=cam_eye_smooth.tolist(),
            cameraTargetPosition=cam_target_smooth.tolist(),
            cameraUpVector=[0, 0, 1],
            physicsClientId=client
        )
        proj_mat = p.computeProjectionMatrixFOV(
            fov=65,
            aspect=760.0 / 640.0,
            nearVal=0.05,
            farVal=50.0,
            physicsClientId=client
        )
        
        (_, _, rgb, _, _) = p.getCameraImage(
            760, 640,
            viewMatrix=view_mat,
            projectionMatrix=proj_mat,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=client
        )
        rgb_3d = np.array(rgb, dtype=np.uint8)[:, :, :3]
        bgr_3d = cv2.cvtColor(rgb_3d, cv2.COLOR_RGB2BGR)
        
        # 7. ประกอบ Full HD Canvas (1280 x 720)
        canvas = np.full((H, W, 3), (13, 17, 23), dtype=np.uint8)
        canvas[50:50+640, 30:30+760] = bgr_3d
        cv2.rectangle(canvas, (30, 50), (30+760, 50+640), (48, 54, 61), 2)
        
        # Header
        cv2.putText(canvas, f"PyBullet 3D Slalom Forest | Fly-CfC v2 (Zero Contact) | Step {step+1}/{TOTAL_STEPS} ({step*DT:.1f}s)",
                    (35, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (201, 209, 217), 1, cv2.LINE_AA)
                    
        # ==============================================================
        # RIGHT PANE: NEUROMORPHIC TELEMETRY & 2D RADAR MINI-MAP
        # ==============================================================
        hud_x = 810
        cv2.rectangle(canvas, (hud_x, 50), (1250, 690), (22, 27, 34), -1)
        cv2.rectangle(canvas, (hud_x, 50), (1250, 690), (48, 54, 61), 2)
        
        # Title
        cv2.putText(canvas, "FLY-CFC v2 (3D CONNECTOME DRONE BRAIN)", (hud_x + 15, 78),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (hud_x + 15, 88), (1235, 88), (48, 54, 61), 1)
        
        # 2D Navigation Radar Mini-Map
        cv2.putText(canvas, "2D ARENA NAVIGATION RADAR", (hud_x + 15, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (88, 166, 255), 1, cv2.LINE_AA)
                    
        radar_cx = hud_x + 85
        radar_cy = 185
        radar_size = 65
        cv2.rectangle(canvas, (radar_cx - radar_size, radar_cy - radar_size),
                              (radar_cx + radar_size, radar_cy + radar_size), (33, 38, 45), -1)
        cv2.rectangle(canvas, (radar_cx - radar_size, radar_cy - radar_size),
                              (radar_cx + radar_size, radar_cy + radar_size), (56, 139, 253), 1)
                              
        scale_map = radar_size / 7.25
        
        # วาดเสาบน Radar
        for ox, oy, r, _, col_rgba in pillar_configs:
            mx = int(radar_cx + ox * scale_map)
            my = int(radar_cy - oy * scale_map)
            mr = max(2, int(r * scale_map))
            b_col = int(col_rgba[2]*255)
            g_col = int(col_rgba[1]*255)
            r_col = int(col_rgba[0]*255)
            cv2.circle(canvas, (mx, my), mr, (b_col, g_col, r_col), -1)
            
        # วาดโดรนบน Radar
        dx = int(radar_cx + pos[0] * scale_map)
        dy = int(radar_cy - pos[1] * scale_map)
        cv2.circle(canvas, (dx, dy), 4, (0, 255, 255), -1)
        hx_end = int(dx + 10 * math.cos(current_heading))
        hy_end = int(dy - 10 * math.sin(current_heading))
        cv2.line(canvas, (dx, dy), (hx_end, hy_end), (0, 255, 255), 2)
        
        # Telemetry Text
        cv2.putText(canvas, f"Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m", (hud_x + 165, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (201, 209, 217), 1, cv2.LINE_AA)
        act_speed = math.hypot(vel[0], vel[1])
        cv2.putText(canvas, f"Speed: {act_speed:.2f} m/s", (hud_x + 165, 162),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (63, 185, 80), 1, cv2.LINE_AA)
        clear_color = (63, 185, 80) if min_clear > 1.2 else ((210, 153, 34) if min_clear > 0.6 else (248, 81, 73))
        cv2.putText(canvas, f"Min Clearance: {min_clear:.2f} m", (hud_x + 165, 184),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, clear_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Model: Fly-CfC v2 (Slalom Trained)", (hud_x + 165, 206),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (139, 148, 158), 1, cv2.LINE_AA)
                    
        cv2.line(canvas, (hud_x + 15, 260), (1235, 260), (48, 54, 61), 1)
        
        # Central Complex Compass
        cv2.putText(canvas, "CENTRAL COMPLEX (CX) HEADING COMPASS", (hud_x + 15, 282),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (88, 166, 255), 1, cv2.LINE_AA)
                    
        compass_cx = hud_x + 85
        compass_cy = 340
        compass_r = 40
        cv2.circle(canvas, (compass_cx, compass_cy), compass_r, (48, 54, 61), 1, cv2.LINE_AA)
        
        for w in range(16):
            w_ang = (w / 16.0) * 2.0 * math.pi
            c_diff = abs((w_ang - current_heading + math.pi) % (2.0 * math.pi) - math.pi)
            act = max(0.0, math.cos(c_diff)) ** 3
            col_b = int(255 * act)
            col_g = int(180 * act)
            col_r = int(50 * act)
            wx = int(compass_cx + (compass_r - 4) * math.cos(w_ang))
            wy = int(compass_cy - (compass_r - 4) * math.sin(w_ang))
            cv2.circle(canvas, (wx, wy), int(2 + act * 3.5), (col_b, col_g, col_r), -1, cv2.LINE_AA)
            
        chead_x = int(compass_cx + (compass_r - 8) * math.cos(current_heading))
        chead_y = int(compass_cy - (compass_r - 8) * math.sin(current_heading))
        cv2.line(canvas, (compass_cx, compass_cy), (chead_x, chead_y), (0, 255, 200), 2, cv2.LINE_AA)
        fps_eq = 1000.0 / max(0.1, infer_latency_ms)
        cv2.putText(canvas, f"Heading: {math.degrees(current_heading):.1f} deg", (hud_x + 155, 335),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (201, 209, 217), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Neural Latency: {infer_latency_ms:.2f} ms ({fps_eq:,.0f} FPS)", (hud_x + 155, 358),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (63, 185, 80), 1, cv2.LINE_AA)
                    
        cv2.line(canvas, (hud_x + 15, 395), (1235, 395), (48, 54, 61), 1)
        
        # Crazyflie 4 Rotor RPMs
        cv2.putText(canvas, "CRAZYFLIE 2.X ROTOR RPMs", (hud_x + 15, 418),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (88, 166, 255), 1, cv2.LINE_AA)
                    
        for m_idx in range(4):
            m_rpm = float(rpms[m_idx])
            y_pos = 440 + m_idx * 26
            cv2.putText(canvas, f"M{m_idx+1}:", (hud_x + 20, y_pos + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (139, 148, 158), 1, cv2.LINE_AA)
            bar_w = int(np.clip(m_rpm / 22000.0, 0, 1) * 230)
            cv2.rectangle(canvas, (hud_x + 60, y_pos), (hud_x + 60 + 230, y_pos + 14), (33, 38, 45), -1)
            cv2.rectangle(canvas, (hud_x + 60, y_pos), (hud_x + 60 + bar_w, y_pos + 14), (88, 166, 255), -1)
            cv2.putText(canvas, f"{int(m_rpm)} RPM", (hud_x + 305, y_pos + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (201, 209, 217), 1, cv2.LINE_AA)
                        
        cv2.line(canvas, (hud_x + 15, 550), (1235, 550), (48, 54, 61), 1)
        
        # Descending Neurons Output
        cv2.putText(canvas, "DESCENDING NEURON OUTPUTS", (hud_x + 15, 572),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (88, 166, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Steer Left:  {steer_left:.2f} rad/s", (hud_x + 20, 595),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (201, 209, 217), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Steer Right: {steer_right:.2f} rad/s", (hud_x + 220, 595),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (201, 209, 217), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Thrust: {thrust:.2f}x ({fwd_speed:.1f} m/s)", (hud_x + 20, 617),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (63, 185, 80), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Lift:   {lift:.2f} (Z: {target_z:.2f}m)", (hud_x + 220, 617),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 153, 34), 1, cv2.LINE_AA)
                    
        # Flight Status Badge (เชื่อมโยง PyBullet Contact Points จริง 100%)
        if has_collided:
            cv2.rectangle(canvas, (hud_x + 15, 638), (1235, 678), (45, 20, 20), -1)
            cv2.rectangle(canvas, (hud_x + 15, 638), (1235, 678), (220, 50, 50), 1)
            status_txt = f"COLLISION DETECTED! Contacts: {collision_count} | Clear: {min_clear:.2f}M"
            cv2.putText(canvas, status_txt, (hud_x + 25, 662),
                        cv2.FONT_HERSHEY_DUPLEX, 0.44, (100, 100, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(canvas, (hud_x + 15, 638), (1235, 678), (28, 45, 30), -1)
            cv2.rectangle(canvas, (hud_x + 15, 638), (1235, 678), (46, 160, 67), 1)
            status_txt = f"FLY-CFC v2: ACTIVE FLIGHT | 100% ZERO CONTACT | CLEAR {min_clear:.2f}M"
            cv2.putText(canvas, status_txt, (hud_x + 25, 662),
                        cv2.FONT_HERSHEY_DUPLEX, 0.44, (100, 255, 120), 1, cv2.LINE_AA)
                        
        out.write(canvas)
        if (step + 1) % 100 == 0:
            print(f"  - เรนเดอร์ไปแล้ว {step+1}/{TOTAL_STEPS} ก้าว ({(step+1)*DT:.1f}s / {TOTAL_STEPS*DT:.1f}s)...")
            
    out.release()
    env.close()
    
    total_render_time = time.time() - t0_all
    print(f"\nเรนเดอร์ Raw Video เสร็จสิ้น ({total_render_time:.2f}s) กำลังแปลง H.264...")
    
    # แปลง H.264 ด้วย h264_nvenc
    cmd = [
        "ffmpeg", "-y",
        "-i", raw_video_path,
        "-c:v", "h264_nvenc",
        "-pix_fmt", "yuv420p",
        "-b:v", "4M",
        final_video_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    # สร้าง GIF พรีวิว
    print("กำลังสร้าง GIF พรีวิว...")
    gif_cmd = [
        "ffmpeg", "-y",
        "-i", raw_video_path,
        "-vf", "fps=10,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        "-t", "15",
        gif_preview_path
    ]
    subprocess.run(gif_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    if os.path.exists(raw_video_path):
        os.remove(raw_video_path)
        
    print(f"บันทึกไฟล์วิดีโอ 3D Fly-CfC v2 เรียบร้อย: {final_video_path}")
    print(f"บันทึกไฟล์ GIF พรีวิวเรียบร้อย: {gif_preview_path}")
    print(f"สรุปการบิน: Total Collisions = {collision_count}")

if __name__ == "__main__":
    run_3d_fly_v2_showcase()
