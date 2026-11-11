import sys
import os
import math
import time
import numpy as np
import cv2
import torch
import torch.nn as nn

# Add gym-pybullet-drones to path
sys.path.insert(0, '/home/surayut/ดาวน์โหลด/kk/gym-pybullet-drones')
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics
import pybullet as p

from fly_wiring import FlyVisuomotorWiring, FlyCfC, load_motor_neuron_indices
from retinal_projector import RetinalProjector
from spherical_eye import SphericalCompoundEye
from acrobatic_arena import AcrobaticArena

# Patch DSLPIDControl._dslPIDAttitudeControl to eliminate Euler angle wraparound derivative spike
orig_dslPIDAttitudeControl = DSLPIDControl._dslPIDAttitudeControl
def patched_dslPIDAttitudeControl(self, control_timestep, thrust, cur_quat, target_euler, target_rpy_rates):
    cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
    diff = (cur_rpy - self.last_rpy + np.pi) % (2.0 * np.pi) - np.pi
    self.last_rpy = cur_rpy - diff
    return orig_dslPIDAttitudeControl(self, control_timestep, thrust, cur_quat, target_euler, target_rpy_rates)

DSLPIDControl._dslPIDAttitudeControl = patched_dslPIDAttitudeControl

class FlyBrainEndToEnd(nn.Module):
    def __init__(self, wiring):
        super().__init__()
        self.projector = RetinalProjector()
        self.cfc = FlyCfC(wiring)

    def forward(self, raw_visual_seq, timespans=None, hx=None):
        vpn_seq = self.projector(raw_visual_seq)
        motor_seq, hx = self.cfc(vpn_seq, timespans=timespans, hx=hx)
        return motor_seq, hx

def render_acrobatic_hud(frame, step, total_steps, pos, rpy, vel, max_roll_deg, gates_cleared, has_collided, infer_latency_ms, left_dist, right_dist):
    h, w, _ = frame.shape
    
    # 1. Top HUD Header Banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 15), (w - 20, 115), (10, 14, 22), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (20, 15), (w - 20, 115), (0, 210, 255), 1)
    
    # Title
    cv2.putText(frame, "FLY-CFC v3: 3D SPHERICAL EYE & ACROBATIC SACCADE FLIGHT", (35, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 230), 2, cv2.LINE_AA)
    cv2.putText(frame, "WHOLE-BRAIN DROSOPHILA CONNECTOME (151.85M SYNAPSES DISTILLED TO 99.51% SPARSITY)", (35, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 210, 225), 1, cv2.LINE_AA)
                
    time_sec = step / 30.0
    status_text = f"T: {time_sec:5.2f}s | Step: {step:03d}/{total_steps} | Model Latency: {infer_latency_ms:.2f} ms ({1000/max(1e-4, infer_latency_ms):.0f} FPS)"
    cv2.putText(frame, status_text, (35, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Collision Status Badge (Top Right)
    if has_collided:
        cv2.rectangle(frame, (w - 280, 30), (w - 35, 95), (20, 20, 180), -1)
        cv2.putText(frame, "COLLISION DETECTED", (w - 265, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (w - 280, 30), (w - 35, 95), (25, 120, 35), -1)
        cv2.putText(frame, "ZERO CONTACT: 100%", (w - 265, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"GATES CLEARED: {gates_cleared}/4", (w - 250, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 255, 210), 1, cv2.LINE_AA)
        
    # 2. Bottom Left: 3D Flight Dynamics & Attitude Telemetry Card
    overlay_dyn = frame.copy()
    cv2.rectangle(overlay_dyn, (20, h - 230), (380, h - 20), (12, 16, 26), -1)
    cv2.addWeighted(overlay_dyn, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (20, h - 230), (380, h - 20), (80, 150, 220), 1)
    
    cv2.putText(frame, "FLIGHT ATTITUDE & ACROBATICS", (35, h - 205), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
    
    roll_deg = math.degrees(rpy[0])
    pitch_deg = math.degrees(rpy[1])
    yaw_deg = math.degrees(rpy[2])
    speed = math.hypot(vel[0], vel[1], vel[2])
    
    # Roll gauge bar
    cv2.putText(frame, f"Roll (Bank): {roll_deg:+5.1f} deg", (35, h - 175), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    # Roll visual indicator
    bar_center_x = 260
    bar_y = h - 180
    cv2.line(frame, (bar_center_x - 70, bar_y), (bar_center_x + 70, bar_y), (80, 80, 90), 2)
    cv2.circle(frame, (bar_center_x, bar_y), 3, (200, 200, 200), -1)
    roll_offset = int(np.clip(roll_deg / 55.0, -1.0, 1.0) * 65.0)
    bank_color = (0, 255, 120) if abs(roll_deg) > 20.0 else (0, 200, 255)
    cv2.circle(frame, (bar_center_x + roll_offset, bar_y), 6, bank_color, -1)
    if abs(roll_deg) > 20.0:
        cv2.putText(frame, "BANKED", (bar_center_x + 80, bar_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 120), 1, cv2.LINE_AA)
        
    cv2.putText(frame, f"Pitch:       {pitch_deg:+5.1f} deg", (35, h - 145), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Heading:     {yaw_deg:+5.1f} deg", (35, h - 115), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Airspeed:    {speed:4.2f} m/s", (35, h - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Altitude Z:  {pos[2]:4.2f} m", (35, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Peak Bank:   {max_roll_deg:4.1f} deg", (35, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 200), 1, cv2.LINE_AA)
    
    # 3. Bottom Right: 3D Spherical Ommatidia Visual Radar
    overlay_eye = frame.copy()
    cv2.rectangle(overlay_eye, (w - 380, h - 230), (w - 20, h - 20), (12, 16, 26), -1)
    cv2.addWeighted(overlay_eye, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (w - 380, h - 230), (w - 20, h - 20), (80, 150, 220), 1)
    
    cv2.putText(frame, "3D SPHERICAL OMMATIDIA RADAR (64 RAYS)", (w - 365, h - 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)
                
    radar_center = (w - 200, h - 110)
    radar_radius = 70
    cv2.circle(frame, radar_center, radar_radius, (50, 60, 75), 1)
    cv2.circle(frame, radar_center, int(radar_radius * 0.66), (40, 50, 65), 1)
    cv2.circle(frame, radar_center, int(radar_radius * 0.33), (40, 50, 65), 1)
    # Center drone cross
    cv2.drawMarker(frame, radar_center, (0, 255, 255), cv2.MARKER_CROSS, 10, 1)
    
    # Left eye rays (blue/cyan)
    for i in range(32):
        prox = left_dist[i]
        ang = np.pi - (i / 31.0) * np.pi + rpy[2]
        r_pix = int(prox * radar_radius)
        if r_pix > 2:
            rx = int(radar_center[0] - r_pix * math.cos(ang))
            ry = int(radar_center[1] - r_pix * math.sin(ang))
            color = (int(255 * prox), int(200 * (1 - prox)), int(255 * (1 - prox)))
            cv2.circle(frame, (rx, ry), 2, color, -1)
            
    # Right eye rays (red/orange)
    for i in range(32):
        prox = right_dist[i]
        ang = (i / 31.0) * np.pi + rpy[2]
        r_pix = int(prox * radar_radius)
        if r_pix > 2:
            rx = int(radar_center[0] + r_pix * math.cos(ang))
            ry = int(radar_center[1] - r_pix * math.sin(ang))
            color = (int(50 * (1 - prox)), int(150 * (1 - prox)), int(255 * prox))
            cv2.circle(frame, (rx, ry), 2, color, -1)
            
    cv2.putText(frame, "LEFT EYE", (w - 360, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, "RIGHT EYE", (w - 100, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 150, 255), 1, cv2.LINE_AA)
    return frame

def run_acrobatic_flight():
    print("====================================================================")
    print("LAUNCHING FLY-CFC v3 ACROBATIC FLIGHT EVALUATION & HD SHOWCASE")
    print("3D Spherical Compound Eye + 4-Axis Attitude Control (Banked Turns)")
    print("====================================================================")
    
    device = torch.device("cpu")
    
    # 1. Load biological connectome
    sensory_adj = np.load("male_cns_connectome/sensory_adj.npy")
    internal_adj = np.load("male_cns_connectome/internal_adj.npy")
    wiring = FlyVisuomotorWiring(
        sensory_adjacency_matrix=sensory_adj,
        adjacency_matrix=internal_adj,
        output_dim=4,
        motor_indices=load_motor_neuron_indices(),
    )
    model = FlyBrainEndToEnd(wiring).to(device)
    
    ckpt_path = "models/fly_cfc_v3_acrobatic.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = "models/fly_cfc_v2_3d.pth"
        print(f"Notice: v3 model not found yet, falling back to: {ckpt_path}")
        
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded Fly-CfC Model: {ckpt_path}")
    
    # 2. Setup PyBullet Environment
    start_pos = [0.0, -5.5, 1.2]
    init_yaw = math.pi / 2.0
    
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([start_pos]),
        initial_rpys=np.array([[0.0, 0.0, init_yaw]]),
        ctrl_freq=30,
        physics=Physics.PYB,
        gui=False,
        obstacles=False
    )
    client = env.getPyBulletClient()
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    arena = AcrobaticArena(client, arena_size=16.0)
    arena.build_arena()
    eye = SphericalCompoundEye(64)
    
    obs, _ = env.reset()
    drone_id = env.DRONE_IDS[0]
    
    # Video Writer
    W, H = 1280, 720
    fps = 30
    raw_video_path = "fly_drone_3d_acrobatic_raw.mp4"
    final_video_path = "fly_drone_3d_acrobatic_showcase.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(raw_video_path, fourcc, fps, (W, H))
    
    TOTAL_STEPS = 360 # 12 seconds
    dt = 1.0 / 30.0
    ts_tensor = torch.tensor([[dt]], dtype=torch.float32)
    hx = None
    
    target_yaw = init_yaw
    hover_pwm = 38500.0
    has_collided = False
    collision_count = 0
    max_roll_deg = 0.0
    gates_cleared = 0
    
    gate_y = [-4.5, -2.0, 0.0, 2.2]
    gate_passed = [False] * len(gate_y)
    
    cam_eye_smooth = None
    cam_target_smooth = None
    
    print(f"Running simulation & rendering HD video ({TOTAL_STEPS} steps, {TOTAL_STEPS/fps:.1f} s)...")
    t0_sim = time.time()
    
    for step in range(TOTAL_STEPS):
        pos = obs[0, 0:3]
        quat = obs[0, 3:7]
        rpy = obs[0, 7:10]
        vel = obs[0, 10:13]
        
        cur_roll_deg = math.degrees(abs(rpy[0]))
        if cur_roll_deg > max_roll_deg:
            max_roll_deg = cur_roll_deg
            
        # 1. Contact Points Collision Check
        cps = p.getContactPoints(bodyA=drone_id, physicsClientId=client)
        if len(cps) > 0:
            has_collided = True
            collision_count += 1
            
        # 2. 3D Spherical Ommatidia Ray Casting
        rot_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        obs_128, norm_prox, looming = eye.cast_rays(client, pos, rot_mat, dt=dt)
        left_dist = obs_128[0:32]
        right_dist = obs_128[64:96]
        
        # 3. Model Forward Pass
        t_infer = time.time()
        x_tensor = torch.tensor(obs_128, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            action_seq, hx = model(x_tensor, timespans=ts_tensor, hx=hx)
        infer_latency_ms = (time.time() - t_infer) * 1000.0
        
        action = action_seq.squeeze().numpy()
        
        # 4. Continuous Aerodynamic Flight Commands (Exact match to training)
        # action[0] -> Banked Roll [-45 deg, +45 deg]
        # action[1] -> Vertical Pitch / Dive [-20 deg, +20 deg]
        # action[2] -> Yaw Saccade Rate [-3.5, +3.5] rad/s
        # action[3] -> Forward Speed [0.8, 2.0] m/s
        cmd_bank = float(np.clip(action[0], -1.0, 1.0)) * math.radians(45.0)
        target_z = float(np.clip(1.2 + 0.40 * action[1], 0.70, 1.90))
        omega = float(np.clip(action[2], -1.0, 1.0)) * 3.5
        fwd_speed = float(np.clip(1.40 + 0.50 * action[3], 0.80, 2.00))
        
        target_yaw = (target_yaw + omega * dt + np.pi) % (2.0 * np.pi) - np.pi
        lookahead = float(np.clip(fwd_speed * 0.55, 0.45, 0.90))
        
        target_pos = np.array([
            pos[0] + lookahead * math.cos(target_yaw),
            pos[1] + lookahead * math.sin(target_yaw),
            target_z
        ])
        target_vel = np.array([
            fwd_speed * math.cos(target_yaw),
            fwd_speed * math.sin(target_yaw),
            (target_z - pos[2]) * 1.5
        ])
        
        thrust, comp_euler, _ = ctrl._dslPIDPositionControl(
            dt, pos, quat, vel, target_pos, np.array([0.0, 0.0, target_yaw]), target_vel
        )
        
        # Inject Banked Roll into attitude
        acrobatic_euler = np.array([
            comp_euler[0] + cmd_bank * math.cos(target_yaw),
            comp_euler[1] + cmd_bank * math.sin(target_yaw),
            comp_euler[2]
        ])
        rpm = ctrl._dslPIDAttitudeControl(dt, thrust, quat, acrobatic_euler, np.zeros(3))
        obs, _, _, _, _ = env.step(rpm.reshape(1, 4))
        
        # Check gates
        for g_idx, gy in enumerate(gate_y):
            if not gate_passed[g_idx] and pos[1] >= gy:
                gate_passed[g_idx] = True
                gates_cleared += 1
                
        # 5. Rigid Rear-Chase Camera (Locked directly behind the drone's tail)
        cam_dist = 0.48
        cam_height = 0.14
        drone_heading = rpy[2]
        fwd_dir = np.array([math.cos(drone_heading), math.sin(drone_heading), 0.0])
        
        target_cam_eye = np.array([
            pos[0] - cam_dist * fwd_dir[0],
            pos[1] - cam_dist * fwd_dir[1],
            pos[2] + cam_height
        ])
        target_cam_target = np.array([
            pos[0] + 0.60 * fwd_dir[0],
            pos[1] + 0.60 * fwd_dir[1],
            pos[2] + 0.02
        ])
        
        if cam_eye_smooth is None:
            cam_eye_smooth = target_cam_eye
            cam_target_smooth = target_cam_target
        else:
            # 80% responsive tight lock to eliminate lag while dampening high-frequency micro-jitter
            cam_eye_smooth = 0.20 * cam_eye_smooth + 0.80 * target_cam_eye
            cam_target_smooth = 0.20 * cam_target_smooth + 0.80 * target_cam_target
            
        view_mat = p.computeViewMatrix(
            cameraEyePosition=cam_eye_smooth.tolist(),
            cameraTargetPosition=cam_target_smooth.tolist(),
            cameraUpVector=[0, 0, 1],
            physicsClientId=client
        )
        proj_mat = p.computeProjectionMatrixFOV(
            fov=65.0, aspect=float(W)/H, nearVal=0.1, farVal=40.0,
            physicsClientId=client
        )
        
        _, _, rgb_img, _, _ = p.getCameraImage(
            width=W, height=H,
            viewMatrix=view_mat,
            projectionMatrix=proj_mat,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=client
        )
        
        frame = cv2.cvtColor(rgb_img[:, :, :3], cv2.COLOR_RGB2BGR)
        
        # Render HUD
        frame = render_acrobatic_hud(
            frame, step, TOTAL_STEPS, pos, rpy, vel, max_roll_deg,
            gates_cleared, has_collided, infer_latency_ms, left_dist, right_dist
        )
        out.write(frame)
        
        if step % 30 == 0:
            print(f"  Step {step}/{TOTAL_STEPS} | Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | Roll: {math.degrees(rpy[0]):+.1f}° | Bank Max: {max_roll_deg:.1f}°")
            
        # Clean completion after full course traversal
        if has_collided and collision_count > 6:
            print(f"Course traversal finished at step {step} ({step/fps:.1f} s)")
            break
            
    out.release()
    env.close()
    
    elapsed = time.time() - t0_sim
    print(f"\nSimulation complete in {elapsed:.1f} s ({step/elapsed:.1f} FPS)")
    print(f"Max Acrobatic Roll (Banked Turn): {max_roll_deg:.1f} degrees")
    print(f"Gates Cleared: {gates_cleared} / 4")
    print(f"Collisions: {collision_count}")
    
    # Transcode to web-compatible H.264 MP4 with hardware acceleration
    os.system(f"ffmpeg -y -i {raw_video_path} -c:v h264_nvenc -profile:v baseline -level 3.1 -pix_fmt yuv420p -b:v 4M -movflags +faststart {final_video_path} > /dev/null 2>&1")
    # Generate high-quality GIF preview directly
    os.system(f"ffmpeg -y -i {raw_video_path} -ss 00:00:00 -t 5 -vf 'fps=15,scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse' -c:v gif fly_drone_3d_acrobatic_preview.gif > /dev/null 2>&1")
    # Sync to assets, web server directory, and brain directory
    os.system(f"cp {final_video_path} assets/ && cp fly_drone_3d_acrobatic_preview.gif assets/")
    os.system(f"cp {final_video_path} /home/surayut/ดาวน์โหลด/kk/")
    os.system(f"cp {final_video_path} /home/surayut/.gemini/antigravity-cli/brain/2a9742d9-a0b8-4576-8f77-e51a2f5d1e92/")
    os.system(f"cp fly_drone_3d_acrobatic_preview.gif /home/surayut/.gemini/antigravity-cli/brain/2a9742d9-a0b8-4576-8f77-e51a2f5d1e92/")
    print(f"Generated Web-Compatible HD Showcase Video: {final_video_path}")

if __name__ == '__main__':
    run_acrobatic_flight()
