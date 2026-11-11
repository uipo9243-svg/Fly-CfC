import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time
import math
import os

from flight_arena import FlightArena
from fly_wiring import FlyVisuomotorWiring, load_motor_neuron_indices
from train_fly_imitation import FlyBrainEndToEnd

def run_evaluation_and_visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"กำลังประเมินผลบนอุปกรณ์: {device}")
    
    # 1. โหลด Wiring และโมเดล
    sensory_adj = np.load("male_cns_connectome/sensory_adj.npy")
    internal_adj = np.load("male_cns_connectome/internal_adj.npy")
    n_sensory = sensory_adj.shape[0]
    
    wiring = FlyVisuomotorWiring(
        sensory_adjacency_matrix=sensory_adj,
        adjacency_matrix=internal_adj,
        output_dim=4,
        motor_indices=load_motor_neuron_indices(),
    )
    
    model = FlyBrainEndToEnd(wiring).to(device)
    checkpoint_path = "models/fly_cfc_v1_final.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "models/fly_cfc_champion.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "models/fly_cfc_v1_final_fp16.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    sd = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
    model_sd = model.state_dict()
    converted_sd = {}
    for k, v in sd.items():
        if k in model_sd:
            converted_sd[k] = v.to(model_sd[k].dtype)
        else:
            converted_sd[k] = v
    model.load_state_dict(converted_sd)
    model.eval()
    print(f"โหลด Checkpoint สำเร็จ ({checkpoint_path}, Validation MSE เดิม: {checkpoint.get('val_mse', 0.0):.4f})")

    
    # 2. ตั้งค่า Arena สำหรับ Visualization: ตรึงชุดสิ่งกีดขวางเดียวกันตลอดการทดสอบ (Fixed Benchmark Arena)
    ARENA_SEED = 777
    arena = FlightArena(size=100.0, num_obstacles=30, seed=ARENA_SEED)
    
    NUM_EPISODES = 5
    MAX_STEPS = 150
    DT = 0.03 # 33.3 Hz Continuous update
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1.2, 1]})
    ax_arena = axes[0]
    ax_telemetry = axes[1]
    
    # วาดขอบ Arena
    ax_arena.set_xlim(0, 100)
    ax_arena.set_ylim(0, 100)
    ax_arena.set_facecolor('#0d1117')
    ax_arena.grid(True, color='#21262d', linestyle='--', alpha=0.7)
    
    # วาดสิ่งกีดขวางที่ใช้จริงในการทดสอบ (ไม่มีการสร้างใหม่ระหว่าง episode)
    for ox, oy, r in arena.obstacles:
        circle = Circle((ox, oy), r, color='#f85149', alpha=0.6, ec='#ff7b72', lw=1.5)
        ax_arena.add_patch(circle)
        
    colors = ['#58a6ff', '#3fb950', '#d29922', '#bc8cff', '#f0883e']
    episode_stats = []
    
    # กำหนดตำแหน่งเกิดที่หลากหลายในสภาพแวดล้อมสิ่งกีดขวางเดียวกัน
    spawn_rng = np.random.RandomState(42)
    
    for ep in range(NUM_EPISODES):
        # สุ่มจุดเกิดใหม่โดยคงสิ่งกีดขวางชุดเดิมไว้ 100%
        for _ in range(300):
            spawn_x = float(spawn_rng.uniform(20.0, arena.size - 20.0))
            spawn_y = float(spawn_rng.uniform(20.0, arena.size - 20.0))
            clearance = min([math.hypot(spawn_x - ox, spawn_y - oy) - r for ox, oy, r in arena.obstacles], default=999.0)
            if clearance > 6.0:
                break
        spawn_heading = float(spawn_rng.uniform(-np.pi, np.pi))
        obs_128, norm_prox = arena.reset_at(spawn_x, spawn_y, spawn_heading)
        
        hx = None

        traj_x = [arena.x]
        traj_y = [arena.y]
        steer_signals = []
        min_obstacle_dists = []
        
        t_infer_total = 0.0
        collided = False
        
        for step in range(MAX_STEPS):
            # ตรวจสอบระยะห่างจากสิ่งกีดขวางและขอบสนามที่ใกล้ที่สุด
            d_min = min(
                min([math.hypot(arena.x - ox, arena.y - oy) - r for ox, oy, r in arena.obstacles], default=999.0),
                arena.x, arena.size - arena.x,
                arena.y, arena.size - arena.y,
            )
            min_obstacle_dists.append(d_min)
            
            x_t = torch.tensor(obs_128, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            ts_t = torch.tensor([[DT]], dtype=torch.float32, device=device)
            
            t0 = time.time()
            with torch.no_grad():
                motor_out, hx = model(x_t, timespans=ts_t, hx=hx)
            t_infer_total += (time.time() - t0)
            
            action = motor_out.squeeze().cpu().numpy()
            steer_val = float(action[0] - action[1])
            steer_signals.append(steer_val)
            
            obs_128, norm_prox, collided = arena.step(action, DT)
            traj_x.append(arena.x)
            traj_y.append(arena.y)
            
            if collided:
                break
                
        survived = len(traj_x) - 1
        avg_lat = (t_infer_total / max(1, survived)) * 1000.0
        dist = arena.distance_traveled
        avg_speed = dist / max(1e-3, survived * DT)
        c = colors[ep % len(colors)]
        
        # พล็อตวิถีการบิน
        status_label = 'Safe' if not collided else 'Hit'
        ax_arena.plot(traj_x, traj_y, color=c, lw=2.2, label=f"Fly {ep+1} ({status_label}: {dist:.1f}m, {survived} steps)")
        ax_arena.scatter([traj_x[0]], [traj_y[0]], color=c, marker='o', s=60, edgecolors='white', zorder=5)
        ax_arena.scatter([traj_x[-1]], [traj_y[-1]], color='#f85149' if collided else c, marker='x' if collided else '^', s=80, zorder=5)
        
        # พล็อตสัญญาณการเลี้ยวครบทุก Steps (MAX_STEPS = 150)
        ax_telemetry.plot(steer_signals, color=c, alpha=0.85, lw=1.5, label=f"Fly {ep+1} Steering")
        
        episode_stats.append({
            'episode': ep + 1,
            'survived': survived,
            'collided': collided,
            'dist': dist,
            'speed': avg_speed,
            'min_dist': min(min_obstacle_dists) if min_obstacle_dists else 0.0,
            'latency_ms': avg_lat
        })
        
    ax_arena.set_title(f"Drosophila Connectome Fly-CfC: 2D Autonomous Flight Arena\n({n_sensory:,} Active Visual Inputs -> 1,961 Connectome Units -> 4 DN-Gated Motors | Fixed 30 Obstacles)", color='white', fontsize=12, pad=12)
    ax_arena.legend(loc='upper right', facecolor='#161b22', edgecolor='#30363d', labelcolor='white', fontsize=9)
    
    ax_telemetry.set_facecolor('#0d1117')
    ax_telemetry.grid(True, color='#21262d', linestyle='--', alpha=0.7)
    ax_telemetry.set_title("Visuomotor Continuous Steering Dynamics (Rad/s)", color='white', fontsize=12, pad=12)
    ax_telemetry.set_xlabel(f"Time Step (dt = {DT*1000:.0f} ms, Total 150 Steps)", color='white')
    ax_telemetry.set_ylabel("Steering Velocity (Turn Left > 0 > Turn Right)", color='white')
    ax_telemetry.tick_params(colors='white')
    ax_telemetry.axhline(0, color='#8b949e', linestyle=':', alpha=0.6)
    ax_telemetry.legend(loc='upper right', facecolor='#161b22', edgecolor='#30363d', labelcolor='white', fontsize=9)
    
    fig.patch.set_facecolor('#010409')
    plt.tight_layout()
    
    output_png = "fly_cfc_flight_arena.png"
    plt.savefig(output_png, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    print("\n--- สรุปผลการบินทดสอบรอบประเมินผล (Telemetry Report) ---")
    safe_count = sum(1 for s in episode_stats if not s['collided'] and s['dist'] >= 15.0 and s['speed'] >= 2.0)
    for s in episode_stats:
        passed = not s['collided'] and s['dist'] >= 15.0 and s['speed'] >= 2.0
        status = "PASSED" if passed else "FAILED"

        print(f"Episode {s['episode']}: {status} | Steps: {s['survived']}/{MAX_STEPS} | Dist: {s['dist']:.1f} m (Speed: {s['speed']:.2f} m/s) | Min Clear: {s['min_dist']:.2f} m | Latency: {s['latency_ms']:.2f} ms")
        
    print(f"\nความสำเร็จโดยรวม: ผ่านการทดสอบ {safe_count}/{NUM_EPISODES} Episodes ({(safe_count/NUM_EPISODES)*100:.1f}%)")
    print(f"บันทึกแผนภาพวิถีการบินที่: {output_png}")

if __name__ == "__main__":
    run_evaluation_and_visualize()
