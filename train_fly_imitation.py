import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import math

from retinal_projector import RetinalProjector
from fly_wiring import FlyVisuomotorWiring, FlyCfC, load_motor_neuron_indices
from flight_arena import FlightArena

class FlyBrainEndToEnd(nn.Module):
    """
    End-to-End Drosophila Connectome Engine:
    Retinal Projector (128 -> active VPNs) + FlyCfC (VPNs -> 1,961 Internal -> 4 Motor Outputs via DNs).
    """
    def __init__(self, wiring):
        super().__init__()
        self.projector = RetinalProjector()
        self.cfc = FlyCfC(wiring)

    def forward(self, raw_visual_seq, timespans=None, hx=None):
        vpn_seq = self.projector(raw_visual_seq) # [batch, seq_len, 4581]
        motor_seq, hx = self.cfc(vpn_seq, timespans=timespans, hx=hx)
        return motor_seq, hx

def collect_expert_dataset(num_episodes=100, seq_len=30, seed=42):
    """
    Collect flight trajectories driven by the biological potential field expert.
    Guarantees non-zero real continuous looming signals.
    """
    print(f"--- กำลังสร้างชุดข้อมูลการบินจำลอง (Expert Flight Trajectories: {num_episodes} Episodes) ---")
    arena = FlightArena(size=100.0, num_obstacles=25, seed=seed)
    rng = np.random.RandomState(seed)
    
    all_visuals = []
    all_timespans = []
    all_expert_actions = []
    
    for ep in range(num_episodes):
        obs_128, norm_prox = arena.reset(seed=seed + ep)
        ep_vis = []
        ep_dt = []
        ep_act = []
        
        # dt ที่เก็บที่ index t คือช่วงเวลาที่ "นำหน้า" observation นั้น (obs_{t-1} -> obs_t)
        # เพื่อให้ FlyCfC ที่อัปเดต H_{t-1} -> H_t ด้วย timespans[:, t] ใช้ dt ได้ตรงช่วงเวลา
        prev_dt = 0.0
        for step in range(seq_len):
            dt = float(rng.uniform(0.015, 0.045))
            expert_act = arena.get_expert_action(norm_prox)
            
            ep_vis.append(obs_128)
            ep_dt.append(prev_dt)
            ep_act.append(expert_act)
            
            # Step in simulation to update position & compute real continuous looming for next step
            obs_128, norm_prox, collided = arena.step(expert_act, dt)
            prev_dt = dt
            
        all_visuals.append(ep_vis)
        all_timespans.append(ep_dt)
        all_expert_actions.append(ep_act)
        
    X = torch.tensor(np.array(all_visuals), dtype=torch.float32)
    T = torch.tensor(np.array(all_timespans), dtype=torch.float32)
    Y = torch.tensor(np.array(all_expert_actions), dtype=torch.float32)
    
    # ยืนยันว่า Looming Channel (Channels 32..63 และ 96..127) มีค่าจริง ไม่ใช่ 0
    left_looming_max = float(X[:, :, 32:64].max())
    right_looming_max = float(X[:, :, 96:128].max())
    print(f"สร้างชุดข้อมูลสำเร็จ: X={list(X.shape)}, T={list(T.shape)}, Y={list(Y.shape)}")
    print(f"  - ตรวจสอบ Looming Channel สูงสุด: Left={left_looming_max:.4f}, Right={right_looming_max:.4f}")
    if not (left_looming_max > 0.0 and right_looming_max > 0.0):
        raise RuntimeError("Looming channels are all zero - dataset collection is broken!")

    
    return X, T, Y

def train_fly_brain():
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ใช้อุปกรณ์ประมวลผล: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # 1. โหลด Adjacency Matrices
    sensory_adj = np.load("male_cns_connectome/sensory_adj.npy")
    internal_adj = np.load("male_cns_connectome/internal_adj.npy")
    
    wiring = FlyVisuomotorWiring(
        sensory_adjacency_matrix=sensory_adj,
        adjacency_matrix=internal_adj,
        output_dim=4,
        motor_indices=load_motor_neuron_indices(),
    )
    
    # 2. สร้างโมเดล End-to-End
    model = FlyBrainEndToEnd(wiring).to(device)
    
    # 3. เตรียมข้อมูล
    TOTAL_EPISODES = 120
    X, T_span, Y = collect_expert_dataset(num_episodes=TOTAL_EPISODES, seq_len=30, seed=SEED)
    
    # Train / Val split ตามอัตราส่วน (80 / 20)
    split = int(0.80 * TOTAL_EPISODES)
    train_X, train_T, train_Y = X[:split].to(device), T_span[:split].to(device), Y[:split].to(device)
    val_X, val_T, val_Y = X[split:].to(device), T_span[split:].to(device), Y[split:].to(device)
    
    # 4. Optimizer และ Loss Setup
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
    mse_criterion = nn.MSELoss()
    
    # ควบคุม Smoothness Penalty ในระดับที่เหมาะสม (~5% ของ MSE) ไม่กลบ Loss หลัก
    LAMBDA_SMOOTH = 0.0005
    EPOCHS = 25
    BATCH_SIZE = 16
    
    print("\n====================================================================")
    print("START CONTINUOUS-TIME FLY-CFC IMITATION LEARNING (FIXED PIPELINE)")
    print("====================================================================")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        permutation = torch.randperm(train_X.size(0))
        total_loss = 0.0
        total_mse = 0.0
        total_smooth = 0.0
        batches = 0
        
        t0 = time.time()
        for i in range(0, train_X.size(0), BATCH_SIZE):
            indices = permutation[i:i + BATCH_SIZE]
            batch_x = train_X[indices]
            batch_t = train_T[indices]
            batch_y = train_Y[indices]
            
            optimizer.zero_grad()
            
            pred_y, _ = model(batch_x, timespans=batch_t)
            
            # 1. Imitation MSE Loss
            loss_mse = mse_criterion(pred_y, batch_y)
            
            # 2. Time-Aware Action Smoothness Regularization: ||(u_{t+1} - u_t) / dt_t||^2
            #    dt_t (index t+1 ใน dataset ที่ shift แล้ว) คือช่วง obs_t -> obs_{t+1}
            delta_u = pred_y[:, 1:] - pred_y[:, :-1]
            dt_step = batch_t[:, 1:].unsqueeze(-1)
            u_velocity = delta_u / torch.clamp(dt_step, min=1e-3)
            loss_smooth = torch.mean(u_velocity ** 2)
            
            loss = loss_mse + LAMBDA_SMOOTH * loss_smooth
            loss.backward()
            
            # Gradient clipping รักษาเสถียรภาพ Liquid Continuous Cell
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += loss_mse.item()
            total_smooth += loss_smooth.item()
            batches += 1
            
        scheduler.step()
        epoch_time = time.time() - t0
        
        # Validation Evaluation
        model.eval()
        with torch.no_grad():
            val_pred, _ = model(val_X, timespans=val_T)
            val_loss = mse_criterion(val_pred, val_Y).item()
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{EPOCHS} | Train Loss: {total_loss/batches:.4f} (MSE: {total_mse/batches:.4f}, Smooth: {total_smooth/batches:.2f}) | Val MSE: {val_loss:.4f} | Time: {epoch_time:.2f}s")

    # บันทึก Checkpoint โมเดล Fly-CfC v1.0 Final Release Candidate
    os.makedirs("models", exist_ok=True)
    save_path = "models/fly_cfc_v1_final.pth"
    ckpt_dict = {
        'model_state': model.state_dict(),
        'wiring_sensory_shape': sensory_adj.shape,
        'wiring_internal_shape': internal_adj.shape,
        'val_mse': val_loss
    }
    torch.save(ckpt_dict, save_path)
    # อัปเดตไปยัง fly_cfc_champion.pth ด้วยเพื่อความเข้ากันได้ของ downstream scripts
    torch.save(ckpt_dict, "models/fly_cfc_champion.pth")
    print(f"\nบันทึกโมเดล Fly-CfC v1.0 Final Checkpoint สำเร็จที่: {save_path} และ models/fly_cfc_champion.pth")
    
    # 5. ทดสอบ Closed-Loop Autonomous Flight ใน Arena พร้อมเมทริกซ์การเคลื่อนที่จริง
    print("\n====================================================================")
    print("CLOSED-LOOP AUTONOMOUS FLIGHT EVALUATION (RIGOROUS METRICS)")
    print("====================================================================")
    
    eval_arena = FlightArena(size=100.0, num_obstacles=30, seed=999)
    model.eval()
    
    successful_episodes = 0
    total_eval_episodes = 5
    eval_steps = 150
    
    for ep in range(total_eval_episodes):
        obs_128, norm_prox = eval_arena.reset(seed=999 + ep)
        hx = None
        survived_steps = 0
        collided = False
        min_clearance = min(
            min([math.hypot(eval_arena.x - ox, eval_arena.y - oy) - r for ox, oy, r in eval_arena.obstacles], default=999.0),
            eval_arena.x, eval_arena.size - eval_arena.x,
            eval_arena.y, eval_arena.size - eval_arena.y,
        )
        
        t_infer_total = 0.0
        for s in range(eval_steps):
            dt = 0.03 # 33.3 FPS
            x_tensor = torch.tensor(obs_128, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            t_tensor = torch.tensor([[dt]], dtype=torch.float32, device=device)
            
            t_inf_start = time.time()
            with torch.no_grad():
                motor_out, hx = model(x_tensor, timespans=t_tensor, hx=hx)
            t_infer_total += (time.time() - t_inf_start)
            
            action = motor_out.squeeze().cpu().numpy()
            obs_128, norm_prox, collided = eval_arena.step(action, dt)
            
            # ตรวจสอบระยะเคลียร์ต่ำสุด (สิ่งกีดขวาง + ขอบสนาม)
            d_min = min(
                min([math.hypot(eval_arena.x - ox, eval_arena.y - oy) - r for ox, oy, r in eval_arena.obstacles], default=999.0),
                eval_arena.x, eval_arena.size - eval_arena.x,
                eval_arena.y, eval_arena.size - eval_arena.y,
            )
            if d_min < min_clearance:
                min_clearance = d_min
                
            if collided:
                break
            survived_steps += 1
            
        avg_lat = (t_infer_total / max(1, survived_steps)) * 1000.0
        dist = eval_arena.distance_traveled
        avg_speed = dist / max(1e-3, survived_steps * 0.03)
        
        # รอดชีวิตจริง = ไม่ชน และบินเคลื่อนที่จริง (ไม่ใช่ลอยนิ่ง)
        passed = (not collided) and (dist >= 15.0) and (avg_speed >= 2.0)
        status = "PASSED" if passed else f"FAILED (collided={collided}, dist={dist:.1f}m)"
        print(f"Eval Episode {ep+1}: {status} | Steps: {survived_steps}/{eval_steps} | Distance: {dist:.1f} m (Speed: {avg_speed:.2f} m/s) | Min Clear: {min_clearance:.2f} m | Latency: {avg_lat:.2f} ms")
        if passed:
            successful_episodes += 1
            
    print(f"\nสรุปผลการทดสอบการบินอัตโนมัติ: สำเร็จ {successful_episodes}/{total_eval_episodes} Episodes ({(successful_episodes/total_eval_episodes)*100:.1f}%)")
    print("====================================================================")

if __name__ == "__main__":
    train_fly_brain()
