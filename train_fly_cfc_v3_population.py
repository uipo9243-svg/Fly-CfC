import sys
import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import multiprocessing as mp

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

def get_trainable_vector(model):
    """
    Extracts the active parameter vector:
      - Motor Head weights (4 x 1300 DNs = 5,200) and bias (4)
      - Projector gain (1683) and bias (1683)
    """
    params = []
    motor_mask = model.cfc.cell.motor_mask.bool().cpu().numpy() # [4, 1961]
    w_m = model.cfc.cell.motor_linear.weight.data.cpu().numpy()
    params.append(w_m[motor_mask].flatten()) # 5,200
    params.append(model.cfc.cell.motor_linear.bias.data.cpu().numpy().flatten()) # 4
    params.append(model.projector.gain.data.cpu().numpy().flatten()) # 1,683
    params.append(model.projector.bias.data.cpu().numpy().flatten()) # 1,683
    return np.concatenate(params)

def set_trainable_vector(model, vector):
    """
    Injects vector back into active parameters strictly adhering to biological masks.
    """
    idx = 0
    motor_mask = model.cfc.cell.motor_mask.bool().cpu().numpy()
    n_m = int(motor_mask.sum())
    w_m_active = vector[idx:idx+n_m]
    idx += n_m
    
    w_m = model.cfc.cell.motor_linear.weight.data.cpu().numpy()
    w_m[motor_mask] = w_m_active
    model.cfc.cell.motor_linear.weight.data.copy_(torch.from_numpy(w_m))
    
    b_m = vector[idx:idx+4]
    idx += 4
    model.cfc.cell.motor_linear.bias.data.copy_(torch.from_numpy(b_m))
    
    gain = vector[idx:idx+1683]
    idx += 1683
    model.projector.gain.data.copy_(torch.from_numpy(gain))
    
    bias = vector[idx:idx+1683]
    idx += 1683
    model.projector.bias.data.copy_(torch.from_numpy(bias))

# Preloaded model in parent process; inherited cleanly via fork by workers
BASE_MODEL = None

def init_base_model():
    global BASE_MODEL
    torch.set_num_threads(1)
    sensory_adj = np.load("male_cns_connectome/sensory_adj.npy")
    internal_adj = np.load("male_cns_connectome/internal_adj.npy")
    wiring = FlyVisuomotorWiring(
        sensory_adjacency_matrix=sensory_adj,
        adjacency_matrix=internal_adj,
        output_dim=4,
        motor_indices=load_motor_neuron_indices(),
    )
    BASE_MODEL = FlyBrainEndToEnd(wiring)
    base_ckpt = torch.load("models/fly_cfc_v2_3d.pth", map_location="cpu", weights_only=False)
    BASE_MODEL.load_state_dict(base_ckpt["model_state"])
    BASE_MODEL.eval()

def evaluate_individual(args):
    """
    Evaluates one candidate parameter vector in a full 3D Acrobatic Arena simulation.
    """
    global BASE_MODEL
    worker_id, param_vec, max_steps = args
    
    # Inject active parameters into inherited model
    set_trainable_vector(BASE_MODEL, param_vec)
    
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
    
    dt = 1.0 / 30.0
    ts_tensor = torch.tensor([[dt]], dtype=torch.float32)
    hx = None
    
    cmd_yaw = init_yaw
    fitness = 0.0
    collided = False
    max_roll_deg = 0.0
    max_pitch_deg = 0.0
    gates_cleared = 0
    total_bank_score = 0.0
    
    # Gates to pass:
    gate_y = [-4.5, -2.0, 0.0, 2.2]
    gate_passed = [False] * len(gate_y)
    
    for step in range(max_steps):
        obs_raw = env._getDroneStateVector(0)
        pos = obs_raw[0:3]
        quat = obs_raw[3:7]
        rpy = obs_raw[7:10]
        vel = obs_raw[10:13]
        ang_v = obs_raw[13:16]
        
        cur_roll_deg = math.degrees(abs(rpy[0]))
        cur_pitch_deg = math.degrees(abs(rpy[1]))
        if cur_roll_deg > max_roll_deg: max_roll_deg = cur_roll_deg
        if cur_pitch_deg > max_pitch_deg: max_pitch_deg = cur_pitch_deg
        
        # 1. Contact Collision Check
        cps = p.getContactPoints(bodyA=drone_id, physicsClientId=client)
        if len(cps) > 0:
            collided = True
            fitness -= 300.0
            break
            
        # 2. Out of bounds check
        if abs(pos[0]) > 7.0 or abs(pos[1]) > 7.2 or pos[2] < 0.25 or pos[2] > 3.2:
            collided = True
            fitness -= 250.0
            break
            
        # 3. 3D Spherical Ommatidia Ray Casting
        rot_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        obs_128, norm_prox, looming = eye.cast_rays(client, pos, rot_mat, dt=dt)
        
        # 4. Model Forward Pass
        x_tensor = torch.tensor(obs_128, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            action_seq, hx = BASE_MODEL(x_tensor, timespans=ts_tensor, hx=hx)
        action = action_seq.squeeze().numpy()
        
        # 5. Continuous Aerodynamic Flight Commands
        # action[0] -> Banked Roll [-45 deg, +45 deg]
        # action[1] -> Vertical Pitch / Dive [-20 deg, +20 deg]
        # action[2] -> Yaw Saccade Rate [-3.5, +3.5] rad/s
        # action[3] -> Forward Speed [0.8, 2.0] m/s
        cmd_bank = float(np.clip(action[0], -1.0, 1.0)) * math.radians(45.0)
        target_z = float(np.clip(1.2 + 0.40 * action[1], 0.70, 1.90))
        omega = float(np.clip(action[2], -1.0, 1.0)) * 3.5
        fwd_speed = float(np.clip(1.40 + 0.50 * action[3], 0.80, 2.00))
        
        cmd_yaw = (cmd_yaw + omega * dt + np.pi) % (2.0 * np.pi) - np.pi
        lookahead = float(np.clip(fwd_speed * 0.55, 0.45, 0.90))
        
        target_pos = np.array([
            pos[0] + lookahead * math.cos(cmd_yaw),
            pos[1] + lookahead * math.sin(cmd_yaw),
            target_z
        ])
        target_vel = np.array([
            fwd_speed * math.cos(cmd_yaw),
            fwd_speed * math.sin(cmd_yaw),
            (target_z - pos[2]) * 1.5
        ])
        
        thrust, comp_euler, _ = ctrl._dslPIDPositionControl(
            dt, pos, quat, vel, target_pos, np.array([0.0, 0.0, cmd_yaw]), target_vel
        )
        
        # Inject Banked Roll into attitude
        acrobatic_euler = np.array([
            comp_euler[0] + cmd_bank * math.cos(cmd_yaw),
            comp_euler[1] + cmd_bank * math.sin(cmd_yaw),
            comp_euler[2]
        ])
        rpm = ctrl._dslPIDAttitudeControl(dt, thrust, quat, acrobatic_euler, np.zeros(3))
        obs, _, _, _, _ = env.step(rpm.reshape(1, 4))
        
        # 6. Gate passing & Bank alignment rewards
        for g_idx, gy in enumerate(gate_y):
            if not gate_passed[g_idx] and pos[1] >= gy:
                gate_passed[g_idx] = True
                gates_cleared += 1
                fitness += 150.0
                
                # Bank alignment reward for slanted gates
                if g_idx == 1: # Slanted +40 deg
                    bank_align = math.cos(rpy[0] - math.radians(40.0))
                    bonus = max(0.0, bank_align) * 120.0
                    fitness += bonus
                    total_bank_score += bonus
                elif g_idx == 3: # Slanted -40 deg
                    bank_align = math.cos(rpy[0] - math.radians(-40.0))
                    bonus = max(0.0, bank_align) * 120.0
                    fitness += bonus
                    total_bank_score += bonus
                elif g_idx == 2: # Crossbar underpass
                    if 0.6 <= pos[2] <= 1.6:
                        fitness += 90.0 # Underpass dive bonus
                        
        # Forward progress reward
        forward_dist = pos[1] - start_pos[1]
        fitness += 5.0 * forward_dist
        
        # Agility bonus: reward banking into turns
        if cur_roll_deg > 12.0:
            fitness += 0.40 * cur_roll_deg
            
    env.close()
    
    total_dist = float(obs[0, 1] - start_pos[1])
    return {
        'worker_id': worker_id,
        'fitness': float(fitness),
        'collided': collided,
        'distance': total_dist,
        'max_roll_deg': float(max_roll_deg),
        'max_pitch_deg': float(max_pitch_deg),
        'gates_cleared': gates_cleared,
        'bank_score': float(total_bank_score)
    }

def train_population(pop_size=24, num_generations=25, max_steps=240, num_workers=8):
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    print("====================================================================")
    print("STARTING FLY-CFC v3 POPULATION-BASED ACROBATIC TRAINING")
    print(f"Population Size: {pop_size} | Generations: {num_generations} | Parallel Workers: {num_workers}")
    print("Objectives: 3D Spherical Vision + 4-Axis Attitude + Banked Saccade Slalom")
    print("====================================================================")
    
    # 1. Initialize Baseline Model in main process
    init_base_model()
    base_vec = get_trainable_vector(BASE_MODEL)
    dim = len(base_vec)
    print(f"Loaded Base Champion (v2 3D). Active trainable vector size: {dim:,}")
    
    # 2. Initialize Population centered around Base Seed
    population = [np.copy(base_vec)]
    sigma = 0.035
    
    for i in range(1, pop_size):
        noise = np.random.randn(dim).astype(np.float32) * sigma
        population.append(base_vec + noise)
        
    best_fitness_history = []
    best_overall_fitness = -1e9
    champion_vector = np.copy(base_vec)
    champion_telemetry = None
    
    t_start_all = time.time()
    
    with mp.Pool(processes=num_workers) as pool:
        for gen in range(1, num_generations + 1):
            t_gen_start = time.time()
            
            # Prepare task arguments
            tasks = [(i, population[i], max_steps) for i in range(pop_size)]
            
            # Parallel evaluation
            results = pool.map(evaluate_individual, tasks)
            
            # Sort by fitness descending
            indexed_results = sorted(zip(range(pop_size), results), key=lambda x: x[1]['fitness'], reverse=True)
            ranked_indices = [idx for idx, _ in indexed_results]
            ranked_fitness = [res['fitness'] for _, res in indexed_results]
            
            gen_best_idx = ranked_indices[0]
            gen_best_res = indexed_results[0][1]
            gen_best_fit = gen_best_res['fitness']
            gen_avg_fit = np.mean(ranked_fitness)
            
            if gen_best_fit > best_overall_fitness:
                best_overall_fitness = gen_best_fit
                champion_vector = np.copy(population[gen_best_idx])
                champion_telemetry = gen_best_res
                
            best_fitness_history.append(best_overall_fitness)
            gen_time = time.time() - t_gen_start
            
            # Print generation report
            print(f"Gen {gen:02d}/{num_generations:02d} | "
                  f"Best Fit: {gen_best_fit:7.1f} (Champ: {best_overall_fitness:7.1f}) | "
                  f"Avg Fit: {gen_avg_fit:6.1f} | "
                  f"Dist: {gen_best_res['distance']:4.1f}m | "
                  f"Gates: {gen_best_res['gates_cleared']}/4 | "
                  f"Max Roll: {gen_best_res['max_roll_deg']:4.1f}° | "
                  f"Bank: {gen_best_res['bank_score']:4.1f} | "
                  f"Crash: {'YES' if gen_best_res['collided'] else 'NO'} | "
                  f"Time: {gen_time:4.1f}s")
                  
            # Elitism: Keep top 4 unchanged
            elites = [np.copy(population[idx]) for idx in ranked_indices[:4]]
            
            # Create next generation
            next_population = list(elites)
            
            # Dynamic mutation rate decay
            sigma_gen = max(0.012, sigma * (0.96 ** gen))
            
            # 12 Mutants from top elites
            for _ in range(12):
                p_idx = np.random.choice(len(elites))
                parent = elites[p_idx]
                noise = np.random.randn(dim).astype(np.float32) * sigma_gen
                next_population.append(parent + noise)
                
            # 6 Crossover children
            for _ in range(6):
                idx_a, idx_b = np.random.choice(len(elites), size=2, replace=False)
                parent_a, parent_b = elites[idx_a], elites[idx_b]
                mask_cross = (np.random.rand(dim) > 0.5).astype(np.float32)
                child = parent_a * mask_cross + parent_b * (1.0 - mask_cross)
                child += np.random.randn(dim).astype(np.float32) * (sigma_gen * 0.7)
                next_population.append(child)
                
            # 2 High-exploration mutants
            for _ in range(pop_size - len(next_population)):
                p_idx = np.random.choice(len(elites))
                parent = elites[p_idx]
                noise = np.random.randn(dim).astype(np.float32) * (sigma_gen * 2.2)
                next_population.append(parent + noise)
                
            population = next_population
            
    total_time = time.time() - t_start_all
    print("\n====================================================================")
    print(f"POPULATION TRAINING COMPLETE in {total_time:.1f} seconds ({(total_time/60):.2f} min)")
    print(f"Champion Fitness: {best_overall_fitness:.1f}")
    if champion_telemetry:
        print(f"  - Distance Cleared: {champion_telemetry['distance']:.2f} m")
        print(f"  - Gates Traversed: {champion_telemetry['gates_cleared']} / 4")
        print(f"  - Max Acrobatic Roll: {champion_telemetry['max_roll_deg']:.1f} degrees (Banked Turn)")
        print(f"  - Max Pitch Dive/Climb: {champion_telemetry['max_pitch_deg']:.1f} degrees")
        print(f"  - Collision Status: {'COLLIDED' if champion_telemetry['collided'] else 'ZERO CONTACT (100% CLEAN)'}")
    print("====================================================================")
    
    # 3. Save Champion Weights
    set_trainable_vector(BASE_MODEL, champion_vector)
    
    save_path = "models/fly_cfc_v3_acrobatic.pth"
    save_dict = {
        'epoch': num_generations,
        'model_state': BASE_MODEL.state_dict(),
        'fitness': best_overall_fitness,
        'telemetry': champion_telemetry,
        'generation_history': best_fitness_history,
        'version': 'v3.0-acrobatic-spherical'
    }
    torch.save(save_dict, save_path)
    print(f"Saved Champion Fly-CfC v3 model to: {save_path}")
    
    # FP16 lightweight edition
    fp16_dict = {k: v.half() if v.is_floating_point() else v for k, v in BASE_MODEL.state_dict().items()}
    save_path_fp16 = "models/fly_cfc_v3_acrobatic_fp16.pth"
    torch.save({'model_state': fp16_dict, 'fitness': best_overall_fitness, 'version': 'v3.0-fp16'}, save_path_fp16)
    print(f"Saved Lightweight FP16 model to: {save_path_fp16}")

if __name__ == '__main__':
    train_population(pop_size=24, num_generations=25, max_steps=240, num_workers=8)
