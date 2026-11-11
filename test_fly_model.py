import numpy as np
import torch
import time
import sys
import os

from fly_wiring import FlyVisuomotorWiring, FlyCfC

print("====================================================================")
print("DROSOPHILA CONNECTOME FLY-CFC v1.0 STRICT ARCHITECTURAL VERIFICATION")
print("====================================================================")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# 1. โหลด Adjacency Matrices ที่ผ่านการ Prune Orphan VPNs
sensory_adj = np.load("male_cns_connectome/sensory_adj.npy")
internal_adj = np.load("male_cns_connectome/internal_adj.npy")

n_sensory = sensory_adj.shape[0]   # 1,683 Active VPNs
n_internal = internal_adj.shape[0] # 1,961 (661 CX + 1,300 DNs)
syn_sensory = np.count_nonzero(sensory_adj)   # 6,290
syn_internal = np.count_nonzero(internal_adj) # 28,592
total_synapses = syn_sensory + syn_internal   # 34,882
total_matrix_entries = (n_sensory + n_internal) * n_internal # 7,145,884

internal_sparsity = 1.0 - (syn_internal / (n_internal * n_internal))
combined_sparsity = 1.0 - (total_synapses / total_matrix_entries)

print("สถิติโครงสร้างคอนเนคโตมชีวภาพ (v1.0 Clean Architecture):")
print(f"  - Active Sensory VPNs (Pruned): {n_sensory:,} (0 Orphan Nodes)")
print(f"  - Internal Units (CX + DN):     {n_internal:,} (661 CX + 1,300 DNs)")
print(f"  - Total Active Synapses:        {total_synapses:,} (Preserved 100%)")
print(f"  - Internal Block Sparsity:      {internal_sparsity:.4%}")
print(f"  - Combined Total Sparsity:      {combined_sparsity:.4%}")

assert n_sensory == 1683, f"Expected 1683 sensory neurons, got {n_sensory}"
assert n_internal == 1961, f"Expected 1961 internal neurons, got {n_internal}"
assert total_synapses == 34882, f"Expected 34882 synapses, got {total_synapses}"

# 2. สร้าง FlyVisuomotorWiring และ FlyCfC
wiring = FlyVisuomotorWiring(
    sensory_adjacency_matrix=sensory_adj,
    adjacency_matrix=internal_adj,
    output_dim=4,
    num_central=661
)
model = FlyCfC(wiring=wiring).to(device)

# ตรวจสอบว่า sparsity_mask และ motor_mask อยู่ใน buffers ไม่ใช่ parameters
param_names = [name for name, _ in model.named_parameters()]
buffer_names = [name for name, _ in model.named_buffers()]
assert "cell.sparsity_mask" in buffer_names, "sparsity_mask must be registered as a buffer!"
assert "cell.motor_mask" in buffer_names, "motor_mask must be registered as a buffer!"
assert "cell.sparsity_mask" not in param_names, "sparsity_mask must NOT be in model.parameters()!"
assert "cell.motor_mask" not in param_names, "motor_mask must NOT be in model.parameters()!"

total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  - Trainable Parameters:         {total_trainable:,}")

# 3. ทดสอบ Continuous-time Forward Pass
print("\n--- ทดสอบ Continuous-time Forward Pass ---")
BATCH_SIZE = 4
SEQ_LEN = 15

x_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, n_sensory, device=device, requires_grad=True)
timespans_tensor = torch.rand(BATCH_SIZE, SEQ_LEN, device=device) * 0.05 + 0.01

t_start = time.time()
output, hx = model(x_tensor, timespans=timespans_tensor)
if torch.cuda.is_available():
    torch.cuda.synchronize()
lat_ms = (time.time() - t_start) * 1000.0

print(f"Forward Tensor Shape:  {list(output.shape)} (Output Dim = 4)")
print(f"Forward Latency:       {lat_ms:.2f} ms ({lat_ms/SEQ_LEN:.3f} ms / step)")

assert output.shape == (BATCH_SIZE, SEQ_LEN, 4), f"Output shape mismatch: {output.shape}"
assert hx.shape == (BATCH_SIZE, n_internal), f"Hidden state shape mismatch: {hx.shape}"

# 4. ทดสอบ Backward Pass & ตรวจสอบ Gradient Leakage
print("\n--- ทดสอบ Backward Pass & ตรวจสอบ Gradient Leakage ---")
loss = output.sum()
loss.backward()

assert x_tensor.grad is not None, "Gradient to input tensor is None!"
print(f"Input Gradient Norm:   {x_tensor.grad.norm().item():.4f}")

# ตรวจสอบ 4 Synaptic Pathways
for name, linear in [('ff1', model.cell.ff1),
                     ('ff2', model.cell.ff2),
                     ('time_a', model.cell.time_a),
                     ('time_b', model.cell.time_b)]:
    grad = linear.weight.grad
    assert grad is not None, f"Gradient for {name} is None!"
    leak = (grad * (1.0 - model.cell.sparsity_mask)).abs().sum().item()
    active_grad = (grad * model.cell.sparsity_mask).abs().sum().item()
    print(f"  - Pathway {name:6s}: Active Grad = {active_grad:.4f} | Non-existent Synapse Leak = {leak:.7f}")
    assert leak == 0.0, f"Critical Leak in {name}: {leak} != 0.0"
    assert active_grad > 0.0, f"No gradient flowing to active synapses in {name}!"

# ตรวจสอบ Motor Head Mask (ห้ามต่อตรงเข้า Central Complex 661 เซลล์แรก)
motor_grad = model.cell.motor_linear.weight.grad
cx_motor_leak = (motor_grad * (1.0 - model.cell.motor_mask)).abs().sum().item()
dn_motor_grad = (motor_grad * model.cell.motor_mask).abs().sum().item()
print(f"  - Motor Head    : DN Active Grad = {dn_motor_grad:.4f} | CX Direct Leak = {cx_motor_leak:.7f}")
assert cx_motor_leak == 0.0, f"Critical Leak: Central Complex directly driving motor head: {cx_motor_leak} != 0.0"
assert dn_motor_grad > 0.0, "Descending Neurons not receiving gradients in motor head!"

# 5. Perturbation Test: กวนสัญญาณน้ำหนักในจุดที่ไม่มีไซแนปส์ทั้งหมด แล้วดูว่าเอาต์พุตเปลี่ยนหรือไม่
with torch.no_grad():
    base_out, _ = model(x_tensor, timespans=timespans_tensor)
    
    # รบกวนน้ำหนักในจุดที่ไม่มีไซแนปส์ด้วย random noise มหาศาล (+100.0)
    noise = torch.randn_like(model.cell.ff1.weight) * 100.0
    model.cell.ff1.weight.data += noise * (1.0 - model.cell.sparsity_mask)
    model.cell.time_a.weight.data += noise * (1.0 - model.cell.sparsity_mask)
    model.cell.motor_linear.weight.data += (torch.randn_like(model.cell.motor_linear.weight) * 100.0) * (1.0 - model.cell.motor_mask)
    
    perturbed_out, _ = model(x_tensor, timespans=timespans_tensor)
    max_out_change = (perturbed_out - base_out).abs().max().item()
    print(f"\nPerturbation Falsification Test:")
    print(f"  - Max output change after perturbing non-existent synapses: {max_out_change:.9f}")
    assert max_out_change == 0.0, f"Falsification Failed! Output changed by {max_out_change}"

print("\n====================================================================")
print("ALL VERIFICATIONS PASSED: 100% CLEAN BIOLOGICAL MAPPING (ZERO LEAK)")
print("====================================================================")
