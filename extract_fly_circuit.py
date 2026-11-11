import pyarrow.feather as feather
import pyarrow.compute as pc
import pyarrow as pa
import numpy as np
import time

t0 = time.time()
print("--- 1. โหลดข้อมูล Body Annotations ---")
annotations_path = "male_cns_connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather"
weights_path = "male_cns_connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather"

df_ann = feather.read_feather(annotations_path)
print(f"โหลดข้อมูลเซลล์สำเร็จ: {len(df_ann):,} นิวรอน (ใช้เวลา {time.time()-t0:.2f} วินาที)")

# --- 2. นิยามเงื่อนไขการคัดเลือกวงจรสายตาและการบิน (Visual-to-Steering Circuit) ---
is_sensory = df_ann['superclass'].isin(['visual_projection']) & df_ann['type'].str.contains('LC|LPLC', na=False)
is_central = df_ann['superclass'].isin(['cb_intrinsic']) & df_ann['type'].str.contains('E-PG|P-EN|Delta', na=False)
is_motor = df_ann['superclass'].isin(['descending_neuron']) & df_ann['type'].str.contains('DN', na=False)

selected_sensory = df_ann[is_sensory]['bodyId'].values
selected_central = df_ann[is_central]['bodyId'].values
selected_motor = df_ann[is_motor]['bodyId'].values

# จัดลำดับ: Central neurons ก่อน แล้วตามด้วย Motor neurons ท้ายสุด
selected_internal = np.concatenate([selected_central, selected_motor])

print(f"พบ Sensory Neurons (VPNs): {len(selected_sensory):,} เซลล์")
print(f"พบ Central Complex (CX):   {len(selected_central):,} เซลล์")
print(f"พบ Motor Neurons (DNs):    {len(selected_motor):,} เซลล์")
print(f"รวม Internal Units:        {len(selected_internal):,} เซลล์")

all_selected = set(selected_sensory).union(set(selected_internal))

# --- 3. โหลดและตัดต่อไซแนปส์ด้วย PyArrow Fast Compute ---
t1 = time.time()
print("\n--- 2. โหลดและตัดต่อไซแนปส์ (Connectome Weights via PyArrow) ---")
table = feather.read_table(weights_path)
print(f"โหลดตารางไซแนปส์: {table.num_rows:,} แถว (ใช้เวลา {time.time()-t1:.2f} วินาที)")

t2 = time.time()
# กรอง weight >= 5 ก่อนเพื่อลดขนาดข้อมูลลงอย่างรวดเร็ว
SYNAPSE_THRESHOLD = 5
mask_w = pc.greater_equal(table['weight'], SYNAPSE_THRESHOLD)
table_w = table.filter(mask_w)
print(f"คงเหลือไซแนปส์ weight >= {SYNAPSE_THRESHOLD}: {table_w.num_rows:,} แถว (ใช้เวลา {time.time()-t2:.2f} วินาที)")

t3 = time.time()
# กรองเฉพาะโหนดที่อยู่ในกลุ่มเซลล์เป้าหมาย
pa_set = pa.array(list(all_selected), type=pa.int64())
mask_pre = pc.is_in(table_w['body_pre'], pa_set)
mask_post = pc.is_in(table_w['body_post'], pa_set)
mask_circuit = pc.and_(mask_pre, mask_post)
sub_table = table_w.filter(mask_circuit)
print(f"คงเหลือไซแนปส์ใน Sub-circuit: {sub_table.num_rows:,} เส้นทาง (ใช้เวลา {time.time()-t3:.2f} วินาที)")

# --- 4. สร้าง Adjacency Matrices สำหรับ ncps ---
t4 = time.time()
print("\n--- 3. สร้าง Adjacency Matrices สำหรับ ncps ---")
sensory_map = {body: idx for idx, body in enumerate(selected_sensory)}
internal_map = {body: idx for idx, body in enumerate(selected_internal)}

N_sensory = len(selected_sensory)
N_internal = len(selected_internal)

sensory_adj = np.zeros((N_sensory, N_internal), dtype=np.int32)
internal_adj = np.zeros((N_internal, N_internal), dtype=np.int32)

pre_arr = sub_table['body_pre'].to_numpy()
post_arr = sub_table['body_post'].to_numpy()

for pre, post in zip(pre_arr, post_arr):
    if pre in sensory_map and post in internal_map:
        sensory_adj[sensory_map[pre], internal_map[post]] = 1
    elif pre in internal_map and post in internal_map:
        internal_adj[internal_map[pre], internal_map[post]] = 1

# --- 5. Prune orphan VPNs: เก็บเฉพาะ sensory units ที่มีเส้นทางเข้า internal จริง ---
active_vpn = np.any(sensory_adj != 0, axis=1)
n_before = int(selected_sensory.shape[0])
selected_sensory = selected_sensory[active_vpn]
sensory_adj = sensory_adj[active_vpn]
print(f"Pruned orphan VPNs: {n_before:,} -> {selected_sensory.shape[0]:,} active sensory units")

# นับ unique edges จากเมทริกซ์จริง (ไม่นับ synapse ซ้ำ)

sensory_count = int(np.count_nonzero(sensory_adj))
internal_count = int(np.count_nonzero(internal_adj))
print(f"Sensory Matrix Shape:  {sensory_adj.shape} (unique edges: {sensory_count:,})")
print(f"Internal Matrix Shape: {internal_adj.shape} (unique edges: {internal_count:,})")
sparsity = 1.0 - (internal_count / internal_adj.size) if internal_adj.size > 0 else 0.0
print(f"Internal Sparsity:     {sparsity:.4%}")



# บันทึกไฟล์ผลลัพธ์
np.save("male_cns_connectome/sensory_adj.npy", sensory_adj)
np.save("male_cns_connectome/internal_adj.npy", internal_adj)
np.save("male_cns_connectome/selected_sensory_ids.npy", selected_sensory)
np.save("male_cns_connectome/selected_internal_ids.npy", selected_internal)
np.save("male_cns_connectome/selected_central_ids.npy", selected_central)
np.save("male_cns_connectome/selected_motor_ids.npy", selected_motor)

print(f"\nบันทึกไฟล์ NumPy สำเร็จทั้งหมด (เวลารวมทั้งสิ้น: {time.time()-t0:.2f} วินาที)")
