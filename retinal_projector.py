import os
import numpy as np
import torch
import torch.nn as nn

class RetinalProjector(nn.Module):
    """
    Biological Retinal Projector:
    Maps 64-ommatidia compound eye panoramic perception (128 raw visual channels)
    into the active Visual Projection Neurons (VPNs) with bilateral receptive fields.
    Orphan VPNs (no downstream synapse) are pruned upstream by extract_fly_circuit.py.
    
    Channel Alignment:
      - 0..31:   Left Eye Distance [0..1]
      - 32..63:  Left Eye Looming Expansion Rate [0..10]
      - 64..95:  Right Eye Distance [0..1]
      - 96..127: Right Eye Looming Expansion Rate [0..10]
    """
    def __init__(self, sensory_ids_path="male_cns_connectome/selected_sensory_ids.npy",
                 proj_matrix_path="male_cns_connectome/retinal_projection_matrix.npy",
                 annotations_path="male_cns_connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather"):
        super().__init__()
        
        if os.path.exists(proj_matrix_path):
            P = np.load(proj_matrix_path)
            N_vpn = P.shape[1]
        else:
            import pyarrow.feather as feather
            sensory_ids = np.load(sensory_ids_path)
            N_vpn = len(sensory_ids) # e.g. 1,683 active VPNs after pruning
            N_raw = 128
            
            # Load biological annotations
            df_ann = feather.read_feather(annotations_path)
            ann_dict = df_ann.set_index('bodyId').to_dict('index')
            
            # Map bodyId directly to index 0..4580 to prevent any ordering misalignment
            raw_sides = [ann_dict.get(int(bid), {}).get('somaSide') for bid in sensory_ids]
            neuron_types = [str(ann_dict.get(int(bid), {}).get('type', '')) for bid in sensory_ids]
            
            left_indices = np.array([i for i, s in enumerate(raw_sides) if s == 'L'], dtype=np.int64)
            right_indices = np.array([i for i, s in enumerate(raw_sides) if s == 'R'], dtype=np.int64)
            unknown = [i for i, s in enumerate(raw_sides) if s not in ('L', 'R')]
            if unknown:
                print(f"[RetinalProjector] WARNING: {len(unknown)} VPNs have no L/R somaSide; routing them to the right eye")
                right_indices = np.sort(np.concatenate([right_indices, np.array(unknown, dtype=np.int64)]))
            
            assert len(left_indices) + len(right_indices) == N_vpn, (
                f"RetinalProjector index mismatch: {len(left_indices)} + {len(right_indices)} != {N_vpn}"
            )
            
            # Looming-detecting neurons (LC4, LPLC1, LPLC2)
            is_looming = np.array(['LPLC' in t or 'LC4' in t for t in neuron_types], dtype=bool)
            
            # Build Sparse Anatomical Projection Matrix P [128, N_vpn]
            P = np.zeros((N_raw, N_vpn), dtype=np.float32)
            
            # 1. Left Eye: Channels 0..63 -> Left VPNs
            left_non_looming = left_indices[~is_looming[left_indices]]
            for i, idx in enumerate(left_non_looming):
                ray_idx = int(np.floor((i / max(1, len(left_non_looming))) * 32))
                P[ray_idx, idx] = 1.0
                
            left_looming_idx = left_indices[is_looming[left_indices]]
            for i, idx in enumerate(left_looming_idx):
                ray_idx = 32 + int(np.floor((i / max(1, len(left_looming_idx))) * 32))
                P[ray_idx, idx] = 0.25
                
            # 2. Right Eye: Channels 64..127 -> Right VPNs
            right_non_looming = right_indices[~is_looming[right_indices]]
            for i, idx in enumerate(right_non_looming):
                ray_idx = 64 + int(np.floor((i / max(1, len(right_non_looming))) * 32))
                P[ray_idx, idx] = 1.0
                
            right_looming_idx = right_indices[is_looming[right_indices]]
            for i, idx in enumerate(right_looming_idx):
                ray_idx = 96 + int(np.floor((i / max(1, len(right_looming_idx))) * 32))
                P[ray_idx, idx] = 0.25
            
            # Cache computed matrix
            os.makedirs(os.path.dirname(proj_matrix_path), exist_ok=True)
            np.save(proj_matrix_path, P)

            
        # Register as buffer
        self.register_buffer("P", torch.from_numpy(P))
        
        # Trainable adaptive scaling
        self.gain = nn.Parameter(torch.ones(N_vpn))
        self.bias = nn.Parameter(torch.zeros(N_vpn))
        self.tanh = nn.Tanh()

    def forward(self, raw_visual_input):
        projected = torch.matmul(raw_visual_input, self.P)
        return self.tanh(projected * self.gain + self.bias)
