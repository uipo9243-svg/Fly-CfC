# Drosophila Melanogaster Connectome Dataset

This directory contains the biological adjacency matrices extracted from the **Janelia Male CNS v1.0 connectome** (Takemura et al., 2023 / Janelia FlyWire Consortium).

## Files Included in Repository
* `sensory_adj.npy` (1,683 x 1,961, ~13 MB): Synaptic connections from active Visual Projection Neurons (VPNs) to internal brain units.
* `internal_adj.npy` (1,961 x 1,961, ~15 MB): Synaptic connections within the internal brain (Central Complex + Descending Neurons).
* `selected_sensory_ids.npy`: Root IDs of the 1,683 active VPNs (after pruning orphan sensory units with zero downstream paths).
* `selected_internal_ids.npy`: Root IDs of the 1,961 internal units (661 CX + 1,300 DNs).
* `selected_central_ids.npy`: Root IDs of the 661 Central Complex neurons.
* `selected_motor_ids.npy`: Root IDs of the 1,300 Descending Neurons.

## Reproducing Extraction from Raw Data
To re-extract the circuit directly from the raw 151.85M synaptic feather dump:
1. Download `body-annotations-male-cns-v1.0-minconf-0.5.feather` and `connectome-weights-male-cns-v1.0-minconf-0.5.feather` from Janelia / FlyWire.
2. Place both files in this directory.
3. Run:
```bash
python extract_fly_circuit.py
```
