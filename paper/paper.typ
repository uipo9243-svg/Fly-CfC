#set page(
  paper: "us-letter",
  margin: (x: 2cm, y: 2.5cm),
  header: align(right)[
    #text(size: 8.5pt, fill: rgb("#666666"))[
      *Fly-CfC: Connectome-Constrained Continuous-Time Networks for Autonomous Flight*
    ]
  ],
  footer: align(center)[#context text(size: 9pt)[#counter(page).display()]]
)

#set text(
  font: ("Liberation Serif", "Nimbus Roman", "Noto Serif"),
  size: 10pt,
)

#set par(justify: true, leading: 0.65em)

// Title Block
#align(center)[
  #v(0.5cm)
  #text(size: 17pt, weight: "bold")[
    Fly-CfC: Connectome-Constrained Closed-Form Continuous-Time Neural Networks for Sub-Millisecond Autonomous Flight
  ]
  #v(0.8em)
  #text(size: 11pt, weight: "medium")[
    *Surayut Chanthanarat*
  ]\
  #text(size: 9.5pt, fill: rgb("#333333"))[
    Independent Researcher, Bangkok, Thailand\
    Email: `uipo9243@gmail.com` | Repository: #link("https://github.com/uipo9243-svg/Fly-CfC")[https://github.com/uipo9243-svg/Fly-CfC]
  ]
  #v(0.5em)
  #text(size: 9.5pt, style: "italic")[
    November 2026
  ]
  #v(1em)
]

// Abstract Box
#rect(
  width: 100%,
  fill: rgb("#f8f9fa"),
  stroke: rgb("#d0d7de"),
  radius: 4pt,
  inset: 12pt,
)[
  #align(center)[#text(weight: "bold", size: 10.5pt)[Abstract]]
  #v(0.3em)
  Continuous-time recurrent models, specifically Liquid Time-Constant (LTC) networks and Closed-Form Continuous-Time (CfC) models, have demonstrated remarkable adaptability, out-of-distribution robustness, and parameter efficiency in robotic control. However, bio-constrained architectures have historically remained confined to the rudimentary 19-to-28-neuron motor circuitry of the nematode _Caenorhabditis elegans_. Conversely, whole-brain connectome simulations of higher organisms typically rely on heavy biophysical Hodgkin-Huxley or spiking formalisms that incur catastrophic computational overheads, rendering closed-loop robotic flight impossible. Here, we present *Fly-CfC*, the first whole-brain connectome-constrained Closed-Form Continuous-Time neural architecture scaled from the complete central nervous system of the adult fruit fly (_Drosophila melanogaster_, `male-cns v1.0`, 166,000 neurons, 151.85 million synapses). We extract an intact visuomotor sub-circuit comprising 1,683 active Visual Projection Neurons (VPNs), 1,961 recurrent internal units (Central Complex and Descending Neurons), and 1,300 Descending Neurons acting as an anatomical motor bottleneck to drive 4 flight actuators, establishing *99.5119% biological sparsity* (34,882 active synapses out of 7.15 million dense pairs). By embedding static biological masks directly into CfC continuous-time state transitions and descending linear readouts, we mathematically prove and empirically verify zero gradient leakage ($partial y \/ partial W_("masked") = 0.0000000$) and invariant output stability under extreme synaptic perturbation noise ($+100.0$). In closed-loop evaluations, Fly-CfC achieves sub-millisecond inference latency (0.45--0.52 ms/step, >1,900 FPS on an RTX 4060 GPU), a 100% survival rate across 30-obstacle benchmark arenas (4.14--4.61 m/s mean velocity, minimum clearance 4.34 m), and zero-contact autonomous slalom navigation across 600 control steps (20.0 s) in full 3D PyBullet quadcopter physics. Our results demonstrate that macroscopic whole-brain connectomics can directly inform ultra-sparse, continuous-time edge neural controllers for agile autonomous aerial vehicles.

  #v(0.5em)
  *Keywords:* Liquid Neural Networks, Closed-Form Continuous-Time (CfC), Drosophila Melanogaster, Whole-Brain Connectome, Autonomous UAV Navigation, Bio-Constrained AI.
]

#v(1em)

= 1. Introduction

Autonomous aerial navigation in unstructured, cluttered environments remains one of the foundational challenges in robotics and artificial intelligence. Micro Air Vehicles (MAVs) and autonomous quadcopters operating in subterranean tunnels, collapsed buildings, and dense forests require agile collision avoidance, rapid sensorimotor reflexes, and high energetic efficiency under severe onboard computational constraints. Conventional deep reinforcement learning (RL) and transformer-based policies operating in discrete time steps typically require millions of parameters, incur tens of milliseconds of latency, and exhibit fragile degradation when confronted with out-of-distribution visual disturbances or abrupt aerodynamic shifts.

Nature has long resolved this challenge through continuous-time neural computing. The common fruit fly (_Drosophila melanogaster_) executes aerial saccades in under 50 milliseconds, navigates dense foliage, and avoids moving predators using a brain smaller than a poppy seed operating on less than one microwatt of metabolic power. Recent breakthroughs in electron microscopy connectomics have culminated in the complete synaptic mapping of the adult _Drosophila_ central nervous system, identifying over 166,000 neurons and 151.85 million synaptic connections (Berg et al., 2026; Takemura et al., 2023; Dorkenwald et al., 2024; Schlegel et al., 2024).

Concurrently, modern continuous-time deep learning has evolved with Neural Ordinary Differential Equations (Neural ODEs) (Chen et al., 2018), Liquid Time-Constant (LTC) networks (Hasani et al., 2021), and Closed-Form Continuous-Time (CfC) models (Hasani et al., 2022). CfCs bypass numerical ODE solvers by deriving an explicit closed-form approximation of the continuous neural state trajectory:
$ x(t) approx sigma(-f(x, I)t) dot g(x, I) + [1 - sigma(-f(x, I)t)] dot h(x, I) $
This formulation delivers continuous-time expressiveness, order-of-magnitude faster training, and microsecond inference speeds. However, existing bio-constrained Neural Circuit Policies (NCPs) (Lechner et al., 2020) have remained strictly limited to the 19--28 neurons of the microscopic roundworm _Caenorhabditis elegans_. While mathematically elegant, nematode circuitry lacks visual processing centers, multi-axis steering bottlenecks, and optical looming reflexes essential for three-dimensional aerial locomotion.

In this work, we bridge this fundamental gap by introducing *Fly-CfC*: the first whole-brain connectome-constrained Closed-Form Continuous-Time neural network for autonomous robotic flight. Our contributions are fourfold:
1. *Biological Circuit Distillation:* We extract a functional sensorimotor graph from the Janelia Male CNS v1.0 connectome, pruning 2,898 orphan sensory units to isolate 1,683 active Visual Projection Neurons (VPNs), 1,961 internal units (Central Complex 661 units + Descending Neurons 1,300 units), and 34,882 verified synapses, achieving 99.5119% anatomical sparsity.
2. *Descending Steering Bottleneck & Zero Leakage:* We enforce an anatomical motor routing constraint where motor commands (thrust, roll, pitch, yaw) are commanded strictly via 1,300 Descending Neurons, masking out direct shortcut leaks from the Central Complex. We formally prove and verify zero gradient leakage to non-biological weights ($partial cal(L) \/ partial W_("masked") = 0.0000000$) and invariant output stability under extreme perturbation.
3. *Continuous-Time Compound Eye Vision:* We design an end-to-end biological retinal projector mapping 64 panoramic ommatidia and optical looming expansion rates into bilateral receptive fields matching anatomical lobula and lobula plate neurons (LC4, LC10, LPLC1, LPLC2).
4. *Empirical Verification in 2D and 3D Physics:* We evaluate Fly-CfC across both dense 2D obstacle benchmark arenas and 3D PyBullet quadcopter physics, demonstrating sub-millisecond latency (0.45--0.52 ms/step, >1,900 FPS on an RTX 4060 GPU), zero collisions (100% survival), and robust slalom flight through tight corridors.

#v(1em)

#figure(
  image("figures/fig3_circuit_architecture.png", width: 95%),
  caption: [
    *Connectome-Constrained Sensorimotor Circuit of Fly-CfC.* Visual signals from 64 panoramic ommatidia (128 channels with distance and looming expansion rate) are projected via anatomical matrix $P$ into 1,683 active Visual Projection Neurons (VPNs). The internal continuous-time CfC recurrent brain comprises 1,961 units (661 Central Complex + 1,300 Descending Neurons) constrained by 28,592 recurrent synapses ($99.26\%$ internal sparsity). Motor commands (thrust, roll, pitch, yaw) are generated exclusively via 1,300 Descending Neurons, guaranteeing zero shortcut leakage from the Central Complex.
  ]
)

#v(1em)

= 2. Related Work

== 2.1 Continuous-Time Deep Learning and Liquid AI
Traditional recurrent neural networks (RNNs, LSTMs, GRUs) model discrete transitions between fixed time steps. Neural ODEs (Chen et al., 2018) extended deep learning to continuous-time dynamical systems $dot(x)(t) = f(x(t), t, theta)$, but relied on adaptive numerical solvers that suffer from significant computational latency and gradient instability during backpropagation.

To address these limitations, Hasani et al. (2021) introduced Liquid Time-Constant (LTC) networks, which incorporate state-dependent time constants inspired by non-linear biological cell membrane equations. Hasani et al. (2022) subsequently solved the numerical bottleneck by deriving Closed-Form Continuous-Time (CfC) networks. CfCs compute closed-form analytical solutions to liquid differential equations, achieving 100x to 1,000x speedups in training and inference while preserving continuous-time temporal adaptation.

== 2.2 Bio-Constrained Architectures & Neural Circuit Policies
The concept of constraining neural network topology using biological wiring diagrams was pioneered by Lechner et al. (2020) with Neural Circuit Policies (NCPs). By embedding the 19-to-28-neuron motor connectome of _Caenorhabditis elegans_, NCPs demonstrated remarkable parameter efficiency, interpretability, and robust autonomous steering. However, _C. elegans_ is a non-visual, non-flying organism with an extremely primitive behavioral repertoire. Scaling bio-constrained architectures to organisms capable of vision and aerial locomotion has previously remained an unsolved challenge.

== 2.3 Whole-Brain Connectomics of _Drosophila melanogaster_
Recent advances in electron microscopy connectomics have unveiled the complete connectome of the adult fruit fly. Berg et al. (2026) and Takemura et al. (2023) mapped the complete Janelia Male CNS v1.0 and ventral nerve cord connectomes, detailing hundreds of millions of synapses across the central brain and motor pathways. Dorkenwald et al. (2024) and Schlegel et al. (2024) published the FlyWire whole-brain connectome, providing comprehensive cell typing and neuropil annotations. While neuroscientists have analyzed these circuits descriptively, Fly-CfC represents the first engineering framework to translate the whole-brain _Drosophila_ connectome into a differentiable continuous-time control network for robotics.

#v(1em)

= 3. Architecture & Mathematical Formulation

The Fly-CfC architecture consists of three interconnected biological modules: (1) The Biological Retinal Projector, (2) The Connectome-Constrained CfC Cell, and (3) The Descending Motor Bottleneck Head.

== 3.1 Connectome Graph Extraction & Pruning
Let $cal(G)_("cns") = (cal(V)_("cns"), cal(E)_("cns"))$ denote the complete directed connectome graph of the _Drosophila_ central nervous system containing $|cal(V)_("cns")| approx 166,000$ vertices and $|cal(E)_("cns")| approx 1.51 times 10^8$ weighted directed edges.

To construct an autonomous flight controller, we isolate three critical neuropils:
1. *Sensory Periphery ($cal(V)_("sensory") $):* Visual Projection Neurons (VPNs) conveying retinotopic visual signals and motion cues from the optic lobes to the central brain.
2. *Central Complex ($cal(V)_("CX") $):* Compass neurons and ring neurons in the Ellipsoid Body (EB) and Fan-Shaped Body (FB) responsible for spatial orientation and heading direction (Seelig & Jayaraman, 2015).
3. *Descending Neurons ($cal(V)_("DN") $):* Motor bottleneck neurons in the gnathal ganglion and ventral brain projecting down the ventral nerve cord (VNC) to wing motor circuits (Namiki et al., 2018).

In the raw dataset (`male-cns v1.0`), $cal(V)_("sensory")$ contains 4,581 putative units. Graph traversal reveals that 2,898 units (63.3%) possess an out-degree of zero ($d_("out") = 0$) with respect to the selected internal circuitry. We prune all sensory units where $d_("out") = 0$, yielding an active sensory population of $N_("sensory") = 1,683$ verified Visual Projection Neurons:
$ cal(V)_("active_sensory") = {v in cal(V)_("sensory") mid d_("out")(v, cal(V)_("internal")) > 0} $

The internal network comprises $N_("internal") = 1,961$ units, partitioned into $N_("CX") = 661$ Central Complex neurons and $N_("DN") = 1,300$ Descending Neurons. The biological connectivity is represented by two sparse adjacency matrices:
- $A_("sensory") in {0, 1}^(1683 times 1961)$ with $||"vec"(A_("sensory"))||_0 = 6,290$ active synapses.
- $A_("internal") in {0, 1}^(1961 times 1961)$ with $||"vec"(A_("internal"))||_0 = 28,592$ active synapses.

The combined recurrent matrix has dimension $(N_("sensory") + N_("internal")) times N_("internal") = 3,644 times 1,961$, totaling 7,145,884 potential entries. The total number of biologically active synapses is:
$ N_("synapses") = 6,290 + 28,592 = 34,882 $
defining an unprecedented combined biological sparsity of:
$ S_("bio") = 1 - 34,882 / 7,145,884 = 99.5119% $

== 3.2 Biological Retinal Projector
We model panoramic compound eye vision using 64 visual ray detectors spanning $360^degree$ horizontally, partitioned into left and right hemispheres (32 rays each). Each ommatidial channel produces two continuous signals:
1. Normalized distance to obstacles $d_i(t) in [0, 1]$.
2. Looming expansion velocity $L_i(t) = max(0, -dot(d)_i \/ d_i) in [0, 10]$ (Klapoetke et al., 2017).

This yields an input vector $I_("raw")(t) in RR^(128)$. We construct an anatomical projection matrix $P in RR^(128 times 1683)$ based on single-cell morphological annotations:
$ P_(r, j) = cases(
  1.0 & "if" j in "Non-Looming VPN and mapped to ray" r,
  0.25 & "if" j in "Looming VPN (LC4/LPLC) and mapped to ray" r,
  0.0 & "otherwise"
) $
The projected sensory activation $I_("vpn")(t) in RR^(1683)$ is computed with learnable gain $gamma in RR^(1683)$ and bias $beta in RR^(1683)$:
$ I_("vpn")(t) = tanh((I_("raw")(t) dot P) dot gamma + beta) $

== 3.3 Closed-Form Continuous-Time Dynamics with Frozen Masks
The recurrent state $x(t) in RR^(1961)$ evolves under Closed-Form Continuous-Time dynamics governed by the combined input-state vector $z(t) = [I_("vpn")(t); x(t)] in RR^(3644)$.

We construct a static biological sparsity buffer $M in {0, 1}^(1961 times 3644)$:
$ M = [A_("sensory")^T mid A_("internal")^T] $
The effective continuous-time transformation matrices are defined via Hadamard masking:
$ W_f^"eff" = W_f dot M, quad W_g^"eff" = W_g dot M, quad W_a^"eff" = W_a dot M, quad W_b^"eff" = W_b dot M $
where $W_f, W_g, W_a, W_b in RR^(1961 times 3644)$ are dense trainable parameter matrices, and $M$ is registered as a non-trainable buffer (`register_buffer`).

Given an elapsed time interval $Delta t$, the continuous state update is evaluated in closed form:
$ f(z) = W_f^"eff" z + b_f, quad g(z) = tanh(W_g^"eff" z + b_g) $
$ tau(z, Delta t) = sigma(-f(z) dot Delta t) $
$ h(z, Delta t) = W_a^"eff" z + b_a + (W_b^"eff" z + b_b) dot Delta t $
$ x(t + Delta t) = tau(z, Delta t) dot g(z) + [1 - tau(z, Delta t)] dot tanh(h(z, Delta t)) $

== 3.4 Descending Motor Bottleneck Head
In biological insects, motor commands driving flight do not originate from the central decision-making neuropils directly; rather, they pass strictly through Descending Neurons (Namiki et al., 2018). We enforce this anatomical constraint by introducing a descending motor mask $M_("motor") in {0, 1}^(4 times 1961)$:
$ (M_("motor"))_(k, j) = cases(
  0 & "if" j < 661 quad ("Central Complex unit"),
  1 & "if" 661 <= j < 1961 quad ("Descending Neuron")
) $
The flight control vector $u(t) = [u_("thrust"), u_("roll"), u_("pitch"), u_("yaw")]^T in RR^4$ is generated exclusively through the masked linear head:
$ u(t) = (W_("motor") dot M_("motor")) x(t) + b_("motor") $

== 3.5 Mathematical Proof & Empirical Verification of Zero Leakage
*Theorem 1 (Zero Synaptic Leakage).* _Let $cal(L)$ be any scalar loss function computed on the network output $u(t)$. Let $W_(i, j)$ be any parameter in $\{W_f, W_g, W_a, W_b\}$ such that $M_(i, j) = 0$. Then:_
$ (partial cal(L)) / (partial W_(i, j)) equiv 0.0000000 $

_Proof._ By the chain rule:
$ (partial cal(L)) / (partial W_(i, j)) = sum_k (partial cal(L)) / (partial W_(i, k)^"eff") (partial W_(i, k)^"eff") / (partial W_(i, j)) $
Since $W_(i, k)^"eff" = W_(i, k) dot M_(i, k)$, we have $(partial W_(i, k)^"eff") / (partial W_(i, j)) = delta_(k, j) M_(i, j)$. Substituting $M_(i, j) = 0$ yields:
$ (partial cal(L)) / (partial W_(i, j)) = (partial cal(L)) / (partial W_(i, j)^"eff") dot 0 equiv 0.0000000 quad square $

We empirically verify Theorem 1 via unit tests in `test_fly_model.py`. Computing the backward pass with respect to random batches yields:
- Active Synapse Gradient Norm: $||nabla_(W_("active")) cal(L)||_2 = 450.607$
- Inactive Synapse Gradient Leak: $||nabla_(W_("masked")) cal(L)||_1 = 0.0000000$
- Central Complex to Motor Leak: $||nabla_(W_("motor")[ :, 0:661 ]) cal(L)||_1 = 0.0000000$
In a perturbation falsification test injecting $cal(N)(0, 100^2)$ Gaussian noise into all non-biological weights, the output deviation was $max |u_("perturbed") - u_("base")| = 0.000000000$, confirming zero leakage.

#v(1em)

= 4. Experiments & Results

== 4.1 Computational Efficiency & Latency Benchmark
Table 1 compares Fly-CfC against standard deep learning and continuous-time baselines. Latency measurements were conducted on an Intel Core i7 processor and an NVIDIA GeForce RTX 4060 GPU (batch size = 1, sequence length = 15).

#align(center)[
  #table(
    columns: (2.8cm, 1.6cm, 1.8cm, 2.4cm, 1.8cm, 1.8cm, 2.6cm),
    stroke: (x, y) => if y == 0 { (bottom: 1.5pt + black) } else if y == 5 { (top: 1pt + black, bottom: 1.5pt + black) } else { (top: 0.5pt + luma(200)) },
    inset: 5pt,
    align: (left, center, center, center, center, center, center),
    [*Model*], [*Units*], [*Sparsity*], [*Parameters*], [*Latency*], [*FPS*], [*Survival*],
    [Standard MLP], [2,048], [0.0%], [7,671,812], [0.62 ms], [1,612], [40.0%],
    [Vanilla CfC], [1,961], [0.0%], [51,328,248], [1.18 ms], [847], [80.0%],
    [NCP (_C. elegans_)], [19], [82.4%], [1,844], [0.12 ms], [8,333], [0.0% (No Vision)],
    [*Fly-CfC v1.0*], [*1,961*], [*99.51%*], [*28,599,228*], [*0.45 ms*], [*2,222*], [*100.0%*],
    [*Fly-CfC v2.0*], [*1,961*], [*99.51%*], [*28,599,228*], [*0.48 ms*], [*2,083*], [*100.0%*],
  )
  #text(size: 8.5pt)[*Table 1:* Computational Efficiency and Real-Time Latency Benchmark on NVIDIA RTX 4060.]
]

Fly-CfC achieves a sub-millisecond inference latency of *0.45 ms/step* (>2,000 FPS), which is 2.6x faster than a dense CfC network of identical neuron count while reducing total memory graph footprint by over 99.5%.

== 4.2 2D Fixed-Obstacle Arena Flight
To evaluate obstacle avoidance and flight vigor, we tested Fly-CfC across 5 consecutive episodes in a dense $100 times 100$ m arena populated with 30 cylindrical obstacles. Survival requires completing 150 control steps ($Delta t = 30$ ms, total 4.5 seconds) while maintaining average velocity $>= 2.0$ m/s and flight distance $>= 15.0$ m.

#align(center)[
  #table(
    columns: (2.2cm, 2.0cm, 2.2cm, 2.2cm, 2.2cm, 2.4cm, 2.0cm),
    stroke: (x, y) => if y == 0 { (bottom: 1.5pt + black) } else if y == 5 { (top: 1pt + black, bottom: 1.5pt + black) } else { (top: 0.5pt + luma(200)) },
    inset: 5pt,
    align: (left, center, center, center, center, center, center),
    [*Episode*], [*Outcome*], [*Steps*], [*Distance*], [*Velocity*], [*Min Clear*], [*Latency*],
    [Episode 1], [PASS], [150 / 150], [20.8 m], [4.61 m/s], [7.75 m], [0.46 ms],
    [Episode 2], [PASS], [150 / 150], [19.6 m], [4.36 m/s], [6.44 m], [0.45 ms],
    [Episode 3], [PASS], [150 / 150], [20.2 m], [4.49 m/s], [7.59 m], [0.45 ms],
    [Episode 4], [PASS], [150 / 150], [18.6 m], [4.14 m/s], [4.34 m], [0.47 ms],
    [Episode 5], [PASS], [150 / 150], [20.6 m], [4.58 m/s], [6.60 m], [0.48 ms],
    [*Overall*], [*100% PASS*], [*750 / 750*], [*20.0 m*], [*4.44 m/s*], [*6.54 m*], [*0.46 ms*],
  )
  #text(size: 8.5pt)[*Table 2:* 2D Fixed-Obstacle Arena Telemetry Report across 5 consecutive episodes.]
]

#figure(
  image("figures/fig1_2d_arena_trajectories.png", width: 85%),
  caption: [
    *Closed-Loop Flight Trajectories in 2D Arena.* Fly-CfC trajectory traces across 5 consecutive episodes in a dense 30-obstacle arena (100% survival rate, 0 collisions, mean velocity 4.44 m/s, minimum clearance 4.34 m).
  ]
)

Fly-CfC attained a *100% survival rate (5/5 PASS)* with zero collisions (Figure 2), maintaining an average velocity of $4.44$ m/s and an ample minimum safety clearance of $4.34$ m.

== 4.3 3D PyBullet Quadcopter Slalom Physics
We embedded Fly-CfC into a 3D PyBullet quadcopter environment (`gym-pybullet-drones`) operating under full aerodynamic drag, ground effect, motor thrust dynamics, and contact point collision physics (`p.getContactPoints()`).

The flight task requires navigating an elongated slalom corridor containing 16 dense pillars with opening gates of 3.6 m. To overcome head-on stagnation and wall-grinding, we integrated:
1. *Front-Biased Receptive Fields:* Gaussian sensitivity peaking at $plus.minus 35^degree$ azimuth with positive frontal baseline weight ($0.41$).
2. *Bilateral Symmetry Breaking:* Emergency yaw torque commanded when frontal proximity exceeds a critical threshold ($d_("norm") > 0.28$), forcing the model to select an evasive saccade direction.

#figure(
  image("figures/fig2_3d_slalom_progression.png", width: 95%),
  caption: [
    *3D Autonomous Quadcopter Slalom Flight in PyBullet.* Time-lapse progression at $t = 2.0" s"$, $t = 6.0" s"$, $t = 10.0" s"$, and $t = 14.0" s"$. Fly-CfC v2 commands continuous thrust, roll, pitch, and yaw, successfully negotiating tight 3.6 m slalom gates with zero collisions across 600 control steps (20.0 s) and maintaining a minimum clearance of 0.393 m.
  ]
)

Figure 3 illustrates the multi-panel time progression of the quadcopter executing closed-loop autonomous slalom flight across 600 control steps (20.0 seconds at 30 Hz control frequency / 240 Hz physics simulation). The real-time Head-Up Display (HUD) telemetry confirms:
- *Total Collisions:* 0 across all 600 steps (*Zero Contact 100%*).
- *Minimum Obstacle Clearance:* $0.393$ m (safely exceeding drone radius $0.15$ m).
- *Control Loop Latency:* $0.48$ ms/step.

#v(1em)

= 5. Discussion, Limitations & Ethics

== 5.1 Biological Plausibility vs. Engineered Utility
Fly-CfC demonstrates that biological connectomes are not merely descriptive anatomical maps, but functional inductive biases for artificial neural networks. By preserving the 99.51% sparsity of _Drosophila_, Fly-CfC eliminates over 99 out of 100 unnecessary synapse calculations. Unlike Spiking Neural Networks (SNNs), which struggle with non-differentiable threshold functions, Closed-Form Continuous-Time models provide exact continuous gradients while retaining biological sparse topology.

== 5.2 Limitations
Several limitations provide fruitful directions for future research:
1. *Sim-to-Real Aerodynamics:* While PyBullet accurately models multi-rotor rigid body dynamics, real-world outdoor flight introduces unmodeled wind gusts, downwash turbulence, and motor saturation. Future work includes deploying the quantized C++ / ONNX implementation of Fly-CfC onto physical micro-quadcopters (e.g., Bitcraze Crazyflie) to benchmark real-world aerodynamic resilience.
2. *Neuromodulators:* The current architecture models fast ionotropic synaptic transmission but omits slow neuromodulators (octopamine, dopamine, serotonin), which modulate arousal and behavioral state in living flies.
3. *Ommatidia Resolution:* The current vision model employs 64 rays across $360^degree$; scaling to full micro-lens imaging (750 ommatidia per eye) with motion-energy optical flow will further improve spatial fidelity.

== 5.3 Ethics & Humanitarian Statement
Autonomous micro-aerial vehicles possess dual-use potential. We explicitly state our commitment that the development of Fly-CfC is intended for *humanitarian and civilian robotics applications*. The ultra-low latency (0.45 ms) and continuous-time adaptation of Fly-CfC are uniquely suited for:
- *Search and Rescue (SAR):* Navigating through collapsed buildings, disaster rubble, and subterranean voids where GPS is denied and human teleoperation is severed.
- *Environmental & Canopy Monitoring:* High-speed autonomous navigation under dense forest canopies for ecological surveys and wildfire detection.
- *Confined Infrastructure Inspection:* Autonomous inspection of industrial pipes, storage tanks, and underground transit tunnels.

The authors strongly oppose the deployment of this technology in autonomous lethal weapon systems and encourage open-source reproducibility under the Apache 2.0 license to foster civilian scientific advancement.

#v(1em)

= 6. Conclusion
We presented *Fly-CfC*, the world's first whole-brain connectome-constrained Closed-Form Continuous-Time neural network. By distilling 151.85 million synapses of _Drosophila melanogaster_ into a 99.51% sparse circuit of 1,683 Visual Projection Neurons, 1,961 internal units, and 1,300 Descending Neurons, Fly-CfC achieves sub-millisecond continuous-time inference (0.45 ms/step, >1,900 FPS) and flawless closed-loop collision avoidance across both 2D benchmark arenas and full 3D quadcopter physics. Fly-CfC bridges the longstanding gap between macroscopic connectomics and real-time robotic intelligence, paving the way for the next generation of brain-constrained edge autonomous vehicles.

#v(1em)

= Acknowledgments
The author thanks the Janelia FlyEM project team and the FlyWire Consortium for reconstructing and annotating the whole-brain _Drosophila_ connectome datasets, as well as the developers of the `ncps` framework and `gym-pybullet-drones` for their open-source tools.

#v(1em)

= Data and Code Availability
All code, models, and simulation environments developed in this work are openly available at https://github.com/uipo9243-svg/Fly-CfC under the Apache 2.0 license. The underlying connectome dataset (`male-cns:v1.0`) is accessible via Janelia neuPrint (https://neuprint.janelia.org/).

#v(1em)

= References
+ Hasani, R., Lechner, M., Amini, A., Liebenwein, L., Ray, A., Tschaikowski, M., Teschl, G., & Rus, D. (2022). Closed-form continuous-time neural networks. _Nature Machine Intelligence_, 4(11), 992–1003.
+ Berg, S., Beckett, I. R., Costa, M., Schlegel, P., Januszewski, M., et al. (2026). Sexual dimorphism in the complete Drosophila male central nervous system connectome. _Cell_, 189(18), 5504–5526. https://doi.org/10.1016/j.cell.2026.08.015
+ Lechner, M., Hasani, R., Amini, A., Henzinger, T. A., Rus, D., & Grosu, R. (2020). Neural circuit policies enabling auditable autonomous driving. _Nature Machine Intelligence_, 2(10), 642–652.
+ Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021). Liquid time-constant networks. _Proceedings of the AAAI Conference on Artificial Intelligence_, 35(9), 7657–7666.
+ Takemura, S., Shinomiya, K., Rivlin, P. K., Xu, C. S., Zhao, A., Huang, G. B., Zhao, T., Scheffer, L., & Plaza, S. M. (2023). A connectome of the male Drosophila ventral nerve cord and brain. _Nature_, 624, 383–392.
+ Dorkenwald, S., Schlegel, P., Sterling, A. R., Henning, J., Joyce, A. T., Kuehner, E., Silverman, B., Yang, R., Murthy, M., & Seung, H. S. (2024). Neuronal wiring diagram of an adult brain. _Nature_, 634, 124–138.
+ Schlegel, P., Yin, Y., Bates, A. S., Dorkenwald, S., Eichler, K., Brooks, P. A., Han, D. S., Gkantia, M., Santos, M. D., & Jefferis, G. S. (2024). Whole-brain annotation and multi-connectome mapping of Drosophila. _Nature_, 634, 139–152.
+ Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural ordinary differential equations. _Advances in Neural Information Processing Systems (NeurIPS)_, 31, 6571–6583.
+ Namiki, S., Dickinson, M. H., Wong, A. M., Korff, W., & Card, G. M. (2018). The functional organization of descending sensory-motor pathways in Drosophila. _eLife_, 7, e34272.
+ Seelig, J. D., & Jayaraman, V. (2015). Neural dynamics for landmark orientation and angular path integration. _Nature_, 521(7551), 186–191.
+ Klapoetke, N. C., Nern, A., Rogerson, D., Korff, W., Rubin, G. M., & Reiser, M. B. (2017). Ultra-selective looming detection in Drosophila. _Nature_, 551, 239–244.
+ Panerati, J., Zheng, H., Zhou, S., Xu, J., & Schoellig, A. P. (2021). Learning to fly---a gym environment with PyBullet physics for reinforcement learning of multi-agent quadcopter control. _IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)_, 7512–7519.
+ Floreano, D., Zufferey, J. C., Srinivasan, M. V., & Ellington, C. (2009). _Flying Insects and Robots_. Springer Science & Business Media.
+ de Croon, G. C. H. E., Dupeyroux, J. J. G., Fuller, S., & Zufferey, J. C. (2018). Insect-inspired autonomous flight of micro air vehicles. _Science Robotics_, 3(23), eaat2785.
+ Franceschini, N., Pichon, J. M., & Blanes, C. (1992). From insect vision to robot vision. _Philosophical Transactions of the Royal Society of London. Series B: Biological Sciences_, 337(1281), 283–294.
+ Coumans, E., & Bai, Y. (2016–2021). PyBullet, a Python module for physics simulation for games, robotics and machine learning.
