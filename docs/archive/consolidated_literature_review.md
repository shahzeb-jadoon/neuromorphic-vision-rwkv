# Consolidated Literature Review: RWKV and Spiking Neural Networks for Autonomous Vehicle Control

## Abstract

This consolidated literature review synthesizes findings from two independent analyses examining the Receptance Weighted Key Value (RWKV) architecture and Spiking Neural Networks (SNNs) as alternatives to Transformer-based systems for autonomous driving applications. Both source reviews converge on the conclusion that RWKV represents a "Third Way" architecture—bridging the gap between the quadratic complexity of Transformers and the training difficulties of SNNs. This document integrates their mathematical derivations, efficiency arguments, vision adaptation strategies, and state-of-the-art positioning while providing critical analysis of their claims and verifying their shared conclusions.

---

## 1. Introduction: The Computational Challenge in Autonomous Perception

The evolution of autonomous driving (AD) systems has been largely shaped by advances in deep learning architectures. Vision Transformers (ViTs) and their multimodal derivatives have established state-of-the-art benchmarks across perception tasks including 3D object detection, semantic segmentation, and trajectory prediction [1, 2]. However, the deployment of these architectures within safety-critical, resource-constrained autonomous vehicles exposes a fundamental tension between model expressivity and computational tractability.

### 1.1 The Quadratic Complexity Bottleneck

The central bottleneck is the **quadratic time and memory complexity** ($O(N^2)$) inherent to the standard self-attention mechanism [1, 3, 4]. In natural language processing (NLP), this constraint limits the context window; in autonomous driving, where the "context" is a continuous, high-bandwidth stream of temporal sensor data—LiDAR point clouds, high-resolution camera feeds, and inertial measurements—this complexity becomes prohibitive.

**Practical Example**: For an AD perception system operating at 30 FPS with a 10-second context window, the sequence involves 300 frames. If each frame is tokenized into 256 patches, the model must handle a sequence length of 76,800 tokens. At this scale, the $O(N^2)$ cost of self-attention becomes computationally infeasible for real-time edge execution [1, 3].

Current state-of-the-art approaches mitigate this through sliding windows or aggressive temporal downsampling, introducing **"context fragmentation"**—the loss of long-range temporal dependencies that compromises the system's ability to reason about latent intentions of other road users over extended horizons [4, 5].

### 1.2 The Energy Efficiency Imperative

The energy efficiency of GPU-based inference for quadratic attention mechanisms poses a tangible threat to the operational viability of electric autonomous vehicles, where the compute stack competes directly with the powertrain for battery resources. Edge compute modules (e.g., NVIDIA Orin, Qualcomm Snapdragon Ride) typically operate within tight power envelopes (< 100W for the entire AD stack) to preserve vehicle range [8, 27].

### 1.3 The RWKV Proposition

The **Receptance Weighted Key Value (RWKV)** architecture represents a novel class of "Linear Transformers" that mathematically reconcile the parallelizable training advantages of Transformers with the efficient, constant-time inference ($O(1)$) of Recurrent Neural Networks (RNNs) [1, 3, 5, 6, 7]. This review posits that RWKV is not merely an optimization but a **structural bridge to Neuromorphic Computing**—serving as a "dense" or "rate-coded" proxy for Leaky Integrate-and-Fire (LIF) neurons that is differentiable and trainable on standard hardware [6, 10, 13].

---

## 2. The Mathematical Bridge: RWKV and Neuromorphic Dynamics

The fundamental insight underlying both source reviews is the mathematical realization that the RWKV architecture's time-mixing mechanism exhibits structural isomorphism with the temporal dynamics of spiking neurons. This section establishes this correspondence rigorously.

### 2.1 The RWKV Time-Mixing Operator

The core innovation of RWKV is the **Time-Mixing block**, which replaces the Self-Attention mechanism of Transformers. Unlike the stateless dot-product of standard attention, Time-Mixing is formulated as a channel-wise recurrence.

#### Recurrent State Formulation (RWKV-4/5/6)

The model maintains a hidden state that is updated recursively. The state at time $t$ comprises a numerator state ($a_t$) and a denominator state ($b_t$). The update rules are governed by:

$$a_t = e^{-w} \cdot a_{t-1} + e^{k_t} \cdot v_t$$

$$b_t = e^{-w} \cdot b_{t-1} + e^{k_t}$$

$$wkv_t = \frac{a_t}{b_t}$$

The final output is gated by the receptance vector $r_t$:

$$o_t = \sigma(r_t) \odot wkv_t$$

**Variable Definitions**:
- $k_t$: **Key vector** — acts as a gating signal for the input
- $v_t$: **Value vector** — carries the semantic content of the input
- $r_t$: **Receptance** — a sigmoid gate that modulates the output
- $w$: **Time Decay vector** — must be non-negative to ensure stability; $e^{-w}$ acts as a "forgetting factor" [3, 5, 6]

#### Global WKV Computation

The global form of the WKV computation at time step $t$ is:

$$wkv_t = \frac{\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} \odot v_i + e^{u+k_t} \odot v_t}{\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} + e^{u+k_t}}$$

where $u$ is a "bonus" vector for the current token to emphasize immediate information [3, 5].

### 2.2 The Leaky Integrate-and-Fire (LIF) Neuron Model

The LIF neuron is the standard model for SNNs, describing the evolution of a membrane potential $V(t)$ over time [15, 16, 17, 18, 19]. The continuous-time differential equation is:

$$\tau_m \frac{dV(t)}{dt} = -(V(t) - E_L) + R_m I(t)$$

**Variable Definitions**:
- $\tau_m = R_m C_m$: **Membrane time constant** — determines how quickly the neuron "forgets" inputs
- $V(t)$: **Membrane potential** (the state)
- $E_L$: **Resting potential** (often normalized to 0)
- $I(t)$: **Synaptic input current**
- $R_m$: **Membrane resistance**

#### Discrete-Time Formulation

When discretized for digital simulation using the Forward Euler method with timestep $\Delta t$:

$$V[t] = V[t-1] + \frac{\Delta t}{\tau_m}(-V[t-1] + I[t])$$

Rearranging and defining the decay factor $\beta = (1 - \frac{\Delta t}{\tau_m})$ or more accurately $\beta = e^{-\Delta t/\tau_m}$:

$$V[t] = \beta \cdot V[t-1] + (1-\beta) \cdot I[t]$$

This equation describes a **discrete first-order low-pass filter**, where the state acts as a leaky integrator of the input stream [13, 15, 20].

### 2.3 Establishing the Isomorphism

The mathematical mapping between RWKV and LIF dynamics is striking. Comparing the equations:

| **RWKV** | $a_t = (e^{-w}) \cdot a_{t-1} + (e^{k_t}) \cdot v_t$ |
|----------|------------------------------------------------------|
| **LIF**  | $V[t] = (\beta) \cdot V[t-1] + (\text{gain}) \cdot I[t]$ |

**Variable Mapping**:

| Neuromorphic Concept | LIF Variable | RWKV Parameter | Interpretation |
|----------------------|--------------|----------------|----------------|
| Membrane Potential | $V[t]$ | $a_t$ | Hidden state = integrated "voltage" of context |
| Leak/Decay | $\beta = e^{-\Delta t/\tau_m}$ | $e^{-w}$ | $w$ ≈ inverse time constant |
| Synaptic Input | $I[t]$ | $v_t$ | Value vector = sensory signal |
| Synaptic Weight | Gain Factor | $e^{k_t}$ | Key vector = dynamic synaptic efficacy |
| Output | Firing Rate/Spike | $wkv_t$ | Channel activation level |

### 2.4 Theoretical Implications

This derivation establishes that the RWKV Time-Mixing block is effectively a **dense, rate-coded simulation of a bank of LIF neurons**:

- **Dense**: Unlike SNNs, which transmit information via sparse binary spikes (0 or 1) only when $V(t)$ crosses a threshold $V_{th}$, RWKV transmits the full floating-point state at every timestep [10, 11]
- **Rate-Coded**: The magnitude of $wkv_t$ can be interpreted as the instantaneous firing rate of the neuron, rather than a spike train

This "Third Way" architecture solves two primary problems of SNNs for autonomous driving:

1. **Training Difficulty**: SNNs are notoriously hard to train due to the non-differentiable nature of the spike generation function (requires surrogate gradients). RWKV remains fully differentiable, allowing stable training on massive datasets [7, 10]
2. **Information Density**: Rate coding carries more information per timestep than binary coding, which is crucial for high-fidelity control tasks like steering angle prediction

### 2.5 Advanced Dynamics: RWKV-7 and Synaptic Plasticity

The latest RWKV-7 "Goose" architecture introduces **Dynamic State Evolution** using a generalized Delta Rule [6, 15]:

$$S_t = S_{t-1} \cdot \text{diag}(w_t) + v_t^T k_t$$

where $w_t$ is now **data-dependent**, evolving with the input. This advancement mirrors **Short-Term Plasticity (STP)** in biological synapses—specifically mechanisms like Facilitation and Depression. The model can "selectively forget" irrelevant history while "attending" strongly to critical transient events, effectively modulating its own membrane resistance in real-time [6, 16].

---

## 3. Computational Efficiency Analysis

### 3.1 Complexity Comparison

| Feature | Transformer (ViT) | RWKV (Linear) | Implication for AD |
|---------|-------------------|---------------|---------------------|
| Training Complexity | $O(N^2)$ | $O(N)$ | RWKV enables training on longer sequences |
| Inference Time | $O(N)$ with KV Cache | $O(1)$ | RWKV: constant, deterministic latency |
| Memory Usage | $O(N^2)$ Attention Map | $O(1)$ Fixed State | RWKV: "Infinite Context" without VRAM explosion |
| State Representation | Explicit KV Cache (Large) | Compressed Recurrent State (Small) | RWKV minimizes memory bandwidth |
| Hardware Fit | GPU (SIMD) | Neuromorphic/FPGA | RWKV aligns with event-driven architectures |

### 3.2 Practical Efficiency Metrics

Empirical data demonstrates significant advantages:

- **VisualRWKV** achieves **3.98x speed advantage** over Transformer-based models like LLaVA-1.5 while saving over **54% of GPU memory** at 24K token inference length [22]
- **TLS-RWKV** for online action detection achieves SOTA performance while running at **>600 FPS on GPU** and **>200 FPS on CPU** [24]
- As image resolution scales (e.g., to 1024×1024), ViT memory usage explodes while VRWKV scales linearly [2]

### 3.3 The "Infinite Context" Hypothesis

Autonomous driving is a **Non-Markovian process** in partially observable environments. The current observation $O_t$ often fails to contain the complete state necessary for optimal decision-making.

**Context Fragmentation in Transformers**: Sliding-window ViTs create artificial boundaries. Objects occluded for longer than the window duration may be treated as new entities or lost entirely.

**Recurrent Continuity in RWKV**: RWKV acts as a **learned state estimator**, analogous to a Kalman Filter but with deep network expressivity. The state $S_t$ compresses entire history into a fixed-size representation, enabling:

- **Object Permanence**: Maintaining belief that a pedestrian is behind a parked car during occlusion [23]
- **Environmental Persistence**: Remembering variable speed limits posted miles back
- **Goal Maintenance**: Retaining high-level navigation goals while processing low-level control [25, 26]

---

## 4. Neuromorphic Hardware Alignment

### 4.1 Hardware Trends

The future of edge AI increasingly involves specialized **Neuromorphic Hardware** (e.g., Intel Loihi 2, BrainChip Akida) designed for event-driven, recurrent computations at milliwatt-scale power consumption [20, 27, 28, 29].

### 4.2 RWKV Hardware Advantages

RWKV offers several hardware-friendly properties:

1. **Memory Bandwidth Reduction**: Standard Transformers are memory-bandwidth bound due to massive KV cache operations. RWKV eliminates the KV cache entirely, replacing it with a compact recurrent state [18]

2. **Quantization Robustness**: RWKVQuant demonstrates effective quantization to INT8 or INT4 with minimal accuracy loss—critical for integer-only neuromorphic accelerators [23]

3. **MatMul-Free Variants**: Recent "MatMul-free" RWKV models using ternary weights ({-1, 0, 1}) achieve **4x higher throughput** and **10x better energy efficiency** on Intel Loihi 2 compared to Transformers on edge GPUs [30, 31, 32, 33]

4. **Sequential Pipelining**: RWKV's dataflow suits deeply pipelined architectures on FPGAs or neuromorphic cores without requiring global sequence synchronization [3, 30]

5. **Spike Conversion**: The underlying isomorphism to LIF neurons enables conversion to spiking models (SpikeRWKV) for deployment on asynchronous hardware like Loihi [14]

---

## 5. Bridging the Vision Gap: RWKV for Visual Perception

### 5.1 The Raster Scan Problem

Applying sequence models to images requires flattening 2D grids to 1D sequences (e.g., raster scan). This makes vertical neighbors in images ($pixel_{x,y}$ and $pixel_{x,y+1}$) distant in the sequence, challenging local spatial feature capture [34, 35].

### 5.2 Vision-RWKV and Quad-directional Token Shift (Q-Shift)

**Vision-RWKV (VRWKV)** addresses this through the **Q-Shift mechanism** [2]:

For a feature map $X$ of shape $(H, W, C)$, the channel dimension $C$ is divided into four groups, each shifted in a cardinal direction:
- Group 1 (0:C/4): Shift Up
- Group 2 (C/4:C/2): Shift Down
- Group 3 (C/2:3C/4): Shift Left
- Group 4 (3C/4:C): Shift Right

This introduces a **Local Inductive Bias** similar to CNNs, providing an effective 3×3 receptive field at every layer without convolution cost or $O(N^2)$ global attention cost [2, 25].

### 5.3 VisualRWKV Architecture

Typical architectures comprise three stages [34, 36]:

1. **Vision Encoder**: Frozen pre-trained encoder (e.g., CLIP-ViT-L/14, SigLIP, DINOv2) extracting robust frame features
2. **Projection Layer**: Trainable alignment layer between visual features and RWKV input space
3. **RWKV Backbone**: Core model for intra-modal reasoning and cross-modal tasks

| Model Variant | Vision Encoder | Tokenization Strategy |
|---------------|----------------|----------------------|
| VisualRWKV v6 | CLIP-ViT-L/14 | 16×16 pixel patches |
| VisualRWKV v7 | SigLIP/DINOv2/SAM | Multi-scale patches |
| VisualRWKV-HD | Hierarchical RWKV | High-resolution grid split |
| Vision-RWKV | Native RWKV backbone | Patch-based sequential scan |

### 5.4 Multimodal Integration: Sandwich Prompting

**VisualRWKV** extends to Visual Language Models through **Sandwich Prompting** [28]:

`<Instruction> → <Image> → <Instruction>`

This structure primes the recurrent state with instructions before processing the visual stream. In RWKV-7's dynamic state evolution, instruction tokens effectively "program" the decay rates ($w_t$), biasing the model to retain specific visual features while filtering background noise—creating a data-dependent attention mechanism crucial for goal-directed driving [28].

---

## 6. State-of-the-Art Landscape

### 6.1 Foundational RWKV Papers

| Version | Key Innovation | Reference |
|---------|----------------|-----------|
| RWKV-4 | First widely-recognized RNN matching Transformer power | [6, 38] |
| RWKV-5/6 (Eagle/Finch) | Matrix-valued states, dynamic recurrence | [5, 6, 35] |
| RWKV-7 (Goose) | Generalized Delta Rule, dynamic state evolution | [6, 15] |

### 6.2 LADY: Linear Attention for Driving

The most critical precedent is **LADY (Linear Attention for Driving)** [3]—an end-to-end generative model for AD using RWKV-7 modules:

- **Architecture**: Fuses multi-frame Camera and LiDAR features
- **Innovation (LICA)**: Linear Cross-Attention approximation for cross-modal fusion using RWKV mechanism
- **Performance**: State-of-the-art on NAVSIM and Bench2Drive benchmarks with constant computational cost regardless of history length

### 6.3 SpikeGPT and Spiking RWKV

**SpikeGPT** is a landmark implementation of generative language modeling with binary, event-driven spiking activations [10, 11, 21, 39]:

- Achieves linear complexity $O(N)$
- Competitive with non-spiking models
- **20x to 32x fewer operations** on neuromorphic hardware [11, 39, 40]
- Built on RWKV framework due to the mathematical bridge enabling efficient spike transition [10, 11]

### 6.4 RWKV vs. Mamba (State Space Models)

| Aspect | RWKV | Mamba |
|--------|------|-------|
| Training | $O(N)$ | $O(N)$ |
| Inference | $O(1)$ | $O(1)$ |
| Mechanism | Simple exponential decay + WKV | Selective Scan (S6) |
| Hardware Fit | Neuromorphic-friendly | GPU-optimized parallel scan |
| Gating | "Soft" sigmoid gating | "Hard" selection gate |
| Robustness | Potentially more robust to sensor noise | More complex dynamics |

RWKV-7's dynamic state evolution matches Mamba's performance on "Copying" tasks while retaining simpler dynamics [15, 16].

### 6.5 Decision-RWKV for Robotic Control

**Decision-RWKV (DRWKV)** integrates RWKV with the Decision Transformer architecture for offline reinforcement learning [25, 26, 35]:

- Optimizes robotic manipulation and lifelong learning
- Efficiently captures temporal correlations without quadratic overhead
- Superior state change prediction for object grasping compared to LSTM/GRU [23]

---

## 7. Comparative Analysis of Source Reviews

### 7.1 Convergence of Conclusions

Both source reviews arrive at **remarkably consistent conclusions**:

| Conclusion | Review 1 | Review 2 |
|------------|----------|----------|
| RWKV is isomorphic to LIF dynamics | ✓ (Detailed derivation) | ✓ (Detailed derivation) |
| RWKV offers O(1) inference complexity | ✓ | ✓ |
| RWKV as "Third Way" between Transformers and SNNs | ✓ | ✓ |
| Neuromorphic hardware compatibility | ✓ | ✓ |
| Infinite context capability | ✓ | ✓ |
| Q-Shift for vision adaptation | ✓ | ✓ |

### 7.2 Alignment of Findings

**Mathematical Foundations**: Both reviews present identical mathematical derivations for the RWKV-LIF isomorphism, arriving at the same variable mapping and decay factor correspondence ($e^{-w}$ ↔ $\beta$).

**Efficiency Arguments**: Both cite the $O(N^2)$ vs $O(N)$ complexity distinction with consistent implications for autonomous driving.

**Vision Adaptation**: Both identify Q-Shift and VisualRWKV variants as solutions to the raster scan problem.

**Hardware Alignment**: Both emphasize MatMul-free variants, quantization robustness, and neuromorphic compatibility.

### 7.3 Unique Contributions

| Review 1 | Review 2 |
|----------|----------|
| Detailed LADY architecture analysis | Stronger emphasis on SpikeGPT mechanisms |
| Proposed "Neuromorphic-LADY" architecture | More hardware deployment metrics |
| Focus on "Object Permanence" for driving | Emphasis on lifelong learning aspects |
| RWKV-7 synaptic plasticity analogy | Decision-RWKV for reinforcement learning |

### 7.4 Shared References

Both reviews cite many of the same foundational works:
- RWKV Survey and architecture papers [1, 5, 6, 35]
- SpikeGPT publications [10, 11, 21, 39]
- LIF neuron dynamics literature [15, 16, 17, 18, 19]
- VisualRWKV papers [22, 34, 36]
- MatMul-free RWKV research [30, 31, 32, 33]

---

## 8. Critical Assessment and Verification

### 8.1 Verification of Core Claims

**Claim: RWKV achieves O(1) inference complexity**
> **Verified.** The recurrent state formulation (fixed-size $a_t$, $b_t$) does not grow with sequence length, enabling constant-time inference per token. This is well-established in the RWKV literature [5, 6].

**Claim: Mathematical isomorphism between RWKV and LIF**
> **Verified with nuance.** The structural correspondence is accurate—both exhibit exponential decay with gated input integration. However, the isomorphism is to the *sub-threshold* LIF dynamics; the spike threshold nonlinearity is absent in standard RWKV. The term "isomorphism" is thus somewhat loose; "structural analogy" may be more precise.

**Claim: 20-32x fewer operations for SpikeGPT on neuromorphic hardware**
> **Verified.** Published benchmarks from the SpikeGPT paper support these efficiency gains on spike-based computation [11, 39, 40].

**Claim: RWKV-7 matches Mamba on state tracking**
> **Partially verified.** The RWKV-7 paper demonstrates improved state tracking, but direct head-to-head comparisons with Mamba vary by task. The claim is directionally correct but task-dependent [15, 16].

### 8.2 Potential Limitations Not Addressed

Both reviews exhibit some optimism that warrants tempering:

1. **Real-World Validation Gap**: Most claims derive from simulation benchmarks (CARLA, nuScenes, NAVSIM). Real-world deployment validation remains limited.

2. **Neuromorphic Hardware Maturity**: Intel Loihi 2 and BrainChip Akida remain research-stage platforms. Production autonomous vehicles predominantly use NVIDIA hardware with established toolchains.

3. **Training Scale**: RWKV's training advantages at extreme scales (billions of parameters) remain less validated than Transformer counterparts.

4. **Multimodal Fusion Complexity**: The reviews focus on sensor fusion conceptually but provide limited empirical comparison against established fusion architectures (e.g., BEVFusion, TransFusion).

### 8.3 Agreement with Findings

**Overall Assessment**: The core thesis—that RWKV offers a compelling "Third Way" for autonomous driving by combining Transformer trainability with RNN efficiency while maintaining neuromorphic compatibility—is **well-supported** by the evidence presented.

The mathematical bridge to LIF dynamics is **legitimate** and provides strong theoretical justification for the approach. The efficiency advantages are **well-documented** in the literature. The neuromorphic alignment is **promising** but requires further hardware validation.

The research direction is **reasonable and timely** given the constraints of edge deployment for autonomous vehicles.

---

## 9. Narrative Synthesis

### 9.1 Problem Statement

> Current End-to-End Autonomous Driving architectures rely predominantly on Vision Transformers (ViTs) to process multi-modal sensor data. While performant, ViTs incur quadratic computational complexity $O(N^2)$ with respect to temporal sequence length. This imposes a 'Context Horizon' limit, forcing perception systems to discard historical driving context to fit within strict memory and latency constraints of edge computing hardware. Furthermore, the high energy cost of attention mechanisms limits operational range and efficiency of autonomous electric vehicles, creating a barrier to widespread adoption.

### 9.2 Hypothesis

> The Receptance Weighted Key Value (RWKV) architecture, specifically its formulation as a Linear Recurrent Neural Network with dynamic state evolution, offers a theoretically superior alternative to Transformers for vehicle control. By leveraging a linear attention mechanism ($O(N)$ complexity) that is mathematically isomorphic to the Leaky Integrate-and-Fire (LIF) dynamics of neuromorphic systems, an RWKV-based driving policy can maintain an effectively infinite temporal context with constant inference cost. This architecture will outperform sliding-window Transformers in long-horizon trajectory prediction tasks while significantly reducing memory footprint and energy consumption on edge hardware.

### 9.3 Proposed Architecture Direction

A "Neuromorphic-RWKV" driving architecture should incorporate:

1. **Backbone**: Vision-RWKV encoder with Q-Shift for Camera and LiDAR streams
2. **Fusion**: Linear Cross-Attention (LICA) for multimodal state integration
3. **Controller**: RWKV policy head outputting control signals directly from fused state
4. **Hardware Mapping**: Explicit mapping of decay parameters ($w$) to SNN time constants ($\tau$) for neuromorphic deployment

### 9.4 Recommended Research Milestones

1. **Simulation Benchmarking**: Implement Decision-RWKV on CARLA simulator comparing against Decision Transformers for success rate and context retention
2. **Hardware Mapping**: Comparative analysis of MatMul-free RWKV on NVIDIA Jetson Orin Nano vs. simulated neuromorphic mesh measuring latency and energy consumption
3. **Visual Integration**: Test VisualRWKV-HD for multi-camera processing evaluating high-resolution perception without frame-dropping

---

## 10. Conclusion

This consolidated review demonstrates that both source analyses arrive at consistent, well-supported conclusions regarding the potential of RWKV for autonomous driving applications. The mathematical bridge between RWKV's time-mixing mechanism and Leaky Integrate-and-Fire neuron dynamics provides strong theoretical justification for the "Third Way" paradigm. The efficiency advantages—linear training complexity, constant inference cost, and neuromorphic compatibility—address real constraints in edge deployment for autonomous vehicles.

RWKV is not merely an "efficient Transformer" but a **convergent architecture** unifying the trainability of deep learning with the continuous, sparse, and stateful dynamics of biological intelligence. For autonomous driving—a task that is inherently continuous, temporal, and resource-constrained—RWKV represents a compelling paradigm shift from "Stateless Frame Processing" to "Stateful Continuous Perception."

---

## References

[1] A Survey of RWKV. arXiv:2412.14847, 2024. https://arxiv.org/abs/2412.14847

[2] Vision-RWKV (VRWKV): A Vision Transformer with Quad-directional Token Shift. arXiv, 2024.

[3] LADY: Linear Attention for Driving. NAVSIM/Bench2Drive benchmarks, 2024.

[4] Thus Spake Long-Context Large Language Model. arXiv:2502.17129, 2025.

[5] RWKV Architecture History. RWKV Wiki. https://wiki.rwkv.com/basic/architecture.html

[6] RWKV Architecture: Combining the Power of Transformers and the Efficiency of Recurrent Neural Networks. https://deepfa.ir/en/blog/rwkv-architecture-rnn-transformer-hybrid

[7] Computational complexity of RWKV vs Transformer. ResearchGate, 2024.

[8] A comparative review of deep and spiking neural networks for edge AI neuromorphic circuits. PMC, 2024.

[9] Learnable Sparsification of Die-to-Die Communication via Spike-Based Encoding. arXiv:2501.08645, 2025.

[10] SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks. arXiv:2302.13939, 2023.

[11] SpikeGPT. OpenReview, ICLR 2023. https://openreview.net/forum?id=gcf1anBL9e

[12] SpikeLORA: Learnable Activation Sparsity for Low-Rank Adaptation Using Spiking Neural Networks. OpenReview, 2024.

[13] Decision SpikeFormer: Spike-Driven Transformer for Decision Making. CVPR 2025.

[14] SpikeRWKV: Spiking Variant of RWKV for Neuromorphic Deployment.

[15] RWKV-7 "Goose" Architecture with Generalized Delta Rule. 2025.

[16] Mamba vs RWKV Comparison Studies. Various sources, 2024-2025.

[17] Sharing leaky-integrate-and-fire neurons for memory-efficient spiking neural networks. Frontiers in Neuroscience, 2023.

[18] HFRWKV: A High-Performance Fully On-Chip Hardware Accelerator for RWKV. arXiv:2601.02135, 2026.

[19] Tutorial 1: The Leaky Integrate-and-Fire (LIF) Neuron Model. Neuromatch Academy. https://compneuro.neuromatch.io/tutorials/W2D3_BiologicalNeuronModels/student/W2D3_Tutorial1.html

[20] LMUFormer: Low Complexity Yet Powerful Spiking Model with Legendre Memory Units. ICLR 2024.

[21] SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks. arXiv PDF, 2023.

[22] VisualRWKV Performance Benchmarks. COLING 2025.

[23] Robotic control optimization based on receptance-weighted reinforcement learning. SPIE Digital Library, 2024.

[24] TLS-RWKV for Online Action Detection. ResearchGate, 2024.

[25] Optimizing Robotic Manipulation with Decision-RWKV: A Recurrent Sequence Modeling Approach for Lifelong Learning. ResearchGate, 2024.

[26] Optimizing Robotic Manipulation With Decision-RWKV. ASME Digital Collection, 2024.

[27] Top Neuromorphic Chips in 2025: BrainChip Akida, Intel Loihi & IBM TrueNorth. ElProCus, 2025.

[28] VisualRWKV Sandwich Prompting for Multimodal Integration.

[29] BrainChip Akida White Paper: A Game Changer in AI Computing. BrainChip, 2025.

[30] MatMul-free LLM Implementation. GitHub: ridgerchu/matmulfreellm.

[31] Scalable MatMul-free Language Modeling. arXiv:2406.02528, 2024.

[32] MatMul-free Language Modeling Benchmarks. arXiv PDF, 2024.

[33] MMfreeLM-370M Performance. Skywork AI, 2024.

[34] RWKV-CLIP: A Robust Vision-Language Representation Learner. ResearchGate, 2024.

[35] MLGroupJLU/RWKV-Survey. GitHub. https://github.com/MLGroupJLU/RWKV-Survey

[36] VisualRWKV Implementation. GitHub: howard-hou/VisualRWKV. https://github.com/howard-hou/VisualRWKV

[37] VisualRWKV-HD for High-Resolution Processing. arXiv, 2024.

[38] Original RWKV Papers by Bo Peng (BlinkDL), 2023-2024.

[39] SpikeGPT Neuromorphic Efficiency Benchmarks. ResearchGate, 2023.

[40] SpikeGPT Operations Reduction Analysis. 2023.

[41] SpikingBERT and SpikeBERT Studies. Various sources, 2023-2024.

---

## Appendix A: Summary Comparison Table

| Aspect | Review 1 | Review 2 | Consensus |
|--------|----------|----------|-----------|
| Core Architecture Focus | RWKV-4/5/6/7 | RWKV-4/5/6/7 | ✓ Aligned |
| Mathematical Bridge | RWKV ↔ LIF isomorphism | RWKV ↔ LIF isomorphism | ✓ Aligned |
| Complexity Analysis | O(N²) → O(N)/O(1) | O(N²) → O(N)/O(1) | ✓ Aligned |
| Vision Adaptation | Q-Shift, VRWKV | Q-Shift, VisualRWKV-HD | ✓ Aligned |
| Neuromorphic Path | SpikeRWKV, Loihi | SpikeGPT, Akida | ✓ Aligned |
| Primary Application | Vehicle Control | Embodied AI | ✓ Compatible |
| Key Precedent | LADY | Decision-RWKV | Complementary |

---

*Document consolidated on January 18, 2026*
