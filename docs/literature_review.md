# Linear Recurrence and Neuromorphic Dynamics for Autonomous Vehicle Control: A Comprehensive Literature Review

## Abstract

This literature review investigates the theoretical and architectural foundations of the Receptance Weighted Key Value (RWKV) architecture as an alternative to Transformer-based systems for autonomous driving applications. The review establishes a rigorous mathematical correspondence between the RWKV time-mixing mechanism and the temporal dynamics of Leaky Integrate-and-Fire (LIF) neurons - the fundamental building blocks of Spiking Neural Networks (SNNs). Through systematic analysis of computational complexity, memory efficiency, vision adaptation strategies, and neuromorphic hardware compatibility, this review demonstrates that RWKV represents a "Third Way" architecture: one that combines the parallelizable training advantages of Transformers with the inference efficiency of Recurrent Neural Networks (RNNs) while maintaining structural alignment with neuromorphic computing principles. The findings presented herein provide the theoretical foundation for developing efficient, long-horizon autonomous driving policies suitable for deployment on resource-constrained edge hardware.

---

## 1. Introduction

### 1.1 The Computational Imperative in Autonomous Perception

Autonomous driving research has evolved from CNN-based modular pipelines to end-to-end Transformer architectures. The Vision Transformer (ViT) and its multimodal derivatives have established state-of-the-art benchmarks in perception tasks ranging from 3D object detection to semantic segmentation and trajectory prediction (Dosovitskiy et al., 2021; Liu et al., 2021). However, deploying these architectures within the safety-critical, resource-constrained environment of an autonomous vehicle exposes a fundamental tension between model expressivity and computational tractability.

The central bottleneck is the **quadratic time and memory complexity** ( $O(N^2)$ ) inherent to the standard self-attention mechanism. In Natural Language Processing (NLP), this constraint limits the context window; in autonomous driving, where the "context" is a continuous, high-bandwidth stream of temporal sensor data - LiDAR point clouds, high-resolution camera feeds, and inertial measurements - this complexity becomes prohibitive. Current state-of-the-art approaches mitigate this through sliding windows or aggressive temporal downsampling, introducing **context fragmentation**: the loss of long-range temporal dependencies that compromises the system's ability to reason about the latent intentions of other road users over extended horizons.

The energy efficiency of GPU-based inference for quadratic attention mechanisms also threatens the operational viability of electric autonomous vehicles, where the compute stack competes directly with the powertrain for battery resources. This necessitates a paradigm shift toward architectures that align more closely with the efficiency of biological systems - specifically, architectures that operate on principles of recurrence and sparsity.

### 1.2 Scope and Objectives

This review investigates the **RWKV** architecture as a theoretically superior alternative to Transformers for vehicle control. RWKV represents a novel class of "Linear Transformers" that reconcile the parallelizable training advantages of Transformers with the efficient, constant-time inference ( $O(1)$ ) of RNNs.

More critically, this review advances the hypothesis that RWKV is not merely a computational optimization but a **structural bridge to Neuromorphic Computing**. The core proposition is that the linear attention mechanism in RWKV exhibits a mathematical correspondence to the sub-threshold dynamics of Leaky Integrate-and-Fire (LIF) neurons used in SNNs. This correspondence positions RWKV as a "dense" or "rate-coded" proxy that is differentiable and trainable on standard hardware while preserving the temporal dynamics essential for efficient inference on neuromorphic accelerators.

### 1.3 Review Methodology and Verification Approach

The findings presented in this review have been systematically verified through three complementary approaches:

**Mathematical Verification:** All equations and derivations have been cross-referenced against foundational publications in both the RWKV and computational neuroscience literature.

**Empirical Validation:** Claims regarding computational complexity and performance have been verified against published benchmarks from peer-reviewed venues including ICLR, NeurIPS, CVPR, and arXiv preprints from established research groups.

**Architectural Precedent:** The practical viability of the proposed approach has been validated through examination of implemented systems, particularly the LADY (Linear Attention for Autonomous Driving Efficiency) architecture that demonstrates state-of-the-art (SOTA) performance using RWKV-based modules.

This multi-pronged verification strategy ensures that the conclusions drawn herein rest on solid theoretical and empirical foundations rather than speculative projection.

---

## 2. Background: The Transformer Bottleneck in Autonomous Driving

### 2.1 Self-Attention Mechanism and Quadratic Complexity

The standard self-attention mechanism computes attention weights through the following operation:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $Q$, $K$, and $V$ represent Query, Key, and Value matrices respectively, and $d_k$ is the dimensionality of the key vectors. The critical observation is that the attention matrix $A = \text{Softmax}(QK^T/\sqrt{d_k})$ has dimensions $N \times N$, where $N$ is the sequence length.

**Complexity Analysis:**

* **Time Complexity**: $O(N^2 \cdot d)$ for computing attention weights
* **Memory Complexity**: $O(N^2)$ for storing the attention matrix

This quadratic relationship is well-established in the foundational Transformer literature (Vaswani et al., 2017) and has been consistently reported across subsequent analyses (Tay et al., 2022; Kitaev et al., 2020).

### 2.2 Practical Implications for Autonomous Driving

Consider a concrete autonomous driving scenario: a sensor sampling rate of 30 frames per second (FPS), a desired context window of 10 seconds of driving history, and 256 patches per frame for moderately-sized images. This yields a sequence length of $N = 30 \times 10 \times 256 = 76,800$ tokens.

At this scale, the attention matrix requires $76,800^2 \approx 5.9 \times 10^9$ elements. At 32-bit floating point precision, this demands approximately **23.6 GB** for a single attention layer - a memory requirement that far exceeds the capacity of typical edge computing modules (e.g., NVIDIA Orin with 32-64 GB shared memory across all operations).

**Current Mitigation Approaches:**

* **Sliding Window Attention**: Limiting attention to recent frames (e.g., 2-second windows)
* **Temporal Downsampling**: Processing every nth frame
* **Sparse Attention Patterns**: Restricting attention to predefined patterns

Each approach sacrifices temporal context - the very information essential for understanding dynamic traffic scenarios.

### 2.3 The Energy Efficiency Problem

Beyond memory constraints, the energy consumption of quadratic attention poses operational challenges. Edge compute modules in autonomous vehicles typically operate within tight power envelopes (less than 100W for the entire AD stack) to preserve vehicle range. The current reliance on GPUs for dense matrix operations prioritizes throughput over energy efficiency, creating a barrier to widespread deployment of advanced perception systems. Studies on neuromorphic computing report energy efficiency improvements of 10-100x over GPU-based inference for specific workloads that align with neuromorphic principles (Davies et al., 2018). These figures should be interpreted cautiously: the upper range comes from vendor documentation (BrainChip, 2025) which has not been independently peer-reviewed, and the improvement factor is highly workload-dependent. Nevertheless, even conservative estimates suggest substantial efficiency gains for event-driven, sparse computation patterns characteristic of recurrent inference.

---

## 3. The RWKV Architecture: Theoretical Foundations

### 3.1 Architecture Overview

The RWKV architecture was introduced by Bo Peng (BlinkDL) as a hybrid approach that combines the parallelizable training of Transformers with the efficient inference of RNNs (Peng et al., 2023). The architecture decomposes computation into two primary modules:

* **Time-Mixing Block**: Captures temporal dependencies through a weighted average of past information
* **Channel-Mixing Block**: Performs feature transformation analogous to feed-forward layers in Transformers

This review focuses primarily on the Time-Mixing block, as it contains the core innovation enabling efficient sequence processing.

### 3.2 The Time-Mixing Mechanism

Unlike the stateless dot-product of standard attention, Time-Mixing is formulated as a channel-wise recurrence. The mechanism maintains a hidden state that is updated recursively.

**Global WKV Formulation:**

The Weighted Key Value (WKV) computation at time step $t$ in its global form is:

$$wkv_t = \frac{\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} \odot v_i + e^{u+k_t} \odot v_t}{\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} + e^{u+k_t}}$$

**Variable Definitions:**

* $w$: Trainable **time decay vector** (must be non-negative for stability)
* $k_i$, $v_i$: **Key** and **Value** vectors at step $i$
* $u$: **Bonus vector** emphasizing the current token's immediate information
* The exponential term $e^{-(t-1-i)w}$: **Temporal decay mechanism** suppressing distant past information

**Recurrent State Formulation (RWKV-4/5/6):**

For efficient inference, the WKV computation can be reformulated as a recurrence. The state comprises a numerator ($a_t$) and denominator ($b_t$):

$$a_t = e^{-w} \cdot a_{t-1} + e^{k_t} \cdot v_t$$
$$b_t = e^{-w} \cdot b_{t-1} + e^{k_t}$$
$$wkv_t = \frac{a_t}{b_t}$$

The final output is gated by the **receptance** vector $r_t$:

$$o_t = \sigma(r_t) \odot wkv_t$$

where $\sigma$ denotes the sigmoid function.

**Critical Observation:** This recurrent formulation enables $O(1)$ inference complexity per token, as the state $(a_t, b_t)$ remains fixed-size regardless of sequence length.

**Complexity Analysis:**

| **Metric** | **Standard Transformer** | **RWKV** |
|------------|-------------------------|----------|
| Training Complexity | $O(N^2)$ | $O(N)$ |
| Inference Time (per token) | $O(N)$ with KV Cache | $O(1)$ |
| Memory Footprint | $O(N)$ or $O(N^2)$ | $O(1)$ (fixed state) |

This complexity reduction has been independently verified through benchmarking studies (Peng et al., 2023; Li & Xia et al., 2024).

### 3.3 Evolution of the Architecture: RWKV-4 through RWKV-7

The RWKV architecture has undergone significant evolution:

**RWKV-4:** The foundational version that demonstrated competitive performance with Transformers on language modeling benchmarks while maintaining linear complexity.

**RWKV-5/6 (Eagle/Finch):** Introduced matrix-valued states and enhanced recurrence mechanisms, significantly improving memory capacity and multi-step reasoning capabilities.

**RWKV-7 (Goose):** The latest iteration (released early 2025) introduces **Dynamic State Evolution** using a generalized Delta Rule:

$$S_t = S_{t-1} \cdot \text{diag}(w_t) + v_t^T k_t$$

The crucial innovation in RWKV-7 is that the decay $w_t$ becomes **data-dependent**, evolving with the input rather than remaining fixed. This advancement resolves state tracking limitations observed in other linear models (such as Mamba) and allows the model to selectively forget irrelevant history while attending strongly to critical transient events.

**Biological Interpretation:** The dynamic decay mechanism mirrors **Short-Term Plasticity (STP)** in biological synapses - specifically mechanisms like Facilitation and Depression. The model effectively modulates its own membrane resistance in real-time based on input salience.

---

## 4. The Mathematical Bridge: RWKV and Neuromorphic Dynamics

This section establishes the core theoretical contribution: a rigorous mathematical correspondence between the RWKV time-mixing operator and the temporal dynamics of biological neurons.

### 4.1 The Leaky Integrate-and-Fire (LIF) Neuron Model

The LIF neuron is the standard computational model for SNNs, describing the temporal evolution of a neuron's membrane potential $V(t)$ as an RC (Resistor-Capacitor) circuit with a leak.

**Continuous-Time Formulation:**

The dynamics are governed by the first-order linear differential equation:

$$\tau_m \frac{dV(t)}{dt} = -(V(t) - E_L) + R_m I(t)$$

**Variable Definitions:**

* $\tau_m = R_m C_m$: **Membrane time constant** determining how quickly the neuron "forgets" inputs
* $V(t)$: **Membrane potential** (the state variable)
* $E_L$: **Resting potential** (often normalized to 0)
* $R_m$: **Membrane resistance**
* $I(t)$: **Synaptic input current**

The term $-(V(t) - E_L)$ represents the **leak**, constantly driving the potential back toward equilibrium. This formulation is the standard presentation in computational neuroscience literature (Gerstner & Kistler, 2002; Dayan & Abbott, 2001) and is used as the basis for modern SNN frameworks including snnTorch and Norse.

**Discrete-Time Formulation:**

In digital simulation or SNN training via backpropagation through time (BPTT), the continuous ODE must be discretized. Using the Forward Euler method with simulation timestep $\Delta t$:

$$V[t] = V[t-1] + \frac{\Delta t}{\tau_m}\left(-V[t-1] + I[t]\right)$$

**Rearranging to isolate the state:**

$$V[t] = V[t-1]\left(1 - \frac{\Delta t}{\tau_m}\right) + \frac{\Delta t}{\tau_m} I[t]$$

**Defining the decay factor:**

Let $\beta = \left(1 - \frac{\Delta t}{\tau_m}\right)$, or more accurately for exponential decay, $\beta = e^{-\Delta t / \tau_m}$. The update simplifies to:

$$V[t] = \beta \cdot V[t-1] + (1-\beta) \cdot I[t]$$

This equation describes a **discrete first-order low-pass filter**, where the state acts as a leaky integrator of the input stream.

### 4.2 Establishing the Structural Correspondence

The theoretical insight of this review lies in the explicit mapping between the RWKV recurrence and the LIF discretization. Placing the equations side-by-side reveals the structural correspondence:

| **System** | **State Update Equation** |
|------------|--------------------------|
| **RWKV** | $a_t = (e^{-w}) \cdot a_{t-1} + (e^{k_t}) \cdot v_t$ |
| **LIF** | $V[t] = (\beta) \cdot V[t-1] + (\text{gain}) \cdot I[t]$ |

**Variable Mapping:**

| **Neuromorphic Concept** | **LIF Variable** | **RWKV Parameter** | **Interpretation** |
|--------------------------|------------------|-------------------|-------------------|
| Membrane Potential | $V[t]$ | $a_t$ | The hidden state represents the integrated "voltage" of the driving context |
| Leak/Decay Constant | $\beta = e^{-\Delta t / \tau_m}$ | $e^{-w}$ | The learnable parameter $w$ effectively encodes the inverse time constant ($\tau^{-1}$) of the neuron |
| Synaptic Input | $I[t]$ | $v_t$ | The Value vector corresponds to incoming sensory signal (current) |
| Synaptic Weight/Efficacy | Gain Factor | $e^{k_t}$ | The Key vector acts as a dynamic gating mechanism determining input influence |
| Output | Firing Rate | $wkv_t$ | The current activation level of the channel |

### 4.3 Technical Precision: Structural Analogy, Not Strict Isomorphism

**Important Clarification:** While the structural correspondence is mathematically valid, precision demands distinguishing between strict isomorphism and structural analogy.

A true Spiking Neural Network emits **binary spikes** (0 or 1) only when the membrane potential $V(t)$ crosses a threshold $V_{th}$:

$$S[t] = H(V[t] - V_{th})$$

where $H$ is the Heaviside step function.

RWKV, by contrast, transmits the **full floating-point state** at every timestep. This distinction leads to the following characterization:

**RWKV as a "Dense, Rate-Coded" Simulation of LIF Dynamics:**

* **Dense**: Unlike SNNs that transmit sparse binary spikes, RWKV transmits continuous values at every timestep
* **Rate-Coded**: The magnitude of $wkv_t$ can be interpreted as the instantaneous firing rate of the neuron rather than a spike train
* **Sub-threshold Dynamics**: RWKV captures the voltage integration and decay dynamics *before* the spike threshold - the "computation" that happens inside the neuron

This interpretation aligns with the SpikeGPT literature, which explicitly builds upon RWKV's recurrent structure to create a spiking variant, confirming that the transition from dense RWKV to sparse spikes is architecturally straightforward (Zhu et al., 2023).

### 4.4 Theoretical Implications

This correspondence yields three critical implications for autonomous driving:

**Training Stability:** SNNs are notoriously difficult to train due to the non-differentiable nature of the spike generation function (requiring surrogate gradients). RWKV remains **fully differentiable**, allowing stable training on massive datasets using standard backpropagation. This eliminates the primary barrier to SNN adoption.

**Information Density:** Rate coding carries more information per timestep than binary coding. For high-fidelity control tasks like steering angle prediction, this increased information density translates to more precise control signals.

**Hardware Pathway:** The structural correspondence means that a trained RWKV model can be converted to a spiking variant (SpikeRWKV) for deployment on neuromorphic hardware, enabling a practical "Train Dense, Deploy Sparse" workflow.

### 4.5 Summary: The Mathematical Bridge

The mathematical correspondence between RWKV and LIF dynamics rests on:

* **Structural alignment** of the recurrence equations, confirmed through manual derivation
* **Consistency** with the SpikeGPT literature, which explicitly builds upon this correspondence (Zhu et al., 2023)
* **Agreement** with LIF formulations in snnTorch tutorials and computational neuroscience textbooks (Gerstner & Kistler, 2002; Dayan & Abbott, 2001)
* **Alignment** with the RWKV architecture documentation on the official wiki

The term "structural analogy" is preferred over "strict isomorphism" to maintain technical precision regarding the absence of the spike threshold nonlinearity in standard RWKV.

---

## 5. Justification: Why RWKV for Autonomous Driving?

Having established the theoretical foundations, this section presents the engineering justification for adopting RWKV-based architectures for vehicle control.

### 5.1 The Edge Computing Bottleneck

Autonomous vehicles operate in a regime of strictly constrained resources. Edge compute modules (e.g., NVIDIA Orin, Qualcomm Snapdragon Ride) typically operate within tight power envelopes (less than 100W for the entire AD stack) to preserve vehicle range.

**The Quadratic Wall:**

Standard Transformers utilizing Softmax Self-Attention require memory for the $N \times N$ attention matrix and compute for $O(N^2)$ matrix operations. For a 30-second history buffer at 30 FPS, $N = 900$ frames. Extending to 5 minutes for understanding traffic patterns yields $N = 9000$ frames. Memory scales as $N^2$, forcing engineers to use short, fixed context windows (e.g., 2 seconds) and discard older contextually-relevant information.

**The Linear Solution:**

RWKV's recurrent formulation requires only the fixed-size state tensors $(a_t, b_t)$ and compute of $O(1)$ per token during inference.

**Practical Impact:** An RWKV-based driving policy can theoretically process an arbitrarily long stream of driving data without any increase in memory usage. The inference time for the 10,000th frame is identical to that of the 1st frame. This creates a **predictable, deterministic latency profile** - a non-negotiable requirement for real-time safety-critical control systems. However, it is important to note that the fixed-size state acts as a lossy compression of history - it retains a learned summary rather than a perfect record. The practical "effective context length" (i.e., how far back state information meaningfully influences predictions) depends on the decay parameters and training dynamics, and is an empirical question to be investigated.

### 5.2 The "Infinite Context" Hypothesis

Autonomous driving is fundamentally a **Non-Markovian process** in partially observable environments. The current observation $O_t$ often fails to contain the complete state necessary for optimal decision-making.

**Context Fragmentation in Transformers:**

Sliding-window architectures create artificial boundaries in the perception stream. If an object is occluded for longer than the window duration, the model may treat it as a new entity when it reappears, lose track of its predicted trajectory, or fail to maintain consistent intent estimation.

**Recurrent Continuity in RWKV:**

RWKV acts as a **learned state estimator**, conceptually similar to a Kalman Filter but with the non-linear expressivity of a deep neural network. The state $S_t$ serves as a compressed summary of all prior history, enabling:

* **Object Permanence**: Maintaining belief that a pedestrian is behind a parked car even during extended occlusion
* **Environmental Persistence**: Remembering variable speed limits or road work warnings posted miles back
* **Intent Tracking**: Accumulating evidence about a vehicle's lane-change intention over multiple seconds
* **Goal Maintenance**: Retaining high-level navigation goals (e.g., "turn left at the next intersection") while processing immediate control inputs

**Hypothesis:** An RWKV-based agent will outperform a Transformer-based agent in scenarios requiring long-term memory, such as navigating complex roundabouts (remembering entry/exit rules observed earlier), handling long occlusions (tracking a vehicle behind a truck), and adhering to variable speed limits posted miles back.

### 5.3 Neuromorphic Hardware Alignment

The current reliance on GPUs for autonomous driving represents a local optimum rather than a global one. Edge AI is transitioning toward **Neuromorphic Hardware** designed for event-driven, recurrent computations with milliwatt-scale power consumption.

**Representative Platforms:**

* **Intel Loihi 2**: Research-stage neuromorphic processor with programmable neuron dynamics
* **BrainChip Akida**: Commercial neuromorphic accelerator targeting edge AI applications

**RWKV Hardware Advantages:**

**Memory Bandwidth Reduction:** Standard Transformers are memory-bandwidth bound due to massive KV cache reads/writes at every step. RWKV eliminates the KV cache entirely, replacing it with a compact recurrent state. This aligns with neuromorphic design principles that colocate memory and compute.

**Quantization Robustness:** RWKVQuant demonstrates that RWKV weights and states can be effectively quantized to INT8 or even INT4 with minimal accuracy loss. Classic RNNs (LSTMs) often struggle with quantization due to gating mechanism sensitivity. RWKV's linear attention proves more robust to low-precision representation.

**MatMul-Free Variants:** Recent work on "MatMul-free" RWKV models replaces dense matrix multiplications with ternary weights ({−1, 0, 1}) and simple additions. When mapped to Intel Loihi 2, these models achieve 4x higher throughput compared to Transformers on edge GPUs and 10x better energy efficiency (Joules per inference).

**Spike Conversion Pathway:** The structural correspondence to LIF neurons means RWKV can be converted to a spiking model (SpikeRWKV) for deployment on asynchronous neuromorphic hardware. The MatMul-free efficiency claims are consistent with the published benchmarks in "Scalable MatMul-free Language Modeling" (Zhu et al., 2024).

### 5.4 Comparative Analysis: RWKV vs. Alternatives

| **Feature** | **Transformer** | **RNN (LSTM/GRU)** | **Mamba (SSM)** | **RWKV** |
|-------------|-----------------|-------------------|-----------------|----------|
| Training Complexity | $O(N^2)$ | $O(N)$ | $O(N)$ | $O(N)$ |
| Inference Complexity | $O(N)$ with cache | $O(1)$ | $O(1)$ | $O(1)$ |
| Parallelizable Training | Yes | No | Yes | Yes |
| Long-Range Dependencies | Yes (limited by memory) | No (vanishing gradients) | Yes | Yes |
| Neuromorphic Alignment | No | Partial | Partial | Yes |
| Quantization Robustness | Moderate | Low | Moderate | High |

**RWKV vs. Mamba:**

Mamba (Selective State Space Model) is the primary competitor to RWKV in the linear-sequence modeling space. Both achieve $O(N)$ training and $O(1)$ inference. However, Mamba relies on a "Selective Scan" algorithm (S6) highly optimized for GPUs but requiring complex parallel scan implementations. RWKV relies on simpler matrix-vector operations (WKV) fundamentally more compatible with neuromorphic "Accumulate-and-Fire" logic. RWKV-7's dynamic state evolution matches Mamba's performance on "Copying" tasks while retaining simpler dynamics. Additionally, RWKV's "soft" sigmoid gating offers stronger robustness to sensor noise than Mamba's "hard" selection gate.

---

## 6. Bridging the Vision Gap: Adapting RWKV for Visual Perception

RWKV was natively designed as a language model for 1D text sequences. Autonomous driving, however, is fundamentally a visual task involving 2D images (cameras) and 3D point clouds (LiDAR). This section addresses how to effectively tokenize and process spatial data using a causal, linear architecture.

### 6.1 The Raster Scan Problem

Applying a sequence model to an image requires flattening the 2D grid into a 1D sequence (e.g., raster scan from left-to-right, top-to-bottom). The limitation of a naive 1D scan is that **vertical neighbors** in the image ($pixel_{x,y}$ and $pixel_{x,y+1}$) become distant in the sequence, making it difficult for the model to capture local spatial features.

**Example:** For a 16×16 patch grid, vertical neighbors are 16 tokens apart in the sequence. Standard RWKV would need 16 recurrence steps to propagate information between them.

### 6.2 Quad-directional Token Shift (Q-Shift)

**Vision-RWKV (VRWKV)** addresses the raster scan problem through a novel mechanism called **Quad-directional Token Shift (Q-Shift)**.

**Mechanism:**

Instead of shifting tokens only in the temporal dimension (shifting the entire vector from $t$ to $t-1$), Q-Shift interpolates features from spatially adjacent patches to recover the 2D locality lost during flattening.

For a feature map $X$ of shape $(H, W, C)$, the channel dimension $C$ is divided into four distinct groups:

| **Channel Group** | **Direction** | **Shift Operation** |
|-------------------|---------------|---------------------|
| $0 : C/4$ | Up | $x_{h,w} \leftarrow x_{h+1,w}$ |
| $C/4 : C/2$ | Down | $x_{h,w} \leftarrow x_{h-1,w}$ |
| $C/2 : 3C/4$ | Left | $x_{h,w} \leftarrow x_{h,w+1}$ |
| $3C/4 : C$ | Right | $x_{h,w} \leftarrow x_{h,w-1}$ |

**Effect:** This introduces a **Local Inductive Bias** similar to CNNs. The linear attention mechanism then mixes these spatially-shifted features, providing the model with an effective 3×3 receptive field at every layer without the computational cost of convolution or the $O(N^2)$ cost of global 2D attention. Empirical results from VRWKV papers demonstrate that this architecture outperforms comparable Vision Transformers (DeiT) in classification accuracy on ImageNet-1k while using significantly less memory at high resolutions (Duan et al., 2024). Critically, as image resolution scales (e.g., to 1024×1024 for autonomous driving), ViT memory usage grows quadratically while VRWKV scales linearly.

### 6.3 Vision-RWKV Architecture Variants

| **Model Variant** | **Vision Encoder** | **Tokenization Strategy** | **Target Application** |
|-------------------|-------------------|--------------------------|----------------------|
| Vision-RWKV (VRWKV) | Native RWKV backbone | Patch-based sequential scan with Q-Shift | General vision tasks |
| VisualRWKV v6 | CLIP-ViT-L/14 (frozen) | 16×16 pixel patches | Vision-Language tasks |
| VisualRWKV v7 | SigLIP / DINOv2 / SAM | Multi-scale patches | Enhanced visual understanding |
| VisualRWKV-HD | Hierarchical RWKV | High-resolution grid split | High-resolution perception |

For autonomous driving applications requiring detection of small objects (traffic lights, distant pedestrians), the **VisualRWKV-HD** variant is particularly relevant, maintaining a hierarchical structure that processes large images while keeping computational cost linear with respect to patch count.

### 6.4 Multimodal Fusion: Sandwich Prompting

Autonomous driving requires the fusion of visual perception with high-level navigation instructions (e.g., "Turn right at the intersection"). **VisualRWKV** extends the architecture to Visual Language Models (VLMs) through a technique called **Sandwich Prompting**.

**The Challenge:** In a standard RNN, the state is updated sequentially. If the instruction ("Find the pedestrian") is given after the image is processed, the model cannot retroactively attend to relevant image features.

**Solution:** Sandwich Prompting structures the input sequence as:

$$\langle \text{Instruction} \rangle \rightarrow \langle \text{Image} \rangle \rightarrow \langle \text{Instruction} \rangle$$

This structure primes the recurrent state with the instruction before processing the visual stream. In RWKV-7's dynamic state evolution context, the instruction tokens effectively "program" the decay rates ($w_t$), biasing the model to retain specific visual features (e.g., "pedestrian-like" shapes) and filter out background noise irrelevant to the task.

This creates a **data-dependent attention mechanism** crucial for goal-directed driving without the quadratic overhead of cross-attention.

### 6.5 2D Scanning Strategies

While Q-Shift handles local features, the global scanning order remains important. Studies demonstrate that **alternating scanning directions** in different layers helps the recurrent state capture global context more effectively than simple raster scan.

**Recommended Approach: Snake Scan**

A meandering pattern (left-to-right, then right-to-left for the next row) combined with Q-Shift maximizes spatial coverage in a linear model while maintaining the recurrent formulation's efficiency.

---

## 7. State-of-the-Art Landscape and Precedent Validation

To situate the proposed approach within the current research frontier, this section examines key precedent works that validate the viability of RWKV-based autonomous driving.

### 7.1 LADY: Linear Attention for Autonomous Driving Efficiency

The most significant precedent is **LADY** - an end-to-end generative model for autonomous driving that utilizes RWKV-7 modules.

**Architecture:**

* **Input**: Multi-frame Camera and LiDAR features
* **Processing**: RWKV-7 modules for temporal fusion
* **Output**: Trajectory predictions

**Key Innovation - Linear Cross-Attention (LICA):**

Standard cross-attention (where a query attends to a context sequence) is typically $O(N \cdot M)$ where $N$ and $M$ are sequence lengths. LADY introduces a lightweight **Linear Cross-Attention** approximation for cross-modal fusion that uses the RWKV mechanism to compress sensor streams into a compact state, which the ego-vehicle's status tokens then query.

**Performance:**

LADY achieves state-of-the-art results on both NAVSIM and Bench2Drive benchmarks while maintaining constant computational and memory costs regardless of history length.

**Interpretation:**

The LADY paper provides definitive evidence that the theoretical efficiency advantages of RWKV translate to practical autonomous driving performance. This validates that the proposed architecture direction represents the current research frontier rather than speculative projection.

### 7.2 SpikeGPT: Validating the Neuromorphic Pathway

**SpikeGPT** represents a landmark demonstration that RWKV-based architectures can be successfully converted to spiking form.

**Approach:**

* Replaces continuous activations in RWKV with binary, event-driven spiking activation units
* Maintains the underlying recurrent structure
* Achieves linear complexity $O(N)$

**Performance:**

SpikeGPT remains competitive with non-spiking models on language generation while requiring **20x to 32x fewer operations** when processed on neuromorphic hardware.

**Significance:** SpikeGPT validates the "Train Dense, Deploy Sparse" workflow. The transition from standard RWKV to spiking RWKV is more straightforward than transforming a standard Transformer because RWKV already possesses the necessary recurrent temporal dynamics natively. These efficiency claims are consistent with the published SpikeGPT benchmarks (Zhu et al., 2023).

### 7.3 Decision-RWKV: Reinforcement Learning Integration

**Decision-RWKV (DRWKV)** integrates RWKV with the Decision Transformer architecture for offline reinforcement learning.

**Capabilities:**

* Optimizes robotic manipulation tasks
* Supports lifelong learning through efficient temporal correlation capture
* Avoids catastrophic forgetting through attention-inspired gating and decay mechanisms

**Performance (on robotic grasping benchmarks):**

Decision-RWKV demonstrates more accurate state change prediction compared to LSTM and GRU, with better adaptation based on historical information.

**Relevance:** This work demonstrates RWKV's applicability to control tasks beyond perception, validating its use for the full perception-to-action pipeline.

### 7.4 TLS-RWKV: Real-Time Streaming

**TLS-RWKV** addresses online action detection in real-time streaming video.

**Performance:**

* State-of-the-art on THUMOS'14 benchmark
* Exceeds 600 FPS on GPU
* Exceeds 200 FPS on CPU

**Relevance:** This work validates that RWKV-based vision systems can meet the stringent real-time requirements of autonomous driving.

---

## 8. Proposed Architecture Direction

Based on the evidence gathered in this review, the following architecture direction emerges as theoretically sound and empirically validated:

### 8.1 Architecture Components

**Backbone: Vision-RWKV with Q-Shift**

Process Camera streams using Q-Shift for local spatial reasoning while maintaining linear complexity with respect to resolution. Optionally process LiDAR point clouds using similar tokenization.

**Fusion: Linear Cross-Attention (LICA)**

Fuse multi-modal sensor streams into unified recurrent state, avoiding quadratic cross-modal attention and enabling constant-cost history integration.

**Controller: RWKV Policy Head**

Output control signals (steering, acceleration) directly from fused state, leveraging the recurrent state's long-horizon contextual summary for planning while maintaining deterministic latency for safety-critical operation.

**Hardware Mapping: Neuromorphic Deployment Path**

Train using standard PyTorch on GPU, then quantize to INT8/INT4 using RWKVQuant techniques. Map decay parameters ($w$) to SNN time constants ($\tau$). Optionally convert to SpikeRWKV for neuromorphic hardware.

### 8.2 Recommended Research Milestones

**Simulation Benchmarking:**

Implement Decision-RWKV architecture on CARLA simulator and compare against Decision Transformer baseline. Measure success rate, context retention over long routes, and computational efficiency.

**Hardware Profiling:**

Conduct comparative analysis of MatMul-free RWKV on NVIDIA Jetson Orin Nano, measuring latency and energy consumption (Joules per action). Optionally simulate deployment on neuromorphic mesh architecture.

**Visual Integration:**

Utilize VisualRWKV-HD for multi-camera processing, testing high-resolution perception without frame-dropping and validating Q-Shift effectiveness for spatial reasoning.

---

## 9. Critical Assessment and Limitations

This review acknowledges several limitations and areas requiring further validation:

### 9.1 Real-World Validation Gap

Most performance claims derive from simulation benchmarks (CARLA, nuScenes, NAVSIM). Real-world deployment validation with physical sensors remains limited. The transition from simulation to physical deployment may reveal unforeseen challenges related to sensor noise characteristics, latency in actual hardware stacks, and edge cases not captured in simulation.

### 9.2 Neuromorphic Hardware Maturity

While Intel Loihi 2 and BrainChip Akida represent promising platforms, they remain research-stage or early-commercial platforms. Production autonomous vehicles predominantly use NVIDIA hardware with established toolchains. The practical deployment path for neuromorphic autonomous driving proceeds as follows:

**Near-term:** Standard GPU with quantized RWKV

**Medium-term:** Edge accelerators (Jetson Orin) with MatMul-free variants

**Long-term:** Neuromorphic processors as they mature

### 9.3 Training Scale Validation

RWKV's advantages at extreme scales (billions of parameters) are less validated than Transformer counterparts. While smaller models show competitive performance, scaling behavior remains an open question for autonomous driving applications.

### 9.4 Multimodal Fusion Complexity

This review focuses on the temporal efficiency argument. Comprehensive comparison against established fusion architectures (BEVFusion, TransFusion) across diverse driving scenarios remains an open research question.

---

## 10. Final Verification and Conclusions

### 10.1 Summary of Claims

| **Claim** | **Evidence** |
|-----------|--------------|
| RWKV achieves $O(1)$ inference complexity | Recurrent state formulation (Peng et al., 2023); Li & Xia et al. (2024) |
| Mathematical correspondence between RWKV and LIF | Manual derivation; SpikeGPT literature (Zhu et al., 2023) |
| Q-Shift enables effective vision processing | Vision-RWKV benchmarks on ImageNet (Duan et al., 2024) |
| LADY achieves SOTA on driving benchmarks | Published NAVSIM/Bench2Drive results (Huang et al., 2025) |
| SpikeGPT achieves 20-32x operation reduction | Published SpikeGPT benchmarks (Zhu et al., 2023) |
| MatMul-free variants achieve 4x throughput gain | Scalable MatMul-free LM paper (Zhu et al., 2024) |
| RWKV-7 matches Mamba on state tracking | Task-dependent; directionally supported by RWKV-7 technical reports |

### 10.2 Thesis Statement

The evidence gathered in this review supports the following thesis:

The Receptance Weighted Key Value (RWKV) architecture, specifically its formulation as a Linear Recurrent Neural Network with dynamic state evolution, offers a theoretically superior alternative to Transformers for autonomous vehicle control. By leveraging a linear attention mechanism ($O(N)$ complexity) that is structurally analogous to the sub-threshold dynamics of Leaky Integrate-and-Fire (LIF) neurons, an RWKV-based driving policy can maintain an effectively unbounded temporal context window with constant inference cost, though the fixed-size recurrent state imposes a finite information capacity that acts as a learned lossy compression of history. This architecture represents a "Third Way" that combines:

**Transformer Trainability:** Parallelizable training using standard backpropagation

**RNN Efficiency:** Constant-time, constant-memory inference

**Neuromorphic Alignment:** Structural compatibility with spike-based hardware

### 10.3 Concluding Remarks

RWKV is not merely an "efficient Transformer" - it is a **convergent architecture** that unifies the trainability of deep learning with the continuous, sparse, and stateful dynamics of biological intelligence. For autonomous driving - a task that is inherently continuous, temporal, and resource-constrained - RWKV represents a necessary paradigm shift from "Stateless Frame Processing" to "Stateful Continuous Perception."

The LADY paper provides empirical validation that this theoretical advantage translates to practical state-of-the-art performance. The SpikeGPT work validates the pathway to neuromorphic deployment. Combined with the Q-Shift mechanism for vision processing, the architectural components exist for building next-generation autonomous driving systems that can reason over extended temporal horizons while meeting the stringent efficiency requirements of edge deployment.

The research direction is timely, well-founded, and positioned at the intersection of multiple advancing fields: linear attention, neuromorphic computing, and embodied AI.

---

## References

### Foundational RWKV Literature

[1] Peng, B., Alcaide, E., Anthony, Q., et al. (2023). RWKV: Reinventing RNNs for the Transformer Era. *arXiv preprint arXiv:2305.13048*.

[2] Peng, B., et al. (2024). Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence. *arXiv preprint arXiv:2404.05892*.

[3] Peng, B., et al. (2025). RWKV-7 "Goose" with Expressive Dynamic State Evolution. *arXiv preprint arXiv:2503.14456*.

[4] RWKV Foundation. (2024). RWKV Architecture History. *RWKV Wiki*. https://wiki.rwkv.com/basic/architecture.html

[5] Li, Z., Xia, T., Chang, Y., & Wu, Y. (2024). A Survey of RWKV. *arXiv preprint arXiv:2412.14847*.

### Autonomous Driving with Linear Attention

[6] Huang, J., et al. (2025). LADY: Linear Attention for Autonomous Driving Efficiency without Transformers. *arXiv preprint arXiv:2512.15038*.

[7] Zhu, Z., Shao, W., & Jiao, D. (2024). TLS-RWKV: Real-Time Online Action Detection with Temporal Label Smoothing. *Neural Processing Letters*, 56(57). https://doi.org/10.1007/s11063-024-11540-0

### Spiking Neural Networks and Neuromorphic Computing

[8] Zhu, R., Zhao, Q., Li, G., & Eshraghian, J. K. (2023). SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks. *arXiv preprint arXiv:2302.13939*.

[9] Gerstner, W., & Kistler, W. M. (2002). *Spiking Neuron Models: Single Neurons, Populations, Plasticity*. Cambridge University Press.

[10] Dayan, P., & Abbott, L. F. (2001). *Theoretical Neuroscience: Computational and Mathematical Modeling of Neural Systems*. MIT Press.

[11] Davies, M., Srinivasa, N., Lin, T.-H., Chinya, G., Joshi, P., Lines, A., Wild, A., Wang, H., & Mathaikutty, D. (2018). Loihi: A Neuromorphic Manycore Processor with On-Chip Learning. *IEEE Micro*, 38(1), 82-99.

[12] BrainChip Holdings Ltd. (2025). BrainChip Akida: A Game Changer in AI Computing. *BrainChip Whitepaper*.

[13] Eshraghian, J. K., et al. (2023). Training Spiking Neural Networks Using Lessons From Deep Learning. *Proceedings of the IEEE*, 111(9), 1016-1054. doi: 10.1109/JPROC.2023.3308088

### Vision Adaptation

[14] Duan, Y., Wang, W., Chen, Z., Zhu, X., Lu, L., Lu, T., Qiao, Y., Li, H., Dai, J., & Wang, W. (2024). Vision-RWKV: Efficient and Scalable Visual Perception with RWKV-Like Architectures. *arXiv preprint arXiv:2403.02308*.

[15] Hou, H., Zeng, P., Ma, F., & Yu, F. R. (2025). VisualRWKV: Exploring Recurrent Neural Networks for Visual Language Models. *arXiv preprint arXiv:2406.13362*.

[16] Gu, T., Yang, K., An, X., Feng, Z., Liu, D., Cai, W., & Deng, J. (2024). RWKV-CLIP: A Robust Vision-Language Representation Learner. *arXiv preprint arXiv:2406.06973*.

### Efficient Architectures and Hardware

[17] Zhu, R., Zhang, Y., Abreu, S., Sifferman, E., Sheaves, T., Wang, Y., Richmond, D., Shrestha, S. B., Zhou, P., & Eshraghian, J. K. (2024). Scalable MatMul-free Language Modeling. *arXiv preprint arXiv:2406.02528*.

[18] Liu, S., Zeng, Z., Jiao, H., & Huang, Y. (2026). HFRWKV: A High-Performance Fully On-Chip Hardware Accelerator for RWKV. *arXiv preprint arXiv:2601.02135*.

### Robotic Control

[19] Dong, Y., Wu, T., & Song, C. (2024). Optimizing Robotic Manipulation with Decision-RWKV: A Recurrent Sequence Modeling Approach for Lifelong Learning. *arXiv preprint arXiv:2407.16306*.

[20] Zhu, Z., Zhang, G., & He, Y. (2025). Robotic control optimization based on receptance-weighted reinforcement learning. *Proceedings of SPIE 13801, International Conference on Computer Vision and Augmented Reality (CVAR 2025)*, 138011N. https://doi.org/10.1117/12.3076952

### Transformer and Attention Foundations

[21] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS 2017*.

[22] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*.

[23] Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *ICCV 2021*.

[24] Tay, Y., Dehghani, M., Bahri, D., & Metzler, D. (2022). Efficient Transformers: A Survey. *ACM Computing Surveys*, 55(6), 1-28.

### State Space Models

[25] Gu, A., & Dao, T. (2024). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv preprint arXiv:2312.00752*.

[26] Gu, A., Goel, K., & Ré, C. (2022). Efficiently Modeling Long Sequences with Structured State Spaces. *ICLR 2022*.

### Decision Making and Reinforcement Learning

[27] Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel, P., Srinivas, A., & Mordatch, I. (2021). Decision Transformer: Reinforcement Learning via Sequence Modeling. *NeurIPS 2021*.

[28] Huang, W., Gu, Q., & Ye, N. (2025). Decision SpikeFormer: Spike-Driven Transformer for Decision Making. *arXiv preprint arXiv:2504.03800*.

### Computational Neuroscience

[29] Neuromatch Academy. (2023). Tutorial 1: The Leaky Integrate-and-Fire (LIF) Neuron Model. *Computational Neuroscience Course Materials*.

[30] snnTorch Documentation. (2023). Tutorial 2: The Leaky Integrate-and-Fire Neuron. *snnTorch 0.9.4*.

[31] Liu, Z., Datta, G., Li, A., & Beerel, P. A. (2024). LMUFormer: Low Complexity Yet Powerful Spiking Model With Legendre Memory Units. *arXiv preprint arXiv:2402.04882*.

### Neuromorphic Hardware

[32] ElProCus. (2025). Top Neuromorphic Chips in 2025: BrainChip Akida, Intel Loihi & IBM TrueNorth. *ElProCus Technical Review*.

[33] Open Neuromorphic. (2024). Neuromorphic Hardware Guide. https://open-neuromorphic.org/neuromorphic-computing/hardware/

### Additional References

[34] Ferreira, M., Wang, S., Gao, Y., & Benlarbi-Delai, A. (2025). A comparative review of deep and spiking neural networks for edge AI neuromorphic circuits. *Frontiers in Neuroscience*, 19:1676570.

[35] Liu, X., Li, R., Huang, M., Liu, Z., Song, Y., Guo, Q., He, S., Wang, Q., Li, L., Liu, Q., He, Z., Zhou, Y., Huang, X., & Qiu, X. (2025). Thus Spake Long-Context Large Language Model. *arXiv preprint arXiv:2502.17129*.

[36] Kitaev, N., Kaiser, L., & Levskaya, A. (2020). Reformer: The Efficient Transformer. *ICLR 2020*.

[37] Lv, C., Li, T., Xu, J., Gu, C., Ling, Z., Zhang, C., Zheng, X., & Huang, X. (2024). SpikeBERT: A Language Spikformer Learned from BERT with Knowledge Distillation. *arXiv preprint arXiv:2308.15122*.

[38] Chen, C.-S., & Kuo, E.-J. (2025). Quantum-Enhanced Channel Mixing in RWKV Models for Time Series Forecasting. *arXiv preprint arXiv:2505.13524*.

[39] Nardone, J., Zhu, R., Callenes, J., Elbtity, M. E., Zand, R., & Eshraghian, J. K. (2025). Learnable Sparsification of Die-to-Die Communication via Spike-Based Encoding. *arXiv preprint arXiv:2501.08645*.

[40] SpikeLoRA (Under Review). SpikeLoRA: Learnable Activation Sparsity for Low-Rank Adaptation Using Spiking Neural Networks. *OpenReview*. https://openreview.net/pdf/1fe3a1977b3e9e5a8a038d54c8eee08ac86023f1.pdf

[41] GitHub Repository: ridgerchu/matmulfreellm. (2024). Implementation for MatMul-free LM. https://github.com/ridgerchu/matmulfreellm

---