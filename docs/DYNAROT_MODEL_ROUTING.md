# ModelFusion DynaRot Hardware-Aware Model Routing & Resource Provisioning

`src/arxiv_to_code/dynarot.py` ports **ModelFusion's DynaRot (Dynamic Router & Hardware Sizing Engine)** into Python, pairing the **Dynamic Available Memory Law**, **4-Objective Pareto Front & Knee-Point Selection**, **UCB1 Multi-Armed Bandit Exploration**, and **Evolutionary Compute Budget Splitting**.

---

## 1. The Dynamic Available Memory Law (Anti-OOM Invariant)

Models are **NEVER** sized against total physical RAM. Sizing against total RAM causes out-of-memory (OOM) crashes when concurrent applications or OS caches are active. `DynaRot` probes **live runtime available free RAM** (`free_ram_gb` via Win32 `GlobalMemoryStatusEx` / `psutil`) and **free GPU VRAM** (`free_vram_mb` via `nvidia-smi`):

| Hardware Tier | Live Available Memory Condition | Primary Model | `ShinkaEvolve` Heavy Model | `OpenEvolve` Fast Mutation Model | Max Ensemble |
|:---|:---|:---:|:---:|:---:|:---:|
| **Tier 1 (Frontier 32B)** | $\text{Free RAM} \ge 48\text{ GB}$ **OR** $\text{Free VRAM} \ge 24\text{ GB}$ ($24{,}000\text{ MB}$) | `qwen2.5:32b` | `qwen2.5:32b` | `qwen2.5:7b` | `7` |
| **Tier 2 (High-Throughput 14B)** | $\text{Free RAM} \ge 24\text{ GB}$ **OR** $\text{Free VRAM} \ge 14\text{ GB}$ ($14{,}000\text{ MB}$) | `qwen2.5:14b` | `qwen2.5:14b` | `qwen2.5:3b` | `5` |
| **Tier 3 (GPU-Resident 7B)** | $\text{Free RAM} \ge 12\text{ GB}$ **OR** $\text{Free VRAM} \ge 4.5\text{ GB}$ ($4{,}500\text{ MB}$) | `qwen2.5:7b` | `qwen2.5:7b` | `qwen2.5:1.5b` | `4` |
| **Tier 4 (Low-Power Edge 3B)** | $\text{Free RAM} \ge 6\text{ GB}$ **OR** $\text{Free VRAM} \ge 2\text{ GB}$ ($2{,}000\text{ MB}$) | `qwen2.5:3b` | `qwen2.5:3b` | `qwen2.5:1.5b` | `3` |
| **Tier 5 (Compact 1.5B)** | $\text{Free RAM} < 6\text{ GB}$ **AND** $\text{Free VRAM} < 2\text{ GB}$ | `qwen2.5:1.5b` | `qwen2.5:1.5b` | `qwen2.5:0.5b` | `2` |

> [!IMPORTANT]
> **Anti-404 Model Tag Sanitization (`sanitize_model_tag`)**: `DynaRot` automatically strips `-coder` suffixes (`qwen2.5-coder:7b` $\to$ `qwen2.5:7b`) so local Ollama pulls and calls never trigger silent 404 aborts.

---

## 2. Multi-Objective Score & UCB1 Bandit Formulation

Each candidate model $i$ in the curated catalog is scored across 5 terms:

$$S_i = w_1 Q_{\text{task}, i} + w_2 \frac{\log_{10}(1 + D_i)}{8} + w_3 V_{\text{trending}, i} + w_4 B_{\text{following}, i} + \alpha \sqrt{\frac{2 \ln N}{n_i}} - P_{\text{memory}, i}$$

where:
- $w_1 = 0.25, \; w_2 = 0.15, \; w_3 = 0.25, \; w_4 = 0.15, \; \alpha = 0.20$.
- $Q_{\text{task}, i} = \frac{0.6 \cdot C_{\text{cap}, i} + 0.4 \cdot C_{\text{dec}, i}}{10}$ is the normalized capability/decision quality.
- $U_i = \alpha \sqrt{\frac{2 \ln N}{n_i}}$ is the Upper Confidence Bound (UCB1) exploration bonus over total pulls $N$ and model pulls $n_i$.
- $P_{\text{memory}, i}$ is the runtime memory penalty when estimated Q4_0 memory ($0.72 \times \text{Params}_{\text{B}}\text{ GB}$) exceeds effective available memory.

---

## 3. 4-Objective Pareto Front & Geometric Knee-Point Selection

Candidates that fit within available runtime memory are projected into a 4-dimensional objective vector:
1. **Objective 0 (Maximize)**: Task Quality $Q_{\text{task}, i}$
2. **Objective 1 (Maximize)**: Trending Velocity + High-Signal Research Org Boost ($V_{\text{trending}, i} + B_{\text{following}, i}$)
3. **Objective 2 (Maximize)**: Efficiency + UCB1 Exploration Bonus ($E_{\text{eff}, i} + U_i$)
4. **Objective 3 (Minimize)**: Runtime Memory Footprint $M_{\text{runtime}, i}$ (GB)

`compute_pareto_front()` extracts all non-dominated models, and `select_knee_point()` selects the candidate on the Pareto frontier with minimal Euclidean distance to the Utopia Point $(\text{Benefit} = 1.0, \; \text{Cost} = 0.0)$.
