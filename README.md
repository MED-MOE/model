
<p align="center">
  <img src="/images/dr_llama.png" width="500" alt="dr llama photo">
</p>



# 🏥 Agent Hospital Optimization via MedLLaMA + Mixture of Experts

**A research project by Xinzhuo Jiang and Dan Harvey**  
**Course: HPML | Columbia University**

## 🧠 Overview

Inspired by Jacobs et al.’s Mixture of Experts (MoE) and the Agent Hospital framework, this project proposes a novel optimization of clinical decision-making systems using **domain-specialized MedLLaMA experts** coordinated by an intelligent **gating network**—a virtual nurse dynamically routing patient queries to the right specialists.

Our goal is to deliver improved medical inference efficiency, diagnostic accuracy, and energy-aware compute through expert distillation, pruning, and smart expert selection.

---

## 🎯 Objectives

1. **Construct a medical-focused MoE system**  
   Build and distill multiple expert models from MedLLaMA to handle medical subdomains (e.g., Neurology, Cardiology).

2. **Optimize the MoE architecture**  
   Develop a novel **fine-grained gating mechanism** inspired by DeepSeekMoE to route queries intelligently with minimal compute overhead.

3. **Benchmark against parent models**  
   Compare our MoE model to the original MedLLaMA in terms of:
   - Accuracy on medical exams and case studies
   - Inference latency
   - Memory and energy usage (via NVIDIA profiler)

---

## ⚙️ Methods

### 🩺 Gating Algorithm
- Sparse gating architecture
- Fine-grained expert segmentation: each expert is subdivided to increase routing flexibility
- Always-on **shared expert** to reduce redundancy
- Top-K routing via softmax scoring

### 🧬 Expert Formation
- Extract 5 domain experts from MedLLaMA (Internal Medicine, Neurology, etc.)
- Two distillation strategies:
  - **Activation-based pruning** (via forward hooks)
  - **Sparse dropout masking**
- All experts fine-tuned post-distillation

### 🧪 Implementation
- Medical data from [MedlinePlus API](https://medlineplus.gov/about/developers/webservices/) and [FDA Drug Labels](https://open.fda.gov/apis/)
- Synthetic case generation for supervised training
- Built using **PyTorch**, trained on **A100 GPUs** (Colab Pro)

---

## 📊 Evaluation

We will assess:
- **Accuracy** on board-style medical questions and synthetic patient scenarios
- **Efficiency** using power/memory profiling tools
- **Zero-shot performance** vs. the base MedLLaMA model

---

## 🧠 Planned Demo

A user-facing consultation platform will allow real-time input of symptoms, showing:
- Which experts were activated
- Routing decisions made by the gating network
- Multi-specialist overlap for complex conditions

---

## 📚 References

1. Jacobs et al. (1991). *Adaptive Mixtures of Local Experts*  
2. Li et al. (2024). *Agent Hospital: A Simulacrum of Hospital with Evolvable Medical Agents*  
3. Dai et al. (2024). *DeepSeekMoE*  
4. Xie et al. (2024). *Me-LLaMA: Foundation LLMs for Medical Applications*
5. https://github.com/OpenSparseLLMs/LLaMA-MoE-v2

---

## 🚧 Challenges & Future Work

- Selecting optimal pruning/distillation strategies
- Balancing expert specialization vs. generalization
- Integrating with existing quantized/LoRA-tuned LLaMA variants
- Exploring Mixtral and DeepSeek insights for model evolution

