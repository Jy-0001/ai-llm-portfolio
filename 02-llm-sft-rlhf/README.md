# 02 · 医疗大模型有监督微调（SFT）+ 人类偏好对齐（RLHF / DPO）

以开源基座（Qwen3-4B / Llama 系列）为底座，针对医疗问诊场景打通"指令微调 → 偏好对齐 → 多维评估"的完整训练链路，并在真实多卡环境完成分布式训练。

## 训练链路
1. **SFT 指令微调**：将约 50 万条医疗问答整理为 Alpaca / ShareGPT 格式，基于 LLaMA-Factory + LoRA 完成低成本指令对齐。
2. **DPO 直接偏好优化**：构建约 50 万条 `(prompt, chosen, rejected)` 偏好三元组，绕开显式奖励模型与在线采样，直接在偏好数据上对齐。
3. **完整 RLHF 三阶段**：SFT → 奖励模型（Reward Model）→ PPO 策略优化，逐阶段独立训练验证；理解 actor-critic、优势估计与 KL 约束。

## 分布式训练（真实实操）
- AutoDL **3×4090** 真实多卡环境，基于 **DeepSpeed ZeRO** 完成分布式微调；自定义 collate 数据整理类、实现并通读 Focal Loss 等核心代码；三阶段累计训练 24h+。
- 掌握 ZeRO（Stage 1/2/3）显存分片差异、**Ring AllReduce** 通信与 **DP / TP / PP** 三维并行的"通讯量↔冗余显存"权衡；实践混合精度训练与 safetensors 存储。

## 强化学习算法对比
手写实现并对比主流对齐/策略算法：`DPO` · `PPO` · `GRPO` · `GSPO` · `SAPO`（见 `27`~`29`）。

## 评估
通用能力用 **MMLU-Pro**（防灾难性遗忘）、垂类效果用 **embedding 语义相似度**、指令跟随用 **LLM-as-judge**，辅以安全红队与服务压测（TTFT / TPOT / TPS / QPS / P95·P99）。

## 关键文件
- `40_rlhf_on_llama/` — DPO 数据处理等本人脚本（不含 LLaMA-Factory 仓库本体）
- `42_eval/` — 大模型微调效果评估方法与代码
- `37_allreduce_zero/` — AllReduce / ZeRO / DP·TP·PP 实现笔记
- `27_PPO_DPO` · `28_GRPO` · `29_GSPO_SAPO` — 强化学习算法实现

## 技术栈
LLaMA-Factory · DeepSpeed ZeRO · PyTorch · LoRA · DPO/PPO/GRPO/GSPO · Qwen3 / Llama / ChatGLM
