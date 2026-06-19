# 02 · 医疗大模型有监督微调（SFT）+ 人类偏好对齐（RLHF / DPO）

以 **Qwen2.5-32B-Instruct** 为基座，针对医疗问诊场景打通"指令微调 → 偏好对齐 → 多维评估"的完整链路，并在真实多卡环境完成分布式训练。32B 全参无法放入 4×4090，故采用 **QLoRA(4-bit NF4) + DeepSpeed ZeRO-3 offload**。

## 训练链路
1. **SFT 指令微调**：约 5 万条医疗问答整理为 Alpaca / ShareGPT 格式，QLoRA 低成本指令对齐。
2. **DPO 直接偏好优化**：约 1 万条 `(prompt, chosen, rejected)` 偏好三元组，绕开显式奖励模型与在线采样直接对齐。
3. **完整 RLHF 三阶段**：SFT → 奖励模型（RM）→ PPO（DeepSpeed-Chat），逐阶段验证；RM 数据有限、信号偏弱 → 据此转向更稳的 DPO。

## 跑 32B 训练（LLaMA-Factory + DeepSpeed ZeRO-3）
> 底座环境见仓库根 `ENVIRONMENT.md`。先装 LLaMA-Factory（源码）与本项目依赖。
```bash
pip install -r 02-llm-sft-rlhf/requirements.txt
# LLaMA-Factory: git clone 后 pip install -e ".[torch,deepspeed,bitsandbytes]"

# 1) 下基座（modelscope）
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-32B-Instruct', cache_dir='/root/autodl-tmp/models')"

# 2) 在 data/dataset_info.json 注册 medical_sft / medical_dpo 数据集

# 3) SFT（QLoRA 4-bit + ZeRO-3）
llamafactory-cli train 02-llm-sft-rlhf/train_configs/qwen2.5_32b_sft_qlora.yaml

# 4) DPO（接 SFT 适配器）
llamafactory-cli train 02-llm-sft-rlhf/train_configs/qwen2.5_32b_dpo_qlora.yaml
```
关键超参（4B→32B 的差异，见 `train_configs/*.yaml`）：4-bit NF4 量化、`per_device_batch=1` + `grad_accum=16`、`gradient_checkpointing=on`、`lora_rank=64/alpha=128`、SFT `lr=1e-4` / DPO `lr=5e-6`、ZeRO-3 + CPU offload。

## 分布式训练
- AutoDL **4×4090**，DeepSpeed **ZeRO-3 + offload**；自定义 collate、实现 Focal Loss 等；理解 actor-critic / 优势估计 / KL 约束。
- 掌握 ZeRO Stage 1/2/3 显存分片、**Ring AllReduce** 通信、**DP/TP/PP** 三维并行的"通信量↔冗余显存"权衡。

## 强化学习算法对比
手写实现并对比：`DPO` · `PPO` · `GRPO` · `GSPO` · `SAPO`（见 `27`~`29`）。

## 评估（`evaluation/`）
通用能力 **MMLU-Pro**（防灾难性遗忘）、垂类 **embedding 语义相似度**、指令跟随 **LLM-as-judge**（DeepSeek 裁判），辅以安全红队与压测（TTFT/TPS/P95·P99，`stress_test/`）。

## 技术栈
DeepSpeed ZeRO-3 · LLaMA-Factory · QLoRA(4-bit) · PyTorch · DPO/PPO/GRPO/GSPO · Qwen2.5-32B · MMLU-Pro · LLM-as-judge

> 本项目为学习/作品集项目；模型权重、数据集、LLaMA-Factory 仓库本体不入库（见 `.gitignore`）。
