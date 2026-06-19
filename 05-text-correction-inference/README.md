# 05 · 中文文本纠错与模型推理加速

中文拼写纠错（CSC）+ 模型工程优化两条线：前者做错误检测与候选筛选，后者把训练好的模型压缩、加速到可低成本部署。

## 中文文本纠错（CSC）
- 基于 **MacBERT4CSC** 实现中文拼写纠错，结合 **SpanPointer** 指针抽取与 **SimCSE** 句向量做错误检测与候选筛选。
- 覆盖序列标注、指针网络、对比学习句向量等 NLP 基础链路。

## 模型推理加速 / 压缩
- **ONNX**：将模型导出为 ONNX 进行推理加速。
- **动态量化（int8）**：BERT 模型体积约由 **392MB → 146MB（≈2.7x）**，精度损失 <1%（Embedding / LayerNorm 保留 float32 以保数值稳定，故未达理论 4x）。
- **知识蒸馏 / 剪枝**：以更小模型逼近大模型效果、裁剪冗余参数，进一步降部署成本。

## 关键文件
- `34_macbert4csc/` — MacBERT4CSC 中文纠错
- `31_SpanPointer/` — SpanPointer 指针抽取
- `32_SimCSE/` — SimCSE 对比学习句向量
- `35_ONNX/` — ONNX 导出与推理加速
- `12-quantization-pruning-Albert-T5/` — 量化（`quantization.py`）、剪枝（`pruning.py`）、ALBERT/T5
- `14_knowledge_distil/` — 知识蒸馏

## 技术栈
MacBERT4CSC · SpanPointer · SimCSE · ONNX · PyTorch · 量化 / 蒸馏 / 剪枝
