# sample_dialog.py
import json
import random

INPUT_PATH = "/root/autodl-tmp/42_eval/dialog.jsonl"
OUTPUT_PATH = "/root/autodl-tmp/42_eval/dialog_200.jsonl"
SAMPLE_SIZE = 200
SEED = 42  # 固定种子,保证每次抽到同一批

# 加载全部
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = [line for line in f if line.strip()]

print(f"原始数据: {len(data)} 条")

# 随机抽样
random.seed(SEED)
sampled = random.sample(data, min(SAMPLE_SIZE, len(data)))

# 写出
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for line in sampled:
        f.write(line if line.endswith("\n") else line + "\n")

print(f"抽样数据: {len(sampled)} 条 → 已保存到 {OUTPUT_PATH}")