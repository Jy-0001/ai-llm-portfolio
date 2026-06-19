import os
from dotenv import load_dotenv
load_dotenv()
import json
import pandas as pd
import numpy as np
from typing import Dict
import time
from dataclasses import dataclass
from tqdm import tqdm
from vllm import LLM, SamplingParams
import warnings
warnings.filterwarnings('ignore')

from zai import ZhipuAiClient

# ====================== 初始化智谱客户端 ======================
client = ZhipuAiClient(api_key=os.getenv("ZHIPU_API_KEY"))

# ====================== 1. 配置与数据结构 ======================
@dataclass
class ModelConfig:
    """模型配置"""
    model_path: str
    model_name: str
    tensor_parallel_size: int = 1


class ModelEvaluator:
    def __init__(self, base_model: ModelConfig, fine_tuned_model: ModelConfig, evaluation_data_path: str):
        self.base_config = base_model
        self.ft_config = fine_tuned_model
        
        # 加载评估数据集
        self.eval_data = self._load_evaluation_data(evaluation_data_path)
        
        # 存储评估结果
        self.results = {
            "base": None,
            "fine_tuned": None
        }

    def _load_evaluation_data(self, path: str) -> list:
        """加载评估数据集"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if line.strip():
                    context = json.loads(line)
                    data.append(context)

        import random
        random.seed(42)
        if len(data) > 1000:
            data = random.sample(data, 1000)

        print(f"Total number of data: {len(data)}")
        return data

    def evaluate_model(self, model_config: ModelConfig) -> Dict:
        """评估单个模型"""
        print(f"\n{'=' * 60}")
        print(f"开始评估模型: {model_config.model_name}")
        print(f"{'=' * 60}")

        # 初始化 vLLM
        llm = LLM(
            model=model_config.model_path,
            tensor_parallel_size=model_config.tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=512
        )

        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_tokens=512,
        )

        print("正在进行领域任务评估...")
        domain_results = self._evaluate_domain_task(llm, sampling_params)
        
        return domain_results

    def _evaluate_domain_task(self, llm, sampling_params) -> Dict:
        """评估领域特定任务"""
        details = []
        
        for item in tqdm(self.eval_data, desc="Evaluating"):
            prompt = self._build_domain_prompt(item['prompt'])
            
            start_time = time.time()
            outputs = llm.generate([prompt], sampling_params)
            latency = (time.time() - start_time) * 1000
            
            prediction = outputs[0].outputs[0].text.strip()
            
            similarity = self._calculate_similarity(prediction, item['chosen'])
            
            details.append({
                "input": item['prompt'][:200],
                "expected": item['chosen'],
                "predicted": prediction,
                "similarity": similarity,
                "latency_ms": latency
            })

        return {"details": details}

    def _calculate_similarity(self, pred: str, expected: str) -> float:
        """使用智谱 Embedding-3 计算余弦相似度"""
        response = client.embeddings.create(
            model="embedding-3",
            input=[pred, expected]
        )
        embeddings = [item.embedding for item in response.data]
        return self._cosine_similarity(embeddings[0], embeddings[1])

    def _cosine_similarity(self, vec1, vec2):
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2 + 1e-8)

    def _build_domain_prompt(self, input_text: str) -> str:
        """构建领域任务提示"""
        return f"""你是一个专业的医生。请根据以下用户问题提供准确、有帮助的回复。
用户问题: {input_text}
回复要求:
1. 直接解决问题核心
2. 语言简洁并严格遵循医学的专业知识
3. 如信息不足，请询问澄清，不要编造不实信息
回复:"""

    def run_comparative_evaluation(self) -> Dict:
        """运行完整对比评估"""
        print("开始对比评估...")

        self.results["base"] = self.evaluate_model(self.base_config)
        self.results["fine_tuned"] = self.evaluate_model(self.ft_config)

        report = self._generate_comparison_report()
        return report

    def _generate_comparison_report(self) -> Dict:
        """生成对比报告"""
        base_details = self.results["base"]["details"]
        finetune_details = self.results["fine_tuned"]["details"]

        num_base, num_finetune = 0.0, 0.0
        for base_v, finetune_v in zip(base_details, finetune_details):
            if base_v['similarity'] > finetune_v['similarity']:
                num_base += 1
            else:
                num_finetune += 1

        if num_base * 0.95 > num_finetune:
            improvement = "显著下降 ❌"
        elif num_finetune > num_base * 1.05:
            improvement = "显著提升 ✅"
        else:
            improvement = "基本持平 ➡️"

        return {
            "summary": {
                "base_better": num_base,
                "finetune_better": num_finetune,
                "total_samples": len(base_details)
            },
            "improvement_analysis": improvement
        }


# ====================== 主程序 ======================
if __name__ == "__main__":
    base_model = ModelConfig(
        model_path="/root/autodl-tmp/40_rlhf_on_llama/Qwen3-4B",
        model_name="Qwen3-4B-Base"
    )

    fine_tuned_model = ModelConfig(
        model_path="/root/autodl-tmp/40_rlhf_on_llama/LLaMA-Factory/output/llama3_lora_ppo/",
        model_name="Qwen3-4B-Finetuned"
    )

    evaluator = ModelEvaluator(
        base_model=base_model,
        fine_tuned_model=fine_tuned_model,
        evaluation_data_path="/root/autodl-tmp/42_eval/dev.jsonl"
    )

    report = evaluator.run_comparative_evaluation()

    print("\n" + "="*80)
    print("评估结果摘要")
    print("="*80)
    print(report)