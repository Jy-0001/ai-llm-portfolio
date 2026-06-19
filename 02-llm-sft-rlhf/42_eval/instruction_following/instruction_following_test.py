import os
from dotenv import load_dotenv
load_dotenv()
import json
import numpy as np
import pandas as pd
import time
import warnings
from dataclasses import dataclass
from tqdm import tqdm
from vllm import LLM, SamplingParams
from openai import OpenAI

warnings.filterwarnings('ignore')


# ====================== 1. 配置与数据结构 ======================
@dataclass
class ModelConfig:
    """模型配置"""
    model_path: str
    model_name: str
    tensor_parallel_size: int = 1


@dataclass
class EvaluationResult:
    """评估结果存储"""
    model_name: str
    metrics: dict
    detailed_results: pd.DataFrame


# ====================== 2. 模型评估器 ======================
class ModelEvaluator:
    def __init__(self, 
                 base_model: ModelConfig, 
                 fine_tuned_model: ModelConfig, 
                 evaluation_data_path: str):
        
        self.base_config = base_model
        self.ft_config = fine_tuned_model
        
        # 加载评估数据集
        self.eval_data = self._load_evaluation_data(evaluation_data_path)
        
        # 初始化 LLM-as-a-Judge (使用 DeepSeek)
        self.judge_client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY")   # ← 替换成你的 DeepSeek Key
        )
        
        self.results = {"base": None, "fine_tuned": None}

    def _load_evaluation_data(self, path: str):
        """加载评估数据集"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        print(f"Total number of data: {len(data)}")
        return data

    def evaluate_model(self, model_config: ModelConfig) -> EvaluationResult:
        """评估单个模型"""
        print(f"\n{'='*60}")
        print(f"开始评估模型: {model_config.model_name}")
        print(f"{'='*60}")

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

        print("正在评估指令跟随能力...")
        instruction_results = self._evaluate_instruction_following(llm, sampling_params)

        result_df = pd.DataFrame(instruction_results["details"])

        return EvaluationResult(
            model_name=model_config.model_name,
            metrics=instruction_results["metrics"],
            detailed_results=result_df
        )

    def _evaluate_instruction_following(self, llm, sampling_params):
        """使用 LLM-as-a-Judge 评估指令跟随能力"""
        details = []
        scores = []

        for item in tqdm(self.eval_data, desc="指令跟随评估"):
            prompt = item['query']
            
            # 生成回复
            outputs = llm.generate([prompt], sampling_params)
            response = outputs[0].outputs[0].text.strip()

            # 使用 DeepSeek 作为裁判打分
            judge_prompt = f"""请评估以下模型回复的质量，从1-5分打分：
                            指令: {prompt}
                            回复: {response}

                            评分标准：
                            5分: 完美遵循指令，回复全面准确
                            4分: 较好遵循指令，少量不足
                            3分: 基本遵循指令，但有关键遗漏
                            2分: 部分偏离指令
                            1分: 完全偏离指令或无意义

                            只输出一个数字（1-5），不要输出其他任何内容："""

            try:
                judge_response = self.judge_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "你是在医学领域非常专业的助手，可以准确评估回复的质量。"},
                        {"role": "user", "content": judge_prompt}
                    ],
                    stream=False
                )
                score = float(judge_response.choices[0].message.content.strip())
            except Exception as e:
                print(f"Judge调用失败: {e}")
                score = 3.0  # 默认分数

            scores.append(score)
            details.append({
                "instruction": prompt[:200],
                "response": response[:200],
                "judge_score": score
            })

        return {
            "metrics": {
                "instruction_score_avg": np.mean(scores),
                "instruction_score_std": np.std(scores)
            },
            "details": details
        }

    def run_comparative_evaluation(self):
        """运行完整对比评估"""
        print("开始对比评估...")

        self.results["base"] = self.evaluate_model(self.base_config)
        self.results["fine_tuned"] = self.evaluate_model(self.ft_config)

        report = self._generate_comparison_report()
        print(report)
        return report

    def _generate_comparison_report(self):
        """生成对比报告"""
        base_metrics = self.results["base"].metrics
        ft_metrics = self.results["fine_tuned"].metrics

        base_score = base_metrics["instruction_score_avg"]
        ft_score = ft_metrics["instruction_score_avg"]

        improvement = ((ft_score - base_score) / base_score * 100) if base_score != 0 else float('inf')

        if ft_score > base_score * 1.05:
            analysis = "显著提升 ✅"
        elif ft_score < base_score * 0.95:
            analysis = "显著下降 ❌"
        else:
            analysis = "基本持平 ➡️"

        return {
            "summary": {
                "base_model_score": round(base_score, 4),
                "fine_tuned_score": round(ft_score, 4),
                "absolute_change": round(ft_score - base_score, 4),
                "relative_change": f"{round(improvement, 2)}%"
            },
            "improvement_analysis": analysis
        }


# ====================== 主程序 ======================
if __name__ == "__main__":
    base_model = ModelConfig(
        model_path="/root/autodl-tmp/40_rlhf_on_llama/Qwen3-4B",
        model_name="Qwen3-4B-Base"
    )

    fine_tuned_model = ModelConfig(
        model_path="/root/autodl-tmp/40_rlhf_on_llama/LLaMA-Factory/output/llama3_lora_ppo",
        model_name="Qwen3-4B-Finetuned"
    )

    evaluator = ModelEvaluator(
        base_model=base_model,
        fine_tuned_model=fine_tuned_model,
        evaluation_data_path="/root/autodl-tmp/42_eval/dialog_200.jsonl"   # ← 修改成你的评估文件路径
    )

    report = evaluator.run_comparative_evaluation()

    print("\n" + "="*80)
    print("评估结果摘要")
    print("="*80)
    print(report)