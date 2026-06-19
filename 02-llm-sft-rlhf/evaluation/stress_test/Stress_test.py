import json
import time
import random
from locust import HttpUser, task, between, events


# 1. 加载真实 Prompt 数据集
# ============================================================
# 模拟真实的 Prompt 数据集(生产环境建议从文件加载)
'''
PROMPTS = [
    "请介绍一下阿里巴巴的 Qwen 大模型。",
    "写一段 Python 代码实现快速排序。",
    "将以下中文翻译成英文: 今天天气真不错, 适合出去压测。",
    "解释量子纠缠的基本原理, 越通俗越好。",
    "写一首关于春天的七言绝句。",
    "分析一下人工智能未来的发展趋势。",
    "Give me a summary of the history of computing.",
    "如何优化 MySQL 的查询性能?",
]
'''

PATH = "/root/autodl-tmp/evaluation/dialog_200.jsonl"
PROMPTS = []
with open(PATH, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        context = json.loads(line)
        PROMPTS.append(context['query'])

print("Total number of PROMPTS: ", len(PROMPTS))


# 2. vLLM 默认参数配置
# ============================================================
# 请替换为你实际部署的模型名称 (vLLM 启动参数中的 model name)
MODEL_NAME = "qwen3-4b-ppo"
MAX_TOKENS = 512
TEMPERATURE = 0.7


# 3. 定义压测用户
# ============================================================
class QwenUser(HttpUser):
    # 思考时间: 模拟用户发出请求后的间隔 (单位: 秒)
    wait_time = between(1, 3)

    # HttpUser 使用 requests, 这里设置超时时间
    def on_start(self):
        self.client.timeout = 600.0

    @task
    def chat_completion(self):
        prompt = random.choice(PROMPTS)

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "你是一位非常专业的医疗助理."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "stream": True   # 关键: 工业界压测必须开启流式, 否则无法计算 TTFT
        }

        headers = {"Content-Type": "application/json"}

        start_time = time.time()
        ttft = 0
        token_count = 0
        first_token_received = False
        request_success = False
        error_msg = ""

        # 发起请求
        try:
            with self.client.post(
                "http://0.0.0.0:6006/v1/chat/completions",
                data=json.dumps(payload),
                headers=headers,
                stream=True,
                catch_response=True
            ) as response:
                if response.status_code != 200:
                    response.failure(f"Status code: {response.status_code}")
                    return

                # 处理流式响应
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')

                        # 过滤掉 SSE 的 ping 或结束标志
                        if decoded_line.startswith("data: ") and decoded_line != "data: [DONE]":
                            try:
                                # 计算 TTFT (收到第一个有效 chunk 的时间)
                                if not first_token_received:
                                    ttft = (time.time() - start_time) * 1000   # 毫秒
                                    first_token_received = True

                                    # 这里可以通过 fire_request 手动上报 TTFT 作为一个单独的指标
                                    events.request.fire(
                                        request_type="LLM_METRIC",
                                        name="TTFT (ms)",
                                        response_time=ttft,
                                        response_length=0,
                                        exception=None,
                                    )

                                # 统计 Token (粗略统计, vLLM API 返回中包含 delta content)
                                chunk_data = json.loads(decoded_line[6:])   # 去掉 "data: "
                                if "content" in chunk_data["choices"][0]["delta"]:
                                    token_count += 1

                            except json.JSONDecodeError:
                                pass
                            except Exception as e:
                                error_msg = str(e)

                # 请求结束
                total_time = (time.time() - start_time) * 1000   # 毫秒
                request_success = True

                # 标记请求成功, 并上报 Token 生成速度
                if token_count > 0:
                    tps = token_count / (total_time / 1000)
                    events.request.fire(
                        request_type="LLM_METRIC",
                        name="Output TPS (Tokens/s)",
                        response_time=tps,            # 这里借用 response_time 字段展示 TPS
                        response_length=token_count,
                        exception=None,
                    )

                response.success()

        except Exception as e:
            # 处理连接超时等异常
            events.request.fire(
                request_type="LLM_API",
                name="Failure",
                response_time=0,
                response_length=0,
                exception=e,
            )


# 4. 测试结束钩子
# ============================================================
# 用于在 Console 输出汇总信息的钩子 (可选)
@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("\n=== Test Finished ===")


# step 3: 启动压力测试评估
# ============================================================
# 开始压力测试:
#   服务端: 启动 vLLM 服务 (vllm serve <model_path> --port 6006)
#   客户端: locust -f 本脚本.py --host http://0.0.0.0:6006