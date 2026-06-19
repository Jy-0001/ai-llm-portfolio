import json
import time
import random
import numpy as np
import os
from locust import HttpUser, task, between, events

# ================= 配置区域 =================
RAG_PROMPTS = [
    "小儿腹泻不能吃什么东西吗?",
    "乳宁颗粒可能会有哪些不良反应？",
    "酸枣仁泡水好处是什么？",
    "一直放屁还很臭是怎么回事？",
    "镶了烤瓷牙可以用电动牙刷吗?"
]

LLM_PROMPTS = [
    "请帮我写一个关于未来AI医疗的科幻故事，500字左右。",
    "用Python写一个分析心电图数据的算法逻辑。",
    "如果人类可以永生，医疗系统会发生什么变化？请详细分析。",
    "解释一下量子计算对新药研发的具体潜在影响。",
    "The quick brown fox jumps over the lazy dog. Translate this to Chinese."
]

LLM_RATIO = 0.5
REPORT_FILE = "benchmark_stream_report.json"
# ===========================================

METRICS_STORE = {
    "LLM_TTFT": [], "LLM_TPS": [], "LLM_LATENCY": [],
    "RAG_LATENCY": [], "ERRORS": 0, "TOTAL_REQS": 0,
    "START_TIME": 0
}


class MedicalBotUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        self.client.timeout = 600.0

    @task
    def chat_request(self):
        is_llm_request = random.random() < LLM_RATIO
        if is_llm_request:
            prompt = random.choice(LLM_PROMPTS)
            request_name = "API_LLM"
        else:
            prompt = random.choice(RAG_PROMPTS)
            request_name = "API_RAG"

        payload = {"messages": [{"role": "user", "content": prompt}]}
        start_time = time.time()

        # 开启 stream=True
        with self.client.post(
            "/",
            json=payload,
            name=request_name,
            stream=True,
            catch_response=True
        ) as response:
            if response.status_code != 200:
                METRICS_STORE["ERRORS"] += 1
                response.failure(f"HTTP {response.status_code}")
                return

            try:
                # --- 流式指标计算逻辑 ---
                ttft = 0
                first_byte_received = False
                content_length = 0

                # 迭代接收流数据
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        # 收到第一个数据包, 计算 TTFT
                        if not first_byte_received:
                            ttft = (time.time() - start_time) * 1000  # ms
                            first_byte_received = True
                        # 累加内容长度 (用于计算 TPS)
                        # 注意: app.py 返回的是 UTF-8 文本流
                        content_length += len(chunk.decode('utf-8', errors='ignore'))

                # 请求结束
                total_time = time.time() - start_time
                total_time_ms = total_time * 1000

                METRICS_STORE["TOTAL_REQS"] += 1

                # --- 记录指标 ---
                if is_llm_request:
                    # 如果内容为空, 可能出错了
                    if content_length == 0:
                        response.failure("Empty stream response")
                        METRICS_STORE["ERRORS"] += 1
                        return

                    tps = content_length / total_time if total_time > 0 else 0
                    METRICS_STORE["LLM_TPS"].append(tps)
                    METRICS_STORE["LLM_TTFT"].append(ttft)
                    METRICS_STORE["LLM_LATENCY"].append(total_time_ms)

                    # 上报实时数据
                    events.request.fire(
                        request_type="METRIC",
                        name="LLM_TPS",
                        response_time=tps,
                        response_length=content_length,
                        exception=None
                    )
                    events.request.fire(
                        request_type="METRIC",
                        name="LLM_TTFT",
                        response_time=ttft,
                        response_length=0,
                        exception=None
                    )
                else:
                    # RAG 场景通常是一次性返回, TTFT 约等于 Latency
                    METRICS_STORE["RAG_LATENCY"].append(total_time_ms)
                    response.success()

            except Exception as e:
                METRICS_STORE["ERRORS"] += 1
                response.failure(f"Stream error: {e}")


# --- 统计与报告生成 ---
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print(f"=== [Start] 流式压测开始 | LLM占比: {LLM_RATIO*100}% ===")
    METRICS_STORE["START_TIME"] = time.time()
    # 重置数据
    for k in METRICS_STORE:
        if isinstance(METRICS_STORE[k], list):
            METRICS_STORE[k] = []
        elif isinstance(METRICS_STORE[k], int):
            METRICS_STORE[k] = 0


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("\n=== [Stop] 压测结束, 计算流式指标... ===")
    duration = time.time() - METRICS_STORE["START_TIME"]
    total_reqs = METRICS_STORE["TOTAL_REQS"]

    def calc_stats(data_list):
        if not data_list:
            return {"avg": 0, "p95": 0, "p99": 0}
        return {
            "count": len(data_list),
            "avg": round(np.mean(data_list), 2),
            "p50": round(np.percentile(data_list, 50), 2),
            "p95": round(np.percentile(data_list, 95), 2),
            "p99": round(np.percentile(data_list, 99), 2),
            "min": round(np.min(data_list), 2),
            "max": round(np.max(data_list), 2)
        }

    stats_report = {
        "summary": {
            "duration": round(duration, 2),
            "total_requests": total_reqs,
            "errors": METRICS_STORE["ERRORS"],
            "qps": round(total_reqs / duration, 2) if duration > 0 else 0
        },
        "metrics": {
            "LLM_TPS": calc_stats(METRICS_STORE["LLM_TPS"]),
            "LLM_TTFT_MS": calc_stats(METRICS_STORE["LLM_TTFT"]),       # 真实的 TTFT
            "LLM_Total_Latency_MS": calc_stats(METRICS_STORE["LLM_LATENCY"]),
            "RAG_Latency_MS": calc_stats(METRICS_STORE["RAG_LATENCY"])
        }
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(stats_report, f, indent=4, ensure_ascii=False)

    print(f"流式压测报告已生成: {REPORT_FILE}")
    print(json.dumps(stats_report, indent=4, ensure_ascii=False))