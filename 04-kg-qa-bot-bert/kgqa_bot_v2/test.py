import requests
import time

# 定义请求url和传⼊的data
url = "http://0.0.0.0:5001/v1/main_server/"
# data = {"uid": "AI-2-20251115", "text": "为什么我最近总失眠?"}
data = {"uid": "AI-2-20251115", "text": "那⽿鸣有什么药吗?"}

start_time = time.time()
# 向服务发送post请求
res = requests.post(url, data=data)

cost_time = time.time() - start_time

# 打印返回的结果
print('⽤户输⼊:', data['text'])
print('红蜘蛛AI:', res.text)
print('单条样本预测耗时: ', cost_time * 1000, 'ms')