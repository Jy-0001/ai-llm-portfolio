import requests
import time
import json

# 注意: 0.0.0.0 是服务端监听地址, 客户端请求要用 127.0.0.1 或 localhost
url = "http://127.0.0.1:8103/"
data = {"question": "平日里蜂蜜加白醋一起喝有什么疗效?"}
# data = {"question": "听说用酸枣仁泡水喝能养生, 是真的吗?"}

start_time = time.time()

data = json.dumps(data)

# 向服务发送请求
res = requests.post(url, data)

cost_time = time.time() - start_time

print('单次查询的耗时:', cost_time, 's')

res = json.loads(res.text)
print(res)
