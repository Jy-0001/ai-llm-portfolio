import requests
import time
import json

url = "http://0.0.0.0:8103/"
data = {"question":"早上起来偏头疼怎么回事？"}
# data = {"question":"leetcode热题有哪些，怎么做？"}

start_time = time.time()

data = json.dumps(data)

# 向服务发送请求
res = requests.post(url, data)

cost_time = time.time() - start_time

print('单词查询的耗时：', cost_time, 's')

res = json.loads(res.text)
print(res)
