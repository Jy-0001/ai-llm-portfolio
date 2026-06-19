# 启动 20 个并发用户, 以每秒 5 个的速度增⻓, 运行 60 秒
locust -f locust_test.py \
    --headless \
    --host http://0.0.0.0:8001 \
    --users 1500 \
    --spawn-rate 5 \
    --run-time 60s 