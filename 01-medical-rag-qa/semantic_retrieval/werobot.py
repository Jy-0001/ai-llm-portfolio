import os
import logging

import werobot
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

TIMEOUT = 10
GREETING = "您好, 我是您的智能医疗助手, 有什么需要帮忙的吗?"
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8103/")

# WeChat official-account connector (token validation handled by the framework)
robot = werobot.WeRoBot(token=os.getenv("WEROBOT_TOKEN", "your-token"))


@robot.handler
def doctor(message, session):
    try:
        uid = message.source
        try:
            # First message from this user -> greet
            if session.get(uid, None) != "1":
                session[uid] = "1"
                return GREETING
            text = message.content
        except Exception:
            # User may have re-followed; message.content can be missing
            return GREETING

        data = {"uid": uid, "text": text}
        res = requests.post(BACKEND_URL, data=data, timeout=TIMEOUT)
        return res.text
    except Exception as e:
        logger.error("Handler error: %s", e)
        return "对不起, 服务暂时不可用, 请稍后再试..."


if __name__ == "__main__":
    robot.config["HOST"] = "0.0.0.0"
    robot.config["PORT"] = 80
    robot.run()
