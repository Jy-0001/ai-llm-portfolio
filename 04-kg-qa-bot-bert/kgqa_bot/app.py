import time
import logging

from flask import Flask, request

from question_classifier import *
from question_parser import *
from answer_search import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# Medical KGQA pipeline: intent classification -> Cypher assembly -> graph search
class MedicalKGQABot:
    def __init__(self):
        self.classifier = QuestionClassifier()
        self.parser = QuestionParser()
        self.searcher = AnswerSearcher()

    def chat_main(self, sentence):
        default_answer = "您好, 我是智能医疗问答助手, 请输入您的问题"

        # Step 1: classify intent; fall back to default reply if out of scope
        res_classify = self.classifier.classify(sentence)
        if not res_classify:
            return default_answer
        logger.info("Intent classification: %s", res_classify)

        # Step 2: parse the classified question into Cypher queries
        res_sql = self.parser.parser_main(res_classify)
        logger.info("Assembled queries: %s", res_sql)

        # Step 3: run the queries against Neo4j and collect answers
        final_answers = self.searcher.search_main(res_sql)
        if not final_answers:
            return default_answer
        return '\n'.join(final_answers)


start_time = time.time()
bot = MedicalKGQABot()
logger.info("Bot initialized in %.2fs", time.time() - start_time)


@app.route('/v1/main_server/', methods=["POST"])
def main_server():
    uid = request.form['uid']
    text = request.form['text']
    answer = bot.chat_main(text)
    return answer