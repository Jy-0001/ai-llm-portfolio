# 导入相关工具包
import os
import sys
sys.path.append('../..')
from config import Args
from utils import Inference
import time


def func_macbert4csc_inference(model, input_text):
    start_time = time.time()
    result = model.predict(input_text)
    end_time = time.time()
    print('单条样本推理耗时: {}s'.format(end_time - start_time))

    return result




if __name__ == '__main__':
    s = 'X'
    model = Inference()

    while True and s != 'q' and s != 'Q':
        person_input = input('Please input a sentence:')
        res = func_macbert4csc_inference(model, person_input)

        print('res = ', res)

        s = input('Continue or not (q/Q)')


# 请问一下,华鑫证卷的客服号马是多少?
# 华通转债今天下午涨俯如和?
# 夹江县本地女雪生酒店过夜特殊按摩哪狸有可以


