import sys
import re
from collections import defaultdict, namedtuple
ANY_SPACE = '<SPACE>'
class FormatError(Exception):
    pass
Metrics = namedtuple('Metrics', 'tp fp fn prec rec fscore')
# 计算字典的类代码
class EvalCounts(object):
    def __init__(self):
        # 初始化5个计数器, 初始值0
        self.correct_chunk = 0
        self.correct_tags = 0
        self.found_correct = 0
        self.found_guessed = 0
        self.token_counter = 0
        # 初始化3个计数器字典, 初始值空
        self.t_correct_chunk = defaultdict(int)
        self.t_found_correct = defaultdict(int)
        self.t_found_guessed = defaultdict(int)
# 添加若⼲特殊的配置参数, 以更好的处理⽂件
def parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser(description='evaluate tagging results using CoNLL criteria', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument
    arg('-b', '--boundary', metavar='STR', default='-X-', help='sentence boundary')
    arg('-d', '--delimiter', metavar='CHAR', default=ANY_SPACE, help='character delimiting items in input')
    arg('-o', '--otag', metavar='CHAR', default='O', help='alternative outside tag')
    arg('file', nargs='?', default=None)
    return parser.parse_args(argv)
def parse_tag(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    # t: B-stock_name
    # m: <_sre.SRE_Match object; span=(0, 12), match='B-stock_name'>
    return m.groups() if m else (t, '')
# 计算关键指标的主要函数, 通过calculate()来调⽤
def evaluate(iterable, options=None):
    # 如果没有传⼊配置信息, 则调⽤本⽂件内的parse_args()函数来获得初始化的配置信息
    if options is None:
        options = parse_args([])
    # 实例化计数器类的对象, ⾥⾯有5个值计数器 + 3个字典类型计数器
    counts = EvalCounts()
    # 当前样本所拥有的实体数量
    num_features = None
    # 截⽌⽬前所有识别出来的实体都正确
    in_correct = False
    # 上⼀个识别的gold字符标签
    last_correct = 'O'
    # 上⼀个识别的gold标签类型
    last_correct_type = ''
    # 上⼀个模型识别的字符标签
    last_guessed = 'O'
    # 上⼀个模型识别的标签类型
    last_guessed_type = ''
    # 遍历评估⽂件的每⼀⾏数据, 本质上有3列[text_char, gold_label, predict_label]
    for line in iterable:
        line = line.rstrip('\r\n')
        # ANY_SPACE = '<SPACE>', 程序代码最开始的定义, 并通过parse_args()函数设置为分隔符了
        if options.delimiter == ANY_SPACE:
            features = line.split()
        else:
            features = line.split(options.delimiter)
        # len(features) = 3
        # features: ['*', 'B-stock_name', 'B-stock_name']
        if len(features) == 2:
            features = [" "]+features
        if num_features is None:
            num_features = len(features)
        elif num_features != len(features) and len(features) != 0:
            raise FormatError('unexpected number of features: %d (%d)' % (len(features), 
num_features))
        # num_features: 3
        if len(features) == 0 or features[0] == options.boundary:
            features = [options.boundary, 'O', 'O']
        if len(features) < 3:
            raise FormatError('unexpected number of features in line %s' % line)
        # features: ['*', 'B-stock_name', 'B-stock_name']
        guessed, guessed_type = parse_tag(features.pop())
        # features: ['*', 'B-stock_name']
        # guessed: B
        # guessed_type: stock_name
        correct, correct_type = parse_tag(features.pop())
        # features: ['*']
        # correct: B
        # correct_type: stock_name
        first_item = features.pop(0)
        # features4: []
        # first_item: '*'
        if first_item == options.boundary:
            guessed = 'O'
        # print('$$$')
        # 通 B-stock_name B-stock_name
        # ⽤ I-stock_name I-stock_name
        # 设 I-stock_name I-stock_name
        # 备 E-stock_name E-stock_name
        # 是 O O
        # ⼲ O O
        # 啥 O O
        # 的 O O
        #
        # 江 B-stock_name B-stock_name
        # ⻄ I-stock_name I-stock_name
        # 铜 I-stock_name I-stock_name
        # 业 E-stock_name E-stock_name
        # 庄 O O
        # 家 O O
        # 持 O O
        # 股 O O
        # 均 O O
        # 价 O O
        # 是 O O
        # 多 O O
        # 少 O O
        # 参考上⾯⽂件数据格式, 每次扫描的⼀⾏数据只是实体中的⼀个tag, ⽐如4个tag组成⼀个有效的chunk = "通⽤设备"
        # 判断当前⾏的标签是不是⼀个"有效实体chunk的结束tag"
        end_correct = end_of_chunk(last_correct, correct, last_correct_type, correct_type)
        end_guessed = end_of_chunk(last_guessed, guessed, last_guessed_type, guessed_type)
        # 判断当前⾏的标签是不是⼀个"有效实体chunk的起始tag"
        start_correct = start_of_chunk(last_correct, correct, last_correct_type, correct_type)
        start_guessed = start_of_chunk(last_guessed, guessed, last_guessed_type, guessed_type)
        # end_correct: False
        # end_guessed: False
        # start_correct: True
        # start_guessed: True
        # 当前的判断⼀直正确, 则有机会完整正确的识别出⼀个"有效实体"
        if in_correct:
            # 完整正确的识别出⼀个有效实体, 计数器+1
            if (end_correct and end_guessed and last_guessed_type == last_correct_type):
                in_correct = False
                counts.correct_chunk += 1
                counts.t_correct_chunk[last_correct_type] += 1
            # 识别错误, 只更改标志变量
            elif (end_correct != end_guessed or guessed_type != correct_type):
                in_correct = False
        # 起始tag识别正确, "预示着"⼀个实体"有机会"被识别出来
        if start_correct and start_guessed and guessed_type == correct_type:
            in_correct = True
        # 起始tag标记正确, gold计数器+1
        if start_correct:
            counts.found_correct += 1
            counts.t_found_correct[correct_type] += 1
        # 起始tag识别正确, gold计数器+1
        if start_guessed:
            counts.found_guessed += 1
            counts.t_found_guessed[guessed_type] += 1
        # first_item不等于换⾏符, 说明处于有效⽂本中间
        if first_item != options.boundary:
            # 每识别正确⼀个tag, 计数器+1
            if correct == guessed and guessed_type == correct_type:
                counts.correct_tags += 1
            # 每扫描⼀⾏(本质上每扫描⼀个tag), 计数器+1
            counts.token_counter += 1
        # 对上⼀个tag进⾏赋值, 以和下⼀⾏tag进⾏⽐对
        last_guessed = guessed
        last_correct = correct
        last_guessed_type = guessed_type
        last_correct_type = correct_type
    # ⽂件中的最后⼀个"识别出的有效实体", 也需要累加进计数器
    if in_correct:
        counts.correct_chunk += 1
        counts.t_correct_chunk[last_correct_type] += 1
    return counts
def uniq(iterable):
    seen = set()
    return [i for i in iterable if not (i in seen or seen.add(i))]
def calculate_metrics(correct, guessed, total):
    tp, fp, fn = correct, guessed-correct, total-correct
    p = 0 if tp + fp == 0 else 1. * tp / (tp + fp)
    r = 0 if tp + fn == 0 else 1. * tp / (tp + fn)
    f = 0 if p + r == 0 else 2 * p * r / (p + r)
    return Metrics(tp, fp, fn, p, r, f)
def metrics(counts):
    c = counts
    overall = calculate_metrics(c.correct_chunk, c.found_guessed, c.found_correct)
    by_type = {}
    for t in uniq(list(c.t_found_correct.keys()) + list(c.t_found_guessed.keys())):
        by_type[t] = calculate_metrics(c.t_correct_chunk[t], c.t_found_guessed[t], 
c.t_found_correct[t])
    return overall, by_type
# 打印相关信息的函数
def report(counts, out=None):
    if out is None:
        out = sys.stdout
    overall, by_type = metrics(counts)
    c = counts
    out.write('processed %d tokens with %d phrases; ' % (c.token_counter, 
c.found_correct))
    out.write('found: %d phrases; correct: %d.\n' % (c.found_guessed, c.correct_chunk))
    if c.token_counter > 0:
        out.write('accuracy: %6.2f%%; ' % (100. * c.correct_tags / c.token_counter))
    out.write('precision: %6.2f%%; ' % (100. * overall.prec))
    out.write('recall: %6.2f%%; ' % (100. * overall.rec))
    out.write('FB1: %6.2f\n' % (100. * overall.fscore))
    return overall.fscore
# 判断当前⾏的某个tag是不是⼀个"有效实体chunk的结束tag"
def end_of_chunk(prev_tag, tag, prev_type, type_):
    # prev_tag: O
    # tag: B
    # prev_type: 
    # type_: stock_name
    chunk_end = False
    # 如果上⼀个字符标签是E, 或者S, 则结束标志位True
    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True
    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True
    if prev_tag == ']': chunk_end = True
    if prev_tag == '[': chunk_end = True
    return chunk_end
# 判断当前⾏的某个tag是不是⼀个"有效实体chunk的起始tag"
def start_of_chunk(prev_tag, tag, prev_type, type_):
    # prev_tag: O
    # tag: B
    # prev_type:
    # type_: stock_name
    chunk_start = False
    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True
    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True
    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True
    if tag == '[': chunk_start = True
    if tag == ']': chunk_start = True
    return chunk_start
def calculate(config):
    with open('./tmp_dev_evaluate_{}'.format(config.task_type), encoding='utf-8') as f:
        counts = evaluate(f)
        f1 = report(counts)
        return f1