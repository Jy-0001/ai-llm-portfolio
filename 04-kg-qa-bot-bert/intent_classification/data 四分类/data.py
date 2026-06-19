with open('bert_model_train.txt', 'r') as data:
    with open('bert_model_train_new.txt', 'w') as new:
        for line in data:
            try:
                s, t = line.split(' ', -1)
                new.write(s + '\t' + t)
            except:
                new.write(line)

with open('bert_model_train.txt', 'r', encoding='utf-8') as data, \
     open('bert_model_train_new.txt', 'w', encoding='utf-8') as new:
    for line in data:
        raw = line.rstrip('\n')
        if '\t' in raw or not raw.strip():
            new.write(line)  # 已经是tab分隔或空行：原样写回
            continue

        parts = raw.strip().rsplit(None, 1)  # 从右边按任意空白分一次
        if len(parts) == 2:
            text, label = parts
            new.write(f"{text}\t{label}\n")
        else:
            new.write(line)