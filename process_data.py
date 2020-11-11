import os 

# 作业给的数据集中情感极性的id
origin_label_id_dic = {"-1": "NEG", 
                       "0":  "NEU",
                       "1":  "POS",}

# BERT模型的情感极性的id
BERT_label_id_dic = {"NEG": 1, 
                     "NEU": 2, 
                     "POS": 0,} 

class Example:
    def __init__(self, example):
        self.sentence_A = example[0]
        self.sentence_B = example[1]
        self.label_id = convert_label_id(example[2])

class Feature:
    def __init__(self, input_ids, input_masks, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.label_id = label_id

def convert_label_id(label_id_str):
    """
    将作业的数据集的情感极性id转换为BERT模型的情感极性的id
    """
    return BERT_label_id_dic[origin_label_id_dic[label_id_str]]

def convert_label_id_back(bert_results):
    """
    将BERT预测出的情感极性id转换为作业给出的情感极性的id
    需要注意转换的顺序, 否则会出现错误
    """
    bert_results[bert_results == 1] = -1
    bert_results[bert_results == 0] = 1
    bert_results[bert_results == 2] = 0
    return bert_results

def transfer_to_examples(raw_data_dir):
    train_examples, test_examples = [], []

    #print(os.listdir(raw_data_dir))
    for filename in os.listdir(raw_data_dir):
        filepath = os.path.join(raw_data_dir, filename)
        if "train" in filename:
            with open(filepath, 'r', encoding="UTF-8") as f:
                cnt = 0
                line = f.readline().strip()
                example = []
                while line and line != '':
                    cnt += 1
                    example.append(line)
                    if cnt % 3 == 0:
                        example[0] = example[0].replace("$T$", example[1], 1)
                        train_examples.append(Example(example))
                        example = []
                    line = f.readline().strip()  
            f.close()
            #for i in range(10):
            #    print(train_examples[i].sentence_A, train_examples[i].sentence_B, 
            #            train_examples[i].label_id)    
        elif "test" in filename:
            with open(filepath, 'r', encoding="utf-8") as f:
                cnt = 0
                line = f.readline().strip()
                example = []
                while line and line != '':
                    cnt += 1
                    example.append(line)
                    if cnt % 2 == 0:
                        example[0] = example[0].replace("$T$", example[1], 1)
                        # 由于BERT模型的输入必须要有一个label_id, 所以把所有
                        # 测试集中样本的label_id都设成 "0" (neual), 在使用中
                        # 不会用到
                        example.append("0") 
                        test_examples.append(Example(example))
                        example = []
                    line = f.readline().strip()
            f.close() 
            #for i in range(10):
            #    print(test_examples[i].sentence_A, test_examples[i].sentence_B, 
            #            test_examples[i].label_id)

    return train_examples, test_examples

def transfer_to_features(args, tokenizer):
    train_examples, test_examples = transfer_to_examples(args.data_dir)
    train_features, test_features = [], []
    tokenized_train_examples, tokenized_test_examples = [], []

    for example in train_examples:
        input_ids, input_masks, segment_ids, tokens_whole = tokenize(example, tokenizer, args.max_seq_len)
        train_features.append(Feature(input_ids, input_masks, segment_ids, example.label_id))
        tokenized_train_examples.append(" ".join([str(token) for token in tokens_whole]))
    
    #print(train_features[0].label_id)
    for example in test_examples:
        input_ids, input_masks, segment_ids, tokens_whole = tokenize(example, tokenizer, args.max_seq_len)
        test_features.append(Feature(input_ids, input_masks, segment_ids, example.label_id))
        tokenized_test_examples.append(" ".join([str(token) for token in tokens_whole]))
    
    return train_features, tokenized_train_examples, test_features, tokenized_test_examples

def tokenize(example, tokenizer, max_seq_len):
    tokens_A = tokenizer.tokenize(example.sentence_A)
    tokens_B = tokenizer.tokenize(example.sentence_B) 
    
    # 第一个句子 (不含开头指示符) 的token以及对应的segment_id
    tokens_whole = tokens_A + ['[SEP]']
    segment_ids = [0] * len(tokens_whole) 

    #第二个句子的token以及对应的segment_id
    tokens_whole += tokens_B + ['[SEP]']
    segment_ids += [1] * (len(tokens_B) + 1)

    #在开头加上[CLS]
    tokens_whole = ['[CLS]'] + tokens_whole
    segment_ids = [1] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens_whole)

    input_masks = [1] * len(input_ids)
    padding_len = max_seq_len - len(input_ids)

    assert padding_len > 0 # 假定不会超过最长串

    padding = [0] * padding_len
    # 在剩余的位置补上 0 (mask 和 id 都是)
    input_ids = input_ids + padding
    input_masks = input_masks + padding
    segment_ids = segment_ids + padding

    assert len(input_ids) == max_seq_len
    assert len(input_masks) == max_seq_len
    assert len(segment_ids) == max_seq_len

    return input_ids, input_masks, segment_ids, tokens_whole

if __name__ == "__main__":
    #transfer_to_examples("./data/raw/")
    print(len(BERT_label_id_dic))
