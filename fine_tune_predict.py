import argparse
from tokenize import Bracket
import numpy as np 
import os 
import random
import sys 

import torch
from pytorch_transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_transformers import AdamW, WarmupLinearSchedule

from process_data import transfer_to_features, convert_label_id_back, print_features, BERT_label_id_dic
from tqdm import tqdm, trange

def train(args, train_dataset, model):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        # 如果指定了最大的执行step , 则重新计算epoch_num
        total_opt_steps = args.max_steps
        args.train_epoch_num = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        total_opt_steps = len(train_dataloader) // args.gradient_accumulation_steps *  args.train_epoch_num

    # 
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_eps)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=total_opt_steps)

    print("*********** Train ***********")
    print("\tExamples num: %d"%(len(train_dataset)))
    print("\tEpoch num: %d"%(args.train_epoch_num))
    print("\tBatch size: %d"%(args.train_batch_size))
    print("\tGradient accumulation steps: %d"%(args.gradient_accumulation_steps))
    print("\tTrain data loader num: %d"%(len(train_dataloader)))
    print("\tTotol optimization steps: %d"%(total_opt_steps))

    global_step = 0
    train_loss = 0.0
    model.zero_grad() #将梯度置为0
    train_iterator = trange(args.train_epoch_num, desc="Epoch")
    fix_seed(args)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            batch = [ex.to(args.device) for ex in batch]

            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2], #segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) #截断梯度, 防止梯度爆炸
            
            train_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # 当达到设定的累积梯度的step数时, 更新梯度, 学习率, 然后将累积的梯度置零

                #根据WARNING信息, 这二者的顺序在目前的pytorch中需要按如下来写
                optimizer.step() #更新梯度
                scheduler.step() #更新学习率
                model.zero_grad() 
                global_step += 1

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    #根据传入的save_steps的数值保存模型
                    checkpoint_path = os.path.join(args.output_model_dir, 'checkpoint-on-step-{}'.format(global_step))
                    if not os.path.exists(checkpoint_path):
                        os.makedirs(checkpoint_path)
                    model.save_pretrained(checkpoint_path)
                    torch.save(args, os.path.join(checkpoint_path, 'training_args.bin'), _use_new_zipfile_serialization=False)
            
            if args.max_steps > 0 and global_step > args.max_steps:
                # 达到最大step数时, 停止迭代
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            # 达到最大step数时, 停止迭代
            train_iterator.close()
            break

    return global_step, train_loss / global_step

def predict(args, test_dataset, model):
    if not os.path.exists(args.predict_output_dir):
        os.makedirs(args.predict_output_dir)
    
    predict_sampler = SequentialSampler(test_dataset)
    predict_dataloader = DataLoader(test_dataset, sampler=predict_sampler, batch_size=args.predict_batch_size)

    print("*********** Predict ***********")
    print("\tExamples num: %d"%(len(test_dataset)))
    print("\tBatch size: %d"%(args.predict_batch_size))

    results = None
    for batch in tqdm(predict_dataloader, desc="Predict"):
        model.eval()
        batch = [ex.to(args.device) for ex in batch]
        
        with torch.no_grad(): #因为是用模型来预测, 所以此时不需要计算和更新梯度
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2], #segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
            logits = outputs[1]
        if results is None:
            results = logits.detach().cpu().numpy()
        else:
            results = np.append(results, logits.detach().cpu().numpy(), axis=0)
    
    results = np.argmax(results, axis=1)
    results = convert_label_id_back(results)
    # 将输出结果保存
    np.savetxt(args.predict_output_dir + "result.txt", results, fmt="%d")

def load_data(args, tokenizer):
    transformed_data_dir = os.path.join(args.data_dir, "transformed/")
    if not os.path.exists(transformed_data_dir):
        os.makedirs(transformed_data_dir)
    
    if not os.listdir(transformed_data_dir): 
        # 如果路径下还没有文件
        train_features, tokenized_train_examples, test_features, tokenized_test_examples = transfer_to_features(args, tokenizer)
        # 将用预训练模型分词后的句子结果保存
        np.save(os.path.join(transformed_data_dir, "cached_train_tokens.npy"), np.array(tokenized_train_examples), allow_pickle=True)
        np.save(os.path.join(transformed_data_dir, "cached_test_tokens.npy"), np.array(tokenized_test_examples), allow_pickle=True)
        
        # 用 numpy 存储调整后的数据集 (数组, 元素类型为Feature类)
        np.save(os.path.join(transformed_data_dir, "cached_train.npy"), np.array(train_features), allow_pickle=True)
        np.save(os.path.join(transformed_data_dir, "cached_test.npy"), np.array(test_features), allow_pickle=True)   

        # 将转换后的数据进一步转换为tensor
        train_dataset = TensorDataset(torch.tensor([feature.input_ids for feature in train_features], dtype=torch.long),
                                      torch.tensor([feature.input_masks for feature in train_features], dtype=torch.long), 
                                      torch.tensor([feature.segment_ids for feature in train_features], dtype=torch.long),
                                      torch.tensor([feature.label_id for feature in train_features], dtype=torch.long))

        test_dataset = TensorDataset(torch.tensor([feature.input_ids for feature in test_features], dtype=torch.long),
                                     torch.tensor([feature.input_masks for feature in test_features], dtype=torch.long), 
                                     torch.tensor([feature.segment_ids for feature in test_features], dtype=torch.long),
                                     torch.tensor([feature.label_id for feature in test_features], dtype=torch.long))
        
        # 直接保存tensor
        torch.save(train_dataset, os.path.join(transformed_data_dir, "cached_train_tensor"))
        torch.save(test_dataset, os.path.join(transformed_data_dir, "cached_test_tensor"))
        

    else:
        # 如果处理后的数据集已经存在, 就直接读取

        # 读取 numpy 保存的文件, 后续需要用TensorDataSet 转换为tensor
        #train_features = list(np.load(os.path.join(transformed_data_dir, "cached_train.npy"), allow_pickle=True))
        #test_features = list(np.load(os.path.join(transformed_data_dir, "cached_test.npy"), allow_pickle=True))
        #tokenized_train_examples = list(np.load(os.path.join(transformed_data_dir, "cached_train_tokens.npy"), allow_pickle=True))
        #tokenized_test_examples = list(np.load(os.path.join(transformed_data_dir, "cached_test_tokens.npy"), allow_pickle=True))
        #train_dataset = TensorDataset(torch.tensor([feature.input_ids for feature in train_features], dtype=torch.long),
        #                          torch.tensor([feature.input_masks for feature in train_features], dtype=torch.long), 
        #                          torch.tensor([feature.segment_ids for feature in train_features], dtype=torch.long),
        #                          torch.tensor([feature.label_id for feature in train_features], dtype=torch.long))
        #test_dataset = TensorDataset(torch.tensor([feature.input_ids for feature in test_features], dtype=torch.long),
        #                            torch.tensor([feature.input_masks for feature in test_features], dtype=torch.long), 
        #                            torch.tensor([feature.segment_ids for feature in test_features], dtype=torch.long),
        #                            torch.tensor([feature.label_id for feature in test_features], dtype=torch.long))

        # 读取保存的tensor
        train_dataset = torch.load(os.path.join(transformed_data_dir, "cached_train_tensor"))
        test_dataset = torch.load(os.path.join(transformed_data_dir, "cached_test_tensor"))
    
    #print_features(train_features, tokenized_train_examples, test_features, tokenized_test_examples, 9)
    
    return train_dataset, test_dataset

def fix_seed(args):
    """
    设定种子
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def main():
    parser = argparse.ArgumentParser()

    # 必要的参数
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The data dir. Contain both train and test data. The transformed data is also saved here.")
    parser.add_argument("--model_dir", default=None, type=str, required=True,
                        help="The model dir")
    
    # 可选的参数
    parser.add_argument("--output_model_dir", default="./data/models/fine_tuned/", type=str, 
                        help="The output models dir")
    parser.add_argument("--train_epoch_num", default=3, type=int, 
                        help="Number of training epochs")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="Set total number of training steps to perform. Override train_epoch_num.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Steps to warm up.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_eps", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of steps to accumulate before backward/update.")
    parser.add_argument("--train_batch_size", default=8, type=int, 
                        help="Batch size when training")
    parser.add_argument("--predict_batch_size", default=8, type=int, 
                        help="Batch size when predicting")
    parser.add_argument("--train", action="store_true", 
                        help="Train (fine-tune) the model.")
    parser.add_argument("--predict", action="store_true", 
                        help="Use the model to predict.")
    parser.add_argument("--max_seq_len", default=128, type=int,
                        help="The max sentence length after tokenization.")
    parser.add_argument("--save_steps", default=sys.maxsize, type=int, 
                        help="Save checkpoint every given steps")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--no_cuda", action="store_true", 
                        help="Use CPU")
    parser.add_argument("--seed", default=10, type=int, 
                        help="Random seed for initialization")
    parser.add_argument("--overwrite_model", action="store_true", 
                        help="Overwrite the output model")        
    parser.add_argument("--predict_output_dir", default="./results/", 
                        help="Set the output prediction dir. Default is ./results")                            

    args = parser.parse_args()
    
    if os.path.exists(args.output_model_dir) and os.listdir(args.output_model_dir) \
            and args.train and not args.overwrite_model:
        raise ValueError("Outuput model dir already exists, use --overwrite_model to overwrite.")

    # 设置运行程序的设备
    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device =device

    # 固定种子
    fix_seed(args)

    # 加载预训练模型, 加载数据
    config = BertConfig.from_pretrained(pretrained_model_name_or_path=args.model_dir, num_labels=len(BERT_label_id_dic))
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=args.model_dir)
    model = BertForSequenceClassification.from_pretrained(args.model_dir, from_tf=bool('.ckpt' in args.model_dir), config=config)
    model.to(args.device)

    train_dateset, test_dataset = load_data(args, tokenizer)
    
    # fine-tune 并保存
    fine_tuned_model_path = None
    if args.train:
        global_step, train_loss = train(args, train_dateset, model)

        fine_tuned_model_path = os.path.join(args.output_model_dir, "final")
        if not os.path.exists(fine_tuned_model_path):
            os.makedirs(fine_tuned_model_path) 
        model.save_pretrained(fine_tuned_model_path)
        tokenizer.save_pretrained(fine_tuned_model_path)
        torch.save(args, os.path.join(fine_tuned_model_path, 'train_args.bin'), _use_new_zipfile_serialization=False)
        

    # Predict
    if args.predict:
        if args.train:
            # 如果之前fine-tune了, 那就调用fine-tune后的模型
            config = BertConfig.from_pretrained(pretrained_model_name_or_path=fine_tuned_model_path, num_labels=len(BERT_label_id_dic))
            model = BertForSequenceClassification.from_pretrained(fine_tuned_model_path,  from_tf=bool('.ckpt' in args.model_dir), config=config)
            tokenizer = BertTokenizer.from_pretrained(fine_tuned_model_path)
            model.to(args.device)
        predict(args, test_dataset, model)

if __name__ == "__main__":
    main()