import argparse
from tokenize import Bracket
import numpy as np 
import os 
import random

import torch
from pytorch_transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_transformers import AdamW, WarmupLinearSchedule

from process_data import transfer_to_features, convert_label_id_back, BERT_label_id_dic
from tqdm import tqdm, trange

def train(args, train_dataset, model):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        total_opt_steps = args.max_steps
        args.train_epoch_num = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        total_opt_steps = len(train_dataloader) // args.gradient_accumulation_steps *  args.train_epoch_num

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
    model.zero_grad()
    train_iterator = trange(args.train_epoch_num, desc="Epoch")
    fix_seed(args)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2], #segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            train_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                #这里的顺序注意一下
                optimizer.step() #更新梯度
                scheduler.step() #更新学习率
                model.zero_grad()
                global_step += 1

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    #根据传入的save_steps的数值保存模型
                    checkpoint_path = os.path.join(args.output_model_dir, 'checkpoint-on-step-{}'.format(global_step))
                    if not os.path.exists(checkpoint_path):
                        os.makedirs(checkpoint_path)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(checkpoint_path)
                    torch.save(args, os.path.join(checkpoint_path, 'training_args.bin'), _use_new_zipfile_serialization=False)
            
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
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
    #for batch in predict_dataloader:
        model.eval()
        batch = tuple(ex.to(args.device) for ex in batch)

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
    np.savetxt(args.predict_output_dir + "result.txt", results, fmt="%d")
    
    
        

def load_data(args, tokenizer):
    if not os.path.exists(args.output_feature_dir):
        os.makedirs(args.output_feature_dir)
    
    if not os.listdir(args.output_feature_dir):
        train_features, tokenized_train_examples, test_features, tokenized_test_examples \
                                                    = transfer_to_features(args, tokenizer)
        #print(train_features[0].label_id)
        np.save(os.path.join(args.output_feature_dir, "cached_train.npy"), np.array(train_features), allow_pickle=True)
        np.save(os.path.join(args.output_feature_dir, "cached_test.npy"), np.array(test_features), allow_pickle=True)   
        np.save(os.path.join(args.output_feature_dir, "cached_train_tokens.npy"), np.array(tokenized_train_examples), allow_pickle=True)
        np.save(os.path.join(args.output_feature_dir, "cached_test_tokens.npy"), np.array(tokenized_test_examples), allow_pickle=True)

    else:
        # 如果处理后的数据集已经存在, 就直接读取
        train_features = list(np.load(os.path.join(args.output_feature_dir, "cached_train.npy"), allow_pickle=True))
        test_features = list(np.load(os.path.join(args.output_feature_dir, "cached_test.npy"), allow_pickle=True))
        #tokenized_train_examples = list(np.load(os.path.join(args.output_feature_dir, "cached_train_tokens.npy"), allow_pickle=True))
        #tokenized_test_examples = list(np.load(os.path.join(args.output_feature_dir, "cached_test_tokens.npy"), allow_pickle=True))
        pass 
    
    #for i in range(9):
        #print("*** Example ***")
        #print("tokens: %s" % (tokenized_train_examples[i]))
        #print("input_ids: %s" % (train_features[i].input_ids))
        #print("input_masks: %s" % (train_features[i].input_masks))
        #print("segment_ids: %s" % (train_features[i].segment_ids))
        #print("label_id = %d)"% (train_features[i].label_id))
    
    train_dataset = TensorDataset(torch.tensor([feature.input_ids for feature in train_features], dtype=torch.long),
                                  torch.tensor([feature.input_masks for feature in train_features], dtype=torch.long), 
                                  torch.tensor([feature.segment_ids for feature in train_features], dtype=torch.long),
                                  torch.tensor([feature.label_id for feature in train_features], dtype=torch.long))

    test_dataset = TensorDataset(torch.tensor([feature.input_ids for feature in test_features], dtype=torch.long),
                                 torch.tensor([feature.input_masks for feature in test_features], dtype=torch.long), 
                                 torch.tensor([feature.segment_ids for feature in test_features], dtype=torch.long),
                                 torch.tensor([feature.label_id for feature in test_features], dtype=torch.long))
    #train_dataset = torch.load(os.path.join(args.output_feature_dir, "cached_train_tensor"))
    #test_dataset = torch.load(os.path.join(args.output_feature_dir, "cached_test_tensor"))
    #torch.save(train_dataset, os.path.join(args.output_feature_dir, "cached_train_tensor"))
    #torch.save(test_dataset, os.path.join(args.output_feature_dir, "cached_test_tensor"))
    return train_dataset, test_dataset

def fix_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def main():
    parser = argparse.ArgumentParser()

    ##Required paras
    parser.add_argument("--raw_data_dir", default=None, type=str, required=True,
                        help="The raw data dir. Contain both train and test data.")
    parser.add_argument("--model_dir", default=None, type=str, required=True,
                        help="The model dir")
    
    ##Alternative paras
    parser.add_argument("--output_model_dir", default="./data/models/fine_tuned/", type=str, 
                        help="The output models dir")
    parser.add_argument("--train_epoch_num", default=3, type=int, 
                        help="Number of training epochs")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="Set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--train_batch_size", default=8, type=int, 
                        help="Batch size when training")
    parser.add_argument("--predict_batch_size", default=8, type=int, 
                        help="Batch size when predicting")
    parser.add_argument("--max_seq_len", default=128, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--save_steps", default=200, type=int, 
                        help="Save checkpoint every given steps")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_eps", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--no_cuda", action="store_true", 
                        help="Do not use CUDA")
    parser.add_argument("--seed", default=10, type=int, 
                        help="Random seed for initialization")
    parser.add_argument("--train", action="store_true", 
                        help="Train (fine-tune) the model.")
    parser.add_argument("--predict", action="store_true", 
                        help="Use the model to predict.")
    parser.add_argument("--overwrite_model", action="store_true", 
                        help="Overwrite the output model")        
    parser.add_argument("--output_feature_dir", default="./data/transformed/", 
                        help="Set the output feature dir. Default is data/transformed")
    parser.add_argument("--predict_output_dir", default="./results/", 
                        help="Set the output prediction dir. Default is ./results")                            
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Use all checkpoints to predict")

    args = parser.parse_args()
    
    
    if os.path.exists(args.output_model_dir) and os.listdir(args.output_model_dir) and args.train and not args.overwrite_model:
        raise ValueError("Outuput model dir already exists, use --overwrite_model to overwrite.")

    # Set device
    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device =device

    fix_seed(args)

    # Load pre-trained model and dataset
    config = BertConfig.from_pretrained(pretrained_model_name_or_path=args.model_dir, num_labels=len(BERT_label_id_dic))
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=args.model_dir)
    model = BertForSequenceClassification.from_pretrained(args.model_dir, from_tf=bool('.ckpt' in args.model_dir), config=config)
    train_dateset, test_dataset = load_data(args, tokenizer)
    model.to(args.device)

    
    # Train and save
    fine_tuned_model_path = None
    if args.train:
        global_step, train_loss = train(args, train_dateset, model)

        fine_tuned_model_path = os.path.join(args.output_model_dir, "final")
        if not os.path.exists(fine_tuned_model_path):
            os.makedirs(fine_tuned_model_path)
        model_to_save = model.module if hasattr(model, 'module') else model 
        model_to_save.save_pretrained(fine_tuned_model_path)
        tokenizer.save_pretrained(fine_tuned_model_path)
        torch.save(args, os.path.join(fine_tuned_model_path, 'train_args.bin'), _use_new_zipfile_serialization=False)
        

    # Predict
    if args.predict:
        if args.train:
            config = BertConfig.from_pretrained(pretrained_model_name_or_path=fine_tuned_model_path, num_labels=len(BERT_label_id_dic))
            model = BertForSequenceClassification.from_pretrained(fine_tuned_model_path,  from_tf=bool('.ckpt' in args.model_dir), config=config)
            tokenizer = BertTokenizer.from_pretrained(fine_tuned_model_path)
            model.to(args.device)
        predict(args, test_dataset, model)



if __name__ == "__main__":
    main()