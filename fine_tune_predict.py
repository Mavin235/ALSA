import argparse
import numpy as np 
import os 
import logging
import random

import torch
from pytorch_transformers.tokenization_bert import BertTokenizer  
from pytorch_transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from torch import dtype
from torch.utils.data import TensorDataset, DataLoader
from process_data import transfer_to_features
        
def train():
    pass 

def predict():
    pass 

def load_data(args, tokenizer):
    if not os.path.exists(args.output_feature_dir):
        os.mkdir(args.output_feature_dir)
    if not os.listdir(args.output_feature_dir):
        train_features, tokenized_train_examples, test_features, tokenized_test_examples \
                                                    = transfer_to_features(args, tokenizer)
        print(train_features[0].label_id)
        np.save(os.path.join(args.output_feature_dir, "cached_train.npy"), np.array(train_features), allow_pickle=True)
        np.save(os.path.join(args.output_feature_dir, "cached_test.npy"), np.array(test_features), allow_pickle=True)

        
    else:
        train_features = list(np.load(os.path.join(args.output_feature_dir, "cached_train.npy"), allow_pickle=True))
        test_features = list(np.load(os.path.join(args.output_feature_dir, "cached_test.npy"), allow_pickle=True))
        print(test_features[0].label_id)
    
    train_dataset = TensorDataset(torch.tensor([feature.input_ids for feature in train_features], dtype=torch.long),
                                  torch.tensor([feature.input_masks for feature in train_features], dtype=torch.long), 
                                  torch.tensor([feature.segment_ids for feature in train_features], dtype=torch.long),
                                  torch.tensor([feature.label_id for feature in train_features], dtype=torch.long))

    test_dataset = TensorDataset(torch.tensor([feature.input_ids for feature in test_features], dtype=torch.long),
                                 torch.tensor([feature.input_masks for feature in test_features], dtype=torch.long), 
                                 torch.tensor([feature.segment_ids for feature in test_features], dtype=torch.long),
                                 torch.tensor([feature.label_id for feature in test_features], dtype=torch.long))
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
    parser.add_argument("--output_model_dir", default=None, type=str, required=True, 
                        help="The output models dir")
    parser.add_argument("--train_epoch_num", default=3, type=int, 
                        help="Number of training epochs")

    ##Alternative paras
    parser.add_argument("--max_seq_len", default=128, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--save_steps", default=50, type=int, 
                        help="Save checkpoint every given steps")
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

    # Load pre-trained model
    config = BertConfig.from_pretrained(pretrained_model_name_or_path=args.model_dir)
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=args.model_dir)
    model = BertForSequenceClassification.from_pretrained(args.model_dir, from_tf=bool('.ckpt' in args.model_dir), config=config)
    train_dateset, test_dataset = load_data(args, tokenizer)

    # Train
    if args.train:
        pass 

    # Predict
    if args.predict:
        pass 


if __name__ == "__main__":
    main()