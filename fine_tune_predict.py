import argparse
from pytorch_transformers.tokenization_bert import BertTokenizer 
import torch 
import numpy as np 
import os 
import logging
import random

from pytorch_transformers import BertConfig, BertForSequenceClassification, BertTokenizer
        
def train():
    pass 

def predict():
    pass 

def load_data():
    pass

def fix_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def main():
    parser = argparse.ArgumentParser()

    ##Required paras
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Contain both train and test data.")
    parser.add_argument("--model_dir", default=None, type=str, required=True,
                        help="The model dir")
    parser.add_argument("--output_dir", default=None, type=str, required=True, 
                        help="The output files dir")
    parser.add_argument("--train_epoch_num", default=None, type=int, 
                        help="Number of training epochs")

    ##Alternative paras
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
    parser.add_argument("--overwrite_output", action="store_true", 
                        help="Overwrite the output files")                        

    args = parser.parse_args()
    
    
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.train and not args.overwrite_output:
        raise ValueError("Outuput dir already exists, use --overwrite_output to overwrite.")

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


    # Train
    if args.train:
        pass 

    # Predict
    if args.predict:
        pass 


if __name__ == "__main__":
    main()