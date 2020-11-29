#!/bin/bash
python fine_tune_predict.py \
--data_dir ./data/dataset/ \
--model_dir ./data/models/restaurants_10mio_ep3/ \
--train --train_epoch_num 6 --train_batch_size 32 \
--warmup_steps 50 --predict --predict_batch_size 32 \
--overwrite_model