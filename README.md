# [NJUAI] NLP课程作业二: 方面级别情感分析
## 实验概述
* 本次实验要求实现方面级别情感分析 (Aspect-Level Sentiment Analysis)
* 可以使用预训练语言模型

## 文件及运行方式说明 
### 文件说明
* `process_data.py` 中包含对原始数据进行处理的相关函数, 在`fine_tune_predict.py`中调用
* `fine_tune_predict.py` 是主要文件, 可以根据传入的参数进行训练 (微调) 或者预测
  * 必须传入的参数有
    * `--data_dir`, 指定数据集的路径
    * `--model_dir`, 指定要使用的模型的路径. 可选参数中
  * 可选参数中, 常用的有
    * `--train`和`--predict`, 控制程序是否执行微调和预测. 如果都不传入, 则程序在处理完数据后就结束
    * `--train_batch_size`和`--predict_batch_size`, 设置微调, 预测时 batch 的大小
    * `--output_model_dir`, 设置微调后模型的保存路径
    * `--train_epoch_num`和`-max_steps`, 前者设置微调时的 epoch 大小, 所有 batch 迭代一遍为一个 epoch ; 后者设置最大的训练步数, 训练一个 batch 为一步. 如果指定了后者的的大小, 那么前者的数值将会被重新计算覆盖掉. 
      * 使用这两个参数时二选一即可
    * `--warmup_step`, 设置预热的步数, 在这些步数内学习率由小变大
    * `--save_steps`, 设置存储的步数, 默认为最大整形值, 即不会在中间步存储模型
    * `--overwrite_model`, 设置是否覆盖掉之前保存的模型, 如果每次都在同一个路径输出, 则需要加上此选项
  * 其他参数的说明请参看代码文件. 
  * 由于传入参数较多, 所以建议将命令放入一个脚本来执行. 这里也提供了一个脚本`run.sh`, 切换到代码文件所在的目录下后, 在终端执行
  
        ./run.sh
    即可运行
  * 训练集和模型的默认目录结构如下

        data
        |--dataset
        |   |--train.txt
        |   |--test.txt
        |   |--transformed/
        |
        |--models
            |--restaurants_10mio_ep3/
            |--fine_tuned/
    其中`transformed` 文件夹下存放处理后的数据集, 由程序运行时创建. `restaurants_10mio_ep3/` 是放置预训练模型的路径
### 配置运行环境
* `Python3.6`
* 建议用`virtualenv`和`virtualenvwrapper`创建一个虚拟环境, 然后用命令
  
        pip install -r requirements.txt
  批量安装所需要的包.
* 此外, 还需要手动安装`pytorch-1.1.0` 版本与`CUDA-10.0`版本. 如果已经有其他版本的`Pytorch` 和 `CUDA`, 可以先试着运行一下看能否执行.
* 由于我的轻薄本显存不够, 所以实验所得结果是用 [Google Colab](https://colab.research.google.com/) 执行得出的, 其环境为`Python-3.6 + Pytorch-1.7.0 + CUDA-10.1`
  
## 实现方法
### 预训练模型来源
* 采用的预训练模型是一个 [BERT](https://drive.google.com/file/d/1DmVrhKQx74p1U5c7oq6qCTVxGIpgvp1c/view) 模型
* 此模型来自论文 [*Adapt or Get Left Behind: Domain Adaptation through BERT Language Model Finetuning for Aspect-Target Sentiment Classification*](https://arxiv.org/pdf/1908.11860v2.pdf)    

  是一个在特定领域上进行预训练之后的模型. 预训练的语料来源为 Amazon 上笔记本电脑的评价, Yelp 上餐馆评价. 作者一共提供了 3 种预训练的模型, 笔记本电脑评价预训练的模型, 餐馆评价预训练的模型, 以及用两个数据集一起预训练的模型  
* 更多的信息可以访问此篇论文的 [GitHub](https://github.com/deepopinion/domain-adapted-atsc) 主页, 本次实验中有关输入数据的格式以及调用模型的规范参考了该 repository.
### 数据的处理
* 此 BERT 模型在ALSA任务中输入数据的格式如下
  
        [CLS] 句子 [SEP] 方面词 [SEP]                # 分割之后的句子以及方面词
        ..., ..., ..., ..., ..., ..., ..., ..., ... # 词向量
        ..., ..., ..., ..., ..., ..., ..., ..., ... # 指示单词出现的向量
        ..., ..., ..., ..., ..., ..., ..., ..., ... # 段向量, 区分两个句子
        ...                                         # 情感极性
        

  以训练集中的第二句为例

        [CLS] the food was lou ##sy - too sweet or too salty and the portions 
        tiny . [SEP] food [SEP]
        
        [101, 1996, 2833, 2001, 10223, 6508, 1011, 2205, 4086, 2030, 2205, 23592, 1998, 
        1996, 8810, 4714, 1012, 102, 2833, 102, 0, ..., 0]

        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,..., 0]

        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, ..., 0]

        1
  句子的分割是由预训练模型完成的, 句子中出现`##`是因为BERT采用了WordPiece的方法, 它会将单词拆成子词单元.    

  词向量, 指示单词出现的向量, 段向量的长度是一样的, 是我们传参时用`--max_seq_len`规定的, 程序中默认的长度是128 (BERT中假设的最长句子为512). 在每个向量中句子长度之后的位置上补0.  
  在这三个向量中, 指示单词出现的向量, 段向量都是手动去编码的, 词向量是由模型导出的.
* 基于以上的知识, 在`process_data.py`中定义了用于将原始数据转换为目标格式的函数`transfer_to_feature()`.  
    
  每个转换后的输入称作一个feature, 用`Feature`类来描述 

        class Feature:
            def __init__(self, input_ids, input_masks, segment_ids, label_id):
                self.input_ids = input_ids
                self.input_masks = input_masks
                self.segment_ids = segment_ids
                self.label_id = label_id
  生成每个feature的`input_ids`, `input_masks` 和 `segment_ids` 由函数 `tokenize()` 实现. 

  `tokenize()`函数中, 首先调用模型的tokenizer的`tokenize()` 方法对句子进行分割, 然后调用它的`convert_tokens_to_ids`来得到`input_ids`

  生成feature的`label_id`, 即情感极性时, 要注意到预训练模型描述三个极性的数值与实验给的数据集中的描述并不相同, 所以通过建立两个字典`origin_label_id_dic`, `BERT_label_id_dic`, 然后用`convert_label_id()`函数来完成转换. 而在预测结束后, 调用`convert_label_id_back()`函数将极性转换为实验数据集中的标签.

* 在`fine_tune_predict.py`中, 用`load_data()`函数来加载微调数据集和测试数据集. 在此函数中, 如果输出转换后数据的路径不存在 (默认为`data/dataset/transformed/`) , 则创建路径并调用`transfer_to_features()`转换数据, 然后再进一步用`TensorDataset`封装成张量, 最后保存数据; 否则直接读入已转换的数据

### 微调预训练模型
* 用`RandomeSampler()`在训练集中取样, 用`Dataloader()`生成一个dataloader (`train_dataloader`)     
  
* `train_dataloader`的长度等于 batch 的数量. 如果在传参时指定了`--max_steps`, 那么用`args.max_steps`, `len(train_dataloader)` 以及 `arg.gradient_accumulation_step` 重新计算`args.train_epoch_num`的值. `--graident_accumulation_step` 参数设置梯度累积的步数, 当显存较小时可以将其设置为大于1的值. 同时也要计算出总的执行步数`tot_train_steps`.
  
* 根据论文, 作者采用的计算梯度的方法是Adam算法, 并且使用了学习率预热的方法. 用`AdamW()`来生成更新梯度的`optimizer`, 用`WarmupLinearSchedule()`生成调整学习率的`scheduler`.  
  
* 以上准备工作就绪后, 就可以进入迭代, 对模型进行微调 (fine-tune) . 需要注意的是, 该预训练模型要求的输入`inputs`是一个字典类型, 它的键已经固定
  
### 使用微调后的模型进行预测
* 此时, 用`SequentailSampler()`在测试集中取样, 因为要按照顺序输出预测结果
* 然后进入迭代, 只不过在预测时, 要在传入inputs时设置 

        with torch.no_grad():
  因为此时只用来预测, 不用更新梯度
* 根据`pytorch_transformers`工具包的文档, 从输出的`logits`中提取出`results`, 然后用`convert_label_id_back()`将它转换为实验数据集中情感极性的`id`, 最后保存到指定路径下的`txt`文档中




