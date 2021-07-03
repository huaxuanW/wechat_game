# 2021中国高校计算机大赛-微信大数据挑战赛

我们的代码参考了6月14日周周星（第一名）分享的lightgbm baseline, 加入了512维embedding的pca降维特征, 历史统计特征, 主要统计id特征和交叉特征, 还加入了6个文本特征和keyword, tag 特征的embedding。模型是单模单折。

# 1. 环境配置

- python=3.8.1
- tqdm=4.61.0
- pandas=1.2.4
- lightgbm=3.2.1
- gensim=4.0.1
- numpy=1.20.3
- scikit-learn=0.24.2

# 2. 运行配置

- CPU

- 最小内存要求
  - 16g
- 耗时
  - 测试环境：内存16G，Apple M1
  - 特征/样本生成: 3600s
  - 模型训练及评估: 7000s

# 3. 目录结构
```
.
├── README.md
├── data
│   ├── submit
│   └── wedata
│       └── wechat_algo_data1
├── init.sh
├── requirements.txt
├── src
│   ├── auc.py
│   ├── embedding_feature.py
│   ├── feature_stat.py
│   ├── id_embedding.py
│   ├── model_lgb.py
│   ├── pca.py
│   └── reduce_mem.py
├── train.sh
└── tree.text
```
# 4. 运行流程

- 安装环境: `sh init.sh`
- 数据准备, 模型训练和预测: `sh train.sh`

# 5. 模型介绍

- 模型: lightgbm
- 参数:
  - learning_rate=0.02

  - n_estimators=50000

  - num_leaves=163

  - subsample=0.8,

  - colsample_bytree=0.8,

  - random_state=2021,

  - reg_alpha = 2

  - reg_lambda = 2
  
  - early_stopping_rounds=100

# 模型结果

|stage|得分|查看评论|点赞|点击头像|转发|
|------|------|------|------|------|------|
|在线|0.66789|0.6367|0.643847|0.744033|0.712495|
|离线|0.67423|0.6540|0.648204|0.730438|0.720747|





