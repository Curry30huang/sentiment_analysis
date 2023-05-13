# 参考网址

KNN: [KNN](https://www.kaggle.com/code/nishankbhola/knn-amazon-sentimental-analysis)

逻辑回归： [逻辑回归](https://www.kaggle.com/code/saishan/sentiment-analysis-logregre-vs-cudnnlstm)

LSTM+GRU: [LSTM+GRU](https://www.kaggle.com/code/dijiswiki/lstm-gru-sentiment-analysis-on-amazon-review)

Bag of Words（词袋技术） and TF-IDF  [SentimentAnalysis | Kaggle](https://www.kaggle.com/code/aravindanr22052001/sentimentanalysis)

# 系统架构

- 数据预处理：将训练数据和测试数据导出到本地 csv 格式。
- 模型构建，格式 py：每个模型一个文件。
- 模型训练：导入处理后的数据，根据 python 模型进行训练，最后导出模型。
- 模型测试：选择模型文件，导入模型，导入测试集，测试数据。

# TODO任务：

数据预处理：

- [ ] 参考给定LSTM+GRU的网址，对基本文本数据先进行处理。
- [ ] 对数据进行分别进行词袋技术、TF—IDF、词向量(torch.nn的，参考LSTM+GRU的网址)技术处理。
