import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import set_seed, BertTokenizer, BertForSequenceClassification, AdamW
import csv
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from ResultTest import ResultTest
from MyUtils import MyUtils
from MBSA import MBSA

# 配置日志记录
import logging

logging.basicConfig(filename='MBCF.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 全局变量
Scorer = 0  # 初始化打分器
counter = 0  # 一个计数器，代码最后的时候用到的。

for epochNum in range(1, 100):

    # 从 CSV 文件加载数据集
    train_data = pd.read_csv('COLD/train.csv')
    dev_data = pd.read_csv('COLD/dev.csv')
    # 加载 BERT 模型和分词器
    model_name = 'bert-base-chinese'
    # 设置随机种子
    set_seed(42)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 二分类任务

    class CustomDataset(Dataset):
        def __init__(self, data, tokenizer, max_length=128):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            if index < len(self.data):
                text = self.data.iloc[index]['TEXT']
                label = int(self.data.iloc[index]['label'])
                encoding = self.tokenizer(text, add_special_tokens=True, truncation=True, max_length=self.max_length,
                                          padding='max_length', return_tensors='pt')
                input_ids = encoding['input_ids'].squeeze()
                attention_mask = encoding['attention_mask'].squeeze()
                return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label}

    batch_size = 32
    train_dataset = CustomDataset(train_data, tokenizer)
    dev_dataset = CustomDataset(dev_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

    # 定义优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def dev():
        model.eval()

        data_loader = dev_loader
        prelabel_list = []  # 保存预测的所有标签
        labels_list = []  # 保存实际的所有标签

        loss_sum = 0.0  # 初始化累计损失

        with torch.no_grad():
            for batch in tqdm(data_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)  # 计算当前批次的损失

                # 累计损失
                loss_sum += loss.item() * len(labels)

                prelabel_list.extend(logits.argmax(dim=1).tolist())
                labels_list.extend(labels.tolist())
        model.train()

        return f1_score(labels_list, prelabel_list), accuracy_score(labels_list, prelabel_list), loss_sum / len(data_loader.dataset)


    # 模型微调训练
    model.to(device)
    model.train()
    best_F1 = 0
    best_acc = 0
    best_loss = 1000000
    best_epoch = 0

    # 创建一个tqdm进度条
    with tqdm(total=100, desc='Training') as pbar:
        for epoch in range(100):
            for batch in tqdm(train_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            # 验证
            dev_F1, dev_acc, avg_loss = dev()

            if best_loss >= avg_loss:
                best_F1 = dev_F1
                best_epoch = epoch
                best_acc = dev_acc
                best_loss = avg_loss
                # 保存微调后的模型
                BestModel = 'BestModel'+str(epochNum)
                model.save_pretrained(BestModel)  # 保存模型的路径
                tokenizer.save_pretrained(BestModel)  # 保存tokenizer的路径
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix(best_F1=best_F1, best_acc=best_acc, best_loss=best_loss, best_epoch=best_epoch+1)

            if epoch - best_epoch >= 5:
                del model
                break

    # 根据“打分”与“偏差感知”来决定是否 数据增强，再来一轮。

    result = ResultTest()
    test_f1, test_acc = result.resultTest(BestModel)

    logging.info(f"第{epochNum}次迭代训练,最终测试集结果：Acc为{test_acc}")

    flag, word_list=MBSA().MBSA_score(BestModel)
    if flag == 0:
        break
    elif test_acc >= Scorer:  # 模型偏差仍然存在，但这一轮的分数比最好的得分高，说明有成长空间，可以进行下一轮。
        Scorer = test_acc
        counter = 0  # 有上升趋势，刷新计数器。
        MyUtils().JAD(word_list)
    elif counter <= 5:  # 这一轮得分比不上最高的得分，给框架5次机会。
        MyUtils().JAD(word_list)
        counter = counter + 1
    else:
        break






