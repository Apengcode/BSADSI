# 打分器
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ResultTest:

    def resultTest(self,BestModel):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = BestModel
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

        # 加载测试数据集
        test_data = pd.read_csv("COLD/test_label_0.csv")

        # 对测试数据集进行标记和填充
        test_texts = test_data['TEXT'].tolist()
        test_labels = test_data['label'].tolist()

        # 将文本转换为token
        test_inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt")

        # 创建TensorDataset和DataLoader
        test_dataset = TensorDataset(test_inputs['input_ids'].to(device), test_inputs['attention_mask'].to(device),
                                     torch.tensor(test_labels).to(device))
        test_loader = DataLoader(test_dataset, batch_size=32)

        # 在测试集上进行预测
        model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                _, predicted = torch.max(logits, 1)
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # 计算评估指标
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)

        del model
        return f1, accuracy
        # print("Accuracy:", accuracy)
        # print("Precision:", precision)
        # print("Recall:", recall)
        # print("F1 Score:", f1)


