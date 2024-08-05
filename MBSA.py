import torch
import csv
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from MyUtils import MyUtils
from collections import Counter
import logging

class MBSA:
    logging.basicConfig(filename='MBSA.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def detectDeviations(self,BestModel):
        k = 0.5
        csv_file_path='COLD/dev.csv'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载预训练的BERT模型和tokenizer
        model_name = BestModel
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        train_texts = []
        train_labels = []
        with open(csv_file_path, 'r', encoding='UTF-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                label = row['label']
                train_labels.append(int(label))
                text = row['TEXT']
                train_texts.append(text)
        imbalancedData = []
        # 构建一个列表：收集标签是0，但模型判断是1的数据。（假阳性数据）
        false_positive_list = []

        for i in range(len(train_texts)):
            inputs = tokenizer(train_texts[i], truncation=True, padding=True, return_tensors='pt').to(device)
            outputs = model(**inputs)
            predictions = outputs.logits.argmax(dim=1).item()
            probabilities = torch.sigmoid(outputs.logits)

            class1_prob = probabilities[0][0]
            class2_prob = probabilities[0][1]

            fin_prob = class2_prob.item() - class1_prob.item()
            # 采集模型认为攻击性概率高，但是分类是错误的数据
            if predictions != train_labels[i] and fin_prob > k:
                imbalancedData.append(train_texts[i])
            # 标签是0，但模型判断是1的数据
            if train_labels[i] == 0 and predictions ==1:
                false_positive_list.append(train_texts[i])
        del model
        return imbalancedData , false_positive_list

    def Mask_verification(self,wordlist,BestModel):
        logging.info("掩码验证：")
        ver_data_list=[]
        for data in wordlist:
            mask_data = data[1].replace(data[0], "XX")  # 将偏差数据掩盖掉中性词，替换为 XX
            ver_data_list.append([data[0],mask_data])

        k = 0.4

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载预训练的BERT模型和tokenizer
        model_name = BestModel
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)


        # 存放验证后的偏差数据
        basic_data_list = []

        for one_data_list in ver_data_list:
            inputs = tokenizer(one_data_list[1], truncation=True, padding=True, return_tensors='pt').to(device)
            outputs = model(**inputs)
            predictions = outputs.logits.argmax(dim=1).item()
            probabilities = torch.sigmoid(outputs.logits)

            class1_prob = probabilities[0][0]
            class2_prob = probabilities[0][1]

            fin_prob = class2_prob.item() - class1_prob.item()
            # 对掩盖数据判断为无毒了，或者置信度下降了
            if predictions == 0 or fin_prob <= k:
                basic_data_list.append(one_data_list)

                # logging.info(f"中性词是:{one_data_list[0]},掩盖掉中性词的数据为：{one_data_list[1]}")
                # logging.info(f"非攻击性概率:{class1_prob.item()}")
                # logging.info(f"攻击性概率:{class2_prob.item()}")
                # logging.info(f"最终判定:{predictions}")

        del model
        return basic_data_list

    # def calculate_pmi(word, label_1):
    #     csv_file_path = 'COLD/train.csv'
    #     train_texts = []
    #     train_labels = []
    #     joint_num=0
    #     word_num=0
    #     label_num=0
    #     with open(csv_file_path, 'r', encoding="utf-8") as csvfile:
    #         reader = csv.DictReader(csvfile)
    #         for row in reader:
    #             label = row['label']
    #             train_labels.append(int(label))
    #             text = row['TEXT']
    #             train_texts.append(text)
    #     for i in range(len(train_texts)):
    #         if word in train_texts[i]:
    #             word_num+=1
    #             if train_labels[i]==label_1:
    #                 joint_num+=1
    #         if train_labels[i]==label_1:
    #             label_num+=1
    #     joint_prob = joint_num/len(train_texts)
    #     marginal_word = word_num/len(train_texts)
    #     marginal_label = label_num/len(train_labels)
    #     if joint_prob == 0 or marginal_word == 0 or marginal_label == 0:
    #         return 0
    #     pmi = np.log(joint_prob / (marginal_word * marginal_label))
    #     return pmi

    # 添加掩码验证的中性词筛选
    def MBSA_score(self,BestModel):
        print('开始偏差感知!!!')
        flag = 0
        dev_biseText_list, false_positive_list = self.detectDeviations(BestModel)  # 获取验证集中的偏差数据和假阳性数据。

        t = " ".join(dev_biseText_list)
        word_list = MyUtils().custom_cut(t)
        word_counts = Counter(word_list)
        sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        word_list1 = []

        allcount = 0

        for rank, (word, count) in enumerate(sorted_word_counts, start=1):
            # print(f"第{rank}名: {word} 出现次数: {count}")
            # if count > 3 and len(word) > 1 and calculate_pmi(word,1)>0:
            if count > 3 and len(word) > 1:
                word_list1.append(word)
        # logging.info(f"第一轮筛选后的词为")
        # print(f"第一轮筛选后的词为：{word_list1}")

        temp_list = []
        for word2 in word_list1:
            for data in dev_biseText_list:
                if word2 in data:
                    temp_list.append([word2, data])

        word_list2 = self.Mask_verification(temp_list,BestModel)

        csv_file_path = 'COLD/train.csv'
        train_texts = []
        train_labels = []
        with open(csv_file_path, 'r', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                label = row['label']
                train_labels.append(int(label))
                text = row['TEXT']
                train_texts.append(text)
        word_list3 = []
        word_set=set()
        for wordtemp in word_list2:
            word_set.add(wordtemp[0])
        word_list2_onlyword=list(word_set)

        for listindex in range(len(word_list2_onlyword)):
            negative_data_num=0
            positive_data_num=0
            for dataindex in range(len(train_texts)):
                if word_list2_onlyword[listindex] in train_texts[dataindex]:
                    if train_labels[dataindex]==0:
                        negative_data_num+=1
                    if train_labels[dataindex]==1:
                        positive_data_num+=1
            cha=positive_data_num-negative_data_num
            if cha>1:
                # word_list3.append([word_list2_onlyword[listindex], cha*Rate])
                word_list3.append([word_list2_onlyword[listindex],cha//3+1])

        if len(word_list3) > 0:
            flag = 1
        return flag, word_list3