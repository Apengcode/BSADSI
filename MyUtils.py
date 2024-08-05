import jieba
import csv
import logging
from transformers import AutoModel, AutoTokenizer
class MyUtils:
    logging.basicConfig(filename='MyUtils.log', level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

    def custom_cut(self, text):
        with open("stop_words.txt", "r", encoding="utf-8") as f:
            stopwords = set(f.read().splitlines())
        words = jieba.cut(text, cut_all=False)
        filtered_words = [word for word in words if word not in stopwords]
        return filtered_words

    def JAD(self, word_list):

        l=len(word_list)

        logging.info(f"偏差中性词数量为{l}个")

        print("开始数据增强......")
        model_name = "chatglm2-6b"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()

        prompt = """
                生成一个包含“xx”这个词，但不具有攻击性的句子,要求不超过40个字。
                """

        for word in word_list:
            logging.info(f"扩充包含中性词{word[0]}的数据{word[1]}条")
            for num in range(word[1]):
                response, his = model.chat(tokenizer, prompt.replace('xx', word[0]),temperature=1.6)
                if len(response) > 40 : # 生成的句子长度超过40，句子就要舍弃了。
                    continue

                logging.info(f"根据中性词：{word[0]}，重新生成的数据：{response}")

                csv_file = 'COLD/train.csv'

                # 将数据写入 CSV 文件,以追加模式写入，指定编码为 UTF-8
                with open(csv_file, 'a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([''] * 3 + [0, response])  # 适配cold数据集

        del model
        print("数据定向增强完成！")

