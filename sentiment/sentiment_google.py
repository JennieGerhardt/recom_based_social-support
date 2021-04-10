# Title     : TODO
# Objective : 利用谷歌API进行情感分析
# Created by: Little Jennie Fairy
# Created on: 2020/11/14
import os
import pandas as pd
import numpy as np
from google.cloud import language_v1
# 改成自己的json许可证路径
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "F:\PythonWorkspace\depression_tieba\sentiment\My Project 19737-d4d9141772c2.json"



# 向谷歌API提交请求
def analyze(content, client):
    """Run a sentiment analysis request on text within a passed filename."""
    document = language_v1.Document(content=content, language='zh', type_=language_v1.Document.Type.PLAIN_TEXT)
    annotations = client.analyze_sentiment(request={'document': document})

    # Print the results
    return annotations


# 保存分数，边分析边保存，防止出错中断没保存
def save(annotations,f):
    score_list = []
    magnitude_list = []

    for index, sentence in enumerate(annotations.sentences):
        sentence_sentiment = sentence.sentiment.score
        sentence_magnitude = sentence.sentiment.magnitude
        print(
            "index:{},score of {} with magnitude of {}，text:{}".format(index, sentence_sentiment, sentence_magnitude,
                                                                       sentence.text.content)
        )

        score_list.append(sentence_sentiment)
        magnitude_list.append(sentence_magnitude)
    score = np.nanmean(score_list)
    magnitude = np.nanmean(magnitude_list)
    f.write(str(score)+','+str(magnitude)+'\n')
    f.flush()
    # score_dic = {'score': score, 'mag': magnitude}
    # score_csv = pd.DataFrame.from_dict(score_dic, orient='index')
    # score_csv.to_csv('../data/new_data/thread_google_score.csv', encoding='utf-8-sig', index=False, mode='a', header=False)
    return score, magnitude


if __name__ == "__main__":

    client = language_v1.LanguageServiceClient()
    score_lists = []
    magnitude_lists = []
    # 修改为自己的读取路径
    comment = pd.read_csv("../data/thread_content2000.csv", encoding='utf-8')
    content = comment['content_new'].tolist()
    num = len(content)
    f = open('../data/thread_google_score.csv', 'a+')
    for i in range(num):
        print('=============剩余========', num)
        annotations = analyze(content[i], client)

        score, magnitude = save(annotations,f)
        num -= 1
        score_lists.append(score)
        magnitude_lists.append(magnitude)

    # result = pd.DataFrame()
    # result['portrait'] = comment['portrait']
    f.close()
    comment['score'] = score_lists
    comment['magnitude'] = magnitude_lists
    # 修改为自己的保存路径
    comment.to_csv('../data/thread_google_result2000.csv', encoding='utf-8-sig', index=False)
