"""
计算现有数据集每个数据集在现有模型下的正确分类情况
所有处理好的数据集存放在 ../static/data/
"""

import pandas as pd
import os
from dataInterface import myInterface


# 计算每个类别的数量
def count_categories(data):
    count = {}
    temp = []
    for i in data["Class"]:
        if i not in temp:
            temp.append(i)
            count[i] = 1
        else:
            count[i] += 1
    return count


if __name__ == "__main__":
    basicData = myInterface.basicData()

    root_path = '../static/data'
    filenames = os.listdir(root_path)

    for i in range(len(filenames)):
        ite = i + 1
        data_iter = 'data' + str(ite)
        print(data_iter)
        path = basicData.get_data_path(data_iter)
        print(path)
        basicData.replace_data(data_iter)

        # 输出各类别数量
        categories = pd.read_csv(path, usecols=['Class'])
        df_num = categories.value_counts()
        print(df_num.index)
        recall = basicData.get_recallRate()
        temp = {'召回率': recall.values()}
        mapCategories = basicData.dataInformation['mapCategories']
        try:
            df_recall = pd.DataFrame(temp, index=[mapCategories[key] for key in recall.keys()])
        except KeyError:
            df_recall = pd.DataFrame(temp, index=recall.keys())
        data = pd.concat([df_recall, df_num], axis=1)
        print(data)

        acc = basicData.get_accuracy()
        print(data_iter, '的正确率是：', acc, '%')
