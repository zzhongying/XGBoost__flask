import pandas as pd
import numpy as np
import sys


# 对训练集进行降维可视化
def data_tsne(data,lable):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt


    """res_U = np.array(res_U)
    test1 = res_U[:,0]"""
    test1 = list(lable)
    tsne = TSNE()
    data = np.array(data)
    data[np.isnan(data)] = 0

    # 检查数据中是否有缺失值
    # Flase:对应特征的特征值中无缺失值
    # True：有缺失值
    print(np.isnan(data).any())
    # 检查是否包含无穷数据
    # False:包含
    # True:不包含
    print(np.isfinite(data).any())
    # False:不包含
    # True:包含
    print(np.isinf(data).any())
    train_inf = np.isinf(data)
    data[train_inf] = 0

    data = tsne.fit_transform(data)  # 进行数据降维,并返回结果

    print(data[50])
    print(data[100])
    print(data[500])
    print(data[1000])
    print(data[2000])
    print(data[5000])

    data = pd.DataFrame(data)
    # 将原始数据中的索引设置成聚类得到的数据类别
    data = pd.DataFrame(data, index=test1)
    print("inde:  ",data.index)
    print(data.index.shape)
    data_tsne = pd.DataFrame(tsne.embedding_, index=data.index)
    print(data_tsne)
    plt.figure(figsize=(8, 6))

    colors = ['blue', 'green', 'orange', 'red', 'cyan', 'magenta', 'yellow', 'black', 'maroon', 'lightblue']  # 可选颜色
    num_class = set(test1)  # 获取类别数目
    print(num_class)

    for class_id in enumerate(num_class):
        ta_data = data_tsne[data_tsne.index == class_id[1]]  # 找出类别为class_id的数据对应的降维结果
        plt.scatter(ta_data[0], ta_data[1], c=colors[class_id[0]], marker='o')
    plt.show()


# 对数据进行空值处理
def process_null_value(data):
    # flag = np.any(data.isnull())
    # print(flag)
    # if flag:
    #     print("数据中存在空值t条数：",end='')
    #     print(data[data.isnull().T.any()].shape[0])
    data.fillna(0, inplace=True)


# 对数据进行inf值处理
def process_inf_value(data):
    replace_value = sys.maxsize
    # data.replace(np.inf, replace_value, inplace=True)
    data.fillna(replace_value, inplace=True)


# 判断数据的合法性并进行处理
def judge_rationality(data):
    # pd.set_option('display.max_columns',None)
    # pd.set_option('display.max_rows',None)
    # 判断数据中是否存在无穷值
    flag = np.isinf(data).any().any()
    if flag:
        print("数据集中存在inf")
        process_inf_value(data)
    # 判断数据中是否存在空值
    flag = np.isnan(data).any().any()
    if flag:
        print("数据集中存在nan")
        process_null_value(data)


if __name__ == "__main__":
    data = pd.read_csv('../static/data/URL.csv')

    # 分离特征以及标签
    data_input = data[[col for col in data.columns if col != 'Class']]
    data_label = data['Class']

    judge_rationality(data_input)

    data_tsne(data_input, data_label)

