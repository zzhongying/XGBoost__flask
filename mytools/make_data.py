import xgboost as xgb
import pandas as pd
import os
import dataInterface.finalXGBoost as myxgb
import numpy as np


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
    for category in count:
        print("类别{}：{}".format(category, count[category]))


# 输出每个数据集的类别分布情况
def all_data_category_distribute():
    root_path = '../static/data'
    filenames = os.listdir(root_path)

    for filename in filenames:
        path = '../static/data/' + filename
        print(path)
        categories = pd.read_csv(path, usecols=['Class'])
        df_num = categories.value_counts()
        print(df_num)


# 对于选定的数据集 URL.csv 进行训练集和验证集的划分
def divide_data(path=None):
    if path is None:
        path = r'../static/data/URL.csv'
    data = pd.read_csv(path)
    # 将数据进行均分
    train_x, test_x, train_y, test_y = myxgb.prepare_data(data, 0.5)
    data = pd.concat([train_x, train_y], axis=1)
    data.to_csv(r'../static/make_data/train.csv', index=False)
    data = pd.concat([test_x, test_y], axis=1)
    data.to_csv(r'../static/make_data/test.csv', index=False)


# 输出划分数据集的类别数量
def data_distribute():
    train = pd.read_csv('../static/make_data/train.csv')
    test = pd.read_csv('../static/make_data/test.csv')
    print(train['Class'].value_counts())
    print(test['Class'].value_counts())


# 实现类别数量不均衡
def unbalance_category():
    train = pd.read_csv('../static/make_data/train.csv')
    test = pd.read_csv('../static/make_data/test.csv')

    # Defacement、benign、phishing、malware、spam
    temp_data = train[(train['Class'] == 'Defacement') | (train['Class'] == 'benign')
                      | (train['Class'] == 'phishing')].sample(300)
    num_data = pd.concat([temp_data, train[(train['Class'] != 'Defacement') & (train['Class'] != 'benign')
                                           & (train['Class'] != 'phishing')]], ignore_index=True)
    num_data.to_csv('../static/make_data/num_train.csv')
    temp_data = test[(test['Class'] == 'malware') | (test['Class'] == 'spam')].sample(300)
    num_test = pd.concat([temp_data, test[(test['Class'] != 'malware') & (test['Class'] != 'spam')]], ignore_index=True)
    num_test.to_csv('../static/make_data/num_test.csv')


# 将训练模型可视化为树结构
# graphviz安装在anaconda的全局区域，需切换虚拟环境
def draw_model(model):
    # import graphviz
    xgb.to_graphviz(model, num_trees=2)
    xgb.plot_tree(model, num_trees=2)
    xgb.to_graphviz(model, num_trees=1, leaf_node_params={'shape': 'plaintext'})
    # 保存所有树图
    for i in range(5):
        src = xgb.to_graphviz(model, num_trees=i)
        src.view("D:/网络安全可视化/XGBoost_Tree/tree"+"_"+str(i))


# 计算质心
def cal_center_mass(data):
    center_mass = np.mean(data, axis=0)
    return center_mass


# 平行偏移
def parallel_offset():
    family = ['Defacement', 'benign', 'phishing', 'malware', 'spam']
    train_data = pd.read_csv(r'../static/make_data/num_train.csv')
    # test_data = pd.read_csv(r'../static/make_data/test.csv')

    c_mass = []
    for category in family:
        temp_data = train_data[train_data['Class'] == category]
        temp_c_mass = cal_center_mass(temp_data)
        c_mass.append(temp_c_mass)
    vector1 = np.array(c_mass[4]) - np.array(c_mass[0])
    vector2 = np.array(c_mass[3]) - np.array(c_mass[1])

    temp_data = train_data[train_data['Class'] == 'Defacement'].iloc[:, :-1].apply(lambda x: np.array(x) + vector1,
                                                                                   axis=1)
    # print(temp_data)
    # print(type(temp_data))
    temp_data = [list(value) for value in temp_data]
    # print(temp_data)
    temp_data = pd.DataFrame(temp_data, columns=train_data.columns[:-1])
    temp_data['Class'] = 'Defacement'
    # print(temp_data)
    train_data = pd.concat([temp_data, train_data[train_data['Class'] != 'Defacement']], axis=0, ignore_index=True)
    print(train_data)

    temp_data = train_data[train_data['Class'] == 'benign'].iloc[:, :-1].apply(lambda x: np.array(x) + vector2, axis=1)
    temp_data = [list(value) for value in temp_data]
    temp_data = pd.DataFrame(temp_data, columns=train_data.columns[:-1])
    temp_data['Class'] = 'benign'
    train_data = pd.concat([temp_data, train_data[train_data['Class'] != 'benign']], axis=0, ignore_index=True)
    print(train_data)

    with open(r'../static/make_data/combined_train.csv', 'w') as file:
        train_data.to_csv(file, index=False)


if __name__ == "__main__":
    all_data_category_distribute()
    # divide_data()
    # data_distribute()
    # unbalance_category()
