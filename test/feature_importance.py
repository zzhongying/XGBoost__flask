import xgboost as xgb
import pandas as pd
import os
from dataInterface import myInterface
from dataInterface import cleanData
import dataInterface.finalXGBoost as myxgb
import json
import pickle
from decimal import Decimal
import sys
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


# 输出随机划分的测试集在模型上的特征重要性
def all_family_feature_importance(data_name='data4'):
    basicData = myInterface.basicData()
    basicData.replace_data(data_name)
    featureImportance, featureNames = basicData.get_featureImportance()

    family_index = ['family1', 'family2', 'family3', 'family4', 'family5']
    family_importance = pd.DataFrame(featureImportance, index=family_index, columns=featureNames)
    family_importance.loc['all_family'] = family_importance.apply(lambda x: x.sum())

    all_importance = list(family_importance.loc['all_family', :])
    columns = list(family_importance.columns)
    for i in range(len(all_importance)):
        max_value = max((all_importance))
        print(columns[all_importance.index(max_value)], ": ", end='')
        print(max_value)
        all_importance.remove(max_value)


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
        src.view("D:/网络安全可视化/XGBoost_Tree/tree" + "_" + str(i))


# 使用类别不平衡数据集进行模型训练评估
def num_unbalance_eval(train_path, test_path):
    # 读取训练集
    train_path = '../static/make_data/combined_train.csv'
    train = myxgb.read_data(train_path)
    cleanData.process_inf_value(train)

    # 读取测试集
    data_info = myxgb.get_dataInformation()
    map_category = {v: k for k, v in data_info['mapCategories'].items()}
    test_path = '../static/make_data/num_test.csv'
    test = pd.read_csv(test_path)
    test.dropna(inplace=True)
    test['Class'].replace(map_category, inplace=True)

    print('训练集集类别数量：')
    print(train['Class'].value_counts())
    print('测试集类别数量：')
    print(test['Class'].value_counts())

    train_x = train[[i for i in train.columns if i != 'Class']]
    train_y = train['Class']
    test_x = test[[i for i in test.columns if i != 'Class']]
    test_y = test['Class']

    params = {'num_boost_round': 10, 'colsample_bytree': 0.8, 'max_depth': 6, 'min_child_weight': 1,
              'subsample': 1, 'gamma': 0, 'lambda': 1, 'alpha': 0.2, 'eta': 0.5}
    model, _ = myxgb.xgb_model(train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y, **params)
    dtest = xgb.DMatrix(test_x)
    print(model.predict(dtest, pred_leaf=True))
    myxgb.evaluationModel(model, test_x, test_y)
    # draw_model(model)


# 获取叶子节点对应id，并获得对应的value值
def get_leaf_value(data_name=None):
    if data_name is None:
        print("输入的数据集名称错误")
        return

    basicData = myInterface.basicData()
    basicData.replace_data(data_name)
    model = basicData.model
    dtest = xgb.DMatrix(basicData.test_x)
    leaf_id = model.predict(dtest, pred_leaf=True)
    print(leaf_id)
    print(len(leaf_id))
    print(len(leaf_id[0]))
    model.save_model('../static/Model/model.json')
    # tree_struct = json.load('../static/Model/model.json')
    with open('../static/Model/model.json') as tree_struct:
        for line in tree_struct:
            pass


# 对树进行迭代
def run_model():
    num_category = 5  # 确定数据集，里面有5个类
    n = 0  # 对已用树计数
    # sample = pd.read_csv(r'../static/data/MalDroid-2020.csv', nrows=2000)
    basic_data = myInterface.basicData()
    basic_data.replace_data('data1')

    # basic_data.model.dump_model('../static/Model/model.json', dump_format='json')
    sample = pd.concat([basic_data.test_x, basic_data.test_y], axis=1)
    print(sample.shape)
    count_categories(sample)
    print(basic_data.dataInformation['mapCategories'])
    with open('../static/Model/data_flow.pickle', 'rb') as f:
        data_flow = pickle.load(f)
    path = r'../static/Model/model.json'
    with open(path) as f:
        tree_model = json.load(f)
    for tree_num in range(len(tree_model)):
        category = tree_num % 5  # 使用第一个类的树
        if category != 0:
            continue
        tree = tree_model[tree_num]
        data = data_flow[n]
        n += 1
        traverse_tree(tree, sample, data, 0)
    # with open(r'../static/Model/data_flow.json', 'w') as f:
    #     json.dump(data_flow, f)
    with open('../static/Model/data_flow.pickle', 'wb') as f:
        pickle.dump(data_flow, f)


# 层次迭代，按照数据流向拆分树结构
def bfs():
    global nodeid_ways, split_ways,nodeid_split_dic
    num_category = 5  # 类别数
    n = 0
    data_flow = []
    feature_names = []

    path = r'../static/Model/model.json'
    with open(path) as f:
        tree_model = json.load(f)
    for tree_num in range(int(len(tree_model) / 5)):
        data_flow.append(dict())
        nodeid_split_dic.append(dict())
        feature_names.append(list())
    way_lis = []
    for tree_num in range(len(tree_model)):
        category = tree_num % num_category
        if category != 0:
            continue
        node = tree_model[tree_num]
        dfs(node, '', '',n)
        way_lis.append({'nodeid_ways': nodeid_ways, 'split_ways': split_ways})
        nodeid_ways = []
        split_ways = []
        # stack = [node]
        # nodeid_list = []
        # split_list = []
        # while stack:
        #     node = stack[0]
        #     if 'leaf' not in node:
        #         nodeid_list.append(node['nodeid'])
        #         if node['split'] in split_list:
        #             split_list.clear()
        #         else:
        #             split_list.append(node['split'])
        #         feature_names[n].append([node['nodeid'], node['split']])
        #
        #         # data_flow[n][(node['nodeid'], node['children'][0]['nodeid'])] = [0, 10,node['split']]
        #         # data_flow[n][(node['nodeid'], node['children'][1]['nodeid'])] = [0, 10,node['split']]
        #
        #         stack.insert(0, node['children'][0])
        #         stack.insert(1, node['children'][1])
        #     else:
        #         if len(split_list) == len(nodeid_list):  # 没有闭环
        #             long = len(nodeid_list)
        #             for i in range(long):
        #                 if i + 1 != long:
        #                     data_flow[n][(nodeid_list[i], nodeid_list[i + 1])] = [0, 10, split_list[i]]
        #         feature_names[n].append([node['nodeid'], 'leaf'])  # 叶子节点，没有特征，以 leaf 代替
        #     stack.pop(0)
        n += 1

    for i in range(len(way_lis)):
        # data_flow[n][(node['nodeid'], node['children'][0]['nodeid'])] = [0, 10, node['split']]
        # data_flow[n][(node['nodeid'], node['children'][1]['nodeid'])] = [0, 10,node['split']]
        nodeids = way_lis[i]['nodeid_ways']
        splits = way_lis[i]['split_ways']
        for j in range(len(nodeids)):
            x='x'
            nodeid=nodeids[j].split('-')
            split=splits[j].split('-')
            if len(nodeid) == len(split):     # 没有闭环
                for k in range(len(nodeid)):
                    if k+1 != len(nodeid):
                        data_flow[i][(int(nodeid[k]), int(nodeid[k+1]))] = [0, 10, nodeid_split_dic[i][int(nodeid[k])]]

    print(way_lis[0]['nodeid_ways'])
    print(way_lis[0]['split_ways'])
    print(nodeid_split_dic[0])
    print("data_flow_->", len(data_flow))
    # with open(r'../static/Model/feature_names.pickle', 'wb') as f:
    #     pickle.dump(feature_names, f)
    with open('../static/Model/nodeid_split.pickle', 'wb') as f:
        pickle.dump(nodeid_split_dic, f)
    with open('../static/Model/data_flow.pickle', 'wb') as f:
        pickle.dump(data_flow, f)


nodeid_ways = []
split_ways = []
nodeid_split_dic=[]

def dfs(node, nodeids, splits,n):
    nodeids += str(node['nodeid'])
    if 'split' in node:
        if nodeids not in nodeid_split_dic:
            nodeid_split_dic[n][node["nodeid"]]=node['split']
        if node['split'] in splits:
            splits = ''
        else:
            splits += node['split']
    if 'leaf' in node:
        nodeid_ways.append(nodeids)
        split_ways.append(splits)
    else:
        dfs(node['children'][0], nodeids + '-', splits + '-',n)
        dfs(node['children'][1], nodeids + '-', splits + '-',n)


def traverse_tree(tree, sample, data_flow, label):
    for _, row in sample.iterrows():
        node = tree
        while node:
            # print(node['nodeid'], end='  ')
            if 'leaf' in node:
                # print(node['leaf'])
                break
            if row[node['split']] < node['split_condition']:
                source_node = node['nodeid']
                node = node['children'][0]
                target_node = node['nodeid']
                if row['Class'] == label:
                    if (source_node, target_node) in data_flow:
                        data_flow[(source_node, target_node)][0] += 1
                else:
                    if (source_node, target_node) in data_flow:
                        data_flow[(source_node, target_node)][1] += Decimal(str(0.001))
            else:
                source_node = node['nodeid']
                node = node['children'][1]
                target_node = node['nodeid']
                if row['Class'] == label:
                    if (source_node, target_node) in data_flow:
                        data_flow[(source_node, target_node)][0] += 1
                else:
                    if (source_node, target_node) in data_flow:
                        data_flow[(source_node, target_node)][1] += Decimal(str(0.001))


# 保存树模型
def save_tree_model():
    basic_data = myInterface.basicData()
    basic_data.replace_data('data1')
    # basic_data.model.save_model('../static/Model/model.json')
    basic_data.model.dump_model('../static/Model/model.json', dump_format='json')


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
    # parallel_offset()
    # get_leaf_value()
    # save_tree_model()
    bfs()
    run_model()
    # sample = pd.read_csv(r'../static/data/MalDroid-2020.csv', nrows=10)
    # for row_index, row in sample.iterrows():
    #     print(type(row_index))
    #     print(type(row))
    # divide_data()
    # data_distribute()
    # unbalance_category()
    # num_unbalance_eval()
    # with open('../static/Model/data_flow.pickle', 'rb') as f:
    #     data_flow = pickle.load(f)
    #     print(data_flow)
    # with open('../static/Model/feature_names.pickle', 'rb') as f:
    #     feature_names = pickle.load(f)
    #     print(feature_names)
