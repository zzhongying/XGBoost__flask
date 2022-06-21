import xgboost as xgb
import dataInterface.finalXGBoost as myxgb
import pandas as pd
import gc
from dataInterface import cleanData
from dataInterface import myInterface
from pathlib import Path
import shap
import numpy as np


# 输出混淆矩阵
def get_confusionMatirx(model, test_x, test_y, n=5):
    """
    Returns : list, [类别 * 类别] ， 列表的下标代表类别
        confusionMatirx : 包含所有类别的混淆矩阵，横向为实际类别，纵向为预测类别；实*预
    """
    dtest = xgb.DMatrix(test_x)
    pred_y = model.predict(dtest)

    confusionMatirx = []
    for i in range(n):
        confusionMatirx.append([i * 0 for i in range(n)])
    for i in range(len(test_y)):
        confusionMatirx[int(test_y[i])][int(pred_y[i])] += 1
    print(confusionMatirx)


# 过采样
def up_sample(data):
    count = data['Class'].value_counts()
    m = max(count)
    for value in count.iteritems():
        if value[1] != m:
            if value[1] == 0:
                t = 1
            else:
                t = m // value[1]
            df1 = data[data['Class'] == value[0]]
            for j in range(t-1):
                data = pd.concat([data, df1], ignore_index=True)
            gc.collect()
    return data


# 综合采样
def mix_sample(data, rank=0):
    count = data['Class'].value_counts().sort_values()
    num = count.iloc[rank]
    new_data = None
    if num == 0:
        print("选择的数据量为0，请重新选择！")
        return
    for value in count.iteritems():
        frac = num / value[1]
        if frac > 1:
            df1 = data[data['Class'] == value[0]].sample(frac=frac, replace=True)
        else:
            df1 = data[data['Class'] == value[0]].sample(frac=frac)
        if new_data is None:
            new_data = df1
        else:
            new_data = pd.concat([new_data, df1], ignore_index=True)
    return new_data


# 降采样
def down_sample(data, rank=0):
    count = data['Class'].value_counts().sort_values()
    num = count.iloc[rank]
    new_data = None
    if num == 0:
        print("选择的数据量为0，请重新选择！")
        return
    for value in count.iteritems():
        if value[1] < num:
            df1 = data[data['Class'] == value[0]]
        else:
            df1 = data[data['Class'] == value[0]].sample(num)
        if new_data is None:
            new_data = df1
        else:
            new_data = pd.concat([new_data, df1], ignore_index=True)
    return new_data


# 平衡数据集类别数量
def balance_amount(data, select='down', rank=None):
    print("采样前数据类别数量分布：")
    print(data['Class'].value_counts().sort_values())
    if select == 'down':
        data = down_sample(data, rank)
    elif select == 'up':
        data = up_sample(data)
    elif select == 'mix':
        data = mix_sample(data, rank)
    print("采样后数据类别数量分布：")
    print(data['Class'].value_counts().sort_values())
    return data


# 得到特征贡献度及特征名
def get_shap_value(model, data):
    # shapely计算
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    return shap_values


# 得到特征贡献度的绝对值
def get_feature_importance(model, data, rank=0.0):
    shap_values = get_shap_value(model, data)

    featureImportance = []   # 类别 * 特征
    for i in range(len(shap_values)):
        values = shap_values[i]  # 针对每个类的shap_value
        featureValues = []
        for j in range(len(values[0])):
            featureValues.append(float(np.mean([abs(m[j]) for m in values])))
        featureImportance.append(featureValues)
    featureNames = data.columns
    print('类别数：', len(featureImportance))

    main_data = []  # [{'特征名': value, ...}, ...]
    for fetureValues in featureImportance:
        selectedFeature = {}
        for i in range(len(fetureValues)):
            value = max(fetureValues)
            if value <= rank:
                break
            index = fetureValues.index(value)
            selectedFeature[featureNames[index]] = value
            fetureValues[index] = -1
        main_data.append(selectedFeature)
    print(main_data[0])
    n = 0
    for i in main_data[0]:
        if n == 60:
            break
        print(main_data[0][i])
        n += 1
    return main_data


# 同一数据集不同类别间的特征(相同特征)相似性
def get_same_feature(shape_values):
    """
    shape_values: 类别 * 特征
    return:
    """
    pass


# 不同数据集相同类别间的特征(不同特征)差异性
def get_dif_feature():
    pass


# 变换same_feature
def transform_same_feature(model, train_x):
    fea_importance = get_feature_importance(model, train_x)
    print(fea_importance)
    return fea_importance


# 变换different_feature
def transform_dif_feature(model, train_x, test_x):
    train_fea = get_feature_importance(model, train_x)
    test_fea = get_feature_importance(model, test_x)
    fea_importance = []
    for i in range(len(test_fea)):
        temp = []
        temp.append([fea for fea in test_fea[i] if fea not in train_fea[i]])    # 测试集多出的特征
        temp.append([fea for fea in train_fea[i] if fea not in test_fea[i]])    # 测试集没有的特征
        fea_importance.append(temp)
    print(fea_importance)
    return fea_importance


# 对特征进行平衡
def balance_feature(data, select='same'):
    pass


# 转换特征
def transform_feature(model, train_x, test_x, select='dif'):
    """
    1、特征改变程度
    2、类别组合
    3、评估方式
    """
    target_fea = []
    if select == 'same':
        transform_same_feature(model, train_x)
        target_fea = ['NumberofDotsinURL', 'domain_token_count']
        pass
    elif select == 'dif':
        select_fea = transform_dif_feature(model, train_x, test_x)
        for i in range(5):
            target_fea += select_fea[i][0]
        target_fea = list(set(target_fea))
    train_x = train_x.drop(target_fea, axis=1)
    test_x = test_x.drop(target_fea, axis=1)
    return train_x, test_x


# 提取特征
def extract_feature(model, train_x, test_x, select='test', rank=0.0):
    select_fea = []
    if select == 'test' or select == 'all':
        temp = get_feature_importance(model, test_x, rank)
        for t in temp:
            select_fea += t.keys()
    elif select == 'train' or select == 'all':
        k = get_feature_importance(model, train_x, rank)
        for t in k:
            select_fea += t.keys()
    select_fea = list(set(select_fea))
    train_x = train_x[select_fea]
    test_x = test_x[select_fea]
    return train_x, test_x


# 得到预测标签
def get_pred_label(model, test_x):
    data = xgb.DMatrix(test_x)
    y_pred = model.predict(data)
    return list(y_pred)


# 同一数据集
def func1(model, train_x, train_y):
    # 选择异常类特征贡献度计算
    un_family = [4]    # 中心类
    ta_family = [0, 2]  # 边缘类
    family = un_family[0]
    shap_values = get_shap_value(model, train_x)
    shap_value = shap_values[family]
    fea_name = list(train_x.columns)
    print(len(fea_name))
    pred_y = get_pred_label(model, train_x)

    fea_positive = [{} for j in range(5)]
    for i in range(5):
        sample = [shap_value[m] for m in range(len(train_y)) if train_y[m] == i and train_y[m] != pred_y[m]]
        for k in range(len(sample[0])):
            temp = np.mean([value[k] for value in sample if value[k] > 0.0])
            if temp > 0.01:
                fea_positive[i][fea_name[k]] = temp

    fea_negative = {ta_family[j]: {} for j in range(len(ta_family))}
    for i in ta_family:
        shap_value = shap_values[i]
        sample = [shap_value[m] for m in range(len(train_y)) if train_y[m] == i and train_y[m] != pred_y[m]]
        for k in range(len(sample[0])):
            temp = np.mean([value[k] for value in sample if value[k] < 0.0])
            if temp > -0.01:
                fea_negative[i][fea_name[k]] = temp

    # print(fea_negative)
    # print(fea_positive)
    result = []
    for fa in ta_family:
        feature = fea_negative[fa]
        for name in feature:
            if name in fea_positive[fa]:
                result.append(name)
    print(len(set(result)))
    print(result)
    return result, fea_positive


# 方法一的实现
def func2(model, train_x, train_y, test_x, test_y):
    target_fea, _ = func1(model, train_x, train_y)
    print("被删掉的特征：", target_fea)
    train_x = train_x.drop(target_fea, axis=1)
    test_x = test_x.drop(target_fea, axis=1)
    # 训练模型
    model, _ = myxgb.xgb_model(train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y, **params)

    # 进行预测评估
    myxgb.evaluationModel(model, test_x, test_y)

    get_confusionMatirx(model, test_x, test_y)


# 方法二
def func3(model, train_x, train_y, test_x, test_y):
    _, fea_positive = func1(model, train_x, train_y)
    fea_positive = fea_positive[4]
    select_fea = [fea for fea in fea_positive if fea_positive[fea] > 0.0]
    pred_y = get_pred_label(model, test_x)

    family = 4
    shap_values = get_shap_value(model, test_x)
    shap_value = shap_values[family]
    fea_name = list(train_x.columns)

    fea_err = {}
    for k in range(len(shap_value[0])):
        temp = np.mean([value[k] for value in shap_value if value[k] > 0.0])
        if temp > 0.01:
            fea_err[fea_name[k]] = temp

    result = [name for name in fea_err if name not in select_fea]
    print("被删掉的特征：", result)

    train_x = train_x.drop(result, axis=1)
    test_x = test_x.drop(result, axis=1)
    # 训练模型
    model, _ = myxgb.xgb_model(train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y, **params)

    # 进行预测评估
    myxgb.evaluationModel(model, test_x, test_y)

    get_confusionMatirx(model, test_x, test_y)


# test
def test():
    path = r'../static/data/AndMal2020-Dynamic.csv'
    basic = myInterface.basicData()
    print(basic.dataInformation)
    basic.replace_data('data3')
    print(basic.dataInformation)
    main_data = get_feature_importance(basic.model, basic.test_x)
    for i in main_data:
        print(len(i))
    # print(len(main_data))
    # print(len(main_data[0]))
    # print(len(main_data[1]))


# if __name__ == "__main__":
#     # 读取数据
#     train_path = r'../static/make_data/num_train.csv'
#     test_path = r'../static/make_data/num_test.csv'
#     train = myxgb.read_data(train_path)
#     data_info = myxgb.get_dataInformation()  # 保证训练集和测试集的字符类别和数字类别转换的一致性
#     map_category = {v: k for k, v in data_info['mapCategories'].items()}
#     test = pd.read_csv(test_path)
#     test['Class'].replace(map_category, inplace=True)
#
#     # 数据分布
#     print("训练集：")
#     print(train['Class'].value_counts())
#     print("测试集：")
#     print(test['Class'].value_counts())
#
#     # 处理数据集中的空值以及 inf 值
#     cleanData.process_inf_value(train)
#     cleanData.process_null_value(train)
#     cleanData.process_inf_value(test)
#     cleanData.process_null_value(test)
#
#     # 对数据进行提升
#     # train = balance_amount(train, select='mix', rank=2)
#     # train = balance_amount(train, select='up')
#
#     # 特征和标签分离
#     train_x = train[[i for i in train.columns if i != 'Class']]
#     train_y = train['Class']
#     test_x = test[[i for i in test.columns if i != 'Class']]
#     test_y = test['Class']
#
#     # 构建参数集
#     params = {'num_boost_round': 10, 'colsample_bytree': 0.8, 'max_depth': 6, 'min_child_weight': 1,
#               'subsample': 1, 'gamma': 0, 'lambda': 1, 'alpha': 0.2, 'eta': 0.5}
#
#     # 训练模型
#     model, _ = myxgb.xgb_model(train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y, **params)
#
#     # # test
#     # train = train.head(3)
#     # x = train[[i for i in train.columns if i != 'Class']]
#     # y = train['Class']
#     # print(y)
#     # shap_value = get_shap_value(model, x)
#     # for value in shap_value:
#     #     print(sum(value[0]), sum(value[1]), sum(value[2]))
#     # exit()
#
#     # 进行预测评估
#     myxgb.evaluationModel(model, test_x, test_y)
#
#     # 输出自制版混淆矩阵
#     get_confusionMatirx(model, test_x, test_y)
#
#     print("这是最新方法：")
#     func3(model, train_x, train_y, test_x, test_y)
#     exit()
#
#     # 对数据进行提升
#     train_x, test_x = transform_feature(model, train_x, test_x, select='dif')
#
#     # 训练模型
#     model, _ = myxgb.xgb_model(train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y, **params)
#
#     # 进行预测评估
#     myxgb.evaluationModel(model, test_x, test_y)
#
#     for i in range(5):
#         # 对数据进行提升
#         train_x, test_x = extract_feature(model, train_x, test_x, select='all', rank=0.1)
#
#         # 训练模型
#         model, _ = myxgb.xgb_model(train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y, **params)
#
#         # 进行预测评估
#         myxgb.evaluationModel(model, test_x, test_y)

