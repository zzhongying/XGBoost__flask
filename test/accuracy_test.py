

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import shap
import warnings
import gc
import json
import os

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# 自定义支持sklearn的评估器
class MyXGBoostEstimator:
    def __init__(self, **params):
        self.params = params
        if 'num_boost_round' in self.params:
            self.num_boost_round = self.params['num_boost_round']
        self.params.update({'silent': 1, 'objective': 'multi:softmax', 'eval_metric': 'merror','seed': 0})

    def fit(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, y_train)
        self.bst = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=self.num_boost_round)

    def predict(self, x_pred):
        dpred = xgb.DMatrix(x_pred)
        return self.bst.predict(dpred)

    def kfold(self, x_train, y_train, nfold=5):
        dtrain = xgb.DMatrix(x_train, y_train)

        cv_rounds = xgb.cv(params=self.params, dtrain=dtrain, num_boost_round=self.num_boost_round,
                           nfold=nfold, early_stopping_rounds=10)

        return cv_rounds.iloc[-1, :]

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        self.params.update(params)
        return self

test_data = {}

# 获取数据===============================================================================================
def read_data(path=''):
    # 读取数据
    if path == '':
        # path = "D:\网络安全可视化\AndMal2020-Dynamic-BeforeAndAfterReboot\Ransomware_after_reboot_Cat.csv"
        path = "../static/AndMal2020-Dynamic-BeforeAndAfterReboot\Backdoor_after_reboot_Cat.csv"
    data = pd.read_csv(r'D:\网络安全可视化\temp\MalDroid-2020\feature_vectors_syscallsbinders_frequency_5_Cat.csv')
    print(data.shape)

    global n
    n = 5

    return data

# 计算每个类别的占比===============================================================================
def counterProCategories(data):
    count = {}
    temp = []
    for i in data["Class"]:
        if i not in temp:
            temp.append(i)
            count[i] = 1
        else:
            count[i] += 1
    return count


# 过采样============================================================================================
def up_sample(data):
    count = counterProCategories(data)
    m = max(count.values())
    for i in count.keys():
        if count[i] != m:
            if count[i] == 0:
                t = 1
            else:
                t = m // count[i]
            df1 = data[data['Class'] == i]
            #print(t)
            for j in range(t):
                data = pd.concat([data, df1], ignore_index=True)
            gc.collect()
    return data


# 综合采样（降采样和过采样相结合）===================================================================
def mixture_samle(data):
    count = counterProCategories(data)
    m = max(count.values())
    for i in count.keys():
        if count[i] != m:
            if count[i] == 0:
                t = 1
            else:
                t = m // count[i]
            df1 = data[data['Family'] == i]
            # print(t)
            for j in range(t):
                data = pd.concat([data, df1], ignore_index=True)
            gc.collect()
    return data


# XGBoost建模数据准备===============================================================================
def prepare_data(data):
    # 添加一列随机数，为了校验因子准确性（重要性排序在随机数后面的可以忽略）
    # data['随机数'] = np.random.randint(0, 4, size=len(data))

    # XGBoost建模数据准备
    data_result = data.iloc[:, 470] # 最后一列(142)为输出结果
    data_input = data.iloc[:, 0:470]    # 第1-141为特征值
    print(data_result)

    # 准备xgb训练集和测试集
    train_x, test_x, train_y, test_y = train_test_split(data_input, data_result, test_size=0.2)

    # 查看训练集和测试集的特征值形状
    print(train_x.shape, test_x.shape)

    return train_x, test_x, train_y, test_y


# xgb：多分类==========================================================================================
def xgb_model(train_x,test_x,train_y,test_y,**params):
    dtrain = xgb.DMatrix(train_x, label=train_y)

    # 参数
    if 'num_boost_round' in params:
        params.pop('num_boost_round')
    print(n)
    fix_params = {'booster': 'gbtree',
                  'objective': 'multi:softmax',  # 多分类'multi:softmax'返回预测的类别(不是概率)，'multi:softprob'返回概率
                  'num_class': 6,
                  'eval_metric': 'merror',  # 二分类用’auc‘，多分类用'mlogloss'或'merror'
                  'seed': 9,
                  'nthread': 8,
                  # 'silent': 1,
                  'alpha': 0.2,
                  'scale_pos_weight':1}
    params.update(fix_params)

    watchlist = [(dtrain, 'train')]

    num_boost_round = 10
    # 建模与预测:NUM_BOOST_round迭代次数和数的个数一致
    model = xgb.train(params, dtrain, evals=watchlist,early_stopping_rounds=5,num_boost_round=num_boost_round)

    # 获取数据temp
    global test_data
    t = ['eta','max_depth','min_child_weight','gamma','lambda','alpha','subsample']
    t_data = {}
    for i in t:
        t_data[i] = params[i]
    test_data['parameter'] = t_data
    test_data['iteNum'] = num_boost_round
    return model


# XGBoost模型评估================================================================
def evaluationModel(model,test_x,test_y):
    dtest = xgb.DMatrix(test_x)
    ypred = model.predict(dtest)

    # test_y为data抽样的索引，重置后便于与模型预测结果比较
    test_y = test_y.reset_index(drop=True)
    # 手动计算准确率
    cnt1 = 0
    cnt2 = 0
    for i in range(len(test_y)):
        if ypred[i] == test_y[i]:
            cnt1 += 1
        else:
            cnt2 += 1
    print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))

    # 计算召回率，即每一类的分类正确率
    # print(metrics.recall_score(test_y,ypred,average=None))
    count1 = {}  # 分母
    count2 = {}  # 分子
    temp = []
    for i in range(len(test_y)):
        if test_y[i] not in temp:
            temp.append(test_y[i])
            count1[test_y[i]] = 1
            count2[test_y[i]] = 0
        else:
            count1[test_y[i]] += 1
    for i in range(len(test_y)):
        if ypred[i] == test_y[i]:
            count2[test_y[i]] += 1
    for i in temp:
        if count2[i] != 0:
            print('类别{}：{}'.format(i, 100 * count2[i] / count1[i]))
        else:
            print('类别{}：{}'.format(i, 0))

    # 计算精确率
    print(metrics.precision_score(test_y, ypred, average=None))

    # 计算F1值（调和平均值）
    print(metrics.f1_score(test_y, ypred, average=None))

    # 计算混淆矩阵
    print(metrics.multilabel_confusion_matrix(test_y, ypred))



# 模型结果解释==================================================================
def model_importance(model):
    xgb.plot_importance(model, max_num_features=10)
    plt.show()
    print(model.get_score())
    # model.save_model('model.json')
    # model.dump_model("22.txt")

    # #import graphviz
    # xgb.to_graphviz(model, num_trees=2)
    # xgb.plot_tree(model, num_trees=2)
    # xgb.to_graphviz(model, num_trees=1, leaf_node_params={'shape': 'plaintext'})
    # # 保存所有树图
    # for i in range(50):
    #     src = xgb.to_graphviz(model, num_trees=i)
    #     src.view("C:/Users/SB/Desktop/XGBoost_tree/tree"+"_"+str(i))

    #获取数据temp
    global test_data
    t_data = {}
    t_data = model.get_score()
    print('importance:',t_data)
    test_data['importance'] = t_data


# SHAP解释模型=================================================================
def model_shap(model,train_x):
    # 解决xgb中utf-8不能编码问题
    # 新版save_raw()开头多4个字符'binf'
    model_modify = model.save_raw()[4:]

    def myfun(self=None):
        return model_modify

    model.save_raw = myfun

    # SHAP计算
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(train_x)
    print(shap_values)

    # 对特征重要度进行解释
    feture_name = []    #防止特征名称过长导致图像显示不全
    for i in range(1,142):
        feture_name.append('FE-'+str(i))
    shap.summary_plot(shap_values, train_x, max_display=10,plot_type='bar',feature_names=feture_name)

    # 对于某一个分类的SHAP值解释
    shap.summary_plot(shap_values[2], train_x, max_display=10,feature_names=feture_name,plot_type='bar')
    shap.summary_plot(shap_values[0], train_x, max_display=10,feature_names=feture_name)

    # 训练集第1个样本对于输出结果为1(类别为1)的SHAP解释（无结果输出）
    shap.force_plot(explainer.expected_value[1], shap_values[1][1],feature_names=feture_name)

    # 对第1个样本对于输出结果为1(类别为1)的统计图解释
    cols = train_x.columns.tolist()
    shap.bar_plot(shap_values[1][1], feature_names=feture_name, max_display=10)

    # 单个特征与模型预测结果的关系
    shap.dependence_plot('Logcat_error', shap_values[1], train_x[cols],
                         display_features=train_x[cols], interaction_index=None)

    # 前1000个样本的shap累计解释，可选择坐标内容（无结果输出）
    shap.force_plot(explainer.expected_value[1], shap_values[1][:1000, :], train_x.iloc[:1000, :])




def score_fn(y_pred, y_true):
    return np.sqrt(mean_squared_error(y_pred, y_true))

cv_score_fn = make_scorer(score_fn, greater_is_better=False)


# 执行网格搜索=============================================================================================
def gridSearch(train_x,train_y,params,param_grid):

    model = MyXGBoostEstimator(**params)
    grid = GridSearchCV(model, param_grid=param_grid, cv=3, scoring=cv_score_fn)
    grid.fit(train_x, train_y)
    return grid.best_params_


"""
由于网格搜索耗费的时间非常长，因此将网格搜索分成三步执行：
1、先调max_depth，min_child_weight，subsample，colsample_bytree
2、再调gamma，lambda
3、再调eta
"""
def selectBestParams(train_x,test_x,train_y,test_y):
    params = {'num_boost_round':100}
    #第一步
    param_grid = {'max_depth': list([5, 6, 7]),
                  'num_class': [n],
                  'min_child_weight': list([1, 5, 9]),
                  'subsample': list([0.8, 1]),
                  'colsample_bytree': list([0.8, 1])}
    param_temp = gridSearch(train_x,train_y,params,param_grid)
    params.update(param_temp)   # 将已经训练好的参数设置为默认参数

    #第二步
    # 这两个参数，调的话要看前面训练loss的下降是情况，可以先拿前面的参数去训一下，如果过拟合严重，再来调这个
    param_grid = {'gamma': list([0, 0.1, 0.5, 1]),
                  'lambda': list([0.5, 1, 1.5])}
    param_temp = gridSearch(train_x, train_y, params,param_grid)
    params.update(param_temp)  # 将已经训练好的参数设置为默认参数

    #第三步
    param_grid = {'eta': list([0.3, 0.5, 1])}
    param_temp = gridSearch(train_x, train_y, params,param_grid)
    params.update(param_temp)  # 将已经训练好的参数设置为默认参数
    return params


# 遍历文件，得到每种恶意软件类型的正确率===============================================================
def traverseDocument():
    root_path = 'D:\网络安全可视化\AndMal2020-Dynamic-BeforeAndAfterReboot'
    filenames = os.listdir(root_path)
    for i in filenames:
        path = os.path.join(root_path,i)
        print(path)
        data = read_data(path)  # 读取原始数据并处理
        train_x, test_x, train_y, test_y = prepare_data(data)  # 为模型准备数据
        data = pd.concat([train_x, train_y], axis=1)

        all = data.shape[0]
        count = counterProCategories(data)
        print("原始数据类别占比情况：")
        for i in count:
            print('{}:{:.2%}'.format(i, count[i] / all))

        data = up_sample(data)  # 对原始数据进行过采样

        all = data.shape[0]
        count = counterProCategories(data)  # 计算数据类别占比
        print("过采样后数据类别占比情况：")
        for i in count:
            print('{}:{:.2%}'.format(i, count[i] / all))

        train_y = data.iloc[:, 141]
        train_x = data.iloc[:, 0:141]
        params = {'num_boost_round': 100, 'colsample_bytree': 0.8, 'max_depth': 6, 'min_child_weight': 1,
                  'num_class': 11, 'subsample': 0.8, 'gamma': 0, 'lambda': 1, 'eta': 0.3}
        xgb_model(train_x, test_x, train_y, test_y, **params)  # 训练模型



if __name__ == "__main__":
    # traverseDocument()
    # exit()

    data = read_data()  # 读取原始数据并处理
    train_x, test_x, train_y, test_y = prepare_data(data)  # 为模型准备数据
    data = pd.concat([train_x,train_y],axis=1)
    print(data.shape)

    all = data.shape[0]
    count = counterProCategories(data)
    print("原始数据类别占比情况：")
    for i in count:
        print('{}:{:.2%}'.format(i, count[i] / all))

    data = up_sample(data)  # 对原始数据进行过采样

    all = data.shape[0]
    count = counterProCategories(data)  # 计算数据类别占比
    print("过采样后数据类别占比情况：")
    for i in count:
        print('{}:{:.2%}'.format(i, count[i] / all))

    count = {}
    temp = []
    for i in test_y:
        if i not in temp:
            temp.append(i)
            count[i] = 1
        else:
            count[i] += 1
    all = test_y.shape[0]
    print("测试集类别数量情况：")
    for i in count:
        print('{}:{}'.format(i, count[i]))

    train_y = data.iloc[:,470]
    train_x = data.iloc[:,0:470]
    #params = selectBestParams(train_x, test_x, train_y, test_y)
    # print(params)
    params = {'num_boost_round': 100, 'colsample_bytree': 0.8, 'max_depth': 6, 'min_child_weight': 1,
                'subsample': 1, 'gamma': 0, 'lambda': 1, 'eta': 0.5}
    model = xgb_model(train_x, test_x, train_y, test_y,**params)  # 训练模型
    evaluationModel(model,test_x,test_y)  # 评估模型
    # get_data(model,train_x)
    model_importance(model) #模型结果解释
    #model_shap(model, test_x)  #shap解释模型
    #print(test_data)



