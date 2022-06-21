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
from pathlib import Path
import json
import gc
import os

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


categories = None   # 保存 Class
n = None    # 保存类别数


# 自定义支持sklearn的评估器=================================================================================
class MyXGBoostEstimator:
    """xgboost原生API不支持sklearn的网格搜索，要么自己写循环调用原生的CV验证，要么写一个自定义类用sklearn的网格搜索，
    此处采用后者"""
    def __init__(self, **params):
        self.params = params
        if 'num_boost_round' in self.params:
            self.num_boost_round = self.params['num_boost_round']
        self.params.update({'objective': 'multi:softmax', 'eval_metric': 'merror', 'seed': 0})

    def fit(self, x_train, y_train):
        # 准备xgb训练集和测试集
        train_x, test_x, train_y, test_y = train_test_split(x_train, y_train, test_size=0.1)
        dtrain = xgb.DMatrix(train_x, train_y)
        dtest = xgb.DMatrix(test_x,test_y)
        watchlist = [(dtrain, 'train'), (dtest, 'test')]
        del self.params['num_boost_round']
        self.bst = xgb.train(params=self.params, dtrain=dtrain, evals=watchlist, num_boost_round=self.num_boost_round,
                             early_stopping_rounds=5, verbose_eval=False)

    def predict(self, x_pred):
        dpred = xgb.DMatrix(x_pred)
        return self.bst.predict(dpred)

    def kfold(self, x_train, y_train, nfold=5):
        dtrain = xgb.DMatrix(x_train, y_train)

        cv_rounds = xgb.cv(params=self.params, dtrain=dtrain, num_boost_round=self.num_boost_round,
                           nfold=nfold, early_stopping_rounds=10)
        print(cv_rounds.iloc[-1, :])
        print("次数：",cv_rounds.shape[0])
        return cv_rounds.iloc[-1, :]

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        self.params.update(params)
        return self


# 获取数据===============================================================================================
def read_data(path=''):
    # 读取数据
    if path == '':
        path = Path(__file__).parent / "../static/data/MalDroid-2020.csv"
    data = pd.read_csv(path)

    # 将类别的数据类型替换成数字
    temp = []
    for i in data["Class"]:
        if i not in temp:
            temp.append(i)
    t = {}
    global n
    n = 0
    for i in temp:
        t[i] = n
        n += 1
    data["Class"] = data["Class"].replace(t)

    # 记录原始类别和转换后的类别关系
    global categories
    # categories = ['Advertising_software','Bank_malware','SMS_malware','Risk_software','Benign_application'] # 默认
    categories = temp
    return data


# 获取数据的基本信息
def get_dataInformation():
    dataInformation = {}
    global n
    global categories
    mapCategories = dict(enumerate(categories))
    dataInformation['numCategories'] = n    # 类别数
    dataInformation['mapCategories'] = mapCategories   # 类别映射表
    return dataInformation


# 计算每个类别的占比===============================================================================
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


# 过采样============================================================================================
def up_sample(data):
    count = count_categories(data)
    m = max(count.values())
    for i in count.keys():
        if count[i] != m:
            if count[i] == 0:
                t = 1
            else:
                t = m // count[i]
            df1 = data[data['Family'] == i]
            #print(t)
            for j in range(t):
                data = pd.concat([data, df1], ignore_index=True)
            gc.collect()
    return data


# 综合采样（降采样和过采样相结合）（待完成）===================================================================
def mixture_sample(data):
    count = count_categories(data)
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
def prepare_data(data, test_size=0.3):
    # 添加一列随机数，为了校验因子准确性（重要性排序在随机数后面的可以忽略）
    # data['随机数'] = np.random.randint(0, 4, size=len(data))

    # XGBoost建模数据准备
    data_result = data['Class']  # 最后一列为输出结果
    data_input = data[[i for i in data.columns if i != 'Class']]    # 除最后一列外为特征值

    # 准备xgb训练集和测试集
    train_x, test_x, train_y, test_y = train_test_split(data_input, data_result, test_size=test_size)

    # 查看训练集和测试集的特征值形状
    print(train_x.shape, test_x.shape)

    return train_x, test_x, train_y, test_y


# xgb：多分类==========================================================================================
def xgb_model(train_x, train_y, test_x=None, test_y=None, **params):
    """
    Return:
        model: 由训练集训练出来的模型

        evals_result: 在不断迭代过程中，训练集和测试集的多分类logloss损失函数的值
            {'train': OrderedDict([ ('mlogloss',[]) ])
             'test': OrderedDict([ ('mlogloss',[]) ])
    """
    # 训练数据及评价数据构造
    dtrain = xgb.DMatrix(train_x, label=train_y)
    if test_x is None or test_y is None:
        dtest = None
        watchlist = ()
    else:
        dtest = xgb.DMatrix(test_x, label=test_y)
        watchlist = [(dtrain, 'train'), (dtest, 'test')]

    # 参数构建
    num_boost_round = 50
    if 'num_boost_round' in params:
        num_boost_round = params.pop('num_boost_round')
    fix_params = {'booster': 'gbtree',
                  'objective': 'multi:softmax',  # 多分类'multi:softmax'返回预测的类别(不是概率)，'multi:softprob'返回概率
                  'num_class': n,
                  'eval_metric': 'mlogloss',  # 模型评价指标。二分类用’auc‘，多分类用'mlogloss'或'merror'
                  'seed': 9}
    params.update(fix_params)

    evals_result = {}
    model = xgb.train(params, dtrain, evals=watchlist, num_boost_round=num_boost_round, evals_result=evals_result)

    return model, evals_result


# XGBoost模型评估================================================================
def evaluationModel(model, test_x, test_y):
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
    # model.dump_model("22.txt")


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


# 定义默认模型损失函数评估==============================================================================
def score_fn(y_pred, y_true):
    return np.sqrt(mean_squared_error(y_pred, y_true))


# 执行网格搜索=============================================================================================
def gridSearch(train_x,train_y,params,param_grid,cv_score_fn=None):

    model = MyXGBoostEstimator(**params)
    grid = GridSearchCV(model, param_grid=param_grid, cv=3, scoring=cv_score_fn)
    grid.fit(train_x, train_y)
    return grid.best_params_


# 选择最佳参数=============================================================================================
def selectBestParams(train_x,train_y,scoring=None,greater_is_better=True):
    """
    由于网格搜索耗费的时间非常长，因此将网格搜索分成三步执行：
    1、先调max_depth，min_child_weight，subsample，colsample_bytree
    2、再调gamma，lambda
    3、再调eta
    """
    if scoring == None:
        cv_score_fn = make_scorer(score_fn, greater_is_better=False)
    else:
        cv_score_fn = make_scorer(scoring, greater_is_better=greater_is_better)
    params = {'num_boost_round':100}
    #第一步
    print("搜索max_depth、min_child_weight、subsample、colsample_bytree中...",end='')
    param_grid = {'max_depth': list([5, 6, 7, 8]),
                  'num_class': [n],
                  'min_child_weight': list([0, 1, 3, 5]),
                  'subsample': list([0.7, 0.8, 0.9, 1]),
                  'colsample_bytree': list([0.7, 0.8, 0.9, 1])}
    param_temp = gridSearch(train_x,train_y,params,param_grid,cv_score_fn)
    params.update(param_temp)   # 将已经训练好的参数设置为默认参数
    print("Success!")

    #第二步
    print("搜索gamma、lambda中...",end='')
    # 这两个参数，调的话要看前面训练loss的下降是情况，可以先拿前面的参数去训练一下，如果过拟合严重，再来调这个
    param_grid = {'gamma': list([0, 0.01, 0.1, 0.5, 1, 2]),
                  'lambda': list([0, 0.2, 0.5, 1, 1.5, 2]),
                  'alpha': list([0, 0.2, 0.5, 1, 1.5, 2])}
    param_temp = gridSearch(train_x, train_y, params,param_grid,cv_score_fn)
    params.update(param_temp)  # 将已经训练好的参数设置为默认参数
    print("Success!")

    #第三步
    print("搜索eta中...",end='')
    param_grid = {'eta': list([0.05, 0.1, 0.3, 0.5, 0.7])}
    param_temp = gridSearch(train_x, train_y, params,param_grid,cv_score_fn)
    params.update(param_temp)  # 将已经训练好的参数设置为默认参数
    print("Success!")
    return params


# 遍历文件，得到每种恶意软件类型的正确率===============================================================
def traverseDocument():
    root_path = Path(__file__).parent / "..\static\AndMal2020-Dynamic-BeforeAndAfterReboot"
    filenames = os.listdir(root_path)
    data = None
    mapCategories = {}  # 类别映射表
    global n
    n = 0
    # 遍历文件夹下的28个数据文件
    for i in filenames:
        mapCategories[n] = i.split('.')[0]

        path = os.path.join(root_path,i)
        print(path)

        # 读取原始数据并处理
        single_data = pd.read_csv(path)
        # 数据清洗
        single_data = single_data.drop("Hash", axis=1)  # 'Hash'，字段不可用
        single_data = single_data.drop("Category", axis=1)  # 'Category"'，字段不可用

        # 将类别的数据类型由字符串替换成数字
        single_data["Family"] = n
        n+=1
        # 用于控制类别数，1 ≤ n ≤ 28
        if n == 6:
            break

        if isinstance(data,pd.DataFrame):
            data = pd.concat([data, single_data], ignore_index=True)
        else:
            data = single_data
    train_x, test_x, train_y, test_y = prepare_data(data)  # 为模型准备数据

    all = data.shape[0]
    count = count_categories(data)
    print("原始数据类别占比情况：")
    for i in count:
        print('{}:{:.2%}'.format(i, count[i] / all))

    params = {'num_boost_round': 100, 'colsample_bytree': 0.8, 'max_depth': 6, 'min_child_weight': 1,
              'num_class': 11, 'subsample': 0.8, 'gamma': 0, 'lambda': 1, 'eta': 0.3}
    model = xgb_model(train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y, **params)  # 训练模型
    evaluationModel(model, test_x, test_y)  # 评估模型

    # model.dump_model("../static/Model/model3.json", dump_format='json')
    fout = open(os.fspath("../static/Model/testmodel.json"), 'w')
    ret = model.get_dump(dump_format='json')
    data = []
    for i,value in enumerate(ret):
        tree = json.loads(value)
        tree['category'] = mapCategories[i%n]
        data.append(tree)
    json.dump(data,fout,indent=2)
    fout.close()

    c = 0
    with open("../static/Model/testmodel.json",'r') as f:
        data = json.load(f)
        for i in data:
            if i['category'] == 'Backdoor_after_reboot_Cat':
                c+=1
        print(c)


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y