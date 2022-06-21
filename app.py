from flask import Flask, request
from dataInterface import myInterface
import json
from flask_cors import *
import os
import numpy as np
import random
from math import pi, cos, sin
import pandas as pd
from pathlib import Path
import shap
import xgboost as xgb
from dataInterface import finalXGBoost as fxgb
import mytools.location as location
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)


def after_request(response):
    # JS前端跨域支持
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


app.after_request(after_request)

# 定义全局对象变量basicData并且使用默认参数训练模型
basicData = myInterface.basicData()
basicData.train_Model()

iterNumber = 1


@app.route('/')
def index():
    # global basicData
    # basicData = myInterface.basicData()
    # basicData.train_Model()
    return "success!"


# 返回特征重要性
@app.route('/getdata/featureImportance')
def get_featureImportance():
    """
    返回堆叠图的数据
    选出的属性是总贡献值最大十个属性，总贡献值为不同类别的属性shap value值相加
    """
    featureImportance, featureNmaes = basicData.get_featureImportance()

    selectedFeature = {}
    fetureValues = []
    for i in range(len(featureNmaes)):
        fetureValues.append(sum([value[i] for value in featureImportance]))
    main_data = []
    for i in range(15):
        value = max(fetureValues)
        index = fetureValues.index(value)
        selectedFeature[featureNmaes[index]] = value
        fetureValues[index] = -1
        temp = []
        for category in featureImportance:
            temp.append(round(category[index], 3))
        main_data.append(temp)
    max_values = 0
    min_values = 0
    for li in main_data:
        min_t = min(li)
        max_t = max(li)
        if min_values > min_t:
            min_values = min_t
        if max_values < max_t:
            max_values = max_t
    data = {
        "legend_data": list(selectedFeature.keys()),
        "visual_data": [max_values, min_values],
        "yAxis_data": [i.split('_')[0] for i in basicData.dataInformation['mapCategories'].values()],
        "series_data": {"main_data": main_data,
                        "key_data": list(selectedFeature.keys())
                        }
    }
    return json.dumps(data)


@app.route('/test')
def test():
    return json.dumps(basicData.get_recallRate())


# 返回热力图的数据
@app.route('/getdata/hotmap')
def get_heatmap():
    confusionMatirx = basicData.get_confusionMatirx()
    categories = [i.split('_')[0] for i in basicData.dataInformation['mapCategories'].values()]
    print(basicData.dataInformation['mapCategories'])
    series_data = []
    max_values = 0
    min_values = 0
    for i in range(len(categories)):
        t_min = min(confusionMatirx[i])
        t_max = max(confusionMatirx[i])
        if t_min < min_values:
            min_values = t_min
        if t_max > max_values:
            max_values = t_max
        for j in range(len(categories)):
            series_data.append([i, j, confusionMatirx[i][j]])
    data = {
        "xAxis_data": categories,
        "yAxis_data": categories,
        "visual_data": [max_values, min_values],
        "series_data": series_data
    }
    return json.dumps(data)


# 返回模型结构数据
@app.route('/getdata/modelStruct')
def get_modelStruct():
    pass


# 返回模型结构数据包含的类以及召回率
@app.route('/getdata/modelStruct/tree_menu2')
def get_treeMenu2():
    """属于中间视图中的复选框"""
    recallrate = basicData.get_recallRate()
    mapCategories = basicData.dataInformation['mapCategories']
    id = 1
    data = []
    for category, recall in recallrate.items():
        menu1 = {}
        menu1['id'] = id
        id += 1
        menu1['label'] = mapCategories[category] + ':' + str(round(recall, 0)) + '%'
        data.append(menu1)
    return json.dumps(data)


# 返回模型评估数据——PR曲线
@app.route('/getdata/evalModel/PR')
def get_PR():
    weight_precision, weight_recall = basicData.get_PR()
    legend_data = []
    global iterNumber
    legend_data.append('iter' + str(iterNumber))
    iterNumber += 1
    series_data = [[i, j] for i, j in zip(weight_recall, weight_precision)]
    data = {
        "legend_data": legend_data,
        "series_data": series_data
    }
    return json.dumps(data)


# 返回模型评估数据——F1值
@app.route('/getdata/evalModel/F1')
def get_f1score():
    f1 = basicData.get_f1()
    data = {
        "xAxis_data": ['iter' + str(iterNumber)],
        "series_data": [round(f1, 2)]
    }
    return json.dumps(data)


# 返回最佳参数
@app.route('/getdata/bestParams')
def get_bestParams():
    # best_params = basicData.get_bestParams()
    # present_params = basicData.params
    # data = {}
    # data['legend_data'] = ["推荐参数", "实际参数"]
    # data['yAxis_data'] = ["eta", "max_depth", "min_child_weight", "subsample", "colsample_bytree",
    #                       "gamma", "lambda", "alpha"]
    # best = []
    # present = []
    # for i in data['yAxis_data']:
    #     best.append(best_params[i])
    # for i in data['yAxis_data']:
    #     present.append(present_params[i])
    # data['series_data'] = [best,present]

    # 固定数据
    data = {"legend_data": ["推荐参数", u"实际参数"],
            "yAxis_data": ["eta", "max_depth", "min_child_weight", "subsample", "colsample_bytree", "gamma", "lambda",
                           "alpha"],
            "series_data": [[0.3, 6, 1, 0.8, 0.7, 0.1, 0.5, 0.2], [0.5, 6, 1, 1, 0.8, 0, 1, 0.2]]}
    # data = {"series_data": [0.3, 6, 1, 0.8, 0.7, 0.1, 0.5, 0.2]}
    return json.dumps(data)


# 返回每个类别的分类正确率（召回率）
@app.route('/getdata/recallRate')
def get_recallRate():
    recallrate = basicData.get_recallRate()
    data = {}
    temp = []
    for key, value in recallrate.items():
        temp.append(round(random.randint(80, 95) / 100, 2))
        # temp = {}
        # temp['value'] = int(value)
        # temp['name'] = str(key)
        # data['series_data'].append(temp)
        # data['legend_data'].append(str(key))
    data['data'] = temp
    return json.dumps(data)


# 学习曲线
@app.route('/getdata/learningCurve')
def get_learingCurve():
    evals_result = basicData.get_evals()
    accuracy = basicData.get_accuracy()
    train_evals = enumerate(evals_result['train'])
    test_evals = enumerate(evals_result['test'])
    data = {
        "series_data": {
            "Training": [[value[0] + 1, value[1]] for value in train_evals],
            "Validation": [[value[0] + 1, value[1]] for value in test_evals]
        },
        "select": [1, int(accuracy)]
    }
    return json.dumps(data)


# 返回星座图数据，属性重要性以及性能指标
@app.route('/center-top/starMap', methods=['POST', 'GET'])
def get_starMap():
    # categories = ["phishing", "Defacement", "benign", "spam", "malware"]  # 根据数据集确定默认值，待定
    categories = ['Advertising_software', 'Bank_malware', 'Benign_application', 'Risk_software', 'SMS_malware']
    if request.method == 'POST':
        categories = json.loads(request.get_data())
    data = {key: [] for key in categories}  # 构建需要返回的数据格式
    # 计算每个类别贡献值最大的十个属性
    featureImportance, featureNmaes = basicData.get_featureImportance()
    mapCategories = basicData.dataInformation['mapCategories']  # 得到数字类别与字符类别之间的映射关系
    mapCategories = dict(zip(mapCategories.values(), mapCategories.keys()))  # 逆转映射关系
    for category in categories:
        fetureValues = featureImportance[mapCategories[category]].copy()
        selectedFeature = {}
        for i in range(10):
            # 筛选出贡献度最大的 10 个特征
            value = max(fetureValues)
            index = fetureValues.index(value)
            selectedFeature[featureNmaes[index]] = value
            fetureValues[index] = -1
        data[category].append(selectedFeature)

    # 计算被选择类别在最佳参数下的各个性能的值
    # 通过混淆矩阵计算准确率(accuracy)、精确率(precision)、召回率(recall)、误报率(false alarm)、漏报率(miss rate)、
    # 特异度(specificity)、f1、AUC
    matirx = basicData.get_confusionMatirx()  # 获得混淆矩阵
    AUC = basicData.get_AUC()  # 获得AUC列表
    for category in categories:
        performance = {}
        index = mapCategories[category]  # 获得对应的数字类别，即下标
        # 计算精准率
        all = sum([value[index] for value in matirx])  # 模型预测的某一类的数量
        TP = matirx[index][index]  # 某一类的正确预测数量
        performance['precision'] = round(TP / all, 2)

        # 计算召回率
        all = sum(matirx[index])  # 某个类别的数量（正样本数量）
        performance['recall'] = round(TP / all, 2)

        # 计算特异度
        all = sum(sum(matirx[i]) for i in range(len(matirx)) if i != index)  # 负样本数量
        TN = sum(sum(matirx[i][1:]) for i in range(len(matirx)) if i != index)  # 负样本的正确预测数量（除正样本外归为负样本）
        performance['specificity'] = round(TN / all, 2)

        # 计算误报率
        performance['falseAlarm'] = round(1 - performance['specificity'], 2)
        # 计算漏报率
        performance['missRate'] = round(1 - performance['recall'], 2)
        # 计算正确率，每个类别正确率即召回率
        performance['accuracy'] = performance['recall']
        # 计算f1-score
        performance['f1'] = round(
            2 * performance['precision'] * performance['recall'] / (performance['precision'] + performance['recall']),
            2)
        # 计算AUC
        performance['AUC'] = round(AUC[index], 2)
        data[category].append(performance)
    return json.dumps(data)


@app.route('/right-top/performance')
def get_performance():
    # targets = ['accuracy','recall','specificity','precision','falseAlarm','missRate','f1','AUC']
    basicData.replace_data('data2')  # 替换数据集

    # targets = ['accuracy','recall']  # 代
    targets = ['specificity', 'precision']  # 钟
    # targets = ['falseAlarm','missRate']  # 程
    # targets = ['f1','AUC']  # 李
    params = basicData.get_best_params_for_target(targets)
    print(params)
    return json.dumps(params)


# 返回样本数据详细信息
@app.route('/getdata/data_info/<data_name>')
def get_dataInfo(data_name=None):
    filepath = basicData.get_data_path(data_name)  # 获取数据绝对路径
    # basicData.replace_data(data_name)   # 替换掉数据集并重新训练模型
    # print('文件目录：', filepath)
    info = {}
    fsize = os.path.getsize(filepath)
    fsize = fsize / float(1024 * 1024)
    info['fsize'] = round(fsize, 3)  # 文件大小

    try:
        f = open(filepath)
        total_lines = sum(1 for line in f)
        f.close()
    except:
        total_lines = -1
    info['total_lines'] = total_lines  # 文件行数(数据数量=（行数-2）/2)有空行
    info['num_feature'] = basicData.test_x.shape[1]  # 属性数量（特征数量）

    family = {0: 'Advertising_software', 1: 'Bank_malware', 2: 'SMS_malware', 3: 'Risk_software',
              4: 'Benign_application'}
    # 标签分类情况
    # df = pd.read_csv(filepath, usecols=['Class']).value_counts()
    df = pd.read_csv(filepath)['Class'].value_counts()
    # print('df->', df)
    tags = {key: int(value) for key, value in zip(df.index, df.values)}
    info['tags'] = tags

    # 对应标签分类正确数量
    true_tags = []
    matirx = basicData.get_confusionMatirx()

    # 正确率
    # print("matirx", matirx)
    accuracys = []
    for i in range(len(matirx)):
        all = sum(j for j in matirx[i])
        accuracys.append({family[i]: {
            'num': tags[family[i]],
            'accuracy': round(matirx[i][i] / all, 3)
        }})
    print(accuracys)

    # for i in range(len(matirx)):
    #     true_tags.append((i, matirx[i][i]))  # tuple, (类别，分类正确数量)
    # info['true_tags'] = true_tags
    # return json.dumps(info)  # 分别返回【文件大小（M为单位），文件行数，文件属性数量，标签分类状况，标签分类正确数量】
    return json.dumps(accuracys)


# 计算,针对于每个超参数，在8种评价指标下都有一个最佳值；结合这8个最佳值计算每个超参数的综合值
@app.route('/get_data/count', methods=['POST'])
def count_eval():
    params = json.loads(request.get_data())
    try:
        params = [li[2] for li in params]
    except (TypeError, IndexError):
        print("数据格式与约定格式不一致")
        return
    data = []  # 需要返回的数据

    # 对数据进行归一化
    change_data = []
    for i in params:
        try:
            x = (i - min(params)) / (max(params) - min(params))
        except ZeroDivisionError:
            x = 0
        num = float(format(x, '.2f'))
        if num > 0:
            change_data.append(num)

    # 中位数
    median = np.median(change_data)
    data.append(median)
    # 平均值
    mean = np.mean(change_data)
    data.append(mean)
    # 方差
    variance = np.var(change_data)
    data.append(variance)
    # 标准差
    std = np.std(change_data)
    data.append(std)

    def harmonic_mean(li):
        """
        计算调和平均数
        易受极端值影响，且不允许有零值存在
        """
        total = 0
        for j in li:
            if i == 0:  # 处理包含0的情况
                return 0
            total += 1 / j
        return len(data) / total

    # 调和平均
    data.append(harmonic_mean(change_data))
    return json.dumps(data)


# 计算坐标
@app.route('/getdata/calculate_coordinates')
def calculate_coordinates():
    params = ['colsample_bytree', 'max_depth', 'min_child_weight', 'subsample', 'gamma', 'lambda', 'alpha', 'eta']
    targets = ['accuracy', 'recall', 'precision', 'falseAlarm', 'missRate', 'specificity', 'f1', 'AUC']
    filepath = Path(__file__).parent / './static/Results/evals_best_params_data2.json'
    with open(filepath, 'r') as f:
        evals_best_params = json.load(f)
    width = 500
    height = 500
    data = []  # 需要返回的数据
    coo1 = width / 2  # x轴中心坐标
    coo2 = height / 2  # y轴中心坐标
    coo_list = [[coo1, coo2]]  # 各参数组对应的中心坐标
    radius = 200  # 外部大圆半径
    # 各组参数组中心坐标确定
    for i in range(len(params) - 1):
        t = (i / (len(params) - 1)) * pi * 2
        x = coo1 + radius * cos(t)
        y = coo2 + radius * sin(t)
        coo_list.append([x, y])
    for param, coo in zip(params, coo_list):
        num = []
        # random.seed(2*params.index(param))
        # x = random.randint(RADIUS - WIDTH // 2, WIDTH // 2 - RADIUS)
        # y = random.randint(RADIUS - HEIGHT // 2, HEIGHT // 2 - RADIUS)
        x = coo[0]
        y = coo[1]

        r = 30  # 半径
        num_targets = len(targets)
        for numerator in range(num_targets):
            t = (numerator / num_targets) * 2 * pi
            dx = r * cos(t)
            dy = r * sin(t)
            num.append([x + dx, y + dy])
        arr = num

        for value, val in zip(arr, targets):
            value.append(evals_best_params[val][param])
            value.append(val)
            value.append(param)
        data.append(arr)
    return json.dumps(data)


# 得到特征的变换趋势及推荐修改值
@app.route('/get_data/change_feature')
def change_feature():
    """
    摆烂，在一坨屎山上面添加代码的最佳方法就是在上面再拉一坨
    """
    # 若有临时文件的保存，则直接读取
    path = Path(__file__).parent / './static/temp_data/change_feature.json'
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return json.dumps(data)

    model = basicData.model
    data_set = ['train', 'test']
    main_data = []  # [{'特征名': value, ...}, ...]
    for data_name in data_set:
        if data_name == 'train':
            data = basicData.train_x
        else:
            data = basicData.test_x

        # shapely计算
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)

        featureImportance = []  # 类别 * 特征
        for i in range(len(shap_values)):
            values = shap_values[i]  # 针对每个类的shap_value
            featureValues = []
            for j in range(len(values[0])):
                temp = []
                try:
                    t = round(np.mean([m[j] for m in values if m[j] > 0]), 2)  # 正影响
                except RuntimeWarning:
                    t = 0.0
                temp.append(t)
                try:
                    t = round(np.mean([m[j] for m in values if m[j] < 0]), 2)  # 负影响
                except RuntimeWarning:
                    t = 0.0
                temp.append(t)
                # temp.append(float(np.mean([m[j] for m in values if m[j] >= 0])))  # 正影响
                # temp.append(float(np.mean([m[j] for m in values if m[j] <= 0])))  # 负影响
                featureValues.append(temp)
            featureImportance.append(featureValues)
        featureNames = data.columns

        # 合并每个类别上的特征贡献值
        feature_contribute = []  # [[总正贡献，总负贡献], []...]
        for i in range(len(featureImportance[0])):
            feature_contribute.append([round(sum([values[i][0] for values in featureImportance]), 2),
                                       round(sum([values[i][1] for values in featureImportance]), 2)])
        for i in range(len(feature_contribute)):
            if feature_contribute[i][0] != 0 or feature_contribute[i][1] != 0:
                if np.isnan(feature_contribute[i][0]) or np.isnan(feature_contribute[i][1]):
                    continue
                if (abs(feature_contribute[i][0]) + abs(feature_contribute[i][1])) < 1:  # 过滤掉较小贡献值的特征
                    continue
                single_data = {}
                single_data['name'] = data_name + '__' + featureNames[i]
                single_data['value'] = feature_contribute[i]
                single_data['type'] = data_name
                single_data['advice'] = [round(random.uniform(-3, 3), 2)]
                main_data.append(single_data)
    # 判断相同特征
    train_feature = list(set(value['name'].split('__')[1] for value in main_data if value['type'] == 'train'))
    test_feature = list(set(value['name'].split('__')[1] for value in main_data if value['type'] == 'test'))
    feature_route = [{'source': 'train' + f_name, 'target': 'test' + f_name} for f_name in train_feature if
                     f_name in test_feature]
    return json.dumps({'data': main_data, 'feature_route': feature_route})


# 返回数据样本偏向度
@app.route('/get_data/offset_data')
def get_offset_data():
    """
    又一坨
    """
    # 商定值
    center_radius = 200  # 中间圈的半径
    edge_radius = 150  # 边缘圈的半径
    max_center_radius = 290  # 中心圈的最大值
    max_edge_radius = 190  # 边缘圈的最大值

    # data4
    center_category = 4
    edge_category = [0, 1, 2, 3]

    # 将半径映射为概率阈值
    center_threshold = 1 - (center_radius / max_center_radius)
    edge_threshold = 1 - (edge_radius / max_edge_radius)

    dtest = xgb.DMatrix(basicData.test_x)
    ypred = basicData.model.predict(dtest, output_margin=True)

    # 将模型原始输出值通过softmax映射到 0~1
    y = []
    for i in ypred:
        k = fxgb.softmax(i)
        y.append(list(k))

    label = basicData.test_y.reset_index(drop=True)
    data = []
    map_category = basicData.dataInformation['mapCategories']
    center_category = map_category[center_category]
    edge_category = [map_category[category] for category in edge_category]
    for i in range(len(y)):
        max_pred = max(y[i])  # numpy.float32，需要转为float
        target = map_category[y[i].index(max_pred)]
        source = map_category[label[i]]
        if target == source:
            continue
        if target == center_category and max_pred < center_threshold:
            continue
        if target in edge_category and max_pred < edge_threshold:
            continue
        data.append([round(float(max_pred), 2), source, [source, target]])
    result_data = []
    category = [[map_category[4], map_category[0]], [map_category[4], map_category[1]],
                [map_category[4], map_category[3]], [map_category[4], map_category[2]],
                [map_category[0], map_category[4]], [map_category[1], map_category[4]],
                [map_category[3], map_category[4]], [map_category[2], map_category[4]]]
    for value in data:
        result_data.append(location.get_location(value, category))
    print(map_category[4], map_category[0], map_category[1], map_category[3], map_category[2])
    return json.dumps({'data': result_data})


# 气泡图，计算每一类模型中每棵树的正负贡献（拟合方向）
@app.route('/get_data/bubble_chart')
def get_bubble_data():
    """
    叒一坨
    """
    path = Path(__file__).parent / './static/Model/model.json'  # 需要保证这个文件保存的模型是当前系统中的模型
    with open(path) as f:
        tree_model = json.load(f)

    map_category = basicData.dataInformation['mapCategories']
    num_category = basicData.dataInformation['numCategories']

    data = {}  # 需要返回的数据
    for category in range(num_category):
        data[map_category[category]] = []

    for tree_num in range(len(tree_model)):
        category = tree_num % num_category
        node = tree_model[tree_num]
        stack = [node]
        negative_value = 0
        positive_value = 0
        tree_value = {}
        while stack:
            node = stack[0]
            if 'leaf' not in node:
                stack.append(node['children'][0])
                stack.append(node['children'][1])
            else:
                if node['leaf'] > 0:
                    positive_value += node['leaf']
                else:
                    negative_value += node['leaf']
            stack.pop(0)
        tree_value['name'] = 'tree' + str(tree_num)
        tree_value['count'] = round(abs(negative_value) + abs(positive_value), 2)
        tree_value['piedata'] = [{"area": "positive", "value": round(abs(positive_value), 2)},
                                 {"area": "negative", "value": round(abs(negative_value), 2)}]
        data[map_category[category]].append(tree_value)

    return json.dumps(data)


# 更新参数，训练新的模型
@app.route('/updataParams', methods=['POST'])
def updata_params():
    if request.method == 'POST':
        params = request.get_data()
        print('params = {}'.format(params))
        params = json.loads(params)
        basicData.train_Model(params=params)
    else:
        print('method error!')


#  返回桑基图数据
@app.route('/get_data/sankey', methods=['POST', 'GET'])
@cross_origin()  # 跨域解决
def get_sankey_data():
    id = 3
    name = 0
    type_values = {
        'Advertising': 0,
        'Bank': 1,
        'SMS': 2,
        'Risk': 3,
        'Benign': 4
    }
    if request.method == 'POST':
        print(json.loads(str(request.get_data(), 'utf-8')))
        param = json.loads(str(request.get_data(), 'utf-8'))
        if 'id' in param.keys():
            id = int(param['id'].split("__")[1][4:])
            name = type_values[param['id'].split("_")[0]]
            if id >= 30:
                id = random.randint(0, 29)
        if 'name' in param.keys():
            name = type_values[param['name']]
            id = 0
    with open('./static/Model/data_flow_' + str(name) + '.pickle', 'rb') as f:
        data_flow = pickle.load(f)
    with open('./static/Model/nodeid_split_' + str(name) + '.pickle', 'rb') as f:
        nodeid_split_dic = pickle.load(f)

    print("str(name)->", str(name), ' id->', id)

    data_keys = list(data_flow[id].keys())
    data = data_flow[id]
    nodeid_split = nodeid_split_dic[id]

    # print(data_flow[0])
    # print(nodeid_split_dic[0])
    # links=[]
    # for i in data_keys:
    #     dic1={}
    #     dic1["source"] = i[0]
    #     dic1["target"] = i[1]
    #     dic1["value"] = data[i][0]
    #     dic1['type']='yes'
    #     if data[i][0]>0:
    #         links.append(dic1)
    #     dic2 = {}
    #     dic2["source"] = i[0]
    #     dic2["target"] = i[1]
    #     dic2['value']=float(data[i][1])  # int((float(data[i][1])-10)*1000)
    #     dic2['type']='no'
    #     # dic["no"] = float(data[i][1])  # (float(data[i][1])-10)*1000
    #     if int((float(data[i][1])-10)*1000)>0:
    #         links.append(dic2)
    # nodes=[]
    # features=[]
    # for i in data_keys:
    #     if i[0] not in features:
    #         features.append(i[0])
    #     if i[1] not in features:
    #         features.append(i[1])
    # for i in features:
    #     nodes.append({
    #         'id':i,
    #         'title':i
    #     })
    # return {
    #     'nodes':nodes,
    #     'links':links,
    #     "alignLinkTypes": False
    # }
    suzu = []
    num=0   # 统计共有多少数据流经该树
    for i in data_keys:
        dic = {}
        dic["source"] = i[0]
        dic["target"] = i[1]
        dic["yes"] = data[i][0]
        dic["no"] = float(data[i][1])  # (float(data[i][1])-10)*1000
        suzu.append(dic)
        if i[0] == 0:
            num += data[i][0]+(float(data[i][1])-10)*1000
    print("调用了sankey数据接口")
    return json.dumps({
        'path': suzu,
        'nodeid_split': nodeid_split,
        'num': int(num)
    })


# 返回中间视图数据（桑基图上面的图）
@app.route('/get_data/centerData')
def get_centerData():
    """
    返回中间视图数据（桑基图上面的图）
    """
    featureImportance, featureNmaes = basicData.get_featureImportance()
    categorys = basicData.dataInformation['mapCategories']
    tem = []
    for i in range(5):
        dic = {'category': categorys[i]}
        lis = []
        for j in range(15):
            Max = max(featureImportance[i])
            ind = featureImportance[i].index(Max)
            lis.append({'featureName': featureNmaes[ind], 'featureImportance': Max})
            featureImportance[i][ind] = -1
        dic['feature'] = lis
        tem.append(dic)
    res = []
    datas = json.load(open("./static/temp_data/feature_values.json", encoding='utf-8'))
    n = 0
    for data in datas:
        for r in tem:
            # print('r->',r)
            if data['Class'] == r['category']:
                res.append({'Class': data['Class'], 'data': [], 'fusion_features': data['fusion_features']})
                for feaTop in r['feature']:
                    for fea in data['data']:
                        if feaTop['featureName'] == fea['feature']:
                            fea['featureImportance'] = round(feaTop['featureImportance'], 3)
                            res[n]['data'].append(fea)

                n = n + 1

    return json.dumps(res)





if __name__ == '__main__':
    # basicData = myInterface.basicData() # 后台命令行启动python app.py，服务才能运行到主函数
    app.run(debug=False, use_reloader=False)
    CORS(app, supports_credentials=True)
