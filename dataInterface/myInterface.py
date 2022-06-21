from dataInterface.finalXGBoost import *
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from dataInterface import myscoring
from dataInterface import cleanData
from improve_model import process_data



def check_model(func):
    """
    作为函数装饰器，在函数运行前判断模型是否存在
    """
    def check(self, *args, **kwargs):
        if self.model is None:
            print("XGBoost模型还未训练！")
            return
        else:
            return func(self, *args, **kwargs)

    return check


class basicData(object):
    def __init__(self):
        self.model = None

        # # 初始化类时 训练集 和 测试集 不从同一个数据集中拆分出来
        # train_path = Path(__file__).parent / "../static/make_data/num_train.csv"
        # test_path = Path(__file__).parent / "../static/make_data/num_test.csv"
        # train_data = read_data(train_path)  # 读取训练集
        # self.dataInformation = get_dataInformation()
        # test_data = read_data(test_path)
        # train_data = process_data.balance_amount(train_data, select='mix', rank=2)
        # # 为模型准备数据
        # self.train_x = train_data[[col for col in train_data.columns if col != 'Class']]
        # self.train_y = train_data['Class']
        # self.test_x = test_data[[col for col in test_data.columns if col != 'Class']]
        # self.test_y = test_data['Class']

        # 得到数据
        path = Path(__file__).parent / "..\static\data\MalDroid-2020.csv"
        print("数据路径：",path)
        data = read_data(path)
        self.dataInformation = get_dataInformation()
        cleanData.process_inf_value(data)
        cleanData.process_null_value(data)

        self.train_x, self.test_x, self.train_y, self.test_y = prepare_data(data)  # 为模型准备数据

        self.params = {'num_boost_round': 30, 'colsample_bytree': 0.8, 'max_depth': 6, 'min_child_weight': 1,
                        'subsample': 1, 'gamma': 0, 'lambda': 1, 'alpha': 0.2, 'eta': 0.5}  # 好默认参数

    def train_Model(self, params=None):
        """
        训练xgboost模型，如果params为None，使用默认参数
        """
        if params is None:
            params = self.params
        else:
            self.params = params
        # 训练模型
        self.model, self.evals_result = xgb_model(self.train_x, self.train_y, self.test_x, self.test_y, **params)
        print(self.evals_result)

    @check_model
    def get_bestParams(self):
        """得到xgboost模型的最佳参数"""
        params = selectBestParams(self.train_x, self.train_y)
        return params

    def get_struct(self):
        """得到训练好的xgboost模型的结构数据
        https://zhuanlan.zhihu.com/p/370761158"""
        if self.model is None:
            raise ValueError('xgboost模型还未训练')
        pass

    @check_model
    def get_featureImportance(self):
        """
        得到xgboost模型的特征重要性，数据类型：类别*特征 特征名称
        """

        # SHAP计算
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.test_x)

        featureImportance = []
        for i in range(len(shap_values)):
            values = shap_values[i]
            featureValues = []
            for j in range(len(values[0])):
                featureValues.append(float(np.mean([abs(m[j]) for m in values])))
            featureImportance.append(featureValues)
        featureNames = self.test_x.columns
        return featureImportance, featureNames

    @check_model
    def get_shapValues(self):
        """
        得到训练好的xgboost模型在测试集上的shap值
        """

        # SHAP计算
        explainer = shap.TreeExplainer(self.model)
        shapValues = explainer.shap_values(self.train_x)
        return shapValues

    @check_model
    def get_recallRate(self):
        """
        return: dict，key是类别，value是召回率
        """
        dtest = xgb.DMatrix(self.test_x)
        ypred = self.model.predict(dtest)

        # test_y为data抽样的索引，重置后便于与模型预测结果比较
        test_y = self.test_y.reset_index(drop=True)

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
        recallrate = {}
        for i in temp:
            if count2[i] != 0:
                recallrate[i] = 100 * count2[i] / count1[i]
            else:
                recallrate[i] = 0
        return recallrate

    @check_model
    def get_confusionMatirx(self):
        """
        Returns : list, [类别 * 类别] ， 列表的下标代表类别
            confusionMatirx : 包含所有类别的混淆矩阵，横向为实际类别，纵向为预测类别；实*预
        """
        dtest = xgb.DMatrix(self.test_x)
        ypred = self.model.predict(dtest)

        # test_y为data抽样的索引，重置后便于与模型预测结果比较
        test_y = self.test_y.reset_index(drop=True)

        numCategories = self.dataInformation['numCategories']
        confusionMatirx = []
        for i in range(numCategories):
            confusionMatirx.append([i*0 for i in range(numCategories)])
        for i in range(len(test_y)):
            confusionMatirx[int(test_y[i])][int(ypred[i])] += 1
        return confusionMatirx

    @check_model
    def get_PR(self):
        """
        Returns : 两个list,[1 * 阈值数]
            weight_precision ： 所有类别精确率（查准率）的加权和，权值的计算采用测试集每个类比的占比
            weight_recall : 所有类别召回率（查全率）的加权和
        """
        dtest = xgb.DMatrix(self.test_x)
        ypred = self.model.predict(dtest, output_margin=True)

        # 将模型原始输出值通过softmax映射到 0~1
        y = []
        for i in ypred:
            k = softmax(i)
            y.append(list(k))

        precision = []  # 精准率,[类别*阈值]
        recall = []  # 召回率,[类别*阈值]
        thresholdValue = [round(i/10-0.1,1) for i in range(1,11)]    # 设置不同的阈值

        # test_y为data抽样的索引，重置后便于与模型预测结果比较
        test_y = self.test_y.reset_index(drop=True)
        test_y = list(test_y)
        numCategories = self.dataInformation['numCategories']
        for i in range(numCategories):
            sin_precision = []
            sin_recall = []
            for threshold in thresholdValue:
                pred_y = [i if j[i] > threshold else -1 for j in y]
                TP = 0
                test_count = test_y.count(i)
                pred_count = pred_y.count(i)
                for k in range(len(test_y)):
                    if test_y[k] == pred_y[k]:
                        TP += 1
                if pred_count == 0:
                    sin_precision.append(0.00)
                else:
                    sin_precision.append(TP/pred_count)
                if test_count == 0:
                    sin_recall.append(0.00)
                else:
                    sin_recall.append(TP/test_count)
            precision.append(sin_precision)
            recall.append(sin_recall)

        weight_recall = []
        weight_precision = []
        for i in range(len(thresholdValue)):
            call = 0
            prec = 0
            for category in range(numCategories):
                if test_y.count(category) == 0:
                    weight = 0
                else:
                    weight = test_y.count(category)/len(test_y)
                call += recall[category][i] * weight
                prec += precision[category][i] * weight
            weight_recall.append(round(call,2))
            weight_precision.append(round(prec,2))
        return weight_precision, weight_recall

    @check_model
    def get_evals(self):
        """
        Returns: dict
            得到模型在学习过程中的多分类logloss损失函数的值
        """
        evals_result = self.evals_result
        evals_results = {}
        evals_results['train'] = evals_result['train']['mlogloss']
        evals_results['test'] = evals_result['test']['mlogloss']
        return evals_results

    @check_model
    def get_accuracy(self):
        """
        Returns: accuracy, 测试集总的正确率
        """
        dtest = xgb.DMatrix(self.test_x)
        ypred = self.model.predict(dtest)

        # test_y为data抽样的索引，重置后便于与模型预测结果比较
        test_y = self.test_y.reset_index(drop=True)
        test_y = list(test_y)
        # 手动计算准确率
        cnt1 = 0
        cnt2 = 0
        for i in range(len(test_y)):
            if ypred[i] == test_y[i]:
                cnt1 += 1
            else:
                cnt2 += 1
        return round(100 * cnt1 / (cnt1 + cnt2), 2)

    @check_model
    def get_f1(self):
        """
        Return: f1-score,
        """
        precision, recall = self.get_PR()
        F1 = 2 * (precision[5] * recall[5]) / (precision[5] + recall[5])    # 取阈值为 0.5 计算f1值
        return F1

    @check_model
    def get_AUC(self):
        """
        Return: list, [1 * 各类别auc值]
        此处计算auc，预测值是类别，而不是概率，不知道 roc_auc_score() 在计算时阈值是怎么取的；
        推测这样计算的结果不会很精确；
        """
        dtest = xgb.DMatrix(self.test_x)
        ypred = self.model.predict(dtest)

        # test_y为data抽样的索引，重置后便于与模型预测结果比较
        test_y = self.test_y.reset_index(drop=True)
        test_y = list(test_y)
        labels = [i for i in range(self.dataInformation['numCategories'])]
        test_y = label_binarize(self.test_y, classes=labels)
        ypred = label_binarize(ypred, classes=labels)
        AUC = []
        for i in range(len(labels)):
            AUC.append(roc_auc_score(test_y[:, i], ypred[:, i]))
        return AUC

    def get_best_params_for_target(self, targets=None):
        """
        Return: dict, {评价指标：{最佳参数}}
            得到不同的评价指标对应的最佳参数
        Parameters:
            targets : 模型采用的评价指标列表
        """
        if targets is None:
            print("target is none")
            return
        params = {}
        for target in targets:
            if target == 'accuracy':
                params['accuracy'] = selectBestParams(self.train_x,self.train_y,scoring=myscoring.score_accuracy)
            elif target == 'recall':
                params['recall'] = selectBestParams(self.train_x,self.train_y,scoring=myscoring.score_recall)
            elif target == 'precision':
                params['precision'] = selectBestParams(self.train_x,self.train_y,scoring=myscoring.score_precision)
            elif target == 'falseAlarm':
                params['falseAlarm'] = selectBestParams(self.train_x,self.train_y,scoring=myscoring.score_falseAlarm,greater_is_better=False)
            elif target == 'missRate':
                params['missRate'] = selectBestParams(self.train_x,self.train_y,scoring=myscoring.score_missRate,greater_is_better=False)
            elif target == 'specificity':
                params['specificity'] = selectBestParams(self.train_x,self.train_y,scoring=myscoring.score_specificity)
            elif target == 'f1':
                params['f1'] = selectBestParams(self.train_x,self.train_y,scoring=myscoring.score_f1)
            elif target == 'AUC':
                num_category = self.dataInformation['numCategories']
                params['AUC'] = selectBestParams(self.train_x, self.train_y, scoring=myscoring.score_auc)
            else:
                pass
        return params

    def replace_data(self, data_name=None, data_path=None):
        """
        对数据集进行替换，并重新训练模型

        Parameters：
            data_name : 数据集的编号
        """
        if data_path is not None:
            filepath = data_path
        else:
            filepath = self.get_data_path(data_name)
            print("filepath->",filepath)
        try:
            data = read_data(filepath)  # 读取原始数据并处理
        except FileNotFoundError:
            print("铁汁，这个文件路径不存在!")
            return
        if data_name == 'data1':
            data = cleanData.clean_data1(data)
        elif data_name == 'data2':
            data = cleanData.clean_data2(data)
        elif data_name == 'data3':
            data = cleanData.clean_data3(data)
        elif data_name == 'data4':
            pass
        elif data_name == 'data5':
            pass
        # 处理空值和极值
        cleanData.process_null_value(data)
        cleanData.process_inf_value(data)
        self.dataInformation = get_dataInformation()
        self.train_x, self.test_x, self.train_y, self.test_y = prepare_data(data)  # 为模型准备数据
        self.train_Model()

    @staticmethod
    def get_data_path(data_name):
        """
        对已有数据集进行统一命名管理，为每个数据集进行编号

        Return: 数据集的绝对路径

        Parameters:
            data_name : 数据集的编号
        """
        filepath = None
        if data_name == "data2":
            filepath = Path(__file__).parent / "../static/data/AndroidAdware2017.csv"
        elif data_name == "data3":
            filepath = Path(__file__).parent / "../static/data/AndMal2020-Dynamic.csv"
        elif data_name == "data1":
            filepath = Path(__file__).parent / "../static/data/MalDroid-2020.csv"
        elif data_name == 'data4':
            filepath = Path(__file__).parent / "../static/data/URL.csv"
        elif data_name == 'data5':
            filepath = Path(__file__).parent / "../static/data/IDS.csv"
        elif data_name == 'train':
            filepath = Path(__file__).parent / "../static/make_data/num_train.csv"
        elif data_name == 'test':
            filepath = Path(__file__).parent / "../static/make_data/num_test.csv"
        return filepath


