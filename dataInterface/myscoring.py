from sklearn import metrics
from sklearn.preprocessing import label_binarize


# 正确率
def score_accuracy(y_true,y_pred):
    return metrics.accuracy_score(y_true,y_pred)


# f1-score
def score_f1(y_true,y_pred):
    return metrics.f1_score(y_true,y_pred,average='macro')


# 召回率
def score_recall(y_true,y_pred):
    return metrics.recall_score(y_true,y_pred,average='macro')


# auc值
def score_auc(y_true,y_pred):
    num_category = max(list(y_true)+list(y_pred)) + 1   # 类别数是从零开始
    labels = [i for i in range(num_category)]
    test_y = label_binarize(y_true, classes=labels)
    ypred = label_binarize(y_pred, classes=labels)
    AUC = []
    for i in range(len(labels)):
        AUC.append(metrics.roc_auc_score(test_y[:, i], ypred[:, i]))
    return sum(AUC)/len(AUC)    # 暂时使用


# 精准率
def score_precision(y_true,y_pred):
    return metrics.precision_score(y_true,y_pred,average='macro')


# 特异度
def score_specificity(y_true,y_pred):
    matrix = metrics.multilabel_confusion_matrix(y_true,y_pred)
    matrix = list(matrix)
    result = []
    for ma in matrix:
        result.append(ma[1][1] / (ma[0][1] + ma[1][1]))
    return sum(result)/len(result)


# 误报率
def score_falseAlarm(y_true,y_pred):
    specificity = score_specificity(y_true,y_pred)
    return 1-specificity


# 漏报率
def score_missRate(y_pred,y_true):
    recall = metrics.recall_score(y_true,y_pred,average='macro')
    return 1-recall