import sys
import numpy as np


# 对数据进行空值处理
def process_null_value(data):
    # flag = np.any(data.isnull())
    # print(flag)
    # if flag:
    #     print("数据中存在空值条数：",end='')
    #     print(data[data.isnull().T.any()].shape[0])
    data.fillna(0, inplace=True)


# 对数据进行inf值处理
def process_inf_value(data):
    replace_value = sys.maxsize
    data.replace(np.inf, replace_value, inplace=True)


#  定制化的数据清洗，只针对某一个数据集
def data1_merge_columns(data):
    # 合并某些相似的属性
    merge = ['FS_ACCESS', 'FS_PIPE_ACCESS', 'NETWORK_ACCESS']
    columns = data.columns
    for i in merge:
        hebing = [c for c in columns if c.find(i) != -1]
        data[i] = data[hebing].apply(lambda x:x.sum(),axis=1)
        data.drop(hebing,axis=1,inplace=True)
    return data


def data1_process_columns(columns):
    # 将属性名称进行一定替换，使得较长的属性变短
    new_columns = []
    fil = ['add', 'has', 'Accessibility', 'To', 'Without', 'cancel', 'current', 'display', 'With', 'Tag',
           'finish', 'Enabled', 'List', 'inKeyguardRestricted', 'is', 'on', 'Get', 'query', 'register',
           'set', 'start', 'stop', 'unregister', 'update', 'regter', 'Master', 'Last', 'Names','Messenger',
           'View', 'Wid', 'play', 'Changedener', 'Changed', 'For', 'Suggesti', 'Requested', 'show']
    replace_fil = {
        'remove': 'rm',
        'Current': 'Cu',
        'Input': 'In',
        'Subtype': 'Sty',
        'get': 'g',
        'Application': 'Ap',
        'App': 'Ap',
        'List': 'Li',
        'Enabled': 'Ab',
        'Automatically': 'Au',
        'Master': 'Ma',
        'Search': 'Se',
        'System': 'Sys',
        'Global': 'Glo',
        'Status': 'Sta',
        'Features': 'Fea',
        'Library': 'Lib',
        'Service': 'Serv',
        'Network': 'Net'
    }
    for column in columns:
        col = column
        col = col.strip('_')  # 去除属性两边的下划线
        if len(col) > 14:
            for k, v in replace_fil.items():
                col = col.replace(k, v)
            for i in fil:
                col = col.replace(i, '')
                if len(col) <= 14:
                    break
        new_columns.append(col)
    return new_columns


def clean_data1(data):
    data = data1_merge_columns(data)
    columns = data1_process_columns(data.columns)
    data.columns = columns
    return data


def clean_data2(data):
    # 处理参数名称过长的问题
    return data


def clean_data3(data):
    # 处理参数名称过长的问题
    return data
