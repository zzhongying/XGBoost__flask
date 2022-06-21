import pandas as pd
import json

def save_columns():
    data = pd.read_csv('../static/MalDroid-2020/feature_vectors_syscallsbinders_frequency_5_Cat.csv')
    data = data.columns
    print(type(data))
    data = list(data)
    with open('../static/data/n_featureNames.json', 'w') as f:
        f.write('[')
        for i in range(len(data)):
            json.dump(data[i], f)
            if i < len(data) - 1:
                f.write(',\n')
        f.write(']')


def merge_columns():
    data = pd.read_csv('../static/MalDroid-2020/feature_vectors_syscallsbinders_frequency_5_Cat.csv')
    merge = ['FS_ACCESS', 'FS_PIPE_ACCESS', 'NETWORK_ACCESS']
    columns = data.columns
    for i in merge:
        hebing = [c for c in columns if c.find(i) != -1]
        data[i] = data[hebing].apply(lambda x:x.sum(),axis=1)
        data.drop(hebing,axis=1,inplace=True)
    columns = data.columns
    columns = list(columns)
    return columns
    # with open('../static/data/featureNames.json', 'w') as f:
    #     f.write('[')
    #     for i in range(len(columns)):
    #         json.dump(columns[i], f)
    #         if i < len(columns) - 1:
    #             f.write(',\n')
    #     f.write(']')


def filter_data(data = None):
    if data == None:
        f = open('../static/data/n_featureNames.json')
        data = json.load(f)
        f.close()
    new_data = []
    fil = ['add','get','has','Accessibility','To','Without','cancel','current','display','With','Tag','finish','Current'
           ,'Enabled','List','inKeyguardRestricted','is','on','Get','query','register','remove','set','start','stop',
           'unregister','update','regter','Master','Last','Subtype','Names','Messenger','App','View','Wid','play',
           'Input','Changedener','Changed','For','Suggesti','Requested','show']
    replace_fil = {
        'remove': 'rm',
        'Current':'Cu',
        'Input': 'In',
        'Subtype':'Sty',
        'get':'g',
        'App':'Ap',
        'Application':'Ap',
        'List':'Li',
        'Enabled':'Ab',
        'Automatically':'Au',
        'Master':'Ma',
        'Search':'Se',
        'System':'Sys',
        'Global':'Glo',
        'Status':'Sta',
        'Features':'Fea',
        'Library':'Lib',
        'Service':'Serv',
        'Network':'Net'
    }
    for column in data:
        col = column
        if len(col) > 14:
            for k,v in replace_fil.items():
                col = col.replace(k,v)
            for i in fil:
                col = col.replace(i,'')
                if len(col) <= 14:
                    break
        new_data.append(col)
    return new_data


def check_repeats():
    with open('../static/data/new_featureNames.json', 'r') as f:
        data = json.load(f)
        s = set(data)
        print(len(s),len(data))
        re_dict = {}
        for col in s:
            count = data.count(col)
            if count > 1:
                print([i+1 for i,x in enumerate(data) if x == col])
                re_dict.update({col:count})
        print(re_dict)


def final_columns(columns):
    new_columns = []
    fil = ['add', 'get', 'has', 'Accessibility', 'To', 'Without', 'cancel', 'current', 'display', 'With', 'Tag',
           'finish', 'Current', 'Enabled', 'List', 'inKeyguardRestricted', 'is', 'on', 'Get', 'query', 'register',
           'remove', 'set', 'start', 'stop', 'unregister', 'update', 'regter', 'Master', 'Last', 'Subtype', 'Names',
           'Messenger', 'App', 'View', 'Wid', 'play', 'Input', 'Changedener', 'Changed', 'For', 'Suggesti', 'Requested',
           'show']
    replace_fil = {
        'remove': 'rm',
        'Current': 'Cu',
        'Input': 'In',
        'Subtype': 'Sty',
        'get': 'g',
        'App': 'Ap',
        'Application': 'Ap',
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
        col.strip('_')
        if len(col) > 14:
            for k, v in replace_fil.items():
                col = col.replace(k, v)
            for i in fil:
                col = col.replace(i, '')
                if len(col) <= 14:
                    break
        new_columns.append(col)
    return new_columns

# filter_data()
# check_repeats()
merge_columns()