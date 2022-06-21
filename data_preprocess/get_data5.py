"""
此数据集来自 https://www.unb.ca/cic/datasets/ids-2017.html
只使用用其中的
Brute Force FTP, Brute Force SSH, DoS, Heartbleed, Web Attack, Infiltration, Botnet and DDoS
"""
import pandas as pd
import os


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


def process_data():
    path = '../static/CIC-IDS2017'

    filenames = os.listdir(path)
    data = None
    for filename in filenames:
        file = os.path.join(path, filename)
        print(file)
        single_data = pd.read_csv(file)
        single_data = single_data[single_data[' Label']!='BENIGN']

        temp = []
        for i in single_data[" Label"]:
            if i not in temp:
                temp.append(i)
        temp1 = [value.split(' ')[0] for value in temp ]
        # replace_data = {k: v for k, v in zip(temp,temp1)}
        single_data[' Label'].replace(temp,temp1,inplace=True)
        if isinstance(data, pd.DataFrame):
            data = pd.concat([data, single_data], ignore_index=True)
        else:
            data = single_data
    data.rename(columns={' Label':'Class'},inplace=True)

    # 保存到data文件夹中
    with open('../static/data/IDS.csv', 'w', newline='') as f:  # 使用newline，可以解决:将数据写入csv文件每条数据间有空行
        data.to_csv(f, index=False)


# process_data()
data = pd.read_csv('../static/data/IDS.csv')
# data.dropna(inplace=True)
# print(data[data['Class'] == 'Heartbleed'])
all = data.shape[0]
print(all)
count = count_categories(data)
print("数据类别占比情况：")
for i in count:
    print('{}:{:.2%}'.format(i, count[i] / all))

# path = '../static/CIC-IDS2017'
#
# filenames = os.listdir(path)
# for filename in filenames:
#     file = os.path.join(path,filename)
#     print(file)
#     data = pd.read_csv(file)
#
#     all = data.shape[0]
#     count = {}
#     temp = []
#     for i in data[" Label"]:
#         if i not in temp:
#             temp.append(i)
#             count[i] = 1
#         else:
#             count[i] += 1
#     print("类别占比情况：")
#     for i in count:
#         print('{}:{:.2%}'.format(i, count[i] / all))