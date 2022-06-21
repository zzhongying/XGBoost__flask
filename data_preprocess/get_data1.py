import pandas as pd


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
    # 读取数据
    path = "..\static/MalDroid-2020/feature_vectors_syscallsbinders_frequency_5_Cat.csv"
    data = pd.read_csv(path)

    # 将类别的数据类型替换成数字
    # temp = []
    # for i in data["Class"]:
    #     if i not in temp:
    #         temp.append(i)
    # t = {}
    #
    # n = 0
    # for i in temp:
    #     t[i] = n
    #     n += 1
    # data["Class"] = data["Class"].replace(t)

    # 将类别的数据类型替换为相应的字符串, 根据 数据集网站 获取数字类别分别代表的类别
    replace_data = {1: 'Advertising_software',
                    2: 'Bank_malware',
                    3: 'SMS_malware',
                    4: 'Risk_software',
                    5: 'Benign_application'}
    data["Class"] = data["Class"].replace(replace_data)

    # 保存到文件夹
    with open('../static/data/MalDroid-2020.csv', 'w') as f:
        data.to_csv(f, index=False)


# process_data()
data = pd.read_csv('../static/data/MalDroid-2020.csv')
all = data.shape[0]
count = count_categories(data)
print("数据类别占比情况：")
for i in count:
    print('{}:{:.2%}'.format(i, count[i] / all))
