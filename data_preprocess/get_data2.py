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
    path = "../static/AndroidAdware2017/TotalFeatures-ISCXFlowMeter.csv"
    data = pd.read_csv(path)
    print(data.shape)
    data.rename(columns={"calss": "Class"}, inplace=True)

    # 对数据量进行平衡，benign类 和 asware类的数量太多，对其进行抽样
    categories = ['asware', 'GeneralMalware', 'benign']
    dict_count = count_categories(data)
    num_genMal = dict_count['GeneralMalware']
    num_asware = dict_count['asware']
    num_benign = dict_count['benign']
    data1 = data[data['Class'] == 'asware'].sample(frac=2 * num_genMal / num_asware, axis=0)
    data2 = data[data['Class'] == 'benign'].sample(frac=2 * num_genMal / num_benign, axis=0)
    data3 = data[data['Class'] == 'GeneralMalware']
    data = pd.concat([data1, data2], ignore_index=True)
    data = pd.concat([data, data3], ignore_index=True)

    # 将类别的数据类型替换成数字
    # t = {value: key for key, value in dict(enumerate(categories)).items()}
    # data["Class"] = data["Class"].replace(t)
    # print(count_categories(data))

    # 保存到data文件夹中
    with open('../static/data/AndroidAdware2017.csv', 'w') as f:
        data.to_csv(f, index=False)


# process_data()
data = pd.read_csv('../static/data/AndroidAdware2017.csv')
all = data.shape[0]
count = count_categories(data)
print("数据类别占比情况：")
for i in count:
    print('{}:{:.2%}'.format(i, count[i] / all))