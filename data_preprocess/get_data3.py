import pandas as pd
import os


# 计算每个类别的占比===============================================================================
def count_Categories(data):
    count = {}
    temp = []
    for i in data["Class"]:
        if i not in temp:
            temp.append(i)
            count[i] = 1
        else:
            count[i] += 1
    return count


def traverseDocument():
    root_path = '../static/AndMal2020-Dynamic-BeforeAndAfterReboot'
    filenames = os.listdir(root_path)
    data = None
    mapCategories = {}  # 类别映射表
    global n
    n = 0
    # 遍历文件夹下的28个数据文件
    for i in filenames:
        category = i.split('_')[0]
        categories = []
        path = os.path.join(root_path,i)
        print(path)

        # 读取原始数据并处理
        single_data = pd.read_csv(path)

        # 排除数据量较小的数据和无类别的数据
        if category == 'No' or category == 'Zero':
            continue
        if single_data.shape[0] < 1000:
            continue

        categories.append(category)
        # 数据清洗
        single_data = single_data.drop("Hash", axis=1)  # 'Hash'，字段不可用
        single_data = single_data.drop("Category", axis=1)  # 'Category"'，字段不可用

        # 替换类别名称
        single_data.rename(columns={'Family': 'Class'}, inplace=True)

        # 将类别..
        single_data["Class"] = category

        if isinstance(data, pd.DataFrame):
            data = pd.concat([data, single_data], ignore_index=True)
        else:
            data = single_data
    # 保存到data文件夹中
    with open('../static/data/AndMal2020-Dynamic.csv', 'w') as f:
        data.to_csv(f, index=False)


# traverseDocument()
data = pd.read_csv('../static/data/AndMal2020-Dynamic.csv')
all = data.shape[0]
count = count_Categories(data)
print("数据类别占比情况：")
for i in count:
    print('{}:{:.2%}'.format(i, count[i] / all))