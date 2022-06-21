"""
此数据集来自 https://www.unb.ca/cic/datasets/url-2016.html
只使用用其中的All.csv文件，里面包含了
"""
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
    path = "../static/ISCX-URL2016/All.csv"
    data = pd.read_csv(path)
    print(data.shape)
    data.rename(columns={"URL_Type_obf_Type": "Class"}, inplace=True)

    # 保存到data文件夹中
    with open('../static/data/URL.csv', 'w', newline='') as f:  # 使用newline，可以解决:将数据写入csv文件每条数据间有空行
        data.to_csv(f, index=False)


# process_data()
data = pd.read_csv('../static/data/URL.csv')
all = data.shape[0]
count = count_categories(data)
print("数据类别占比情况：")
for i in count:
    print('{}:{:.2%}'.format(i, count[i] / all))
