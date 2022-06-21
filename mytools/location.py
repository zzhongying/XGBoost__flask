import cmath
import json
import random

# path1 = './static/data/sca_location.json'
# path2 = './static/data/sca_out_location.json'
# load = open(path1, encoding='utf-8')
# out = open(path2, 'w', encoding='utf-8')
# data = json.load(load)
# ready_data = data['data']
#
# init_data = []
# category = [['a', 'b'], ['a', 'c'], ['a', 'd'],
#             ['a', 'e'], ['b', 'a'], ['c', 'a'], ['d', 'a'],
#             ['e', 'a']]
center_sca = [[37.5, 37.5], [37.5, 12.5], [12.5, 12.5], [12.5, 37.5]]


# [0.5, 'a', ['a', 'b']]
def get_location(datas, category):
    change_data = []
    p = 17.6 * datas[0]
    # a = random.uniform(0.125 * cmath.pi, 0.375 * cmath.pi)
    loc = datas[2]
    if loc == category[0]:
        a = random.uniform(0.125 * cmath.pi, 0.375 * cmath.pi)
        cn1 = cmath.rect(p, a)
        x = round(cn1.real + center_sca[0][0], 2)
        y = round(cn1.imag + center_sca[0][1], 2)
    elif loc == category[1]:
        a = random.uniform(1.625 * cmath.pi, 1.875 * cmath.pi)
        cn1 = cmath.rect(p, a)
        x = round(cn1.real + center_sca[1][0], 2)
        y = round(cn1.imag + center_sca[1][1], 2)
    elif loc == category[2]:
        a = random.uniform(1.125 * cmath.pi, 1.375 * cmath.pi)
        cn1 = cmath.rect(p, a)
        x = round(cn1.real + center_sca[2][0], 2)
        y = round(cn1.imag + center_sca[2][1], 2)
    elif loc == category[3]:
        a = random.uniform(0.625 * cmath.pi, 0.875 * cmath.pi)
        cn1 = cmath.rect(p, a)
        x = round(cn1.real + center_sca[3][0], 2)
        y = round(cn1.imag + center_sca[3][1], 2)
    elif loc == category[4]:
        a = random.uniform(1.125 * cmath.pi, 1.375 * cmath.pi)
        cn1 = cmath.rect(p, a)
        x = round(cn1.real + center_sca[0][0], 2)
        y = round(cn1.imag + center_sca[0][1], 2)
    elif loc == category[5]:
        a = random.uniform(0.625 * cmath.pi, 0.875 * cmath.pi)
        cn1 = cmath.rect(p, a)
        x = round(cn1.real + center_sca[1][0], 2)
        y = round(cn1.imag + center_sca[1][1], 2)
    elif loc == category[6]:
        a = random.uniform(0.125 * cmath.pi, 0.375 * cmath.pi)
        cn1 = cmath.rect(p, a)
        x = round(cn1.real + center_sca[2][0], 2)
        y = round(cn1.imag + center_sca[2][1], 2)
    else:
        a = random.uniform(1.625 * cmath.pi, 1.875 * cmath.pi)
        cn1 = cmath.rect(p, a)
        x = round(cn1.real + center_sca[3][0], 2)
        y = round(cn1.imag + center_sca[3][1], 2)
    change_data.append(x)
    change_data.append(y)
    change_data.append(1)
    change_data.append(datas[1])
    change_data.append(datas[0])
    return change_data
#
#
# print(ready_data)
# # 数据：[0.45,0.25,"a"]
# for i in ready_data:
#     locc = get_location(i)
#     init_data.append(locc)
#
# dicts = {}
# dicts["data"] = init_data
# json.dump(dicts, out)