#基于movielens数据集的item协同过滤（字典版）
import math
import random
import pprint

#数据导入函数,评分数据为整形1-5
def loaddata(filename = 'u.data'):
    data = []
    for line in open(filename):
        user, item, record, timestamp = line
        data.append((user, item, int(record)))
    return data

def splitData(data, split_percent, seed = 2020):
    random.seed(seed)
    test = {}
    train = {}
    for user, item, record in data:
        if random.uniform(0, 1) >= split_percent:
            test.setdefault(user, {})
            test[user][item] = record
        else:
            train.setdefault(user, {})
            train[user][item] = record
    return train, test

#计算物品相似度
def itemSimilarity(train, method = 'IUF'):
    C = dict()#物品相似度矩阵中间量
    N = dict()#物品总数统计
    W = dict()#物品相似度结果

    for user, items in train.items():
        #遍历每一个user
        for item in items:
            N[item] = N.get(i, 0) + 1
            for j in items:
                if item == j:
                    continue
                C.setdefault(item, {})
                if method == 'IUF':
                    C[item][j] = C[item].get(j, 0) + 1/math.log(1 + len(items) * 1.0)
                else:
                    C[item][j] = C[item].get(j, 0) + 1
        for i, related_items in C.items():
            for j, cij in related_items.items():
                W.setdefault(i, {})
                W[i][j] = cij / math.sqrt(N[i] * N[j])
    return W
#计算完相似度以后利用item矩阵相乘推荐物品，遍历看过的电影，每个点一个相似度最高的先选一遍，然后根据相似度和评分再排一次序
def recommender(user, train, W, K, N):
    rank = dict()
    interacted_items = train[user]
    for i, pi in interacted_items.items():
        for j, wj in sorted(W[i].items(), key = lambda x:x[1], reverse = True)[0: K]:
            if j in interacted_items:
                continue
            rank[j] = rank.get(j, 0) + pi * wj
    return dict(sorted(rank.items(), key = lambda x: x[1], reverse = True)[0:K])
#计算准确率和召回率
def precisionandrecall(train, test, W, K, N) :
    hit = 0
    pre = 0
    rec = 0
    for user in train.keys():
        tu = test.get(user, {})
        rank = recommender(user, train, W, K, N)
        for item, pui in rank.items():
            if item in tu:
                hit += 1
        pre += N
        rec += len(tu)
    return hit/(pre * 1.0), hit/(rec * 1.0)
#计算覆盖率，覆盖率等于召回元素占总元素的占比
def Coverage(train, test, W, K = 1, N = 2):
    train = train
    recommend_items = set()
    all_items = set()
    for user, items in train.items():
        for i in items.key():
            all_items.add(i)
        rank = recommender(user, train, W, K, N)
        for i, _ in rank.items:
            recommend_items.add(i)
    return len(recommend_items) / len(all_items)

#计算物品流行度
def Popularity(train, test, W, K = 1, N = 2):
    item_popularity = dict()
    for user, item in train.items():
        for i in item.keys()
            item_popularity.setdefault(i, 0)
            item_popularity[i] += 1
    ret = 0
    n = 0
    for user in train.keys():
        rank = recommender(user, train, W, K = K, N = N)
        for item, _ in rank.items()
            ret += math.log(1 + item_popularity[item])
            n += 1
    ret /= n * 1.0
    return ret

def testRecommend(filename):
    data = loaddata(filename)
    train, test = splitData(data, 0.7, 2020)
    W = itemSimilarity(train, method= 'IUF')
    rank = recommender('344', train, W, 3, 10)
    pprint.pprint('向用户344推荐电影为：' + rank)
if __name__ == '__main__':
    testRecommend('u.data')



