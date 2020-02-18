import json
import os
import math
import random
class item_based_CF:

    def __init__(self,K,n_items):
        self.K=K  # 选择k个相似item
        self.n_items=n_items  # 每个用户推荐n_item个item
        self.train,self.test=self.get_train_test(3,47)
        self.N_item,self.N_u=self.get_nItem_nU()
        self.W=self.get_similarity()

    def get_train_test(self,k,seed,M=9):
        train_path='./data/movieLens_data/train.json'
        test_path ='./data/movieLens_data/test.json'
        if (os.path.exists(train_path)) and (os.path.exists(test_path)):
            train_dict=json.load(open(train_path))
            test_dict=json.load(open(test_path))
            print('训练集、测试集已存在，可直接使用！')
            return train_dict,test_dict
        else:
            ratings_path='./data/movieLens_data/ratings.csv'
            train_dict=dict()
            test_dict=dict()
            random.seed(seed)
            with open(ratings_path,'r') as f:
                for line in f.readlines():
                    if not line.startswith('UserID'):
                        user,item,rating=line.split(',')[:3]
                        if random.randint(0,M)==k:
                            test_dict.setdefault(user,{})[item]=rating
                        else:
                            train_dict.setdefault(user,{})[item]=rating
            json.dump(train_dict,open(train_path,'w'))
            json.dump(test_dict, open(test_path, 'w'))
            print('训练集测试集已保存')
            return train_dict,test_dict

    def get_nItem_nU(self):
        N_item = dict()
        N_u = dict()
        for u, item_rating in self.train.items():
            for item in item_rating.keys():
                N_item.setdefault(item, set())
                N_u.setdefault(u, set())
                N_item[item].add(u)
                N_u[u].add(item)
        return N_item,N_u

    def get_similarity(self):
        sim_path='./data/movieLens_data/similarity.json'
        if os.path.exists(sim_path):
            W=json.load(open(sim_path))
            print('相似度矩阵已存在，路径为{}可直接使用'.format(sim_path))
            return W
        else:
            # 构建相似度矩阵的分子（即item共现矩阵）
            C=dict()
            W=dict()
            for item in self.N_u.values():
                for item_1 in item:
                    C.setdefault(item_1,{})
                    for item_2 in item:
                        C[item_1].setdefault(item_2, 0)
                        if item_1==item_2:
                            continue
                        C[item_1][item_2]+=1

            # 考虑进分母，得到item的相似度矩阵
            for it,it_coocur in C.items():
                W.setdefault(it, {})
                for ite,occur in it_coocur.items():
                    fenmu=math.sqrt(len(self.N_item[it])*len(self.N_item[ite]))
                    W[it].setdefault(ite,0.0)
                    W[it][ite]=occur/fenmu
            json.dump(W,open(sim_path,'w'))
            print('相似度矩阵已保存在{}中'.format(sim_path))
            return W

    def recommend(self,user):
        p=dict()
        for item in self.N_u[user]:
            item_neighbors=sorted(self.W[item].items(),key=lambda k:k[1],reverse=True)[:self.K]
            for item_neigh, sim in item_neighbors:
                if item_neigh in self.N_u[user]:  # 判定item的邻居是否是user评分过的。
                    # self.train[str(user)][item_neigh] 是字符串格式
                    continue
                p.setdefault(item_neigh,0.0)
                p[item_neigh] += (self.W[item][item_neigh] * int(self.train[user][item]))
        recommend_list = sorted(p.items(), key=lambda k: k[1], reverse=True)[:self.n_items]
        print('用户{}的top{}推荐列表为\n{}'.format(user, self.n_items, recommend_list))
        return recommend_list

    def evaluate(self):
        N_U_test=dict()
        users=self.test.keys()
        print('test中有{}个用户'.format(len(users)))
        for user,item_score in self.test.items():
            N_U_test.setdefault(user,set())
            for item,score in item_score.items():
                N_U_test[user].add(item)
        precision_list=[]
        recall_list=[]
        for u in users:
            hit=0
            rec=self.recommend(u)
            for rec_item,pro in rec:
                if rec_item in N_U_test[u]:
                    hit+=1
            precision=hit/self.n_items
            recall=hit/len(N_U_test[u])
            precision_list.append(precision)
            recall_list.append(recall)
        print('准确率为{},召回率为{}'.format(sum(precision_list)/len(users),sum(recall_list)/len(users)))
        return sum(precision_list)/len(users),sum(recall_list)/len(users)

if __name__=='__main__':
    item_CF=item_based_CF(8,10)
    item_CF.recommend('1')
    item_CF.evaluate()