from tqdm import tqdm as tqdm
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
import BitVector


def read_lst(dpth):
    res_lst = []
    with open(dpth, 'r', encoding='utf-8') as fp:
        for x in tqdm(fp):
            res_lst.append(x.strip().strip("\n"))
    print(dpth, " reading finish!")
    return res_lst


if __name__ == '__main__':
    template_lst = read_lst('/Users/liuzixuan/Downloads/template_all.txt')
    train_str_lst = read_lst('/Users/liuzixuan/Downloads/da.txt')
    train_lst = []
    for i in range(len(train_str_lst)):
        train_lst.append([])
        for ele in train_str_lst[i]:
            train_lst[i].append(int(ele))
        train_lst[i] = BitVector.BitVector(bitlist=train_lst[i])

    # for key, value in smiles_dict.items():
    #     print(key, value, sep='\t')
    train_np_lst = np.array(train_lst)
    # min_max_scaler = preprocessing.MinMaxScaler()
    # train_np_lst = min_max_scaler.fit_transform(train_np_lst)
    # print('min_max_scaler_over')
    minibatch_kmeans = MiniBatchKMeans(n_clusters=10, max_iter=100)
    minibatch_kmeans.fit(train_np_lst)
    train_labels = minibatch_kmeans.labels_
    print('model_train_over')
    # with open('/Users/liuzixuan/Downloads/Scaler_Result.txt', 'w') as fp:
    #     for i in range(len(train_lst)):
    #         fp.write(smiles_dict[tuple(train_lst[i])])
    #         fp.write('\t' + str(train_labels[i]) + '\n')

    fp_lst = []
    fp0 = open('/Users/liuzixuan/Downloads/fingerprint_cluster/cluster_0.txt', 'a')
    fp_lst.append(fp0)

    fp1 = open('/Users/liuzixuan/Downloads/fingerprint_cluster/cluster_1.txt', 'a')
    fp_lst.append(fp1)

    fp2 = open('/Users/liuzixuan/Downloads/fingerprint_cluster/cluster_2.txt', 'a')
    fp_lst.append(fp2)

    fp3 = open('/Users/liuzixuan/Downloads/fingerprint_cluster/cluster_3.txt', 'a')
    fp_lst.append(fp3)

    fp4 = open('/Users/liuzixuan/Downloads/fingerprint_cluster/cluster_4.txt', 'a')
    fp_lst.append(fp4)

    fp5 = open('/Users/liuzixuan/Downloads/fingerprint_cluster/cluster_5.txt', 'a')
    fp_lst.append(fp5)

    fp6 = open('/Users/liuzixuan/Downloads/fingerprint_cluster/cluster_6.txt', 'a')
    fp_lst.append(fp6)

    fp7 = open('/Users/liuzixuan/Downloads/fingerprint_cluster/cluster_7.txt', 'a')
    fp_lst.append(fp7)

    fp8 = open('/Users/liuzixuan/Downloads/fingerprint_cluster/cluster_8.txt', 'a')
    fp_lst.append(fp8)

    fp9 = open('/Users/liuzixuan/Downloads/fingerprint_cluster/cluster_9.txt', 'a')
    fp_lst.append(fp9)

    for i in tqdm(range(len(train_lst))):
        fp_lst[int(train_labels[i])].write(template_lst[i] + '\n')
