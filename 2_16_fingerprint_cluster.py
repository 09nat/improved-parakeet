from tqdm import tqdm as tqdm
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs

wrong_cnt = 0


def read_lst(dpth):
    res_lst = []
    with open(dpth, 'r', encoding='utf-8') as fp:
        for x in tqdm(fp):
            res_lst.append(x.strip().strip("\n"))
    print(dpth, " reading finish!")
    return res_lst


def get_set(tmp_lst):
    ele_set = set()
    for tmp in tmp_lst:
        for ele in tmp:
            ele_set.add(ele)
    return ele_set


def smarts_to_vector(tmp_lst):
    global wrong_cnt
    smiles_dict = dict()
    train_lst = []
    for template in tqdm(tmp_lst):
        express = template.split('>')
        if len(express) != 3:
            print(template)
            wrong_cnt += 1
            continue
        # try:
        #     arr_tgt = preprocess(express[0].lstrip('(').rstrip(')'))
        #     arr_src = preprocess(express[-1])
        #     arr = np.concatenate((arr_tgt, arr_src))
        #     smiles_dict[arr] = template
        #     train_lst.append(arr)
        # except:
        #     wrong_cnt += 1

        arr_tgt = preprocess(express[0].lstrip('(').rstrip(')'))
        arr_src = preprocess(express[-1])
        arr = np.concatenate((arr_tgt, arr_src))
        smiles_dict[tuple(arr)] = template
        train_lst.append(arr)

    return train_lst, smiles_dict


def preprocess(X):
    # Compute fingerprint from mol to feature
    # mol = Chem.MolFromSmiles(X)
    mol = Chem.MolFromSmarts(X)
    # fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=int(fp_dim), useChirality=True)
    fp = Chem.RDKFingerprint(mol)

    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits())
    arr[onbits] = 1
    return arr


if __name__ == '__main__':
    template_lst = read_lst('/Users/liuzixuan/Downloads/template_all.txt')

    # template_lst = template_lst[:100]

    train_lst, smiles_dict = smarts_to_vector(template_lst)
    print('smile_to_vector_over')
    print(wrong_cnt)
    pass
    # for key, value in smiles_dict.items():
    #     print(key, value, sep='\t')
    train_np_lst = np.array(train_lst)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_np_lst = min_max_scaler.fit_transform(train_np_lst)
    print('min_max_scaler_over')
    minibatch_kmeans = MiniBatchKMeans(n_clusters=10, max_iter=100)
    minibatch_kmeans.fit(train_np_lst)
    train_labels = minibatch_kmeans.labels_
    vector_labels = np.column_stack((train_np_lst, train_labels))
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
        fp_lst[int(train_labels[i])].write(smiles_dict[tuple(train_lst[i])] + '\n')
