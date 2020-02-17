from tqdm import tqdm as tqdm
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs


def read_lst(dpth):
    res_lst = []
    with open(dpth, 'r', encoding='utf-8') as fp:
        for x in tqdm(fp):
            res_lst.append(x.strip().strip("\n"))
    print(dpth, " reading finish!")
    return res_lst


def smarts_to_vector(tmp_lst):
    train_lst = []
    for template in tqdm(tmp_lst):
        express = template.split('>')
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

        train_lst.append(np.concatenate((arr_tgt, arr_src)))

    return train_lst


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


def save_list(lst, path):
    with open(path, 'w') as f:
        for i in lst:
            for ele in i:
                f.write(str(int(ele)))
            f.write('\n')


if __name__ == '__main__':
    template_lst = read_lst('/Users/liuzixuan/Downloads/template_all.txt')
    train_lst = smarts_to_vector(template_lst[:100])
    save_list(train_lst, '/Users/liuzixuan/Downloads/da.txt')
