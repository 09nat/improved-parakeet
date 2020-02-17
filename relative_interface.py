import pandas as pd
from tqdm import tqdm as tqdm
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
import BitVector
import math


def preprocess(X, fp_dim):
    # Compute fingerprint from mol to feature
    mol = Chem.MolFromSmiles(X)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=int(fp_dim), useChirality=True)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits())
    arr[onbits] = 1
    return arr


def smiles_to_vector(tmp_lst):
    fp_lst = []
    for template in tqdm(tmp_lst):
        express = template.split('>')
        arr_tgt = preprocess(express[0], 2048)
        arr_src = preprocess(express[-1], 2048)
        arr = np.concatenate((arr_tgt, arr_src))
        fp_lst.append(BitVector.BitVector(bitlist=arr))
    return fp_lst


def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    # zero_list = [0] * len(x)
    x = list(map(int, list(str(x))))
    y = list(map(int, list(str(y))))
    #
    # if x == zero_list or y == zero_list:
    #     return float(1) if x == y else float(0)

    res = [[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))]
    sum0 = 0
    for i in res:
        sum0 += i[0]
    sum1 = 0
    for i in res:
        sum1 += i[1]
    sum2 = 0
    for i in res:
        sum2 += i[2]

    cos = sum0 / (math.sqrt(sum1) * math.sqrt(sum2))

    # Nomalize to [0, 1]
    return 0.5 * cos + 0.5 if norm else cos


def get_relative(reactions_lst):
    df = pd.read_csv('/Users/liuzixuan/Downloads/proc_small.csv')
    cano_rxn_smiles = df['cano_rxn_smiles']
    smiles_lst = []
    for ele in cano_rxn_smiles:
        smiles_lst.append(str(ele))
    fp_lst = smiles_to_vector(smiles_lst)
    reactions_fp_lst = smiles_to_vector(reactions_lst)

    relative_lst = []
    for i in range(len(reactions_lst)):
        relative_lst.append([])
        for fp_i in range(len(fp_lst)):
            if len(relative_lst[i]) < 5:
                relative_lst[i].append((smiles_lst[fp_i], cosine_similarity(reactions_fp_lst[i], fp_lst[fp_i])))
                # print(relative_lst[i][-1])
                if len(relative_lst[i]) == 5:
                    relative_lst[i].sort(key=lambda x: x[1], reverse=True)
                continue
            tup = (smiles_lst[fp_i], cosine_similarity(reactions_fp_lst[i], fp_lst[fp_i]))
            # print(tup)
            if tup[1] > relative_lst[i][4][1]:
                # print('wowowo')
                relative_lst[i].pop()
                relative_lst[i].append(tup)
                relative_lst[i].sort(key=lambda x: x[1], reverse=True)
    # print(relative_lst[0])
    ans_lst = []
    for i in range(len(relative_lst)):
        ans_lst.append([])
        for j in relative_lst[i]:
            ans_lst[i].append(j[0])
    return ans_lst


if __name__ == '__main__':
    l = []
    l.append('C#CC(C)Br.O=CC(Cl)(Cl)Cl>>C#CC(C)C(O)C(Cl)(Cl)Cl')
    l.append('CCC#Cc1ccccc1.COC(=O)c1ccc(-n2ccc3ccccc32)c(N)c1>>CCC1C(c2ccccc2)=Nc2cc(C(=O)OC)ccc2-n2c1cc1ccccc12')
    ans = get_relative(l)
    print(ans[0], ans[1], sep='\n')
