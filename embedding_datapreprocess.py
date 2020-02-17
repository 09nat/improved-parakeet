from tqdm import tqdm as tqdm
import fasttext
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans


def read_lst(dpth):
    res_lst = []
    with open(dpth, 'r', encoding='utf-8') as fp:
        for x in tqdm(fp):
            res_lst.append(x.strip().strip("\n"))
    print(dpth, " reading finish!")
    return res_lst


def template_to_vector(tmp_lst, model):
    train_lst = []
    for template in tqdm(tmp_lst):
        arr = np.zeros(100)
        for ele in template:
            arr += model[ele]
        arr /= 100
        train_lst.append(arr)
    return train_lst


def save_list(lst, path):
    with open(path, 'w') as f:
        for i in lst:
            for ele in i:
                f.write('%.1f ' % ele)
            f.write('\n')


if __name__ == '__main__':
    template_lst = read_lst('/Users/liuzixuan/Downloads/template_all.txt')
    model = fasttext.load_model('/Users/liuzixuan/Downloads/template_word2vec_model.bin')
    train_lst = template_to_vector(template_lst[:100], model)
    save_list(train_lst, '/Users/liuzixuan/Downloads/dada.txt')
