import csv
import numpy as np
from sklearn.decomposition import PCA
import sys
from io import StringIO
import pickle
import os

def load_yr(yr, loc):
    vectors = np.load(f'{loc}{yr}-w.npy')
    words   = np.load(f'{loc}{yr}-vocab.pkl', allow_pickle=True)
    counts_path = os.path.join(loc, '..', 'counts', f'{yr}-counts.pkl')
    with open(counts_path, 'rb') as f:
        counts = pickle.load(f, encoding='latin1')
    return vectors, words, counts


def save_files(yrs, oldloc, newloc, label):
    for yr in yrs:
        print()
        print(yr)
        vectors, words, counts = load_yr(yr, oldloc)

        out_vec_path = f'{newloc}vectors_{label}{yr}.txt'
        out_vocab_path = f'{newloc}vocab/vocab_{label}{yr}.txt'

        with open(out_vec_path, 'w', newline='', encoding='utf-8') as f_vec, \
             open(out_vocab_path, 'w', newline='', encoding='utf-8') as f_vocab:

            csvwriter = csv.writer(f_vec, delimiter=' ')
            csvwritervoc = csv.writer(f_vocab, delimiter=' ')

            for en in range(len(vectors)):
                try:
                    row = [words[en]]
                    row.extend(vectors[en])
                    csvwriter.writerow(row)
                    csvwritervoc.writerow([words[en], counts[words[en]]])
                except Exception as e:
                    print('Error on word:', words[en], counts[words[en]], e)


if __name__ == "__main__":
    yrs = range(1910, 2000, 10)
    newloc = '../data/vectors/normalized_clean/'
    svd_loc = '../data/vectors/raw/svd/'
    save_files(yrs, svd_loc, newloc, 'svd')
    sgns_loc = '../data/vectors/raw/sgns/'
    save_files(yrs, sgns_loc, newloc, 'sgns')