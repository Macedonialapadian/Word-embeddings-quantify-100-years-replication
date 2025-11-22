import csv
import numpy as np
from sklearn.decomposition import PCA
import sys
from io import StringIO
import pickle
import os



def save_files(yrs, oldloc, newloc, label):
    for yr in yrs:
        print()
        print(yr)
        vectors = np.load(f'{oldloc}{yr}-w.npy')
        words   = np.load(f'{oldloc}{yr}-vocab.pkl', allow_pickle=True)

        out_vec_path = f'{newloc}vectors_{label}{yr}.txt'

        with open(out_vec_path, 'w', newline='', encoding='utf-8') as f_vec:

            csvwriter = csv.writer(f_vec, delimiter=' ')

            for en in range(len(vectors)):
                try:
                    row = [words[en]]
                    row.extend(vectors[en])
                    csvwriter.writerow(row)
                except Exception as e:
                    print('Error on word:', words[en], counts[words[en]], e)


if __name__ == "__main__":
    yrs = range(1910, 2000, 10)
    newloc = 'vectors/normalized_clean/'
    all_english_loc = 'vectors/raw/all_english/'
    save_files(yrs, all_english_loc, newloc, 'sgns')
    # here we still make the all english data with label sgns so that we can directly use changes_over_time.py without any modification