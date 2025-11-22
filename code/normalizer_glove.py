import numpy as np
import os
import csv
import re
import sys

def normalize(filename, filename_output):
    countnorm0 = 0
    countnormal = 0

    output_dir = os.path.dirname(filename_output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(filename, 'r', encoding='utf-8') as f_in, \
         open(filename_output, 'w', encoding='utf-8', newline='') as f_out:
        for line in f_in:
            parts = line.rstrip().split()
            if not parts:
                continue

            word_raw = parts[0]
            word = re.sub('[^a-z]+', '', word_raw.strip().lower())
            if len(word) < 2:
                continue

            vec = [float(x) for x in parts[1:]]
            norm = np.linalg.norm(vec)
            if norm < 1e-2:
                countnorm0 += 1
                continue

            vec_normed = [str(v / norm) for v in vec]
            f_out.write(word + ' ' + ' '.join(vec_normed) + '\n')
            countnormal += 1

    print(countnorm0, countnormal)

if __name__ == "__main__":
    filename = "vectors/raw/glove.42B.300d.txt"
    filename_output = "vectors/normalized_clean/vectorscommoncrawlglove.txt"
    normalize(filename, filename_output)
