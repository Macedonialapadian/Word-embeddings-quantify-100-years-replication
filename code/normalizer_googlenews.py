import numpy as np
import os
import re
from gensim.models import KeyedVectors

def normalize_googlenews(bin_filename, filename_output):
    countnorm0 = 0
    countnormal = 0

    output_dir = os.path.dirname(filename_output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    model = KeyedVectors.load_word2vec_format(bin_filename, binary=True)

    with open(filename_output, "w", encoding="utf-8", newline="") as f_out:
        for word_raw in model.index_to_key:
            word = re.sub('[^a-z]+', '', word_raw.strip().lower())
            if len(word) < 2:
                continue

            vec = model[word_raw]
            norm = np.linalg.norm(vec)
            if norm < 1e-2:
                countnorm0 += 1
                continue

            vec_normed = vec / norm
            vec_str = " ".join(str(float(v)) for v in vec_normed)
            f_out.write(word + " " + vec_str + "\n")
            countnormal += 1

    print(countnorm0, countnormal)

if __name__ == "__main__":
    filename = "vectors/raw/GoogleNews-vectors-negative300.bin"
    filename_output = "vectors/normalized_clean/vectorsGoogleNews_exactclean.txt"
    normalize_googlenews(filename, filename_output)