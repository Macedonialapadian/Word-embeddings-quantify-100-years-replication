# from gensim.models import KeyedVectors
# import numpy as np

# filename = "vectors/raw/GoogleNews-vectors-negative300.bin"

# model = KeyedVectors.load_word2vec_format(filename, binary=True)

# for i in range(10):
#     word = model.index_to_key[i]
#     vec = model[word]
#     print(i + 1, word, vec)


#filename = "vectors/raw/wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt"
filename = "vectors/raw/glove.42B.300d.txt"

with open(filename, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        if i > 10:
            break
        print(line.rstrip("\n"))
