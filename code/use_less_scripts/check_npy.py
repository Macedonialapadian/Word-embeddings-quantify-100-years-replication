import numpy as np

filename = "vectors/raw/sgns/1910-w.npy"

arr = np.load(filename)

print("Array shape:", arr.shape)



for i, vec in enumerate(arr):
    print(f"Vector {i}: {vec}")

    if i >= 10:
        break




# the code below is for counting the number for total vectors and non-zero vectors.

# j=0
# n=0

# for i, vec in enumerate(arr):
#     if vec[1]!=0:
#         j+=1
#     n+=1
# print(j,n)