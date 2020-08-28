import numpy as np


with open("train_all.txt", "r", encoding="utf8") as f:
    data = f.readlines()

length = []
for line in data:
    text = line.strip().split("\t")
    text_a = text[1]
    text_b = text[2]
    length.append(len(text_a + text_b))

length1 = np.array(length)
mean = np.mean(length1)
max_v = np.max(length1)
min_v = np.min(length1)
print(max_v, min_v, mean)

max_i = np.where(length1 == np.max(length1))
print(data[length.index(166)])
print(data[length.index(10)])