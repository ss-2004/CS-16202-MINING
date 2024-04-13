# Q1 : WAP to obtain a sorted list of words with their counts & make sure that
#     all words are lower-cased & contain only letters from a-z.

import string

text = open("/content/drive/MyDrive/Colab/24-01-17/data.txt", "r")
#text = open("data.txt", "r")
d = dict()

for line in text:
    line = line.strip()
    line = line.lower()
    line = line.translate(line.maketrans("", "", string.punctuation))
    words = line.split(" ")

    for word in words:
        if word in d:
            d[word] = d[word] + 1
        else:
            d[word] = 1

for key in sorted(d.keys()):
    print(key, ":", d[key])
