import string

text = open("/content/24-01-17/data.txt", "r") 
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

for key in list(d.keys()): 
    print(key, "\t\t : ", d[key]) 
