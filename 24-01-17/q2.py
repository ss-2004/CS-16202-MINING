stpwrd_lst = ["the", "and", "if", "which", "on"]
with open("/content/24-01-17/data.txt", "r") as file:
    txt = file.read()

words = txt.split()
fltr_txt = [word for word in words if word.lower() not in stpwrd_lst]
fltr_txt = ' '.join(fltr_txt)
print(fltr_txt)
