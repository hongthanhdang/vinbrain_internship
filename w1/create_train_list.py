import os

datafilenames="C:\\Users\\thanhdh6\\Documents\\datasets\\lfw\\files\\peopleDevTrain.txt"
file_name_list="C:\\Users\\thanhdh6\Documents\\project\\faceX-Zoo\data\\files\\MS-Celeb-1M-v1c-r_train_list.txt"
peopleNames=[]
numberOfImage=[]
with open(datafilenames) as f:
    lines=f.readlines()
    for line in lines[1:]:
        # print(line)
        # print("\n")
        name,number=line.split('\t')
        peopleNames.append(name)
        numberOfImage.append(int(number))
print(len(peopleNames))
print(len(numberOfImage))
outFileName="C:\\Users\\thanhdh6\\Documents\\datasets\\lfw\\files\\peopleDevTrain_list.txt"
with open(outFileName,'w') as f:
    for i,name in enumerate(peopleNames):
        for j in range(numberOfImage[i]):
            line="{}/{}_{:04}.jpg {}\n".format(name,name,j+1,i)
            f.write(line)