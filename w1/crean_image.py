datafilenames="C:\\Users\\thanhdh6\\Documents\\project\\faceX-Zoo\\list.txt"
outfilepath="C:\\Users\\thanhdh6\\Documents\\project\\faceX-Zoo\\list1.txt"
outfile=open(outfilepath,'w')
with open(datafilenames) as f:
    lines=f.readlines()
    for line in lines:
        outfile.write(line[46:].replace('\\','/'))
outfile.close()