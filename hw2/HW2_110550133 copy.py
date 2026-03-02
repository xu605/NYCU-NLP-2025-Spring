# compare two csv files with accuracy

import csv

def compare_csv(file1, file2):
    cnt=0
    num=0
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)
        next(reader1)
        next(reader2)
        for row1, row2 in zip(reader1, reader2):
            num+=1
            if row1 != row2:
                cnt+=1
    return 1-cnt/num

file1 = "dataset/0/output_0.5.csv"
file2 = "dataset/15/output_0.5.csv"

print(compare_csv(file1, file2))