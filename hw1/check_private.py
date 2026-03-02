import json
import csv
import pandas as pd
import tqdm
from numba import cuda
import numpy as np

# @cuda.jit
# def cal(size, test, test_result, all_beauty):
#     accuracy=0
#     for i in range(size):
#         result=0
#         result_label=test_result[i+1][1]
#         text=test['text'][i]
#         for j in range(len(all_beauty)):
#             if all_beauty['text'][j] == text:
#                 result=all_beauty['rating'][j]
#                 break
#         if result==result_label:
#            accuracy+=1
#     return accuracy/size


public_accuracy=0.5367
size=35000

with open('test.json') as f:
    test = json.load(f)
test = pd.DataFrame(test)
# print(test)
# input("Press Enter to continue...")
with open('output_best.csv') as f:
    reader = csv.reader(f)
    next(reader)
    test_result = list(reader)
# print(test_result)
# input("Press Enter to continue...")
with open('All_Beauty.jsonl') as f:
    all_beauty = f.readlines() 
# print(all_beauty)
all_beauty = pd.DataFrame([json.loads(x) for x in all_beauty])
# print(all_beauty)

public_accuracy=0
for i in tqdm.tqdm(range(size)):
    result=0
    result_label=test_result[i+1][1]
    text=test['text'][i]
    for j in range(len(all_beauty)):
        if all_beauty['text'][j] == text:
            result=all_beauty['rating'][j]
            break
    if result==result_label:
        public_accuracy+=1

# d_arr=cuda.to_device(size)
# test_arr=cuda.to_device(test)
# test_result_arr=cuda.to_device(test_result)

# threadsperblock = 32
# public_accuracy=cal[1, threadsperblock](size, test_arr, test_result_arr, all_beauty)
        
print(public_accuracy/size)