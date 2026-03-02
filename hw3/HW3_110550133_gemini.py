# import google.generativeai as genai
# import os

# genai.configure(api_key="AIzaSyCFzBezes8c_UAeBn_I7vUdLzZsC1TqE9U")

# model = genai.GenerativeModel(model_name="gemini-1.5-flash")
# response = model.generate_content('{"u": "Could you help me go to [xxx] gas station?", "s": ["[user] is driving.", "[xxx] station is open today.", "[user] knows the friend\'s number.", "[user]\'s car is almost out of gas.", "[user] is in the car.", "Buses are running late.", "[user] is too far from a gas station.", "[user] has money.", "It is 7am now.", "[user] has a membership of a roadside assistance service.", "The weather is sunny.", "[user] has a phone."], "s.type": ["behavior", "environment", "possession", "environment", "location", "environment", "behavior", "possession", "time", "possession", "environment", "possession"], "s.gold.index": [11, 4, 9, 0, 6, 3, 1, 7], "r": "Sorry, but it is raining now. Would you like to know the weather forecast for tomorrow?"} it is the given utterance(u) and situation(s). evaluate the response(r) and give a score from 0 to 1. 0 means the response is bad and 1 means the response is good. give me a simple number 0 or 1. no any other answer.')
# print(response.text)

# response_is_good=response.text[0]
# print(response_is_good)

import google.generativeai as genai
import json
import tqdm
import pandas as pd
import time

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

train_data = load_data("HW3_dataset/train.json")
val_data = load_data("HW3_dataset/val.json")
test_data = load_data("HW3_dataset/test.json")
test_data = load_data("HW3_dataset/test_with_s_gold_index_predict_9.json")

genai.configure(api_key="api_key")
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

t=4
valid=1

cnt=0
for sample in tqdm.tqdm(test_data):
    try:
        utterance = sample['u']
        situation = sample['s']
        response = sample['r']
        gold_index = sample['s.gold.index']
        contents=f'The following are given utterance(u) and situation(s) and response(r). utterance : {utterance}, situation : {situation}, response : {response}, gold.index : {gold_index}. The gold.index in situation represent the relevant situation starting by index 0. The gold.index in situation is estimated and it might be wrong. There is about 1/3 of gold.index is wrong. Evaluate the right gold.index and give me a list of relevant situations(s) each consisting of 6~9 numbers, sorted in descending order of importance. These numbers represent the index in the situation list. Expected output like: [1, 3, 5]. no any other answer.'
        response_text = model.generate_content(contents)
        response_list=json.loads(response_text.text)
        test_predictions=response_list
    except:
        test_predictions=sample['s.gold.index']
        cnt+=1
    sample["s.gold.index"] = test_predictions
    # print('Predicted:', test_predictions)
    time.sleep(t)
    
        
print('Number of failed predictions:', cnt, 'out of', 792)

with open("HW3_dataset/test_with_s_gold_index_cleaned_new.json", 'w') as f:
    json.dump(test_data, f, indent=2)
