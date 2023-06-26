from PIL import Image
import numpy as np
import csv
import os
from itertools import chain
import random

dir_neg = 'negatives/negatives/'
dir_pos = 'positives/positives/'
negative_features = []
positive_features = []
training_data = []
test_data = []

with open('features.txt', 'w') as f:
    for filename in os.listdir(dir_neg):
        img = Image.open(dir_neg + filename)
        arr = np.array(img)
        mean_val = np.mean(arr.mean(axis=0), axis=0)
        min_val = np.min(arr.min(axis=0), axis=0)
        #negative_features.append(list(chain([-1], mean_val.tolist(), min_val.tolist())))
        training_data.append(list(chain([-1], mean_val.tolist(), min_val.tolist())))


for filename in os.listdir(dir_pos):
    img = Image.open(dir_pos + filename)
    arr_pos = np.array(img)
    mean_val = np.mean(arr_pos.mean(axis=0), axis=0)
    min_val = np.min(arr_pos.min(axis=0), axis=0)
    #positive_features.append(list(chain([1], mean_val.tolist(), min_val.tolist())))
    training_data.append(list(chain([1], mean_val.tolist(), min_val.tolist())))
#print(training_data)

random.shuffle(training_data)
# with open('features.txt', 'w') as f:
#     for filename in os.listdir(dir_neg):
#         img = Image.open(dir_neg + filename)
#         arr = np.array(img)
#         mean_val = np.mean(arr.mean(axis=0), axis=0)
#         min_val = np.min(arr.min(axis=0), axis=0)
#         #negative_features.append(list(chain([-1], mean_val.tolist(), min_val.tolist())))
#         test_data.append(list(chain([-1], mean_val.tolist(), min_val.tolist())))
#
#
# for filename in os.listdir(dir_pos):
#     img = Image.open(dir_pos + filename)
#     arr_pos = np.array(img)
#     mean_val = np.mean(arr_pos.mean(axis=0), axis=0)
#     min_val = np.min(arr_pos.min(axis=0), axis=0)
#     #positive_features.append(list(chain([1], mean_val.tolist(), min_val.tolist())))
#     test_data.append(list(chain([1], mean_val.tolist(), min_val.tolist())))

#print(test_data)

with open('conversion.csv', 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerows(training_data)

# with open('conversion.csv', 'w') as myfile:
#     wr = csv.writer(myfile)
#     wr.writerows(positive_features)

