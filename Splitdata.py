import numpy as np
import os
import splitfolders
from prettytable import PrettyTable

input_folder = "E:/dataset MIDV HOLO/Mosaics/"
output_folder = "E:/dataset MIDV HOLO/Mosaics V2 splited/"

#splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.8, 0.1, 0.1))

# statestics after spliting the dataset
# stats train
num_train_Holo = len(os.listdir('E:/dataset MIDV HOLO/Mosaics V2 splited/train/Holo'))
num_train_No_Holo = len(os.listdir('E:/dataset MIDV HOLO/Mosaics V2 splited/train/No-Holo'))
TOT_train = num_train_Holo + num_train_No_Holo

# stats test
num_test_Holo = len(os.listdir('E:/dataset MIDV HOLO/Mosaics V2 splited/test/Holo'))
num_test_No_Holo = len(os.listdir('E:/dataset MIDV HOLO/Mosaics V2 splited/test/No-Holo'))
TOT_test = num_test_No_Holo + num_test_Holo
# stats valdidation
num_valid_Holo = len(os.listdir('E:/dataset MIDV HOLO/Mosaics V2 splited/val/Holo'))
num_valid_No_Holo = len(os.listdir('E:/dataset MIDV HOLO/Mosaics V2 splited/val/No-Holo'))
TOT_valid = num_valid_Holo + num_valid_No_Holo

TOT_Holo = num_valid_Holo + num_test_Holo + num_train_Holo
TOT_No_Holo = num_train_No_Holo + num_test_No_Holo + num_valid_No_Holo
TOT_all = TOT_Holo + TOT_No_Holo

# table stat general
t = PrettyTable(['Class', 'Train', 'Validation', 'Test', 'Total'])
t.add_row(['Holo', num_train_Holo, num_valid_Holo, num_test_Holo, TOT_Holo])
t.add_row(['No-Holo', num_train_No_Holo, num_valid_No_Holo, num_test_No_Holo, TOT_No_Holo])
t.add_row(['Total', TOT_train, TOT_valid, TOT_test, TOT_all])
print(t)
