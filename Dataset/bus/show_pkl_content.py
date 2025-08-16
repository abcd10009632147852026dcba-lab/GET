# import pickle

# #dataset_path = 'bus_train_test_names.pkl'
# #dataset_path = 'bus2busi_train_test_names.pkl'

# with open(dataset_path, 'rb') as file:
#     loaded_dict = pickle.load(file)

# train_name_list = loaded_dict['train']['name_list']
# test_name_list  = loaded_dict['test']['name_list']

# print('train num: {}'.format(len(train_name_list)))
# print('test num: {}'.format(len(test_name_list)))
# print('all num: {}'.format(len(train_name_list) + len(test_name_list)))


import pickle
import os

dataset_paths = [
    'bus_train_test_names.pkl',
    'bus2busi_train_test_names.pkl'
]

for pkl_path in dataset_paths:
    # open expects one filepath at a time
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    train_list = data['train']['name_list']
    test_list  = data['test']['name_list']

    print(f"--- {os.path.basename(pkl_path)} ---")
    print(f" train num: {len(train_list)}")
    print(f" test num : {len(test_list)}")
    print(f" all num  : {len(train_list) + len(test_list)}\n")

