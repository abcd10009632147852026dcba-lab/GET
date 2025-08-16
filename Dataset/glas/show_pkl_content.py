import pickle

dataset_path = 'glas_train_test_names.pkl'

with open(dataset_path, 'rb') as file:
    loaded_dict = pickle.load(file)

train_name_list = loaded_dict['train']['name_list']
test_name_list  = loaded_dict['test']['name_list']

print('train num: {}'.format(len(train_name_list)))
print('test num: {}'.format(len(test_name_list)))
print('all num: {}'.format(len(train_name_list) + len(test_name_list)))
