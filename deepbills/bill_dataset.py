import pandas as pd
import os
from datasets import load_dataset
from pathlib import Path

def load_bill_dataset(problem_name = 'multiclass'):
    
    unique_name = f'{problem_name}_{position}'
    Path(f"./data/").mkdir(parents=True, exist_ok=True)
    path_name = f"./data/{unique_name}"
    print(path_name)
    
    file_path = os.getcwd() + '/data'
    if problem_name == 'multiclass':
        train_file = os.path.join(file_path, 'df_multi_train.csv')
        test_file  = os.path.join(file_path, 'df_multi_test.csv')
        train_df = pd.read_csv(train_file)
        test_df  = pd.read_csv(test_file)
        
    else:
        train_file = os.path.join(file_path, 'df_binary_train.csv')
        test_file  = os.path.join(file_path, 'df_binary_test.csv')
        train_df = pd.read_csv(train_file)
        test_df  = pd.read_csv(test_file)
    

    train_df.to_csv(f'{path_name}_train.csv', index = False)
    test_df.to_csv(f'{path_name}_test.csv', index = False)
    
    print(f'{path_name}_test.csv')
    
    dataset = load_dataset('csv', data_files = {'train': f'{path_name}_train.csv', 
                                                'test' : f'{path_name}_test.csv'})
    dataset = dataset.remove_columns(['bill_year', 'title', 'type', 'source', 'key'])
    dataset = dataset.rename_column('status', 'labels')
    dataset = dataset.rename_column('bill_content', 'Text')
    
    print(dataset)
    return dataset