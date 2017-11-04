'''
Nodes: Exploratory Data

'''
import os
path = os.getcwd()
data_path= path+'/node_LMP/'


def get_csv_names(folder):
    ''' Read all the names of files in a folder and store as list

    Args:
        folder:      str; path to folder of interest

    Returns: 
        csv_names:  list of str; contains file names
    '''

    csv_names_temp = os.listdir(folder)
    csv_names = os.listdir(folder)

    # Remove hidden folders and also any files that are not necessary (lack _)
    for f in csv_names_temp:
       if (f.startswith('.')==True):
           csv_names.remove(f)
       elif '_' not in f:
           csv_names.remove(f)
    return csv_names
    
    

csv_names = get_csv_names(data_path)

import pandas as pd 
pd.set_option('display.max_columns', 0)

finished_LMP_nodes =  open('merged_lmp_nodes.txt', 'r')
finished_LMP_nodes = finished_LMP_nodes.read().split(',')

# Find new csvs to be concatenated
todo_nodes = list(set(csv_names)  - set(finished_LMP_nodes))

# Concat new data to a dataframe
def concat_missing(todo, data_path, save_name):
    '''
    Concats missing csv information to an existing csv
    inputs:
        todo -- list of missing file names
        data_path -- path to missing csv files
        save_name -- name of final CSV
    returns:
        None
    '''
    
    final_df = pd.read_csv(save_name, sep=',')
    for csv in todo:
        tmp = pd.read_csv(data_path+csv, sep=',')
        final_df = pd.concat([final_df, tmp])

        with open('merged_lmp_nodes.txt', 'a') as done_file:
            # record as concatenated
            done_file.write(','+csv)
    
    final_df.to_csv(save_name, sep=',', index=False)

# Concat newly acquired data
# concat_missing(todo_nodes, data_path, 'LMPnodes_all_features.csv')

# Open main file
LMP_nodes = pd.read_csv('LMPnodes_all_features.csv', sep=',')
LMP_nodes.columns =  ['grp', 'grp_type', 'time', 'lmp_type', 'market_id', 
    'price_per_mw', 'node_id', 'node_id2', 'node_id_xml', 'opr_dt', 'opr_hr', 
    'opr_interval', 'pnode_resmrid', 'pos', 'unnamed', 'xml_data_item']

LMP_nodes['year'], LMP_nodes['month'], LMP_nodes['date'] =                  \
    LMP_nodes['opr_dt'].str.split('-', 2).str
    
# add the days of the week 
LMP_nodes['day'] = LMP_nodes['date']
LMP_nodes['day'] = LMP_nodes['day'].replace(['01', '08', '15', '22', '29'], '0')
LMP_nodes['day'] = LMP_nodes['day'].replace(['02', '09', '16', '23', '30'], '1')
LMP_nodes['day'] = LMP_nodes['day'].replace(['03', '10', '17', '24', '31'], '2')
LMP_nodes['day'] = LMP_nodes['day'].replace(['04', '11', '18', '25'], '3')
LMP_nodes['day'] = LMP_nodes['day'].replace(['05', '12', '19', '26'], '4')
LMP_nodes['day'] = LMP_nodes['day'].replace(['06', '13', '20', '27'], '5')
LMP_nodes['day'] = LMP_nodes['day'].replace(['07', '14', '21', '28'], '6')


LMP_nodes = LMP_nodes[['year', 'month', 'date', 'day', 'opr_hr', 'node_id', 
    'market_id', 'price_per_mw']]