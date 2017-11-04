''' 
Applied Machine Learning Final Project
Pyiso script for gathering node data

'''
import time as timeit
import os
import numpy as np
import pandas as pd
pd.set_option('display.max_columns',0)
# from CAISOClient import *
from pyiso import caiso
from datetime import datetime, timedelta, time
import os

path=os.getcwd()

caiso_data = caiso.CAISOClient()
caiso_data.timeout_seconds = 120

starttime = caiso_data.utcify('2017-10-01 00:00')
endtime = caiso_data.utcify('2017-10-01 23:00')
now = caiso_data.utcify(datetime.utcnow(), tz_name='utc')

caiso_data.handle_options(freq='1hr', market='DAM')

# # Get node locations
# LMP_locs = pd.DataFrame(caiso.get_lmp_loc())
# LMP_locs=LMP_locs[LMP_locs['area']=='CA']
# LMP_locs.to_csv('LMP_locs.csv', sep=',', index=False)
LMP_locs = pd.read_csv('LMP_locs.csv', sep=',')
node_names = list(LMP_locs['node_id'].unique())

# Fuel Types
fuel = pd.DataFrame(caiso_data._generation_historical())
fuel.to_csv('fuel.csv', sep=',', index=False)


# time1 = timeit.time()
num_nodes = len(node_names)
# # LMP DAMs
for node_id in node_names:
    print(num_nodes-node_names.index(node_id), 'nodes left')
    lmp = caiso_data.get_lmp_as_dataframe(node_id, False, starttime, endtime, freq='1hr', market='DAM', market_run_id='DAM')
    lmp.to_csv(path+'/node_LMP/'+ lmp['PNODE_RESMRID'][0]+'.csv', sep=',', index=False)

# tot_time = timeit.time() - time1
# print("Time taken: {:.2f} seconds".format(tot_time))
