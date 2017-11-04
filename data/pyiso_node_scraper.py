''' 
Applied Machine Learning Final Project
Pyiso script for gathering node data

'''
import time as timeit
import os
import pandas as pd
pd.set_option('display.max_columns',0)
from CAISOClient import *
import os
path=os.getcwd()

caiso = CAISOClient()

starttime = caiso.utcify('2017-11-01 00:00')
endtime = caiso.utcify('2017-11-01 23:00')
now = caiso.utcify(datetime.utcnow(), tz_name='utc')

options = {
    'start_at': starttime,
    'end_at': endtime,
    'freq': '1hr',
    'market': 'DAM'
    }
    
caiso.options = options

# # Get node locations
# LMP_locs = pd.DataFrame(caiso.get_lmp_loc())
# LMP_locs=LMP_locs[LMP_locs['area']=='CA']
# LMP_locs.to_csv('LMP_locs.csv', sep=',', index=False)
LMP_locs = pd.read_csv('LMP_locs.csv', sep=',')
node_names = LMP_locs['node_id'].unique()

# # Demand Forecast
time1 = timeit.time()

node_id = node_names[100]

lmp = caiso.get_lmp_as_dataframe(node_id, False, starttime, endtime)
lmp.to_csv(path+'/node_LMP/'+ lmp['PNODE_RESMRID'][0]+'.csv', sep=',', index=False)

tot_time = timeit.time() - time1
print("Time taken: {:.2f} seconds".format(tot_time))
