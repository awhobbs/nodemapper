#!/usr/bin/env python3
'''
Applied Machine Learning Final Project
Pyiso script for gathering node data

'''
import os
import time
import calendar
import pickle
import datetime
import logging
import pandas as pd
import mycaiso

# set up logging
logging.basicConfig(filename='caiso_parser.log', level=logging.DEBUG)

pd.set_option('display.max_columns', 0)

path = os.getcwd()

caiso_data = mycaiso.CAISOClient()
caiso_data.timeout_seconds = 240

caiso_data.handle_options(freq='1hr', market='DAM')

# # Get node locations
LMP_locs = pd.read_csv('LMP_locs.csv', sep=',')
node_names = list(LMP_locs['node_id'].unique())

# LMP DAMs
num_nodes = len(node_names)
lmpData = []


def get_data(starttime, endtime):
    for node_id in node_names: # Change start index if time out
        logging.info(str(num_nodes-node_names.index(node_id)) + ' nodes left')
        lmp = caiso_data.get_lmp_as_dataframe(node_id, False, starttime, endtime,
                                              freq='1hr',
                                              market='DAM',
                                              market_run_id='DAM')
        lmpData.append(lmp)
        pickle.dump(lmpData, open("save.p", "wb"))
        logging.info('Data saved, waiting 5 seconds.')
        time.sleep(5)

for year in range(2017, 2000, -1):
    if year == 2017:
        start_month = 10
    else:
        start_month = 12
    for month in range(start_month, 0, -1):
        month_end = calendar.monthrange(year, month)[1]
        start = datetime.datetime(year, month, 1, 0).strftime('%Y-%m-%d %H:%M')
        end = datetime.datetime(year, month, month_end, 23).strftime('%Y-%m-%d %H:%M')
        start_c = caiso_data.utcify(start)
        end_c = caiso_data.utcify(end)
        print("Starting ", start, " to ", end)
        get_data(start_c, end_c)
        picklename = start + ".p"
        pickle.dump(lmpData, open( picklename, "wb" ) )
