import pandas as pd
import os
pd.set_option('display.max_columns',0)

path = os.getcwd() 

# Open CSV concatenating 6 different networks from May 2017-Aug 2017
# Includes duplicate parent nodes. Remove olders, keep updated
all_nodes_w_dup = pd.read_csv(path+'/all_nodes_w_duplicates.csv', sep=',')
all_nodes_w_dup.sort_values(['PARENT_ID', 'Number'], inplace=True)
all_nodes_w_dup.drop_duplicates(subset=['PARENT_ID', 'CHILD_ID'], keep='first', inplace = True)
all_nodes_w_dup.to_csv(path+'/all_nodes.csv', sep=',', index=False)

import geopandas as gpd
import matplotlib as plt

power_lines = gpd.read_file(path+'/Electric_Power_Transmission_Lines/Electric_Power_Transmission_Lines.shp')
power_lines.plot()
plt.show()