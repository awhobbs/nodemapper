''' 
Applied Machine Learning Final Project
Pyiso script for gathering node data

'''


from pyiso import client_factory
from CAISOClient import *

caiso = CAISOClient()

# Get node locations
LMP_locs = pd.DataFrame(caiso.get_lmp_loc())
LMP_locs=LMP_locs[LMP_locs['area']=='CA']
LMP_locs.to_csv('LMP_locs.csv', sep=',', index=False)

