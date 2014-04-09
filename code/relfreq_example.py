#!/usr/bin/env python

import numpy as np
import numpy.ma as ma
from research.reliability import reliability_diagram as rd
from research.reliability import relia_roc as rr
from research.explore import relative_frequency as rf
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from fortran_routine import read_lsmask_1deg


# --- Get domain indices
minlat = 26
maxlat = 50
minlon = 254
maxlon = 290


# --- Now, let's get the masks out...
lts=np.arange(minlat,maxlat+1,1)
lns=np.arange(minlon,maxlon+1,1)
lats = np.arange(10,61,1)
lons = np.arange(220,301,1)
lats_mask = np.in1d(lats,lts)
lons_mask = np.in1d(lons,lns)

# --- Not too interested in forecasts over the ocean or in Canada/Mexico, so...
nx = 360
ny = 181
lsmask, rlons_mask, rlats_mask = read_lsmask_1deg(nx,ny)
lsmask_t = np.transpose(lsmask)
latindex = np.where(rlats_mask[0,:]==minlat)
lonindex = np.where(rlons_mask[:,0]==minlon)
lsmask_t = lsmask_t[int(latindex[0]):int(latindex[0]+(((maxlat)-(minlat))+1)),int(lonindex[0]):int(lonindex[0]+(((maxlon)-(minlon))+1))]

predictor_nc = Dataset('refcst2_01km_bulk_shear_day1.nc','r')
predictand_nc = Dataset('tor_day_data_1985-2011_day1.nc','r')

raw_predictor = predictor_nc.variables['01_shear'][:,lats_mask,lons_mask]
raw_predictand = predictand_nc.variables['Tornado_reports'][:,0,lats_mask,lons_mask]

predictor = ma.zeros((raw_predictor.shape))
predictand = ma.zeros((raw_predictor.shape))
for i in xrange(predictor.shape[0]):
    predictor[i,:,:] = ma.array(raw_predictor[i,:,:],mask=1-lsmask_t)
    predictand[i,:,:] = ma.array(raw_predictand[i,:,:],mask=1-lsmask_t)      
predictor = predictor.reshape(-1).compressed()
predictand = predictand.reshape(-1).compressed()

fig = plt.figure(figsize=(13.5,13.5))
rf(fig,predictor,predictand,'0-1 km Shear')
plt.show()
plt.close()
