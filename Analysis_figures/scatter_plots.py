import xarray as xr
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cmocean as cmo


tar = xr.open_dataset('./ofam3_sst_historic.nc')
sst_tar = tar.SST

clim_tar = sst_tar.resample(time="1M").mean().groupby("time.month").mean("time")
anom_tar = sst_tar.resample(time="1M").mean().groupby("time.month") - clim_past

#prediction
pred = xr.open_dataset('./ofam3_sst_prediction.nc')
sst_pred = pred.SST

clim_pred = sst_pred.resample(time="1M").mean().groupby("time.month").mean("time")
anom_pred = sst_pred.resample(time="1M").mean().groupby("time.month") - clim_pred

####choose locations along GBR

#nGBR
apa1 = anom_tar.sel(lon_sst = 145,lat_sst=-14,method ="nearest")
apa1_mn=apa1.groupby("time.month")

apr1 = anom_pred.sel(lon_sst = 145,lat_sst=-14,method ="nearest")
apr1_mn = apr1.groupby("time.month")

#sGBR
apa3 = anom_tar.sel(lon_sst = 151,lat_sst=-22,method ="nearest")
apa3_mn=apa3.groupby("time.month")

apr3 = anom_pred.sel(lon_sst = 151,lat_sst=-22,method ="nearest")
apr3_mn = apr3.groupby("time.month")

#cGBR
apa2 = anom_tar.sel(lon_sst = 147.5,lat_sst=-18.5,method ="nearest")#.resample(time="1M").mean()
apa2_mn=apa2.groupby("time.month")#.mean('time')

apr2 = anom_pred.sel(lon_sst = 147.5,lat_sst=-18.5,method ="nearest")#.resample(time="1M").mean()
apr2_mn = apr2.groupby("time.month")

## Do plot
#loop to plot truth, prediction

month = ['JAN', 'FEB', 'MAR', 'APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']

fig, axs = plt.subplots(4, 3, figsize=(12, 12), layout='constrained',
                        sharex=True, sharey=True)
ax=axs.ravel()

for i, month in enumerate(('JAN', 'FEB', 'MAR', 'APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC')):
    ax[i].scatter(apa1_mn[i+1],apr1_mn[i+1], c = 'r', s = 61,alpha=0.3,label = "nGBR")
    ax[i].scatter(apa2_mn[i+1],apr2_mn[i+1], c = 'b', s = 61,alpha=0.3,label = "cGBR")
    ax[i].scatter(apa3_mn[i+1],apr3_mn[i+1], c = 'g', s = 61,alpha=0.3,label = "sGBR")
    
    #find line of best fit
    a, b = np.polyfit(apa1_mn[i+1],apr1_mn[i+1], 1)
    #add line of best fit to plot
    ax[i].plot(apa1_mn[i+1], a*apa1_mn[i+1]+b, color="r", linestyle = "dashed")

    #find line of best fit
    a, b = np.polyfit(apa2_mn[i+1],apr2_mn[i+1], 1)
    #add line of best fit to plot
    ax[i].plot(apa2_mn[i+1], a*apa2_mn[i+1]+b, color="b", linestyle = "dashed")

    #find line of best fit
    a, b = np.polyfit(apa3_mn[i+1],apr3_mn[i+1], 1)
    #add line of best fit to plot
    ax[i].plot(apa3_mn[i+1], a*apa3_mn[i+1]+b, color="g", linestyle = "dashed")

    ax[i].set_title(month.upper(), fontsize =14, fontweight="bold")
    ax[i].legend(loc='upper left')
    ax[i].set_xlim([-2.5,2.5])
    ax[i].set_ylim([-2.5,2.5])
    
    # add text box for the statistics
    stats1 = (f'$r$ = {np.corrcoef(apa1_mn[i+1],apr1_mn[i+1])[0][1]:.2f}\n')
    stats2 = (f'$r$ = {np.corrcoef(apa2_mn[i+1],apr2_mn[i+1])[0][1]:.2f}\n')
    stats3 = (f'$r$ = {np.corrcoef(apa3_mn[i+1],apr3_mn[i+1])[0][1]:.2f}\n')
    # bbox = dict(boxstyle='round',fc='blanchedalmond', ec='orange', alpha=0.5)
    ax[i].text(0.95, 0.07, stats1, fontsize=9, color = 'red',
            transform=ax[i].transAxes, horizontalalignment='right')
    ax[i].text(0.95, 0.13, stats2, fontsize=9, color = 'blue',
            transform=ax[i].transAxes, horizontalalignment='right')
    ax[i].text(0.95, 0.19, stats3, fontsize=9, color = 'green',
            transform=ax[i].transAxes, horizontalalignment='right')
    
fig.supxlabel('Target',fontsize =14, fontweight="bold")
fig.supylabel('Downscaling',fontsize =14, fontweight="bold")

plt.show()
