######

import xarray as xr
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cmocean as cmo


######
#BRAN data

#target data
tar = xr.open_dataset('./bran_sst_historic.nc')
sst_tar = tar.SST
lon_sst = tar.yt_ocean.data
lat_sst = tar.xt_ocean.data
bran_time = tar.time.data

bran_sst_mn = sst_tar.mean('time')
bran_spat_tar_mn = sst_tar.mean(('xt_ocean','yt_ocean'))

#99 percentile
perc_tar_bran = np.percentile(sst_tar,99,axis=0)

ds_past_bran = xr.Dataset( 
    {"SST": (("lat_sst", "lon_sst"), perc_tar_bran),},
    coords={ "lon_sst": lon_sst,
            "lat_sst": lat_sst,})

#prediction
pred = xr.open_dataset('./bran_sst_prediction.nc')
sst_pred = pred.SST

bran_pred_mn = sst_pred.mean('time')
bran_spat_pred_mn = sst_pred.mean(('xt_ocean','yt_ocean'))

#99 percentile
perc_pred_bran = np.percentile(sst_pred,99,axis=0)

ds_pred_bran = xr.Dataset( 
    {"SST": (("lat_sst", "lon_sst"), perc_pred_bran),},
    coords={ "lon_sst": lon_sst,
            "lat_sst": lat_sst,})


###interpolation 
bran_interp = xr.open_dataset('./bran_sst_interpolation.nc')
sst_interp = bran_interp.SST

bran_int_mn = sst_interp.mean('time')
bran_spat_int_mn =sst_interp.mean(('lon_sst','lat_sst'))

#99 percentile
perc_interp_bran = np.percentile(sst_interp,99,axis=0)

ds_interp_bran = xr.Dataset( 
    {"SST": (("lat_sst", "lon_sst"), perc_interp_bran),},
    coords={ "lon_sst": lon_sst,
            "lat_sst": lat_sst,})


#######NOAA

#target data
tar = xr.open_dataset('./noaa_sst_historic.nc')
sst_tar = tar.SST
lon_sst = tar.lon.data
lat_sst = tar.lat.data

noaa_sst_mn = sst_tar.mean('time')
noaa_tar_spat_mn = sst_tar.mean(('lat','lon'))

#99 percentile
perc_past_noaa = np.percentile(sst_tar,99,axis=0)

ds_past_noaa = xr.Dataset( 
    {"SST": (("lat_sst", "lon_sst"), perc_past_noaa),},
    coords={ "lon_sst": lon_sst,
            "lat_sst": lat_sst,})

#prediction

pred = xr.open_dataset('./noaa_sst_prediction.nc')
sst_pred = pred.SST

noaa_pred_mn = sst_pred.mean('time')
noaa_pred_spat_mn = sst_pred.mean(('lon','lat'))

#99 percentile
perc_pred_noaa = np.percentile(sst_pred,99,axis=0)

ds_pred_noaa = xr.Dataset( 
    {"SST": (("lat_sst", "lon_sst"), perc_pred_noaa),},
    coords={ "lon_sst": lon_sst,
            "lat_sst": lat_sst,})

###interpolation 
noaa_interp = xr.open_dataset('./noaa_sst_interpolation.nc')
sst_interp = noaa_interp.SST

noaa_int_mn = sst_interp.mean('time')
noaa_int_spat_mn =sst_interp.mean(('lon_sst','lat_sst'))

#99 percentile
perc_interp_noaa = np.percentile(sst_interp,99,axis=0)

ds_interp_noaa = xr.Dataset( 
    {"SST": (("lat_sst", "lon_sst"), perc_interp_noaa),},
    coords={ "lon_sst": lon_sst,
            "lat_sst": lat_sst,})



####plot the data

##RMSE and correlation
def rmse(predictions, targets):
    return np.sqrt(np.mean(((predictions - targets) ** 2)))
####

import matplotlib.gridspec as gridspec

fig = plt.figure(tight_layout=True,figsize=(7,10))
gs = gridspec.GridSpec(4, 4)

ax = plt.subplot(gs[0, :])
bran_spat_tar_mn.plot(color = 'k',marker='o',markersize=0.5,label ='Target')
bran_spat_pred_mn.plot(color = 'r',marker='o',markersize=0.5, label = 'ML downscaling')
bran_spat_int_mn.plot(color = 'g',marker='o',markersize=0.5, label = 'Interpolation')
plt.legend(loc = 'lower right')

plt.axhline(y=perc_tar_bran.mean(), color='grey', linestyle='-.')
plt.text(bran_time[0],29.5,"(a)",fontsize=10)
ax.set_ylabel('SST [$^\circ$C]')
ax.set_xlabel('')
ax.set_xlim(['1993-01-01', '2022-12-30'])
#plot the extreme event summer (Dec-Feb) periods
ax.axvspan('1997-12-01', '1998-02-28', alpha=0.4, color='cyan')
ax.axvspan('2001-12-01', '2002-02-28', alpha=0.4, color='cyan')
ax.axvspan('2015-12-01', '2016-02-28', alpha=0.4, color='cyan')
ax.axvspan('2016-12-01', '2017-02-28', alpha=0.4, color='cyan')
ax.axvspan('2019-12-01', '2020-02-28', alpha=0.4, color='cyan')
ax.axvspan('2021-12-01', '2022-02-28', alpha=0.4, color='cyan')

plt.title('BRAN2020',fontsize=12, fontweight='bold')

##################
ax1 = plt.subplot(gs[1, 0])
m = Basemap(llcrnrlon=142.,llcrnrlat=-25.,urcrnrlon=154.,urcrnrlat=-10.,
    resolution='l',projection='merc',lat_ts=-15,lat_0=lat_sst.mean(),lon_0=lon_sst.mean(),ax=ax1)

#create grid to plot
lonx,latx = np.meshgrid(lon_sst,lat_sst)
xi,yi=m(lonx,latx)

ax1.set_title("Target")
cmap=mpl.cm.Spectral_r
bounds = list(np.arange(28.5,32.5,0.1))
norm =  mpl.colors.BoundaryNorm(bounds, ncolors=cmap.N, extend = "both")

cs = m.pcolor(xi,yi,ds_tar_bran.SST.data,cmap=cmap,norm=norm)
# Add Grid Lines
m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=7,color ="grey")
m.drawmeridians(np.arange(142., 194., 10.), labels=[0,0,0,1], fontsize=7,color ="grey")
m.drawcoastlines()
m.fillcontinents(color='grey')

plt.annotate("(b)", m(143.4,-20), fontsize=10)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.9, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]
x, y = m(lons, lats)
m.plot(x, y, color='k', linewidth=1.5, linestyle = 'dotted')

##################
ax2 = plt.subplot(gs[1, 1])
m = Basemap(llcrnrlon=142.,llcrnrlat=-25.,urcrnrlon=154.,urcrnrlat=-10.,
    resolution='l',projection='merc',lat_ts=-15,lat_0=lat_sst.mean(),lon_0=lon_sst.mean(),ax=ax2)

#create grid to plot
lonx,latx = np.meshgrid(lon_sst,lat_sst)
xi,yi=m(lonx,latx)

ax2.set_title("ML downscaling")
cs = m.pcolor(xi,yi,ds_pred_bran.SST.data,cmap=cmap,norm=norm)
# Add Grid Lines
m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=7,color ="grey")
m.drawmeridians(np.arange(142., 194., 10.), labels=[0,0,0,1], fontsize=7,color ="grey")
m.drawcoastlines()
m.fillcontinents(color='grey')

plt.annotate("(c)", m(143.4,-20), fontsize=10)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.9, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]
x, y = m(lons, lats)
m.plot(x, y, color='k', linewidth=1.5, linestyle = 'dotted')

# add text box for the statistics
stats1 = (f'rmse = {rmse(perc_pred_bran,perc_tar_bran):.2f}\n')
stats2 = (f'r = {np.corrcoef(perc_pred_bran,perc_tar_bran)[0][1]:.2f}\n')
ax2.text(0.95, 0.78, stats1, fontsize=9, color = 'black', fontweight = 'bold',
        transform=ax2.transAxes, horizontalalignment='right')
ax2.text(0.95, 0.70, stats2, fontsize=9, color = 'black', fontweight = 'bold',
        transform=ax2.transAxes, horizontalalignment='right')

#####################

ax2 = plt.subplot(gs[1, 2])
m = Basemap(llcrnrlon=142.,llcrnrlat=-25.,urcrnrlon=154.,urcrnrlat=-10.,
    resolution='l',projection='merc',lat_ts=-15,lat_0=lat_sst.mean(),lon_0=lon_sst.mean(),ax=ax2)

#create grid to plot
lonx,latx = np.meshgrid(lon_sst,lat_sst)
xi,yi=m(lonx,latx)

ax2.set_title("Interpolation")
cs = m.pcolor(xi,yi,ds_interp_bran.SST.data,cmap=cmap,norm=norm)
# Add Grid Lines
m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=7,color ="grey")
m.drawmeridians(np.arange(142., 194., 10.), labels=[0,0,0,1], fontsize=7,color ="grey")
m.drawcoastlines()
m.fillcontinents(color='grey')

plt.annotate("(d)", m(143.4,-20), fontsize=10)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.9, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]
x, y = m(lons, lats)
m.plot(x, y, color='k', linewidth=1.5, linestyle = 'dotted')

cax = plt.axes((0.15, 0.51, 0.52, 0.01))
fig.colorbar(cs, cax=cax,label='SST [$^\circ$C]',orientation='horizontal')

# add text box for the statistics
stats1 = (f'rmse = {rmse(perc_interp_bran,perc_tar_bran):.2f}\n')
stats2 = (f'r = {np.corrcoef(perc_interp_bran,perc_tar_bran)[0][1]:.2f}\n')
ax2.text(0.95, 0.78, stats1, fontsize=9, color = 'black', fontweight = 'bold',
        transform=ax2.transAxes, horizontalalignment='right')
ax2.text(0.95, 0.70, stats2, fontsize=9, color = 'black', fontweight = 'bold',
        transform=ax2.transAxes, horizontalalignment='right')

#################

#difference
ax3 = plt.subplot(gs[1, 3])

m = Basemap(llcrnrlon=142.,llcrnrlat=-25.,urcrnrlon=154.,urcrnrlat=-10.,
    resolution='l',projection='merc',lat_ts=-15,lat_0=lat_sst.mean(),lon_0=lon_sst.mean(),ax=ax3)

#create grid to plot
lonx,latx = np.meshgrid(lon_sst,lat_sst)
xi,yi=m(lonx,latx)

ax3.set_title("Difference")
cmap=mpl.cm.PiYG_r
bounds = list(np.arange(-0.8,0.8,0.05))
norm =  mpl.colors.BoundaryNorm(bounds, ncolors=cmap.N, extend = "both")

cs = m.pcolor(xi,yi,ds_pred_bra.SST.data-ds_past.SST.data,cmap=cmap,\
             vmin=-0.5,vmax=0.5)
# Add Grid Lines
m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=7,color ="grey")
m.drawmeridians(np.arange(142., 194., 10.), labels=[0,0,0,1], fontsize=7,color ="grey")
m.drawcoastlines()
m.fillcontinents(color='grey')

plt.annotate("(e)", m(143.4,-20), fontsize=10)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.9, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]
x, y = m(lons, lats)
m.plot(x, y, color='k', linewidth=1.5, linestyle = 'dotted')

cax = plt.axes((0.995, 0.54, 0.015, 0.15))
fig.colorbar(cs, cax=cax,label='Target - ML downscaling [$^\circ$C]',extend='both')


#####################NOAA

ax = plt.subplot(gs[2, :])
noaa_past_mn.plot(color = 'k',marker='o',markersize=0.5,label ='Target')
noaa_pred_mn.plot(color = 'm',marker='o',markersize=0.5, label = 'ML downscaling')
ss_interp_noaa.plot(color = 'b',marker='o',markersize=0.5, label = 'Interpolation')
plt.legend(loc = 'lower right')

plt.axhline(y=perc_noaa.mean(), color='grey', linestyle='-.')
plt.text(noaa_time[0],28.5,"(f)",fontsize=10)
ax.set_ylabel('SST [$^\circ$C]')
ax.set_xlabel('')
ax.set_xlim(['1985-01-01', '2020-12-30'])
#plot the extreme event summer (Dec-Feb) periods
ax.axvspan('1997-12-01', '1998-02-28', alpha=0.4, color='cyan')
ax.axvspan('2001-12-01', '2002-02-28', alpha=0.4, color='cyan')
ax.axvspan('2015-12-01', '2016-02-28', alpha=0.4, color='cyan')
ax.axvspan('2016-12-01', '2017-02-28', alpha=0.4, color='cyan')
ax.axvspan('2019-12-01', '2020-02-28', alpha=0.4, color='cyan')

plt.title('NOAA',fontsize=12, fontweight='bold')

##################

ax1 = plt.subplot(gs[3, 0])
m = Basemap(llcrnrlon=142.,llcrnrlat=-25.,urcrnrlon=154.,urcrnrlat=-10.,
    resolution='l',projection='merc',lat_ts=-15,lat_0=lat_sst.mean(),lon_0=lon_sst.mean(),ax=ax1)

#create grid to plot
lonx,latx = np.meshgrid(lon_sst,lat_sst)
xi,yi=m(lonx,latx)

ax1.set_title("Target")
cmap=mpl.cm.Spectral_r
bounds = list(np.arange(28.5,32.5,0.1))
norm =  mpl.colors.BoundaryNorm(bounds, ncolors=cmap.N, extend = "both")

cs = m.pcolor(xi,yi,ds_noaa.SST.data,cmap=cmap,norm=norm)
# Add Grid Lines
m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=7,color ="grey")
m.drawmeridians(np.arange(142., 194., 10.), labels=[0,0,0,1], fontsize=7,color ="grey")
m.drawcoastlines()
m.fillcontinents(color='grey')

plt.annotate("(g)", m(143.4,-20), fontsize=10)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.9, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]
x, y = m(lons, lats)
m.plot(x, y, color='k', linewidth=1.5, linestyle = 'dotted')

##############

ax2 = plt.subplot(gs[3, 1])
m = Basemap(llcrnrlon=142.,llcrnrlat=-25.,urcrnrlon=154.,urcrnrlat=-10.,
    resolution='l',projection='merc',lat_ts=-15,lat_0=lat_sst.mean(),lon_0=lon_sst.mean(),ax=ax2)

#create grid to plot
lonx,latx = np.meshgrid(lon_sst,lat_sst)
xi,yi=m(lonx,latx)

ax2.set_title("ML downscaling")
cs = m.pcolor(xi,yi,ds_noaa_p.SST.data,cmap=cmap,norm=norm)
# Add Grid Lines
m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=7,color ="grey")
m.drawmeridians(np.arange(142., 194., 10.), labels=[0,0,0,1], fontsize=7,color ="grey")
m.drawcoastlines()
m.fillcontinents(color='grey')

plt.annotate("(h)", m(143.4,-20), fontsize=10)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.9, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]
x, y = m(lons, lats)
m.plot(x, y, color='k', linewidth=1.5, linestyle = 'dotted')

# add text box for the statistics
stats1 = (f'rmse = {rmse(perc_noaa_p,perc_noaa):.2f}\n')
stats2 = (f'r = {np.corrcoef(perc_noaa_p,perc_noaa)[0][1]:.2f}\n')
ax2.text(0.95, 0.78, stats1, fontsize=9, color = 'black', fontweight = 'bold',
        transform=ax2.transAxes, horizontalalignment='right')
ax2.text(0.95, 0.70, stats2, fontsize=9, color = 'black', fontweight = 'bold',
        transform=ax2.transAxes, horizontalalignment='right')

###################

ax2 = plt.subplot(gs[3, 2])
m = Basemap(llcrnrlon=142.,llcrnrlat=-25.,urcrnrlon=154.,urcrnrlat=-10.,
    resolution='l',projection='merc',lat_ts=-15,lat_0=lat_sst.mean(),lon_0=lon_sst.mean(),ax=ax2)

#create grid to plot
lonx,latx = np.meshgrid(lon_sst,lat_sst)
xi,yi=m(lonx,latx)

ax2.set_title("Interpolation")
cs = m.pcolor(xi,yi,ds_interp_noaa.SST.data,cmap=cmap,norm=norm)
# Add Grid Lines
m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=7,color ="grey")
m.drawmeridians(np.arange(142., 194., 10.), labels=[0,0,0,1], fontsize=7,color ="grey")
m.drawcoastlines()
m.fillcontinents(color='grey')

plt.annotate("(i)", m(143.4,-20), fontsize=10)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.9, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]
x, y = m(lons, lats)
m.plot(x, y, color='k', linewidth=1.5, linestyle = 'dotted')

# add text box for the statistics
stats1 = (f'rmse = {rmse(perc_interp_noaa,perc_noaa):.2f}\n')
stats2 = (f'r = {np.corrcoef(perc_interp_noaa,perc_noaa)[0][1]:.2f}\n')
ax2.text(0.95, 0.78, stats1, fontsize=9, color = 'black', fontweight = 'bold',
        transform=ax2.transAxes, horizontalalignment='right')
ax2.text(0.95, 0.70, stats2, fontsize=9, color = 'black', fontweight = 'bold',
        transform=ax2.transAxes, horizontalalignment='right')

cax = plt.axes((0.15, 0.0004, 0.54, 0.01))
fig.colorbar(cs, cax=cax,label='SST [$^\circ$C]',orientation='horizontal')

###################

#difference
ax3 = plt.subplot(gs[3, 3])

m = Basemap(llcrnrlon=142.,llcrnrlat=-25.,urcrnrlon=154.,urcrnrlat=-10.,
    resolution='l',projection='merc',lat_ts=-15,lat_0=lat_sst.mean(),lon_0=lon_sst.mean(),ax=ax3)

#create grid to plot
lonx,latx = np.meshgrid(lon_sst,lat_sst)
xi,yi=m(lonx,latx)

ax3.set_title("Difference")
cmap=mpl.cm.PiYG_r
bounds = list(np.arange(-0.8,0.8,0.05))
norm =  mpl.colors.BoundaryNorm(bounds, ncolors=cmap.N, extend = "both")

cs = m.pcolor(xi,yi,ds_noaa.SST.data-ds_noaa_p.SST.data,cmap=cmap,\
             vmin=-0.5,vmax=0.5)
# Add Grid Lines
m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=7,color ="grey")
m.drawmeridians(np.arange(142., 194., 10.), labels=[0,0,0,1], fontsize=7,color ="grey")
m.drawcoastlines()
m.fillcontinents(color='grey')

plt.annotate("(j)", m(143.4,-20), fontsize=10)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.9, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]
x, y = m(lons, lats)
m.plot(x, y, color='k', linewidth=1.5, linestyle = 'dotted')

cax = plt.axes((0.995, 0.03, 0.015, 0.15))
fig.colorbar(cs, cax=cax,label='Target - ML downscaling [$^\circ$C]',extend='both')


plt.show()
