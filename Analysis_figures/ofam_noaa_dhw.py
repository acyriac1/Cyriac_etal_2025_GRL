import xarray as xr
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cmocean as cmo


##-----load the data: OFAM
tar = xr.open_dataset('./ofam3_dhw_historic.nc')
ofam_dhw_tar = tar.DHW.sel(time = slice("2002-03-01","2002-03-31"))
lon_sst = tar.longitude.data
lat_sst = tar.latitude.data

pred = xr.open_dataset('./ofam3_dhw_prediction.nc')
ofam_dhw_pred = pred.DHW.sel(time = slice("2002-03-01","2002-03-31"))


inter = xr.open_dataset('./ofam3_dhw_interpolation.nc')
ofam_dhw_interp = inter.DHW.sel(time = slice("2002-03-01","2002-03-31"))

#Difference
ofam_dhw_diff_pr = ofam_dhw_pred - ofam_dhw_tar
ofam_dhw_diff_interp = ofam_dhw_interp - ofam_dhw_tar


truth_mn_o = ofam_dhw_tar.mean(dim='time')
pred_mn_o = ofam_dhw_pred.mean(dim='time')
interp_mn_o = ofam_dhw_interp.mean(dim='time')

pred_diff_mn_o = ofam_dhw_diff_pr.mean(dim='time')
interp_diff_mn_o = ofam_dhw_diff_interp.mean(dim='time')


##-----load the data:NOAA
del tar, pred, inter
tar = xr.open_dataset('./noaa_dhw_historic.nc')
noaa_dhw_tar = tar.DHW.sel(time = slice("2002-03-01","2002-03-31"))

pred = xr.open_dataset('./noaa_dhw_prediction.nc')
noaa_dhw_pred = pred.DHW.sel(time = slice("2002-03-01","2002-03-31"))


inter = xr.open_dataset('./noaa_dhw_interpolation.nc')
noaa_dhw_interp = inter.DHW.sel(time = slice("2002-03-01","2002-03-31"))

#Difference
noaa_dhw_diff_pr = noaa_dhw_pred - noaa_dhw_tar
noaa_dhw_diff_interp = noaa_dhw_interp - noaa_dhw_tar


truth_mn_n = noaa_dhw_tar.mean(dim='time')
pred_mn_n = noaa_dhw_pred.mean(dim='time')
interp_mn_n = noaa_dhw_interp.mean(dim='time')

pred_diff_mn_n = noaa_dhw_diff_pr.mean(dim='time')
interp_diff_mn_n = noaa_dhw_diff_interp.mean(dim='time')


##RMSE and correlation
def rmse_ml(predictions, targets):
    return np.sqrt(np.mean(((predictions - targets) ** 2)))



#Do plot

m = Basemap(llcrnrlon=142.,llcrnrlat=-25.,urcrnrlon=154.,urcrnrlat=-10.,
    resolution='l',projection='merc',lat_ts=-15,lat_0=lat_sst.mean(),lon_0=lon_sst.mean())

#create grid to plot
lonx,latx = np.meshgrid(lon_sst,lat_sst)
xi,yi=m(lonx,latx)

fig, axs = plt.subplots(2, 5,figsize = (11.0, 5.0))

h1 = plt.subplot(2,5,1)
h1.set_title("Target", fontsize=10)
h1.set_ylabel("OFAM", fontweight = 'bold')

cmap=mpl.cm.magma_r
bounds = list(np.arange(0,8,0.5))
norm =  mpl.colors.BoundaryNorm(bounds, ncolors=cmap.N, extend ="both")

cs = m.pcolor(xi,yi,truth_mn_o.data,cmap=mpl.cm.magma_r, vmin=0,vmax=12)
# Add Grid Lines
m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=7, color = "grey")
m.drawmeridians(np.arange(143., 194., 10.), labels=[0,0,0,1], fontsize=7, color = "grey")
m.drawcoastlines()
m.fillcontinents(color='grey')
plt.annotate("(a)", m(143.1,-20), fontsize=12)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.5, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]

x, y = m(lons, lats)
m.plot(x, y, marker=None,color='k', linewidth=1.5, linestyle = 'dotted')


ax2 = plt.subplot(2,5,2)
ax2.set_title("ML Downscaling", fontsize=10)


cs = m.pcolor(xi,yi,pred_mn_o.data, cmap =mpl.cm.magma_r,  vmin=0,vmax=12)
# Add Grid Lines
m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=7, color = "grey")
m.drawmeridians(np.arange(143., 194., 10.), labels=[0,0,0,1], fontsize=7, color = "grey")
m.drawcoastlines()
m.fillcontinents(color='grey')
plt.annotate("(b)", m(143.1,-20), fontsize=12)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.5, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]

x, y = m(lons, lats)
m.plot(x, y, marker=None,color='k', linewidth=1.5, linestyle = 'dotted')

# add text box for the statistics
stats1 = (f'rmse = {rmse_ml(ofam_dhw_pred,ofam_dhw_tar).data:.2f}\n')
stats2 = (f'r = {xr.corr(ofam_dhw_pred,ofam_dhw_tar).data:.2f}\n')
ax2.text(0.95, 0.78, stats1, fontsize=9, color = 'black', fontweight = 'bold',
        transform=ax2.transAxes, horizontalalignment='right')
ax2.text(0.95, 0.70, stats2, fontsize=9, color = 'black', fontweight = 'bold',
        transform=ax2.transAxes, horizontalalignment='right')


ax2 = plt.subplot(2,5,3)
ax2.set_title("Interpolation", fontsize=10)

cs = m.pcolor(xi,yi,interp_mn, cmap =mpl.cm.magma_r,  vmin=0,vmax=12)
# Add Grid Lines
m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=7, color = "grey")
m.drawmeridians(np.arange(143., 194., 10.), labels=[0,0,0,1], fontsize=7, color = "grey")
m.drawcoastlines()
m.fillcontinents(color='grey')
plt.annotate("(c)", m(143.1,-20), fontsize=12)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.5, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]

x, y = m(lons, lats)
m.plot(x, y, marker=None,color='k', linewidth=1.5, linestyle = 'dotted')

# add text box for the statistics
stats1 = (f'rmse = {rmse_ml(ofam_dhw_interp,ofam_dhw_tar).data:.2f}\n')
stats2 = (f'r = {xr.corr(ofam_dhw_interp,ofam_dhw_tar).data:.2f}\n')
ax2.text(0.95, 0.78, stats1, fontsize=9, color = 'black', fontweight = 'bold',
        transform=ax2.transAxes, horizontalalignment='right')
ax2.text(0.95, 0.70, stats2, fontsize=9, color = 'black', fontweight = 'bold',
        transform=ax2.transAxes, horizontalalignment='right')



cax = plt.axes((0.15, 0.0008, 0.33, 0.015))
cc=fig.colorbar(cs, cax=cax,orientation='horizontal')
cc.ax.tick_params(labelsize=10) 
cc.ax.set_title('DHW [$^\circ$C - weeks]')


ax3 = plt.subplot(2,5,4)
ax3.set_title("ML Downscaling - Target", fontsize=10)

cs = m.pcolor(xi,yi,pred_diff_mn_o.data, cmap = cmo.cm.curl,vmin=-4,vmax=4)
# Add Grid Lines
m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=7, color = "grey")
m.drawmeridians(np.arange(143., 194., 10.), labels=[0,0,0,1], fontsize=7, color = "grey")
m.drawcoastlines()
m.fillcontinents(color='grey')
plt.annotate("(d)", m(143.1,-20), fontsize=12)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.5, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]

x, y = m(lons, lats)
m.plot(x, y, marker=None,color='k', linewidth=1.5, linestyle = 'dotted')


ax3 = plt.subplot(2,5,5)
ax3.set_title("Interpolation - Target", fontsize=10)

cs = m.pcolor(xi,yi,interp_diff_mn_o.data, cmap = cmo.cm.curl,vmin=-4,vmax=4)
# Add Grid Lines
m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=7, color = "grey")
m.drawmeridians(np.arange(143., 194., 10.), labels=[0,0,0,1], fontsize=7, color = "grey")
m.drawcoastlines()
m.fillcontinents(color='grey')
plt.annotate("(e)", m(143.1,-20), fontsize=12)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.5, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]

x, y = m(lons, lats)
m.plot(x, y, marker=None,color='k', linewidth=1.5, linestyle = 'dotted')


# Add Colorbar
cax = plt.axes((0.65, 0.008, 0.20, 0.015))
cc=fig.colorbar(cs, cax=cax,orientation='horizontal')
cc.ax.tick_params(labelsize=10) 
cc.ax.set_title('[$^\circ$C - weeks]')

ax1 = plt.subplot(2,5,6)
ax1.set_ylabel("NOAA", fontweight = 'bold')

cmap=mpl.cm.magma_r
bounds = list(np.arange(0,15,1))
norm =  mpl.colors.BoundaryNorm(bounds, ncolors=cmap.N, extend ="both")

cs = m.pcolor(xi,yi,truth_mn_n.data,cmap=mpl.cm.magma_r, vmin=0,vmax=12)
# Add Grid Lines
m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=7, color = "grey")
m.drawmeridians(np.arange(143., 194., 10.), labels=[0,0,0,1], fontsize=7, color = "grey")
m.drawcoastlines()
m.fillcontinents(color='grey')
plt.annotate("(f)", m(143.1,-20), fontsize=12)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.5, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]

x, y = m(lons, lats)
m.plot(x, y, marker=None,color='k', linewidth=1.5, linestyle = 'dotted')


ax2 = plt.subplot(2,5,7)
# ax2.set_title("ML Downscaling", fontsize=9)


cs = m.pcolor(xi,yi,pred_mn_n.data, cmap =mpl.cm.magma_r,  vmin=0,vmax=12)
# Add Grid Lines
m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=7, color = "grey")
m.drawmeridians(np.arange(143., 194., 10.), labels=[0,0,0,1], fontsize=7, color = "grey")
m.drawcoastlines()
m.fillcontinents(color='grey')
plt.annotate("(g)", m(143.1,-20), fontsize=12)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.5, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]

x, y = m(lons, lats)
m.plot(x, y, marker=None,color='k', linewidth=1.5, linestyle = 'dotted')

# add text box for the statistics
stats1 = (f'rmse = {rmse_ml(noaa_dhw_pred,noaa_dhw_tar).data:.2f}\n')
stats2 = (f'r = {xr.corr(noaa_dhw_pred,noaa_dhw_tar).data:.2f}\n')
ax2.text(0.95, 0.78, stats1, fontsize=9, color = 'black', fontweight = 'bold',
        transform=ax2.transAxes, horizontalalignment='right')
ax2.text(0.95, 0.70, stats2, fontsize=9, color = 'black', fontweight = 'bold',
        transform=ax2.transAxes, horizontalalignment='right')


ax2 = plt.subplot(2,5,8)
# ax2.set_title("Interpolation", fontsize=9)


cs = m.pcolor(xi,yi,interp_mn.data, cmap =mpl.cm.magma_r,  vmin=0,vmax=12)
# Add Grid Lines
m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=7, color = "grey")
m.drawmeridians(np.arange(143., 194., 10.), labels=[0,0,0,1], fontsize=7, color = "grey")
m.drawcoastlines()
m.fillcontinents(color='grey')
plt.annotate("(h)", m(143.1,-20), fontsize=12)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.5, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]

x, y = m(lons, lats)
m.plot(x, y, marker=None,color='k', linewidth=1.5, linestyle = 'dotted')

# add text box for the statistics
stats1 = (f'rmse = {rmse_ml(noaa_dhw_interp,noaa_dhw_tar).data:.2f}\n')
stats2 = (f'r = {xr.corr(noaa_dhw_interp,noaa_dhw_tar).data:.2f}\n')
ax2.text(0.95, 0.78, stats1, fontsize=9, color = 'black', fontweight = 'bold',
        transform=ax2.transAxes, horizontalalignment='right')
ax2.text(0.95, 0.70, stats2, fontsize=9, color = 'black', fontweight = 'bold',
        transform=ax2.transAxes, horizontalalignment='right')


ax3 = plt.subplot(2,5,9)

cs = m.pcolor(xi,yi,pred_diff_mn_n.data, cmap = cmo.cm.curl,vmin=-4,vmax=4)#cmo.cm.curl
# Add Grid Lines
m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=7, color = "grey")
m.drawmeridians(np.arange(143., 194., 10.), labels=[0,0,0,1], fontsize=7, color = "grey")
m.drawcoastlines()
m.fillcontinents(color='grey')
plt.annotate("(i)", m(143.1,-20), fontsize=12)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.5, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]

x, y = m(lons, lats)
m.plot(x, y, marker=None,color='k', linewidth=1.5, linestyle = 'dotted')


ax3 = plt.subplot(2,5,10)

cs = m.pcolor(xi,yi,interp_diff_mn_n.data, cmap = cmo.cm.curl,vmin=-4,vmax=4)#cmo.cm.curl
# Add Grid Lines
m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=7, color = "grey")
m.drawmeridians(np.arange(143., 194., 10.), labels=[0,0,0,1], fontsize=7, color = "grey")
m.drawcoastlines()
m.fillcontinents(color='grey')
plt.annotate("(j)", m(143.1,-20), fontsize=12)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.5, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]

x, y = m(lons, lats)
m.plot(x, y, marker=None,color='k', linewidth=1.5, linestyle = 'dotted')

plt.show()