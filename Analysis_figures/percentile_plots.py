import xarray as xr
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cmocean as cmo
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot_2samples


##-----load the data
tar = xr.open_dataset('./ofam3_sst_historic.nc')
sst_tar = tar.SST

pred = xr.open_dataset('./ofam3_sst_prediction.nc')
sst_pred = pred.SST


inter = xr.open_dataset('./ofam3_sst_interpolation.nc')
sst_interp = inter.SST

#99 percentile
perc_tar = np.percentile(sst_tar,99,axis=0)
perc_pred = np.percentile(sst_pred,99,axis=0)
perc_interp = np.percentile(sst_interp,99,axis=0)

##extreme difference
diff_pred = perc_pred - perc_tar
diff_interp = perc_interp - perc_tar


##RMSE and correlation
def rmse(predictions, targets):
    return np.sqrt(np.mean(((predictions - targets) ** 2)))



####------------------variance plots

def estimate_variance_from_99th_percentile(data, axis=0):
 """
 Estimates the variance of an array from its 99th percentile value along a selected axis.

 Args:
   data: A NumPy array.
   axis: The axis along which to estimate the variance (default: 0).

 Returns:
   A NumPy array containing the estimated variance for each element along the specified axis.
 """

 # Calculate the 99th percentile
 percentiles = np.percentile(data, 99, axis=axis)

 # Calculate the squared deviations from the 99th percentile
 squared_deviations = (data - percentiles)**2

 # Estimate the variance as the mean of the squared deviations
 variance_estimates = np.mean(squared_deviations, axis=axis)

 return variance_estimates

var_tar = estimate_variance_from_99th_percentile(sst_tar, axis=0)
var_pred = estimate_variance_from_99th_percentile(sst_pred, axis=0)
var_interp = estimate_variance_from_99th_percentile(sst_interp, axis=0)

std_tar = np.sqrt(var_tar)
std_pred = np.sqrt(var_pred)
std_interp = np.sqrt(var_interp)

###--------do plot

m = Basemap(llcrnrlon=142.,llcrnrlat=-25.,urcrnrlon=154.,urcrnrlat=-10.,
    resolution='l',projection='merc',lat_ts=-15,lat_0=lat_ap.mean(),lon_0=lon_ap.mean())

#create grid to plot
lonx,latx = np.meshgrid(lon_ap,lat_ap)
xi,yi=m(lonx,latx)

fig = plt.figure(figsize = (10.0, 12.0),layout='constrained')
# fig.suptitle('Variance from 99th percentile', y=0.95)

ax1 = fig.add_subplot(3,3,1)
ax1.set_title("Target", fontsize=14)
cmap=mpl.cm.Spectral_r#CMRmap
bounds = list(np.arange(28.5,32.5,0.2))
norm =  mpl.colors.BoundaryNorm(bounds, ncolors=cmap.N, extend = "both")

cs = m.pcolor(xi,yi,perc_tar,cmap=cmap,norm=norm)
# Add Grid Lines
m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=10,color ="grey")
m.drawmeridians(np.arange(142., 194., 10.), labels=[0,0,0,1], fontsize=10,color ="grey")
m.drawcoastlines()
m.fillcontinents(color='grey')

plt.annotate("(a)", m(143.4,-20), fontsize=14)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.5, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]
x, y = m(lons, lats)
m.plot(x, y, color='k', linewidth=1.5, linestyle = 'dotted')

ax2 = fig.add_subplot(3,3,2)
ax2.set_title("ML downscaling", fontsize=14)
cs = m.pcolor(xi,yi,perc_pred,cmap=cmap,norm=norm)
# Add Grid Lines
m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=10,color ="grey")
m.drawmeridians(np.arange(142., 194., 10.), labels=[0,0,0,1], fontsize=10,color ="grey")
m.drawcoastlines()
m.fillcontinents(color='grey')
plt.annotate("(b)", m(143.4,-20), fontsize=14)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.5, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]
x, y = m(lons, lats)
m.plot(x, y, color='k', linewidth=1.5, linestyle = 'dotted')

# add text box for the statistics
stats1 = (f'rmse = {rmse(perc_pred,perc_tar):.2f}\n')
stats2 = (f'r = {np.corrcoef(perc_pred,perc_tar)[0][1]:.2f}\n')
ax2.text(0.95, 0.85, stats1, fontsize=9, color = 'black', fontweight = 'bold',
        transform=ax2.transAxes, horizontalalignment='right')
ax2.text(0.95, 0.78, stats2, fontsize=9, color = 'black', fontweight = 'bold',
        transform=ax2.transAxes, horizontalalignment='right')

ax3 = fig.add_subplot(3,3,3)
ax3.set_title("Interpolation", fontsize=14)
cs = m.pcolor(xi,yi,perc_interp,cmap=cmap,norm=norm)
# Add Grid Lines
m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=10,color ="grey")
m.drawmeridians(np.arange(142., 194., 10.), labels=[0,0,0,1], fontsize=10,color ="grey")
m.drawcoastlines()
m.fillcontinents(color='grey')
plt.annotate("(c)", m(143.4,-20), fontsize=14)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.5, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]
x, y = m(lons, lats)
m.plot(x, y, color='k', linewidth=1.5, linestyle = 'dotted')

# add text box for the statistics
stats1 = (f'rmse = {rmse(perc_interp,perc_tar):.2f}\n')
stats2 = (f'r = {np.corrcoef(perc_interp,perc_tar)[0][1]:.2f}\n')
ax3.text(0.95, 0.85, stats1, fontsize=9, color = 'black', fontweight = 'bold',
        transform=ax3.transAxes, horizontalalignment='right')
ax3.text(0.95, 0.78, stats2, fontsize=9, color = 'black', fontweight = 'bold',
        transform=ax3.transAxes, horizontalalignment='right')

cax = plt.axes((0.99, 0.7, 0.010, 0.27))
plt.colorbar(cax=cax).set_label(label='99th percentile SST [$^\circ$C]',size=14)
cax.tick_params(labelsize=13)

cmap=mpl.cm.rainbow#cubehelix_r
bounds = list(np.arange(2.0,6.5,0.2))
norm =  mpl.colors.BoundaryNorm(bounds, ncolors=cmap.N, extend = "both")

#variance plots
ax4 = fig.add_subplot(3,3,4)
# ax4.set_title("Downscaling")
cs = m.pcolor(xi,yi,std_tar,cmap=cmap,norm=norm)
# Add Grid Lines
m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=10,color ="grey")
m.drawmeridians(np.arange(142., 194., 10.), labels=[0,0,0,1], fontsize=10,color ="grey")
m.drawcoastlines()
m.fillcontinents(color='grey')
plt.annotate("(d)", m(143.4,-20), fontsize=14)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.5, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]
x, y = m(lons, lats)
m.plot(x, y, color='k', linewidth=1.5, linestyle = 'dotted')

ax5 = fig.add_subplot(3,3,5)
# ax5.set_title("Downscaling")
cs = m.pcolor(xi,yi,std_pred,cmap=cmap,norm=norm)
# Add Grid Lines
m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=10,color ="grey")
m.drawmeridians(np.arange(142., 194., 10.), labels=[0,0,0,1], fontsize=10,color ="grey")
m.drawcoastlines()
m.fillcontinents(color='grey')
plt.annotate("(e)", m(143.4,-20), fontsize=14)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.5, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]
x, y = m(lons, lats)
m.plot(x, y, color='k', linewidth=1.5, linestyle = 'dotted')

ax6 = fig.add_subplot(3,3,6)
# ax5.set_title("Downscaling")
cs = m.pcolor(xi,yi,std_interp,cmap=cmap,norm=norm)
# Add Grid Lines
m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=10,color ="grey")
m.drawmeridians(np.arange(142., 194., 10.), labels=[0,0,0,1], fontsize=10,color ="grey")
m.drawcoastlines()
m.fillcontinents(color='grey')
plt.annotate("(f)", m(143.4,-20), fontsize=14)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.5, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]
x, y = m(lons, lats)
m.plot(x, y, color='k', linewidth=1.5, linestyle = 'dotted')


cax = plt.axes((0.99, 0.37, 0.010, 0.27))
# plt.colorbar(cax=cax,label='SST variance [$^\circ$C^2]')
plt.colorbar(cax=cax).set_label(label='SST st.deviation [$^\circ$C]',size=14)
cax.tick_params(labelsize=13)

cmap=mpl.cm.PiYG_r
bounds = list(np.arange(-0.8,0.8,0.05))
norm =  mpl.colors.BoundaryNorm(bounds, ncolors=cmap.N, extend = "both")

ax7 = fig.add_subplot(3,3,7)
ax7.set_title("ML downscaling - Target", fontsize=14)
cs = m.pcolor(xi,yi,diff_pred,cmap=cmap,vmin=-0.5,vmax=0.5)
# Add Grid Lines
m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=10,color ="grey")
m.drawmeridians(np.arange(142., 194., 10.), labels=[0,0,0,1], fontsize=10,color ="grey")
m.drawcoastlines()
m.fillcontinents(color='grey')
plt.annotate("(g)", m(143.4,-20), fontsize=14)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.5, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]
x, y = m(lons, lats)
m.plot(x, y, color='k', linewidth=1.5, linestyle = 'dotted')


# cax = plt.axes((0.91, 0.55, 0.015, 0.3))
# plt.colorbar(cax=cax,label='ML downscaling - Truth [$^\circ$C]')


ax8 = fig.add_subplot(3,3,8)
ax8.set_title("Interpolation - Target", fontsize=14)
cs = m.pcolor(xi,yi,diff_interp,cmap=cmap,vmin=-0.5,vmax=0.5)
# Add Grid Lines
m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=10,color ="grey")
m.drawmeridians(np.arange(142., 194., 10.), labels=[0,0,0,1], fontsize=10,color ="grey")
m.drawcoastlines()
m.fillcontinents(color='grey')
plt.annotate("(h)", m(143.4,-20), fontsize=14)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.5, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]
x, y = m(lons, lats)
m.plot(x, y, color='k', linewidth=1.5, linestyle = 'dotted')

cax = plt.axes((0.67, 0.04, 0.010, 0.27))
cbar = plt.colorbar(cax=cax,extend='both').set_label(label='SST [$^\circ$C]',size=14)
cax.tick_params(labelsize=13)

plt.show()