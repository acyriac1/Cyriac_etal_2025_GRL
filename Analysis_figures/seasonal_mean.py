# ### This code is to calculate seasonal mean of SST, prediction and the difference
# ####------------------------------------------
import xarray as xr
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cmocean as cmo
import pandas as pd

##-----load the data
tar = xr.open_dataset('./ofam3_sst_historic.nc')
sst_tar = tar.SST
lat_10 = sst_tar.lat_sst.data
lon_10 = tar.lon_sst.data

pred = xr.open_dataset('./ofam3_sst_prediction.nc')
sst_pred = pred.SST

lowres = xr.open_dataset('./ofam3_sst_lowres80.nc')
sst_lowres = lowres.SST
lon_80 = lowres.lon_sst.data
lat_80 = lowres.lat_sst.data

inter = xr.open_dataset('./ofam3_sst_interpolation.nc')
sst_interp = inter.SST

#function to calculate the weighted seasonal mean
def season_mean(ds, calendar="standard"):
    # Make a DataArray with the number of days in each month, size = len(time)
    month_length = ds.time.dt.days_in_month

    # Calculate the weights by grouping by 'time.season'
    weights = (
        month_length.groupby("time.season") / month_length.groupby("time.season").sum()
    )

    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby("time.season").sum().values, np.ones(4))

    # Calculate the weighted average
    return (ds * weights).groupby("time.season").sum(dim="time")


#Apply seasonal mean
sst_tar_mn = season_mean(sst_tar,calendar="standard")
sst_pred_mn = season_mean(sst_pred,calendar="standard")
sst_lres_mn = season_mean(sst_lowres,calendar="standard")
sst_interp_mn = season_mean(sst_interp,calendar="standard")
sst_diff = season_mean(sst_pred-sst_tar,calendar="standard")

#RMSE
def rmse_ml(predictions, targets):
    return np.sqrt(np.mean(((predictions - targets) ** 2)))

#####-------------------do plot

m = Basemap(llcrnrlon=142.,llcrnrlat=-25.,urcrnrlon=154.,urcrnrlat=-10.,
    resolution='l',projection='merc',lat_ts=-15,lat_0=lat_10.mean(),lon_0=lon_10.mean())

#create grid to plot
lonx,latx = np.meshgrid(lon_10,lat_10)
xi,yi=m(lonx,latx)

cmap = mpl.cm.Spectral_r
bounds = list(np.arange(22,31,0.3))
norm =  mpl.colors.BoundaryNorm(bounds, ncolors=cmap.N, clip=True)

fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(12, 12),sharex =True, sharey=True)
for i, season in enumerate(("DJF", "MAM", "JJA", "SON")):
    cs1 = m.pcolormesh(xi,yi,sst_past_mn.sel(season=season),
        ax=axes[i, 0],
        norm=norm,
        cmap="Spectral_r",
    )
    m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=7,ax=axes[i, 0],color ="grey")
    m.drawmeridians(np.arange(143., 194., 10.), labels=[0,0,0,1], fontsize=7,ax=axes[i, 0],color ="grey")
    m.drawcoastlines(ax=axes[i, 0])
    m.fillcontinents(ax=axes[i, 0],color='grey')
    
    ## Add the polygon for GBR conservation area
    lons = [142.6, 145, 145, 146, 147, 152.9, 154, 152]
    lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]
    x, y = m(lons, lats)
    m.plot(x, y, ax=axes[i, 0],color='k', linewidth=1.5, linestyle = 'dotted')
    
    m = Basemap(llcrnrlon=142.,llcrnrlat=-25.,urcrnrlon=154.,urcrnrlat=-10.,
    resolution='l',projection='merc',lat_ts=-15,lat_0=lat_80.mean(),lon_0=lon_80.mean())

    #create grid to plot
    lonx,latx = np.meshgrid(lon_80,lat_80)
    xi,yi=m(lonx,latx)

    cs = m.pcolormesh(xi,yi,sst_80_mn.sel(season=season),
        ax=axes[i, 1],
        norm=norm,
        cmap="Spectral_r",
    )
    m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=7,ax=axes[i, 1],color ="grey")
    m.drawmeridians(np.arange(143., 194., 10.), labels=[0,0,0,1], fontsize=7,ax=axes[i, 1],color ="grey")
    m.drawcoastlines(ax=axes[i, 1])
    m.fillcontinents(ax=axes[i, 1],color='grey')
    
    # # Add Colorbar
    # cbar = m.colorbar(cs, location='right',pad="4%", ax = axes[i,1],extend = "both")
    # # cbar.set_label('SST')

    ## Add the polygon for GBR conservation area
    lons = [142.6, 145, 145, 146, 147, 152.9, 154, 152]
    lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]
    x, y = m(lons, lats)
    m.plot(x, y, ax=axes[i, 1], color='k', linewidth=1.5, linestyle = 'dotted')
    
    m = Basemap(llcrnrlon=142.,llcrnrlat=-25.,urcrnrlon=154.,urcrnrlat=-10.,
    resolution='l',projection='merc',lat_ts=-15,lat_0=lat_10.mean(),lon_0=lon_10.mean())

    #create grid to plot
    lonx,latx = np.meshgrid(lon_10,lat_10)
    xi,yi=m(lonx,latx)
    
    cs = m.pcolormesh(xi,yi,sst_interp_mn.sel(season=season),
        ax=axes[i, 2],
        norm=norm,
        cmap="Spectral_r",
    )
    m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=7,ax=axes[i, 2],color ="grey")
    m.drawmeridians(np.arange(143., 194., 10.), labels=[0,0,0,1], fontsize=7,ax=axes[i, 2],color ="grey")
    m.drawcoastlines(ax=axes[i, 2])
    m.fillcontinents(ax=axes[i, 2],color='grey')
    
    # # Add correlation and rmse values 
    # add text box for the statistics
    stat1 = (f'rmse = {rmse_ml(sst_interp_mn_mask.sel(season=season),sst_past_mn_mask.sel(season=season)).data:.2f}\n')
    stat2 = (f'r = {xr.corr(sst_interp_mn_mask.sel(season=season),sst_past_mn_mask.sel(season=season)).data:.2f}\n')
    axes[i,2].text(0.95, 0.85, stat1, fontsize=9, color = 'black',
            transform=axes[i,2].transAxes, horizontalalignment='right', fontweight = 'bold')
    axes[i,2].text(0.95, 0.78, stat2, fontsize=9, color = 'black',
            transform=axes[i,2].transAxes, horizontalalignment='right', fontweight = 'bold')


    ## Add the polygon for GBR conservation area
    lons = [142.6, 145, 145, 146, 147, 152.9, 154, 152]
    lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]
    x, y = m(lons, lats)
    m.plot(x, y, ax=axes[i, 2], color='k', linewidth=1.5, linestyle = 'dotted')
    

    cs = m.pcolormesh(xi,yi,sst_pred_mn.sel(season=season),
        ax=axes[i, 3],
        norm=norm,
        cmap="Spectral_r",
    )
    m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=7,ax=axes[i, 3],color ="grey")
    m.drawmeridians(np.arange(143., 194., 10.), labels=[0,0,0,1], fontsize=7,ax=axes[i, 3],color ="grey")
    m.drawcoastlines(ax=axes[i, 3])
    m.fillcontinents(ax=axes[i, 3],color='grey')
    
    # # Add correlation and rmse values 
    # add text box for the statistics
    stat1 = (f'rmse = {rmse_ml(sst_pred_mn_mask.sel(season=season),sst_past_mn_mask.sel(season=season)).data:.2f}\n')
    stat2 = (f'r = {xr.corr(sst_pred_mn_mask.sel(season=season),sst_past_mn_mask.sel(season=season)).data:.2f}\n')
    axes[i,3].text(0.95, 0.85, stat1, fontsize=9, color = 'black',
            transform=axes[i,3].transAxes, horizontalalignment='right', fontweight = 'bold')
    axes[i,3].text(0.95, 0.78, stat2, fontsize=9, color = 'black',
            transform=axes[i,3].transAxes, horizontalalignment='right', fontweight = 'bold')

    ## Add the polygon for GBR conservation area
    lons = [142.6, 145, 145, 146, 147, 152.9, 154, 152]
    lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]
    x, y = m(lons, lats)
    m.plot(x, y, ax=axes[i, 3], color='k', linewidth=1.5, linestyle = 'dotted')

    cs4 = m.pcolormesh(xi,yi,sst_diff.sel(season=season),
        ax=axes[i, 4],
        vmin=-0.5,
        vmax=0.5,
        cmap="coolwarm",
    )
    m.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=7,ax=axes[i, 4],color ="grey")
    m.drawmeridians(np.arange(143., 194., 10.), labels=[0,0,0,1], fontsize=7,ax=axes[i, 4],color ="grey")
    m.drawcoastlines(ax=axes[i, 4])
    m.fillcontinents(ax=axes[i, 4],color='grey')
    
    ## Add the polygon for GBR conservation area
    lons = [142.6, 145, 145, 146, 147, 152.9, 154, 152]
    lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]
    x, y = m(lons, lats)
    m.plot(x, y, ax=axes[i, 4], color='k', linewidth=1.5, linestyle = 'dotted')

    axes[i, 0].set_ylabel(season, fontsize =14, fontweight="bold")
    axes[i, 1].set_ylabel("")
    axes[i, 2].set_ylabel("")
    axes[i, 3].set_ylabel("")
    axes[i, 4].set_ylabel("")
    
    
axes[0, 0].set_title("Target [10 km]",fontsize =14)
axes[0, 1].set_title("Input [80 km]",fontsize =14)
axes[0, 2].set_title("Interpolation [10 km]",fontsize =14)
axes[0, 3].set_title("ML downscaling [10 km]",fontsize =14)
axes[0, 4].set_title("Difference",fontsize =14)

# Add Colorbar
cax = plt.axes((0.15, 0.0001, 0.46, 0.008))
cbar = fig.colorbar(cs1,cax=cax,ax = axes[1,0],extend = "both",orientation = 'horizontal')
cbar.set_label('SST [$^\circ$C]', fontweight="bold")

cax = plt.axes((0.82, 0.0001, 0.16, 0.008))
cbar = fig.colorbar(cs4,cax=cax,ax = axes[3,3],extend = "both",orientation = 'horizontal')
cbar.set_label('ML downscaling-Target [$^\circ$C]', fontweight="bold")

plt.tight_layout()
plt.show()
