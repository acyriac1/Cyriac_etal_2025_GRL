import xarray as xr
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cmocean as cmo
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.gridspec as gridspec



##-----load the data
tar = xr.open_dataset('./ofam3_sst_historic.nc')
sst_tar = tar.SST.sel(time = slice('2008-01-01','2014-12-31'))
mon_tar = sst_past.mean(('lon_sst','lat_sst')).groupby("time.month")

pred = xr.open_dataset('./ofam3_sst_prediction.nc')
sst_pred = pred.SST.sel(time = slice('2008-01-01','2014-12-31'))
mon_pred = sst_pred.mean(('lon_sst','lat_sst')).groupby("time.month")

inter = xr.open_dataset('./ofam3_sst_interpolation.nc')
sst_interp = inter.SST.sel(time = slice('2008-01-01','2014-12-31'))
mon_interp = sst_interp.mean(('lon_sst','lat_sst')).groupby("time.month")


#RMSE
def rmse(predictions, targets):
    return np.sqrt(np.mean(((predictions - targets) ** 2),axis=0))

#pandas frame to store the data
df = pd.DataFrame(columns=['Month','RMSE10','RMSE80'])

# create an empty list to store dictionaries
dict_list = []
month = ['JAN', 'FEB', 'MAR', 'APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']

# write the for loop and store output in a dictionary
for i, mon in enumerate(month):
    row_dict = {'Month': mon,\
                'RMSE10': (rmse(mon_pred[i+1].data,mon_tar[i+1].data)),\
               'RMSE80': (rmse(mon_interp[i+1].data,mon_tar[i+1].data))}
    #append each dictionary to a list
    dict_list.append(row_dict)

# convert list of dictionaries to pandas dataframe
df = pd.DataFrame.from_dict(dict_list)

###histogram 

##take the difference and then get the area averaged mean
sst_hist_down = sst_pred - sst_tar
sst_hist_interp = sst_interp - sst_tar

sst_down_mn = sst_hist_down.mean(('lon_sst','lat_sst')).data
sst_interp_mn = sst_hist_interp.mean(('lon_sst','lat_sst')).data

##overview plot
sst_data = np.squeeze(sst_tar.sel(time = '2012-01-01')).data

####do plot

fig = plt.figure(tight_layout=True,figsize = (11.0, 6.0))
gs = gridspec.GridSpec(2, 2)

#RMSE plot

ax = fig.add_subplot(gs[1, 0:])
ax.plot_date(df.Month, df["RMSE10"], color="blue", label="ML downscaling", linestyle="-",linewidth=2)
ax.plot_date(df.Month, df["RMSE80"], color="red", label="Interpolation", linestyle="-",linewidth=2)
plt.ylabel('RMSE [$^\circ$C]', fontsize=14)
ax.legend(fontsize=12)
ax.tick_params(labelsize=12) 
plt.text(0,0.035,"(c)",fontsize=13)


print('RMSE_ML = ', df["RMSE10"].mean())
print('RMSE_intp = ', df["RMSE80"].mean())

#snapshot
ax = fig.add_subplot(gs[0,0])


#SST
m3 = Basemap(llcrnrlon=141.,llcrnrlat=-40.,urcrnrlon=188.,urcrnrlat=-5.,
    resolution='l',projection='merc',lat_ts=-25,lat_0=lat_sst.mean(),lon_0=lon_sst.mean())

#create grid to plot
lonx,latx = np.meshgrid(lon_sst,lat_sst)
xi,yi=m3(lonx,latx)

cmap = mpl.cm.Spectral_r
bounds = list(np.arange(16,33,1))
norm =  mpl.colors.BoundaryNorm(bounds, ncolors=cmap.N, extend ="both")
cs3 = m3.pcolor(xi,yi,sst_data,cmap=cmap, norm=norm)
# Add Grid Lines
m3.drawparallels(np.arange(-45., 6., 10.), labels=[1,0,0,0], fontsize=12,color = "grey")
m3.drawmeridians(np.arange(143., 194., 10.), labels=[0,0,0,1], fontsize=12,color = "grey")
m3.drawcoastlines(color ="black")
m3.fillcontinents(color='grey')

#texts
plt.annotate("Great Barrier Reef", m3(145,-15), rotation=307, rotation_mode = 'anchor', fontsize=8, fontweight='bold')
plt.annotate("Coral Sea", m3(147,-15),fontsize=8, fontweight='bold')
plt.annotate("AUSTRALIA", m3(142,-35),fontsize=9, fontweight='bold',rotation=45)

plt.annotate("(a)", m3(143.1,-20), fontsize=13)

## Add the polygon for GBR conservation area
lons = [142.6, 145, 145, 146, 147, 152.9, 154, 152]
lats = [-10.65, -10.65, -13, -15, -17.5, -21, -24.5, -24.5]
x, y = m3(lons, lats)
m3.plot(x, y, marker=None,color='g', linewidth=1.2)

#Add the box for GBR region
lons = [142, 154, 154, 142, 142]
lats = [-10, -10, -25, -25, -10]
x, y = m3(lons, lats)
m3.plot(x, y, marker=None,color='b', linewidth=1.5)

cc=fig.colorbar(cs3, orientation='vertical')
cc.ax.tick_params(labelsize=12) 
cc.ax.set_title('SST [$^\circ$C]',fontsize=12)

###### inset axes to zoom into GBR
axins = zoomed_inset_axes(ax, 0.09, loc=1)
axins.set_xlim(110,188)
axins.set_ylim(-50,5)

plt.xticks(visible=False)
plt.yticks(visible=False)

#larger plot
#SST
m3 = Basemap(llcrnrlon=40,llcrnrlat=-70.,urcrnrlon=210.,urcrnrlat=15.,
    resolution='l', projection='merc', lat_0 = -25, lon_0 = 130)

m3.drawcoastlines(color ="#636363",linewidth=0.9)
m3.fillcontinents(color='grey')

#Add the box for model domain
lons = [142, 188, 188, 142, 142]
lats = [-5, -5, -40, -40, -5]
x, y = m3(lons, lats)
m3.plot(x, y, marker=None,color='k', linewidth=1.5)


#Histograms

ax = fig.add_subplot(gs[0, 1])

bins = 30
bins = np.histogram(np.hstack((sst_down_mn,sst_interp_mn)), bins=bins)[1]

ax.hist(sst_down_mn, bins=bins, density = True,
         label = ["ML Downscaling - Target"],color = ["g"],alpha =0.3, ec = "g",lw =1.1 )
ax.hist(sst_interp_mn, bins=bins, density = True,
         label = ["Interpolation - Target"],color = ["r"],alpha =0.3, ec = "m",lw =1.1 )

ax.set_xlabel("SST [$^\circ$C]", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
# pos = axs.get_position()
# h1.set_position([pos.x0, pos.y0, pos.width, pos.height * 1])
ax.legend(
    loc='upper right', 
    # bbox_to_anchor=(1.0, 1.25),
    ncol=1,
    fontsize=10)
plt.text(-0.045,140,"(b)",fontsize=13)
ax.tick_params(labelsize=12) 

plt.show()
