import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import glob
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm, TwoSlopeNorm, LinearSegmentedColormap, BoundaryNorm
import scipy.ndimage
import cmocean
from datetime import datetime
import os
from scipy.ndimage import rotate
import cmocean.cm as cmo
from matplotlib.colorbar import Colorbar
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FormatStrFormatter
from matplotlib import gridspec
import matplotlib.colors as mcolors
import cmcrameri.cm as cmc
from  scipy.interpolate import RectBivariateSpline
import matplotlib.patheffects as pe
from labellines import labelLines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
import argparse
from ISMIP6_plots_cli_main_figures_2 import *


################################ Supplementary figure 2 ################################################################
## Need to download Antarctic velocity (measures 450m antarctic velocity data) dataset to do comparison plot
vel_path = '/Volumes/T7/antarctica_ice_velocity_450m_v2.nc'
def get_gl(bed, thickness):
    Hf = -1028/928*bed
    glmask = np.where(thickness > Hf, 1, np.zeros_like(bed))
    return glmask

def regrid(infield):
    x = np.arange(-3040.0e+3,3040.0e+3 + 1.0,8.0e+3)
    y = x.copy()
    x_0  = -3071000.0 - 500.0
    xc = np.arange(x_0 + 4.0e+3, x_0 + 8.0e+3*768 ,8.0e+3)
    yc = xc.copy()
    spl = RectBivariateSpline(xc,yc,infield,kx=1,ky=1)
    f_c = spl(x,y)
    return f_c

def regrid_vel_obs(infile):
    xc = infile.x.data
    yc = infile.y.data[::-1]
    x = np.arange(-3040.0e+3,3040.0e+3 + 1.0,8.0e+3)
    y = x.copy()
    inspeed = np.sqrt(infile.VX.data[::-1,::]**2+infile.VY.data[::-1,::]**2)
    spl = RectBivariateSpline(xc,yc,np.nan_to_num(inspeed, nan=1e9),kx=1,ky=1)
    f_c = spl(x,y)
    out = np.where(f_c > np.nanmax(inspeed), np.nan, f_c)
    return out

def regrid_vel_obs_error(infile):
    xc = infile.x.data
    yc = infile.y.data[::-1]
    x = np.arange(-3040.0e+3,3040.0e+3 + 1.0,8.0e+3)
    y = x.copy()
    inerr = np.sqrt(infile.ERRX.data[::-1,::]**2+infile.ERRY.data[::-1,::]**2)
    spl = RectBivariateSpline(xc,yc,np.nan_to_num(inerr, nan=1e9),kx=1,ky=1)
    f_c = spl(x,y)
    out = np.where(f_c > np.nanmax(inerr), np.nan, f_c)
    return out

if os.path.exists(vel_path):
    vel_obs = xr.open_dataset(vel_path)
    speed_obs = regrid_vel_obs(vel_obs)
    speed_err = regrid_vel_obs_error(vel_obs)

    relaxation_files = netcdf_dir + '/relaxation_run/'
    # spatial plots
    plotfiles = sorted(glob.glob(relaxation_files + '/*.nc'))

    thk0 = regrid(xr.open_dataset(plotfiles[0]).thickness)
    thk9 = regrid(xr.open_dataset(plotfiles[-1]).thickness)

    gl0 = regrid(get_gl(xr.open_dataset(plotfiles[0]).Z_base, xr.open_dataset(plotfiles[0]).thickness))
    gl9 = regrid(get_gl(xr.open_dataset(plotfiles[-1]).Z_base, xr.open_dataset(plotfiles[-1]).thickness))

    spd0 = regrid(np.sqrt(xr.open_dataset(plotfiles[0]).xVel.data**2 + xr.open_dataset(plotfiles[0]).yVel.data**2))
    spd9 = regrid(np.sqrt(xr.open_dataset(plotfiles[-1]).xVel.data**2 + xr.open_dataset(plotfiles[-1]).yVel.data**2))

    ## redo plot but vertical
    def add_colorbar_vert(mappable, ax, position='left', **kwargs):
        image=im
        ax.axis('off')
        insax = ax.inset_axes([0.0, 0.165, 1, 0.66])
        cb = plt.colorbar(mappable, cax=insax, **kwargs)
        cb.ax.yaxis.set_label_position(position)
        cb.ax.yaxis.set_ticks_position(position)
        return cb

    fig = plt.figure(figsize=(6,8))
    gs = GridSpec(3, 4, figure=fig, hspace=0.0, wspace=0.0, width_ratios=[0.1,1,1,0.1])

    ax0 = fig.add_subplot(gs[0,1])
    ax1 = fig.add_subplot(gs[0,2])
    ax2 = fig.add_subplot(gs[1,1])
    ax3 = fig.add_subplot(gs[1,2])
    ax4 = fig.add_subplot(gs[2,1])
    ax5 = fig.add_subplot(gs[2,2])

    # add plots
    im0 = ax0.imshow(np.ma.masked_where(thk9==0, thk9), origin='lower', vmax=3000, vmin=0, cmap='Blues_r')
    im1 = ax1.imshow(np.ma.masked_where(thk9==0, spd9), origin='lower', norm=LogNorm(vmin=0.01, vmax=1e3),
                     cmap=cmocean.cm.thermal)
    im2 = ax2.imshow(thk9 - thk0, origin='lower', vmin=-50, vmax=50, cmap='bwr_r')
    im3 = ax3.imshow(spd9 - spd0, origin='lower', vmin=-200, vmax=200, cmap='seismic')
    im4 = ax4.imshow(speed_obs, origin='lower', norm=LogNorm(vmin=0.01, vmax=1e3),
                     cmap=cmocean.cm.thermal)
    im5 = ax5.imshow(speed_obs - spd9, origin='lower', vmin=-200, vmax=200,
                     cmap='seismic')

    add_colorbar_vert(im0, fig.add_subplot(gs[0,0]),
                      position='left', label='Thickness (m)', orientation='vertical', extend='max')
    add_colorbar_vert(im1, fig.add_subplot(gs[0,-1]),
                      position='right', label='Speed (m yr$^{-1}$)', orientation='vertical', extend='max')
    add_colorbar_vert(im2, fig.add_subplot(gs[1,0]),
                      position='left', label='$\Delta$ Thickness (m)', orientation='vertical',
                      extend='both')
    add_colorbar_vert(im3, fig.add_subplot(gs[1,-1]),
                      position='right', label='$\Delta$ Speed (m yr$^{-1}$)', orientation='vertical',
                      extend='both')
    add_colorbar_vert(im4, fig.add_subplot(gs[2,0]),
                      position='left', label='Obs peed (m yr$^{-1}$)', orientation='vertical', extend='max')
    add_colorbar_vert(im5, fig.add_subplot(gs[2,-1]),
                      position='right', label='Speed obs - model\n(m yr$^{-1}$)', orientation='vertical',
                      extend='both')

    labels = 'abcdef'
    for i, ax in enumerate([ax0, ax1, ax2, ax3, ax4, ax5]):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.contour(gl9, [0.5], colors='k', linewidths=0.3)
        ax.contour(thk9, [0.5], colors='k', linestyles=':', linewidths=0.3)
        ax.text(0.05,0.9,'({})'.format(labels[i]), transform=ax.transAxes)

    plt.savefig(savedir + 'supplementary_fig02.png', bbox_inches='tight', dpi=dpi)

    # speed and thickness RMSE
    mask = np.where((np.isnan(spd9)==True) & (np.isnan(speed_obs)==True), 1, np.zeros_like(spd9))
    RMSE_speed = np.sqrt(np.nanmean((spd9 - speed_obs)**2))
    thck_diff_obs = np.where(bedmap_8km.icemask_grounded + bedmap_8km.icemask_shelves == 0,
                             np.nan, thk9) \
                    - np.where(bedmap_8km.icemask_grounded + bedmap_8km.icemask_shelves == 0,
                               np.nan, bedmap_8km.thickness.data)
    RMSE_thick = np.sqrt(np.nanmean((thck_diff_obs)**2))
else:
    print('Cant plot supplementary fig 2, download Measures Antarctic data...')


################################ Supplementary figure 3 ################################################################
def add_melt_subplot(ax, array, x0, x1, y0, y1, boundcol, edgelinestyle):
    melt_cmap = cmocean.cm.thermal_r
    melt_norm = BoundaryNorm([0,0.1, 0.5, 1, 2, 5, 10, 50, 100], ncolors=256)
    im = ax.imshow(array[y0:y1,x0:x1], origin='lower',
                   cmap=melt_cmap, norm=melt_norm)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(color=boundcol, labelcolor=boundcol)
    for spine in ax.spines.values():
        spine.set_edgecolor(boundcol)
        spine.set_linestyle(edgelinestyle)

    return im, ax

def add_contour_smb_bmb(ax, array_gl, array_shelf, x0, x1, y0, y1, lc):
    ax.contour(array_gl[y0:y1,x0:x1], [0.5], linewidths=0.5, colors=lc)
    ax.contour(array_shelf[y0:y1,x0:x1], [0.5], linewidths=1, colors=lc, linestyles=':')

# open smb and basal melt files
acabf = xr.open_dataset(netcdf_root.format('ctrl', 'acabf', 'ctrl'))
libmassbffl = xr.open_dataset(netcdf_root.format('ctrl', 'libmassbffl', 'ctrl'))
# open ice shelf mask, grounded mask and ice mask
sftflf = xr.open_dataset(netcdf_root.format('ctrl', 'sftflf', 'ctrl'))
sftgrf = xr.open_dataset(netcdf_root.format('ctrl', 'sftgrf', 'ctrl'))
sftgif = xr.open_dataset(netcdf_root.format('ctrl', 'sftgif', 'ctrl'))

# setup colormap to match Mottram 2021 paper smb plot mean
mottram_2021_colors=['#FFFFDB', '#EBF6BB', '#BFE5B9', '#7EC1BE', '#53A2C1', '#386BAA', '#263890', '#0B1A57']
def convert_hex_rgb_norm(hex):
    h = hex.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))
mot_cols = [convert_hex_rgb_norm(hex) for hex in mottram_2021_colors]

smb_cmap = LinearSegmentedColormap.from_list('mottram_2021', mot_cols, N=len(mottram_2021_colors))
smb_field = np.mean(acabf.sel(time=slice(2015,2100)).acabf.data, axis=0)
smb_norm = BoundaryNorm([-1, 0, 25,50, 100, 200, 400, 800, 1000], ncolors=8)

avmelt = np.mean(libmassbffl.sel(time=slice(2015, 2100)).libmassbffl.data/918, axis=0)
melt_field = np.ma.masked_where(np.max(sftflf.sftflf.data, axis=0)==0, -avmelt)

fig = plt.figure(figsize=(8,8))
gs = GridSpec(6, 6, figure=fig, hspace=0.05, wspace=0.05)
ax0 = fig.add_subplot(gs[0:2,0:2])
im0 = ax0.imshow(np.ma.masked_where(sftgif.sftgif.sel(time=2010).data==0, smb_field),
                 norm = smb_norm, origin='lower', cmap=smb_cmap)
ax0.set_xticks([])
ax0.set_yticks([])
add_contour_smb_bmb(ax0, sftgrf.sftgrf.sel(time=2015), sftgif.sftgif.sel(time=2015),
                    None, None, None, None, 'k') #'lightgray')

ax1 =  fig.add_subplot(gs[0:2,2:4])
im1, ax1 = add_melt_subplot(ax1, melt_field, None, None, None, None, 'k', '-')
add_contour_smb_bmb(ax1, sftgrf.sftgrf.sel(time=2015), sftgif.sftgif.sel(time=2015),
                    None, None, None, None, 'k')

ax2 = fig.add_subplot(gs[0:2,4:6])
im2, ax2 = add_melt_subplot(ax2, melt_field, 180, 330, 390, 520, 'y', '-')
add_contour_smb_bmb(ax2, sftgrf.sftgrf.sel(time=2015), sftgif.sftgif.sel(time=2015),
                    180, 330, 390, 520, 'k')

ax3 = fig.add_subplot(gs[2:4,4:6])
im3, ax3 = add_melt_subplot(ax3, melt_field, 290, 440, 190, 330, 'b', '-')
add_contour_smb_bmb(ax3, sftgrf.sftgrf.sel(time=2015), sftgif.sftgif.sel(time=2015),
                    290, 440, 190, 330, 'k')

ax4 = fig.add_subplot(gs[2:5,2:4])
im4, ax4 = add_melt_subplot(ax4, melt_field, 120, 220, 220, 370, 'r', '-')
add_contour_smb_bmb(ax4, sftgrf.sftgrf.sel(time=2015), sftgif.sftgif.sel(time=2015),
                    120, 220, 220, 370, 'k')

ax6 = fig.add_subplot(gs[4:5,4:6])
im6, ax6 = add_melt_subplot(ax6, melt_field, 280, 550, 530, 680, 'g', '-')
add_contour_smb_bmb(ax6, sftgrf.sftgrf.sel(time=2015), sftgif.sftgif.sel(time=2015),
                    280, 550, 530, 680, 'k')

#[210:270, 645:690]
ax7 = fig.add_subplot(gs[2:4,0:2])
im7, ax7 = add_melt_subplot(ax7, melt_field, 645, 690, 210, 270, 'm', '-')
add_contour_smb_bmb(ax7, sftgrf.sftgrf.sel(time=2015), sftgif.sftgif.sel(time=2015),
                    645, 690, 210, 270, 'k')


# add bounding boxes onto melt plot
add_bounding_box(ax1, 0, 290, 440, 190, 330, 'b', '-')
add_bounding_box(ax1, 0, 120, 220, 220, 370, 'r', '-')
add_bounding_box(ax1, 0, 280, 550, 530, 680, 'g', '-')
add_bounding_box(ax1, 0, 180, 330, 390, 520,  'y', '-')
add_bounding_box(ax1, 0, 645, 690, 210, 270,  'm', '-')

ax5 = fig.add_subplot(gs[4:5,0:2])
ax5.axis('off')
axc1 = ax5.inset_axes([0, 0.7, 0.9, 0.15])
axc2 = ax5.inset_axes([0, -0.1, 0.9, 0.15])
cb0 = plt.colorbar(im0, cax=axc1, label='SMB (kg m$^{-2}$ yr$^{-1}$)', orientation='horizontal', extend='both',
                   ticks = [0, 25,50, 100, 200, 400, 800])
cb0.ax.tick_params(rotation=45)
for t in cb0.ax.get_xticklabels():
    t.set_fontsize(10)
cb1 = plt.colorbar(im1, cax=axc2, label='Mean basal melt rate\n(m yr$^{-1}$)', orientation='horizontal', extend='max',
                   ticks = [0,0.1, 0.5, 1, 2, 5, 10, 50, 100])
cb1.ax.tick_params(rotation=45)
for t in cb1.ax.get_xticklabels():
    t.set_fontsize(10)

# add labels
labels = 'abcdefghijklmnopqrst'
for i, ax in enumerate([ax0, ax1, ax2, ax7, ax4, ax3, ax6]):
    ax.text(0.01, 0.9, '({})'.format(labels[i]), transform=ax.transAxes)

## try adding scale bars
for i, ax in enumerate([ax0, ax1, ax2, ax7, ax4, ax3, ax6]):
    ax.plot([20, 32.5], [10,10], 'k-', linewidth=4)

plt.savefig(savedir + 'supplementary_fig03.png', dpi=dpi, bbox_inches='tight')


################################ Supplementary figure 4 ################################################################
def add_melt_diff_subplot(ax, array, x0, x1, y0, y1, boundcol, edgelinestyle, mdiff_bs):
    mdiff_norm = BoundaryNorm(mdiff_bs, ncolors=256)
    im = ax.imshow(array[y0:y1,x0:x1], origin='lower',
                   cmap=cmocean.cm.balance, norm=mdiff_norm)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(color=boundcol, labelcolor=boundcol)
    for spine in ax.spines.values():
        spine.set_edgecolor(boundcol)
        spine.set_linestyle(edgelinestyle)

    return im, ax

# read in regridded Adusumilli basal melt rates
obs_melt = xr.open_dataset(netcdf_dir + 'Adsumilli_melt_rate_regridded_8km_761.nc')

model_melt = np.ma.masked_where(np.max(sftflf.sftflf.data, axis=0)==0, -avmelt)

melt_field = obs_melt.wb - model_melt

# set colour bounds
mdiff_bs = [-20,-10,-5, -2, -1, 0, 1, 2, 5, 10, 20]

fig = plt.figure(figsize=(8,8))
gs = GridSpec(6, 6, figure=fig, hspace=0.05, wspace=0.05)
ax0 = fig.add_subplot(gs[0:2,0:2])
im0, ax0 = add_melt_subplot(ax0, obs_melt.wb, None, None, None, None, 'k', '-')
add_contour_smb_bmb(ax0, sftgrf.sftgrf.sel(time=2015), sftgif.sftgif.sel(time=2015), None, None, None, None, 'k')

ax1 =  fig.add_subplot(gs[0:2,2:4])
im1, ax1 = add_melt_diff_subplot(ax1, melt_field, None, None, None, None, 'k', '-', mdiff_bs)
add_contour_smb_bmb(ax1, sftgrf.sftgrf.sel(time=2015), sftgif.sftgif.sel(time=2015), None, None, None, None, 'k')

ax2 = fig.add_subplot(gs[0:2,4:6])
im2, ax2 = add_melt_diff_subplot(ax2, melt_field, 180, 330, 390, 520, 'y', '-', mdiff_bs)
add_contour_smb_bmb(ax2, sftgrf.sftgrf.sel(time=2015), sftgif.sftgif.sel(time=2015), 180, 330, 390, 520, 'k')

ax3 = fig.add_subplot(gs[2:4,4:6])
im3, ax3 = add_melt_diff_subplot(ax3, melt_field, 290, 440, 190, 330, 'b', '-', mdiff_bs)
add_contour_smb_bmb(ax3, sftgrf.sftgrf.sel(time=2015), sftgif.sftgif.sel(time=2015), 290, 440, 190, 330, 'k')

ax4 = fig.add_subplot(gs[2:5,2:4])
im4, ax4 = add_melt_diff_subplot(ax4, melt_field, 120, 220, 220, 370, 'r', '-', mdiff_bs)
add_contour_smb_bmb(ax4, sftgrf.sftgrf.sel(time=2015), sftgif.sftgif.sel(time=2015), 120, 220, 220, 370, 'k')

ax6 = fig.add_subplot(gs[4:5,4:6])
im6, ax6 = add_melt_diff_subplot(ax6, melt_field, 280, 550, 530, 680, 'g', '-', mdiff_bs)
add_contour_smb_bmb(ax6, sftgrf.sftgrf.sel(time=2015), sftgif.sftgif.sel(time=2015), 280, 550, 530, 680, 'k')

ax7 = fig.add_subplot(gs[2:4,0:2])
im7, ax7 = add_melt_diff_subplot(ax7, melt_field, 645, 690, 210, 270, 'm', '-', mdiff_bs)
add_contour_smb_bmb(ax7, sftgrf.sftgrf.sel(time=2015), sftgif.sftgif.sel(time=2015), 645, 690, 210, 270, 'k')

# add bounding boxes onto melt plot
add_bounding_box(ax1, 0, 290, 440, 190, 330, 'b', '-')
add_bounding_box(ax1, 0, 120, 220, 220, 370, 'r', '-')
add_bounding_box(ax1, 0, 280, 550, 530, 680, 'g', '-')
add_bounding_box(ax1, 0, 180, 330, 390, 520,  'y', '-')
add_bounding_box(ax1, 0, 645, 690, 210, 270,  'm', '-')

ax5 = fig.add_subplot(gs[4:5,0:2])
ax5.axis('off')
axc1 = ax5.inset_axes([0, 0.75, 0.9, 0.15])
axc2 = ax5.inset_axes([0, -0.3, 0.9, 0.15])
cb0 = plt.colorbar(im0, cax=axc1, label='Observed melt rate \n(m yr$^{-1}$)', orientation='horizontal',
                   extend='max',
                   ticks = [0,0.1, 0.5, 1, 2, 5, 10, 50, 100])
cb0.ax.tick_params(rotation=45)
for t in cb0.ax.get_xticklabels():
    t.set_fontsize(10)
cb1 = plt.colorbar(im1, cax=axc2, label='Observed - model \n(m yr$^{-1}$)', orientation='horizontal',
                   extend='both',
                   ticks = mdiff_bs)
cb1.ax.tick_params(rotation=45)
for t in cb1.ax.get_xticklabels():
    t.set_fontsize(10)

# add subplot labels
labels = 'abcdefghijklmnopqrst'
for i, ax in enumerate([ax0, ax1, ax2, ax7, ax4, ax3, ax6]):
    ax.text(0.01, 0.9, '({})'.format(labels[i]), transform=ax.transAxes)

## try adding scale bars
for i, ax in enumerate([ax0, ax1, ax2, ax7, ax4, ax3, ax6]):
    ax.plot([20, 32.5], [10,10], 'k-', linewidth=4)

plt.savefig(savedir + 'supplementary_fig04.png', dpi=dpi, bbox_inches='tight')


################################ Supplementary figure 5 ################################################################

ctrl = xr.open_dataset(netcdf_root.format('ctrl', 'limnsw', 'ctrl'))
sorted_exps = {}
for i, exp in enumerate(AllExps):
    limnsw = xr.open_dataset(netcdf_root.format(exp, 'limnsw', exp))
    y = -((limnsw.sel(time=slice(2015,2100)).limnsw
           - limnsw.sel(time=2015).limnsw)/1e12)/362.5
    sorted_exps[exp]=np.round(y[-1].data, 2)

All_exps_ordered = sorted(sorted_exps, key=sorted_exps.get)
# plot sorted
fig = plt.figure(figsize=(9,11))
gs = gridspec.GridSpec(5, 4, hspace=0.1, wspace=0.1)#, width_ratios=[1,1,1,0.1])

def transform_y(OldValue, OldMax = 60, OldMin = -130, NewMax = 768, NewMin = 0):
    OldRange = (OldMax - OldMin)
    NewRange = (NewMax - NewMin)
    NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    return NewValue

vming=-150
vmaxg=50
vminf=-400
vmaxf=50
divnorm = TwoSlopeNorm(vmin=vming, vcenter=0, vmax=vmaxg)
divnorm2 = TwoSlopeNorm(vmin=vminf, vcenter=0, vmax=vmaxf)
lithk_root = netcdf_dir + '/ismip6_{}*/lithk_AIS_CPOM-LBL_BISICLES_ismip6_{}*.nc'
gl_root = netcdf_dir + '/ismip6_{}*/sftgrf_AIS_CPOM-LBL_BISICLES_ismip6_{}*.nc'
row = [0, 0, 0,1, 1, 1, 2, 2, 2, 3,3,3,4,4,4,5,5,5]
col = [0,1,2,0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2]
row = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5]
col = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
labels = 'abcdefghijklmnopqrstuvwxyz'
ind = zip(row, col)
TScol = 'fuchsia'
TScol = 'r'

flcmap = cmc.bam
grcmap = cmc.broc_r

ctrl = xr.open_dataset(netcdf_root.format('ctrl', 'limnsw', 'ctrl'))
for i, exp in enumerate(All_exps_ordered[::-1]):
    ax = plt.subplot(gs[row[i], col[i]])
    axin = ax.inset_axes([0.0, 0.0, 1, 1])
    axin.patch.set_alpha(0.0)
    gldat = xr.open_dataset(netcdf_root.format(exp, 'sftgrf', exp))
    data = xr.open_dataset(netcdf_root.format(exp, 'lithk', exp))
    shelf_mask = np.where(data.lithk.isel(time=0) > 0, 1, np.zeros_like(data.lithk.isel(time=0)).data)
    diff = data.lithk.sel(time=2100) - data.lithk.sel(time=2015)
    gl_last = np.round(gldat.sftgrf.isel(time=-1), 0)
    gl_first = np.round(gldat.sftgrf.isel(time=0), 0)
    field2 = np.ma.masked_where(gl_last==1, np.where(shelf_mask==1, diff, np.ones_like(diff)*np.nan))
    im1 = ax.imshow(field2, origin='lower', norm=divnorm2, cmap=flcmap) #cmap=cmocean.cm.tarn)
    im = ax.imshow(np.ma.masked_where(gl_first==0, diff), origin='lower', norm=divnorm, cmap=grcmap)
    ax.contour(gl_last, colors='k', linestyles=':', linewidths=0.3)
    ax.contour(gl_last, colors='k', linestyles='-', linewidths=0.3)
    # add shelf contour
    ax.contour(shelf_mask, colors='k', linestyles='--', linewidths=0.3)
    ax.axis('off')
    limnsw = xr.open_dataset(netcdf_root.format(exp, 'limnsw', exp))
    y = -((limnsw.sel(time=slice(2015,2100)).limnsw #- ctrl.sel(time=slice(2015,2100)).limnsw
           - limnsw.sel(time=2015).limnsw)/1e12)/362.5
    axin.plot(y.time, y, color=TScol, linewidth=2, linestyle=':', zorder=0)
    axin.set_facecolor(None)
    #axin.set_ylim(-60, 130)
    axin.set_ylim(-10, 190)
    axin.spines['bottom'].set_color(TScol)
    axin.spines['left'].set_color(TScol)
    axin.tick_params(axis='x', colors=TScol)
    axin.tick_params(axis='y', colors=TScol)
    axin.yaxis.label.set_color(TScol)
    axin.xaxis.label.set_color(TScol)
    axin.spines['right'].set_visible(False)
    axin.spines['top'].set_visible(False)
    if len(exp) == 1:
        axin.annotate('Exp0{}'.format(exp), xy = (2070, -10), size=8, weight='bold')
    else:
        axin.annotate('Exp{}'.format(exp), xy = (2070, -10), size=8, weight='bold')
    annotation = '{},\n{},\n{},\nCollapse {}'.format(Exps_dict[exp]['gcm'], Exps_dict[exp]['scenario'], Exps_dict[exp]['gamma0pcntile'], Exps_dict[exp]['collapse'])
    axin.annotate(annotation, xy = (2015, -10), size=7, color='k')
    if labels[i] in ['a', 'e', 'i', 'm']:
        axin.xaxis.set_ticklabels([])
    elif labels[i] in ['r', 'o', 'p']:
        axin.yaxis.set_ticklabels([])
    elif labels[i] in ['q']:
        print('keep both labels')
    else:
        axin.xaxis.set_ticklabels([])
        axin.yaxis.set_ticklabels([])

    ax.annotate('2100 SLC:\n{} mm'.format(str(int(np.round(y[-1].data, 0)))), (500, 670), c=TScol, size=7)
    ax.annotate('({})'.format(labels[i]), xy=(10, 670))
    sorted_exps[exp]=np.round(y[-1].data, 2)
cbax = plt.subplot(gs[4, 2:4])
cbax.axis('off')
cbar_ax = cbax.inset_axes([0.15, 0.7, 0.6, 0.15])
fig.colorbar(im, cax=cbar_ax, orientation='horizontal', extend='both', label='Grounded ice thickness change (m)',
             ticks = [vming, 0, vmaxg])

cbar_ax2 = cbax.inset_axes([0.15, 0.2, 0.6, 0.15])
fig.colorbar(im1, cax=cbar_ax2, orientation='horizontal', extend='both', label='Floating ice thickness change (m)',
             ticks = [vminf, 0, vmaxf])


fig.text(0.4, 0.06, 'Year', ha='center', va='center', c=TScol)
fig.text(0.06, 0.5, 'Sea level contribution (mm)', ha='center', va='center', rotation='vertical', c=TScol)
plt.savefig(savedir + 'supplementary_fig05.png',
            dpi=dpi, bbox_inches='tight')
