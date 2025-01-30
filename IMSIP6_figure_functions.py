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

################################### Figure 1 ###########################################

def calc_temporal_depth_mean(TF_dir, model, scenario, floor):
    TF_root = TF_dir + '{}*{}*thermal_forcing_8km_x_60m.nc'.format(model, scenario)
    dataset = xr.open_dataset(glob.glob(TF_root)[0], decode_times=False)
    thermal_forcing = dataset.thermal_forcing.sel(z=slice(0,floor)).isel(time=slice(10,-1))
    return thermal_forcing.mean('time').mean('z')

def annual_spatial_mean_500m(TF_dir, model, scenario, t0, mask):
    TF_root = TF_dir + '{}*{}*thermal_forcing_8km_x_60m.nc'.format(model, scenario)
    dset = xr.open_dataset(glob.glob(TF_root)[0])
    av_tf = []
    surface_tf = dset.sel(z=slice(0, -500))
    Mask = np.where(mask==0, np.nan, mask)
    for i in range(t0, 106):
        masked = surface_tf.thermal_forcing.isel(time=i).data * Mask
        mean = np.nanmean(masked)
        av_tf.append(mean)
    return np.asarray(av_tf)

def figure1(TF_dir, netcdf_dir, savedir, dpi):
    # If thermal forcing data downloaded, plot, skip if not
    if len(glob.glob(TF_dir + '*.nc')) >= 6:
        print('Making figure 1')
        TF_root = TF_dir + '{}*{}*thermal_forcing_8km_x_60m.nc'
        models = ['CNRM-CM6-1', 'NorESM1-M', 'CCSM4', 'MIROC-ESM-CHEM']
        scenarios = ['RCP26', 'RCP85', 'ssp126', 'ssp585']

        # generate means
        miroc_rcp85 = calc_temporal_depth_mean(TF_dir, 'MIROC-ESM-CHEM', 'RCP85', -500)
        ccsm_rcp85 = calc_temporal_depth_mean(TF_dir, 'CCSM4', 'RCP85', -500)
        noresm_rcp85 = calc_temporal_depth_mean(TF_dir, 'NorESM1-M', 'RCP85', -500)
        cnrm_ssp585 = calc_temporal_depth_mean(TF_dir, 'CNRM-CM6-1', 'ssp585', -500)

        noresm_rcp26 = calc_temporal_depth_mean(TF_dir, 'NorESM1-M', 'RCP26', -500)
        cnrm_ssp126 = calc_temporal_depth_mean(TF_dir, 'CNRM-CM6-1', 'ssp126', -500)

        #read in bedmap
        bedmap_8km = xr.open_dataset(netcdf_dir + 'bedmap2_8km.nc')
        grounded = bedmap_8km.icemask_grounded
        shelves = bedmap_8km.icemask_shelves
        cmap=cmo.thermal

        levels=np.linspace(0, 5, 11)
        contC='w'
        cmap=cmocean.cm.thermal
        lightcmap = cmocean.tools.lighten(cmap, 0.5)
        vmin=0
        vmax=5
        lw = 0.3
        tc = 'w'
        mirocCol='olivedrab'
        ccsmCol='mediumorchid'
        noresmCol='cornflowerblue'
        cnrmCol='lightcoral'
        size = 5
        fig = plt.figure(figsize=((8/3)*size,size))
        gs = gridspec.GridSpec(2, 4,
                               wspace=0.1, hspace=0.0, top=0.95, bottom=0.05, left=0.17, right=0.845)

        ax0 = plt.subplot(gs[0,0])
        ax0.contourf(miroc_rcp85, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap)
        ax0.contour(grounded, colors=contC, linewidths=0.4)
        ax0.contour(shelves, colors=contC, linewidths=0.4, linestyles=':')
        ax0.annotate('RCP8.5', (100, 600), fontsize=8, c=tc)
        ax0.annotate('MIROC-ESM-CHEM', (170,700), fontsize=8, c=tc, weight='bold')
        ax0.annotate('(a)', (10, 700))

        ax1 = plt.subplot(gs[0,1])
        ax1.contourf(ccsm_rcp85, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap)
        ax1.contour(grounded, colors=contC, linewidths=lw)
        ax1.contour(shelves, colors=contC, linewidths=lw, linestyles=':')
        ax1.annotate('RCP8.5', (100, 600), fontsize=8, c=tc)
        ax1.annotate('CCSM4', (320,700), fontsize=8, c=tc, weight='bold')
        ax1.annotate('(b)', (10, 700))

        ax2 = plt.subplot(gs[0,2])
        ax2.contourf(noresm_rcp85, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap)
        ax2.contour(grounded, colors=contC, linewidths=lw)
        ax2.contour(shelves, colors=contC, linewidths=lw, linestyles=':')
        ax2.annotate('RCP8.5', (100, 600), fontsize=8, c=tc)
        ax2.annotate('NorESM1-M', (260,700), fontsize=8, c=tc, weight='bold')
        ax2.annotate('(c)', (10, 700))

        ax3 = plt.subplot(gs[0,3])
        ax3.contourf(cnrm_ssp585, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap)
        ax3.contour(grounded, colors=contC, linewidths=lw)
        ax3.contour(shelves, colors=contC, linewidths=lw, linestyles=':')
        ax3.annotate('SSP585', (100, 600), fontsize=8, c=tc)
        ax3.annotate('CNRM-CM6-1', (250,700), fontsize=8, c=tc, weight='bold')
        ax3.annotate('(d)', (10, 700))

        ax4 = plt.subplot(gs[1,2])
        ax4.contourf(noresm_rcp26, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap)
        ax4.contour(grounded, colors=contC, linewidths=lw)
        ax4.contour(shelves, colors=contC, linewidths=lw, linestyles=':')
        ax4.annotate('RCP2.6', (100, 600), fontsize=8, c=tc)
        ax4.annotate('NorESM1-M', (260,700), fontsize=8, c=tc, weight='bold')
        ax4.annotate('(f)', (10, 700))

        ax5 = plt.subplot(gs[1,3])
        im = ax5.contourf(cnrm_ssp126, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap)
        ax5.contour(grounded, colors=contC, linewidths=lw)
        ax5.contour(shelves, colors=contC, linewidths=lw, linestyles=':')
        ax5.annotate('SSP126', (100, 600), fontsize=8, c=tc)
        ax5.annotate('CNRM-CM6-1', (250,700), fontsize=8, c=tc, weight='bold')
        ax5.annotate('(g)', (10, 700))

        for i in range(2):
            for j in range(4):
                plt.subplot(gs[i, j]).axis('off')
                plt.subplot(gs[i, j]).set_aspect(1)

        #ax6 = plt.subplots(gs[1,:2])
        ax6 = fig.add_subplot(gs[1, :2])

        mask = bedmap_8km.icemask_shelves
        ax6.plot(np.arange(2015, 2101), annual_spatial_mean_500m(TF_dir, 'MIROC-ESM-CHEM', 'RCP85',20,mask), c=mirocCol,label='MIROC-ESM-CHEM, RCP8.5')
        ax6.plot(np.arange(2015, 2101), annual_spatial_mean_500m(TF_dir, 'CCSM4', 'RCP85',20,mask), c=ccsmCol,label='CCSM4, RCP8.5')
        ax6.plot(np.arange(2015, 2101), annual_spatial_mean_500m(TF_dir, 'NorESM1-M', 'RCP85',20,mask),c=noresmCol, label='NorESM1-M, RCP8.5')
        ax6.plot(np.arange(2015, 2101), annual_spatial_mean_500m(TF_dir, 'NorESM1-M', 'RCP26',20,mask), c=noresmCol, linestyle=':', label='NorESM1-M, RCP2.6')
        ax6.plot(np.arange(2015, 2101), annual_spatial_mean_500m(TF_dir, 'CNRM-CM6-1', 'ssp585',20,mask),c=cnrmCol, label='CNRM-CM6-1, SSP585')
        ax6.plot(np.arange(2015, 2101), annual_spatial_mean_500m(TF_dir, 'CNRM-CM6-1', 'ssp126',20,mask),c=cnrmCol, linestyle=':', label='CNRM-CM6-1, SSP126')
        ax6.legend(loc='upper left', ncol=2, fontsize='small')
        ax6.tick_params(axis='both', which='major', labelsize=10)
        ax6.tick_params(axis='both', which='minor', labelsize=10)
        ax6.set_xlabel('Year', size=12)
        ax6.set_ylabel('Thermal forcing (K)', size=12)
        ax6.text(0, 1.01, '(e)', transform=ax6.transAxes)

        line = plt.Line2D([0.68,0.68],[0.2,0.8], transform=fig.transFigure, color="black")
        fig.add_artist(line)
        cax = plt.axes([0.85, 0.2, 0.02, 0.6])
        plt.colorbar(im, cax=cax, label='Thermal forcing (K)', ticklocation='right', extend='max')
        #fig.suptitle('Average surface 500m thermal forcing, 2015-2100 average')
        plt.savefig(savedir + 'fig01.png', dpi=dpi,
                    bbox_inches='tight')

    else:
        print('Thermal forcing data not accessible, download from ISMIP6')


################################### Figure 2 ###########################################################################

def average_smb_anomaly(SMB_dir, model, scenario, t0):
    smb_root = SMB_dir + '{}*{}*.nc'.format(model, scenario)
    print(smb_root)
    dataset = xr.open_dataset(glob.glob(smb_root)[0], decode_times=False)
    dataset = dataset.isel(time=slice(t0, 106))
    average_smb_anom = dataset.smb_anomaly.mean('time')
    return average_smb_anom

def annual_mean_smb_anom(SMB_dir, model, scenario, t0, mask):
    smb_root = SMB_dir + '{}*{}*.nc'.format(model, scenario)
    dset = xr.open_dataset(glob.glob(smb_root)[0])
    av_tf = []
    Mask = np.where(mask==0, np.nan, mask)
    for i in range(t0, 106):
        masked = dset.smb_anomaly.isel(time=i).data * Mask
        mean = np.nanmean(masked)
        av_tf.append(mean)
    return np.asarray(av_tf)

import matplotlib as mpl
class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))

def figure2(SMB_dir, netcdf_dir, savedir, dpi):
    # if SMB data has been downloaded, plot, skip if not
    if len(glob.glob(SMB_dir  + '*.nc')) >= 6:
        print('Making figure 2')
        # get spatial plots
        miroc_rcp85_smb = average_smb_anomaly(SMB_dir, 'MIROC-ESM-CHEM', 'rcp85', 20)
        ccsm_rcp85_smb = average_smb_anomaly(SMB_dir, 'CCSM4', 'rcp85', 20)
        noresm_rcp85_smb= average_smb_anomaly(SMB_dir, 'NorESM1-M', 'rcp85', 20)
        cnrm_ssp585_smb = average_smb_anomaly(SMB_dir, 'CNRM', 'ssp585', 20)

        noresm_rcp26_smb = average_smb_anomaly(SMB_dir, 'NorESM1-M', 'rcp26', 20)
        cnrm_ssp126_smb = average_smb_anomaly(SMB_dir, 'CNRM', 'ssp126', 20)
        # get timeseries
        # use bedmap grounded mask
        bedmap_8km = xr.open_dataset(netcdf_dir + 'bedmap2_8km.nc')
        GroundedMask = bedmap_8km.icemask_grounded

        miroc_rcp85_smb_TS = annual_mean_smb_anom(SMB_dir, 'MIROC-ESM-CHEM', 'rcp85', 20, GroundedMask)
        ccsm_rcp85_smb_TS = annual_mean_smb_anom(SMB_dir, 'CCSM4', 'rcp85', 20, GroundedMask)
        noresm_rcp85_smb_TS = annual_mean_smb_anom(SMB_dir, 'NorESM1-M', 'rcp85', 20, GroundedMask)
        cnrm_ssp585_smb_TS = annual_mean_smb_anom(SMB_dir, 'CNRM', 'ssp585', 20, GroundedMask)

        noresm_rcp26_smb_TS = annual_mean_smb_anom(SMB_dir, 'NorESM1-M', 'rcp26', 20, GroundedMask)
        cnrm_ssp126_smb_TS = annual_mean_smb_anom(SMB_dir, 'CNRM', 'ssp126', 20, GroundedMask)

        # masks for contours #
        grounded = bedmap_8km.icemask_grounded
        shelves = bedmap_8km.icemask_shelves

        contC='k'
        cmap=cmo.balance_r
        lightcmap = cmocean.tools.lighten(cmap, 0.5)
        vmin=-500 #/ 1e12
        vmax=250 #/1e12
        levels=11

        norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)

        lw = 0.3
        conversion = 365.24*24*60*60 #/1e12
        tc = 'k'
        mirocCol='olivedrab'
        ccsmCol='mediumorchid'
        noresmCol='cornflowerblue'
        cnrmCol='lightcoral'
        size = 5
        fig = plt.figure(figsize=((8/3)*size,size))
        gs = gridspec.GridSpec(2, 4,
                               wspace=0.1, hspace=0.1, top=0.95, bottom=0.05, left=0.17, right=0.845)

        ax0 = plt.subplot(gs[0,0])
        ax0.contourf(miroc_rcp85_smb*conversion, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)
        ax0.contour(grounded, colors=contC, linewidths=0.4)
        ax0.contour(shelves, colors=contC, linewidths=0.4, linestyles=':')
        ax0.annotate('RCP8.5', (100, 600), fontsize=8, c=tc, weight='bold')
        ax0.annotate('MIROC-ESM-CHEM', (170,700), fontsize=8, c=tc, weight='bold')
        ax0.annotate('(a)', (10, 700))

        ax1 = plt.subplot(gs[0,1])
        ax1.contourf(ccsm_rcp85_smb*conversion, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)
        ax1.contour(grounded, colors=contC, linewidths=lw)
        ax1.contour(shelves, colors=contC, linewidths=lw, linestyles=':')
        ax1.annotate('RCP8.5', (100, 600), fontsize=8, c=tc, weight='bold')
        ax1.annotate('CCSM4', (320,700), fontsize=8, c=tc, weight='bold')
        ax1.annotate('(b)', (10, 700))

        ax2 = plt.subplot(gs[0,2])
        ax2.contourf(noresm_rcp85_smb*conversion, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)
        ax2.contour(grounded, colors=contC, linewidths=lw)
        ax2.contour(shelves, colors=contC, linewidths=lw, linestyles=':')
        ax2.annotate('RCP8.5', (100, 600), fontsize=8, c=tc, weight='bold')
        ax2.annotate('NorESM1-M', (260,700), fontsize=8, c=tc, weight='bold')
        ax2.annotate('(c)', (10, 700))

        ax3 = plt.subplot(gs[0,3])
        ax3.contourf(cnrm_ssp585_smb*conversion, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)
        ax3.contour(grounded, colors=contC, linewidths=lw)
        ax3.contour(shelves, colors=contC, linewidths=lw, linestyles=':')
        ax3.annotate('SSP585', (100, 600), fontsize=8, c=tc, weight='bold')
        ax3.annotate('CNRM-CM6-1', (250,700), fontsize=8, c=tc, weight='bold')
        ax3.annotate('(d)', (10, 700))

        ax4 = plt.subplot(gs[1,2])
        ax4.contourf(noresm_rcp26_smb*conversion, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)
        ax4.contour(grounded, colors=contC, linewidths=lw)
        ax4.contour(shelves, colors=contC, linewidths=lw, linestyles=':')
        ax4.annotate('RCP2.6', (100, 600), fontsize=8, c=tc, weight='bold')
        ax4.annotate('NorESM1-M', (260,700), fontsize=8, c=tc, weight='bold')
        ax4.annotate('(f)', (10, 700))

        ax5 = plt.subplot(gs[1,3])
        im = ax5.contourf(cnrm_ssp126_smb*conversion, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)
        ax5.contour(grounded, colors=contC, linewidths=lw)
        ax5.contour(shelves, colors=contC, linewidths=lw, linestyles=':')
        ax5.annotate('SSP126', (100, 600), fontsize=8, c=tc, weight='bold')
        ax5.annotate('CNRM-CM6-1', (250,700), fontsize=8, c=tc, weight='bold')
        ax5.annotate('(g)', (10, 700))

        for i in range(2):
            for j in range(4):
                plt.subplot(gs[i, j]).axis('off')
                plt.subplot(gs[i, j]).set_aspect(1)


        # plot as Gt/yr
        multiplier = (np.sum(grounded.data) * 8000 * 8000 * 365.24*24*60*60)/1e12
        ax6 = fig.add_subplot(gs[1, :2])
        ax6.plot(np.arange(2015, 2101), miroc_rcp85_smb_TS*multiplier, c=mirocCol,label='MIROC-ESM-CHEM, RCP8.5')
        ax6.plot(np.arange(2015, 2101), ccsm_rcp85_smb_TS*multiplier, c=ccsmCol,label='CCSM4, RCP8.5')
        ax6.plot(np.arange(2015, 2101), noresm_rcp85_smb_TS*multiplier,c=noresmCol, label='NorESM1-M, RCP8.5')
        ax6.plot(np.arange(2015, 2101), noresm_rcp26_smb_TS*multiplier, c=noresmCol, linestyle=':', label='NorESM1-M, RCP2.6')
        ax6.plot(np.arange(2015, 2101), cnrm_ssp585_smb_TS*multiplier,c=cnrmCol, label='CNRM-CM6-1, SSP585')
        ax6.plot(np.arange(2015, 2101), cnrm_ssp126_smb_TS*multiplier,c=cnrmCol, linestyle=':', label='CNRM-CM6-1, SSP126')
        ax6.set_ylim(-250, 1400)
        ax6.legend(loc='upper left', ncol=2, fontsize='small')
        ax6.tick_params(axis='both', which='major', labelsize=10)
        ax6.tick_params(axis='both', which='minor', labelsize=10)
        ax6.set_xlabel('Year', size=12)
        ax6.set_ylabel('Average SMB anomaly\n grounded (Gt yr$^{-1}$)', size=12)
        ax6.text(0, 1.01, '(e)', transform=ax6.transAxes)

        line = plt.Line2D([0.68,0.68],[0.2,0.8], transform=fig.transFigure, color="black")
        fig.add_artist(line)
        cax = plt.axes([0.85, 0.2, 0.02, 0.6])
        cb = plt.colorbar(im, cax=cax, ticklocation='right', extend='max')
        cb.set_label(label='Average SMB anomaly (Kg m$^{-2}$ yr$^{-1}$)', fontsize=12)
        plt.savefig(savedir + 'fig02.png', dpi=dpi, bbox_inches='tight')

    else:
        print('Surface mass balance data not accessible, download from ISMIP6')

#################################### RESULTS PLOTS #####################################################################
###################################### Figure 3 ########################################################################
def get_speed(year, xvelmean, yvelmean):
    speed = np.sqrt(np.power(xvelmean.xvelmean.sel(time=year), 2) + np.power(yvelmean.yvelmean.sel(time=year), 2))
    return speed

def add_speed_subplot(ax, array, x0, x1, y0, y1, boundcol, mask, edgelinestyle):
    speeddivnorm=TwoSlopeNorm(vmin=-500, vcenter=0, vmax=500)
    im = ax.imshow(np.ma.masked_where(mask[y0:y1,x0:x1]<0.1,
                                      array[y0:y1,x0:x1]), origin='lower', cmap='seismic', norm=speeddivnorm)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(color=boundcol, labelcolor=boundcol)
    for spine in ax.spines.values():
        spine.set_edgecolor(boundcol)
        spine.set_linestyle(edgelinestyle)

    return im, ax

def add_thck_subplot(ax, array, x0, x1, y0, y1, boundcol, mask, edgelinestyle):
    divnorm=TwoSlopeNorm(vmin=-200, vcenter=0, vmax=50)
    im = ax.imshow(np.ma.masked_where(mask[y0:y1,x0:x1]<0.1,
                                      array[y0:y1,x0:x1]), origin='lower',
                   cmap=cmocean.cm.balance_r, norm=divnorm)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(color=boundcol, labelcolor=boundcol)
    for spine in ax.spines.values():
        spine.set_edgecolor(boundcol)
        spine.set_linestyle(edgelinestyle)

    return im, ax

def add_contours(ax, array, array2, x0, x1, y0, y1, glc, sec, array3):
    ax.contour(array[y0:y1,x0:x1], linewidths=0.3, colors=glc)
    ax.contour(array2[y0:y1,x0:x1], linewidths=0.3, colors=glc, linestyles='--')
    # e.g. array 2 sftgrf.sftgrf.isel(time=90).astype(int)
    ax.contour(array3[y0:y1,x0:x1], linewidths=0.3, colors=sec)

def add_bounding_box(ax, rot, x0, x1, y0, y1, bcol, ls):
    xx, yy = np.meshgrid(np.arange(0,761), np.arange(0,761))
    xx, yy = rotate(xx,rot), rotate(yy, rot)
    xll, yll = xx[y0:y1,x0:x1][0, 0], yy[y0:y1,x0:x1][0,0]
    xul, yul = xx[y0:y1,x0:x1][-1, 0], yy[y0:y1,x0:x1][-1, 0]
    xur, yur = xx[y0:y1,x0:x1][-1, -1], yy[y0:y1,x0:x1][-1,-1]
    # issue with bottom right rotating out of frame
    xlr = xur + abs(xul-xll)
    ylr = yur - (yul-yll)
    ax.plot([xll, xul, xur, xlr, xll], [yll, yul, yur, ylr, yll],bcol,linestyle=ls)


def figure3(netcdf_root, savedir, dpi):
    print('Making figure 3')
    # open velocity data & thickness
    xvelmean = xr.open_dataset(netcdf_root.format('ctrl', 'xvelmean', 'ctrl'))
    yvelmean = xr.open_dataset(netcdf_root.format('ctrl', 'yvelmean', 'ctrl'))
    lithk = xr.open_dataset(netcdf_root.format('ctrl', 'lithk', 'ctrl'))

    # open ice mask and grounding line mask
    sftgrf = xr.open_dataset(netcdf_root.format('ctrl', 'sftgrf', 'ctrl'))
    sftgif = xr.open_dataset(netcdf_root.format('ctrl', 'sftgif', 'ctrl'))

    # change fields for plotting
    delthck = lithk.lithk.sel(time=2100) - lithk.lithk.sel(time=2015)
    delspeed = get_speed(2100, xvelmean, yvelmean) - get_speed(2015, xvelmean, yvelmean)

    # mass above floatation
    limnsw = xr.open_dataset(netcdf_root.format('ctrl', 'limnsw', 'ctrl'))

    gl0 = sftgrf.sftgrf.sel(time=2015).astype(int)
    gl1 = sftgrf.sftgrf.sel(time=2100).astype(int)
    shlf = np.where(sftgif.sftgif.isel(time=0)==0, 0, np.ones_like(sftgif.sftgif.isel(time=0)))
    glc = 'k'
    sec = 'm'
    fig = plt.figure(figsize=(10,10))
    gs = GridSpec(5, 3, figure=fig, hspace=0.05, wspace=0.05)#, height_ratios=[1,1,0.6], width_ratios=[0.1,1,1,1,0.1])
    ax0 = fig.add_subplot(gs[0:2,0])
    im0, ax0 = add_thck_subplot(ax0, delthck, None,  None,  None,  None, 'k', sftgif.sftgif.isel(time=0), None)
    add_contours(ax0, gl0, gl1, None,  None,  None,  None, glc, sec, shlf)
    ax0.annotate('DML',xy=(500,620),xytext=(500,700),arrowprops={'arrowstyle':'-|>', 'color':'k'}
                 ,horizontalalignment='center', fontsize=12)
    ax0.annotate('DML',xy=(500,620),xytext=(500,700),arrowprops={'arrowstyle':'-|>', 'color':'k'}
                 ,horizontalalignment='center', fontsize=12)
    ax0.annotate('WIS',xy=(680,450),xytext=(680,700),arrowprops={'arrowstyle':'-|>', 'color':'k'}
                 ,horizontalalignment='center', fontsize=12)

    ax1 = fig.add_subplot(gs[0,1])
    im1, ax1 = add_thck_subplot(ax1, delthck, 500, 700, 400, 500, 'brown', sftgif.sftgif.isel(time=0), 'dotted')
    add_contours(ax1, gl0, gl1, 500, 700, 400, 500, glc, sec, shlf)
    ax1.annotate('AmIS',xy=(150,67),xytext=(110,26),arrowprops={'arrowstyle':'-|>', 'color':'k'}
                 ,horizontalalignment='center', fontsize=12)

    ax2 = fig.add_subplot(gs[1,1])
    im2, ax2 = add_thck_subplot(ax2, rotate(delthck, angle=30),400, 800, 180, 350,
                                'r', rotate(sftgif.sftgif.isel(time=0), angle=30), 'dotted')
    add_contours(ax2, rotate(gl0, 30), rotate(gl1, 30),400, 800, 180, 350, glc,  sec,
                 np.where(rotate(shlf, 30)<0.1, 0, np.ones_like(rotate(shlf, 30))))
    ax2.annotate('TotG',xy=(287,95),xytext=(183,119),arrowprops={'arrowstyle':'-|>', 'color':'k'}
                 ,horizontalalignment='center', fontsize=12)

    # add boxes
    add_bounding_box(ax0, 30, 400, 800, 180, 350, 'r', ':')
    add_bounding_box(ax0, 0, 500, 700, 400, 500, 'brown', ':')

    # speed plots
    ax3 = fig.add_subplot(gs[0:2,2])
    im3, ax3 = add_speed_subplot(ax3, delspeed, None,  None,  None,  None, 'k', sftgif.sftgif.isel(time=0), None)
    add_contours(ax3, gl0, gl1, None,  None,  None,  None, glc, sec, shlf)
    # add labels for RIS and FRIS
    ax3.annotate('RIS',xy=(350,200),xytext=(300,20),arrowprops={'arrowstyle':'-|>', 'color':'k'}
                 ,horizontalalignment='center', fontsize=12)
    ax3.annotate('FRIS',xy=(220,500),xytext=(150,700),arrowprops={'arrowstyle':'-|>', 'color':'k'}
                 ,horizontalalignment='center', fontsize=12)

    ax4 = fig.add_subplot(gs[2:4,-1])
    im4, ax4 = add_speed_subplot(ax4, delspeed, 300,  450,  189,  330, 'green', sftgif.sftgif.isel(time=0), 'dotted')
    add_contours(ax4, gl0, gl1, 300,  450,  189,  330, glc, sec, shlf)

    ax5 = fig.add_subplot(gs[2:4,1:-1])
    im5, ax5 = add_speed_subplot(ax5, delspeed, 130,  250,  250,  400, 'blue', sftgif.sftgif.isel(time=0), 'dotted')
    add_contours(ax5, gl0, gl1, 130,  250,  250,  400, glc, sec, shlf)
    ax5.annotate('PIG',xy=(52,109),xytext=(51,130),arrowprops={'arrowstyle':'-|>', 'color':'k'}
                 ,horizontalalignment='center', fontsize=12)
    ax5.annotate('ThwG',xy=(77,77),xytext=(102,77),arrowprops={'arrowstyle':'-|>', 'color':'k'}
                 ,horizontalalignment='center', fontsize=12)

    # add boxes
    add_bounding_box(ax3, 0, 300,  450,  189,  330, 'green', ':')
    add_bounding_box(ax3, 0, 130,  250,  250,  400, 'blue', ':')

    ax6 = fig.add_subplot(gs[3,0])
    limnsw = limnsw.sel(time=slice(2015, 2100))
    ax6.plot(limnsw.time, -(((limnsw.limnsw - limnsw.limnsw.sel(time=2015))/1e12)/362.5))
    ax6.set_xlabel('Year')
    ax6.set_ylabel('Sea level contribution\n(mm)')
    ax6.set_xlim(2010, 2100)

    # add colorbars
    ax7 = fig.add_subplot(gs[2,0])
    ax7.axis('off')
    axc1 = ax7.inset_axes([0, 1.0, 1, 0.2])
    axc2 = ax7.inset_axes([0, 0.4, 1, 0.2])
    plt.colorbar(im0, cax=axc1, label='Thickness change (m)', orientation='horizontal')
    plt.colorbar(im3, cax=axc2, label='Speed change (m yr$^{-1}$)', orientation='horizontal')

    # add subplot labels
    labels = 'abcdefghijklmnopqrst'
    for i, ax in enumerate([ax0, ax1, ax2, ax3, ax4, ax5, ax6]):
        ax.text(0.025, 0.9, '({})'.format(labels[i]), transform=ax.transAxes)

    ## add scale bars
    for i, ax in enumerate([ax0, ax1, ax2, ax3, ax4, ax5, ax6]):
        ax.plot([20, 32.5], [20,20], 'k-', linewidth=4)


    plt.savefig(savedir + 'fig03.png', dpi=dpi, bbox_inches='tight')

###################################### Figure 4 ########################################################################

def figure4(netcdf_root, savedir, dpi, **kwargs):
    print('Making figure 4')
    # load in exp dict
    var = 'acabf'
    Exps_dict = kwargs['Exps_dict']
    Exps_DF = pd.DataFrame.from_dict(Exps_dict)
    Exps_DFT = Exps_DF.T
    colors=kwargs['colors']
    Formatting_dict_allExps = kwargs['Formatting_dict_allExps']
    AllExps = ['5', '6', '7', '8', '9', '10', '12', '13', 'D52', 'D53', 'D55',
               'D56', 'D58', 'T71', 'T73', 'TD58', 'B6', 'B7']

    Formatting_dict_allExps = {'linestyles':{'RCP2.6':':', 'RCP8.5':'-', 'SSP126':'-.', 'SSP585':'--'},
                               'colors':{'NorESM1-M':'blue', 'CCSM4':'red', 'MIROC-ESM':'cyan', 'CNRM-CM6-1':'magenta'}}

    # SLC dict
    SLC_dict = {}
    # make dictionary of results
    #control
    ctrl = xr.open_dataset(netcdf_root.format('ctrl', 'limnsw', 'ctrl'))
    for i, exp in enumerate(AllExps):
        SLC_dict[exp]={}
        SLC_dict[exp]['SLC'] = {}
        data = xr.open_dataset(netcdf_root.format(exp, 'limnsw', exp))
        SLC = (data.sel(time=slice(2015,2100)).limnsw - data.sel(time=2015).limnsw)
        SLC = (SLC/1e12)/362.5
        SLC_dict[exp]['time'] = SLC.time.data
        SLC_dict[exp]['SLC'] = SLC.data

    SLC_dict['control']={}
    SLC_dict['control']['SLC']=((ctrl.sel(time=slice(2015, 2100)).limnsw - ctrl.sel(time=2015).limnsw)/1e12)/362.5
    SLC_dict['control']['time']=ctrl.sel(time=slice(2015, 2100)).time
    Formatting_dict_allExps = {'linestyles':{'RCP2.6':':', 'RCP8.5':'-', 'SSP126':'-.', 'SSP585':'--'},
                               'colors':{'NorESM1-M':'blue', 'CCSM4':'red', 'MIROC-ESM':'cyan', 'CNRM-CM6-1':'magenta'}}
    SLC_DF = pd.DataFrame.from_dict(SLC_dict)
    SLC_DFT = SLC_DF.transpose()

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    # plot control run
    ax.plot(SLC_DFT.loc['control'].time, -SLC_DFT.loc['control'].SLC, 'k-', marker='|')
    for exp in AllExps:
        alpha=1
        y = -SLC_DFT.loc[exp].SLC
        x = SLC_DFT.loc[exp].time
        label = 'exp:{}, GCM:{}, RCP:{}, collapse:{},gamma:{}'.format(exp, [Exps_dict[exp]['gcm']], [Exps_dict[exp]['scenario']],
                                                                      [Exps_dict[exp]['collapse']],[Exps_dict[exp]['gamma0pcntile']])
        if exp in ['TD58', '12']:
            ax.scatter(x, y, label=label, c=colors[exp], marker='^',
                       alpha=alpha)
        else:
            ax.plot(x, y, label=label, c=colors[exp],
                    linestyle=Formatting_dict_allExps['linestyles'][Exps_dict[exp]['scenario']], alpha=alpha)
    #ax.legend(ncol=2, fontsize='x-small')
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Sea level contribution (mm)', fontsize=14)

    ax.legend(handles=kwargs['legend_elements'], loc='upper left', ncol=3, fontsize='medium')
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(axis='both', labelsize=12)
    ax.grid()
    plt.savefig(savedir + 'fig04.png',
                dpi=300, bbox_inches='tight')

    # make dataframe of results relative to control
    SLC_2100_allexps={}
    for exp in AllExps:
        SLC_2100_allexps[exp]={}
        SLC_2100_allexps[exp]['delVAF_SLE']=SLC_DFT.loc[exp].SLC[-1]
        SLC_2100_allexps[exp]['delVAF_SLE_round']=np.round(SLC_DFT.loc[exp].SLC[-1])

    SLC_2100_allexps['control']={}
    SLC_2100_allexps['control']['delVAF_SLE']=SLC_DFT.loc['control'].SLC[-1].item()
    SLC_2100_allexps['control']['delVAF_SLE_round']=SLC_DFT.loc['control'].SLC[-1].round().item()
    out = pd.DataFrame.from_dict(SLC_2100_allexps)
    out.T.to_csv(savedir + 'all_results_not_minus_control_2100_unmasked.csv')

###################################### Figure 5 ########################################################################

def figure5(netcdf_root, savedir, dpi, **kwargs):
    print('Making figure 5')
    Exps_dict = kwargs['Exps_dict']
    colors = kwargs['colors']
    Formatting_dict_allExps = kwargs['Formatting_dict_allExps']
    # use iareag
    GrIA_dict = {}
    # make dictionary of results
    #control
    ctrl = xr.open_dataset(netcdf_root.format('ctrl', 'iareag', 'ctrl'))
    AllExps = ['5', '6', '7', '8', '9', '10', '12', '13', 'D52', 'D53', 'D55', 'D56', 'D58', 'T71', 'T73', 'TD58','B7', 'B6']
    for i, exp in enumerate(AllExps):
        GrIA_dict[exp]={}
        data = xr.open_dataset(netcdf_root.format(exp, 'iareag', exp))
        GrIA = data.iareag.sel(time=slice(2015, 2100)).data
        GrIA_dict[exp]['GrIA'] = GrIA
        GrIA_dict[exp]['time'] = data.iareag.sel(time=slice(2015, 2100)).time.data

    GrIA_dict['control'] = {}
    GrIA_dict['control']['GrIA'] = ctrl.iareag.sel(time=slice(2015, 2100)).data
    GrIA_dict['control']['time'] = ctrl.iareag.sel(time=slice(2015, 2100)).time.data

    GrIA_DF = pd.DataFrame.from_dict(GrIA_dict)
    GrIA_DFT = GrIA_DF.transpose()

    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(2, 2, hspace=0.1, wspace=0.1)#, wi
    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[0, 1])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[1, 1])
    axis_dict_allExps = {'NorESM1-M':ax0, 'CCSM4':ax1, 'MIROC-ESM':ax2, 'CNRM-CM6-1':ax3}
    for exp in AllExps:
        alpha=1
        y = ((GrIA_DF[exp].GrIA)/(1000*1000))/1e6
        x = GrIA_DF[exp].time
        label = 'exp:{}, GCM:{}, RCP:{}, collapse:{},gamma:{}'.format(exp, [Exps_dict[exp]['gcm']], [Exps_dict[exp]['scenario']],
                                                                      [Exps_dict[exp]['collapse']],[Exps_dict[exp]['gamma0pcntile']])
        axis=axis_dict_allExps[Exps_dict[exp]['gcm']]
        if exp in ['TD58', '12']:
            axis.scatter(x, y, label=label, c=colors[exp], marker='^',
                         alpha=alpha)
        else:
            axis.plot(x, y, label=label, c=colors[exp],
                      linestyle=Formatting_dict_allExps['linestyles'][Exps_dict[exp]['scenario']], alpha=alpha)
        axis.set_ylim(11.85, 12.2)

    for ax in [ax0, ax1, ax2, ax3]:
        ax.plot(GrIA_DF['control'].time, ((GrIA_DF['control'].GrIA)/(1000*1000))/1e6, 'k-', marker='|',
                zorder=0, alpha=0.5)

    ax0.set_xticklabels([])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax3.set_yticklabels([])
    legend_elements3 =[Line2D([0], [0],marker='None', color='None', label=r'$\bf{CNRM-CM6}$'),
                       Patch(facecolor='fuchsia', edgecolor='fuchsia', label='ANT50'),
                       Line2D([0], [0],linestyle='-.', color='k', label='SSP126',
                              markeredgecolor='k', markersize=10),
                       Line2D([0], [0],linestyle='--', color='k', label='SSP585'),
                       Line2D([0], [0],marker='|', color='k', label='control', alpha=0.5)]
    ax3.legend(handles=legend_elements3, loc='lower left', ncol=3, fontsize='x-small')

    legend_elements0 =[Line2D([0], [0],marker='None', color='None', label=r'$\bf{NorESM1-M}$'),
                       Patch(facecolor='aqua', edgecolor='aqua', label='ANT5'),
                       Patch(facecolor='c', edgecolor='c', label='ANT50'),
                       Patch(facecolor='deepskyblue', edgecolor='deepskyblue', label='ANT95'),
                       Patch(facecolor='royalblue', edgecolor='royalblue', label='PIG50'),
                       Patch(facecolor='navy', edgecolor='navy', label='PIG95'),
                       Line2D([0], [0],linestyle=':', color='k', label='RCP2.6',
                              markeredgecolor='k', markersize=10),
                       Line2D([0], [0],linestyle='-', color='k', label='RCP8.5',
                              markeredgecolor='k', markersize=10),
                       Line2D([0], [0],marker='|', color='k', label='control', alpha=0.5)]
    ax0.legend(handles=legend_elements0, loc='lower left', ncol=3, fontsize='x-small')

    legend_elements2 =[Line2D([0], [0],marker='None', color='None', label=r'$\bf{MIROC-ESM-CHEM}$'),
                       Patch(facecolor='gold', edgecolor='gold', label='ANT50'),
                       Patch(facecolor='darkorange', edgecolor='darkorange', label='PIG50'),
                       Patch(facecolor='chocolate', edgecolor='chocolate', label='PIG95'),
                       Line2D([0], [0],linestyle='-', color='k', label='RCP8.5',
                              markeredgecolor='k', markersize=10),
                       Line2D([0], [0],marker='|', color='k', label='control', alpha=0.5)]
    ax2.legend(handles=legend_elements2, loc='lower left', ncol=3, fontsize='x-small')

    legend_elements1 =[Line2D([0], [0],marker='None', color='None', label=r'$\bf{CCSM4}$'),
                       Patch(facecolor='lightcoral', edgecolor='lightcoral', label='ANT50'),
                       Patch(facecolor='red', edgecolor='red', label='PIG50'),
                       Patch(facecolor='firebrick', edgecolor='firebrick', label='PIG95'),
                       Line2D([0], [0],marker='^', color='w', label='RCP8.5, collapse on',
                              markeredgecolor='k', markersize=10),
                       Line2D([0], [0],linestyle='-', color='k', label='RCP8.5',
                              markeredgecolor='k', markersize=10),
                       Line2D([0], [0],marker='|', color='k', label='control', alpha=0.5)]
    ax1.legend(handles=legend_elements1, loc='lower left', ncol=3, fontsize='x-small')
    labels = 'abcdefghijklmnopqrst'
    for i, ax in enumerate([ax0, ax1, ax2, ax3]):
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.grid()
        ax.text(0.01, 0.95, '({})'.format(labels[i]), transform=ax.transAxes)

    fig.text(0.5, 0.04, 'Year', ha='center', fontsize=14)
    fig.text(0.04, 0.5, 'Grounded Area ($x10^{6}$$Km^{-2}$)', va='center', rotation='vertical', fontsize=14)
    plt.savefig(savedir + 'fig05.png', bbox_inches='tight', dpi=dpi)


###################################### Figure 6 ########################################################################

def figure6(netcdf_root, savedir, dpi, **kwargs):
    print('Making figure 6')
    Exps_dict = kwargs['Exps_dict']
    colors = kwargs['colors']
    Formatting_dict_allExps = kwargs['Formatting_dict_allExps']

    ctrl = xr.open_dataset(netcdf_root.format('ctrl', 'iareaf', 'ctrl'))
    AllExps = ['5', '6', '7', '8', '9', '10', '12', '13', 'D52', 'D53', 'D55', 'D56', 'D58', 'T71', 'T73', 'TD58','B7', 'B6']
    FlIA_dict = {}
    for i, exp in enumerate(AllExps):
        FlIA_dict[exp]={}
        data = xr.open_dataset(netcdf_root.format(exp, 'iareaf', exp))
        FlIA = data.iareaf.sel(time=slice(2015, 2100)).data
        FlIA_dict[exp]['FlIA'] = FlIA
        FlIA_dict[exp]['time'] = data.iareaf.sel(time=slice(2015, 2100)).time.data

    FlIA_dict['control']={}
    FlIA_dict['control']['FlIA'] = ctrl.iareaf.sel(time=slice(2015, 2100)).data
    FlIA_dict['control']['time'] = ctrl.iareaf.sel(time=slice(2015, 2100)).time.data

    FlIA_DF = pd.DataFrame.from_dict(FlIA_dict)

    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(2, 2, hspace=0.1, wspace=0.1)#, wi
    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[0, 1])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[1, 1])
    axis_dict_allExps = {'NorESM1-M':ax0, 'CCSM4':ax1, 'MIROC-ESM':ax2, 'CNRM-CM6-1':ax3}
    for exp in AllExps:
        alpha=1
        y = ((FlIA_DF[exp].FlIA)/(1000*1000))/1e6
        x = FlIA_DF[exp].time
        label = 'exp:{}, GCM:{}, RCP:{}, collapse:{},gamma:{}'.format(exp, [Exps_dict[exp]['gcm']], [Exps_dict[exp]['scenario']],
                                                                      [Exps_dict[exp]['collapse']],[Exps_dict[exp]['gamma0pcntile']])
        axis = axis_dict_allExps[Exps_dict[exp]['gcm']]
        if exp in ['TD58', '12']:
            axis.scatter(x, y, label=label, c=colors[exp], marker='^',
                         alpha=alpha)
        else:
            axis.plot(x, y, label=label, c=colors[exp],
                      linestyle=Formatting_dict_allExps['linestyles'][Exps_dict[exp]['scenario']], alpha=alpha)
        axis.set_ylim(1.35, 1.7)

    for ax in [ax0, ax1, ax2, ax3]:
        ax.plot(FlIA_DF['control'].time, ((FlIA_DF['control'].FlIA)/(1000*1000))/1e6, 'k-', marker='|',
                zorder=0, alpha=0.5)
    #ax.legend(ncol=2, fontsize='x-small')
    ax0.set_xticklabels([])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax3.set_yticklabels([])
    # same legends as previous
    legend_elements3 =[Line2D([0], [0],marker='None', color='None', label=r'$\bf{CNRM-CM6}$'),
                       Patch(facecolor='fuchsia', edgecolor='fuchsia', label='ANT50'),
                       Line2D([0], [0],linestyle='-.', color='k', label='SSP126',
                              markeredgecolor='k', markersize=10),
                       Line2D([0], [0],linestyle='--', color='k', label='SSP585'),
                       Line2D([0], [0],marker='|', color='k', label='control', alpha=0.5)]
    ax3.legend(handles=legend_elements3, loc='lower left', ncol=3, fontsize='x-small')

    legend_elements0 =[Line2D([0], [0],marker='None', color='None', label=r'$\bf{NorESM1-M}$'),
                       Patch(facecolor='aqua', edgecolor='aqua', label='ANT5'),
                       Patch(facecolor='c', edgecolor='c', label='ANT50'),
                       Patch(facecolor='deepskyblue', edgecolor='deepskyblue', label='ANT95'),
                       Patch(facecolor='royalblue', edgecolor='royalblue', label='PIG50'),
                       Patch(facecolor='navy', edgecolor='navy', label='PIG95'),
                       Line2D([0], [0],linestyle=':', color='k', label='RCP2.6',
                              markeredgecolor='k', markersize=10),
                       Line2D([0], [0],linestyle='-', color='k', label='RCP8.5',
                              markeredgecolor='k', markersize=10),
                       Line2D([0], [0],marker='|', color='k', label='control', alpha=0.5)]
    ax0.legend(handles=legend_elements0, loc='lower left', ncol=3, fontsize='x-small')

    legend_elements2 =[Line2D([0], [0],marker='None', color='None', label=r'$\bf{MIROC-ESM-CHEM}$'),
                       Patch(facecolor='gold', edgecolor='gold', label='ANT50'),
                       Patch(facecolor='darkorange', edgecolor='darkorange', label='PIG50'),
                       Patch(facecolor='chocolate', edgecolor='chocolate', label='PIG95'),
                       Line2D([0], [0],linestyle='-', color='k', label='RCP8.5',
                              markeredgecolor='k', markersize=10),
                       Line2D([0], [0],marker='|', color='k', label='control', alpha=0.5)]
    ax2.legend(handles=legend_elements2, loc='lower left', ncol=3, fontsize='x-small')

    legend_elements1 =[Line2D([0], [0],marker='None', color='None', label=r'$\bf{CCSM4}$'),
                       Patch(facecolor='lightcoral', edgecolor='lightcoral', label='ANT50'),
                       Patch(facecolor='red', edgecolor='red', label='PIG50'),
                       Patch(facecolor='firebrick', edgecolor='firebrick', label='PIG95'),
                       Line2D([0], [0],marker='^', color='w', label='RCP8.5, collapse on',
                              markeredgecolor='k', markersize=10),
                       Line2D([0], [0],linestyle='-', color='k', label='RCP8.5',
                              markeredgecolor='k', markersize=10),
                       Line2D([0], [0],marker='|', color='k', label='control', alpha=0.5)]
    ax1.legend(handles=legend_elements1, loc='lower left', ncol=3, fontsize='x-small')
    fig.text(0.5, 0.04, 'Year', ha='center', fontsize=14)
    fig.text(0.04, 0.5, 'Floating Area ($x10^{6}$$Km^{-2}$)', va='center', rotation='vertical', fontsize=14)
    labels = 'abcdefghijklmnopqrst'
    for i, ax in enumerate([ax0, ax1, ax2, ax3]):
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.grid()
        ax.text(0.01, 0.95, '({})'.format(labels[i]), transform=ax.transAxes)

    plt.savefig(savedir + 'fig06.png', bbox_inches='tight', dpi=dpi)


###################################### Figure 7 ########################################################################

def figure7(netcdf_dir, savedir, dpi, **kwargs):
    print('Making figure 7')
    Exps_dict = kwargs['Exps_dict']
    colors = kwargs['colors']
    Formatting_dict_allExps = kwargs['Formatting_dict_allExps']

    AllExps = ['5', '6', '7', '8', '9', '10', '12', '13', 'D52', 'D53', 'D55', 'D56', 'D58', 'T71', 'T73', 'TD58','B7', 'B6']
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 10.5), gridspec_kw={'hspace':0.15, 'wspace':0.3})
    plt.subplots_adjust(hspace=0.2)
    #fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    dir = netcdf_dir + 'maskedStats_regions/ismip6_{}.maskedStatsRegions'
    alpha=1
    axs = axs.flatten()
    # save sectoral results to data frame
    SLC_2100_masked={}
    regions = ['WAIS', 'EAIS', 'AP']
    for i, exp in enumerate(AllExps):
        file = pd.read_csv(dir.format(exp))
        ctrl = pd.read_csv(dir.format('ctrl2'))
        ctrl = ctrl[ctrl['time']>=2015][ctrl['time']<=2100]
        file = file[file['time']>=2015][file['time']<=2100]
        SLC_2100_masked[exp]={}
        for j in range(3):
            SLC_2100_masked[exp][regions[j]]={}
            plt.sca(axs[j])
            plt.axhline(0, color='k', alpha=0.7, linewidth=0.4)
            x = file[file['sector']==j]['time']
            y = file[file['sector']==j]['iceMassAbove']
            y = y - y.iloc[0]
            SLC_2100_masked[exp][regions[j]]=-np.round(y.iloc[-1]/1e12/362.5, 2)
            yctrl = ctrl[ctrl['sector']==j]['iceMassAbove']
            yctrl = yctrl - yctrl.iloc[0]
            if exp in ['12', 'TD58']:
                plt.plot(x, -((np.asarray(y))/1e12)/362.5, c=colors[exp],
                         linestyle=Formatting_dict_allExps['linestyles'][Exps_dict[exp]['scenario']], alpha=alpha,
                         marker='^', zorder=0, markersize=3)
            else:
                plt.plot(x, -((np.asarray(y))/1e12)/362.5, c=colors[exp], #label=label, c=colors[exp],
                         linestyle=Formatting_dict_allExps['linestyles'][Exps_dict[exp]['scenario']], alpha=alpha)
            if j < 2:
                plt.ylim(-50, 140) #100, 50)
            elif j==2:
                plt.ylim(-8, 15) #-12, 8)
            plt.ylabel('Sea level contribution (mm)')
            plt.xlabel('Year')
            plt.grid()

    # add control
    ctrl = pd.read_csv(dir.format('ctrl2'))
    ctrl = ctrl[ctrl['time']>=2015][ctrl['time']<=2100]
    SLC_2100_masked['control']={}
    for j in range(3):
        SLC_2100_masked['control'][regions[j]]={}
        plt.sca(axs[j])
        x = ctrl[ctrl['sector']==j]['time']
        yctrl = ctrl[ctrl['sector']==j]['iceMassAbove']
        yctrl = yctrl - yctrl.iloc[0]
        plt.plot(x, -((np.asarray(yctrl))/1e12)/362.5, 'k-', marker='|', zorder=0, alpha=0.4)
        SLC_2100_masked['control'][regions[j]]=-np.round(yctrl.iloc[-1]/1e12/362.5, 2)

    axs[0].annotate('WAIS', (2015, -40), weight='bold', c='r')
    axs[1].annotate('EAIS', (2015, -40), weight='bold', c='r')
    axs[2].annotate('AP', (2015, -7.5), weight='bold', c='r')
    # add legend, same as fig. 4
    axs[1].legend(handles=kwargs['legend_elements'], loc='upper left', bbox_to_anchor=(0,1), ncol=2, fontsize='x-small', framealpha=0.1,
                  bbox_transform=axs[1].transAxes)
    inset_axes=axs[-1]
    # read in mask and bedmap
    mask = xr.open_dataset(netcdf_dir + 'masks/Mask_EAIS_WAIS_PEN_seroussi_0_1_2_NN.nc')
    # read in 761 mask and pad
    bedmap = xr.open_dataset(netcdf_dir + 'bedmap2_8km.nc')
    bmap_is = np.pad(bedmap.icemask_grounded + 2 * bedmap.icemask_shelves, ((3, 4), (3, 4)), 'edge')
    inset_axes.imshow(np.ma.masked_where(bmap_is==0, bmap_is), origin='lower', cmap='Blues', vmax=4)
    inset_axes.contour(mask.mask, colors='r', linewidths=0.3, linestyles=':')
    inset_axes.axis('off')
    inset_axes.annotate('AP', (0,600), c='r', weight='bold')
    inset_axes.annotate('WAIS', (0,100), c='r', weight='bold')
    inset_axes.annotate('EAIS', (400,500), c='r', weight='bold')
    labels = 'abcdefghijklmnopqrst'
    for i, ax in enumerate([axs[0], axs[1], axs[2], inset_axes]):
        ax.text(0.01, 1.01, '({})'.format(labels[i]), transform=ax.transAxes)

    plt.savefig(savedir + 'fig07.png', dpi=dpi,
                bbox_inches='tight')
    # save table of masked results
    SLC_masked_DF = pd.DataFrame.from_dict(SLC_2100_masked)
    SLC_masked_DFT = SLC_masked_DF.transpose()
    SLC_masked_DFT.to_csv(savedir + 'regional_masked_results_not_minus_control_2100.csv')

###################################### Figure 8 ########################################################################
def figure8(netcdf_dir, savedir, dpi, **kwargs):
    print('Making figure 8')
    Exps_dict = kwargs['Exps_dict']
    colors = kwargs['colors']
    Formatting_dict_allExps = kwargs['Formatting_dict_allExps']

    AllExps = ['5', '6', '7', '8', '9', '10', '12', '13', 'D52', 'D53', 'D55', 'D56', 'D58', 'T71', 'T73', 'TD58','B7', 'B6']

    fig, axs = plt.subplots(6, 4, figsize=(8, 10))
    plt.subplots_adjust(hspace=0.1, wspace=0.3)
    axs = axs.flatten()
    alpha=0.8
    csv_root = netcdf_dir + 'maskedStats_sectors/ismip6_{}_maskedStats'
    basin_sector_contributions = {}
    SLC_2100_masked={}
    labels = 'abcdefghijklmnopqrst'
    for sector in range(16):
        basin_sector_contributions[sector+1] = {}
        conversion = (918/1e9)/362.5
        for exp in AllExps:
            SLC_2100_masked[exp]={}
            SLC_2100_masked[exp][sector+1]={}
            dset = pd.read_csv(csv_root.format(exp))
            ctrl = pd.read_csv(csv_root.format('ctrl2'))
            ctrl = ctrl[ctrl['time']<=2100][ctrl['time']>=2015]
            dset = dset[dset['time']<=2100][dset['time']>=2015]
            x = dset[dset['sector'] == sector]['time']
            y = (dset[dset['sector'] == sector]['iceMassAbove'] - dset[dset['sector'] == sector]['iceMassAbove'].iloc[0])
            SLC_2100_masked[exp][sector+1]=-np.round(y.iloc[-1]/1e12/362.5, 2)
            if exp in ['TD58', '12']:
                axs[sector].scatter(x, -(((np.asarray(y))/1e12)/362.5), c=colors[exp], marker='^', s=8, #label=label, c=colors[exp], marker='^', s=8,
                                    alpha=alpha)
            else:
                axs[sector].plot(x, -(((np.asarray(y))/1e12)/362.5), c=colors[exp], #label=label, c=colors[exp],
                                 linestyle=Formatting_dict_allExps['linestyles'][Exps_dict[exp]['scenario']], alpha=alpha)
            # sort sector 2, 6 and 13 y scales
            if sector == 2:
                axs[sector].set_ylim(-10,3)
            if sector in [6,13]:
                axs[sector].set_ylim(-4,2)
            # tick options
            if sector in [1,2,3,5,6,7,9,10,11]:
                axs[sector].set_xticklabels([])
            elif sector in [0,4,8]:
                axs[sector].set_ylabel('SLC (mm)')
                axs[sector].set_xticklabels([])
            elif sector in [13,14,15]:
                axs[sector].set_xlabel('Year')
            else:
                axs[sector].set_xlabel('Year')
                axs[sector].set_ylabel('SLC (mm)')


            basin_sector_contributions[sector+1][exp]=round((((np.asarray(y))/1e12)/362.5)[-1], 2)
        axs[sector].text(0.25, 0.95, str(sector + 1), color='r', transform=axs[sector].transAxes, weight='bold',
                         bbox=dict(boxstyle="square",
                                   ec=(1., 0.5, 0.5),
                                   fc=(1., 0.8, 0.8),
                                   facecolor='white')
                         )
        axs[sector].text(0.01, 0.85, '({})'.format(labels[sector]), transform=axs[sector].transAxes)

    SLC_2100_masked['control']={}
    for sector in range(16):
        ctrl = pd.read_csv(csv_root.format('ctrl2'))
        ctrl = ctrl[ctrl['time']<=2100][ctrl['time']>=2015]
        xctrl = ctrl[ctrl['sector'] == sector]['time']
        yctrl = (ctrl[ctrl['sector'] == sector]['iceMassAbove'] - ctrl[ctrl['sector'] == sector]['iceMassAbove'].iloc[0])
        axs[sector].plot(xctrl, -((np.asarray(yctrl)/1e12)/362.5), 'k-', marker='|', alpha=0.4, zorder=0)
        SLC_2100_masked['control'][sector]=-np.round(yctrl.iloc[-1]/1e12/362.5, 2)


    for i in [16, 17, 18, 19, 20, 21, 22, 23]:
        axs[i].axis('off')
    gs1 = fig.add_gridspec(nrows=6, ncols=4, wspace=0.1, hspace=0.1)#, left=0.05, right=0.48, wspace=0.05)
    gs_ax0 = fig.add_subplot(gs1[4:, :2])
    gs_ax1 = gs_ax0.inset_axes([-0.3, 0.0, 1, 0.86])
    gs_ax0.axis('off')
    # add basins
    IMBIEmask = xr.open_dataset(netcdf_dir + 'masks/imbie2_basin_numbers_8km.nc')
    bedmap8km = xr.open_dataset(netcdf_dir + 'bedmap2_8km.nc')
    gs_ax1.contour(bedmap8km.icemask_shelves, colors='grey', linestyles='--', linewidths=0.3, alpha=0.8)
    gs_ax1.contour(bedmap8km.icemask_grounded, colors='grey', linestyles='-', linewidths=0.3, alpha=0.8)
    gs_ax1.contour(IMBIEmask.basinNumber,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], colors='r', linestyles=':', linewidths=0.6)
    gs_ax1.annotate('1.', (430,700), color='r', weight='bold')
    gs_ax1.annotate('2.', (680,680), color='r', weight='bold')
    gs_ax1.annotate('3.', (540,450), color='r', weight='bold')
    gs_ax1.annotate('4.', (700,380), color='r', weight='bold')
    gs_ax1.annotate('5.', (700,170), color='r', weight='bold')
    gs_ax1.annotate('6.', (600,70), color='r', weight='bold')
    gs_ax1.annotate('7.', (400,70), color='r', weight='bold')
    gs_ax1.annotate('8.', (400,280), color='r', weight='bold')
    gs_ax1.annotate('9.', (140,90), color='r', weight='bold')
    gs_ax1.annotate('10.', (60,250), color='r', weight='bold')
    gs_ax1.annotate('11.', (30,360), color='r', weight='bold')
    gs_ax1.annotate('12.', (80,430), color='r', weight='bold',
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")]) # illegible otherwise
    gs_ax1.annotate('13.', (60,670), color='r', weight='bold')
    gs_ax1.annotate('14.', (130,510), color='r', weight='bold')
    gs_ax1.annotate('15.', (350,440), color='r', weight='bold')
    gs_ax1.annotate('16.', (240,640), color='r', weight='bold')
    gs_ax1.set_xticks([])
    gs_ax1.set_yticks([])
    gs_ax0.set_aspect('equal')
    gs_ax1.set_aspect('equal')
    gs_ax1.text(0.05, 0.05, '(q)', transform=gs_ax1.transAxes)

    gs_ax2 = fig.add_subplot(gs1[4:, 2:])
    gs_ax3 = gs_ax2.inset_axes([-0.1,0,1,1])
    # add legend elements
    gs_ax3.legend(handles=kwargs['legend_elements'], loc='center', ncol=3, fontsize='small')
    gs_ax2.axis('off')
    gs_ax3.axis('off')
    fig.savefig(savedir + 'fig08.png', dpi=dpi, bbox_inches='tight')

    # save table of masked results
    SLC_masked_DF = pd.DataFrame.from_dict(SLC_2100_masked)
    SLC_masked_DFT = SLC_masked_DF.transpose()
    SLC_masked_DFT.to_csv(savedir + 'sectoral_masked_results_not_minus_control_2100.csv')

###################################### Figure 9 ########################################################################

def calculate_GLF_smb_dltihkdt(netcdf_root, netcdf_dir, exp, start_year, sector):
    """
    :param exp: netcdf root
    :param exp: experiment
    :param start_year: start year
    :param sector: sector number 1: WAIS, 2:EAIS, 3: AP
    :return: time, GLFlux calculated from masked SMB and dlithkdt, assumes static ice divides,
    Gt/yr
    """
    sftgrf = xr.open_dataset(netcdf_root.format(exp, 'sftgrf', exp))
    # some quantities have half year time axes, so need 2014.5
    sftgrf = sftgrf.sel(time=slice(start_year, 2100))

    dlithkdt = xr.open_dataset(netcdf_root.format(exp, 'dlithkdt', exp))
    dlithkdt = dlithkdt.sel(time=slice(start_year-1, 2100))

    acabf = xr.open_dataset(netcdf_root.format(exp, 'acabf', exp))
    acabf = acabf.sel(time=slice(start_year-1, 2100))

    seroussi_mask = xr.open_dataset(netcdf_dir + 'masks/sectors_8km.nc')
    mask = np.where(seroussi_mask.regions == sector, 1, np.zeros_like(seroussi_mask.regions))

    res = acabf.x.data[1] - acabf.x.data[0]
    GLF = ((dlithkdt.dlithkdt.data - acabf.acabf.data/918) * sftgrf.sftgrf.data) * mask.data
    GLFsum = np.sum(GLF, axis=(1,2)) * res ** 2 * 918/ 1e12
    return acabf.time, GLFsum

def figure9(netcdf_dir, netcdf_root, savedir, dpi, **kwargs):
    print('Making figure 9')
    Exps_dict = kwargs['Exps_dict']
    colors = kwargs['colors']
    Formatting_dict_allExps = kwargs['Formatting_dict_allExps']
    AllExps = ['5', '6', '7', '8', '9', '10', '12', '13', 'D52', 'D53', 'D55', 'D56', 'D58', 'T71', 'T73', 'TD58','B7', 'B6']

    ## add zorder dictionary
    zdict = {'5':None,'6':None,'7':None, '8':None, '9':None, '10':None, '12':0, 'B6':None, 'B7':None, '13':None,
             'D52':None, 'D53':None, 'D55':None, 'D56':-1, 'D58':None, 'T71':None, 'T73':None, 'TD58':0}

    fig, axs = plt.subplots(5, 3, figsize=(13, 10.5), gridspec_kw={'hspace':0.07, 'wspace':0.01,
                                                                   'width_ratios':[1,0.1,1],
                                                                   'height_ratios':[1,1,0.1,1,1]})
    axes = axs.flatten()
    mask = xr.open_dataset(netcdf_dir + 'masks/ISMIP6_masks_8km_761_seroussi.nc')
    experiments = ['5','6','7','8','9','10','12','13','D52','D53','D55','D56','D58','T71','T73','TD58','B6','B7']
    alpha=1
    for i, exp in enumerate(experiments):
        # read in thickness
        for j in range(3):
            x, y = calculate_GLF_smb_dltihkdt(netcdf_root, netcdf_dir, exp, 2015, j+1)
            plt.sca(axes[[0,2,9][j]])
            if exp in ['12', 'TD58']:
                plt.plot(x, -y/362.5, c=colors[exp],
                         linestyle=Formatting_dict_allExps['linestyles'][Exps_dict[exp]['scenario']],
                         alpha=alpha, marker='^', zorder=0)
            else:
                plt.plot(x, -y/362.5, c=colors[exp],
                         linestyle=Formatting_dict_allExps['linestyles'][Exps_dict[exp]['scenario']],
                         alpha=alpha, zorder=zdict[exp])
            plt.grid()

    # add control
    for j in range(3):
        plt.sca(axes[[0,2,9][j]])
        x, y = calculate_GLF_smb_dltihkdt(netcdf_root, netcdf_dir,'ctrl', 2015, j+1)
        plt.plot(x, -y/362.5, linestyle='-', marker='|', color='k', label='control', alpha=0.4, zorder=0)

    axs[0,0].set_ylabel('GL flux (mm SLC)')
    axs[3,0].set_ylabel('GL flux (mm SLC)')

    axs[0,0].text(0.1, 0.9, 'WAIS', weight='bold', c='r', transform=axs[0,0].transAxes)
    axs[0,2].text(0.1, 0.9, 'EAIS', weight='bold', c='r', transform=axs[0,2].transAxes)
    axs[3,0].text(0.1, 0.9,'AP', weight='bold', c='r', transform=axs[3,0].transAxes)

    for i in [1,4,6,7,8,10,13]:
        plt.sca(axes[i])
        plt.axis('off')


    ############# now add in plots for SMB ###########
    dir = netcdf_dir + 'maskedStats_regions/ismip6_{}.maskedStatsRegions'
    alpha=1
    for i, exp in enumerate(experiments):
        file = pd.read_csv(dir.format(exp))
        ctrl = pd.read_csv(dir.format('ctrl2'))
        ctrl = ctrl[ctrl['time']>=2015][ctrl['time']<=2100]
        file = file[file['time']>=2015][file['time']<=2100]
        for j in range(3):
            plt.sca(axes[[3,5,12][j]])
            x = file[file['sector']==j]['time']
            y = file[file['sector']==j]['Grounded_SMB']
            if exp in ['12', 'TD58']:
                plt.plot(x, -((np.asarray(y*918))/1e12)/362.5, c=colors[exp],
                         linestyle=Formatting_dict_allExps['linestyles'][Exps_dict[exp]['scenario']],
                         alpha=alpha, marker='^', zorder=0)
            else:
                plt.plot(x, -((np.asarray(y*918))/1e12)/362.5, c=colors[exp],
                         linestyle=Formatting_dict_allExps['linestyles'][Exps_dict[exp]['scenario']],
                         alpha=alpha, zorder=zdict[exp])
            plt.grid()

    # add control
    ctrl = pd.read_csv(dir.format('ctrl2'))
    ctrl = ctrl[ctrl['time']>=2015][ctrl['time']<=2100]
    for j in range(3):
        plt.sca(axes[[3,5,12][j]])
        x = ctrl[ctrl['sector']==j]['time']
        yctrl = ctrl[ctrl['sector']==j]['Grounded_SMB']
        plt.plot(x, -((np.asarray(yctrl*918))/1e12)/362.5, 'k-', zorder=0, alpha=0.4, marker='|')

    axs[1,0].set_ylabel('Grounded SMB (mm SLC)')
    axs[4,0].set_ylabel('Grounded SMB (mm SLC)')
    axs[4,0].set_xlabel('Year')

    # reuse legend elements
    axs[3,2].legend(handles=kwargs['legend_elements'], loc='center', ncol=3, fontsize='small', framealpha=0.3)
    axs[3,2].axis('off')

    inset_axes=axs[4,2]
    # read in mask and bedmap
    mask = xr.open_dataset(netcdf_dir + 'masks/Mask_EAIS_WAIS_PEN_seroussi_0_1_2_NN.nc')
    # read in 761 mask and pad
    bedmap = xr.open_dataset(netcdf_dir + 'bedmap2_8km.nc')
    bmap_is = np.pad(bedmap.icemask_grounded + 2 * bedmap.icemask_shelves, ((3, 4), (3, 4)), 'edge')
    inset_axes.imshow(np.ma.masked_where(bmap_is==0, bmap_is), origin='lower', cmap='Blues', vmax=4)
    inset_axes.contour(mask.mask, colors='r', linewidths=0.3, linestyles=':')
    inset_axes.axis('off')
    inset_axes.annotate('AP', (0,600), c='r', weight='bold')
    inset_axes.annotate('WAIS', (0,100), c='r', weight='bold')
    inset_axes.annotate('EAIS', (400,500), c='r', weight='bold')

    # add subplots lables
    labels = 'abcdefg'
    pos = [0,2,3,5,9,12,14]
    for i, l in enumerate(labels):
        plt.sca(axes[pos[i]])
        plt.text(0.01, 0.9, '({})'.format(l), transform=axes[pos[i]].transAxes)

    plt.savefig(savedir + 'fig09.png', dpi=dpi, bbox_inches='tight')

###################################### Figure 10 ########################################################################

def figure10(netcdf_dir, savedir, dpi, **kwargs):
    print('Making figure 10')
    Exps_dict = kwargs['Exps_dict']
    Formatting_dict = {'colors':{'RCP2.6':'blue', 'RCP8.5':'red', 'SSP126':'cyan', 'SSP585':'magenta'},
                       'Marker':{'NorESM1-M':'s', 'CCSM4':'o', 'MIROC-ESM':'<', 'CNRM-CM6-1':'H'}}

    MaskedStatsRoot=netcdf_dir + 'maskedStats_regions/ismip6_{}.maskedStatsRegions'
    ctrl_path = netcdf_dir + 'maskedStats_regions/ismip6_{}.maskedStatsRegions'.format('ctrl2')
    sectorNames = ['WAIS', 'EAIS', 'AP']
    fig, axs = plt.subplots(2, 2, figsize=(8,8), gridspec_kw={'wspace':0.3})
    axs = axs.flatten()
    for j in [0,1,2]:
        plt.sca(axs[j])
        for i, exp in enumerate(['T71', 'T73','5', '6', '7', '8', '9', '10', '12', '13', 'D52', 'D53', 'D55', 'D56', 'D58', 'TD58', 'B7','B6']):
            if Exps_dict[exp]['collapse'] == 'ON':
                fillstyle = 'bottom'
            else:
                fillstyle = 'full'
            path = glob.glob(MaskedStatsRoot.format(exp))[0]
            data = pd.read_csv(path, sep=',')
            control = pd.read_csv(ctrl_path, sep=',')
            control = control[control['sector']==j]
            data = data[data['sector']==j]
            slc = data[data['time']==2100].iceVolumeAbove.item() - data[data['time']==2015].iceVolumeAbove.item()
            ctrl = control[control['time']==2100].iceVolumeAbove.item() - control[control['time']==2015].iceVolumeAbove.item()
            SLC = ((((slc)*918)/1e12)/362.5)#/10 #<- cm
            plt.scatter(Exps_dict[exp]['gamma0'] / 1000, - SLC,
                        c=Formatting_dict['colors'][Exps_dict[exp]['scenario']],
                        marker=MarkerStyle(Formatting_dict['Marker'][Exps_dict[exp]['gcm']], fillstyle=fillstyle),
                        edgecolors=Formatting_dict['colors'][Exps_dict[exp]['scenario']], s=30, alpha=0.5)
        if j in [1, 2]:
            plt.xlabel('Antarctic basal melt parameter (x$10^{3}$ma$^{-1}$)')
        plt.ylabel('Sea level contribution (mm)')
        #plt.ylim(-10, 30)
        plt.title(sectorNames[j], y=0.9, weight='bold')
        axs[j].text(0.01, 1.01, '({})'.format('abc'[j]), transform=axs[j].transAxes)
        plt.grid()
    plt.sca(axs[-1])


    Formatting_dict = {'colors':{'RCP2.6':'blue', 'RCP8.5':'red', 'SSP126':'cyan', 'SSP585':'magenta'},
                       'Marker':{'NorESM1-M':'s', 'CCSM4':'o', 'MIROC-ESM':'<', 'CNRM-CM6-1':'H'}}

    legend_elements =[Patch(facecolor='blue', edgecolor='blue', label='RCP2.6'),
                      Patch(facecolor='red', edgecolor='red', label='RCP8.5'),
                      Patch(facecolor='cyan', edgecolor='cyan', label='SSP126'),
                      Patch(facecolor='magenta', edgecolor='magenta', label='SSP585'),
                      Line2D([0], [0], marker='s', color='k', label='NorESM1-M',
                             markeredgecolor='k', markersize=10, lw=0),
                      Line2D([0], [0], marker='o', color='k', label='CCSM4',
                             markeredgecolor='k', markersize=10, lw=0),
                      Line2D([0], [0], marker=MarkerStyle('o', fillstyle='bottom'), color='k',
                             label='Collapse', markeredgecolor='k', markersize=10, lw=0),
                      Line2D([0], [0], marker='<', color='k', label='MIROC-ESM',
                             markeredgecolor='k', markersize=10, lw=0),
                      Line2D([0], [0], marker='H', color='k', label='CNRM-CM6-1',
                             markeredgecolor='k', markersize=10, lw=0)]

    plt.legend(handles=legend_elements, loc='center', ncol=2, fontsize='small', framealpha=0.1)
    plt.axis('off')
    plt.savefig(savedir + 'fig10.png',
                dpi=dpi, bbox_inches='tight')

###################################### Figure 11 ########################################################################
# Nb: FIGURE WON'T WORK IF TF DATA NOT DOWNLOADED
def generate_TF_gradient_masked(model, scenario, year, mask, TF_dir):
    TF_root = TF_dir + '{}*{}*thermal_forcing_8km_x_60m.nc'.format(model, scenario)
    if len(glob.glob(TF_root))==1: # check thermal forcing exists
        dataset = xr.open_dataset(glob.glob(TF_root)[0], decode_times=False)
        thermal_forcing = dataset.isel(time=year)
        TF_gradient = []
        for i in range(len(thermal_forcing.z)):
            layer = thermal_forcing.thermal_forcing.isel(z=i)
            layer = mask * layer
            mean = np.nanmean(layer)
            TF_gradient.append(mean)
        return np.asarray(TF_gradient), thermal_forcing.z
    else:
        print('Need to download thermal forcing data from ISMIP6....')

def plot_transect_thermal_forcing_NorESM_multitimes_seperate(exp1, exp2, scenario, GCM, BN, Y, X, netcdf_root, netcdf_dir,
                                                             TF_dir, savedir, dpi):
    # could calculate base and surface from lithk and bed elevation and mask, but included data for exps for ease

    orog1 = xr.open_dataset(netcdf_root.format(exp1, 'orog', exp1))
    orog2 = xr.open_dataset(netcdf_root.format(exp2, 'orog', exp2))
    base1 = xr.open_dataset(netcdf_root.format(exp1, 'base', exp1))
    base2 = xr.open_dataset(netcdf_root.format(exp2, 'base', exp2))
    topg = xr.open_dataset(netcdf_dir + 'BISICLES_ISMIP6_bed_elevation_8km.nc')

    bedmap_8km = xr.open_dataset(netcdf_dir + 'bedmap2_8km.nc')
    shelf_mask = bedmap_8km.icemask_shelves
    imbie_basins = xr.open_dataset(netcdf_dir + 'masks/imbie2_basin_numbers_8km.nc')
    ross_basin = np.where(imbie_basins.basinNumber == BN, 1, np.zeros_like(imbie_basins.basinNumber))
    template = np.ones((761, 761))*np.nan
    ross_shelf_mask = np.where((ross_basin == 1) & (shelf_mask == 1), 1, template)

    # generate NorESM temp gradients
    noresm_rcp85_tfg_2015 = generate_TF_gradient_masked(GCM, 'RCP85', 20, ross_shelf_mask, TF_dir)[0]
    noresm_rcp85_tfg_2050 = generate_TF_gradient_masked(GCM, 'RCP85', 55, ross_shelf_mask, TF_dir)[0]
    noresm_rcp85_tfg_2100 = generate_TF_gradient_masked(GCM, 'RCP85', 95, ross_shelf_mask, TF_dir)[0]

    noresm_rcp26_tfg_2015 = generate_TF_gradient_masked(GCM, 'RCP26', 20, ross_shelf_mask, TF_dir)[0]
    noresm_rcp26_tfg_2050 = generate_TF_gradient_masked(GCM, 'RCP26', 55, ross_shelf_mask, TF_dir)[0]
    noresm_rcp26_tfg_2100 = generate_TF_gradient_masked(GCM, 'RCP26', 95, ross_shelf_mask, TF_dir)[0]

    i = 0
    z = generate_TF_gradient_masked(GCM, 'RCP85', i + 15, ross_shelf_mask, TF_dir)[1]

    y = Y
    x = X
    topg_transect = []
    for j in range(len(x)):
        topg_transect.append(topg.topg[x[j], y[j]].data)

    # try plotting with transect show aswell
    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1], 'wspace':0.05},
                            figsize=(9, 5))
    # simple cross sections
    exp1_2015_base = []
    exp2_2015_base = []
    exp1_2050_base = []
    exp2_2050_base = []
    exp1_2100_base = []
    exp2_2100_base = []
    exp1_2015_orog = []
    exp2_2015_orog = []
    exp1_2050_orog = []
    exp2_2050_orog = []
    exp1_2100_orog = []
    exp2_2100_orog = []
    for j in range(len(x)):
        exp1_2015_base.append(base1.base.sel(time=2015)[x[j], y[j]].data)
        exp2_2015_base.append(base2.base.sel(time=2015)[x[j], y[j]].data)
        exp1_2050_base.append(base1.base.sel(time=2050)[x[j], y[j]].data)
        exp2_2050_base.append(base2.base.sel(time=2050)[x[j], y[j]].data)
        exp1_2100_base.append(base1.base.sel(time=2100)[x[j], y[j]].data)
        exp2_2100_base.append(base2.base.sel(time=2100)[x[j], y[j]].data)
        exp1_2015_orog.append(orog1.orog.sel(time=2015)[x[j], y[j]].data)
        exp2_2015_orog.append(orog2.orog.sel(time=2015)[x[j], y[j]].data)
        exp1_2050_orog.append(orog1.orog.sel(time=2050)[x[j], y[j]].data)
        exp2_2050_orog.append(orog2.orog.sel(time=2050)[x[j], y[j]].data)
        exp1_2100_orog.append(orog1.orog.sel(time=2100)[x[j], y[j]].data)
        exp2_2100_orog.append(orog2.orog.sel(time=2100)[x[j], y[j]].data)

    ax1 = axs[0]
    ax1.set_xlabel('Distance along transect (km)')
    ax1.set_ylabel('Depth (m)')
    c1 = 'navy'
    alpha = 0.5
    cxlabel_box = dict(facecolor='w', edgecolor='none', alpha=0.6)
    ax1.plot(np.arange(len(exp1_2015_base)) * 8, exp1_2015_base, c=c1, linestyle='--', label='2015', alpha=alpha)
    yi = np.searchsorted(np.arange(len(exp1_2015_base)) * 8, 400)
    ax1.annotate('2015', (400, exp1_2015_base[yi]), size=8, c=c1, bbox=cxlabel_box)
    ax1.plot(np.arange(len(exp1_2050_base)) * 8, exp1_2050_base, c=c1, linestyle='--', label='2050', alpha=alpha)
    ax1.annotate('2050', (400, exp1_2050_base[yi]), size=8, c=c1, bbox=cxlabel_box)
    ax1.plot(np.arange(len(exp1_2100_base)) * 8, exp1_2100_base, c=c1, linestyle='--', label='2100', alpha=alpha)
    ax1.annotate('2100', (400, exp1_2100_base[yi]), size=8, c=c1, bbox=cxlabel_box)

    ax1.plot(np.arange(len(exp1_2015_orog)) * 8, exp1_2015_orog, c=c1, linestyle='--', label='2015', alpha=alpha)
    yi = np.searchsorted(np.arange(len(exp1_2015_orog)) * 8, 400)
    ax1.annotate('2015', (400, exp1_2015_orog[yi]), size=8, c=c1, bbox=cxlabel_box)
    ax1.plot(np.arange(len(exp1_2050_orog)) * 8, exp1_2050_orog, c=c1, linestyle='--', label='2050', alpha=alpha)
    ax1.annotate('2050', (400, exp1_2050_orog[yi]), size=8, c=c1, bbox=cxlabel_box)
    ax1.plot(np.arange(len(exp1_2100_orog)) * 8, exp1_2100_orog, c=c1, linestyle='--', label='2100', alpha=alpha)
    ax1.annotate('2100', (400, exp1_2100_orog[yi]), size=8, c=c1, bbox=cxlabel_box)

    c2='steelblue'
    yi = np.searchsorted(np.arange(len(exp1_2015_base)) * 8, 350)
    ax1.plot(np.arange(len(exp2_2015_base)) * 8, exp2_2015_base, c=c2, linestyle='--', label='2015', alpha=alpha)
    ax1.annotate('2015', (350, exp2_2015_base[yi]), size=8, c=c2, bbox=cxlabel_box)
    ax1.plot(np.arange(len(exp2_2050_base)) * 8, exp2_2050_base, c=c2, linestyle='--', label='2050', alpha=alpha)
    ax1.annotate('2050', (350, exp2_2050_base[yi]), size=8, c=c2, bbox=cxlabel_box)
    ax1.plot(np.arange(len(exp2_2100_base)) * 8, exp2_2100_base, c=c2, linestyle='--', label='2100', alpha=alpha)
    ax1.annotate('2100', (350, exp2_2100_base[yi]), size=8, c=c2, bbox=cxlabel_box)

    yi = np.searchsorted(np.arange(len(exp1_2015_orog)) * 8, 350)
    ax1.plot(np.arange(len(exp2_2015_orog)) * 8, exp2_2015_orog, c=c2, linestyle='--', label='2015', alpha=alpha)
    ax1.annotate('2015', (350, exp2_2015_orog[yi]), size=8, c=c2, bbox=cxlabel_box)
    ax1.plot(np.arange(len(exp2_2050_orog)) * 8, exp2_2050_orog, c=c2, linestyle='--', label='2050', alpha=alpha)
    ax1.annotate('2050', (350, exp2_2050_orog[yi]), size=8, c=c2, bbox=cxlabel_box)
    ax1.plot(np.arange(len(exp2_2100_orog)) * 8, exp2_2100_orog, c=c2, linestyle='--', label='2100', alpha=alpha)
    ax1.annotate('2100', (350, exp2_2100_orog[yi]), size=8, c=c2, bbox=cxlabel_box)


    ax1.plot(np.arange(len(topg_transect)) * 8, topg_transect, c='tan', linestyle='--')
    ax1.fill_between(np.arange(len(exp1_2015_base)) * 8, topg_transect, min(exp1_2015_base, exp2_2015_base), color='skyblue', alpha=0.2)
    ax1.fill_between(np.arange(len(topg_transect)) * 8, -1770, topg_transect, color='tan', alpha=0.4)
    ax1.annotate('A', (0, -1750), color='r')
    ax1.annotate('B', ((np.arange(len(exp1_2015_base)) * 8)[-1], -1750), color='r')

    ax2 = axs[1]
    color = 'r' #'tomato'
    ls  = '-'
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                      '#f781bf', '#a65628', '#984ea3',
                      '#999999', '#e41a1c', '#dede00']
    lw=2
    rot= -55 #-45
    depth = -1250
    lc15 = CB_color_cycle[3]
    lc50 = CB_color_cycle[1]
    lc100 = CB_color_cycle[7]
    yi = np.searchsorted(-z, -depth)
    if scenario == 'RCP26':
        ax2.plot(noresm_rcp26_tfg_2015, z, color=lc15, linestyle=ls, linewidth=lw,
                 label='NorESM1-M {},\n2015'.format(scenario))
        ax2.annotate('2015', (noresm_rcp26_tfg_2015[yi]-0.1, depth), color=lc15, size=7, rotation=rot)
        ax2.plot(noresm_rcp26_tfg_2050, z, color=lc50, linestyle=ls, linewidth=lw,
                 label='NorESM1-M {},\n2050'.format(scenario))
        ax2.annotate('2050', (noresm_rcp26_tfg_2050[yi]-0.1, depth), color=lc50, size=7, rotation=rot)
        ax2.plot(noresm_rcp26_tfg_2100, z, color=lc100, linestyle=ls, linewidth=lw,
                 label='NorESM1-M {},\n2100'.format(scenario))
        ax2.annotate('2100', (noresm_rcp26_tfg_2100[yi]-0.1, depth), color=lc100, size=7, rotation=rot)
        #labelLines(plt.gca().get_lines(), zorder=2.5)
    elif scenario == 'RCP85':
        ax2.plot(noresm_rcp85_tfg_2015, z, color=lc15, linestyle=ls, linewidth=lw,
                 label='NorESM1-M {},\n2015'.format(scenario))
        ax2.annotate('2015', (noresm_rcp85_tfg_2015[yi]-0.1, depth), color=lc15, size=7, rotation=rot)
        ax2.plot(noresm_rcp85_tfg_2050, z, color=lc50, linestyle=ls, linewidth=lw,
                 label='NorESM1-M {},\n2050'.format(scenario))
        ax2.annotate('2050', (noresm_rcp85_tfg_2050[yi]-0.1, depth), color=lc50, size=7, rotation=rot)
        ax2.plot(noresm_rcp85_tfg_2100, z, color=lc100, linestyle=ls, linewidth=lw,
                 label='NorESM1-M {},\n2100'.format(scenario))
        ax2.annotate('2100', (noresm_rcp85_tfg_2100[yi]-0.1, depth), color=lc100, size=7, rotation=rot)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels([0, 1, 2])
    ax2.set_xlabel('Average thermal forcing (K)', c='k') #color)
    for a in [ax1, ax2]:
        a.set_yticks(np.linspace(0,-1800, 10))
        a.set_ylim(-1800, 0)
    ax2.set_yticklabels([])
    ax2.set_title('(c)', x=0.02, y=1.05)
    ax2.text(0.3, 1.05, 'NorESM1-M {}'.format(scenario),
             transform=ax2.transAxes)
    # do legend
    legend_elements = [Line2D([0], [0], linestyle='--', color=c1, label='Exp{}, PIG50'.format(exp1)),
                       Line2D([0], [0], linestyle='--', color=c2, label='Exp{}, PIG95'.format(exp2))]
    ax1.legend(handles=legend_elements, loc='upper left', ncol=2, fontsize='small')

    # add plot of transect
    ax3 = ax1.inset_axes([0.3, -0.05, 0.6, 0.7])
    ax3.imshow(np.ma.masked_where(bedmap_8km.open_ocean_mask == 1,
                                  4 * bedmap_8km.icemask_shelves + bedmap_8km.icemask_grounded), origin='lower', vmax=10,
               cmap='Blues')
    ax3.contour(bedmap_8km.icemask_grounded, linewidths=0.1, colors='k')
    ax3.contour(bedmap_8km.icemask_shelves, linewidths=0.1, colors='k')
    ax3.axis('off')
    #ax3.plot()
    ax3.plot(y, x, 'r-', linewidth=1.2)
    ax3.annotate('A', (y[0] - 20, x[0] + 20), c='r')
    #ax3.annotate('B', (y[-1]-20, x[-1] - 20), c='r')
    ax3.annotate('B', (y[-1] - 10, x[-1] - 50), c='r')

    ax1.set_title('(a)', x=0.02, y=1.05)
    ax3.set_title('(b)', x=0.2, y=0.8)
    plt.savefig(savedir + 'fig11.png', dpi=dpi, bbox_inches='tight')
    return np.arange(len(exp1_2015_base)) * 8


def figure11(netcdf_dir, netcdf_root, TF_dir, savedir, dpi):
    print('Making figure 11')
    exp1 = '13'
    exp2 = 'D52'
    scenario = 'RCP85'
    GCM = 'NorESM1-M'
    BN = 7 # basinnumber, ross is 7
    X = np.linspace(255, 230, 51).astype(int)
    Y = np.linspace(625, 675, 51).astype(int)

    X = np.linspace(330, 280, 61).astype(int)
    Y = np.linspace(330, 380, 61).astype(int)

    bedmap_8km = xr.open_dataset(netcdf_dir + 'bedmap2_8km.nc')
    shelf_mask = bedmap_8km.icemask_shelves
    imbie_basins = xr.open_dataset(netcdf_dir + 'masks/imbie2_basin_numbers_8km.nc')
    ross_basin = np.where(imbie_basins.basinNumber==7, 1, np.zeros_like(imbie_basins.basinNumber))
    template = np.zeros((761,761))
    template[:]=np.nan
    ross_shelf_mask = np.where((ross_basin==1) & (shelf_mask==1), 1, template)

    # check thermal forcing data present before trying to plot
    thermal_forcing_path = '/Volumes/T7/thermal_forcings_ismip6/' # would need to replace with path to download
    TF_root = TF_dir + '{}*{}*thermal_forcing_8km_x_60m.nc'.format(GCM, scenario)
    if len(glob.glob(TF_root))==1:
        test = plot_transect_thermal_forcing_NorESM_multitimes_seperate(exp1, exp2, scenario, GCM, BN, Y, X,
                                                                        netcdf_root, netcdf_dir, TF_dir, savedir, dpi)
    else:
        print('Thermal forcing data not downloaded, cant plot figure 11...')

###################################### Figure 12 ########################################################################
def figure12(netcdf_dir, savedir, dpi, **kwargs):
    print('Making figure 12')
    Exps_dict = kwargs['Exps_dict']

    Other_models_formatter = {'AISMPALEO':'saddlebrown', 'CISM':'fuchsia', 'ElmerIce':'turquoise', 'GRISLI':'teal', 'IMAUICE1':'darkmagenta',
                              'IMAUICE2':'chocolate', 'ISSM':'orange', 'ISSM2':'gold', 'MALI':'darkolivegreen', 'PISM':'lightskyblue', 'PISM1':'navy', 'PISM2':'b', 'SICOPOLIS':'orchid',
                              'fETISh_16km':'orangered', 'fETISh_32km':'red', 'this_study':'k'}
    Other_models_formatter_ls = {'AISMPALEO':'--', 'CISM':'--', 'ElmerIce':'--', 'GRISLI':'--', 'IMAUICE1':':',
                                 'IMAUICE2':'--', 'ISSM':'-.', 'ISSM2':':','MALI':':', 'PISM':'--', 'PISM1':':', 'PISM2':'-.', 'SICOPOLIS':'--',
                                 'fETISh_16km':':', 'fETISh_32km':'--', 'this_study':'-'}

    Other_models_formatter_marker = {'AWI':None, 'DOE':'x', 'ILTS_PIK':'x', 'IMAU':None, 'JPL1':'x', 'LSCE':None, 'NCAR':None, 'PIK':None,
                                     'UCIJPL':None, 'ULB':None, 'UTAS':'x', 'VUB':None, 'VUW':None}

    exps = ['13', 'D52', 'D53', 'D55', 'D56', 'D58', 'T71', 'T73', 'TD58']
    scenario = ['RCP8.5', 'RCP8.5','RCP8.5', 'RCP8.5','RCP8.5', 'RCP8.5', 'RCP2.6','RCP2.6','RCP8.5']
    gcm = ['NorESM1-M', 'NorESM1-M', 'MIROC-ESM', 'MIROC-ESM', 'CCSM4', 'CCSM4', 'NorESM1-M', 'NorESM1-M', 'CCSM4']
    gamma0 = [159188.5, 471264.3, 159188.5, 471264.3, 159188.5, 471264.3, 159188.5, 471264.3, 471264.3]
    GamPcntle = ['PIG50', 'PIG95', 'PIG50', 'PIG95', 'PIG50', 'PIG95', 'PIG50', 'PIG95', 'PIG95']
    Collapse = ['OFF', 'OFF', 'OFF','OFF','OFF','OFF','OFF','OFF','ON']


    experiments=['5','6','7','8','9','10', '12', '13', 'D52','D53','D55','D56','D58','TD58', 'B6','B7', 'T71','T73'] #, 'ctrl2']
    dir = netcdf_dir + 'maskedStats_regions/ismip6_{}.maskedStatsRegions'
    alpha=1

    TE21results = pd.read_excel(netcdf_dir + 'ISMIP6_model_results_TE21.xlsx')
    # drop BISICLES so don't duplicate
    TE21results = TE21results[TE21results['model']!='BISICLES']

    # try just reducing to AIS
    AIS_TE21results = TE21results[TE21results['ice_source']=='AIS']

    legend_elements = []
    legend_elements.append(Line2D([0], [0],
                                  linestyle='-',
                                  color='k',label='{}:{}'.format('CPOM', 'BISICLES')))

    experiments = ['5', '7', '9', '10',
                   '5', '6', '7', '8', '9', '10', '12']
    region = ['EAIS', 'EAIS', 'EAIS', 'EAIS',
              'WAIS', 'WAIS', 'WAIS', 'WAIS', 'WAIS', 'WAIS', 'WAIS']
    regdict = {'EAIS':1, 'WAIS':0}

    # set up figure, divider first
    fig = plt.figure(figsize=(10,10))

    gs = fig.add_gridspec(8, 6, height_ratios = [1,0.1, 1,0.1, 1,0.1, 1,0.1],
                          width_ratios = [1,0.1, 1,0.1, 1,0.1],
                          hspace=0.2, wspace=0.1)

    divax1 = fig.add_subplot(gs[1, 1:-1])
    divax2 = fig.add_subplot(gs[1:4, 1])
    divax3 = fig.add_subplot(gs[3, :2])
    for ax in [divax1, divax2,divax3]:
        ax.set_facecolor('lightgrey')
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines:
            ax.spines[s].set_visible(False)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 4])
    ax4 = fig.add_subplot(gs[2, 0])

    ax5 = fig.add_subplot(gs[2, 2])
    ax6 = fig.add_subplot(gs[2, 4])
    ax7 = fig.add_subplot(gs[4, 0])
    ax8 = fig.add_subplot(gs[4, 2])
    ax9 = fig.add_subplot(gs[4, 4])
    ax10 = fig.add_subplot(gs[6, 0])
    ax11 = fig.add_subplot(gs[6, 2])

    spl = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11]
    lo = 0.0
    hi = 0.6
    posan = [lo, hi, lo,
             lo, hi, hi,
             hi, hi, hi,
             hi, hi]

    for i, exp in enumerate(experiments):
        plt.sca(spl[i])
        file = pd.read_csv(dir.format(exp))
        ctrl = pd.read_csv(dir.format('ctrl2'))
        ctrl = ctrl[ctrl['time']>=2015][ctrl['time']<=2100]
        file = file[file['time']>=2015][file['time']<=2100]
        sector = region[i]
        xx = file[file['sector']==regdict[region[i]]]['time']
        y = file[file['sector']==regdict[region[i]]]['iceMassAbove']
        y = y - y.iloc[0]
        yctrl = ctrl[ctrl['sector']==regdict[region[i]]]['iceMassAbove']
        yctrl = yctrl - yctrl.iloc[0]
        plt.plot(xx.to_numpy()[:86], -((((np.asarray(y) - np.asarray(yctrl))/1e12)/362.5)/10)[:86], c='k')
        txt = spl[i].text(0, posan[i],
                          'scenario:{},\nGCM:{},\ngamma:{},\nCollapse:{}'.format(Exps_dict[exp]['scenario'],
                                                                                 Exps_dict[exp]['gcm'],
                                                                                 Exps_dict[exp]['gamma0pcntile'],
                                                                                 Exps_dict[exp]['collapse']),
                          transform=spl[i].transAxes, fontsize=8.5)
        # add results forother models
        if exp in ['5','6','7','8','9']:
            expid = 'exp0' + exp
        elif exp in ['10', '12', '13', 'D52','D53','D55','D56','D58','TD58','B6','B7']:
            expid = 'exp' + exp
        elif exp in ['T71', 'T73']:
            expid = 'exp' + exp.replace('7', '07')
        plt.title('{},   {}'.format(expid.replace('e', 'E'), region[i]), x = 0.7, y=0.95, fontsize=10)
        Tres = AIS_TE21results[AIS_TE21results['ice_source']=='AIS'][AIS_TE21results['region']==region[i]][AIS_TE21results['exp_id']==expid]
        linestyles=[':', '--', '-.']
        for group in np.unique(Tres['group']):
            reduced = Tres[Tres['group']==group]
            j=0
            for model in np.unique(reduced['model']):
                if model=='ISSM':
                    mody = reduced[reduced['model']==model].loc[:,'y2015':'y2100'].T
                else:
                    mody = reduced[reduced['model']==model].loc[:,'y2015':'y2100'].T
                x = np.arange(2015, 2101, 1)
                spl[i].plot(x, mody, label=model, alpha=0.6,
                            c=Other_models_formatter[model], linestyle=Other_models_formatter_ls[model])
                # do marker plot to pick out JPL1 ISSM
                spl[i].plot(x[::10], mody[::10], label=model, alpha=0.6,
                            c=Other_models_formatter[model], linestyle=Other_models_formatter_ls[model],
                            marker=Other_models_formatter_marker[group])
                if i == 0:
                    legend_elements.append(Line2D([0], [0],
                                                  linestyle=Other_models_formatter_ls[model],
                                                  color=Other_models_formatter[model],label='{}:{}'.format(group, model),
                                                  marker=Other_models_formatter_marker[group]))
                j+=1

    # add on y label, x label and legend
    ylab = 'SLC vs control (cm)'
    for ax in [ax1, ax4, ax7, ax10]:
        ax.set_ylabel(ylab)

    for ax in [ax10, ax11]:
        ax.set_xlabel('Year')

    labels = 'abcdefghijk'
    for i, ax in enumerate(spl):
        ax.text(0,1.05, '({})'.format(labels[i]), transform=ax.transAxes)


    leg_ax = fig.add_subplot(gs[6, 3:])
    leg_ax.legend(handles=legend_elements[:14], loc='lower left', ncol=2, fontsize='x-small', framealpha=0.1)
    leg_ax.axis('off')

    plt.savefig(savedir + 'fig12.png', bbox_inches='tight', dpi=dpi)
