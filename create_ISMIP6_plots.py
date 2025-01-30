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


def main(args):
    """
    Script plots main figures from paper, if run as main doesn't plot supplementary figures. However,
    code is included here, at some point will adapt so script produces these too.
    """
    netcdf_dir = args.netcdf_dir
    TF_dir = args.TF_dir
    SMB_dir = args.SMB_dir

    # template path for netcdf files
    netcdf_root = netcdf_dir + 'ismip6_{}_8km/{}_AIS_CPOM-LBL_BISICLES_ismip6_{}_8km.nc'
    # make directory for saving plots to
    savedir = netcdf_dir + 'figures/'
    print(savedir)
    if os.path.exists(savedir):
        print('{} exists...'.format(savedir))
    else:
        os.mkdir(savedir)

    # set dpi for plots
    dpi=300

    # set up common experiments dictionary, colors, formatting dict, legend handles for reusing
    exps = ['13', 'D52', 'D53', 'D55', 'D56', 'D58', 'T71', 'T73', 'TD58']
    scenario = ['RCP8.5', 'RCP8.5','RCP8.5', 'RCP8.5','RCP8.5', 'RCP8.5', 'RCP2.6','RCP2.6','RCP8.5']
    gcm = ['NorESM1-M', 'NorESM1-M', 'MIROC-ESM', 'MIROC-ESM', 'CCSM4', 'CCSM4', 'NorESM1-M', 'NorESM1-M', 'CCSM4']
    gamma0 = [159188.5, 471264.3, 159188.5, 471264.3, 159188.5, 471264.3, 159188.5, 471264.3, 471264.3]
    GamPcntle = ['PIG50', 'PIG95', 'PIG50', 'PIG95', 'PIG50', 'PIG95', 'PIG50', 'PIG95', 'PIG95']
    Collapse = ['OFF', 'OFF', 'OFF','OFF','OFF','OFF','OFF','OFF','ON']

    Exps_dict={}
    for i, exp in enumerate(exps):
        Exps_dict[exp]={}
        Exps_dict[exp]['scenario']=scenario[i]
        Exps_dict[exp]['gcm'] = gcm[i]
        Exps_dict[exp]['gamma0'] = gamma0[i]
        Exps_dict[exp]['gamma0pcntile'] = GamPcntle[i]
        Exps_dict[exp]['collapse'] = Collapse[i]
        Exps_dict[exp]['exp_id'] = exp

    ## now add core experiments
    exps = ['5', '6', '7', '8', '9', '10', '12', 'B6', 'B7']
    scenario = ['RCP8.5', 'RCP8.5', 'RCP2.6', 'RCP8.5', 'RCP8.5', 'RCP8.5', 'RCP8.5', 'SSP585', 'SSP126']
    gcm = ['NorESM1-M', 'MIROC-ESM', 'NorESM1-M', 'CCSM4', 'NorESM1-M', 'NorESM1-M', 'CCSM4', 'CNRM-CM6-1', 'CNRM-CM6-1']
    gamma0 = [14477.3, 14477.3, 14477.3, 14477.3, 21005.3, 9618.9, 14477.3, 14477.3, 14477.3]
    GamPcntle = ['ANT50', 'ANT50', 'ANT50', 'ANT50','ANT95', 'ANT5', 'ANT50', 'ANT5', 'ANT50']
    Collapse = ['OFF','OFF','OFF','OFF','OFF','OFF','ON', 'OFF', 'OFF']
    for i, exp in enumerate(exps):
        Exps_dict[exp]={}
        Exps_dict[exp]['scenario']=scenario[i]
        Exps_dict[exp]['gcm'] = gcm[i]
        Exps_dict[exp]['gamma0'] = gamma0[i]
        Exps_dict[exp]['gamma0pcntile'] = GamPcntle[i]
        Exps_dict[exp]['collapse'] = Collapse[i]
        Exps_dict[exp]['exp_id'] = exp

    ## add B experiments
    exps = ['B6', 'B7']
    scenario = ['SSP585', 'SSP126']
    gcm = ['CNRM-CM6-1', 'CNRM-CM6-1']
    gamma0 = [14477.3, 14477.3]
    GamPcntle = ['ANT50', 'ANT50']
    Collapse = ['OFF', 'OFF']
    for i, exp in enumerate(exps):
        Exps_dict[exp]={}
        Exps_dict[exp]['scenario']=scenario[i]
        Exps_dict[exp]['gcm'] = gcm[i]
        Exps_dict[exp]['gamma0'] = gamma0[i]
        Exps_dict[exp]['gamma0pcntile'] = GamPcntle[i]
        Exps_dict[exp]['collapse'] = Collapse[i]
        Exps_dict[exp]['exp_id'] = exp

    colors = {'B6':'fuchsia', 'B7':'fuchsia', '5':'c', '6':'gold', '7':'skyblue',
              '8':'lightcoral', '9':'deepskyblue', '10':'aqua', '12':'lightcoral',
              '13':'royalblue', 'D52':'navy', 'D53':'darkorange', 'D55':'chocolate',
              'D56':'red', 'D58':'firebrick', 'T71':'royalblue', 'T73':'navy',
              'TD58':'firebrick'}

    Formatting_dict_allExps = {'linestyles':{'RCP2.6':':', 'RCP8.5':'-', 'SSP126':'-.', 'SSP585':'--'},
                               'colors':{'NorESM1-M':'blue', 'CCSM4':'red', 'MIROC-ESM':'cyan', 'CNRM-CM6-1':'magenta'}}

    legend_elements =[Line2D([0], [0],marker='None', color='None', label=r'$\bf{CNRM-CM6}$'),
                      Patch(facecolor='fuchsia', edgecolor='fuchsia', label='ANT50'),
                      Line2D([0], [0],marker='None', color='None', label=r'$\bf{NorESM1-M}$'),
                      Patch(facecolor='aqua', edgecolor='aqua', label='ANT5'),
                      Patch(facecolor='c', edgecolor='c', label='ANT50'),
                      Patch(facecolor='deepskyblue', edgecolor='deepskyblue', label='ANT95'),
                      Patch(facecolor='royalblue', edgecolor='royalblue', label='PIG50'),
                      Patch(facecolor='navy', edgecolor='navy', label='PIG95'),
                      Line2D([0], [0],marker='None', color='None', label=r'$\bf{MIROC-ESM-CHEM}$'),
                      Patch(facecolor='gold', edgecolor='gold', label='ANT50'),
                      Patch(facecolor='darkorange', edgecolor='darkorange', label='PIG50'),
                      Patch(facecolor='chocolate', edgecolor='chocolate', label='PIG95'),
                      Line2D([0], [0],marker='None', color='None', label=r'$\bf{CCSM4}$'),
                      Patch(facecolor='lightcoral', edgecolor='lightcoral', label='ANT50'),
                      Patch(facecolor='red', edgecolor='red', label='PIG50'),
                      Patch(facecolor='firebrick', edgecolor='firebrick', label='PIG95'),
                      Line2D([0], [0],marker='^', color='w', label='RCP8.5, collapse on',
                             markeredgecolor='k', markersize=10),
                      Line2D([0], [0],linestyle=':', color='k', label='RCP2.6',
                             markeredgecolor='k', markersize=10),
                      Line2D([0], [0],linestyle='-', color='k', label='RCP8.5',
                             markeredgecolor='k', markersize=10),
                      Line2D([0], [0],linestyle='-.', color='k', label='SSP126',
                             markeredgecolor='k', markersize=10),
                      Line2D([0], [0],linestyle='--', color='k', label='SSP585'),
                      Line2D([0], [0],marker='None', color='None', label=''),
                      Line2D([0], [0],marker='None', color='None', label=''),
                      Line2D([0], [0],marker='|', color='k', label='Control')]


    if (TF_dir != "NA" ) & (os.path.exists(TF_dir)):
        figure1(TF_dir, netcdf_dir, savedir, dpi)
    else:
        print("ISMIP6 thermal forcing data not downloaded")
    if (SMB_dir != "NA") & (os.path.exists(SMB_dir)):
        figure2(SMB_dir, netcdf_dir, savedir, dpi)
    else:
        print("ISMIP6 SMB data not downloaded")
    figure3(netcdf_root, savedir, dpi)
    figure4(netcdf_root, savedir, dpi, Exps_dict=Exps_dict, colors=colors,
            Formatting_dict_allExps=Formatting_dict_allExps, legend_elements=legend_elements)
    figure5(netcdf_root, savedir, dpi, Exps_dict=Exps_dict, colors=colors,
            Formatting_dict_allExps=Formatting_dict_allExps)
    figure6(netcdf_root, savedir, dpi, Exps_dict=Exps_dict, colors=colors,
            Formatting_dict_allExps=Formatting_dict_allExps)
    figure7(netcdf_dir, savedir, dpi, Exps_dict=Exps_dict, colors=colors,
            Formatting_dict_allExps=Formatting_dict_allExps, legend_elements=legend_elements)
    figure8(netcdf_dir, savedir, dpi, Exps_dict=Exps_dict, colors=colors,
            Formatting_dict_allExps=Formatting_dict_allExps, legend_elements=legend_elements)
    figure9(netcdf_dir, netcdf_root, savedir, dpi, Exps_dict=Exps_dict, colors=colors,
            Formatting_dict_allExps=Formatting_dict_allExps, legend_elements=legend_elements)
    figure10(netcdf_dir, savedir, dpi, Exps_dict=Exps_dict)
    figure11(netcdf_dir, netcdf_root, TF_dir, savedir, dpi)
    figure12(netcdf_dir, savedir, dpi, Exps_dict=Exps_dict)


if __name__ == "__main__":
    # Initialize argparse
    parser = argparse.ArgumentParser(
        description="Reproduce plotfiles for ONeill et al. BISICLES ISMIP6 paper"
    )
    # add arguments
    parser.add_argument("netcdf_dir", type=str, help="Path to netcdf files downloaded from https://zenodo.org/records/13880450")
    parser.add_argument("TF_dir", type=str, help="Path to netcdfs with thermal forcing anomaly data for ISMIP6 (if not downloaded, type NA to skip figure 1))")
    parser.add_argument("SMB_dir", type=str, help="Path to netcdfs with SMB anomaly data for ISMIP6 (if not downloaded, type NA to skip figure 2)")

    args = parser.parse_args()
    main(args)
