Code for reproducing main text figures and results from O'Neill et al. 2025 'ISMIP6-based Antarctic Projections to 2100: simulations with the BISICLES ice sheet model' https://egusphere.copernicus.org/preprints/2024/egusphere-2024-441/. To plot figures 1 and 2, need to have thermal forcing and surface mass balance data from ISMIP6. Code for supplementary figures is also included for completeness, but is not currently run in the main script create_ISMIP6_plots.py. 

To recreate conda environment, run:

conda env create -f ismip6_plotting_env.yml

then:

conda activate ismip6_plotting_env

To reproduce figures, you can then run:

python create_ISMIP6_plots.py /path/to/BISICLES_ismip6_output_8km_2/ /path/to/thermal_forcings_ismip6/ /path/to/smb_anomaly_ismip6/
