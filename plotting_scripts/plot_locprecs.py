from drift_quant.plotting_scripts import plot_drift

## Load Data

loc_path = input('Please enter path to localisations: ')

out = input('Please enter path for saving plots: ')

roi_data = plot_drift.load_roi_locs(loc_path)

## Plot

plot_drift.plot_locprec(roi_data, out)