## Imports and variables

from drift_quant.plotting_scripts import plot_drift

pixel_size = 1

exp_time = 0.03

## Load data

mat_data = input("Please enter path to file: ")
output = input("Please enter where you want the plot to be saved: ")

all_data = plot_drift.load_mat_data(mat_data)

## Extract

bins = plot_drift.extract_mat_bins(all_data)

lines = plot_drift.extract_mat_cont_data(all_data)

## Plot

plot_drift.plot_all_data(bins, lines, 0.03, output)

plot_drift.plot_x_histogram(bins, output)

plot_drift.plot_y_histogram(bins, output)

## Mean

plot_drift.mean_and_std(bins, output)