## Imports and variables

from drift_quant.plotting_scripts import plot_drift

pixel_size = 100

timestep = 30

## Load data

plot_data = input('Please enter path to file: ')
output_location = input('Please enter where you want the plot to be saved: ')

title = input('Please enter title: ')

all_data = plot_drift.load_plot_data(plot_data)

## Extract

bins = plot_drift.extract_bins(all_data)

lines = plot_drift.extract_cont_data(all_data)

bins_cor, lines_cor = plot_drift.zero_initials(bins, lines)

## Plot

plot_drift.plot_all_data(bins_cor, lines_cor, pixel_size, timestep, title, output_location)

plot_drift.plot_x_histogram(bins_cor, pixel_size, title, output_location)

plot_drift.plot_y_histogram(bins_cor, pixel_size, title, output_location)

plot_drift.mean_and_std(bins_cor, pixel_size, title, output_location)
