## Imports and variables

from drift_quant.plotting_scripts import plot_drift

pixel_size = 97.5

timestep = 0.03

title = str()

## Load data

plot_data = input('Please enter path to file: ')
output_location = input('Please enter where you want the plot to be saved: ')

title = input('Please enter title: ')

all_data = plot_drift.load_plot_data(plot_data)

## Extract

bins = plot_drift.extract_bins(all_data)

lines = plot_drift.extract_cont_data(all_data)

## Plot

plot_drift.plot_all_data(bins, lines, pixel_size, timestep, title, output_location)

plot_drift.plot_x_histogram(bins, pixel_size, title, output_location)

plot_drift.plot_y_histogram(bins, pixel_size, title, output_location)

plot_drift.mean_and_std(bins, pixel_size, title, output_location)
