from drift_quant.plotting_scripts import plot_drift

path = input('Please enter path to localisations: ')

out = input('Please enter folder where you want xy locs to be saved: ')

name = input('Give the file a name: ')

all_locs = plot_drift.load_all_data(path)

xy = plot_drift.extract_xy(all_locs)

plot_drift.save_xy_locs(xy, name, out)

print('Your data is now ready for FRC analysis using Fiji.')