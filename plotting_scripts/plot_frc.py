from drift_quant.plotting_scripts import plot_drift

path = input('Please enter path to FRC plot data: ')

out = input('Please enter folder where you want plot to be saved: ')

title = input('Please give a name to files: ')

plot_data = plot_drift.load_frc_plot_data(path)

x_int, y_int = plot_drift.calc_frc_res(plot_data, title, out)

plot_drift.plot_frc_curve(plot_data, x_int, y_int, title, out)