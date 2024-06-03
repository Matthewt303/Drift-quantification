from drift_quant.plotting_scripts import plot_drift

path = input('Please enter path to FRC plot data: ')

out = input('Please enter folder where you want plot to be saved: ')

plot_data = plot_drift.load_frc_plot_data(path)

plot_drift.plot_frc_curve(plot_data, out)