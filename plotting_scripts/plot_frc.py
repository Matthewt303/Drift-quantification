from drift_quant.plotting_scripts import plot_drift

path = input('Please enter path to FRC plot data without rcc: ')

path2 = input('Please enter path to FRC plot data with rcc:eex ')

out = input('Please enter folder where you want plot to be saved: ')

title = input('Please give a name to files: ')

plot_data = plot_drift.load_frc_plot_data(path)

plot_data2 = plot_drift.load_frc_plot_data(path2)

x_int, y_int = plot_drift.calc_frc_res(plot_data, title, out)

x_rcc, y_rcc = plot_drift.calc_frc_res(plot_data2, title + 'rcc_', out)

plot_drift.plot_frc_curve(plot_data, plot_data2, x_int, y_int, x_rcc, y_rcc, title, out)