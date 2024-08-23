## import

from drift_quant.plotting_scripts import plot_drift

pixel_resol = 1

## 1

folder = input('Enter folder path: ')
out = input('Where things are saved: ')

plots = plot_drift.extract_files(folder)

## 2

meansds = plot_drift.calculate_mean_sd(plots, pixel_resol)

plot_drift.save_mean_sd(meansds, out)

## 3

plot_drift.overall_meansd(out)

plot_drift.plot_dotplot(out)

plot_drift.plot_max_dotplot(out)
