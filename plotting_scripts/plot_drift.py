## Modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.io
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from tqdm import tqdm
import os
import math
import seaborn as sns

## Function to take user input

def load_user_input():

    """ This function loads user input for file paths, folders, or 
    titles for plots. It will not accept blank inputs and users will
    have three attempts to enter a valid string before a system error
    is raised.

    Inputs: none
    Outputs: The user input (str)
    
    """

    counter = 0

    input_path = ''

    while True:

        if counter == 3:

            raise SystemError('Too many incorrect file paths. Exiting')

        input_path = input('')

        if len(input_path) > 0:

            break

        else:

            counter += 1

            print('Invalid path entered. You have 3 -' + str(counter) + 'attempts remaining')

    return input_path

## Functions for Fiji drift plots

def load_data(path):

    """
    This function loads the content of a .csv file into a numpy array.
    Typically used for loading drift trajectory plots exported from Fiji.

    Input: .csv file path (str)
    Output: The .csv file content (numpy array)
    """

    # Check for valid path
    if os.path.isfile(path) is False:

        raise OSError('Invalid path entered or file does not exist.')
    
    data = np.genfromtxt(path, dtype=float,
                           skip_header=1, delimiter=',')

    return data

def extract_bins(all_data):

    """
    Extracts bins (in nm) from drift trajectory plots exported from Fiji. Also
    extracts the timepoints (in frames) corresponding to each bin. x and y
    drift are in different columns.

    Input: numpy array extracted from Fiji plot
    Output: bins of drift trajectory (numpy array)
    """

    return all_data[:, 0:4]

def extract_cont_data(all_data):

    """
    Extracts curves (in nm) from plots exported from Fiji. Also
    extracts the timepoints (in frames). x and y drift are in different columns.

    Input: numpy array extracted from Fiji plot
    Output: curve of drift trajectory (numpy array)
    """

    return all_data[:, 4:8]

def zero_initials(bin_data, line_data):

    """
    Corrects the curve such that the firsts point for x-drift and y-drift
    are zero when t=0. Bins are shifted accordingly. Shifts are independent
    for x-drift and y-drift. 

    Inputs: bins and curves from drift trajectory (numpy array)
    Outputs: corrected bins and curves from drift trajectory.

    """

    if bin_data.shape[1] != 4:

        raise ValueError('Bins cannot have more than four columns.')
    
    elif line_data.shape[1] != 4:

        raise ValueError('Continuous data cannot have more than four columns.')

    bins_cor = bin_data.copy()

    lines_cor = line_data.copy()

    x0 = lines_cor[0, 1]

    y0 = lines_cor[0, 3]

    if x0 > 0:

        lines_cor[:, 1] -= x0

        bins_cor[:, 1] -= x0

    elif x0 < 0:

        lines_cor[:, 1] += np.abs(x0)

        bins_cor[:, 1] += np.abs(x0)

    if y0 > 0:

        lines_cor[:, 3] -= y0

        bins_cor[:, 3] -= y0

    elif y0 < 0:

        lines_cor[:, 3] += np.abs(y0)

        bins_cor[:, 3] += np.abs(y0)

    lines_cor = lines_cor[~np.isnan(lines_cor)].reshape(-1, 4)

    bines_cor = bins_cor[~np.isnan(bins_cor)].reshape(-1, 4)
    
    return bins_cor, lines_cor


## Functions for SMAP drift plots

def load_mat_data(mat_path):

    if os.path.isfile(mat_path) is False:

        raise OSError('Invalid path specified or file does not exist.')

    plot_data = scipy.io.loadmat(mat_path)

    return plot_data

def extract_mat_bins(mat_data):

    bin_data = np.hstack((mat_data['frame_bins'], mat_data['x_bins'],
                          mat_data['frame_bins'], mat_data['y_bins']))

    return bin_data

def extract_mat_cont_data(mat_data):

    cont_data = np.hstack((mat_data['frames'], mat_data['x'],
                           mat_data['frames'], mat_data['y']))

    return cont_data

def frame_filter(data, cutoff):

    return data[(data[:, 0]) < cutoff]

def save_mat_bins(mat_bins, num, out):

    np.savetxt(out + 'plot_values' + str(num) + '.csv',
               mat_bins, fmt='%.5e', delimiter=',')

## Plots for both

def plot_all_data(bin_data, line_data, pixel_size, exp_time, title, out):

    plt.ioff()

    frames1 = bin_data[:, 0] * exp_time

    x_drift_bins = bin_data[:, 1] * pixel_size

    y_drift_bins = bin_data[:, 3] * pixel_size

    frames2 = line_data[:, 0] * exp_time

    x_drift = line_data[:, 1] * pixel_size

    y_drift = line_data[:, 3] * pixel_size

    mpl.rcParams['font.sans-serif'] = "Arial"
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 28

    fig, ax = plt.subplots(figsize=(11, 11), dpi=500)

    ax.scatter(frames1, x_drift_bins, s=220, alpha=1.0,
                facecolors='none', edgecolors='darkred')
    ax.scatter(frames1, y_drift_bins, s=220, alpha=1.0,
                facecolors='none', edgecolors='mediumblue')
    ax.plot(frames2, x_drift, 'darkred', linewidth=4.5, label='x-axis drift')
    ax.plot(frames2, y_drift, 'mediumblue', linewidth=4.5, label='y-axis drift')

    leg = plt.legend(bbox_to_anchor=(0.5, 1.13), loc='upper center', ncol=2)

    for line in leg.get_lines():
        line.set_linewidth(3.5)
    
    for text in leg.get_texts():
        text.set_fontsize(28)

    ax.set_xlim(left=0, right=np.max(frames2) + 100)

    ratio = 1.0

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high)) * ratio)

    ax.tick_params(axis='y', which='major', length=10, direction='in')
    ax.tick_params(axis='y', which='minor', length=5, direction='in')
    ax.tick_params(axis='x', which='major', length=10, direction='in')
    ax.tick_params(axis='x', which='minor', length=5, direction='in')

    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))

    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['top'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)

    ax.set_xlabel('Time (s)', labelpad=6, fontsize=40)
    ax.set_ylabel('Drift (nm)', labelpad=-6, fontsize=40)

    plt.savefig(out + '/' + str(title) + '.png')
    plt.savefig(out + '/' + str(title) + '.svg')

def plot_x_histogram(bin_data, pixel_size, title, out):

    plt.ioff()

    x = np.abs(bin_data[:, 1] * pixel_size)

    x = x[~np.isnan(x)]

    weights = np.ones_like(x) / float(len(x))

    mpl.rcParams['font.sans-serif'] = "Arial"
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 28

    fig, ax = plt.subplots(figsize=(11, 11), dpi=500)

    plt.hist(x, bins=20, weights=weights, edgecolor='black', linewidth=1.5, color='darkred')

    ratio = 1.0

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high)) * ratio)

    ax.tick_params(axis='y', which='major', length=6, direction='in')
    ax.tick_params(axis='y', which='minor', length=3, direction='in')
    ax.tick_params(axis='x', which='major', length=6, direction='out')
    ax.tick_params(axis='x', which='minor', length=3, direction='out')

    #ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))

    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['top'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)

    ax.set_xlabel('Drift along x-axis (nm)', labelpad=6, fontsize=40)
    ax.set_ylabel('Normalized frequency', labelpad=3, fontsize=40)

    plt.savefig(out + '/' + str(title) + 'x_hist.png')
    plt.savefig(out + '/' + str(title) + 'x_hist.svg')


def plot_y_histogram(bin_data, pixel_size, title, out):

    plt.ioff()

    y = np.abs(bin_data[:, 3] * pixel_size)

    y = y[~np.isnan(y)]

    weights = np.ones_like(y) / float(len(y))

    mpl.rcParams['font.sans-serif'] = "Arial"
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 28

    fig, ax = plt.subplots(figsize=(11, 11), dpi=500)

    plt.hist(y, bins=20, weights=weights, edgecolor='black', linewidth=1.5, color='darkred')

    ratio = 1.0

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    ax.tick_params(axis='y', which='major', length=6, direction='in')
    ax.tick_params(axis='y', which='minor', length=3, direction='in')
    ax.tick_params(axis='x', which='major', length=6, direction='out')
    ax.tick_params(axis='x', which='minor', length=3, direction='out')

    #ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))

    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['top'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)

    ax.set_xlabel('Drift along y-axis (nm)', labelpad=6, fontsize=40)
    ax.set_ylabel('Normalized frequency', labelpad=3, fontsize=40)

    plt.savefig(out + '/' + str(title) + 'y_hist.png')
    plt.savefig(out + '/' + str(title) + 'y_hist.svg')

def mean_and_std(bin_data, pixel_size, title, out):

    x = bin_data[:, 1] * pixel_size

    y = bin_data[:, 3] * pixel_size

    mean_x = np.nanmean(np.abs(x))
    mean_y = np.nanmean(np.abs(y))

    std_x = np.nanstd(np.abs(x))
    std_y = np.nanstd(np.abs(y))

    with open(out + '/' + str(title) + 'mean_and_std.txt', 'w') as f:
        f.write('The mean drift along x is: ' + str(mean_x) + 'nm' + '\n')
        f.write('The mean drift along y is: ' + str(mean_y) + 'nm' + '\n')
        f.write('The drift standard deviation along x is: ' + str(std_x) + 'nm' + '\n')
        f.write('The drift standard deviation along y is: ' + str(std_y) + 'nm')

## Load ROI data and plot localisation precision

def load_roi_locs(path):

    """" Function is probably redundant """

    roi_data = np.genfromtxt(path, dtype=float,
                           skip_header=1, delimiter=',')

    return roi_data

def plot_locprec(loc_data, out):

    plt.ioff()

    locprec = loc_data[:, -1]

    locprec = locprec[(locprec < 100)]

    weights = np.ones_like(locprec) / float(len(locprec))

    mpl.rcParams['font.sans-serif'] = "Arial"
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 28

    fig, ax = plt.subplots(figsize=(11, 11), dpi=500)


    ax.hist(locprec, bins=20, weights=weights, edgecolor='black', linewidth=1.5, color='darkred')

    ratio = 1.0

    ax.set_xlim(left=0, right=20)

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    ax.tick_params(axis='y', which='major', length=6, direction='in')
    ax.tick_params(axis='y', which='minor', length=3, direction='in')
    ax.tick_params(axis='x', which='major', length=6, direction='in')
    ax.tick_params(axis='x', which='minor', length=3, direction='in')

    ax.yaxis.set_minor_locator(AutoMinorLocator(10))

    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['top'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)

    ax.set_xlabel('Localization precision (nm)', labelpad=6, fontsize=40)
    ax.set_ylabel('Normalized frequency', labelpad=1, fontsize=40)

    plt.savefig(out + '/loc_prec_hist.png')
    plt.savefig(out + '/loc_prec_hist.svg')

## Functions for FRC calculations

def load_all_data(path):

    """ Function is probably redundant"""

    locs = np.genfromtxt(path, dtype=float,
                  skip_header=1, delimiter=',')

    return locs

def extract_xy(loc_data):

    x = loc_data[:, 2].reshape(loc_data.shape[0], 1)

    y = loc_data[:, 3].reshape(loc_data.shape[0], 1)

    xy_only = np.hstack((x, y))

    return xy_only

def save_xy_locs(xy_data, title, out):

    np.savetxt(out + '/' + str(title) + 'xy_for_FRC.txt', xy_data,
               fmt='%.5e', delimiter='\t')

def load_frc_plot_data(path):

    """Function probably redundant"""

    plot_data = np.genfromtxt(path, dtype=float,
                              skip_header=1, delimiter=',')

    return plot_data

def calc_frc_res(frc_data, title, out):

    v = frc_data[:, 0]
    frc = frc_data[:, 1]
    thold = frc_data[:, 2]

    intercept = np.argwhere(np.diff(np.sign(frc - thold))).flatten()

    res_fourier = v[intercept][0]

    res_real_space = 1 / res_fourier

    with open(out + '/' + str(title) + 'resolution.txt', 'w') as f:

        f.write('The resolution in spatial frequency is: '
                + str(res_fourier) + ' nm^-1' + '\n')

        f.write('The real space resolution is: '
                + str(res_real_space) + ' nm' + '\n')

    return res_fourier, frc[intercept][0]


def plot_frc_curve(frc_data, rcc_data, x1, y1, x2, y2, title, out):

    plt.ioff()

    spatial_freq = frc_data[:, 0]

    frc = frc_data[:, 1]

    thold = frc_data[:, 2]

    spatial_freq_rcc = rcc_data[:, 0]

    frc_rcc = rcc_data[:, 1]

    mpl.rcParams['font.sans-serif'] = "Arial"
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 28

    fig, ax = plt.subplots(figsize=(11, 11), dpi=500)

    ax.plot(spatial_freq, frc, 'darkmagenta', label='Without RCC', linewidth=4.5)
    ax.plot(spatial_freq, thold, 'royalblue', label='Threshold', linewidth=4.5)
    ax.plot(spatial_freq_rcc, frc_rcc, 'salmon', label='With RCC', linewidth=4.5)
    ax.plot(x1, thold[0], 'blueviolet', marker='.', markersize=24, markeredgecolor='k')
    ax.plot(x2, thold[0], 'blueviolet', marker='.', markersize=24, markeredgecolor='k')

    leg = plt.legend(loc='upper right')

    for line in leg.get_lines():
        line.set_linewidth(3.5)
    
    for text in leg.get_texts():
        text.set_fontsize(28)

    ax.set_ylim(bottom=np.min(frc) - 0.05, top=1)
    ax.set_xlim(left=0)

    ratio = 1.0

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    ax.tick_params(axis='y', which='major', length=6, direction='in', pad=10)
    ax.tick_params(axis='y', which='minor', length=3, direction='in')
    ax.tick_params(axis='x', which='major', length=6, direction='in', pad=10)
    ax.tick_params(axis='x', which='minor', length=3, direction='in')

    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))

    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['top'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)

    ax.set_xlabel(r'Spatial frequency $\mathregular{(nm^{-1})}$', labelpad=-3, fontsize=40)
    ax.set_ylabel('Fourier Ring Correlation', labelpad=2, fontsize=40)

    plt.savefig(out + '/' + str(title) + 'frc_plot.png')
    plt.savefig(out + '/' + str(title) + 'frc_plot.svg')

## Calculate mean and SD for all beads

def extract_files(folder_path):

    plot_files = os.listdir(folder_path)

    all_plot_paths = []

    for plot_file in tqdm(plot_files):

        if plot_file[0:4] == 'fidu':

            print('Extracting {}'.format(plot_file))

            all_plot_paths.append(os.path.join(folder_path, plot_file))

    return all_plot_paths

def calculate_mean_sd(all_paths, pixel_res):

    z = len(all_paths)

    mean_std_max_data = np.zeros((2 * z, 3))

    axes = np.empty((2 * z, 1), dtype=str)

    for i in tqdm(range(0, z)):

        print('Quantifying drift in file ' + str(i))

        values = load_data(all_paths[i])

        bins_cor, lines_cor = zero_initials(extract_bins(values),
                                            extract_cont_data(values))

        mean_x = np.nanmean(np.abs(bins_cor[:, 1])) * pixel_res

        std_x = np.nanstd(np.abs(bins_cor[:, 1])) * pixel_res

        max_x = np.nanmax(np.abs(bins_cor[:, 1])) * pixel_res

        mean_y = np.nanmean(np.abs(bins_cor[:, 3])) * pixel_res

        std_y = np.nanstd(np.abs(bins_cor[:, 3])) * pixel_res

        max_y = np.nanmax(np.abs(bins_cor[:, 3])) * pixel_res

        mean_std_max_data[i, 0], mean_std_max_data[i, 1], mean_std_max_data[i, 2] = mean_x, std_x, max_x

        mean_std_max_data[i + z, 0], mean_std_max_data[i + z, 1], mean_std_max_data[i + z, 2] = mean_y, std_y, max_y

        axes[i], axes[i + z] = 'x', 'y'

    all_drift = np.hstack((axes, mean_std_max_data))

    return all_drift

def save_mean_sd(drift_values, out):

    np.savetxt(out + '/all_bead_drift.txt', drift_values,
               fmt='%s', delimiter='\t',
               header='Axis \t Mean drift (nm) \t Drift SD (nm) \t Maximum drift (nm)')


def overall_meansd(out):

    full_path = os.path.join(out, 'all_bead_drift.txt')

    if os.path.isfile(full_path) is False:
        raise OSError('Bead drift data not found in specified folder.')

    df = pd.read_csv(full_path, sep='\t')

    means = df[df.columns[1]]

    sds = df[df.columns[2]]

    maxima = df[df.columns[3]]

    mean_x = np.mean(means[0: int(len(means) / 2)])

    mean_y = np.mean(means[int(len(means) / 2): len(means)])

    mean_sd_x = np.mean(sds[0: int(len(sds) / 2)])

    mean_sd_y = np.mean(sds[int(len(means) / 2): len(sds)])

    mean_max_x = np.mean(maxima[0: int(len(maxima) / 2)])

    mean_max_y = np.mean(maxima[int(len(maxima) / 2): len(maxima)])

    with open(out + '/overall_stats.txt', 'w') as f:

        f.write('Mean drift along x: ' + str(mean_x) + 'nm \n')
        f.write('Mean drift along y: ' + str(mean_y) + 'nm \n')
        f.write('Mean sd along x: ' + str(mean_sd_x) + 'nm \n')
        f.write('Mean sd along y: ' + str(mean_sd_y) + 'nm \n')
        f.write('Mean maxima along x: ' + str(mean_max_x) + 'nm \n')
        f.write('Mean maxima along y: ' + str(mean_max_y) + 'nm \n')


def plot_dotplot(out):

    plt.ioff()

    full_path = os.path.join(out, 'all_bead_drift.txt')

    if os.path.isfile(full_path) is False:

        raise OSError('Bead drift data not found in specified folder.')

    df = pd.read_csv(full_path, sep='\t')

    fig, ax = plt.subplots(figsize=(11, 11), dpi=500)

    sns.set_theme(font='Arial')

    graph = sns.stripplot(x=df.columns[0], y=df[df.columns[1]], data=df,
                          s=15, color='midnightblue')
    sns.pointplot(data=df, x=df.columns[0], y=df[df.columns[1]], errorbar='sd',
                  markers='_', linestyles='none', capsize=0.2,
                  linewidth=4.5, color='darkgreen')
    graph.tick_params(labelsize=30, pad=5)

    ratio = 1.0

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    ax.tick_params(axis='y', which='major', length=6, direction='in')
    ax.tick_params(axis='y', which='minor', length=3, direction='in')
    ax.tick_params(axis='x', which='major', length=6, direction='in')
    ax.tick_params(axis='x', which='minor', length=3, direction='in')

    ax.yaxis.set_minor_locator(AutoMinorLocator(11))

    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['top'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)

    ax.set_xlabel('Axis', labelpad=12, fontsize=40)
    ax.set_ylabel(df.columns[1], labelpad=3, fontsize=40)

    plt.savefig(out + '/dotplot_beads.png')
    plt.savefig(out + '/dotplot_beads.svg')

def plot_max_dotplot(out):

    plt.ioff()

    full_path = os.path.join(out, 'all_bead_drift.txt')

    if os.path.isfile(full_path) is False:

        raise OSError('Bead drift data not found in specified folder.')

    df = pd.read_csv(full_path, sep='\t')

    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True

    fig, ax = plt.subplots(figsize=(12, 12), dpi=500)

    sns.set_theme(font='Arial')

    sns.set_style('ticks')

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    graph = sns.stripplot(x=df.columns[0], y=df[df.columns[3]], data=df,
                          s=15, color='midnightblue')
    sns.pointplot(data=df, x=df.columns[0], y=df[df.columns[3]], errorbar='sd',
                  markers='_', linestyles='none', capsize=0.2,
                  linewidth=4.5, color='darkgreen')
    graph.tick_params(labelsize=30, pad=5)

    ratio = 1.0

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    ax.tick_params(axis='y', which='major', length=6, direction='in')
    ax.tick_params(axis='y', which='minor', length=3, direction='in')
    ax.tick_params(axis='x', which='major', length=6, direction='in')
    ax.tick_params(axis='x', which='minor', length=3, direction='in')

    ax.yaxis.set_minor_locator(AutoMinorLocator(11))

    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['top'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)

    ax.set_xlabel('Axis', labelpad=12, fontsize=40)
    ax.set_ylabel(df.columns[3], labelpad=3, fontsize=40)

    plt.savefig(out + '/dotplot_beads_maxima.png')
    plt.savefig(out + '/dotplot_beads_maxima.svg')
    plt.close(fig)

def filter_frame(in_folder, out_folder, frame):

    file_number = 0

    filtered_files = []

    for file in tqdm(os.listdir(in_folder)):

        if file.endswith('.csv'):

            print('Converting {}'.format(file))

            localisations = load_data(os.path.join(in_folder, file))

            filtered_locs = localisations[(localisations[:, 0]) < frame]

            file_name = out_folder + '/' + str(file_number) + '_' + file

            np.savetxt(file_name, filtered_locs, fmt='%.5e', delimiter=',')

            filtered_files.append(file_name)

            file_number += 1
    
    return filtered_files

## Combined functions

def plot_from_fiji():

    pixel_size, exp_time = 1, 100

    print('First enter input file.')
    input_file = load_user_input()

    print('Then enter output folder.')
    output_folder = load_user_input()

    print('Finally enter title.')
    title = load_user_input()

    data = load_data(input_file)

    bins = extract_bins(data)

    lines = extract_cont_data(data)

    bins_cor, lines_cor = zero_initials(bins, lines)

    plot_all_data(bins_cor, lines_cor, pixel_size, exp_time, title, output_folder)

    plot_x_histogram(bins_cor, pixel_size, title, output_folder)

    plot_y_histogram(bins_cor, pixel_size, title, output_folder)

    mean_and_std(bins_cor, pixel_size, title, output_folder)

def analyse_all_beads():

    pixel_size = 1

    print('First enter folder where all plots are stored.')
    input_folder = load_user_input()

    print('Enter output folder.')
    output_folder = load_user_input()

    all_plots = extract_files(input_folder)

    all_drift_stats = calculate_mean_sd(all_plots, pixel_size)

    save_mean_sd(all_drift_stats, output_folder)

    overall_meansd(output_folder)

    plot_dotplot(output_folder)

    plot_max_dotplot(output_folder)

def analyse_beads_frame_cutoff():

    pixel_size = 1

    print('First enter folder where all plots are stored.')
    input_folder = load_user_input()

    print('Enter output folder.')
    output_folder = load_user_input()

    filtered_plots = filter_frame(input_folder, output_folder, frame=31)

    drift_stats = calculate_mean_sd(filtered_plots, pixel_res=pixel_size)

    save_mean_sd(drift_values=drift_stats, out=output_folder)

    overall_meansd(out=output_folder)

    plot_dotplot(output_folder)

    plot_max_dotplot(output_folder)

def plot_from_mat_file():

    pixel_size, exp_time = 1, 0.03

    print('First enter input file.')
    input_file = load_user_input()

    print('Then enter output folder.')
    output_folder = load_user_input()

    print('Finally enter title.')
    title = load_user_input()

    data = load_mat_data(input_file)

    bins = extract_mat_bins(data)

    lines = extract_mat_cont_data(data)

    plot_all_data(bins, lines, pixel_size, exp_time, title, output_folder)

    plot_x_histogram(bins, pixel_size, title, output_folder)

    plot_y_histogram(bins, pixel_size, title, output_folder)

    mean_and_std(bins, pixel_size, title, output_folder)

def plot_from_mat_frame_cutoff():

    pixel_size, exp_time = 1, 0.03

    print('First enter input file.')
    input_file = load_user_input()

    print('Then enter output folder.')
    output_folder = load_user_input()

    print('Finally enter title.')
    title = load_user_input()

    data = load_mat_data(input_file)

    bins = frame_filter(extract_mat_bins(data), cutoff=33000)

    lines = frame_filter(extract_mat_cont_data(data), cutoff=33000)

    plot_all_data(bins, lines, pixel_size, exp_time, title, output_folder)

    plot_x_histogram(bins, pixel_size, title, output_folder)

    plot_y_histogram(bins, pixel_size, title, output_folder)

    mean_and_std(bins, pixel_size, title, output_folder)


def load_frc_data():

    print('Enter path to localisation files.')
    loc_file = load_user_input()

    print('Enter where you want files to be saved.')
    output_folder = load_user_input()

    print('Finally enter a title for this set of data.')
    title = load_user_input()

    localisations = load_data(loc_file)

    xy_coords = extract_xy(localisations)

    save_xy_locs(xy_coords, title, output_folder)

    print('Data are now ready for FRC analysis.')

def plot_frc_data():

    print('Enter non-RCC data.')
    no_rcc_data = load_user_input()

    print('Enter RCC data.')
    rcc_data = load_user_input()

    print('Enter title')
    title = load_user_input()

    print('Enter output folder.')
    output_folder = load_user_input()

    no_rcc_plot, rcc_plot = load_data(no_rcc_data), load_data(rcc_data)

    x_frc, y_frc = calc_frc_res(no_rcc_plot, title, output_folder)

    x_frc_rcc, y_frc_rcc = calc_frc_res(rcc_plot, title + 'rcc_', output_folder)

    plot_frc_curve(no_rcc_plot, rcc_plot, x_frc, y_frc,
                   x_frc_rcc, y_frc_rcc, title, output_folder)

def plot_localisation_precision():

    print('Enter path to localisation table.')
    localisation_data_path = load_user_input()

    print('Enter output folder.')
    output_folder = load_user_input()

    localisation_table = load_data(localisation_data_path)

    plot_locprec(loc_data=localisation_table, out=output_folder)
