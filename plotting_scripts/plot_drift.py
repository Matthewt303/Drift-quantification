## Modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.io
from matplotlib.ticker import MultipleLocator
import os
import math
import seaborn as sns

## Functions for Fiji drift plots

def load_plot_data(path):
    
    plot_data = np.genfromtxt(path, dtype=float,
                           skip_header=1, delimiter=',')

    return plot_data

def extract_bins(all_data):

    bins = all_data[:, 0:4]

    return bins

def extract_cont_data(all_data):

    cont_data = all_data[:, 4:8]

    return cont_data

## Functions for SMAP drift plots

def load_mat_data(mat_path):

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

def save_mat_bins(mat_bins, num, out):

    np.savetxt(out + 'plot_values' + str(num) + '.csv',
               xy_data, fmt='%.5e', delimiter=',')

## Plots for both

def plot_all_data(bin_data, line_data, pixel_size, exp_time, title, out):

    plt.ioff()

    frames1 = bin_data[:, 0] * exp_time

    x_drift_bins = bin_data[:, 1] * pixel_size

    y_drift_bins = bin_data[:, 3] * pixel_size

    frames2 = line_data[:, 0] * exp_time

    x_drift = line_data[:, 1] * pixel_size

    y_drift = line_data[:, 3] * pixel_size

    maxima = np.maximum(x_drift, y_drift)

    minima = np.minimum(x_drift, y_drift)

    maxima_bins = np.maximum(x_drift_bins, y_drift_bins)

    minima_bins = np.minimum(x_drift_bins, y_drift_bins)

    true_max = int()

    true_min = int()

    if np.max(maxima) > np.max(maxima_bins):

        true_max = round(np.nanmax(maxima), 5)

    else:

        true_max = round(np.nanmax(maxima_bins), 5)

    if np.min(minima) < np.min(minima_bins):

        true_min = round(np.nanmin(minima), 5)

    else:

        true_min = round(np.nanmin(minima_bins), 5)

    increment = round((true_max - true_min) / 10, 1)

    x_incr = (np.max(frames2) - np.min(frames2)) / 10

    t_incr = int(math.ceil(x_incr / 100.0)) * 100

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 12

    fig, ax = plt.subplots(figsize=(10, 10), dpi=500)

    ax.scatter(frames1, x_drift_bins, s=25,
                facecolors='none', edgecolors='r')
    ax.scatter(frames1, y_drift_bins, s=25,
                facecolors='none', edgecolors='b')
    ax.plot(frames2, x_drift, 'r', label='x-axis drift')
    ax.plot(frames2, y_drift, 'b', label='y-axis drift')

    ax.legend(loc='upper left', fontsize=10)

    ratio = 1.0

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high)) * ratio)

    ax.tick_params(axis='y', which='major', length=6, direction='in')
    ax.tick_params(axis='y', which='minor', length=3, direction='in')
    ax.tick_params(axis='x', which='major', length=6, direction='in')
    ax.tick_params(axis='x', which='minor', length=3, direction='in')

    ax.xaxis.set_major_locator(MultipleLocator(t_incr))
    ax.xaxis.set_minor_locator(MultipleLocator(t_incr/ 10))
    ax.yaxis.set_major_locator(MultipleLocator(increment))
    ax.yaxis.set_minor_locator(MultipleLocator(increment / 10))

    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)

    ax.set_xlabel('Time (s)', labelpad=12, fontsize=16)
    ax.set_ylabel('Drift (nm)', labelpad=12, fontsize=16)

    plt.savefig(out + '/' + str(title) + '.png')


def plot_x_histogram(bin_data, pixel_size, title, out):

    plt.ioff()

    x = np.abs(bin_data[:, 1] * pixel_size)

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 12

    fig, ax = plt.subplots(figsize=(10, 10), dpi=500)

    plt.hist(x, bins=10, edgecolor='black', linewidth=1.1, color='C3')

    ratio = 1.0

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high)) * ratio)

    ax.tick_params(axis='y', which='major', length=6, direction='in')
    ax.tick_params(axis='y', which='minor', length=3, direction='in')
    ax.tick_params(axis='x', which='major', length=6, direction='in')
    ax.tick_params(axis='x', which='minor', length=3, direction='in')

    #ax.xaxis.set_major_locator(MultipleLocator(10000))
    #ax.xaxis.set_minor_locator(MultipleLocator(1000))
    #ax.yaxis.set_major_locator(MultipleLocator(increment))
    #ax.yaxis.set_minor_locator(MultipleLocator(increment / 10))

    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)

    ax.set_xlabel('Drift along x-axis (nm)', labelpad=12, fontsize=16)
    ax.set_ylabel('Number', labelpad=12, fontsize=16)

    plt.savefig(out + '/' + str(title) + 'x_hist.png')


def plot_y_histogram(bin_data, pixel_size, title, out):

    plt.ioff()

    y = np.abs(bin_data[:, 3] * pixel_size)

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 12

    fig, ax = plt.subplots(figsize=(10, 10), dpi=500)

    plt.hist(y, bins=10, edgecolor='black', linewidth=1.1, color='C3')

    ratio = 1.0

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    ax.tick_params(axis='y', which='major', length=6, direction='in')
    ax.tick_params(axis='y', which='minor', length=3, direction='in')
    ax.tick_params(axis='x', which='major', length=6, direction='in')
    ax.tick_params(axis='x', which='minor', length=3, direction='in')

    # ax.xaxis.set_major_locator(MultipleLocator(10000))
    # ax.xaxis.set_minor_locator(MultipleLocator(1000))
    # ax.yaxis.set_major_locator(MultipleLocator(increment))
    # ax.yaxis.set_minor_locator(MultipleLocator(increment / 10))

    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)

    ax.set_xlabel('Drift along y-axis (nm)', labelpad=12, fontsize=16)
    ax.set_ylabel('Number', labelpad=12, fontsize=16)

    plt.savefig(out + '/' + str(title) + 'y_hist.png')

def mean_and_std(bin_data, pixel_size, title, out):

    x = bin_data[:, 1] * pixel_size

    y = bin_data[:, 3] * pixel_size

    mean_x = np.nanmean(np.abs(x))
    mean_y = np.nanmean(np.abs(y))

    std_x = np.nanstd(x)
    std_y = np.nanstd(y)

    with open(out + '/' + str(title) + 'mean_and_std.txt', 'w') as f:
        f.write('The mean drift along x is: ' + str(mean_x) + 'nm' + '\n')
        f.write('The mean drift along y is: ' + str(mean_y) + 'nm' + '\n')
        f.write('The drift standard deviation along x is: ' + str(std_x) + 'nm' + '\n')
        f.write('The drift standard deviation along y is: ' + str(std_y) + 'nm')

## Load ROI data and plot localisation precision

def load_roi_locs(path):

    roi_data = np.genfromtxt(path, dtype=float,
                           skip_header=1, delimiter=',')

    return roi_data

def plot_locprec(loc_data, out):

    plt.ioff()

    locprec =  loc_data[:, -1]

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 12

    fig, ax = plt.subplots(figsize=(10, 10), dpi=500)

    plt.hist(locprec, bins='rice', edgecolor='black', linewidth=1.1, color='C3')

    ratio = 1.0

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    ax.tick_params(axis='y', which='major', length=6, direction='in')
    ax.tick_params(axis='y', which='minor', length=3, direction='in')
    ax.tick_params(axis='x', which='major', length=6, direction='in')
    ax.tick_params(axis='x', which='minor', length=3, direction='in')

    # ax.xaxis.set_major_locator(MultipleLocator(10000))
    # ax.xaxis.set_minor_locator(MultipleLocator(1000))
    # ax.yaxis.set_major_locator(MultipleLocator(increment))
    # ax.yaxis.set_minor_locator(MultipleLocator(increment / 10))

    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)

    ax.set_xlabel('Localisation Precision (nm)', labelpad=12, fontsize=16)
    ax.set_ylabel('Number', labelpad=12, fontsize=16)

    plt.savefig(out + '/roi_loc_prec_hist.png')

## Functions for FRC calculations

def load_all_data(path):

    locs = np.genfromtxt(path, dtype=float,
                  skip_header=1, delimiter=',')

    return locs

def extract_xy(loc_data):

    x = loc_data[:, 2]

    y = loc_data[:, 3]

    xy_only = np.hstack((x, y))

    return xy_only

def save_xy_locs(xy_data, title, out):

    np.savetxt(out + str(title) + 'xy_for_FRC.txt', xy_data,
               fmt='%.5e')

def load_frc_plot_data(path):

    plot_data = np.genfromtxt(path, dtype=float,
                              skip_header=1, delimiter=',')

    return plot_data

def plot_frc_curve(frc_data, out):

    plt.ioff()

    spatial_freq = frc_data[:, 0]

    frc = frc_data[:, 1]

    thold = frc_data[:, 2]

    x_incr = (np.max(spatial_freq) - np.min(spatial_freq)) / 10

    x_incr = round(x_incr, 3)

    y_incr = (np.max(frc) - np.min(frc)) / 10

    y_incr = round(y_incr, 3)

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 12

    fig, ax = plt.subplots(figsize=(10, 10), dpi=500)

    ax.plot(spatial_freq, frc, 'C3', label='correlation')
    ax.plot(spatial_freq, thold, 'b', label='threshold')

    ax.legend(loc='upper left', fontsize=10)

    ratio = 1.0

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    ax.tick_params(axis='y', which='major', length=6, direction='in')
    ax.tick_params(axis='y', which='minor', length=3, direction='in')
    ax.tick_params(axis='x', which='major', length=6, direction='in')
    ax.tick_params(axis='x', which='minor', length=3, direction='in')

    ax.xaxis.set_major_locator(MultipleLocator(x_incr))
    ax.xaxis.set_minor_locator(MultipleLocator(x_incr / 10))
    ax.yaxis.set_major_locator(MultipleLocator(y_incr))
    ax.yaxis.set_minor_locator(MultipleLocator(y_incr / 10))

    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)

    ax.set_xlabel('Spatial frequency (nm)', labelpad=12, fontsize=16)
    ax.set_ylabel('Correlation', labelpad=12, fontsize=16)

    plt.savefig(out + '/frc_plot.png')

## Calculate mean and SD for all beads

def extract_files(folder_path):

    plot_files = os.listdir(folder_path)

    all_plot_paths = []

    for plot_file in plot_files:

        if plot_file[0:4] == 'plot':

            plot_path = os.path.join(folder_path, plot_file)

            all_plot_paths.append(plot_path)

    return all_plot_paths

def calculate_mean_sd(all_paths, pixel_res):

    z = len(all_paths)

    mean_std_data = np.zeros((2 * z, 2))

    axes = np.empty((2 * z, 1), dtype=str)

    for i in range(0, z):

        values = np.genfromtxt(all_paths[i], dtype=float,
                      skip_header=1, delimiter= ',')

        mean_x = np.nanmean(values[:, 1]) * pixel_res

        std_x = np.nanstd(values[:, 1]) * pixel_res

        mean_y = np.nanmean(values[:, 3]) * pixel_res

        std_y = np.nanstd(values[:, 3]) * pixel_res

        mean_std_data[i, 0], mean_std_data[i, 1] = mean_x, std_x

        mean_std_data[i + z, 0], mean_std_data[i + z, 1] = mean_y, std_y

        axes[i], axes[i + z] = 'x', 'y'

    all_drift = np.hstack((axes, mean_std_data))

    return all_drift

def save_mean_sd(drift_values, folder_path):

    np.savetxt(folder_path + '/all_bead_drift.txt', drift_values,
               fmt='%s', delimiter='\t',
               header='Axis \t Mean drift (nm) \t Drift SD (nm)')


def overall_meansd(folder_path, out):
    full_path = os.path.join(folder_path, 'all_bead_drift.txt')

    if os.path.isfile(full_path) is False:
        raise OSError('Bead drift data not found in specified folder.')

    df = pd.read_csv(full_path, sep='\t')

    means = df[df.columns[1]]

    means = means.abs()

    sds = df[df.columns[2]]

    mean_x = np.mean(means[0: int(len(means) / 2)])

    mean_y = np.mean(means[int(len(means) / 2): len(means)])

    mean_sd_x = np.mean(sds[0: int(len(sds) / 2)])

    mean_sd_y = np.mean(sds[int(len(means) / 2): len(sds)])

    with open(folder_path + '/overall_stats.txt', 'w') as f:

        f.write('Mean drift along x: ' + str(mean_x) + 'nm \n')
        f.write('Mean drift along y: ' + str(mean_y) + 'nm \n')
        f.write('Mean sd along x: ' + str(mean_sd_x) + 'nm \n')
        f.write('Mean sd along y: ' + str(mean_sd_y) + 'nm')


def plot_stripplot(folder_path, out):

    plt.ioff()

    full_path = os.path.join(folder_path, 'all_bead_drift.txt')

    if os.path.isfile(full_path) is False:

        raise OSError('Bead drift data not found in specified folder.')

    df = pd.read_csv(full_path, sep='\t')

    means = df[df.columns[1]]

    mean_x = np.mean(means[0: int(len(means) / 2)])

    mean_y = np.mean(means[int(len(means) / 2): len(means)])

    fig, ax = plt.subplots(figsize=(10, 10), dpi=500)

    sns.set(font_scale=1.2)

    sns.set_context(rc={'xtick.labelsize': 14, 'ytick.labelsize': 14})
    sns.set_theme(font='sans-serif')

    graph = sns.stripplot(x=df.columns[0], y=df.columns[1], data=df,
                          color='#00008b')
    graph.axhline(mean_x, xmin=0.1, xmax=0.4)
    graph.axhline(mean_y, xmin=0.6, xmax=0.9)

    ratio = 1.0

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    ax.tick_params(axis='y', which='major', length=6, direction='in')
    ax.tick_params(axis='y', which='minor', length=3, direction='in')
    ax.tick_params(axis='x', which='major', length=6, direction='in')
    ax.tick_params(axis='x', which='minor', length=3, direction='in')

    #ax.xaxis.set_major_locator(MultipleLocator(x_incr))
    #ax.xaxis.set_minor_locator(MultipleLocator(x_incr / 10))
    #ax.yaxis.set_major_locator(MultipleLocator(y_incr))
    #ax.yaxis.set_minor_locator(MultipleLocator(y_incr / 10))

    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)

    ax.set_xlabel('Axis', labelpad=12, fontsize=18)
    ax.set_ylabel(df.columns[1], labelpad=12, fontsize=18)

    plt.savefig(out + '/all_beads.png')
