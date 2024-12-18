# SMLM drift analysis
A series of python scripts to analyse and plot the sample drift of fluorescent beads and molecules, used as part 
of the paper titled: "Reinforced optical cage systems enable drift-free single-molecule localization 
microscopy".
 
# About
The aim of this project was to develop a novel optical microscope construction to reduce sample drift without 
the use of fiducial markers, additional hardware, or postprocessing. As part of this, we wrote scripts to 
quantify and plot drift following fiducial marker identification and tracking using ThunderSTORM. We also wrote 
scripts to quantify and plot drift following redundant cross-correlation of localisation data from STORM. Note 
that these scripts are purely for quantifying lateral drift. There are no methods to analyse axial drift in this 
repository.

# Drift quantification
The drift of fluorescent beads was calculated by applying a built-in algorithm in ThunderSTORM that detects and 
tracks fiducial markers. From these data, ThunderSTORM returns the drift trajectory, as well as the positions of 
the beads over time. We calculated the mean of the absolute values of the positions of the beads as our metric 
for sample drift.

For STORM data, we carried out redundant cross-correlation, so that we can track drift across the image series. 
We calculated sample drift by taking the mean of the absolute values of the drift for each image subset.

The remainder of the scripts are used to plot drift trajectories, convert localization data to a .txt file for 
FRC (Fourier ring correlation) analysis, plot FRC data, plot localization precision, and plot dotplots for the 
experimental repeats of fluorescent bead drift.

# Installation
Please feel free to download or clone the repository to use the code.

# Usage
It is highly recommended to use a virtual environment and to use conda or pip to install the necessary packages. 
The scripts require the following packages:
- Numpy 2.1.1
- Pandas 2.2.2
- Matplotlib 3.9.2
- Scipy 1.14.1
- tqdm 4.66.5
- Seaborn 0.13.2

# Contact
Please contact matthew.tang@stfc.ac.uk or matthew.tang@linacre.ox.ac.uk for any questions.
