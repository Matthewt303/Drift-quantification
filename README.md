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

There are also scripts to plot the FRC resolution and localisation precision.

# Drift quantification
The drift of fluorescent beads was calculated by applying a built-in algorithm in ThunderSTORM that detects and 
tracks fiducial markers. From these data, the mean of the absolute values...
