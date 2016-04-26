# Calibrated Analysis
This directory provides a way to find the central values and resolution of a set of calibrated data (e.g. response in a given bin of truth pT).

Three files are provided:
`fithist.py`: A package for calculating the central value and resolution of a set of data and weights.  
`calibrated_resolution.py`: A sample python script for reading in a Root histogram from file and converting it to a list of data/weights.   `sample_script.sh`: A sample bash script for running `calibrated_resolution`.

## Fithist
There is one provided function in the package, `fithist(data,weightdata,central,eff=eff,plotDir=None)`.

The arguments are:  
`data` - A numpy array of data values (e.g. a list of response values).  
`weightdata` - A numpy array of weight values (e.g. a list of event weights for each response value).  
`central` - A string indicating which definition of central tendency and resolution to use. The options are `mean`,`median`,`trimmed`,`mode`, and `absolute_median`. See the top-level README for explanations of each option.  
`eff` - The reconstruction efficiency. Only required if the options `mode` or `absolute_median` are selected.  
`plotDir` - The directory to store plots. If unspecified, the function does not store any plots.

## Calibrated_resolution
This script provides an example of reading a histogram in from a Root file and converting it to a list of data/weights to use in `fithist`. For any function that uses median/IQR calculations (`median`,`mode`,`absolute_median`), the best performance is provided in the limit of infinitely small bin sizes. 
