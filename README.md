# Resolution - A script to calculate jet resolutions

This is a script to calculate the resolution of a reconstructed jet collection, taking into account the effects of the jet response function.
The script does not actually go through the process of numerical inversion, but rather estimates the jet resolution as sigma(x)/f'(x), as outlined in [https://cds.cern.ch/record/2045523/](https://cds.cern.ch/record/2045523/).
The jets are broken down into bins of NPV and pT, so it's recommended to have statistics of at least 50000 jets in order to have reasonable error bars.

## Dependencies
Should work with Python 2.7.2 and above. Uses numpy and PyRoot.

## Input
The input to the script can either be a Root Ntuple or numpy arrays.

### Root
If you want to use Root files as input, use the `-r` option.

Root files are searched for in the `inputDir` directory. The script picks up all Root files stored in this directory.

The only required branches are `jetpt` (the pT of reconstructed jets), `tjetpt` (the pT of the truth jets matched to those reconstructed jets), and `NPV` (the NPV of the event).
Other optional branches are `tjeteta` (the eta of the truth jets), `tjetmindr` (the minimum dR from the truth jet to any other truth jet - for most studies, it's recommend to examine only isolated truth jets), and `event_weight` (the event weight).
The script reads in every Root file in the `intputDir` directory, but you can limit the number of events run over with the `numEvents` keyword.

The script automatically writes out the files to numpy arrays and stores them in the `submitDir` directory. Because of this, you only need to run with the `-r` keyword once, and after that the script will read the .npy files it created, which is much faster than reading Root files.

### Numpy
The default is to use numpy files as the input.

Numpy files are searched for in the `submitDir` directory. The only required files are:  
`'truepts_'+identifier+'.npy'` (the pT of reconstructed jets)  
`'recopts_'+identifier+'.npy'` (the pT of the truth jets matched to those reconstructed jets)  
`'npvs_'+identifier+'.npy'` (the NPV of the event that the jet is in - should be the same length as the above two arrays)  

The optional files are:  
`'etas_'+identifier+'.npy'` (the eta of the truth jets)  
`'mindrs_'+identifier+'.npy'` (the minimum dR from the truth jet to any other truth jet - for most studies, it's recommend to examine only isolated truth jets)  
`'weights_'+identifier+'.npy'` (the weight of the event that the jet is in - should be the same length as the above two arrays)  

## Output

### Plots
Plots are saved in the `plotDir` directory. The `-i` option sets the `identifier` keyword, which is a string that will end up in all plots made with this tool in order to identify the sample/jet collection being studied.

#### Output Plots
In NPV bins (indicated by `'_NPV##_'` string in the filename):
`'resbin%d'%ptbin`: Distribution of reconstructed pT response binned in truth pT.
`'fbin%d'%ptbin`: Distribution of reconstructed pT binned in truth pT.
`'jetresponse_pttrue'`: Jet response curve.
`'jetf_pttrue'`: Jet reconstructed pT curve.
`'jetf1_pttrue'`: Estimated pT of average reconstructed pT. This is *not* the closure, but rather a test of how well the fitted response curve fits the average response data points. See the paper as to how this differs from the closure.
`'jetsigma_pttrue'`: Estimated calibrated jet pT resolution vs. truth pT.
`'jetsigmaR_pttrue'`: Estimated calibrated fractional jet pT recolusion vs. truth pT.

In truth pT bins (indicated by `'_pt##'` string in the filename):
`'jetsigma_NPV'`: Estimated calibrated jet pT resolution vs. NPV.
`'jetsigmaR_NPV'`: Estimated calibrated fractional jet pT recolusion vs. NPV.

Inclusive in NPV:
`'jetsigma_pttrue'`: Estimated calibrated jet pT resolution vs. truth pT.
`'jetsigmaR_pttrue'`: Estimated calibrated fractional jet pT recolusion vs. truth pT.

### Data
Some output data is stored in the `submitDir` directory, which allows reconstruction of some of the created plots. In particular this is useful for comparing multiple jet collections, as the tool does not yet support that functionality. The data are stored in the standard Python [pickle](https://docs.python.org/2/library/pickle.html) format.

The stored data are:
`'fit_'+options.identifier+'.p''`: The fit parameters to the response function. The curve is parameterized as a+b/log(x+10)+c/(log(x+10))^2, where x is the truth pT in GeV.
`''pttruebins_'+options.identifier+'.p'`: The average truth pT in the truth pT bins. Basically, the x-axis of the data points in many of the above plots.
`'sigmas_'+options.identifier+'.p''`: The estimated calibrated jet pT resolution, stored as a dictionary of arrays. The keys of the dictionary are the max NPV of the NPV bin, and the array is listed in order of increasing pT bin.
`'sigmaRs_'+options.identifier+'.p''`: The estimated calibrated fractional jet pT resolution, stored as a dictionary of arrays. The keys of the dictionary are the max NPV of the NPV bin, and the array is listed in order of increasing pT bin.

## Options
Many of the options are described above, but all the options in the script are described below.
### Job Configuration
Option | Description
--- | ---
`inputDir` | Directory where Root files are located.
`submitDir` | Directory where numpy files and output data are located.
`plotDir` | Directory where plots are located.
`-i`,`identifier` | String to identify sample/jet collection; all output plots and data are labeled with this string.
`-r` | Option to read in data from Root. Default is from numpy.
### Root Configuration
Option | Description
--- | ---
`jetpt` | Branch name for reconstructed jet pT.
`tjetpt` | Branch name for truth jet pT.
`npv` | Branch name for NPV of event.
`tjeteta` | Branch name for truth jet eta.
`tjetmindr` | Branch name for minimum distance from truth jet to any other truth jet.
`event_weight` | Branch name for event weight.
### Jet Selection Configuration
Option | Description
--- | ---
`-c`,`cut` | Cut out reconustructed jets with pT less than this value (in GeV).
`mineta` | Minimum absolute value of truth jet eta.
`maxeta` | Maximum absolute value of truth jet eta.
`mindr` | Minimum distance from truth jet to any other truth jet.
### Analysis Configuration
Options | Description
--- | ---
`minnpv` | Minimum NPV to analyze.
`maxnpv` | Maximum NPV to analyze.
`npvbin` | Size of NPV bins.
`minpt` | Minimum truth pT to analyze.
`maxpt` | Maximum truth pT to analyze.
`ptbin` | Size of pT bins.
Note: Even if you're only considering jets with low pT, it's recommended to make the `maxpt` value considerably higher than the analysis regime, so that f'(x) can be measured properly. In particular, if you only analyze one pT bin then the analysis will fail, because no fit can be made to the response function.


