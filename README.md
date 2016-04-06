# Resolution - A script to calculate jet resolutions

This is a script to calculate the resolution of a reconstructed jet collection, taking into account the effects of the jet response function.
The script gives you an option to actually go through the process of numerical inversion or to estimate the jet resolution as sigma(x)/f'(x), as outlined in [https://cds.cern.ch/record/2045523/](https://cds.cern.ch/record/2045523/).
The jets are broken down into bins of NPV and pT, so it's recommended to have statistics of at least 50000 jets in order to have reasonable error bars.

The main python script is `resolution.py`, and a sample run script is given in `run_jz2_j0.sh`. The script does not compare different jet collections; it only calculates the resolution of a particular jet collection in NPV and pT bins. This is because the process of comparing different jet collections will probably vary a lot depending on the analysis being done. A sample python and accompanying shell script to compare different jet collections can be found in `comare_resolution.py` and `compare_sigmas_EM.sh`, with `output/EM_collections.json` giving some plotting parameters for the comparison script.

## Dependencies
Should work with Python 2.7.2 and above. Uses numpy, PyRoot, and SciPy. If RootPy is installed, plots are made with ATLAS style, but it's not necessary for running the script.

## Input
The input to the script can either be a Root Ntuple or numpy arrays.

### Root
If you want to use Root files as input, use the `-r` option.

Root files are searched for in the `inputDir` directory. The script picks up all Root files stored in this directory.

The only required branches are `jetpt` (the pT of reconstructed jets), `tjetpt` (the pT of the truth jets matched to those reconstructed jets), and `NPV` (the NPV of the event).
Other optional branches are `tjeteta` (the eta of the matched truth jets), `tjetmindr` (the minimum dR from the matched truth jet to any other truth jet - for most studies, it's recommend to examine only isolated truth jets), and `event_weight` (the event weight).
If the `-e` flag is set to calculate efficiencies, then the required branches are `all_tjetpt` (the pT of all truth jets, not just the matched ones), and the optional branches are `all_tjeteta` (the eta of all truth jets) and `all_tjetmindr` (the minimum dR for each truth jet in the set of all truth jets).
The script reads in every Root file in the `intputDir` directory, but you can limit the number of events run over with the `numEvents` keyword.

The script automatically writes out the files to numpy arrays and stores them in the `submitDir` directory. Because of this, you only need to run with the `-r` keyword once, and after that the script will read the .npy files it created, which is much faster than reading Root files.

### Numpy
The default is to use numpy files as the input.

Numpy files are searched for in the `submitDir` directory. The only required files are:  
`'truepts_'+identifier+'.npy'` (the pT of reconstructed jets)  
`'recopts_'+identifier+'.npy'` (the pT of the truth jets matched to those reconstructed jets)  
`'npvs_'+identifier+'.npy'` (the NPV of the event that the jet is in)  

The optional files are:  
`'etas_'+identifier+'.npy'` (the eta of the matched truth jets)  
`'mindrs_'+identifier+'.npy'` (the minimum dR from the matched truth jets to any other truth jet - for most studies, it's recommend to examine only isolated truth jets)  
`'weights_'+identifier+'.npy'` (the weight of the event that the jet is in)  

Note that all of the above arrays should be the same length.

If the `-e` flag is set to calculate efficiencies, the required files are:

`'all_truepts_'+identifier+'.npy'` (the pT of all truth jets)  
`'all_npvs_'+identifier+'.npy'` (the NPV of the event that the truth jet is in)
And the optional files are:

`'all_etas_'+identifier+'.npy'` (the eta of the truth jets)
`'all_mindrs_'+identifier+'.npy'` (the minimum dR from the truth jets to any other truth jet)  
`'all_weights_'+identifier+'.npy'` (the weight of the event that the truth jet is in) 

Note that all of the above arrays `all_*.npy` should be the same length, and larger in length than the other numpy arrays above without the `all_` prefix.

## Analysis Configuration
The calibration is done via the process of numerical inversion, through applying f^-1(x) to reconstructed jets rather than the numerical inversion process described in [https://cds.cern.ch/record/1201006](https://cds.cern.ch/record/1201006). These methods are shown to be mathematically equivalent in the note mentioned in the introduction. However both methods suffer from the problem of extrapolating f(x) to unseen values. This script assumes the response R(x) is even and positive, so that f(x)=R(x)*x is odd and passes through 0.

One important option int the analysis configuration is the `-e` flag. This flag tells the script to calculate jet collection efficiencies, and is required for some definitions of central tendency/resolution. Some extra branches/files are required in order to utilize this option.

A very important option in the analysis configuration is the `-m` option. This indicates which notion of central tendency and resolution to use in the calibration. For many experimental jet collections, the response distributions are non-Gaussian, and so the mean, median, and mode are generally all different, requiring some thought on the part of the analyzer as to which notion of central tendency to use. See, e.g., a discussion in the [JES/JER group](https://indico.cern.ch/event/478081/) about different approaches to this answering this question. The options are:

Option | Central Tendency | Resolution
--- | --- | ---
mean | naive Gaussian mean | naive Gaussian width
median | median of response distribution | right-hand IQR of response distribution
trimmed | trimmed Gaussian mean (~mode) | trimmed Gaussian width
mode | trimmed Gaussian mean (~mode) | absolute right-hand IQR
absolute_median | absolute median of response distribution | absolute right-hand IQR
For an explanation of the "absolute" options, check out [this talk](https://indico.cern.ch/event/514730/session/0/attachments/1249497/1841653/PM_3.28.16.pdf). The default option is `mode`, based on discussions in JES/JER and pileup mitigation meetings.

Note that for the `mode` and `absolute_median` options, the flag `-e` is required.

If the `-doEst` flag is set in the analysis, then the full calibration is not run, and estimates for the calibrated resolution of the jets are used.

## Output

### Plots
Plots are saved in the `plotDir` directory. The `-i` option sets the `identifier` keyword, which is a string that will end up in all plots made with this tool in order to identify the sample/jet collection being studied.

#### Output Plots
In NPV bins (indicated by `'_NPV##_'` string in the filename):  
`'resbin%d'%ptbin`: Distribution of reconstructed pT response binned in truth pT.  
`'fbin%d'%ptbin`: Distribution of reconstructed pT binned in truth pT.  
`'jetresponse_pttrue'`: Jet response curve.  
`'jetf_pttrue'`: Jet reconstructed pT curve.  
`'closurebin%d'%ptbin`: Distribution of calibrated reconstructed pT response binned in truth pT. Only made if `-n` option is set.  
`'f1bin%d'%ptbin`: Distribution of calibrated reconstructed pT binned in truth pT. Only made if `-n` option is set.  
`'jetf1_pttrue'`: Calibrated reconstructed pT. Only made if `-n` option is set.
`'jetclosure_pttrue'`: Calibrated reconstructed pT, divided by truth pT. Only made if `-n` option is set.  
`'jetclosure_pttrue_zoom'`: Calibrated reconstructed pT, divided by truth pT. Zoomed to small range around 1. Only made if `-n` option is set.  
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
`''avgpttrue_'+options.identifier+'.p'`: The average truth pT in the truth pT bins. Basically, the x-axis of the data points in many of the above plots.  
`''ptedges_'+options.identifier+'.p'`: A list of the truth pT bins for the analysis.  
`'sigmas_'+options.identifier+'.p''`: The estimated calibrated jet pT resolution, stored as a dictionary of arrays. The keys of the dictionary are the max NPV of the NPV bin, and the array is listed in order of increasing pT bin.  
`'sigma_errs_'+options.identifier+'.p''`: The error on the estimated calibrated jet pT resolution, stored as a dictionary of arrays. The keys of the dictionary are the max NPV of the NPV bin, and the array is listed in order of increasing pT bin.  
`'sigmaRs_'+options.identifier+'.p''`: The estimated calibrated fractional jet pT resolution, stored as a dictionary of arrays. The keys of the dictionary are the max NPV of the NPV bin, and the array is listed in order of increasing pT bin.  
`'sigmaR_errs_'+options.identifier+'.p''`: The erros on the estimated calibrated fractional jet pT resolution, stored as a dictionary of arrays. The keys of the dictionary are the max NPV of the NPV bin, and the array is listed in order of increasing pT bin. 
If the `-e` flag is set:`'efficiencies_'+options.identifier+'.p''`: The efficiency of the jet collection in the given pT and NPV bin, stored as a dictionary of arrays. The keys of the dictionary are the max NPV of the NPV bin, and the array is listed in order of increasing pT bin. 

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
`tjetpt` | Branch name for matched truth jet pT.
`npv` | Branch name for NPV of event.
`tjeteta` | Branch name for matched truth jet eta.
`tjetmindr` | Branch name for minimum distance from matched truth jet to any other truth jet.
`event_weight` | Branch name for event weight.
`all_tjetpt` | Branch name for all truth jet pTs.
`all_tjeteta` | Branch name for all truth jet etas.
`all_tjetmindr` | Branch name for minimum distance from each truth jet in the set of all truth jets to any other truth jet.
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
`-e`,`absolute` | Calculate efficiencies of jet collection.
`-m`,`central` | What notion of central tendency and resolution to use.
`minnpv` | Minimum NPV to analyze.
`maxnpv` | Maximum NPV to analyze.
`npvbin` | Size of NPV bins.
`minpt` | Minimum truth pT to analyze.
`maxpt` | Maximum truth pT to analyze.
`ptbin` | Size of pT bins.
`doEst` | Use estimate for resolution rather than doing full numerical inversion.

N. B. Even if you're only considering jets with low pT, it's recommended to make the `maxpt` value considerably higher than the analysis regime, so that f'(x) can be measured properly. In particular, if you only analyze one pT bin then the analysis will fail, because no fit can be made to the response function.


