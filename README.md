# Resolution - A script to calculate jet resolutions

This is a script to calculate the resolution of a reconstructed jet collection, taking into account the effects of the jet response function.
The script does not actually go through the process of numerical inversion, but rather estimates the jet resolution as sigma(t)/g'(t), as outlined in (ATLAS internal note).
The jets are broken down into bins of NPV and pT, so it's recommended to have statistics of at least 50000 jets in order to have reasonable error bars.

## Dependencies
Should work with Python 2.7.2 and above. Uses numpy and PyRoot.

## Input
The input to the script can either be a Root Ntuple or numpy arrays. Either way, the `-i` option sets the `identifier` keyword, which identifies the sample/jet collection being studied.
### Root
If you want to use Root files as input, use the `-r` option.

You should store your Root files in the `inputDir` directory. The only required branches are `jetpt` (the pT of reconstructed jets), `tjetpt` (the pT of the truth jets matched to those reconstructed jets), and `NPV` (the NPV of the event).
Other optional branches are `tjeteta` (the eta of the truth jets), `tjetmindr` (the minimum dR from the truth jet to any other truth jet - for most studies, it's recommend to examine only isolated truth jets), and `event_weight` (the event weight).
The script reads in every Root file in the `intputDir` directory, but you can limit the number of events run over with the `eventNum` keyword.

The script automatically writes out the files to numpy arrays and stores them in the `submitDir` directory. Because of this, you only need to run with the `-r` keyword once, and after that the script will read the .npy files it created, which is much faster than reading Root files.

### Numpy
The default is to use numpy files as the input.

You should store your numpy files in the `submitDir` directory. The only required files are:  
`'truepts_'+identifier+'.npy'` (the pT of reconstructed jets)  
`'recopts_'+identifier+'.npy'` (the pT of the truth jets matched to those reconstructed jets)  
`'npvs_'+identifier+'.npy'` (the NPV of the event that the jet is in - should be the same length as the above two arrays)  

The optional files are:  
`'etas_'+identifier+'.npy'` (the eta of the truth jets)  
`'mindrs_'+identifier+'.npy'` (the minimum dR from the truth jet to any other truth jet - for most studies, it's recommend to examine only isolated truth jets)  
`'weights_'+identifier+'.npy'` (the weight of the event that the jet is in - should be the same length as the above two arrays)  
