import os
from numpy import array
from optparse import OptionParser
os.environ[ 'MPLCONFIGDIR' ] = '/tmp/' #to get matplotlib to work

parser = OptionParser()

# job configuration
parser.add_option("--inputFile", help="Root file containing input histogram",type=str, default="test.root")
parser.add_option("--outFile", help="Output file containing fitted values",type=str, default="test")
parser.add_option("--plotDir", help="Directory containing plots",type=str, default=".")

# Root configuration
## Reconstructed jets and matched truth jets
parser.add_option("--hist", help="name of histogram in Root file",type=str, default="j0pt")
parser.add_option("--eff", help="efficiency of jet collection",type=float, default=-1)
parser.add_option("-m","--central",help="Choice of notion of central tendency/resolution (mean, mode, median, absolute_median, or trimmed)",type='choice',choices=['mean','mode','median','absolute_median','trimmed'],default='mode')

(options, args) = parser.parse_args()

if not os.path.exists(options.inputFile): raise OSError(options.inputFile +' does not exist. This is the input Root file.')

import ROOT as r
def readRoot(root_file):
  try:
    hist = getattr(root_file,options.hist)
  except:
    raise RuntimeError('Can\'t find histogram '+options.hist+' in Root file.') 
  return hist

root_file = r.TFile(options.inputFile)
hist = readRoot(root_file)

data = []
weightdata = []
for i in range(hist.GetXaxis().GetNbins()):
  data.append(hist.GetBinCenter(i))
  weightdata.append(hist.GetBinContent(i)) #weight according to height
data = array(data)
weightdata = array(weightdata)

from fithist import fithist
(mu,mu_err,sigma,sigma_err) = fithist(data,weightdata,options.central,options.eff,plotDir=options.plotDir)
print mu,mu_err,sigma,sigma_err

storedict = {'mu':mu,'mu_err':mu_err,'sigma':sigma,'sigma_err':sigma_err}
import json
with open(options.outFile+'_'+options.central+'.json', 'w') as outfile:
  json.dump(storedict, outfile)
