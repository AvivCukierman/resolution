from numpy import array,all
import os
import numpy
from numpy import save
from helper_functions import distribution_values
from optparse import OptionParser
from scipy.stats import norm
os.environ[ 'MPLCONFIGDIR' ] = '/tmp/' #to get matplotlib to work

try:
  from rootpy.plotting.style import set_style, get_style
  print '== Using ATLAS style =='
  atlas = get_style('ATLAS')
  atlas.SetPalette(51)
  set_style(atlas)
  set_style('ATLAS',mpl=True)
except ImportError: print '== Not using ATLAS style (Can\'t import rootpy.) =='

parser = OptionParser()

# job configuration
parser.add_option("--inputFile", help="Root file containing input histogram",type=str, default="../data")
parser.add_option("--submitDir", help="Directory containing output files",type=str, default="../output")
parser.add_option("--plotDir", help="Directory containing plots",type=str, default="../plots")
parser.add_option("-i","--identifier", help="sample identifier",type=str, default="myjets")

# Root configuration
## Reconstructed jets and matched truth jets
parser.add_option("--hist", help="name of histogram in Root file",type=str, default="j0pt")
parser.add_option("--eff", help="efficiency of jet collection",type=float, default=-1)
parser.add_option("-m","--central",help="Choice of notion of central tendency/resolution (mean, mode, median, absolute_median, or trimmed)",type='choice',choices=['mean','mode','median','absolute_median','trimmed'],default='mode')

(options, args) = parser.parse_args()

if not os.path.exists(options.inputFile): raise OSError(options.inputFile +' does not exist. This is the input Root file.')
if not os.path.exists(options.submitDir):
  print '== Making folder '+options.submitDir+' =='
  os.makedirs(options.submitDir)
if not os.path.exists(options.plotDir):
  print '== Making folder '+options.plotDir+' =='
  os.makedirs(options.plotDir)

identifier = options.identifier

import pdb

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
#plt.style.use('atlas')
import matplotlib.mlab as mlab

import ROOT as r
def readRoot(root_file):

  try:
    hist = getattr(root_file,options.hist)
  except:
    raise RuntimeError('Can\'t find histogram '+options.hist+' in Root file.') 

  return hist

def fithist(hist,central,eff,plotDir):
  data = []
  weightdata = []
  for i in range(hist.GetXaxis().GetNbins()):
    data.append(hist.GetBinCenter(i))
    weightdata.append(hist.GetBinContent(i)) #weight according to height
  data = array(data)
  weightdata = array(weightdata)

  n,bins,patches = plt.hist(data,normed=True,bins=100,weights=weightdata,facecolor='b',histtype='stepfilled')
  weightdata = weightdata/sum(weightdata)
  if central == 'absolute_median' or central == 'mode':
    if eff<0 or eff>1: raise RuntimeError('In order to use absolute IQR, you have to provide the reconstruction efficiency. Use --eff option.')
    (mu,mu_err,sigma,sigma_err,upper_quantile,lower_quantile,err) = distribution_values(data,weightdata,central,eff=eff)
    plt.plot((mu,mu),(0,plt.ylim()[1]),'r--',linewidth=2)
    height = 0.607*max(n) #height at x=1*sigma in normal distribution
    if lower_quantile>float('-inf'):
      plt.plot((lower_quantile,upper_quantile),(height,height),'r--',linewidth=2)
      plt.plot((lower_quantile,lower_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
    else:
      plt.plot((mu,upper_quantile),(height,height),'r--',linewidth=2)
    plt.plot((upper_quantile,upper_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
  if central == 'median':
    (mu,mu_err,sigma,sigma_err,upper_quantile,lower_quantile) = distribution_values(data,weightdata,central)
    plt.plot((mu,mu),(0,plt.ylim()[1]),'r--',linewidth=2)
    height = 0.607*max(n) #height at x=1*sigma in normal distribution
    plt.plot((lower_quantile,upper_quantile),(height,height),'r--',linewidth=2)
    plt.plot((lower_quantile,lower_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
    plt.plot((upper_quantile,upper_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
  if central == 'mean':
    (mu,mu_err,sigma,sigma_err) = distribution_values(data,weightdata,central)
    gfunc = norm
    y = gfunc.pdf( bins, mu, sigma)
    plt.plot((mu,mu),(0,gfunc.pdf(mu,mu,sigma)),'r--',linewidth=2)
    l = plt.plot(bins, y, 'r--', linewidth=2)
  if central == 'trimmed':
    (mu,mu_err,sigma,sigma_err,lower,upper) = distribution_values(data,weightdata,central)
    #print mu,sigma,ptbin
    gfunc = norm
    y = gfunc.pdf(bins, mu, sigma)
    plt.plot((mu,mu),(0,gfunc.pdf(mu,mu,sigma)),'r--',linewidth=2)
    newbins = bins[all([bins>lower,bins<upper],axis=0)]
    newy = y[all([bins>lower,bins<upper],axis=0)]
    l = plt.plot(newbins, newy, 'r--', linewidth=2)
  plt.xlabel('$p_T^{reco}/p_T^{true}$')
  plt.ylabel('a.u.')
  plt.savefig(plotDir+'/'+'histfit_'+central+'.png')
  plt.close()
  #avgres.append(mu)
  #avgres_errs.append(mu_err)
  #sigmaRs.append(sigma)
  #sigmaR_errs.append(sigma_err)

  return mu,mu_err,sigma,sigma_err 

root_file = r.TFile(options.inputFile)
hist = readRoot(root_file)
(mu,mu_err,sigma,sigma_err) = fithist(hist,options.central,options.eff,options.plotDir)
print mu,mu_err,sigma,sigma_err

