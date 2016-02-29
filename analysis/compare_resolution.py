import os
import json
from pprint import pprint
from numpy import array
from optparse import OptionParser
import matplotlib.pyplot as plt 
os.environ[ 'MPLCONFIGDIR' ] = '/tmp/' #to get matplotlib to work

try:
  from rootpy.plotting.style import set_style, get_style
  print 'ATLAS style!'
  atlas = get_style('ATLAS')
  atlas.SetPalette(51)
  set_style(atlas)
except ImportError: print 'Not using ATLAS style (Can\'t import rootpy. Try setting up a virtual environment.)'

parser = OptionParser()

# job configuration
parser.add_option("--submitDir", help="Directory containing output files",type=str, default="../output")
parser.add_option("--plotDir", help="Directory containing plots",type=str, default="../plots")
parser.add_option("--collections", help="file containing jet collection identifiers and labels",type=str, default="collections")

# analysis configuration
parser.add_option("--minnpv", help="min NPV", type=int, default=5)
parser.add_option("--maxnpv", help="max NPV", type=int, default=30)
parser.add_option("--npvbin", help="size of NPV bins", type=int, default=5)
parser.add_option("--minpt", help="min truth pt", type=int, default=20)
parser.add_option("--maxpt", help="max truth pt", type=int, default=80)
parser.add_option("--ptbin", help="size of pT bins", type=int, default=2)

(options, args) = parser.parse_args()

if not os.path.exists(options.submitDir): raise OSError(options.submitDir+' does not exist. This is where the input pickle files go.')
if not os.path.exists(options.plotDir):
  print '== Making folder '+options.plotDir+' =='
  os.makedirs(options.plotDir)

def readCollections():
  with open(options.submitDir+'/'+options.collections+'.json') as data_file:
    data = json.load(data_file)
    data = data['collections']
  return data

import pickle
def plot_sigmas():
  collections_list = readCollections()
  maxpt = options.maxpt
  if (options.maxpt-options.minpt)%options.ptbin==0: maxpt+=1
  ptedges = range(options.minpt,maxpt,options.ptbin)
  for i,ptbin in enumerate(ptedges):
    if i==0: continue

    lowlim = float('inf')
    highlim = float('-inf')
    for c in collections_list:
      identifier = c['identifier']
      npv_sigmas = pickle.load(open(options.submitDir+'/'+'sigmas_'+identifier+'.p','rb'))
      npv_sigma_errs = pickle.load(open(options.submitDir+'/'+'sigma_errs_'+identifier+'.p','rb'))

      npv_keys = npv_sigmas.keys() 
      npv_keys.sort()

      plt.errorbar(array(npv_keys)-0.5*options.npvbin,[npv_sigmas[n][i-1] for n in npv_keys],yerr=[npv_sigma_errs[n][i-1] for n in npv_keys],color=c['color'],linestyle=c['ls'],label=c['label'])
      lowlim = min(lowlim,min(npv_sigmas[n][i-1] for n in npv_keys))
      highlim = max(highlim,max(npv_sigmas[n][i-1] for n in npv_keys))
    plt.errorbar([0],[0],linestyle=' ',label=str(ptedges[i-1])+' GeV $< p_T^{true} < $'+str(ptedges[i])+' GeV')
    plt.xlabel('NPV')
    plt.ylabel('$\sigma[p_T^{reco}]$ [GeV]')
    plt.ylim(lowlim-1,highlim+1)
    plt.xlim(options.minnpv,options.maxnpv)
    plt.legend(loc='upper left',frameon=False,numpoints=1)
    plt.savefig(options.plotDir+'/jetsigma_NPV_pt'+str(ptedges[i-1])+str(ptedges[i])+'_'+options.collections+'.png')
    plt.close()

    highlim = float('-inf')
    for c in collections_list:
      identifier = c['identifier']
      npv_sigmaRs = pickle.load(open(options.submitDir+'/'+'sigmaRs_'+identifier+'.p','rb'))
      npv_sigmaR_errs = pickle.load(open(options.submitDir+'/'+'sigmaR_errs_'+identifier+'.p','rb'))

      npv_keys = npv_sigmas.keys() 
      npv_keys.sort()

      plt.errorbar(array(npv_keys)-0.5*options.npvbin,[npv_sigmaRs[n][i-1] for n in npv_keys],yerr=[npv_sigmaR_errs[n][i-1] for n in npv_keys],color=c['color'],linestyle=c['ls'],label=c['label'])
      highlim = max(max(npv_sigmaRs[n][i-1] for n in npv_keys),highlim)
    plt.errorbar([0],[0],linestyle=' ',label=str(ptedges[i-1])+' GeV $< p_T^{true} < $'+str(ptedges[i])+' GeV')
    plt.xlabel('NPV')
    plt.ylabel('$\sigma[p_T^{reco}/p_T^{true}]$')
    lowlim = 0 
    plt.ylim(lowlim,highlim+0.1)
    plt.xlim(options.minnpv,options.maxnpv)
    plt.legend(loc='upper left',frameon=False,numpoints=1)
    plt.savefig(options.plotDir+'/jetsigmaR_NPV_pt'+str(ptedges[i-1])+str(ptedges[i])+'_'+options.collections+'.png')
    plt.close()

plot_sigmas()
