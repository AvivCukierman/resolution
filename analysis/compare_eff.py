import os
import pdb
import json
from pprint import pprint
from numpy import array,append
import numpy
from optparse import OptionParser
import matplotlib.pyplot as plt 
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
os.environ[ 'MPLCONFIGDIR' ] = '/tmp/' #to get matplotlib to work

atlas_style = False
try:
  from rootpy.plotting.style import set_style, get_style
  print 'ATLAS style!'
  atlas = get_style('ATLAS')
  atlas.SetPalette(51)
  set_style(atlas)
  set_style('ATLAS',mpl=True)
  atlas_style=True
except ImportError: print 'Not using ATLAS style (Can\'t import rootpy. Try setting up a virtual environment.)'

parser = OptionParser()

# job configuration
parser.add_option("--submitDir", help="Directory containing output files",type=str, default="../output_absolute")
parser.add_option("--plotDir", help="Directory containing plots",type=str, default="../plots")
parser.add_option("--collections", help="file containing jet collection identifiers and labels",type=str, default="collections")

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
  identifier = collections_list[0]['identifier']
  ptedges = pickle.load(open(options.submitDir+'/'+'ptedges_'+identifier+'.p','rb')) #assumes all the collections have the same ptedges
  for i,ptbin in enumerate(ptedges):
    if i==0: continue

    lowlim = float('inf')
    highlim = float('-inf')

    effs = [] 
    eff_errs = []

    for c in collections_list:
      identifier = c['identifier']
      npv_effs = pickle.load(open(options.submitDir+'/'+'efficiencies_'+identifier+'.p','rb'))
      npv_eff_errs = pickle.load(open(options.submitDir+'/'+'efficiency_errs_'+identifier+'.p','rb'))

      npv_keys = npv_effs.keys() 
      npv_keys.sort()
      npvbin = npv_keys[1]-npv_keys[0]

      effs.append([npv_effs[n][i-1] for n in npv_keys])
      eff_errs.append([npv_eff_errs[n][i-1] for n in npv_keys])

      plt.errorbar(array(npv_keys)-0.5*npvbin,[npv_effs[n][i-1] for n in npv_keys],yerr=[npv_eff_errs[n][i-1] for n in npv_keys],color=c['color'],linestyle=c['ls'],label=c['label'])
      lowlim = min(lowlim,min([npv_effs[n][i-1] for n in npv_keys]))

    plt.errorbar([0],[0],linestyle=' ',label=str(ptedges[i-1])+' GeV $< p_T^{true} < $'+str(ptedges[i])+' GeV')
    plt.xlabel('NPV')
    plt.ylabel('Reconstruction Efficiency')
    plt.ylim(lowlim-0.1,1.1)
    plt.xlim(min(npv_keys)-npvbin,max(npv_keys))
    # legend without errors: 
    axes = plt.axes()
    handles, labels = axes.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles,labels,loc='upper right',frameon=False,numpoints=1,prop={'size':14})
    plt.savefig(options.plotDir+'/jetefficiency_NPV_pt'+str(ptedges[i-1])+str(ptedges[i])+'_'+options.collections+'.png')
    plt.close()

plot_sigmas()
