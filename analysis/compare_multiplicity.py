import os
import json
from pprint import pprint
from numpy import array
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
parser.add_option("-c","--cut", help="Low pT cut on reco (calibrated) jets",type=str, default="20")
parser.add_option("--plotlabel", help="label going on every plot (only if using ATLAS style)",type=str, default='$\mathregular{\sqrt{s}=13}$ TeV, $\mathregular{<\mu>=20}$')

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
def plot_mults():
  collections_list = readCollections()
  identifier = collections_list[0]['identifier']
  highlim = float('-inf')
  for c in collections_list:
    identifier = c['identifier']
    avg_mults = pickle.load(open(options.submitDir+'/'+'avg_mults_'+identifier+'.p','rb'))
    err_mults = pickle.load(open(options.submitDir+'/'+'err_mults_'+identifier+'.p','rb'))

    npv_keys = avg_mults.keys() 
    npv_keys.sort()
    npvbin = npv_keys[1]-npv_keys[0]

    plt.errorbar(array(npv_keys)-0.5*npvbin,[avg_mults[n] for n in npv_keys],yerr=[err_mults[n] for n in npv_keys],color=c['color'],linestyle=c['ls'],label=c['label'])
    highlim = max(highlim,max(avg_mults[n] for n in npv_keys))
  #ATLAS style
  axes = plt.axes()
  if atlas_style:
    axes.xaxis.set_minor_locator(AutoMinorLocator())
    axes.yaxis.set_minor_locator(AutoMinorLocator())
    plt.xlabel('NPV', position=(1., 0.), va='bottom', ha='right')
    plt.ylabel('Average Jet Multiplicity ($pT>$ '+options.cut+' GeV)', position=(0., 1.), va='top', ha='right')
    axes.xaxis.set_label_coords(1., -0.15)
    axes.yaxis.set_label_coords(-0.15, 1.)
    axes.text(0.05,0.9,'ATLAS', transform=axes.transAxes,size='larger',weight='bold',style='oblique')
    axes.text(0.18,0.9,'Simulation', transform=axes.transAxes,size='larger')
    axes.text(0.05,0.65,options.plotlabel+'\nPythia8 dijets', transform=axes.transAxes,linespacing=1.5,size='larger')
  else:
    plt.xlabel('NPV')
    plt.ylabel('Average Jet Multiplicity ($pT>$ '+options.cut+' GeV)')
  plt.ylim(0,highlim+0.5)
  plt.xlim(min(npv_keys)-npvbin,max(npv_keys))
  # legend without errors: 
  handles, labels = axes.get_legend_handles_labels()
  handles = [h[0] for h in handles]
  plt.legend(handles,labels,loc='upper right',frameon=False,numpoints=1,prop={'size':14})
  plt.savefig(options.plotDir+'/jetmultiplicity'+'_'+options.collections+'.png')
  plt.close()


plot_mults()
