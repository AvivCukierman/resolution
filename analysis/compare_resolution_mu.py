import os
import json
from pprint import pprint
from numpy import array
from optparse import OptionParser
import matplotlib.pyplot as plt 
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
os.environ[ 'MPLCONFIGDIR' ] = '/tmp/' #to get matplotlib to work

import pdb

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
#parser.add_option("--plotlabel", help="label going on every plot (only if using ATLAS style)",type=str, default='$\mathregular{\sqrt{s}=13}$ TeV, $\mathregular{<\mu>=20}$')
parser.add_option("--mu",help="Name of thing to plot against. Doesn't have to be mu.",type=str,default="mus")
parser.add_option("--mu_label",help="Label of x-axis value. Doesn't have to be mu.",type=str,default="<$\mu$>")

(options, args) = parser.parse_args()

if not os.path.exists(options.submitDir): raise OSError(options.submitDir+' does not exist. This is where the input pickle files go.')
if not os.path.exists(options.plotDir):
  print '== Making folder '+options.plotDir+' =='
  os.makedirs(options.plotDir)

def readCollections():
  with open(options.submitDir+'/'+options.collections+'.json') as data_file:
    data = json.load(data_file)
    mus = data[options.mu]
    data = data['collections']
  return data,mus

import pickle
def plot_sigma_mu(collections_list,mus):
  identifier = collections_list[0]['identifiers'][0]
  ptedges = pickle.load(open(options.submitDir+'/'+'ptedges_'+identifier+'.p','rb')) #assumes all the collections have the same ptedges

  incl_sigmas_mu = []
  incl_sigma_errs_mu = []
  for c in collections_list:
    incl_sigmas = []
    incl_sigma_errs = []
    for identifier in c['identifiers']:
      incl_sigmas.append(pickle.load(open(options.submitDir+'/'+'incl_sigmas_'+identifier+'.p','rb')))
      incl_sigma_errs.append(pickle.load(open(options.submitDir+'/'+'incl_sigma_errs_'+identifier+'.p','rb')))
    incl_sigmas_mu.append(incl_sigmas)
    incl_sigma_errs_mu.append(incl_sigma_errs)

  for i,ptbin in enumerate(ptedges):
    if i==0: continue

    lowlim = float('inf')
    highlim = float('-inf')
    npv_sigmas = []
    npv_sigma_errs = []

    for j,c in enumerate(collections_list):
      incl_sigmas = [sigmas[i-1] for sigmas in incl_sigmas_mu[j]]
      incl_sigma_errs = [sigma_errs[i-1] for sigma_errs in incl_sigma_errs_mu[j]]

      plt.errorbar(mus,incl_sigmas,yerr=incl_sigma_errs,color=c['color'],linestyle=c['ls'],label=c['label'])
      lowlim = min(lowlim,min(incl_sigmas))
      highlim = max(highlim,max(incl_sigmas))
    #ATLAS style
    axes = plt.axes()
    if atlas_style:
      axes.xaxis.set_minor_locator(AutoMinorLocator())
      axes.yaxis.set_minor_locator(AutoMinorLocator())
      plt.xlabel(options.mu_label, position=(1., 0.), va='bottom', ha='right')
      plt.ylabel('$\sigma[p_T^{reco}]$ [GeV]', position=(0., 1.), va='top', ha='right')
      axes.xaxis.set_label_coords(1., -0.15)
      axes.yaxis.set_label_coords(-0.15, 1.)
      axes.text(0.05,0.9,'ATLAS', transform=axes.transAxes,size='larger',weight='bold',style='oblique')
      axes.text(0.18,0.9,'Simulation', transform=axes.transAxes,size='larger')
      axes.text(0.05,0.65,'Pythia8 dijets'+'\n'+str(ptedges[i-1])+' GeV $< p_T^{true} < $'+str(ptedges[i])+' GeV', transform=axes.transAxes,linespacing=1.5,size='larger')
    else:
      plt.errorbar([0],[0],linestyle=' ',label=str(ptedges[i-1])+' GeV $< p_T^{true} < $'+str(ptedges[i])+' GeV')
      plt.xlabel(options.mu_label)
      plt.ylabel('$\sigma[p_T^{reco}]$ [GeV]')
    plt.ylim(lowlim-0.5,highlim+2)
    plt.xlim(0,max(mus)+min(mus))
    # legend without errors: 
    handles, labels = axes.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles,labels,loc='upper right',frameon=False,numpoints=1,prop={'size':14})
    plt.savefig(options.plotDir+'/jetsigma_mu_pt'+str(ptedges[i-1])+str(ptedges[i])+'_'+options.collections+'.png')
    plt.savefig(options.plotDir+'/jetsigma_mu_pt'+str(ptedges[i-1])+str(ptedges[i])+'_'+options.collections+'.pdf')
    plt.close()

  incl_sigmas_mu = []
  incl_sigma_errs_mu = []
  for c in collections_list:
    incl_sigmas = []
    incl_sigma_errs = []
    for identifier in c['identifiers']:
      incl_sigmas.append(pickle.load(open(options.submitDir+'/'+'incl_sigmaRs_'+identifier+'.p','rb')))
      incl_sigma_errs.append(pickle.load(open(options.submitDir+'/'+'incl_sigmaR_errs_'+identifier+'.p','rb')))
    incl_sigmas_mu.append(incl_sigmas)
    incl_sigma_errs_mu.append(incl_sigma_errs)

  for i,ptbin in enumerate(ptedges):
    if i==0: continue

    lowlim = float('inf')
    highlim = float('-inf')
    npv_sigmas = []
    npv_sigma_errs = []

    for j,c in enumerate(collections_list):
      incl_sigmas = [sigmas[i-1] for sigmas in incl_sigmas_mu[j]]
      incl_sigma_errs = [sigma_errs[i-1] for sigma_errs in incl_sigma_errs_mu[j]]

      plt.errorbar(mus,incl_sigmas,yerr=incl_sigma_errs,color=c['color'],linestyle=c['ls'],label=c['label'])
      lowlim = min(lowlim,min(incl_sigmas))
      highlim = max(highlim,max(incl_sigmas))
    #ATLAS style
    axes = plt.axes()
    if atlas_style:
      axes.xaxis.set_minor_locator(AutoMinorLocator())
      axes.yaxis.set_minor_locator(AutoMinorLocator())
      plt.xlabel(options.mu_label, position=(1., 0.), va='bottom', ha='right')
      plt.ylabel('$\sigma[p_T^{reco}/p_T^{true}]$', position=(0., 1.), va='top', ha='right')
      axes.xaxis.set_label_coords(1., -0.15)
      axes.yaxis.set_label_coords(-0.15, 1.)
      axes.text(0.05,0.9,'ATLAS', transform=axes.transAxes,size='larger',weight='bold',style='oblique')
      axes.text(0.18,0.9,'Simulation', transform=axes.transAxes,size='larger')
      axes.text(0.05,0.65,'Pythia8 dijets'+'\n'+str(ptedges[i-1])+' GeV $< p_T^{true} < $'+str(ptedges[i])+' GeV', transform=axes.transAxes,linespacing=1.5,size='larger')
    else:
      plt.errorbar([0],[0],linestyle=' ',label=str(ptedges[i-1])+' GeV $< p_T^{true} < $'+str(ptedges[i])+' GeV')
      plt.xlabel(options.mu_label)
      plt.ylabel('$\sigma[p_T^{reco}/p_T^{true}]$')
    plt.ylim(lowlim-0.05,highlim+.1)
    plt.xlim(0,max(mus)+min(mus))
    # legend without errors: 
    handles, labels = axes.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles,labels,loc='upper right',frameon=False,numpoints=1,prop={'size':14})
    plt.savefig(options.plotDir+'/jetsigmaR_mu_pt'+str(ptedges[i-1])+str(ptedges[i])+'_'+options.collections+'.png')
    plt.savefig(options.plotDir+'/jetsigmaR_mu_pt'+str(ptedges[i-1])+str(ptedges[i])+'_'+options.collections+'.pdf')
    plt.close()

collections_list,mus = readCollections()
plot_sigma_mu(collections_list,mus)
