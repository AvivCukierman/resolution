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
def plot_sigma_npv(collections_list):
  identifier = collections_list[0]['identifier']
  ptedges = pickle.load(open(options.submitDir+'/'+'ptedges_'+identifier+'.p','rb')) #assumes all the collections have the same ptedges
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
      npvbin = npv_keys[1]-npv_keys[0]

      plt.errorbar(array(npv_keys)-0.5*npvbin,[npv_sigmas[n][i-1] for n in npv_keys],yerr=[npv_sigma_errs[n][i-1] for n in npv_keys],color=c['color'],linestyle=c['ls'],label=c['label'])
      lowlim = min(lowlim,min(npv_sigmas[n][i-1] for n in npv_keys))
      highlim = max(highlim,max(npv_sigmas[n][i-1] for n in npv_keys))
    #ATLAS style
    axes = plt.axes()
    if atlas_style:
      axes.xaxis.set_minor_locator(AutoMinorLocator())
      axes.yaxis.set_minor_locator(AutoMinorLocator())
      plt.xlabel('NPV', position=(1., 0.), va='bottom', ha='right')
      plt.ylabel('$\sigma[p_T^{reco}]$ [GeV]', position=(0., 1.), va='top', ha='right')
      axes.xaxis.set_label_coords(1., -0.15)
      axes.yaxis.set_label_coords(-0.15, 1.)
      axes.text(0.05,0.9,'ATLAS', transform=axes.transAxes,size='larger',weight='bold',style='oblique')
      axes.text(0.18,0.9,'Simulation', transform=axes.transAxes,size='larger')
      axes.text(0.05,0.65,options.plotlabel+'\nPythia8 dijets'+'\n'+str(ptedges[i-1])+' GeV $< p_T^{true} < $'+str(ptedges[i])+' GeV', transform=axes.transAxes,linespacing=1.5,size='larger')
    else:
      plt.errorbar([0],[0],linestyle=' ',label=str(ptedges[i-1])+' GeV $< p_T^{true} < $'+str(ptedges[i])+' GeV')
      plt.xlabel('NPV')
      plt.ylabel('$\sigma[p_T^{reco}]$ [GeV]')
    plt.ylim(lowlim-0.5,highlim+2)
    plt.xlim(min(npv_keys)-npvbin,max(npv_keys))
    # legend without errors: 
    handles, labels = axes.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles,labels,loc='upper right',frameon=False,numpoints=1,prop={'size':14})
    plt.savefig(options.plotDir+'/jetsigma_NPV_pt'+str(ptedges[i-1])+str(ptedges[i])+'_'+options.collections+'.png')
    plt.savefig(options.plotDir+'/jetsigma_NPV_pt'+str(ptedges[i-1])+str(ptedges[i])+'_'+options.collections+'.pdf')
    plt.close()

    highlim = float('-inf')
    lowlim = float('inf') 
    for c in collections_list:
      identifier = c['identifier']
      npv_sigmaRs = pickle.load(open(options.submitDir+'/'+'sigmaRs_'+identifier+'.p','rb'))
      npv_sigmaR_errs = pickle.load(open(options.submitDir+'/'+'sigmaR_errs_'+identifier+'.p','rb'))

      npv_keys = npv_sigmaRs.keys() 
      npv_keys.sort()
      npvbin = npv_keys[1]-npv_keys[0]

      plt.errorbar(array(npv_keys)-0.5*npvbin,[npv_sigmaRs[n][i-1] for n in npv_keys],yerr=[npv_sigmaR_errs[n][i-1] for n in npv_keys],color=c['color'],linestyle=c['ls'],label=c['label'])
      highlim = max(max(npv_sigmaRs[n][i-1] for n in npv_keys),highlim)
      lowlim = min(min(npv_sigmaRs[n][i-1] for n in npv_keys),lowlim)
    #ATLAS style
    axes = plt.axes()
    if atlas_style:
      axes.xaxis.set_minor_locator(AutoMinorLocator())
      axes.yaxis.set_minor_locator(AutoMinorLocator())
      plt.xlabel('NPV', position=(1., 0.), va='bottom', ha='right')
      plt.ylabel('$\sigma[p_T^{reco}/p_T^{true}]$', position=(0., 1.), va='top', ha='right')
      axes.xaxis.set_label_coords(1., -0.15)
      axes.yaxis.set_label_coords(-0.15, 1.)
      axes.text(0.05,0.9,'ATLAS', transform=axes.transAxes,size='larger',weight='bold',style='oblique')
      axes.text(0.18,0.9,'Simulation', transform=axes.transAxes,size='larger')
      axes.text(0.05,0.65,options.plotlabel+'\nPythia8 dijets'+'\n'+str(ptedges[i-1])+' GeV $< p_T^{true} < $'+str(ptedges[i])+' GeV', transform=axes.transAxes,linespacing=1.5,size='larger')
    else:
      plt.errorbar([0],[0],linestyle=' ',label=str(ptedges[i-1])+' GeV $< p_T^{true} < $'+str(ptedges[i])+' GeV')
      plt.xlabel('NPV')
      plt.ylabel('$\sigma[p_T^{reco}/p_T^{true}]$')
    plt.ylim(lowlim-0.05,highlim+0.1)
    plt.xlim(min(npv_keys)-npvbin,max(npv_keys))
    # legend without errors: 
    handles, labels = axes.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles,labels,loc='upper right',frameon=False,numpoints=1,prop={'size':14})
    plt.savefig(options.plotDir+'/jetsigmaR_NPV_pt'+str(ptedges[i-1])+str(ptedges[i])+'_'+options.collections+'.png')
    plt.savefig(options.plotDir+'/jetsigmaR_NPV_pt'+str(ptedges[i-1])+str(ptedges[i])+'_'+options.collections+'.pdf')
    plt.close()

def plot_sigma_pt(collections_list):
  identifier = collections_list[0]['identifier']
  ptedges = pickle.load(open(options.submitDir+'/'+'ptedges_'+identifier+'.p','rb')) #assumes all the collections have the same ptedges
  ptedges.sort()
  ptbin = ptedges[1]-ptedges[0]
  npv_sigmas = pickle.load(open(options.submitDir+'/'+'sigmas_'+identifier+'.p','rb')) #assumes all algorithms have the same NPV range
  npv_keys = npv_sigmas.keys() 
  npv_keys.sort()
  npvbin = npv_keys[1]-npv_keys[0]
  highlim = {npv:float('-inf') for npv in npv_keys}
  lowlim = {npv:float('inf') for npv in npv_keys}
  for c in collections_list:
    identifier = c['identifier']
    avgpt = pickle.load(open(options.submitDir+'/'+'avgpttrue_'+identifier+'.p','rb')) #assumes all algorithms have the same avg pT true
    npv_sigmas = pickle.load(open(options.submitDir+'/'+'sigmas_'+identifier+'.p','rb'))
    npv_sigma_errs = pickle.load(open(options.submitDir+'/'+'sigma_errs_'+identifier+'.p','rb'))

    for i,npv in enumerate(npv_keys):
      plt.figure(i)
      plt.errorbar(avgpt,npv_sigmas[npv],yerr=npv_sigma_errs[npv],color=c['color'],linestyle=c['ls'],label=c['label'])
      highlim[npv] = max(highlim[npv],max(npv_sigmas[npv]))
      lowlim[npv] = min(lowlim[npv],min(npv_sigmas[npv]))
  #ATLAS style
  for i,npv in enumerate(npv_keys):
    plt.figure(i)
    axes = plt.axes()
    if atlas_style:
      axes.xaxis.set_minor_locator(AutoMinorLocator())
      axes.yaxis.set_minor_locator(AutoMinorLocator())
      plt.xlabel('$p_T^{true}$ [GeV]', position=(1., 0.), va='bottom', ha='right')
      plt.ylabel('$\sigma[p_T^{reco}]$ [GeV]', position=(0., 1.), va='top', ha='right')
      axes.xaxis.set_label_coords(1., -0.15)
      axes.yaxis.set_label_coords(-0.15, 1.)
      axes.text(0.05,0.9,'ATLAS', transform=axes.transAxes,size='larger',weight='bold',style='oblique')
      axes.text(0.18,0.9,'Simulation', transform=axes.transAxes,size='larger')
      axes.text(0.05,0.65,options.plotlabel+'\nPythia8 dijets'+'\n'+str(ptedges[i-1])+' GeV $< p_T^{true} < $'+str(ptedges[i])+' GeV', transform=axes.transAxes,linespacing=1.5,size='larger')
    else:
      plt.errorbar([0],[0],linestyle=' ',label=str(npv-npvbin)+' < NPV < '+str(npv))
      plt.xlabel('$p_T^{true}$ [GeV]')
      plt.ylabel('$\sigma[p_T^{reco}]$ [GeV]')
    plt.ylim(lowlim[npv]-0.5,highlim[npv]+2)
    plt.xlim(min(ptedges),max(ptedges))
    # legend without errors: 
    handles, labels = axes.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles,labels,loc='upper right',frameon=False,numpoints=1,prop={'size':14})
    plt.savefig(options.plotDir+'/jetsigma_pt_NPV'+str(npv-npvbin)+str(npv)+'_'+options.collections+'.png')
    plt.savefig(options.plotDir+'/jetsigma_pt_NPV'+str(npv-npvbin)+str(npv)+'_'+options.collections+'.pdf')
    plt.close()

  highlim = {npv:float('-inf') for npv in npv_keys}
  lowlim = {npv:float('inf') for npv in npv_keys}
  for c in collections_list:
    identifier = c['identifier']
    avgpt = pickle.load(open(options.submitDir+'/'+'avgpttrue_'+identifier+'.p','rb')) #assumes all algorithms have the same avg pT true
    npv_sigmas = pickle.load(open(options.submitDir+'/'+'sigmaRs_'+identifier+'.p','rb'))
    npv_sigma_errs = pickle.load(open(options.submitDir+'/'+'sigmaR_errs_'+identifier+'.p','rb'))

    for i,npv in enumerate(npv_keys):
      plt.figure(i)
      plt.errorbar(avgpt,npv_sigmas[npv],yerr=npv_sigma_errs[npv],color=c['color'],linestyle=c['ls'],label=c['label'])
      highlim[npv] = max(highlim[npv],max(npv_sigmas[npv]))
      lowlim[npv] = min(lowlim[npv],min(npv_sigmas[npv]))
  #ATLAS style
  for i,npv in enumerate(npv_keys):
    plt.figure(i)
    axes = plt.axes()
    if atlas_style:
      axes.xaxis.set_minor_locator(AutoMinorLocator())
      axes.yaxis.set_minor_locator(AutoMinorLocator())
      plt.xlabel('$p_T^{true}$ [GeV]', position=(1., 0.), va='bottom', ha='right')
      plt.ylabel('$\sigma[p_T^{reco}/p_T^{true}]$', position=(0., 1.), va='top', ha='right')
      axes.xaxis.set_label_coords(1., -0.15)
      axes.yaxis.set_label_coords(-0.15, 1.)
      axes.text(0.05,0.9,'ATLAS', transform=axes.transAxes,size='larger',weight='bold',style='oblique')
      axes.text(0.18,0.9,'Simulation', transform=axes.transAxes,size='larger')
      axes.text(0.05,0.65,options.plotlabel+'\nPythia8 dijets'+'\n'+str(ptedges[i-1])+' GeV $< p_T^{true} < $'+str(ptedges[i])+' GeV', transform=axes.transAxes,linespacing=1.5,size='larger')
    else:
      plt.errorbar([0],[0],linestyle=' ',label=str(npv-npvbin)+' < NPV < '+str(npv))
      plt.xlabel('$p_T^{true}$ [GeV]')
      plt.ylabel('$\sigma[p_T^{reco}/p_T^{true}]$')
    plt.ylim(lowlim[npv]-0.05,highlim[npv]+.1)
    plt.xlim(min(ptedges),max(ptedges))
    # legend without errors: 
    handles, labels = axes.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles,labels,loc='upper right',frameon=False,numpoints=1,prop={'size':14})
    plt.savefig(options.plotDir+'/jetsigmaR_pt_NPV'+str(npv-npvbin)+str(npv)+'_'+options.collections+'.png')
    plt.savefig(options.plotDir+'/jetsigmaR_pt_NPV'+str(npv-npvbin)+str(npv)+'_'+options.collections+'.pdf')
    plt.close()

collections_list = readCollections()
plot_sigma_npv(collections_list)
plot_sigma_pt(collections_list)
