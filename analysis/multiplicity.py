from numpy import load,log,linspace,digitize,array,mean,std,exp,all,average,sqrt,asarray,sign
import os
import numpy
from numpy import save
from scipy.optimize import curve_fit,fsolve
from scipy.stats import norm
from operator import sub
from optparse import OptionParser
import pickle
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
parser.add_option("--inputDir", help="Directory containing input files",type=str, default="../data")
parser.add_option("--submitDir", help="Directory containing output files",type=str, default="../output")
parser.add_option("--plotDir", help="Directory containing plots",type=str, default="../plots")
parser.add_option("--numEvents", help="How many events to include (set to -1 for all events)",type=int, default=100000)
parser.add_option("-i","--identifier", help="sample identifier",type=str, default="myjets")
parser.add_option("-r","--root", help="Root input",action="store_true", default=False)

# Root configuration
parser.add_option("--jetpt", help="reco jet pT branch name",type=str, default="j0pt")
parser.add_option("--jeteta", help="reco jet eta branch name",type=str, default="j0eta")
parser.add_option("--npv", help="NPV branch name",type=str, default="NPV")
parser.add_option("--event_weight", help="event weight branch name",type=str, default="event_weight")

# jet configuration
parser.add_option("-c","--cut", default=20, type=float, help="low pT cut on (calibrated) reco jets")
parser.add_option("--mineta", help="min abs(eta) on reco jets", type=float, default=0)
parser.add_option("--maxeta", help="max abs(eta) on reco jets", type=float, default=float('inf'))

(options, args) = parser.parse_args()

if options.root and not os.path.exists(options.inputDir): raise OSError(options.inputDir +' does not exist. This is where the input Root files go.')
if not options.root and not os.path.exists(options.submitDir): raise OSError(options.submitDir+' does not exist. This is where the input numpy files go.')
if options.root and not os.path.exists(options.submitDir):
  print '== Making folder '+options.submitDir+' =='
  os.makedirs(options.submitDir)
if not os.path.exists(options.plotDir):
  print '== Making folder '+options.plotDir+' =='
  os.makedirs(options.plotDir)

import pdb

asym = 10 # shift distribution to the right to get better fit and have R(0) be finite
def R(x,a,b,c):
    ax = abs(array(x))
    result = a + b/log(ax+asym) + c/log(ax+asym)**2
    return result 

def g(x,a,b,c):
    ax = array(x)
    return R(x,a,b,c)*ax 

memoized = {} #memoize results to reduce computation time
def g1(x,a,b,c):
    ax = array(x)
    result = []
    for x in ax:
      approx = (round(x,1),round(a,1),round(b,1),round(c,1))
      if approx not in memoized:
        func = lambda y: approx[0]-g(y,a,b,c)
        memoized[approx] = fsolve(func,approx)[0]
        #print approx,memoized[approx]
      result.append(memoized[approx])
    return array(result)

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
#plt.style.use('atlas')
import matplotlib.mlab as mlab

def readRoot():
  import ROOT as r
  from sys import stdout,argv
  from math import fabs
  finalmu = options.identifier 
  from dataset import getsamp

  import glob
  #sampweight = getweight(mu)
  sampweight = 1
  filenamebase = options.inputDir 
  filenames = glob.glob(filenamebase+'/*.root')
  if len(filenames) == 0: raise OSError('Can\'t find any Root files to read in '+options.inputDir) 
  tree = r.TChain('oTree')
  for filename in filenames:
    statinfo = os.stat(filename)
    if statinfo.st_size < 10000: continue #sometimes batch jobs fail
    print '== Reading in '+filename+' =='
    tree.Add(filename) 

  # make sure the branches are compatible between the two
  branches = set(i.GetName() for i in tree.GetListOfBranches())
  # required:
  if options.jetpt not in branches: raise RuntimeError(options.jetpt+' branch does not exist. This is the branch containing reco jet pTs.')
  else: print '== \''+options.jetpt+'\' branch is being read as reco jet pTs =='
  if options.npv not in branches: raise RuntimeError(options.npv+' branch does not exist. This is the branch containing NPVs.')
  else: print '== \''+options.npv+'\' branch is being read as NPVs =='
  # optional:
  has_event_weight = False
  has_eta = False
  if options.event_weight not in branches: print '== \''+options.event_weight+'\' branch does not exist; weighting every event the same =='  
  else:
    has_event_weight=True
    print '== \''+options.event_weight+'\' branch is being read as event weights =='
  if options.jeteta not in branches: print '== \''+options.jeteta+'\' branch does not exist; no eta cuts set =='  
  else:
    has_eta = True
    print '== \''+options.jeteta+'\' branch being read as reco jet etas =='

  nentries = tree.GetEntries()

  npvs = [] 
  recopts = []
  weights = [] 
  event_numbers = []

  event_number = 0 
  for jentry in xrange(nentries):
      if jentry>options.numEvents and options.numEvents>0: continue
      tree.GetEntry(jentry)
      
      if not jentry%1000:
          stdout.write('== \r%d events read ==\n'%jentry)
          stdout.flush()

      jpts = getattr(tree,options.jetpt)
      npv = tree.NPV

      if has_eta: jetas = getattr(tree,options.jeteta)
      if has_event_weight: event_weight = tree.event_weight*sampweight

      recopt = []
      weightjets = []
      for i,jpt in enumerate(jpts):
          if has_eta:
            jeta = jetas[i]
            if fabs(jeta)>options.maxeta or fabs(jeta)<options.mineta: continue
          recopt.append(jpt)
          if has_event_weight:
            weightjets.append(event_weight)
          else: weightjets.append(1) #set all events to have the same weight

      npv = [npv]*len(recopt)
      npvs += npv
      event_numbers += [event_number]*len(recopt)
      event_number += 1
      recopts += recopt
      weights += weightjets

  save(options.submitDir+'/recopts_all_'+finalmu,recopts)
  save(options.submitDir+'/npvs_all_'+finalmu,npvs)
  save(options.submitDir+'/event_numbers_all_'+finalmu,event_numbers)
  if has_event_weight: save(options.submitDir+'/weights_all_'+finalmu,weights)

  return array(recopts),array(npvs),array(event_numbers),array(weights)

def calibrate():
  if options.root: 
    recopts,npvs,event_numbers,weights = readRoot()
    eta_cuts = [True]*len(recopts) 
    print '== Root files read. Data saved in '+options.submitDir+'. Next time you can run without -r option and it should be faster. =='
    print '== There are '+str(len(recopts))+' total jets =='
  else:
    filename = options.submitDir+'/'+'recopts_all_'+options.identifier+'.npy'
    if not os.path.exists(filename): raise OSError(filename +' does not exist')
    print '== Loading file <'+filename+'> as reco jet pTs =='
    recopts = load(filename)
    print '== There are '+str(len(recopts))+' total jets =='

    filename = options.submitDir+'/'+'npvs_all_'+options.identifier+'.npy'
    if not os.path.exists(filename): raise OSError(filename +' does not exist')
    print '== Loading file <'+filename+'> as NPVs =='
    npvs = load(filename)
    if not len(npvs)==len(recopts):
      raise RuntimeError('There should be the same number of npvs as truth jets (format is one entry per truth jet)')

    filename = options.submitDir+'/'+'event_numbers_all_'+options.identifier+'.npy'
    if not os.path.exists(filename): raise OSError(filename +' does not exist')
    print '== Loading file <'+filename+'> as event numbers =='
    event_numbers = load(filename)
    if not len(event_numbers)==len(recopts):
      raise RuntimeError('There should be the same number of npvs as truth jets (format is one entry per truth jet)')

    filename = options.submitDir+'/'+'weights_all_'+options.identifier+'.npy'
    if os.path.exists(filename): 
      print '== Loading file <'+filename+'> as event weights =='
      weights = load(filename)
      if not len(weights)==len(recopts):
        raise RuntimeError('There should be the same number of weights as truth jets (format is one entry per truth jet)')
    else:
      print '== '+filename+' does not exist; weighting every event the same =='
      weights = array([1]*len(recopts))

    filename = options.submitDir+'/'+'etas_all_'+options.identifier+'.npy'
    if os.path.exists(filename): 
      print '== Loading file <'+filename+'> as reco jet etas =='
      etas = load(filename)
      if not len(etas)==len(recopts):
        raise RuntimeError('There should be the same number of etas as truth jets')
      eta_cuts = numpy.all([abs(etas)<options.mineta,abs(etas)>options.maxeta]) 
    else:
      print '== '+filename+' does not exist; no additional eta cuts set (if you started reading from a root file, this is ok) =='
      eta_cuts = [True]*len(recopts) 

  
  fits = pickle.load(open(options.submitDir+'/'+'fit_'+options.identifier+'.p','rb'))
  npvedges = fits.keys()
  avg_mults = {n:0 for n in npvedges} 
  err_mults = {n:0 for n in npvedges} 
  npvedges.sort()
  npvbinsize = npvedges[1]-npvedges[0]
  npvedges.insert(0,npvedges[0]-npvbinsize)
  npvbins = digitize(npvs,npvedges)


  for npvbin in xrange(1,len(npvedges)):
    print '>> Processing NPV bin '+str(npvedges[npvbin-1])+'-'+str(npvedges[npvbin])
    ptdata = recopts[npvbins==npvbin]
    npvdata = npvs[npvbins==npvbin]
    weightdata = weights[npvbins==npvbin]
    event_numberdata = event_numbers[npvbins==npvbin]

    fit = fits[npvedges[npvbin]]
    calibptdata = g1(ptdata,*fit)

    event_multiplicities = []
    event_weights = []

    unique_events = set(event_numberdata)
    for event in unique_events:
      event_calib_jets = calibptdata[event_numberdata==event]
      event_weight = weightdata[event_numberdata==event]
      if len(set(event_weight))>1: raise RuntimeError('Not all the event weights for a single event are the same.')
      event_weight = event_weight[0]
      event_multiplicity = len(event_calib_jets[event_calib_jets>options.cut])
      event_multiplicities.append(event_multiplicity)
      event_weights.append(event_weight)

    n,bins,patches = plt.hist(event_multiplicities,normed=True,bins=max(event_multiplicities)-min(event_multiplicities),weights=event_weights,facecolor='b')
    plt.xlabel('Jet Multiplicity ($p_T >$ '+str(options.cut)+' GeV)')
    plt.ylabel('Fraction of Events')
    plt.savefig(options.plotDir+'/'+'jetmultiplicity'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.identifier+'.png')
    plt.close()

    event_weights = event_weights/sum(event_weights)
    avg_mult = average(event_multiplicities,weights=event_weights)
    std_mult = sqrt(average((event_multiplicities-avg_mult)**2,weights=event_weights))
    err_mult = std_mult*sqrt(sum(event_weights**2))
    avg_mults[npvedges[npvbin]] = avg_mult
    err_mults[npvedges[npvbin]] = err_mult

  xs = [npvedges[npvbin]-0.5*npvbinsize for npvbin in xrange(1,len(npvedges))]
  ys = [avg_mults[npvedges[npvbin]] for npvbin in xrange(1,len(npvedges))]
  errs = [err_mults[npvedges[npvbin]] for npvbin in xrange(1,len(npvedges))]
  plt.errorbar(xs,ys,yerr=errs)
  plt.ylabel('Average Jet Multiplicity ($p_T >$ '+str(options.cut)+' GeV)')
  plt.xlabel('NPV')
  plt.xlim(min(npvedges),max(npvedges))
  plt.savefig(options.plotDir+'/'+'jetmultiplicity'+'_'+options.identifier+'.png')

  return avg_mults,err_mults
      
(avg_mults,err_mults) = calibrate()
pickle.dump(avg_mults,open(options.submitDir+'/avg_mults_'+options.identifier+'.p','wb'))
pickle.dump(err_mults,open(options.submitDir+'/err_mults_'+options.identifier+'.p','wb'))
