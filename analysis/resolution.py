from numpy import load,log,linspace,digitize,array,mean,std,exp,all,average,sqrt,asarray,sign,zeros
import os
import numpy
from numpy import save
from scipy.optimize import curve_fit,fsolve
from scipy.stats import norm
from helper_functions import distribution_values
from operator import sub
from optparse import OptionParser
from copy import copy
os.environ[ 'MPLCONFIGDIR' ] = '/tmp/' #to get matplotlib to work

'''try:
  from rootpy.plotting.style import set_style, get_style
  print '== Using ATLAS style =='
  atlas = get_style('ATLAS')
  atlas.SetPalette(51)
  set_style(atlas)
  set_style('ATLAS',mpl=True)
except ImportError: print '== Not using ATLAS style (Can\'t import rootpy.) =='
'''

parser = OptionParser()

# job configuration
parser.add_option("--inputDir", help="Directory containing input files",type=str, default="../data")
parser.add_option("--submitDir", help="Directory containing output files",type=str, default="../output")
parser.add_option("--plotDir", help="Directory containing plots",type=str, default="../plots")
parser.add_option("--numEvents", help="How many events to include (set to -1 for all events)",type=int, default=100000)
parser.add_option("-i","--identifier", help="sample identifier",type=str, default="myjets")
parser.add_option("-r","--root", help="Root input",action="store_true", default=False)
parser.add_option("-d","--debug", help="Debug",action="store_true", default=False)

# Root configuration
## Reconstructed jets and matched truth jets
parser.add_option("--jetpt", help="reco jet pT branch name",type=str, default="j0pt")
parser.add_option("--tjetpt", help="matched truth jet pT branch name",type=str, default="tj0pt")
parser.add_option("--npv", help="NPV branch name",type=str, default="NPV")
parser.add_option("--mu", help="mu branch name",type=str, default="mu")
parser.add_option("--tjeteta", help="matched truth jet eta branch name",type=str, default="tj0eta")
parser.add_option("--tjetmindr", help="matched truth jet mindr branch name",type=str, default="tj0mindr")
parser.add_option("--jetmindr", help="reco jet mindr branch name",type=str, default="j0mindr")
parser.add_option("--event_weight", help="event weight branch name",type=str, default="event_weight")
parser.add_option("--jetisPU", help="branch name for is pileup indicator on reco jets (only necessary for fake calculation)",type=str, default="j0isPU")
parser.add_option("--jeteta", help="branch name for jet eta (only necessary for fake calculation)",type=str, default="j0eta")
## All truth jets (only required if using absolute scale) 
parser.add_option("--all_tjetpt", help="all truth jet pT branch name",type=str, default="tjpt")
parser.add_option("--all_tjeteta", help="all truth jet eta branch name",type=str, default="tjeta")
parser.add_option("--all_tjetmindr", help="all truth jet mindr branch name",type=str, default="tjmindr")

# jet configuration
parser.add_option("-c","--cut", default=float('-inf'), type=float, help="low pT cut on reco jets")
parser.add_option("--mineta", help="min abs(eta) on truth jets", type=float, default=0)
parser.add_option("--maxeta", help="max abs(eta) on truth jets", type=float, default=float('inf'))
parser.add_option("--mindr", help="min dr on truth jets", type=float, default=0)
parser.add_option("--reco_mindr", help="min dr on reco jets", type=float, default=0)

# analysis configuration
parser.add_option("-e","--absolute",help="Calculate efficiency as well",action="store_true",default=False)
parser.add_option("--doEst",help="Estimate resolution rather than doing full numerical inversion",action="store_true",default=False)
parser.add_option("-m","--central",help="Choice of notion of central tendency/resolution (mean, mode, median, absolute_median, or trimmed)",type='choice',choices=['mean','mode','median','absolute_median','trimmed','kde_mode'],default='mode')
parser.add_option("--minnpv", help="min NPV", type=int, default=5)
parser.add_option("--maxnpv", help="max NPV", type=int, default=30)
parser.add_option("--npvbin", help="size of NPV bins", type=int, default=5)
parser.add_option("--minpt", help="min truth pt", type=int, default=20)
parser.add_option("--maxpt", help="max truth pt", type=int, default=80)
parser.add_option("--ptbin", help="size of pT bins", type=int, default=2)
parser.add_option("--fakept", help="minmum (calibrated) reco pT for fake", type=float, default=20)
parser.add_option("--doFake", help="calculate fake rate", action="store_true", default=False)

(options, args) = parser.parse_args()

if options.central == 'kde_mode': raise RuntimeError('kde_mode option is deprecated currently. Use mode option.')
if options.doEst: raise RuntimeError('doEst is currently deprecated. Last version that worked with doEst was v1.0.')

if options.root and not os.path.exists(options.inputDir): raise OSError(options.inputDir +' does not exist. This is where the input Root files go.')
if not options.root and not os.path.exists(options.submitDir): raise OSError(options.submitDir+' does not exist. This is where the input numpy files go.')
if options.root and not os.path.exists(options.submitDir):
  print '== Making folder '+options.submitDir+' =='
  os.makedirs(options.submitDir)
if not os.path.exists(options.plotDir):
  print '== Making folder '+options.plotDir+' =='
  os.makedirs(options.plotDir)

identifier = options.identifier
do_all = False
if options.cut==float('-inf'): do_all=True 
if not do_all: identifier+='_c'+str(int(options.cut))

absolute = options.absolute

import pdb

asym = 10 # shift distribution to the right to get better fit and have R(0) be finite
def R(x,a,b,c):
    ax = abs(array(x))
    result = a + b/log(ax+asym) + c/log(ax+asym)**2
    return result 

def g(x,a,b,c):
    ax = array(x)
    return R(x,a,b,c)*ax 

#derivative of g
def dg(x,a,b,c):
    ax = abs(array(x))
    #chain rule:
    result = a + b/log(ax+asym) + c/log(ax+asym)**2
    result = result + (- b/log(ax+asym)**2*ax/(ax+asym) - 2*c/log(ax+asym)**3*ax/(ax+asym))
    return result

memoized = {} #memoize results to reduce computation time
def g1(x,a,b,c):
    ax = array(x)
    result = []
    for x in ax:
      approx = (round(x,2),round(a,2),round(b,2),round(c,2))
      if approx not in memoized:
        func = lambda y: approx[0]-g(y,a,b,c)
        if approx[0]>=0: x0 = max([approx[0],10])
        if approx[0]<0: x0 = min([approx[0],-10])
        memoized[approx] = fsolve(func,x0)[0]
        #print approx,memoized[approx]
      result.append(memoized[approx])
    return array(result)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
#plt.style.use('atlas')
import matplotlib.mlab as mlab

import ROOT as r
def readRoot():
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
  if options.tjetpt not in branches: raise RuntimeError(options.tjetpt+' branch does not exist. This is the branch containing matched truth jet pTs.')
  else: print '== \''+options.tjetpt+'\' branch is being read as truth jet pTs =='
  if options.npv not in branches: raise RuntimeError(options.npv+' branch does not exist. This is the branch containing NPVs.')
  else: print '== \''+options.npv+'\' branch is being read as NPVs =='
  if options.mu not in branches: print '== \''+options.mu+'\' branch does not exist; mu is not being recorded =='
  else: print '== \''+options.mu+'\' branch is being read as mus =='
  # optional:
  has_event_weight = False
  has_eta = False
  has_mindr = False
  if options.event_weight not in branches: print '== \''+options.event_weight+'\' branch does not exist; weighting every event the same =='  
  else:
    has_event_weight=True
    print '== \''+options.event_weight+'\' branch is being read as event weights =='
  if options.tjeteta not in branches: print '== \''+options.tjeteta+'\' branch does not exist; no eta cuts set on matched truth jets =='  
  else:
    has_eta = True
    print '== \''+options.tjeteta+'\' branch being read as matched truth jet etas =='
  if options.tjetmindr not in branches: print '== \''+options.tjetmindr+'\' branch does not exist; no mindr cuts set on matched truth jets=='  
  else:
    has_mindr = True
    print '== \''+options.tjetmindr+'\' branch being read as matched truth jet mindrs =='
  if options.jetmindr not in branches: print '== \''+options.jetmindr+'\' branch does not exist; no mindr cuts set on reco jets=='  
  else:
    has_reco_mindr = True
    print '== \''+options.jetmindr+'\' branch being read as reco jet mindrs =='
  if options.doFake:
    if options.jetisPU not in branches: print '== \''+options.jetisPU+'\' branch does not exist; not calculating fake rate=='  
    else:
      has_jetisPU = True
      print '== \''+options.jetisPU+'\' branch being read as indicator that reco jet is PU =='
    if options.jeteta not in branches: print '== \''+options.jeteta+'\' branch does not exist; not calculating fake rate=='  
    else:
      has_jeteta = True
      print '== \''+options.jeteta+'\' branch being read as jet etas =='
  doFake = has_jetisPU and has_jeteta and options.doFake

  if absolute:
    if options.all_tjetpt not in branches: raise RuntimeError(options.all_tjetpt+' branch does not exist. This is the branch containing all the truth jet pTs. Required for absolute/efficiency calculation.')
    has_all_eta = False
    has_all_mindr = False
    if options.all_tjeteta not in branches: print '== \''+options.all_tjeteta+'\' branch does not exist; no eta cuts set on all truth jets =='  
    else:
      has_all_eta = True
      print '== \''+options.tjeteta+'\' branch being read as all truth jet etas =='
    if options.all_tjetmindr not in branches: print '== \''+options.all_tjetmindr+'\' branch does not exist; no mindr cuts set on all truth jets =='  
    else:
      has_all_mindr = True
      print '== \''+options.tjetmindr+'\' branch being read as all truth jet mindrs =='

  nentries = tree.GetEntries()

  npvs = [] 
  mus = []
  responses = [] 
  truepts = [] 
  recopts = []
  weights = [] 
  etas = []
  mindrs = []
  reco_mindrs = []
  PU_recopts = {npv: [] for npv in range(options.minnpv,options.maxnpv)}
  PU_etas = {npv: [] for npv in range(options.minnpv,options.maxnpv)}
  PU_weights = {npv: [] for npv in range(options.minnpv,options.maxnpv)}
  store_fakept = float('-inf') #store PU jets down to x to save space; update continuously as gather more data to estimate response at fake pT cutoff

  if absolute:
    all_weights = []
    all_npvs = []
    all_mus = []
    all_truepts = []
    all_etas = []
    all_mindrs = []

  for jentry in xrange(nentries):
      if jentry>options.numEvents and options.numEvents>0: continue
      tree.GetEntry(jentry)
      
      if not jentry%1000:
        stdout.write('== \r%d events read ==\n'%jentry)
        stdout.flush()
        if jentry>0:
          arr_truepts = array(truepts)
          arr_recopts = array(recopts)
          arr_weights = array(weights)
          ptdata = arr_recopts[all([arr_truepts>options.fakept,arr_truepts<options.fakept+options.ptbin],axis=0)]
          weightdata = arr_weights[all([arr_truepts>options.fakept,arr_truepts<options.fakept+options.ptbin],axis=0)]
          store_fakept = average(ptdata,weights=weightdata)*0.5

      jpts = getattr(tree,options.jetpt)
      tjpts = getattr(tree,options.tjetpt)
      if not len(jpts)==len(tjpts):
        raise RuntimeError('There should be the same number of reco jets as truth jets')
      npv = tree.NPV
      mu = tree.mu

      if has_eta:
        tjetas = getattr(tree,options.tjeteta)
        if not len(tjetas)==len(tjpts):
          raise RuntimeError('There should be the same number of truth etas as truth jets')
      if has_mindr:
        tjmindrs = getattr(tree,options.tjetmindr)
        if not len(tjmindrs)==len(tjpts):
          raise RuntimeError('There should be the same number of truth mindrs as truth jets')
      if has_reco_mindr:
        jmindrs = getattr(tree,options.jetmindr)
        if options.reco_mindr==0: jmindrs = [1]*len(jpts) #if reco mindr cut is 0, don't care what they actually are
        if not len(jmindrs)==len(jpts):
          raise RuntimeError('There should be the same number of reco mindrs as reco jets')
      if doFake:
        jisPUs = getattr(tree,options.jetisPU)
        jetas = getattr(tree,options.jeteta)
        if not len(jisPUs)==len(jpts):
          raise RuntimeError('Each reco jet should have an indicator that it is PU or not')
        if not len(jetas)==len(jpts):
          raise RuntimeError('There should be the same number of reco etas as reco jets')
      if has_event_weight: event_weight = tree.event_weight*sampweight

      truept = []
      recopt = []
      weightjets = []
      eta = []
      mindr = []
      reco_mindr = []
      PU_recopt = []
      PU_eta = []
      for i,(jpt,tjpt) in enumerate(zip(jpts,tjpts)):
          if doFake and jpt>store_fakept:
            jisPU = jisPUs[i]
            jeta = jetas[i]
            if jisPU and fabs(jeta)<options.maxeta and fabs(jeta)>options.mineta:
              PU_recopt.append(jpt) #PU jets - no requirement
              PU_eta.append(jeta) #PU jets - no requirement
          if has_eta:
            tjeta = tjetas[i]
            if fabs(tjeta)>options.maxeta or fabs(tjeta)<options.mineta: continue
          if has_mindr:
            tjmindr = tjmindrs[i]
            if tjmindr<options.mindr: continue
          if has_reco_mindr:
            jmindr = jmindrs[i]
            #if jmindr<options.reco_mindr: continue
          truept.append(tjpt)
          recopt.append(jpt)
          if has_eta: eta.append(tjeta)
          if has_mindr: mindr.append(tjmindr)
          if has_reco_mindr: reco_mindr.append(jmindr)
          if has_event_weight:
            weightjets.append(event_weight)
          else: weightjets.append(1) #set all events to have the same weight

      jet_npv = [npv]*len(truept)
      npvs += jet_npv
      jet_mu = [mu]*len(truept)
      mus += jet_mu
      truepts += truept
      recopts += recopt
      weights += weightjets
      etas += eta
      mindrs += mindr
      reco_mindrs += reco_mindr
      if npv in PU_recopts:
        PU_recopts[npv].append(PU_recopt)
        PU_etas[npv].append(PU_eta)
        if has_event_weight: PU_weights[npv].append(event_weight)
        else: PU_weights[npv].append(1)
      else:
        PU_recopts[npv] = [PU_recopt]
        PU_etas[npv] = [PU_eta]
        if has_event_weight: PU_weights[npv]=[event_weight]
        else: PU_weights[npv]=[1]

      if absolute:
        all_tjpts = getattr(tree,options.all_tjetpt)

        if has_all_eta: all_tjetas = getattr(tree,options.all_tjeteta)
        if has_all_mindr: all_tjmindrs = getattr(tree,options.all_tjetmindr)

        all_truept = []
        all_weightjets = []
        all_eta = []
        all_mindr = []
        for i,tjpt in enumerate(all_tjpts):
            if has_all_eta:
              all_tjeta = all_tjetas[i]
              if fabs(all_tjeta)>options.maxeta or fabs(all_tjeta)<options.mineta: continue
            if has_all_mindr:
              all_tjmindr = all_tjmindrs[i]
              if all_tjmindr<options.mindr: continue
            all_truept.append(tjpt)
            all_eta.append(all_tjeta)
            all_mindr.append(all_tjmindr)
            if has_event_weight:
              all_weightjets.append(event_weight)
            else: weightjets.append(1) #set all events to have the same weight

        all_npv = [npv]*len(all_truept)
        all_npvs += all_npv
        all_mu = [mu]*len(all_truept)
        all_mus += all_mu
        all_truepts += all_truept
        all_weights += all_weightjets
        all_etas += all_eta
        all_mindrs += all_mindr
  #end loop over entries

  save(options.submitDir+'/truepts_'+finalmu,truepts)
  save(options.submitDir+'/recopts_'+finalmu,recopts)
  save(options.submitDir+'/npvs_'+finalmu,npvs)
  save(options.submitDir+'/mus_'+finalmu,mus)
  if has_eta: save(options.submitDir+'/etas_'+finalmu,etas)
  if has_mindr: save(options.submitDir+'/mindrs_'+finalmu,mindrs)
  if has_reco_mindr: save(options.submitDir+'/reco_mindrs_'+finalmu,reco_mindrs)
  if has_event_weight: save(options.submitDir+'/weights_'+finalmu,weights)
  if doFake:
    pickle.dump(PU_recopts,open(options.submitDir+'/PU_recopts_'+finalmu+'.p','wb'))
    pickle.dump(PU_etas,open(options.submitDir+'/PU_etas_'+finalmu+'.p','wb'))
    pickle.dump(PU_weights,open(options.submitDir+'/PU_weights_'+finalmu+'.p','wb'))

  if absolute:
    save(options.submitDir+'/all_truepts_'+finalmu,all_truepts)
    save(options.submitDir+'/all_npvs_'+finalmu,all_npvs)
    save(options.submitDir+'/all_mus_'+finalmu,all_mus)
    save(options.submitDir+'/all_etas_'+finalmu,all_etas)
    save(options.submitDir+'/all_mindrs_'+finalmu,all_mindrs)
    if has_event_weight: save(options.submitDir+'/all_weights_'+finalmu,all_weights)

  if absolute:
    return array(recopts),array(truepts),array(npvs),array(weights),array(all_truepts),array(all_npvs),array(all_weights)
  else:
    return array(recopts),array(truepts),array(npvs),array(weights)

def fitres(params=[]):
  if options.root: 
    if absolute:
      recopts,truepts,npvs,weights,all_truepts,all_npvs,all_weights = readRoot()
      all_eta_cuts = [True]*len(all_truepts)
      all_mindr_cuts = [True]*len(all_truepts)
    else:
      recopts,truepts,npvs,weights = readRoot()
    eta_cuts = [True]*len(truepts) 
    mindr_cuts = [True]*len(truepts)
    print '== Root files read. Data saved in '+options.submitDir+'. Next time you can run without -r option and it should be significantly faster. =='
    print '== There are '+str(len(truepts))+' total matched jets =='
    if absolute: print '== There are '+str(len(all_truepts))+' total truth jets =='
  else:
    filename = options.submitDir+'/'+'truepts_'+options.identifier+'.npy'
    print filename
    if not os.path.exists(filename): raise OSError(filename +' does not exist')
    print '== Loading file <'+filename+'> as truth jet pTs =='
    truepts = load(filename)
    print '== There are '+str(len(truepts))+' total jets =='

    filename = options.submitDir+'/'+'recopts_'+options.identifier+'.npy'
    if not os.path.exists(filename): raise OSError(filename +' does not exist')
    print '== Loading file <'+filename+'> as reco jet pTs =='
    recopts = load(filename)
    if not len(recopts)==len(truepts):
      raise RuntimeError('There should be the same number of reco jets as truth jets')

    filename = options.submitDir+'/'+'npvs_'+options.identifier+'.npy'
    if not os.path.exists(filename): raise OSError(filename +' does not exist')
    print '== Loading file <'+filename+'> as NPVs =='
    npvs = load(filename)
    if not len(npvs)==len(truepts):
      raise RuntimeError('There should be the same number of npvs as truth jets (format is one entry per truth jet)')

    filename = options.submitDir+'/'+'weights_'+options.identifier+'.npy'
    if os.path.exists(filename): 
      print '== Loading file <'+filename+'> as event weights =='
      weights = load(filename)
      if not len(weights)==len(truepts):
        raise RuntimeError('There should be the same number of weights as truth jets (format is one entry per truth jet)')
    else:
      print '== '+filename+' does not exist; weighting every event the same =='
      weights = array([1]*len(truepts))

    filename = options.submitDir+'/'+'etas_'+options.identifier+'.npy'
    if os.path.exists(filename): 
      print '== Loading file <'+filename+'> as truth jet etas =='
      etas = load(filename)
      if not len(etas)==len(truepts):
        raise RuntimeError('There should be the same number of etas as truth jets')
      eta_cuts = numpy.all([abs(etas)>options.mineta,abs(etas)<options.maxeta],axis=0) 
    else:
      print '== '+filename+' does not exist; no eta cuts set =='
      eta_cuts = [True]*len(truepts) 

    filename = options.submitDir+'/'+'mindrs_'+options.identifier+'.npy'
    if os.path.exists(filename):
      print '== Loading file <'+filename+'> as truth jet mindRs =='
      mindrs = load(filename)
      if not len(mindrs)==len(truepts):
        raise RuntimeError('There should be the same number of mindRs as truth jets')
      mindr_cuts = mindrs>options.mindr
    else:
      print '== '+filename+' does not exist; no mindR cuts set =='
      mindr_cuts = [True]*len(truepts) 
  

    if absolute:
      filename = options.submitDir+'/'+'all_truepts_'+options.identifier+'.npy'
      if not os.path.exists(filename): raise OSError(filename +' does not exist')
      print '== Loading file <'+filename+'> as all truth jet pTs =='
      all_truepts = load(filename)
      print '== There are '+str(len(all_truepts))+' total truth jets =='

      filename = options.submitDir+'/'+'all_npvs_'+options.identifier+'.npy'
      if not os.path.exists(filename): raise OSError(filename +' does not exist')
      print '== Loading file <'+filename+'> as NPVs =='
      all_npvs = load(filename)
      if not len(all_npvs)==len(all_truepts):
        raise RuntimeError('There should be the same number of npvs as truth jets (format is one entry per truth jet)')

      filename = options.submitDir+'/'+'all_weights_'+options.identifier+'.npy'
      if os.path.exists(filename): 
        print '== Loading file <'+filename+'> as event weights =='
        all_weights = load(filename)
        if not len(all_weights)==len(all_truepts):
          raise RuntimeError('There should be the same number of weights as truth jets (format is one entry per truth jet)')
      else:
        print '== '+filename+' does not exist; weighting every event the same =='
        all_weights = array([1]*len(all_truepts))

      filename = options.submitDir+'/'+'all_etas_'+options.identifier+'.npy'
      if os.path.exists(filename): 
        print '== Loading file <'+filename+'> as truth jet etas =='
        all_etas = load(filename)
        if not len(all_etas)==len(all_truepts):
          raise RuntimeError('There should be the same number of etas as truth jets')
        all_eta_cuts = numpy.all([abs(all_etas)>options.mineta,abs(all_etas)<options.maxeta],axis=0) 
      else:
        print '== '+filename+' does not exist; no eta cuts set =='
        all_eta_cuts = [True]*len(all_truepts) 

      filename = options.submitDir+'/'+'all_mindrs_'+options.identifier+'.npy'
      if os.path.exists(filename):
        print '== Loading file <'+filename+'> as truth jet mindRs =='
        all_mindrs = load(filename)
        if not len(all_mindrs)==len(all_truepts):
          raise RuntimeError('There should be the same number of mindRs as truth jets')
        all_mindr_cuts = all_mindrs>options.mindr
      else:
        print '== '+filename+' does not exist; no mindR cuts set =='
        all_mindr_cuts = [True]*len(all_truepts) 

  #do this if reading from root or not
  filename = options.submitDir+'/'+'reco_mindrs_'+options.identifier+'.npy'
  if os.path.exists(filename):
    if not options.root: print '== Loading file <'+filename+'> as reco jet mindRs =='
    reco_mindrs = load(filename)
    if not len(reco_mindrs)==len(truepts):
      raise RuntimeError('There should be the same number of mindRs as truth jets')
    reco_mindr_cuts = reco_mindrs>=options.reco_mindr
  else:
    print '== '+filename+' does not exist; no reco mindR cuts set =='
    reco_mindr_cuts = [True]*len(truepts) 

  filenames = [options.submitDir+'/PU_recopts_'+options.identifier+'.p',
               options.submitDir+'/PU_etas_'+options.identifier+'.p',
               options.submitDir+'/PU_weights_'+options.identifier+'.p']
  doFake = options.doFake 
  for filename in filenames:
    if not os.path.exists(filename) and doFake:
      print '== '+filename+' does not exist; not calculating fake rates'
      doFake = False
  if doFake:
    for filename in filenames: print '== Loading file <'+filename+'> for fake jet calculation =='
    PU_recopts = pickle.load(open(filenames[0],'rb'))
    PU_etas = pickle.load(open(filenames[1],'rb'))
    PU_weights = pickle.load(open(filenames[2],'rb'))
    if not (PU_recopts.keys()==PU_etas.keys() and PU_recopts.keys()==PU_weights.keys()): raise RuntimeError('NPVs don\'t match between fake jet files')
    for k in PU_recopts.keys():
      if not (len(PU_recopts[k])==len(PU_etas[k]) and len(PU_recopts[k])==len(PU_etas[k])): raise RuntimeError('Don\'t have same number of events in fake jet files for NPV = '+k)

  maxpt = options.maxpt
  if (options.maxpt-options.minpt)%options.ptbin==0: maxpt+=1
  ptedges = range(options.minpt,maxpt,options.ptbin)
  cuts = all([truepts>min(ptedges),recopts>options.cut,eta_cuts,mindr_cuts,reco_mindr_cuts],axis=0)

  recopts = recopts[cuts]
  truepts = truepts[cuts]
  responses = recopts/truepts
  npvs = npvs[cuts]
  weights = weights[cuts]

  incl_ptests = copy(recopts) #replace recopts with estpts as they come up (only within given NPV and pT ranges)
  incl_resests = copy(responses) #replace recopts with resests as they come up (only within given NPV and pT ranges)

  ptbins = digitize(truepts,ptedges)

  maxnpv = options.maxnpv
  if (options.maxnpv-options.minnpv)%options.npvbin==0: maxnpv+=1
  npvedges = range(options.minnpv,maxnpv,options.npvbin)
  npvbins = digitize(npvs,npvedges)

  if absolute:
    all_cuts = all([all_truepts>options.minpt,all_eta_cuts,all_mindr_cuts],axis=0)
    all_truepts = all_truepts[all_cuts]
    all_npvs = all_npvs[all_cuts]
    all_weights = all_weights[all_cuts]

    all_ptbins = digitize(all_truepts,ptedges)
    all_npvbins = digitize(all_npvs,npvedges)

  npv_sigmas = {npvedges[npvbin]: [] for npvbin in xrange(1,len(npvedges))}
  npv_sigma_errs = {npvedges[npvbin]: [] for npvbin in xrange(1,len(npvedges))}
  npv_sigmaRs = {npvedges[npvbin]: [] for npvbin in xrange(1,len(npvedges))}
  npv_sigmaR_errs = {npvedges[npvbin]: [] for npvbin in xrange(1,len(npvedges))}
  npv_closures = {npvedges[npvbin]: [] for npvbin in xrange(1,len(npvedges))}
  npv_closure_errs = {npvedges[npvbin]: [] for npvbin in xrange(1,len(npvedges))}
  Ropts = {npvedges[npvbin]: [] for npvbin in xrange(1,len(npvedges))}

  if absolute:
    npv_efficiencies = {npvedges[npvbin]: [] for npvbin in xrange(1,len(npvedges))}
    npv_efficiencies_err = {npvedges[npvbin]: [] for npvbin in xrange(1,len(npvedges))}
    npv_efficiencies_fom = {npvedges[npvbin]: [] for npvbin in xrange(1,len(npvedges))}
    npv_efficiencies_err_fom = {npvedges[npvbin]: [] for npvbin in xrange(1,len(npvedges))}

  #responses,recopts,trupts,weights
  for npvbin in xrange(1,len(npvedges)):
    print '>> Processing NPV bin '+str(npvedges[npvbin-1])+'-'+str(npvedges[npvbin])
    avgtruept = []
    avgres = []
    avgres_errs = []
    sigmaRs = []
    sigmaR_errs = []
    avgpt = []
    avgpt_errs = []
    sigmas = []
    sigma_errs = []
    if absolute:
      efficiencies = [] 
      efficiencies_err = [] 

    for ptbin in xrange(1,len(ptedges)): 
      #print '>> >> Processing pT bin '+str(ptedges[ptbin-1])+'-'+str(ptedges[ptbin])+' GeV'
      resdata = responses[all([ptbins==ptbin,npvbins==npvbin],axis=0)]
      ptdata = recopts[all([ptbins==ptbin,npvbins==npvbin],axis=0)]
      trueptdata = truepts[all([ptbins==ptbin,npvbins==npvbin],axis=0)]
      weightdata = weights[all([ptbins==ptbin,npvbins==npvbin],axis=0)]

      avgtruept.append(average(trueptdata,weights=weightdata))
      if len(resdata)<100: print 'Low statistics ('+str(len(resdata))+' jets) in bin with pT = ' +str(ptedges[ptbin])+' and NPV between '+str(npvedges[npvbin-1])+' and '+str(npvedges[npvbin])
      n,bins,patches = plt.hist(resdata,normed=True,bins=int((max(resdata)-min(resdata))*60/2),weights=weightdata,facecolor='b',histtype='stepfilled')
      if absolute:
        all_weightdata = all_weights[all([all_ptbins==ptbin,all_npvbins==npvbin],axis=0)]
        efficiency = sum(weightdata)/sum(all_weightdata)
        efficiency_err = sqrt(sum(weightdata**2))/sum(all_weightdata)
        if efficiency>1:
          #raise RuntimeError('Efficiency > 1. Check truth jets?')
          efficiency=1
        efficiencies.append(efficiency)
        efficiencies_err.append(efficiency_err)

      if options.central == 'absolute_median' or options.central == 'mode' or options.central == 'kde_mode':
        if not absolute: raise RuntimeError('In order to use absolute IQR, you have to have all truth jets and calculate efficiency. Use -e option.')
        (mu,mu_err,sigma,sigma_err,upper_quantile,lower_quantile,err) = distribution_values(resdata,weightdata,options.central,eff=efficiency)
        if options.debug: print 'pT = ' +str(ptedges[ptbin-1])+'-'+str(ptedges[ptbin])+': '+'Mode ' + str(mu) + '; Absolute IQR ' + str(sigma)
        if err: print '<< In pT bin '+str(ptbin)+' ('+str(ptedges[ptbin-1])+'-'+str(ptedges[ptbin])+' GeV) >>'
        plt.plot((mu,mu),(0,plt.ylim()[1]),'r--',linewidth=2)
        height = 0.607*max(n) #height at x=1*sigma in normal distribution
        if lower_quantile>float('-inf'):
          plt.plot((lower_quantile,upper_quantile),(height,height),'r--',linewidth=2)
          plt.plot((lower_quantile,lower_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
        else:
          plt.plot((mu,upper_quantile),(height,height),'r--',linewidth=2)
        plt.plot((upper_quantile,upper_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
      if options.central == 'median':
        (mu,mu_err,sigma,sigma_err,upper_quantile,lower_quantile) = distribution_values(resdata,weightdata,options.central)
        print mu
        plt.plot((mu,mu),(0,plt.ylim()[1]),'r--',linewidth=2)
        height = 0.607*max(n) #height at x=1*sigma in normal distribution
        plt.plot((lower_quantile,upper_quantile),(height,height),'r--',linewidth=2)
        plt.plot((lower_quantile,lower_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
        plt.plot((upper_quantile,upper_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
      if options.central == 'mean':
        (mu,mu_err,sigma,sigma_err) = distribution_values(resdata,weightdata,options.central)
        gfunc = norm
        y = gfunc.pdf( bins, mu, sigma)
        normal = sum(n)/sum(y) 
        y = y*normal #normalize
        plt.plot((mu,mu),(0,gfunc.pdf(mu,mu,sigma)*normal),'r--',linewidth=2)
        l = plt.plot(bins, y, 'r--', linewidth=2)
      '''if options.central == 'mode':
        (mu,mu_err,sigma,sigma_err,kernel) = distribution_values(resdata,weightdata,options.central)
        y = kernel(bins)
        plt.plot(bins,y,'r--',linewidth=2)
        plt.plot((mu,mu),(0,kernel(mu)),'r--',linewidth=2)'''
      if options.central == 'trimmed':
        (mu,mu_err,sigma,sigma_err,lower,upper) = distribution_values(resdata,weightdata,options.central)
        if options.debug: print 'pT = ' +str(ptedges[ptbin-1])+'-'+str(ptedges[ptbin])+': '+'Mode ' + str(mu) + '; Fitted width ' + str(sigma)
        gfunc = norm
        y = gfunc.pdf(bins, mu, sigma)
        normal = sum(n)/sum(y) 
        y = y*normal #normalize
        plt.plot((mu,mu),(0,gfunc.pdf(mu,mu,sigma)*normal),'r--',linewidth=2)
        newbins = bins[all([bins>lower,bins<upper],axis=0)]
        newy = y[all([bins>lower,bins<upper],axis=0)]
        l = plt.plot(newbins, newy, 'r--', linewidth=2)
      plt.xlabel('$p_T^{reco}/p_T^{true}$')
      plt.ylabel('a.u.')
      plt.xlim(-0.5,3.0)
      plt.ylim(0,3.5)
      plt.savefig(options.plotDir+'/resbin%d'%ptbin+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.central+'_'+identifier+'.png')
      plt.close()
      avgres.append(mu)
      avgres_errs.append(mu_err)
      sigmaRs.append(sigma)
      sigmaR_errs.append(sigma_err)

      n,bins,patches = plt.hist(ptdata,normed=True,bins=100,weights=weightdata,histtype='stepfilled')
      if options.central == 'absolute_median' or options.central == 'mode' or options.central == 'kde_mode':
        (mu,mu_err,sigma,sigma_err,upper_quantile,lower_quantile,err) = distribution_values(ptdata,weightdata,options.central,eff=efficiency)
        if err: print '<< In pT bin '+str(ptbin)+' ('+str(ptedges[ptbin-1])+'-'+str(ptedges[ptbin])+' GeV) >>'
        plt.plot((mu,mu),(0,max(n)),'r--',linewidth=2)
        height = 0.607*max(n) #height at x=1*sigma in normal distribution
        if lower_quantile>float('-inf'):
          plt.plot((lower_quantile,upper_quantile),(height,height),'r--',linewidth=2)
          plt.plot((lower_quantile,lower_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
        else:
          plt.plot((mu,upper_quantile),(height,height),'r--',linewidth=2)
        plt.plot((upper_quantile,upper_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
      if options.central == 'median':
        (mu,mu_err,sigma,sigma_err,upper_quantile,lower_quantile) = distribution_values(ptdata,weightdata,options.central)
        plt.plot((mu,mu),(0,max(n)),'r--',linewidth=2)
        height = 0.607*max(n) #height at x=1*sigma in normal distribution
        plt.plot((lower_quantile,upper_quantile),(height,height),'r--',linewidth=2)
        plt.plot((lower_quantile,lower_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
        plt.plot((upper_quantile,upper_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
      if options.central == 'mean':
        (mu,mu_err,sigma,sigma_err) = distribution_values(ptdata,weightdata,options.central)
        gfunc = norm
        y = gfunc.pdf( bins, mu, sigma)
        normal = sum(n)/sum(y) 
        y = y*normal #normalize
        plt.plot((mu,mu),(0,gfunc.pdf(mu,mu,sigma)*normal),'r--',linewidth=2)
        l = plt.plot(bins, y, 'r--', linewidth=2)
      '''if options.central == 'mode':
        (mu,mu_err,sigma,sigma_err,kernel) = distribution_values(ptdata,weightdata,options.central)
        y = kernel(bins)
        plt.plot((mu,mu),(0,kernel(mu)),'r--',linewidth=2)
        plt.plot(bins,y,'r--',linewidth=2)'''
      if options.central == 'trimmed':
        (mu,mu_err,sigma,sigma_err,lower,upper) = distribution_values(ptdata,weightdata,options.central)
        #print mu,sigma,ptbin
        newbins = bins[all([bins>lower,bins<upper],axis=0)]
        gfunc = norm
        y = gfunc.pdf( bins, mu, sigma)
        normal = sum(n)/sum(y) 
        y = y*normal #normalize
        plt.plot((mu,mu),(0,gfunc.pdf(mu,mu,sigma)*normal),'r--',linewidth=2)
        newbins = bins[all([bins>lower,bins<upper],axis=0)]
        newy = y[all([bins>lower,bins<upper],axis=0)]
        l = plt.plot(newbins, newy, 'r--', linewidth=2)
      plt.xlabel('$p_T^{reco}$')
      plt.ylabel('a.u.')
      plt.savefig(options.plotDir+'/fbin%d'%ptbin+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.central+'_'+identifier+'.png')
      plt.close()
      avgpt.append(mu)
      sigmas.append(sigma)
      avgpt_errs.append(mu_err)
      sigma_errs.append(sigma_err)

    if absolute:
      npv_efficiencies[npvedges[npvbin]] = efficiencies
      npv_efficiencies_err[npvedges[npvbin]] = efficiencies_err
      plt.errorbar(avgtruept,efficiencies,color='g',marker='o',linestyle='',yerr=efficiencies_err)
      plt.xlabel('$p_T^{true}$ [GeV]')
      plt.ylabel('Reconstruction Efficiency')
      plt.xlim(0,options.maxpt+10)
      plt.ylim(min(efficiencies)-0.1,1.0)
      plt.savefig(options.plotDir+'/jetefficiency_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.central+'_'+identifier+'.png')
      plt.close()

    xp = linspace(5,150,75)

    #Fit to response vs. pTtrue
    Ropt, Rcov = curve_fit(R, avgtruept, avgres)
    Ropts[npvedges[npvbin]] = Ropt 

    plt.plot(truepts[npvbins==npvbin],responses[npvbins==npvbin],'.',xp,R(xp,*Ropt),'r-')
    plt.errorbar(avgtruept,avgres,color='g',marker='o',linestyle='',yerr=avgres_errs)
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('$p_T^{reco}/p_T^{true}$')
    if do_all: plt.ylim(-0.5,2)
    else: plt.ylim(0,2)
    plt.xlim(0,options.maxpt+10)
    plt.savefig(options.plotDir+'/jetresponse_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.central+'_'+identifier+'.png')
    plt.close()

    #g = R*t:
    print Ropt
    plt.plot(truepts[npvbins==npvbin],recopts[npvbins==npvbin],'.',xp,R(xp,*Ropt)*array(xp),'r-')
    plt.errorbar(avgtruept,avgpt,color='g',marker='o',linestyle='',yerr=avgpt_errs)
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('$p_T^{reco}$ [GeV]')
    if do_all: plt.ylim(-10,options.maxpt+10)
    else: plt.ylim(0,options.maxpt+10)
    plt.xlim(0,options.maxpt+10)
    plt.savefig(options.plotDir+'/jetf_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.central+'_'+identifier+'.png')
    plt.close()

    #dg = d(R*t):
    plt.plot(xp,dg(xp,*Ropt),'r-')
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('$f\'(p_T^{true})$')
    plt.ylim(0,1)
    plt.xlim(0,options.maxpt+10)
    plt.savefig(options.plotDir+'/jetdf_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.central+'_'+identifier+'.png')
    plt.close()

    calmuRs = []
    calmuR_errs = []
    calsigmaRs = []
    calsigmaR_errs = []
    calmus = []
    calmu_errs = []
    calsigmas = []
    calsigma_errs = []
    if absolute:
      efficiencies_fom = []
      efficiency_errs_fom = []
    for ptbin in xrange(1,len(ptedges)): 
      ptdata = recopts[all([ptbins==ptbin,npvbins==npvbin],axis=0)]
      trueptdata = truepts[all([ptbins==ptbin,npvbins==npvbin],axis=0)]
      weightdata = weights[all([ptbins==ptbin,npvbins==npvbin],axis=0)]
      ptestdata = g1(ptdata,*Ropt)
      incl_ptests[all([ptbins==ptbin,npvbins==npvbin],axis=0)] = ptestdata
      resestdata = ptestdata/trueptdata
      incl_resests[all([ptbins==ptbin,npvbins==npvbin],axis=0)] = ptestdata/trueptdata

      if absolute:
        all_weightdata = all_weights[all([all_ptbins==ptbin,all_npvbins==npvbin],axis=0)]
        efficiency = sum(weightdata)/sum(all_weightdata)
        if efficiency>1:
          #raise RuntimeError('Efficiency > 1. Check truth jets?')
          efficiency=1
        efficiency_fom = sum(weightdata[ptestdata>20])/sum(all_weightdata) #FoM: efficiency on all truth jets if cutting at 20 GeV on reco
        efficiency_err_fom = sqrt(sum(weightdata[ptestdata>20]**2))/sum(all_weightdata)
      n,bins,patches = plt.hist(resestdata,normed=True,bins=int((max(resestdata)-min(resestdata))*60/2),weights=weightdata,facecolor='b',histtype='stepfilled')
      if options.central == 'absolute_median' or options.central == 'mode' or options.central == 'kde_mode':
        (muR,muR_err,sigmaR,sigmaR_err,upper_quantile,lower_quantile,err) = distribution_values(resestdata,weightdata,options.central,eff=efficiency)
        if options.debug: print 'pT = ' +str(ptedges[ptbin-1])+'-'+str(ptedges[ptbin])+': '+'Mode ' + str(muR) + '; Absolute IQR ' + str(sigmaR)
        if err: print '<< In pT bin '+str(ptbin)+' ('+str(ptedges[ptbin-1])+'-'+str(ptedges[ptbin])+' GeV) >>'
        plt.plot((muR,muR),(0,plt.ylim()[1]),'r--',linewidth=2)
        height = 0.607*max(n) #height at x=1*sigma in normal distribution
        if lower_quantile>float('-inf'):
          plt.plot((lower_quantile,upper_quantile),(height,height),'r--',linewidth=2)
          plt.plot((lower_quantile,lower_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
        else:
          plt.plot((muR,upper_quantile),(height,height),'r--',linewidth=2)
        plt.plot((upper_quantile,upper_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
      if options.central == 'median':
        (muR,muR_err,sigmaR,sigmaR_err,upper_quantile,lower_quantile) = distribution_values(resestdata,weightdata,options.central)
        plt.plot((muR,muR),(0,plt.ylim()[1]),'r--',linewidth=2)
        height = 0.607*max(n) #height at x=1*sigma in normal distribution
        plt.plot((lower_quantile,upper_quantile),(height,height),'r--',linewidth=2)
        plt.plot((lower_quantile,lower_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
        plt.plot((upper_quantile,upper_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
      if options.central == 'mean':
        (muR,muR_err,sigmaR,sigmaR_err) = distribution_values(resestdata,weightdata,options.central)
        gfunc = norm
        y = gfunc.pdf( bins, muR, sigmaR)
        normal = sum(n)/sum(y) 
        y = y*normal #normalize
        plt.plot((muR,muR),(0,gfunc.pdf(muR,muR,sigmaR)*normal),'r--',linewidth=2)
        l = plt.plot(bins, y, 'r--', linewidth=2)
      '''if options.central == 'mode':
        (muR,muR_err,sigmaR,sigmaR_err,kernel) = distribution_values(resestdata,weightdata,options.central)
        y = kernel(bins)
        plt.plot((muR,muR),(0,kernel(muR)),'r--',linewidth=2)
        plt.plot(bins,y,'r--',linewidth=2)'''
      if options.central == 'trimmed':
        (muR,muR_err,sigmaR,sigmaR_err,lower,upper) = distribution_values(resestdata,weightdata,options.central)
        if options.debug: print 'pT = ' +str(ptedges[ptbin-1])+'-'+str(ptedges[ptbin])+': '+'Mode ' + str(muR) + '; Fitted width ' + str(sigmaR)
        #print muR,sigmaR,ptbin
        newbins = bins[all([bins>lower,bins<upper],axis=0)]
        gfunc = norm
        y = gfunc.pdf( bins, muR, sigmaR)
        normal = sum(n)/sum(y) 
        y = y*normal #normalize
        plt.plot((muR,muR),(0,gfunc.pdf(muR,muR,sigmaR)*normal),'r--',linewidth=2)
        newbins = bins[all([bins>lower,bins<upper],axis=0)]
        newy = y[all([bins>lower,bins<upper],axis=0)]
        l = plt.plot(newbins, newy, 'r--', linewidth=2)
      plt.xlabel('$p_T^{reco}/p_T^{true}$')
      plt.ylabel('a.u.')
      plt.xlim(-0.5,3.0)
      plt.ylim(0,1.5)
      plt.savefig(options.plotDir+'/closurebin%d'%ptbin+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.central+'_'+identifier+'.png')
      plt.close()

      n,bins,patches = plt.hist(ptestdata,normed=True,bins=100,weights=weightdata,facecolor='b',histtype='stepfilled')
      if options.central == 'absolute_median' or options.central == 'mode' or options.central == 'kde_mode':
        (mu,mu_err,sigma,sigma_err,upper_quantile,lower_quantile,err) = distribution_values(ptestdata,weightdata,options.central,eff=efficiency)
        if err: print '<< In pT bin '+str(ptbin)+' ('+str(ptedges[ptbin-1])+'-'+str(ptedges[ptbin])+' GeV) >>'
        plt.plot((mu,mu),(0,max(n)),'r--',linewidth=2)
        height = 0.607*max(n) #height at x=1*sigma in normal distribution
        if lower_quantile>float('-inf'):
          plt.plot((lower_quantile,upper_quantile),(height,height),'r--',linewidth=2)
          plt.plot((lower_quantile,lower_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
        else:
          plt.plot((mu,upper_quantile),(height,height),'r--',linewidth=2)
        plt.plot((upper_quantile,upper_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
      if options.central == 'median':
        (mu,mu_err,sigma,sigma_err,upper_quantile,lower_quantile) = distribution_values(ptestdata,weightdata,options.central)
        plt.plot((mu,mu),(0,max(n)),'r--',linewidth=2)
        height = 0.607*max(n) #height at x=1*sigma in normal distribution
        plt.plot((lower_quantile,upper_quantile),(height,height),'r--',linewidth=2)
        plt.plot((lower_quantile,lower_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
        plt.plot((upper_quantile,upper_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
      if options.central == 'mean':
        (mu,mu_err,sigma,sigma_err) = distribution_values(ptestdata,weightdata,options.central)
        gfunc = norm
        y = gfunc.pdf( bins, mu, sigma)
        normal = sum(n)/sum(y) 
        y = y*normal #normalize
        plt.plot((mu,mu),(0,gfunc.pdf(mu,mu,sigma)*normal),'r--',linewidth=2)
        l = plt.plot(bins, y, 'r--', linewidth=2)
      '''if options.central == 'mode':
        (mu,mu_err,sigma,sigma_err,kernel) = distribution_values(ptestdata,weightdata,options.central)
        y = kernel(bins)
        plt.plot((mu,mu),(0,kernel(mu)),'r--',linewidth=2)
        plt.plot(bins,y,'r--',linewidth=2)'''
      if options.central == 'trimmed':
        (mu,mu_err,sigma,sigma_err,lower,upper) = distribution_values(ptestdata,weightdata,options.central)
        #print mu,sigma,ptbin
        newbins = bins[all([bins>lower,bins<upper],axis=0)]
        gfunc = norm
        y = gfunc.pdf( bins, mu, sigma)
        normal = sum(n)/sum(y) 
        y = y*normal #normalize
        plt.plot((mu,mu),(0,gfunc.pdf(mu,mu,sigma)*normal),'r--',linewidth=2)
        newbins = bins[all([bins>lower,bins<upper],axis=0)]
        newy = y[all([bins>lower,bins<upper],axis=0)]
        l = plt.plot(newbins, newy, 'r--', linewidth=2)
      plt.xlabel('$p_T^{reco}$')
      plt.ylabel('a.u.')
      plt.savefig(options.plotDir+'/f1bin%d'%ptbin+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.central+'_'+identifier+'.png')
      plt.close()

      calmuRs.append(muR)
      calmuR_errs.append(muR_err)
      calsigmaRs.append(sigmaR)
      calsigmaR_errs.append(sigmaR_err)
      calmus.append(mu)
      calmu_errs.append(mu_err)
      calsigmas.append(sigma)
      calsigma_errs.append(sigma_err)
      if absolute:
        efficiencies_fom.append(efficiency_fom)
        efficiency_errs_fom.append(efficiency_err_fom)

    #estpts = g1(recopts,*Ropt) #shouldn't take more time because of memoization
    indices = all([npvbins==npvbin,truepts>=options.minpt,truepts<options.maxpt],axis=0)
    plt.plot(truepts[indices],incl_ptests[indices],'.')
    plt.errorbar(avgtruept,calmus,color='g',marker='o',linestyle='',yerr=calmu_errs)
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('$p_T^{reco,cal}$ [GeV]')
    plt.xlim(0,options.maxpt+10)
    plt.ylim(0,options.maxpt+10)
    plt.savefig(options.plotDir+'/jetf1_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.central+'_'+identifier+'.png')
    plt.close()
    
    #closure = estpts/truepts
    plt.plot(truepts[indices],incl_resests[indices],'.')
    plt.errorbar(avgtruept,calmuRs,color='g',marker='o',linestyle='',yerr=calmuR_errs)
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('$p_T^{reco,cal}/p_T^{true}$')
    plt.xlim(0,options.maxpt+10)
    plt.ylim(0,2)
    plt.savefig(options.plotDir+'/jetclosure_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.central+'_'+identifier+'.png')
    plt.close()

    npv_closures[npvedges[npvbin]] = calmuRs
    npv_closure_errs[npvedges[npvbin]] = calmuR_errs
    plt.errorbar(avgtruept,calmuRs,color='g',marker='o',linestyle='',yerr=calmuR_errs)
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('$p_T^{reco,cal}/p_T^{true}$')
    plt.xlim(0,options.maxpt+10)
    plt.ylim(.90,1.1)
    plt.savefig(options.plotDir+'/jetclosure_pttrue_zoom'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.central+'_'+identifier+'.png')
    plt.close()

    if absolute:
      plt.errorbar(avgtruept,efficiencies_fom,color='g',marker='o',linestyle='',yerr=efficiency_errs_fom)
      plt.xlabel('$p_T^{true}$ [GeV]')
      plt.ylabel('Efficiency ($p_T^{reco,cal}>$20 GeV)')
      plt.xlim(0,options.maxpt+10)
      plt.ylim(.50,1.1)
      plt.savefig(options.plotDir+'/jetefficiency_fom_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.central+'_'+identifier+'.png')
      plt.close()
      npv_efficiencies_fom[npvedges[npvbin]] = efficiencies_fom
      npv_efficiencies_err_fom[npvedges[npvbin]] = efficiency_errs_fom


    sigma_calculation = calsigmas
    sigma_err_calculation = calsigma_errs
    npv_sigmas[npvedges[npvbin]] = sigma_calculation
    npv_sigma_errs[npvedges[npvbin]] = sigma_err_calculation
    plt.errorbar(avgtruept,sigma_calculation,yerr=sigma_err_calculation,color='b',linestyle='-',label='NPV '+str(npvedges[npvbin-1])+'-'+str(npvedges[npvbin]))
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('$\sigma[p_T^{reco}]$ [GeV]')
    plt.ylim(min(sigma_calculation)-1,max(sigma_calculation)+1)
    plt.xlim(0,options.maxpt+10)
    plt.legend(loc='upper left',frameon=False,numpoints=1)
    plt.savefig(options.plotDir+'/jetsigma_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.central+'_'+identifier+'.png')
    plt.close()
    
    sigma_calculation = calsigmaRs 
    sigma_err_calculation = calsigmaR_errs
    npv_sigmaRs[npvedges[npvbin]] = sigma_calculation
    npv_sigmaR_errs[npvedges[npvbin]] = sigma_err_calculation
    plt.errorbar(avgtruept,sigma_calculation,yerr=sigma_err_calculation,color='b',linestyle='-',label='NPV '+str(npvedges[npvbin-1])+'-'+str(npvedges[npvbin]))
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('$\sigma[p_T^{reco}/p_T^{true}]$')
    plt.ylim(0,max(sigma_calculation)+0.1) 
    plt.xlim(0,options.maxpt+10)
    plt.legend(loc='upper right',frameon=False,numpoints=1)
    plt.savefig(options.plotDir+'/jetsigmaR_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.central+'_'+identifier+'.png')
    plt.close()

  colors = ['b','r','g','purple','orange','black']
  linestyles = ['-']*6
  labels = ['NPV '+str(npvedges[npvbin-1])+'-'+str(npvedges[npvbin]) for npvbin in xrange(1,len(npvedges))] 
  if len(labels)>6:
    colors = colors*2
    linestyles+=['--']*6
  if len(labels)>12:
    raise RuntimeError('Too many NPV bins. Try increasing the size of the bins.')

  npv_keys = npv_sigmas.keys() 
  npv_keys.sort()
  for i,npv in enumerate(npv_keys):
    plt.errorbar(avgtruept,npv_sigmas[npv],yerr=npv_sigma_errs[npv],color=colors[i],linestyle=linestyles[i],label=labels[i])
  plt.xlabel('$p_T^{true}$ [GeV]')
  plt.ylabel('$\sigma[p_T^{reco}]$ [GeV]')
  lowlim = min([min(s) for s in npv_sigmas.values()])
  highlim = max([max(s) for s in npv_sigmas.values()])
  plt.ylim(lowlim-1,highlim+1)
  plt.xlim(0,options.maxpt+10)
  plt.legend(loc='upper left',frameon=False,numpoints=1)
  plt.savefig(options.plotDir+'/jetsigma_pttrue_'+options.central+'_'+identifier+'.png')
  plt.close()

  for i,npv in enumerate(npv_keys):
    plt.errorbar(avgtruept,npv_sigmaRs[npv],yerr=npv_sigmaR_errs[npv],color=colors[i],linestyle=linestyles[i],label=labels[i])
  plt.xlabel('$p_T^{true}$ [GeV]')
  plt.ylabel('$\sigma[p_T^{reco}/p_T^{true}]$')
  highlim = max([max(s) for s in npv_sigmaRs.values()])
  plt.ylim(0,highlim+0.1)
  plt.xlim(0,options.maxpt+10)
  plt.legend(loc='upper left',frameon=False,numpoints=1)
  plt.savefig(options.plotDir+'/jetsigmaR_pttrue_'+options.central+'_'+identifier+'.png')
  plt.close()

  for i,npv in enumerate(npv_keys):
    plt.errorbar(avgtruept,npv_closures[npv],yerr=npv_closure_errs[npv],color=colors[i],linestyle=linestyles[i],label=labels[i])
  plt.xlabel('$p_T^{true}$ [GeV]')
  plt.ylabel('$p_T^{reco,cal}/p_T^{true}$')
  plt.ylim(0.8,1.2)
  plt.xlim(0,options.maxpt+10)
  plt.legend(loc='upper left',frameon=False,numpoints=1)
  plt.savefig(options.plotDir+'/jetclosure_pttrue_'+options.central+'_'+identifier+'.png')
  plt.close()

  if absolute:
    for i,npv in enumerate(npv_keys):
      plt.errorbar(avgtruept,npv_efficiencies[npv],color=colors[i],linestyle=linestyles[i],label=labels[i],yerr=npv_efficiencies_err[npv])
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('Reconstruction Efficiency')
    lowlim = min([min(e) for e in npv_efficiencies.values()])
    plt.ylim(lowlim-0.1,1.0)
    plt.xlim(0,options.maxpt+10)
    plt.legend(loc='upper left',frameon=False,numpoints=1)
    plt.savefig(options.plotDir+'/jetefficiency_pttrue'+'_'+options.central+'_'+identifier+'.png')
    plt.close()

    for i,npv in enumerate(npv_keys):
      plt.errorbar(avgtruept,npv_efficiencies_fom[npv],color=colors[i],linestyle=linestyles[i],label=labels[i],yerr=npv_efficiencies_err_fom[npv])
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('Reconstruction Efficiency ($p_T^{reco}>20$ GeV)')
    lowlim = min([min(e) for e in npv_efficiencies_fom.values()])
    plt.ylim(lowlim-0.1,1.0)
    plt.xlim(0,options.maxpt+10)
    plt.legend(loc='upper left',frameon=False,numpoints=1)
    plt.savefig(options.plotDir+'/jetefficiency_fom_pttrue'+'_'+options.central+'_'+identifier+'.png')
    plt.close()

  for i,ptbin in enumerate(ptedges):
    if i==0: continue
    plt.errorbar(array(npv_keys)-0.5*options.npvbin,[npv_sigmas[n][i-1] for n in npv_keys],yerr=[npv_sigma_errs[n][i-1] for n in npv_keys],color='b',linestyle='-',label=str(ptedges[i-1])+' GeV $< p_T^{true} < $'+str(ptedges[i])+' GeV')
    plt.xlabel('NPV')
    plt.ylabel('$\sigma[p_T^{reco}]$ [GeV]')
    lowlim = min(npv_sigmas[n][i-1] for n in npv_keys)
    highlim = max(npv_sigmas[n][i-1] for n in npv_keys)
    plt.ylim(lowlim-1,highlim+1)
    plt.xlim(options.minnpv,options.maxnpv)
    plt.legend(loc='upper left',frameon=False,numpoints=1)
    plt.savefig(options.plotDir+'/jetsigma_NPV_pt'+str(ptedges[i-1])+str(ptedges[i])+'_'+options.central+'_'+identifier+'.png')
    plt.close()

  for i,ptbin in enumerate(ptedges):
    if i==0: continue
    plt.errorbar(array(npv_keys)-0.5*options.npvbin,[npv_sigmaRs[n][i-1] for n in npv_keys],yerr=[npv_sigmaR_errs[n][i-1] for n in npv_keys],color='b',linestyle='-',label=str(ptedges[i-1])+' GeV $< p_T^{true} < $'+str(ptedges[i])+' GeV')
    plt.xlabel('NPV')
    plt.ylabel('$\sigma[p_T^{reco}/p_T^{true}]$')
    lowlim = 0 
    highlim = max(npv_sigmaRs[n][i-1] for n in npv_keys)
    plt.ylim(lowlim,highlim+0.1)
    plt.xlim(options.minnpv,options.maxnpv)
    plt.legend(loc='upper left',frameon=False,numpoints=1)
    plt.savefig(options.plotDir+'/jetsigmaR_NPV_pt'+str(ptedges[i-1])+str(ptedges[i])+'_'+options.central+'_'+identifier+'.png')
    plt.close()

  if absolute:
    for i,ptbin in enumerate(ptedges):
      if i==0: continue
      plt.errorbar(array(npv_keys)-0.5*options.npvbin,[npv_efficiencies[n][i-1] for n in npv_keys],yerr=[npv_efficiencies_err[n][i-1] for n in npv_keys],color='b',linestyle='-',label=str(ptedges[i-1])+' GeV $< p_T^{true} < $'+str(ptedges[i])+' GeV')
      plt.xlabel('NPV')
      plt.ylabel('Reconstruction Efficiency')
      lowlim = min(npv_efficiencies[n][i-1] for n in npv_keys)
      plt.ylim(lowlim-0.1,1)
      plt.xlim(options.minnpv,options.maxnpv)
      plt.legend(loc='upper left',frameon=False,numpoints=1)
      plt.savefig(options.plotDir+'/jetefficiency_NPV_pt'+str(ptedges[i-1])+str(ptedges[i])+'_'+options.central+'_'+identifier+'.png')
      plt.close()

    for i,ptbin in enumerate(ptedges):
      if i==0: continue
      plt.errorbar(array(npv_keys)-0.5*options.npvbin,[npv_efficiencies_fom[n][i-1] for n in npv_keys],yerr=[npv_efficiencies_err_fom[n][i-1] for n in npv_keys],color='b',linestyle='-',label=str(ptedges[i-1])+' GeV $< p_T^{true} < $'+str(ptedges[i])+' GeV')
      plt.xlabel('NPV')
      plt.ylabel('Reconstruction Efficiency ($p_T^{reco}>20$ GeV)')
      lowlim = min(npv_efficiencies_fom[n][i-1] for n in npv_keys)
      plt.ylim(lowlim-0.1,1)
      plt.xlim(options.minnpv,options.maxnpv)
      plt.legend(loc='upper left',frameon=False,numpoints=1)
      plt.savefig(options.plotDir+'/jetefficiency_fom_NPV_pt'+str(ptedges[i-1])+str(ptedges[i])+'_'+options.central+'_'+identifier+'.png')
      plt.close()

  if absolute:
    pickle.dump(npv_efficiencies,open(options.submitDir+'/efficiencies_'+options.central+'_'+identifier+'.p','wb'))
    pickle.dump(npv_efficiencies_err,open(options.submitDir+'/efficiency_errs_'+options.central+'_'+identifier+'.p','wb'))
    pickle.dump(npv_efficiencies_fom,open(options.submitDir+'/efficiencies_fom_'+options.central+'_'+identifier+'.p','wb'))
    pickle.dump(npv_efficiencies_err_fom,open(options.submitDir+'/efficiency_errs_fom_'+options.central+'_'+identifier+'.p','wb'))

  #################
  ### inclusive ###
  #################
  incl_sigmas = []
  incl_sigma_errs = []
  incl_sigmaRs = []
  incl_sigmaR_errs = []
  incl_calmus = []
  incl_calmu_errs = []
  incl_calmuRs = []
  incl_calmuR_errs = []
  if absolute:
    incl_efficiencies = []
    incl_efficiencies_err = []
    incl_efficiencies_fom = []
    incl_efficiencies_err_fom = []
  for ptbin in xrange(1,len(ptedges)): 
    ptestdata = incl_ptests[all([ptbins==ptbin,npvs>=options.minnpv,npvs<options.maxnpv],axis=0)]
    trueptdata = truepts[all([ptbins==ptbin,npvs>=options.minnpv,npvs<options.maxnpv],axis=0)]
    weightdata = weights[all([ptbins==ptbin,npvs>=options.minnpv,npvs<options.maxnpv],axis=0)]
    resestdata = ptestdata/trueptdata

    if absolute:
      all_weightdata = all_weights[all([all_ptbins==ptbin,all_npvs>=options.minnpv,all_npvs<options.maxnpv],axis=0)]
      efficiency = sum(weightdata)/sum(all_weightdata)
      if efficiency>1:
        #raise RuntimeError('Efficiency > 1. Check truth jets?')
        efficiency=1
      efficiency_fom = sum(weightdata[ptestdata>20])/sum(all_weightdata) #FoM: efficiency on all truth jets if cutting at 20 GeV on reco
      efficiency_err_fom = sqrt(sum(weightdata[ptestdata>20]**2))/sum(all_weightdata)
    n,bins,patches = plt.hist(resestdata,normed=True,bins=100,weights=weightdata,facecolor='b',histtype='stepfilled')
    if options.central == 'absolute_median' or options.central == 'mode' or options.central == 'kde_mode':
      (muR,muR_err,sigmaR,sigmaR_err,upper_quantile,lower_quantile,err) = distribution_values(resestdata,weightdata,options.central,eff=efficiency)
      if err: print '<< In pT bin '+str(ptbin)+' ('+str(ptedges[ptbin-1])+'-'+str(ptedges[ptbin])+' GeV) >>'
      plt.plot((muR,muR),(0,plt.ylim()[1]),'r--',linewidth=2)
      height = 0.607*max(n) #height at x=1*sigma in normal distribution
      if lower_quantile>float('-inf'):
        plt.plot((lower_quantile,upper_quantile),(height,height),'r--',linewidth=2)
        plt.plot((lower_quantile,lower_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
      else:
        plt.plot((muR,upper_quantile),(height,height),'r--',linewidth=2)
      plt.plot((upper_quantile,upper_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
    if options.central == 'median':
      (muR,muR_err,sigmaR,sigmaR_err,upper_quantile,lower_quantile) = distribution_values(resestdata,weightdata,options.central)
      plt.plot((muR,muR),(0,plt.ylim()[1]),'r--',linewidth=2)
      height = 0.607*max(n) #height at x=1*sigma in normal distribution
      plt.plot((lower_quantile,upper_quantile),(height,height),'r--',linewidth=2)
      plt.plot((lower_quantile,lower_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
      plt.plot((upper_quantile,upper_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
    if options.central == 'mean':
      (muR,muR_err,sigmaR,sigmaR_err) = distribution_values(resestdata,weightdata,options.central)
      gfunc = norm
      y = gfunc.pdf( bins, muR, sigmaR)
      normal = sum(n)/sum(y) 
      y = y*normal #normalize
      plt.plot((muR,muR),(0,gfunc.pdf(muR,muR,sigmaR)*normal),'r--',linewidth=2)
      l = plt.plot(bins, y, 'r--', linewidth=2)
    '''if options.central == 'mode':
      (muR,muR_err,sigmaR,sigmaR_err,kernel) = distribution_values(resestdata,weightdata,options.central)
      y = kernel(bins)
      plt.plot((muR,muR),(0,kernel(muR)),'r--',linewidth=2)
      plt.plot(bins,y,'r--',linewidth=2)'''
    if options.central == 'trimmed':
      (muR,muR_err,sigmaR,sigmaR_err,lower,upper) = distribution_values(resestdata,weightdata,options.central)
      #print muR,sigmaR,ptbin
      newbins = bins[all([bins>lower,bins<upper],axis=0)]
      gfunc = norm
      y = gfunc.pdf( bins, muR, sigmaR)
      normal = sum(n)/sum(y) 
      y = y*normal #normalize
      plt.plot((muR,muR),(0,gfunc.pdf(muR,muR,sigmaR)*normal),'r--',linewidth=2)
      newbins = bins[all([bins>lower,bins<upper],axis=0)]
      newy = y[all([bins>lower,bins<upper],axis=0)]
      l = plt.plot(newbins, newy, 'r--', linewidth=2)
    plt.xlabel('$p_T^{reco}/p_T^{true}$')
    plt.ylabel('a.u.')
    plt.savefig(options.plotDir+'/closurebin%d'%ptbin+'_NPVincl'+'_'+options.central+'_'+identifier+'.png')
    plt.close()

    n,bins,patches = plt.hist(ptestdata,normed=True,bins=100,weights=weightdata,facecolor='b',histtype='stepfilled')
    if options.central == 'absolute_median' or options.central == 'mode' or options.central == 'kde_mode':
      (mu,mu_err,sigma,sigma_err,upper_quantile,lower_quantile,err) = distribution_values(ptestdata,weightdata,options.central,eff=efficiency)
      if err: print '<< In pT bin '+str(ptbin)+' ('+str(ptedges[ptbin-1])+'-'+str(ptedges[ptbin])+' GeV) >>'
      plt.plot((mu,mu),(0,max(n)),'r--',linewidth=2)
      height = 0.607*max(n) #height at x=1*sigma in normal distribution
      if lower_quantile>float('-inf'):
        plt.plot((lower_quantile,upper_quantile),(height,height),'r--',linewidth=2)
        plt.plot((lower_quantile,lower_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
      else:
        plt.plot((mu,upper_quantile),(height,height),'r--',linewidth=2)
      plt.plot((upper_quantile,upper_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
    if options.central == 'median':
      (mu,mu_err,sigma,sigma_err,upper_quantile,lower_quantile) = distribution_values(ptestdata,weightdata,options.central)
      plt.plot((mu,mu),(0,max(n)),'r--',linewidth=2)
      height = 0.607*max(n) #height at x=1*sigma in normal distribution
      plt.plot((lower_quantile,upper_quantile),(height,height),'r--',linewidth=2)
      plt.plot((lower_quantile,lower_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
      plt.plot((upper_quantile,upper_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
    if options.central == 'mean':
      (mu,mu_err,sigma,sigma_err) = distribution_values(ptestdata,weightdata,options.central)
      gfunc = norm
      y = gfunc.pdf( bins, mu, sigma)
      normal = sum(n)/sum(y) 
      y = y*normal #normalize
      plt.plot((mu,mu),(0,gfunc.pdf(mu,mu,sigma)*normal),'r--',linewidth=2)
      l = plt.plot(bins, y, 'r--', linewidth=2)
    '''if options.central == 'mode':
      (mu,mu_err,sigma,sigma_err,kernel) = distribution_values(ptestdata,weightdata,options.central)
      y = kernel(bins)
      plt.plot((mu,mu),(0,kernel(mu)),'r--',linewidth=2)
      plt.plot(bins,y,'r--',linewidth=2)'''
    if options.central == 'trimmed':
      (mu,mu_err,sigma,sigma_err,lower,upper) = distribution_values(ptestdata,weightdata,options.central)
      #print mu,sigma,ptbin
      newbins = bins[all([bins>lower,bins<upper],axis=0)]
      gfunc = norm
      y = gfunc.pdf( bins, mu, sigma)
      normal = sum(n)/sum(y) 
      y = y*normal #normalize
      plt.plot((mu,mu),(0,gfunc.pdf(mu,mu,sigma)*normal),'r--',linewidth=2)
      newbins = bins[all([bins>lower,bins<upper],axis=0)]
      newy = y[all([bins>lower,bins<upper],axis=0)]
      l = plt.plot(newbins, newy, 'r--', linewidth=2)
    plt.xlabel('$p_T^{reco}$')
    plt.ylabel('a.u.')
    plt.savefig(options.plotDir+'/f1bin%d'%ptbin+'_NPVincl'+'_'+options.central+'_'+identifier+'.png')
    plt.close()

    incl_calmus.append(mu)
    incl_calmu_errs.append(mu_err)
    incl_calmuRs.append(muR)
    incl_calmuR_errs.append(muR_err)
    incl_sigmas.append(sigma)
    incl_sigma_errs.append(sigma_err)
    incl_sigmaRs.append(sigmaR)
    incl_sigmaR_errs.append(sigmaR_err)
    if absolute:
      incl_efficiencies.append(efficiency)
      incl_efficiencies_err.append(efficiency_err)
      incl_efficiencies_fom.append(efficiency_fom)
      incl_efficiencies_err_fom.append(efficiency_err_fom)

  f = r.TFile(options.submitDir+'/'+options.central+'_'+identifier+'.root','recreate')
  
  indices = all([npvs>=options.minnpv,npvs<options.maxnpv,truepts>=options.minpt,truepts<options.maxpt],axis=0)
  plt.plot(truepts[indices],incl_ptests[indices],'.')
  plt.errorbar(avgtruept,incl_calmus,color='g',marker='o',linestyle='',yerr=incl_calmu_errs)
  plt.xlabel('$p_T^{true}$ [GeV]')
  plt.ylabel('$p_T^{reco,cal}$ [GeV]')
  plt.xlim(0,options.maxpt+10)
  plt.ylim(0,options.maxpt+10)
  plt.savefig(options.plotDir+'/jetf1_pttrue'+'_NPVincl'+'_'+options.central+'_'+identifier+'.png')
  plt.close()
  
  #closure = incl_ptests/truepts
  plt.plot(truepts[indices],incl_resests[indices],'.')
  plt.errorbar(avgtruept,incl_calmuRs,color='g',marker='o',linestyle='',yerr=incl_calmuR_errs)
  plt.xlabel('$p_T^{true}$ [GeV]')
  plt.ylabel('$p_T^{reco,cal}/p_T^{true}$')
  plt.xlim(0,options.maxpt+10)
  plt.ylim(0,2)
  plt.savefig(options.plotDir+'/jetclosure_pttrue'+'_NPVincl'+'_'+options.central+'_'+identifier+'.png')
  plt.close()

  plt.errorbar(avgtruept,incl_calmuRs,color='g',marker='o',linestyle='',yerr=incl_calmuR_errs)
  plt.xlabel('$p_T^{true}$ [GeV]')
  plt.ylabel('$p_T^{reco,cal}/p_T^{true}$')
  plt.xlim(0,options.maxpt+10)
  plt.ylim(.90,1.1)
  plt.savefig(options.plotDir+'/jetclosure_pttrue_zoom'+'_NPVincl'+'_'+options.central+'_'+identifier+'.png')
  plt.close()
  #root
  t = r.TGraphErrors(len(avgtruept),array(avgtruept),array(incl_calmuRs),zeros(len(avgtruept)),array(incl_calmuR_errs))
  t.SetName('jetclosure_pttrue_NPVincl')
  t.Write()

  if absolute:
    plt.errorbar(avgtruept,incl_efficiencies,color='g',marker='o',linestyle='',yerr=incl_efficiencies_err)
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('Efficiency')
    plt.xlim(0,options.maxpt+10)
    plt.ylim(.90,1.1)
    plt.savefig(options.plotDir+'/jetefficiency_pttrue'+'_NPVincl'+'_'+options.central+'_'+identifier+'.png')
    plt.close()

    plt.errorbar(avgtruept,incl_efficiencies_fom,color='g',marker='o',linestyle='',yerr=incl_efficiencies_err_fom)
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('Efficiency ($p_T^{reco,cal}>$20 GeV)')
    plt.xlim(0,options.maxpt+10)
    plt.ylim(.50,1.1)
    plt.savefig(options.plotDir+'/jetefficiency_fom_pttrue'+'_NPVincl'+'_'+options.central+'_'+identifier+'.png')
    plt.close()
    #root
    t = r.TGraphErrors(len(avgtruept),array(avgtruept),array(incl_efficiencies_fom),zeros(len(avgtruept)),array(incl_efficiencies_err_fom))
    t.SetName('jetefficiency_pttrue_NPVincl')
    t.Write()

  sigma_calculation = incl_sigmas
  sigma_err_calculation = incl_sigma_errs
  plt.errorbar(avgtruept,sigma_calculation,yerr=sigma_err_calculation,color='b',linestyle='-',label='NPV Incl.')
  plt.xlabel('$p_T^{true}$ [GeV]')
  plt.ylabel('$\sigma[p_T^{reco}]$ [GeV]')
  plt.ylim(min(sigma_calculation)-1,max(sigma_calculation)+1)
  plt.xlim(0,options.maxpt+10)
  plt.legend(loc='upper left',frameon=False,numpoints=1)
  plt.savefig(options.plotDir+'/jetsigma_pttrue'+'_NPVincl'+'_'+options.central+'_'+identifier+'.png')
  plt.close()
  #root
  t = r.TGraphErrors(len(avgtruept),array(avgtruept),array(sigma_calculation),zeros(len(avgtruept)),array(sigma_err_calculation))
  t.SetName('jetsigma_pttrue_NPVincl')
  t.Write()
  
  sigma_calculation = incl_sigmaRs 
  sigma_err_calculation = incl_sigmaR_errs
  plt.errorbar(avgtruept,sigma_calculation,yerr=sigma_err_calculation,color='b',linestyle='-',label='NPV Incl.')
  plt.xlabel('$p_T^{true}$ [GeV]')
  plt.ylabel('$\sigma[p_T^{reco}/p_T^{true}]$')
  plt.ylim(0,max(sigma_calculation)+0.1) 
  plt.xlim(0,options.maxpt+10)
  plt.legend(loc='upper right',frameon=False,numpoints=1)
  plt.savefig(options.plotDir+'/jetsigmaR_pttrue'+'_NPVincl'+'_'+options.central+'_'+identifier+'.png')
  plt.close()
  #root
  t = r.TGraphErrors(len(avgtruept),array(avgtruept),array(sigma_calculation),zeros(len(avgtruept)),array(sigma_err_calculation))
  t.SetName('jetsigmaR_pttrue_NPVincl')
  t.Write()

  #fake jets
  if doFake:
    fakejetmults = {npv: [] for npv in npvedges[1:]} 
    fakeweights = {npv: [] for npv in npvedges[1:]} 
    fakejetmults_avg = {npv: 0 for npv in npvedges[1:]} 
    fakejetmults_err = {npv: 0 for npv in npvedges[1:]} 
    try:
      gfakejetpts = {npv: g(options.fakept,*Ropts[npv]) for npv in npvedges[1:]} 
      for npv in range(options.minnpv,options.maxnpv):
        npvedge = min([npve for npve in npvedges[1:] if npve>=npv])
        gfakejetpt = gfakejetpts[npvedge]
        for ept,eeta,eweight in zip(PU_recopts[npv],PU_etas[npv],PU_weights[npv]):
          if not len(ept)==len(eeta):
            doFake=False
            raise RuntimeError('== Different length of reco pTs and reco etas for fake jets. Not storing fake jet information. ==')
          ept = array(ept)
          eeta = array(eeta)
          fakejetmult = len(ept[all([ept>gfakejetpt,abs(eeta)<options.maxeta,abs(eeta)>options.mineta],axis=0)])
          fakejetmults[npvedge].append(fakejetmult)
          fakeweights[npvedge].append(eweight)
      for npvbin,npvedge in enumerate(npvedges):
        if npvbin==0: continue
        data = array(fakejetmults[npvedge])
        weights = array(fakeweights[npvedge])
        n,bins,patches = plt.hist(data,normed=True,bins=max(data),weights=weights,facecolor='b',histtype='stepfilled',label='NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin]))
        plt.xlabel('Fake Jet Multiplicity $(p_T>20$ GeV)')
        plt.ylabel('a.u.')
        plt.legend(loc='upper right',frameon=False,numpoints=1)
        plt.savefig(options.plotDir+'/fakejets'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.central+'_'+identifier+'.png')
        plt.close()
        (mu,mu_err,sigma,sigma_err) = distribution_values(data,weights,'mean')
        fakejetmults_avg[npvedge] = mu
        fakejetmults_err[npvedge] = mu_err
      x = array(npvedges[1:])-0.5*options.npvbin
      y = [fakejetmults_avg[npvedge] for npvedge in npvedges[1:]]
      yerr = [fakejetmults_err[npvedge] for npvedge in npvedges[1:]]
      plt.errorbar(x,y,yerr=yerr,color='b',linestyle='-')
      plt.xlabel('NPV')
      plt.ylabel('Fake Jet Multiplicity $(p_T>20$ GeV)')
      plt.xlim(npvedges[0],npvedges[len(npvedges)-1])
      plt.ylim(0,max([fakejetmults_avg[npvedge] for npvedge in npvedges[1:]])+1) 
      #plt.legend(loc='upper right',frameon=False,numpoints=1)
      plt.savefig(options.plotDir+'/fakejets_NPV'+'_'+options.central+'_'+identifier+'.png')
      plt.close()
      #root
      t = r.TGraphErrors(len(x),array(x),array(y),zeros(len(x)),array(yerr))
      t.SetName('fakejets_NPV')
      t.Write()

      pickle.dump(fakejetmults_avg,open(options.submitDir+'/fakejetmults_avg_'+options.central+'_'+identifier+'.p','wb'))
      pickle.dump(fakejetmults_err,open(options.submitDir+'/fakejetmults_err_'+options.central+'_'+identifier+'.p','wb'))
    except Exception,e: print 'Error in fake jet calculation: '+str(e) 

  f.Close()

  pickle.dump(Ropts,open(options.submitDir+'/fit_'+options.central+'_'+identifier+'.p','wb'))
  pickle.dump(npv_sigmas,open(options.submitDir+'/sigmas_'+options.central+'_'+identifier+'.p','wb'))
  pickle.dump(npv_sigma_errs,open(options.submitDir+'/sigma_errs_'+options.central+'_'+identifier+'.p','wb'))
  pickle.dump(npv_sigmaRs,open(options.submitDir+'/sigmaRs_'+options.central+'_'+identifier+'.p','wb'))
  pickle.dump(npv_sigmaR_errs,open(options.submitDir+'/sigmaR_errs_'+options.central+'_'+identifier+'.p','wb'))
  pickle.dump(avgtruept,open(options.submitDir+'/avgpttrue_'+options.central+'_'+identifier+'.p','wb'))
  pickle.dump(ptedges,open(options.submitDir+'/ptedges_'+options.central+'_'+identifier+'.p','wb'))

  if absolute:
    pickle.dump(incl_efficiencies,open(options.submitDir+'/incl_efficiencies_'+options.central+'_'+identifier+'.p','wb'))
    pickle.dump(incl_efficiencies_err,open(options.submitDir+'/incl_efficiency_errs_'+options.central+'_'+identifier+'.p','wb'))
    pickle.dump(incl_efficiencies_fom,open(options.submitDir+'/incl_efficiencies_fom_'+options.central+'_'+identifier+'.p','wb'))
    pickle.dump(incl_efficiencies_err_fom,open(options.submitDir+'/incl_efficiency_errs_fom_'+options.central+'_'+identifier+'.p','wb'))

  pickle.dump(incl_calmus,open(options.submitDir+'/incl_calmus_'+options.central+'_'+identifier+'.p','wb'))
  pickle.dump(incl_calmu_errs,open(options.submitDir+'/incl_calmu_errs_'+options.central+'_'+identifier+'.p','wb'))
  pickle.dump(incl_calmuRs,open(options.submitDir+'/incl_calmuRs_'+options.central+'_'+identifier+'.p','wb'))
  pickle.dump(incl_calmuR_errs,open(options.submitDir+'/incl_calmuR_errs_'+options.central+'_'+identifier+'.p','wb'))

  pickle.dump(incl_sigmas,open(options.submitDir+'/incl_sigmas_'+options.central+'_'+identifier+'.p','wb'))
  pickle.dump(incl_sigma_errs,open(options.submitDir+'/incl_sigma_errs_'+options.central+'_'+identifier+'.p','wb'))
  pickle.dump(incl_sigmaRs,open(options.submitDir+'/incl_sigmaRs_'+options.central+'_'+identifier+'.p','wb'))
  pickle.dump(incl_sigmaR_errs,open(options.submitDir+'/incl_sigmaR_errs_'+options.central+'_'+identifier+'.p','wb'))


  return 

import pickle
fitres()
