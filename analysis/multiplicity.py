from numpy import load,log,linspace,digitize,array,mean,std,exp,all,average,sqrt,asarray,sign
import os
import numpy
from numpy import save
from scipy.optimize import curve_fit,fsolve
from scipy.stats import norm
from operator import sub
from optparse import OptionParser
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
parser.add_option("-c","--cut", default=float('-inf'), type=float, help="low pT cut on reco jets")
parser.add_option("--mineta", help="min abs(eta) on reco jets", type=float, default=0)
parser.add_option("--maxeta", help="max abs(eta) on reco jets", type=float, default=float('inf'))

# analysis configuration
parser.add_option("--minnpv", help="min NPV", type=int, default=5)
parser.add_option("--maxnpv", help="max NPV", type=int, default=30)
parser.add_option("--npvbin", help="size of NPV bins", type=int, default=5)
parser.add_option("--minpt", help="min truth pt", type=int, default=20)
parser.add_option("--maxpt", help="max truth pt", type=int, default=80)
parser.add_option("--ptbin", help="size of pT bins", type=int, default=2)

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

  
  import pickle
  fits = pickle.load(open(options.submitDir+'/'+'fit_'+options.identifier+'.p','rb'))
  npvedges = fits.keys()
  npvedges.sort()
  npvbinsize = npvedges[1]-npvedges[0]
  npvedges.insert(0,npvedges[0]-npvbinsize)
  npvbins = digitize(npvs,npvedges)

  for npvbin in xrange(1,len(npvedges)):
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
      event_multiplicity = len(event_calib_jets[event_calib_jets>20])
      event_multiplicities.append(event_multiplicity)
      event_weights.append(event_weight)
    pdb.set_trace()
      



def plot():
  maxpt = options.maxpt
  if (options.maxpt-options.minpt)%options.ptbin==0: maxpt+=1
  ptedges = range(options.minpt,maxpt,options.ptbin)
  cuts = all([truepts>min(ptedges),recopts>options.cut,eta_cuts,mindr_cuts],axis=0)

  recopts = recopts[cuts]
  truepts = truepts[cuts]
  responses = recopts/truepts
  npvs = npvs[cuts]
  weights = weights[cuts]

  ptbins = digitize(truepts,ptedges)

  maxnpv = options.maxnpv
  if (options.maxnpv-options.minnpv)%options.npvbin==0: maxnpv+=1
  npvedges = range(options.minnpv,maxnpv,options.npvbin)
  npvbins = digitize(npvs,npvedges)

  npv_sigmas = {npvedges[npvbin]: [] for npvbin in xrange(1,len(npvedges))}
  npv_sigma_errs = {npvedges[npvbin]: [] for npvbin in xrange(1,len(npvedges))}
  npv_sigmaRs = {npvedges[npvbin]: [] for npvbin in xrange(1,len(npvedges))}
  npv_sigmaR_errs = {npvedges[npvbin]: [] for npvbin in xrange(1,len(npvedges))}
  Ropts = {npvedges[npvbin]: [] for npvbin in xrange(1,len(npvedges))}

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

    for ptbin in xrange(1,len(ptedges)): 
      #print '>> >> Processing pT bin '+str(ptedges[ptbin-1])+'-'+str(ptedges[ptbin])+' GeV'
      resdata = responses[all([ptbins==ptbin,npvbins==npvbin],axis=0)]
      ptdata = recopts[all([ptbins==ptbin,npvbins==npvbin],axis=0)]
      trueptdata = truepts[all([ptbins==ptbin,npvbins==npvbin],axis=0)]
      weightdata = weights[all([ptbins==ptbin,npvbins==npvbin],axis=0)]
      weightdata = weightdata/sum(weightdata)
      avgtruept.append(average(trueptdata,weights=weightdata))
      if len(resdata)<100: print 'Low statistics ('+str(len(resdata))+' jets) in bin with pT = ' +str(ptedges[ptbin])+' and NPV between '+str(npvedges[npvbin-1])+' and '+str(npvedges[npvbin])
      # maximum likelihood estimates
      mu = average(resdata,weights=weightdata)
      var = average((resdata-mu)**2,weights=weightdata)
      sigma = sqrt(var)
      mu_err = sigma*sqrt(sum(weightdata**2))
      var_err = var*sqrt(2*sum(weightdata**2)) # from https://web.eecs.umich.edu/~fessler/papers/files/tr/stderr.pdf
      #var = sigma^2 -> var_err/var = 2*sigma_err/sigma
      sigma_err = 0.5*var_err/sigma
      n,bins,patches = plt.hist(resdata,normed=True,bins=50,weights=weightdata,facecolor='b')
      gfunc = norm
      y = gfunc.pdf( bins, mu, sigma)
      l = plt.plot(bins, y, 'r--', linewidth=2)
      plt.xlabel('$p_T^{reco}/p_T^{true}$')
      plt.ylabel('a.u.')
      plt.savefig(options.plotDir+'/resbin%d'%ptbin+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.identifier+'.png')
      plt.close()
      avgres.append(mu)
      avgres_errs.append(mu_err)
      sigmaRs.append(sigma)
      sigmaR_errs.append(sigma_err)

      n,bins,patches = plt.hist(ptdata,normed=True,bins=50,weights=weightdata)
      # maximum likelihood estimates
      mu = average(ptdata,weights=weightdata)
      var = average((ptdata-mu)**2,weights=weightdata)
      sigma = sqrt(var)
      mu_err = sigma*sqrt(sum(weightdata**2))
      var_err = var*sqrt(2*sum(weightdata**2)) # from https://web.eecs.umich.edu/~fessler/papers/files/tr/stderr.pdf
      #var = sigma^2 -> var_err/var = 2*sigma_err/sigma
      sigma_err = 0.5*var_err/sigma
      n,bins,patches = plt.hist(ptdata,normed=True,bins=50,weights=weightdata,facecolor='b')
      gfunc = norm
      y = gfunc.pdf( bins, mu, sigma)
      l = plt.plot(bins, y, 'r--', linewidth=2)
      plt.xlabel('$p_T^{reco}$')
      plt.ylabel('a.u.')
      plt.savefig(options.plotDir+'/fbin%d'%ptbin+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.identifier+'.png')
      plt.close()
      avgpt.append(mu)
      sigmas.append(sigma)
      avgpt_errs.append(mu_err)
      sigma_errs.append(sigma_err)

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
    plt.savefig(options.plotDir+'/jetresponse_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.identifier+'.png')
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
    plt.savefig(options.plotDir+'/jetf_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.identifier+'.png')
    plt.close()

    #dg = d(R*t):
    plt.plot(xp,dg(xp,*Ropt),'r-')
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('$f\'(p_T^{true})$')
    plt.ylim(0,1)
    plt.xlim(0,options.maxpt+10)
    plt.savefig(options.plotDir+'/jetdf_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.identifier+'.png')
    plt.close()

    if options.doCal:
      calmuRs = []
      calmuR_errs = []
      calsigmaRs = []
      calsigmaR_errs = []
      calmus = []
      calmu_errs = []
      calsigmas = []
      calsigma_errs = []
      for ptbin in xrange(1,len(ptedges)): 
        ptdata = recopts[all([ptbins==ptbin,npvbins==npvbin],axis=0)]
        trueptdata = truepts[all([ptbins==ptbin,npvbins==npvbin],axis=0)]
        weightdata = weights[all([ptbins==ptbin,npvbins==npvbin],axis=0)]
        ptestdata = g1(ptdata,*Ropt)
        muR,muR_err,sigmaR,sigmaR_err,mu,mu_err,sigma,sigma_err = numerical_inversion(ptestdata,trueptdata,weightdata,ptbin,npvedges,npvbin)
        calmuRs.append(muR)
        calmuR_errs.append(muR_err)
        calsigmaRs.append(sigmaR)
        calsigmaR_errs.append(sigmaR_err)
        calmus.append(mu)
        calmu_errs.append(mu_err)
        calsigmas.append(sigma)
        calsigma_errs.append(sigma_err)

      estpts = g1(recopts,*Ropt) #shouldn't take more time because of memoization
      plt.plot(truepts[npvbins==npvbin],estpts[npvbins==npvbin],'.')
      plt.errorbar(avgtruept,calmus,color='g',marker='o',linestyle='',yerr=calmu_errs)
      plt.xlabel('$p_T^{true}$ [GeV]')
      plt.ylabel('$p_T^{reco,cal}$ [GeV]')
      plt.xlim(0,options.maxpt+10)
      plt.ylim(0,options.maxpt+10)
      plt.savefig(options.plotDir+'/jetf1_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.identifier+'.png')
      plt.close()
      
      closure = estpts/truepts
      plt.plot(truepts[npvbins==npvbin],closure[npvbins==npvbin],'.')
      plt.errorbar(avgtruept,calmuRs,color='g',marker='o',linestyle='',yerr=calmuR_errs)
      plt.xlabel('$p_T^{true}$ [GeV]')
      plt.ylabel('$p_T^{reco,cal}/p_T^{true}$')
      plt.xlim(0,options.maxpt+10)
      plt.ylim(0,2)
      plt.savefig(options.plotDir+'/jetclosure_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.identifier+'.png')
      plt.close()

      plt.errorbar(avgtruept,calmuRs,color='g',marker='o',linestyle='',yerr=calmuR_errs)
      plt.xlabel('$p_T^{true}$ [GeV]')
      plt.ylabel('$p_T^{reco,cal}/p_T^{true}$')
      plt.xlim(0,options.maxpt+10)
      plt.ylim(.95,1.05)
      plt.savefig(options.plotDir+'/jetclosure_pttrue_zoom'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.identifier+'.png')
      plt.close()


    if options.doCal:
      sigma_calculation = calsigmas
      sigma_err_calculation = calsigma_errs
    else:
      sigma_calculation = array(sigmas)/dg(avgtruept,*Ropt)
      sigma_err_calculation = array(sigma_errs)/dg(avgtruept,*Ropt)
    npv_sigmas[npvedges[npvbin]] = sigma_calculation
    npv_sigma_errs[npvedges[npvbin]] = sigma_err_calculation
    plt.errorbar(avgtruept,sigma_calculation,yerr=sigma_err_calculation,color='b',linestyle='-',label='NPV '+str(npvedges[npvbin-1])+'-'+str(npvedges[npvbin]))
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('$\sigma[p_T^{reco}]$ [GeV]')
    plt.ylim(min(sigma_calculation)-1,max(sigma_calculation)+1)
    plt.xlim(0,options.maxpt+10)
    plt.legend(loc='upper left',frameon=False,numpoints=1)
    plt.savefig(options.plotDir+'/jetsigma_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.identifier+'.png')
    plt.close()
    
    if options.doCal:
      sigma_calculation = calsigmaRs 
      sigma_err_calculation = calsigmaR_errs
    else:
      sigma_calculation = array(sigmaRs)/dg(avgtruept,*Ropt)
      sigma_err_calculation = array(sigmaR_errs)/dg(avgtruept,*Ropt) 
    npv_sigmaRs[npvedges[npvbin]] = sigma_calculation
    npv_sigmaR_errs[npvedges[npvbin]] = sigma_err_calculation
    plt.errorbar(avgtruept,sigma_calculation,yerr=sigma_err_calculation,color='b',linestyle='-',label='NPV '+str(npvedges[npvbin-1])+'-'+str(npvedges[npvbin]))
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('$\sigma[p_T^{reco}/p_T^{true}]$')
    plt.ylim(0,max(sigma_calculation)+0.1) 
    plt.xlim(0,options.maxpt+10)
    plt.legend(loc='upper right',frameon=False,numpoints=1)
    plt.savefig(options.plotDir+'/jetsigmaR_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.identifier+'.png')
    plt.close()

  colors = ['b','r','g','purple','orange','black']
  linestyles = ['-']*6
  labels = ['NPV '+str(npvedges[npvbin-1])+'-'+str(npvedges[npvbin]) for npvbin in xrange(1,len(npvedges))] 
  if len(labels)>6:
    colors = colors*2
    linestyles+=['--']*6
  if len(labels)>12:
    raise RuntimeError('NPV bins are too small. Make them bigger.')

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
  plt.savefig(options.plotDir+'/jetsigma_pttrue_'+options.identifier+'.png')
  plt.close()

  for i,npv in enumerate(npv_keys):
    plt.errorbar(avgtruept,npv_sigmaRs[npv],yerr=npv_sigmaR_errs[npv],color=colors[i],linestyle=linestyles[i],label=labels[i])
  plt.xlabel('$p_T^{true}$ [GeV]')
  plt.ylabel('$\sigma[p_T^{reco}/p_T^{true}]$')
  highlim = max([max(s) for s in npv_sigmaRs.values()])
  plt.ylim(0,highlim+0.1)
  plt.xlim(0,options.maxpt+10)
  plt.legend(loc='upper left',frameon=False,numpoints=1)
  plt.savefig(options.plotDir+'/jetsigmaR_pttrue_'+options.identifier+'.png')
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
    plt.savefig(options.plotDir+'/jetsigma_NPV_pt'+str(ptedges[i-1])+str(ptedges[i])+'_'+options.identifier+'.png')
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
    plt.savefig(options.plotDir+'/jetsigmaR_NPV_pt'+str(ptedges[i-1])+str(ptedges[i])+'_'+options.identifier+'.png')
    plt.close()



  return Ropts,npv_sigmas,npv_sigma_errs,npv_sigmaRs,npv_sigmaR_errs,avgtruept,ptedges

def numerical_inversion(ptestdata,trueptdata,weightdata,ptbin,npvedges,npvbin):
  weightdata = weightdata/sum(weightdata)
  resdata = ptestdata/trueptdata
  muR = average(resdata,weights=weightdata)
  varR = average((resdata-muR)**2,weights=weightdata)
  sigmaR = sqrt(varR)
  muR_err = sigmaR*sqrt(sum(weightdata**2))
  varR_err = varR*sqrt(2*sum(weightdata**2)) # from https://web.eecs.umich.edu/~fessler/papers/files/tr/stderr.pdf
  #var = sigma^2 -> var_err/var = 2*sigma_err/sigma
  sigmaR_err = 0.5*varR_err/sigmaR
  n,bins,patches = plt.hist(resdata,normed=True,bins=50,weights=weightdata,facecolor='b')
  gfunc = norm
  y = gfunc.pdf( bins, muR, sigmaR)
  l = plt.plot(bins, y, 'r--', linewidth=2)
  plt.xlabel('$p_T^{reco}/p_T^{true}$')
  plt.ylabel('a.u.')
  plt.savefig(options.plotDir+'/closurebin%d'%ptbin+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.identifier+'.png')
  plt.close()
  #avgres.append(mu)
  #avgpt.append(average(ptdata,weights=weightdata))
  #avgtruept.append(average(trueptdata,weights=weightdata))
  #sigmaRs.append(sigma)

  n,bins,patches = plt.hist(ptestdata,normed=True,bins=50,weights=weightdata,facecolor='b')
  # maximum likelihood estimates
  mu = average(ptestdata,weights=weightdata)
  var = average((ptestdata-mu)**2,weights=weightdata)
  sigma = sqrt(var)
  mu_err = sigma*sqrt(sum(weightdata**2))
  var_err = var*sqrt(2*sum(weightdata**2)) # from https://web.eecs.umich.edu/~fessler/papers/files/tr/stderr.pdf
  #var = sigma^2 -> var_err/var = 2*sigma_err/sigma
  sigma_err = 0.5*var_err/sigma
  gfunc = norm
  y = gfunc.pdf( bins, mu, sigma)
  l = plt.plot(bins, y, 'r--', linewidth=2)
  plt.xlabel('$p_T^{reco}$')
  plt.ylabel('a.u.')
  plt.savefig(options.plotDir+'/f1bin%d'%ptbin+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.identifier+'.png')
  plt.close()
  
  return muR,muR_err,sigmaR,sigmaR_err,mu,mu_err,sigma,sigma_err

calibrate()
