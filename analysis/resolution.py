from numpy import load,log,linspace,digitize,array,mean,std,exp,all,average,sqrt,asarray,sign
import os
import numpy
from numpy import save
from scipy.optimize import curve_fit,fsolve
from scipy.stats import norm
from operator import sub
from optparse import OptionParser
os.environ[ 'MPLCONFIGDIR' ] = '/tmp/' #to get matplotlib to work

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
parser.add_option("--tjetpt", help="truth jet pT branch name",type=str, default="tj0pt")
parser.add_option("--npv", help="NPV branch name",type=str, default="NPV")
parser.add_option("--tjeteta", help="truth jet eta branch name",type=str, default="tj0eta")
parser.add_option("--tjetmindr", help="truth jet mindr branch name",type=str, default="tj0mindr")
parser.add_option("--event_weight", help="event weight branch name",type=str, default="event_weight")

# jet configuration
parser.add_option("-c","--cut", default=float('-inf'), type=float, help="low pT cut on reco jets")
parser.add_option("--mineta", help="min abs(eta) on truth jets", type=float, default=0)
parser.add_option("--maxeta", help="max abs(eta) on truth jets", type=float, default=float('inf'))
parser.add_option("--mindr", help="min dr on truth jets", type=float, default=0)

# analysis configuration
parser.add_option("-n","--doCal",help="Do full numerical inversion calibration",action="store_true",default=False)
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

do_all = False
if options.cut==float('-inf'): do_all=True 

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
  if options.tjetpt not in branches: raise RuntimeError(options.tjetpt+' branch does not exist. This is the branch containing truth jet pTs.')
  else: print '== \''+options.tjetpt+'\' branch is being read as truth jet pTs =='
  if options.npv not in branches: raise RuntimeError(options.npv+' branch does not exist. This is the branch containing NPVs.')
  else: print '== \''+options.npv+'\' branch is being read as NPVs =='
  # optional:
  has_event_weight = False
  has_eta = False
  has_mindr = False
  if options.event_weight not in branches: print '== \''+options.event_weight+'\' branch does not exist; weighting every event the same =='  
  else:
    has_event_weight=True
    print '== \''+options.event_weight+'\' branch is being read as event weights =='
  if options.tjeteta not in branches: print '== \''+options.tjeteta+'\' branch does not exist; no eta cuts set =='  
  else:
    has_eta = True
    print '== \''+options.tjeteta+'\' branch being read as truth jet etas =='
  if options.tjetmindr not in branches: print '== \''+options.tjetmindr+'\' branch does not exist; no mindr cuts set =='  
  else:
    has_mindr = True
    print '== \''+options.tjetmindr+'\' branch being read as truth jet mindrs =='

  nentries = tree.GetEntries()

  npvs = [] 
  responses = [] 
  truepts = [] 
  recopts = []
  weights = [] 
  for jentry in xrange(nentries):
      if jentry>options.numEvents and options.numEvents>0: continue
      tree.GetEntry(jentry)
      
      if not jentry%1000:
          stdout.write('== \r%d events read ==\n'%jentry)
          stdout.flush()

      jpts = getattr(tree,options.jetpt)
      tjpts = getattr(tree,options.tjetpt)
      npv = tree.NPV

      if has_eta: tjetas = getattr(tree,options.tjeteta)
      if has_mindr: tjmindrs = getattr(tree,options.tjetmindr)
      if has_event_weight: event_weight = tree.event_weight*sampweight

      truept = []
      recopt = []
      weightjets = []
      for i,(jpt,tjpt) in enumerate(zip(jpts,tjpts)):
          if has_eta:
            tjeta = tjetas[i]
            if fabs(tjeta)>options.maxeta or fabs(tjeta)<options.mineta: continue
          if has_mindr:
            tjmindr = tjmindrs[i]
            if tjmindr<options.mindr: continue
          truept.append(tjpt)
          recopt.append(jpt)
          if has_event_weight:
            weightjets.append(event_weight)
          else: weightjets.append(1) #set all events to have the same weight

      npv = [npv]*len(truept)
      npvs += npv
      truepts += truept
      recopts += recopt
      weights += weightjets

  save(options.submitDir+'/truepts_'+finalmu,truepts)
  save(options.submitDir+'/recopts_'+finalmu,recopts)
  save(options.submitDir+'/npvs_'+finalmu,npvs)
  if has_event_weight: save(options.submitDir+'/weights_'+finalmu,weights)

  return array(recopts),array(truepts),array(npvs),array(weights)

def fitres(params=[]):
  if options.root: 
    recopts,truepts,npvs,weights = readRoot()
    eta_cuts = [True]*len(truepts) 
    mindr_cuts = [True]*len(truepts) 
    print '== Root files read. Data saved in '+options.submitDir+'. Next time you can run without -r option and it should be faster. =='
    print '== There are '+str(len(truepts))+' total jets =='
  else:
    # truepts, recopts, npvs required
    filename = options.submitDir+'/'+'truepts_'+options.identifier+'.npy'
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
      eta_cuts = numpy.all([abs(etas)<options.mineta,abs(etas)>options.maxeta]) 
    else:
      print '== '+filename+' does not exist; no additional eta cuts set (if you started reading from a root file, this is ok) =='
      eta_cuts = [True]*len(truepts) 

    filename = options.submitDir+'/'+'mindrs_'+options.identifier+'.npy'
    if os.path.exists(filename):
      print '== Loading file <'+filename+'> as truth jet mindRs =='
      mindrs = load(filename)
      if not len(mindrs)==len(truepts):
        raise RuntimeError('There should be the same number of mindRs as truth jets')
      mindr_cuts = mindrs>options.mindr
    else:
      print '== '+filename+' does not exist; no additional mindR cuts set (if you started reading from a root file, this is ok) =='
      mindr_cuts = [True]*len(truepts) 

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
  npv_sigmaRs = {npvedges[npvbin]: [] for npvbin in xrange(1,len(npvedges))}
  Ropts = {npvedges[npvbin]: [] for npvbin in xrange(1,len(npvedges))}

  for npvbin in xrange(1,len(npvedges)):
    print '>> Processing NPV bin '+str(npvedges[npvbin-1])+'-'+str(npvedges[npvbin])
    avgres = []
    avgpt = []
    avgtruept = []
    sigmas = []
    sigmaRs = []

    for ptbin in xrange(1,len(ptedges)): 
      #print '>> >> Processing pT bin '+str(ptedges[ptbin-1])+'-'+str(ptedges[ptbin])+' GeV'
      resdata = responses[all([ptbins==ptbin,npvbins==npvbin],axis=0)]
      ptdata = recopts[all([ptbins==ptbin,npvbins==npvbin],axis=0)]
      trueptdata = truepts[all([ptbins==ptbin,npvbins==npvbin],axis=0)]
      weightdata = weights[all([ptbins==ptbin,npvbins==npvbin],axis=0)]
      if len(resdata)<100: print 'Low statistics ('+str(len(resdata))+' jets) in bin with pT = ' +str(ptedges[ptbin])+' and NPV between '+str(npvedges[npvbin-1])+' and '+str(npvedges[npvbin])
      # maximum likelihood estimates
      mu = average(resdata,weights=weightdata)
      sigma = sqrt(average((resdata-mu)**2,weights=weightdata))
      n,bins,patches = plt.hist(resdata,normed=True,bins=50,weights=weightdata)
      gfunc = norm
      y = gfunc.pdf( bins, mu, sigma)
      l = plt.plot(bins, y, 'r--', linewidth=2)
      plt.xlabel('$p_T^{reco}/p_T^{true}$')
      plt.ylabel('a.u.')
      plt.savefig(options.plotDir+'/resbin%d'%ptbin+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.identifier+'.png')
      plt.close()
      avgres.append(mu)
      avgpt.append(average(ptdata,weights=weightdata))
      avgtruept.append(average(trueptdata,weights=weightdata))
      sigmaRs.append(sigma)

      n,bins,patches = plt.hist(ptdata,normed=True,bins=50,weights=weightdata)
      # maximum likelihood estimates
      mu = average(ptdata,weights=weightdata)
      sigma = sqrt(average((ptdata-mu)**2,weights=weightdata))
      gfunc = norm
      y = gfunc.pdf( bins, mu, sigma)
      l = plt.plot(bins, y, 'r--', linewidth=2)
      plt.xlabel('$p_T^{reco}$')
      plt.ylabel('a.u.')
      plt.savefig(options.plotDir+'/fbin%d'%ptbin+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.identifier+'.png')
      plt.close()
      sigmas.append(sigma)

    xp = linspace(5,150,75)

    #Fit to response vs. pTtrue
    Ropt, Rcov = curve_fit(R, avgtruept, avgres)
    Ropts[npvedges[npvbin]] = Ropt 

    plt.plot(truepts[npvbins==npvbin],responses[npvbins==npvbin],'.',avgtruept,avgres,'o',xp,R(xp,*Ropt),'r-')
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('$p_T^{reco}/p_T^{true}$')
    if do_all: plt.ylim(-0.5,2)
    else: plt.ylim(0,2)
    plt.xlim(0,80)
    plt.savefig(options.plotDir+'/jetresponse_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.identifier+'.png')
    plt.close()

    #g = R*t:
    print Ropt
    plt.plot(truepts[npvbins==npvbin],recopts[npvbins==npvbin],'.',avgtruept,avgpt,'o',xp,R(xp,*Ropt)*array(xp),'r-')
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('$p_T^{reco}$ [GeV]')
    if do_all: plt.ylim(-10,80)
    else: plt.ylim(0,80)
    plt.xlim(0,80)
    plt.savefig(options.plotDir+'/jetf_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.identifier+'.png')
    plt.close()

    #dg = d(R*t):
    plt.plot(xp,dg(xp,*Ropt),'r-')
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('$f\'(p_T^{true})$')
    plt.ylim(0,1)
    plt.xlim(0,80)
    plt.savefig(options.plotDir+'/jetdf_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.identifier+'.png')
    plt.close()

    if options.doCal:
      calmus = []
      calmuRs = []
      calsigmas = []
      calsigmaRs = []
      for ptbin in xrange(1,len(ptedges)): 
        ptdata = recopts[all([ptbins==ptbin,npvbins==npvbin],axis=0)]
        trueptdata = truepts[all([ptbins==ptbin,npvbins==npvbin],axis=0)]
        weightdata = weights[all([ptbins==ptbin,npvbins==npvbin],axis=0)]
        ptestdata = g1(ptdata,*Ropt)
        muR,sigmaR,mu,sigma = numerical_inversion(ptestdata,trueptdata,weightdata,Ropt,ptbin,npvedges,npvbin)
        calmuRs.append(muR)
        calsigmaRs.append(sigmaR)
        calmus.append(mu)
        calsigmas.append(sigma)

      estpts = g1(recopts,*Ropt)
      plt.plot(truepts[npvbins==npvbin],estpts[npvbins==npvbin],'.',avgtruept,calmus,'go')
      plt.xlabel('$p_T^{true}$ [GeV]')
      plt.ylabel('$p_T^{reco,cal}$ [GeV]')
      plt.xlim(0,80)
      plt.ylim(0,80)
      plt.savefig(options.plotDir+'/jetf1_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.identifier+'.png')
      plt.close()
      
      closure = estpts/truepts
      plt.plot(truepts[npvbins==npvbin],closure[npvbins==npvbin],'.',avgtruept,calmuRs,'go')
      plt.xlabel('$p_T^{true}$ [GeV]')
      plt.ylabel('$p_T^{reco,cal}/p_T^{true}$')
      plt.xlim(0,80)
      plt.ylim(0,2)
      plt.savefig(options.plotDir+'/jetclosure_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.identifier+'.png')
      plt.close()

      plt.plot(avgtruept,calmuRs,'go')
      plt.xlabel('$p_T^{true}$ [GeV]')
      plt.ylabel('$p_T^{reco,cal}/p_T^{true}$')
      plt.xlim(0,80)
      plt.ylim(.95,1.05)
      plt.savefig(options.plotDir+'/jetclosure_pttrue_zoom'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.identifier+'.png')
      plt.close()


    if options.doCal: sigma_calculation = calsigmas
    else: sigma_calculation = array(sigmas)/dg(avgtruept,*Ropt)
    npv_sigmas[npvedges[npvbin]] = sigma_calculation
    plt.plot(avgtruept,sigma_calculation,color='b',linestyle='-',label='NPV '+str(npvedges[npvbin-1])+'-'+str(npvedges[npvbin]))
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('$\sigma[p_T^{reco}]$ [GeV]')
    plt.ylim(min(sigma_calculation)-1,max(sigma_calculation)+1)
    plt.xlim(0,80)
    plt.legend(loc='upper left',frameon=False,numpoints=1)
    plt.savefig(options.plotDir+'/jetsigma_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.identifier+'.png')
    plt.close()
    
    if options.doCal: sigma_calculation = calsigmaRs 
    else: sigma_calculation = sigma_calculation=array(sigmaRs)/dg(avgtruept,*Ropt)
    npv_sigmaRs[npvedges[npvbin]] = sigma_calculation
    plt.plot(avgtruept,sigma_calculation,color='b',linestyle='-',label='NPV '+str(npvedges[npvbin-1])+'-'+str(npvedges[npvbin]))
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('$\sigma[p_T^{reco}/p_T^{true}]$')
    plt.ylim(0,max(sigma_calculation)+0.1) 
    plt.xlim(0,150)
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
    plt.plot(avgtruept,npv_sigmas[npv],color=colors[i],linestyle=linestyles[i],label=labels[i])
  plt.xlabel('$p_T^{true}$ [GeV]')
  plt.ylabel('$\sigma[p_T^{reco}]$ [GeV]')
  lowlim = min([min(s) for s in npv_sigmas.values()])
  highlim = max([max(s) for s in npv_sigmas.values()])
  plt.ylim(lowlim-1,highlim+1)
  plt.xlim(0,80)
  plt.legend(loc='upper left',frameon=False,numpoints=1)
  plt.savefig(options.plotDir+'/jetsigma_pttrue_'+options.identifier+'.png')
  plt.close()

  for i,npv in enumerate(npv_keys):
    plt.plot(avgtruept,npv_sigmaRs[npv],color=colors[i],linestyle=linestyles[i],label=labels[i])
  plt.xlabel('$p_T^{true}$ [GeV]')
  plt.ylabel('$\sigma[p_T^{reco}/p_T^{true}]$')
  highlim = max([max(s) for s in npv_sigmaRs.values()])
  plt.ylim(0,highlim+0.1)
  plt.xlim(0,80)
  plt.legend(loc='upper left',frameon=False,numpoints=1)
  plt.savefig(options.plotDir+'/jetsigmaR_pttrue_'+options.identifier+'.png')
  plt.close()

  for i,ptbin in enumerate(ptedges):
    if i==0: continue
    plt.plot(array(npv_keys)-0.5*options.npvbin,[npv_sigmas[n][i-1] for n in npv_keys],color='b',linestyle='-',label=str(ptedges[i-1])+' GeV $< p_T^{true} < $'+str(ptedges[i])+' GeV')
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
    plt.plot(array(npv_keys)-0.5*options.npvbin,[npv_sigmaRs[n][i-1] for n in npv_keys],color='b',linestyle='-',label=str(ptedges[i-1])+' GeV $< p_T^{true} < $'+str(ptedges[i])+' GeV')
    plt.xlabel('NPV')
    plt.ylabel('$\sigma[p_T^{reco}/p_T^{true}]$')
    lowlim = 0 
    highlim = max(npv_sigmaRs[n][i-1] for n in npv_keys)
    plt.ylim(lowlim,highlim+0.1)
    plt.xlim(options.minnpv,options.maxnpv)
    plt.legend(loc='upper left',frameon=False,numpoints=1)
    plt.savefig(options.plotDir+'/jetsigmaR_NPV_pt'+str(ptedges[i-1])+str(ptedges[i])+'_'+options.identifier+'.png')
    plt.close()



  return Ropts,npv_sigmas,npv_sigmaRs,avgtruept

def numerical_inversion(ptestdata,trueptdata,weightdata,Ropt,ptbin,npvedges,npvbin):
  resdata = ptestdata/trueptdata
  muR = average(resdata,weights=weightdata)
  sigmaR = sqrt(average((resdata-muR)**2,weights=weightdata))
  n,bins,patches = plt.hist(resdata,normed=True,bins=50,weights=weightdata)
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

  n,bins,patches = plt.hist(ptestdata,normed=True,bins=50,weights=weightdata)
  # maximum likelihood estimates
  mu = average(ptestdata,weights=weightdata)
  sigma = sqrt(average((ptestdata-mu)**2,weights=weightdata))
  gfunc = norm
  y = gfunc.pdf( bins, mu, sigma)
  l = plt.plot(bins, y, 'r--', linewidth=2)
  plt.xlabel('$p_T^{reco}$')
  plt.ylabel('a.u.')
  plt.savefig(options.plotDir+'/f1bin%d'%ptbin+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.identifier+'.png')
  plt.close()
  
  return muR,sigmaR,mu,sigma

(fit,sigmas,sigmaRs,pttrue) = fitres()
import pickle
pickle.dump(fit,open(options.submitDir+'/fit_'+options.identifier+'.p','wb'))
pickle.dump(sigmas,open(options.submitDir+'/sigmas_'+options.identifier+'.p','wb'))
pickle.dump(sigmaRs,open(options.submitDir+'/sigmaRs_'+options.identifier+'.p','wb'))
pickle.dump(pttrue,open(options.submitDir+'/pttruebins_'+options.identifier+'.p','wb'))
