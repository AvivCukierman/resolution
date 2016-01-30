from numpy import load,log,linspace,digitize,array,mean,std,exp,all,average,sqrt
import os
import numpy
from scipy.optimize import curve_fit,fsolve
from scipy.stats import norm
from operator import sub
from optparse import OptionParser

parser = OptionParser()

# job configuration
parser.add_option("--inputDir", help="Directory containing input files",type=str, default="../data")
parser.add_option("--submitDir", help="Directory containing output files",type=str, default="../output")
parser.add_option("--plotDir", help="Directory containing plots",type=str, default="../plots")
parser.add_option("--numEvents", help="How many events to include",type=int, default=100000)
parser.add_option("-i","--identifier", help="sample identifier",type=str, default="myjets")
parser.add_option("-r","--root", help="Root input",action="store_true", default=False)

# jet configuration
parser.add_option("-j","--jet", help="jet identifier",type=str, default="j0")
parser.add_option("-c","--cut", default=float('-inf'), type=float, help="low pT cut on reco jets")
parser.add_option("--mineta", help="min abs(eta) on truth jets", type=float, default=0)
parser.add_option("--maxeta", help="max abs(eta) on truth jets", type=float, default=float('inf'))
parser.add_option("--mindr", help="min dr on truth jets", type=float, default=0)

# analysis configuration
parser.add_option("--minnpv", help="min NPV", type=int, default=5)
parser.add_option("--maxnpv", help="max NPV", type=int, default=30)
parser.add_option("--npvbin", help="size of NPV bins", type=int, default=5)

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

asym = 10 # shift distribution to the right to get better fit and deal with negative pTs
def R(x,a,b,c):
    ax = array(x)
    result = a + b/log(ax+asym) + c/log(ax+asym)**2
    return result 

def g(x,a,b,c):
    ax = array(x)
    return R(x,a,b,c)*x 

#derivative of g
def dg(x,a,b,c):
    ax = array(x)
    result =  a + b/log(ax+asym) - b/log(ax+asym)**2*ax/(ax+asym) + c/log(ax+asym)**2 - 2*c/log(ax+asym)**3*ax/(ax+asym)
    return result

def g1(x,a,b,c):
    ax = array(x)
    func = lambda y: ax-g(y,a,b,c)
    return fsolve(func,ax)

#ptedges = range(20,60,2)+range(60,150,5)
ptedges = range(20,80,2)

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
#plt.style.use('atlas')
import matplotlib.mlab as mlab

def readRoot(jet='j0'):
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
  if '%spt'%jet not in branches: raise RuntimeError('%spt branch does not exist. This is the branch containing reco jet pTs.'%jet)
  else: print '== \'%spt\' branch is being read as reco jet pTs =='%jet
  if 't%spt'%jet not in branches: raise RuntimeError('t%spt branch does not exist. This is the branch containing truth jet pTs.'%jet)
  else: print '== \'t%spt\' branch is being read as truth jet pTs =='%jet
  if 'NPV' not in branches: raise RuntimeError('NPV branch does not exist. This is the branch containing NPVs.')
  else: print '== \'NPV\' branch is being read as NPVs =='
  if 'event_weight' not in branches: print '== \'event_weight\' branch does not exist; weighting every event the same =='  
  else: print '== \'event_weight\' branch is being read as event weights =='
  if 't'+jet+'eta' not in branches: print '== \''+'t'+jet+'eta'+'\' branch does not exist; no eta cuts set =='  
  else: print '== \''+'t'+jet+'eta'+'\' branch being read as truth jet etas =='
  if 't'+jet+'mindr' not in branches: print '== \''+'t'+jet+'mindr'+'\' branch does not exist; no mindr cuts set =='  
  else: print '== \''+'t'+jet+'mindr'+'\' branch being read as truth jet mindrs =='

  nentries = tree.GetEntries()

  npvs = [] 
  responses = [] 
  truepts = [] 
  recopts = []
  weights = [] 
  for jentry in xrange(nentries):
      if jentry>options.numEvents: continue
      tree.GetEntry(jentry)
      
      if not jentry%1000:
          stdout.write('== \r%d events read ==\n'%jentry)
          stdout.flush()

      jpts = getattr(tree,'%spt'%jet)
      tjpts = getattr(tree,'t%spt'%jet)
      tjetas = getattr(tree,'t%seta'%jet)
      tjmindr = getattr(tree,'t%smindr'%jet)
      npv = tree.NPV
      #rho = tree.rho
      event_weight = tree.event_weight*sampweight

      truept = []
      recopt = []
      weightjets = []
      for jpt,tjpt,tjeta,tjmindr in zip(jpts,tjpts,tjetas,tjmindr):
          if fabs(tjeta)>options.maxeta or fabs(tjeta)<options.mineta: continue
          if tjmindr<options.mindr: continue
          truept.append(tjpt)
          recopt.append(jpt)
          weightjets.append(event_weight)

      npv = [npv]*len(truept)
      npvs += npv
      truepts += truept
      recopts += recopt
      weights += weightjets

  from numpy import save
  save(options.submitDir+'/truepts_'+jet+'_'+finalmu,truepts)
  save(options.submitDir+'/recopts_'+jet+'_'+finalmu,recopts)
  save(options.submitDir+'/weights_'+jet+'_'+finalmu,weights)
  save(options.submitDir+'/npvs_'+jet+'_'+finalmu,npvs)

  return array(recopts),array(truepts),array(npvs),array(weights)

def fitres(jet='j0',params=[]):
  if options.root: 
    recopts,truepts,npvs,weights = readRoot(jet)
    eta_cuts = [True]*len(truepts) 
    mindr_cuts = [True]*len(truepts) 
    print '== Root files read. Data saved in '+options.submitDir+'. Next time you can run without -r option and it should be faster. =='
    print '== There are '+str(len(truepts))+' total jets =='
  else:
    # truepts, recopts, npvs required
    filename = options.submitDir+'/'+'truepts_'+jet+'_'+options.identifier+'.npy'
    if not os.path.exists(filename): raise OSError(filename +' does not exist')
    print '== Loading file <'+filename+'> as truth jet pTs =='
    truepts = load(filename)
    print '== There are '+str(len(truepts))+' total jets =='

    filename = options.submitDir+'/'+'recopts_'+jet+'_'+options.identifier+'.npy'
    if not os.path.exists(filename): raise OSError(filename +' does not exist')
    print '== Loading file <'+filename+'> as reco jet pTs =='
    recopts = load(filename)
    if not len(recopts)==len(truepts):
      raise RuntimeError('There should be the same number of reco jets as truth jets')

    filename = options.submitDir+'/'+'npvs_'+jet+'_'+options.identifier+'.npy'
    if not os.path.exists(filename): raise OSError(filename +' does not exist')
    print '== Loading file <'+filename+'> as NPVs =='
    npvs = load(filename)
    if not len(npvs)==len(truepts):
      raise RuntimeError('There should be the same number of npvs as truth jets (format is one entry per truth jet)')

    filename = options.submitDir+'/'+'weights_'+jet+'_'+options.identifier+'.npy'
    if os.path.exists(filename): 
      print '== Loading file <'+filename+'> as event weights =='
      weights = load(filename)
      if not len(weights)==len(truepts):
        raise RuntimeError('There should be the same number of weights as truth jets (format is one entry per truth jet)')
    else:
      print '== '+filename+' does not exist; weighting every event the same =='
      weights = array([1]*len(truepts))

    filename = options.submitDir+'/'+'etas_'+jet+'_'+options.identifier+'.npy'
    if os.path.exists(filename): 
      print '== Loading file <'+filename+'> as truth jet etas =='
      etas = load(filename)
      if not len(etas)==len(truepts):
        raise RuntimeError('There should be the same number of etas as truth jets')
      eta_cuts = numpy.all([abs(etas)<options.mineta,abs(etas)>options.maxeta]) 
    else:
      print '== '+filename+' does not exist; no additional eta cuts set (if you started reading from a root file, this is ok) =='
      eta_cuts = [True]*len(truepts) 

    filename = options.submitDir+'/'+'mindrs_'+jet+'_'+options.identifier+'.npy'
    if os.path.exists(filename):
      print '== Loading file <'+filename+'> as truth jet mindRs =='
      mindrs = load(filename)
      if not len(mindrs)==len(truepts):
        raise RuntimeError('There should be the same number of mindRs as truth jets')
      mindr_cuts = mindrs>options.mindr
    else:
      print '== '+filename+' does not exist; no additional mindR cuts set (if you started reading from a root file, this is ok) =='
      mindr_cuts = [True]*len(truepts) 

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

  for npvbin in xrange(1,len(npvedges)):
    avgres = []
    avgpt = []
    avgtruept = []
    sigmas = []
    sigmaRs = []

    for ptbin in xrange(1,len(ptedges)): 
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
      plt.savefig(options.plotDir+'/resbin%d'%ptbin+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+jet+'_'+options.identifier+'.png')
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
      plt.savefig(options.plotDir+'/gbin%d'%ptbin+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+jet+'_'+options.identifier+'.png')
      plt.close()
      sigmas.append(sigma)

    xp = linspace(5,150,75)

    #Fit to response vs. pTtrue
    Ropt, Rcov = curve_fit(R, avgtruept, avgres)

    plt.plot(truepts,responses,'.',avgtruept,avgres,'o',xp,R(xp,*Ropt),'r-')
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('$p_T^{reco}/p_T^{true}$')
    if do_all: plt.ylim(-0.5,2)
    else: plt.ylim(0,2)
    plt.xlim(0,80)
    plt.savefig(options.plotDir+'/jetresponse_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+jet+'_'+options.identifier+'.png')
    plt.close()

    #g = R*t:
    plt.plot(truepts,recopts,'.',avgtruept,avgpt,'o',xp,R(xp,*Ropt)*array(xp),'r-')
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('$p_T^{reco}$ [GeV]')
    if do_all: plt.ylim(-10,80)
    else: plt.ylim(0,80)
    plt.xlim(0,80)
    plt.savefig(options.plotDir+'/jetg_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+jet+'_'+options.identifier+'.png')
    plt.close()

    #dg = d(R*t):
    plt.plot(xp,dg(xp,*Ropt),'r-')
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('$g\'(p_T^{true})$')
    plt.ylim(0,1)
    plt.xlim(0,80)
    plt.savefig(options.plotDir+'/jetdg_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+jet+'_'+options.identifier+'.png')
    plt.close()

    plt.plot(avgtruept,g1(avgpt,*Ropt),'.')
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('$g^{-1}(<p_T^{reco}>)$ [GeV]')
    plt.xlim(0,80)
    plt.ylim(0,80)
    plt.savefig(options.plotDir+'/jetg1_ptttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+jet+'_'+options.identifier+'.png')
    plt.close()

    plt.plot(avgtruept,g1(avgpt,*Ropt)/avgtruept,'.')
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('$g^{-1}(<p_T^{reco}>)/p_T^{true}$')
    plt.xlim(0,80)
    plt.ylim(0.95,1.05)
    plt.savefig(options.plotDir+'/jetclosure_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+jet+'_'+options.identifier+'.png')
    plt.close()


    sigma_calculation=array(sigmas)/dg(avgtruept,*Ropt)
    npv_sigmas[npvedges[npvbin]] = sigma_calculation
    plt.plot(avgtruept,sigma_calculation,color='b',linestyle='-',label='NPV '+str(npvedges[npvbin-1])+'-'+str(npvedges[npvbin]))
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('$\sigma[p_T^{reco}]$ [GeV]')
    plt.ylim(min(sigma_calculation)-1,max(sigma_calculation)+1)
    plt.xlim(0,80)
    plt.legend(loc='upper left',frameon=False,numpoints=1)
    plt.savefig(options.plotDir+'/jetsigma_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+jet+'_'+options.identifier+'.png')
    plt.close()
    
    sigma_calculation=array(sigmaRs)/dg(avgtruept,*Ropt)
    npv_sigmaRs[npvedges[npvbin]] = sigma_calculation
    plt.plot(avgtruept,sigma_calculation,color='b',linestyle='-',label='NPV '+str(npvedges[npvbin-1])+'-'+str(npvedges[npvbin]))
    plt.xlabel('$p_T^{true}$ [GeV]')
    plt.ylabel('$\sigma[p_T^{reco}/p_T^{true}]$')
    plt.ylim(0,max(sigma_calculation)+0.1) 
    plt.xlim(0,150)
    plt.legend(loc='upper right',frameon=False,numpoints=1)
    plt.savefig(options.plotDir+'/jetsigmaR_pttrue'+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+jet+'_'+options.identifier+'.png')
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
  plt.savefig(options.plotDir+'/jetsigma_pttrue_'+jet+'_'+options.identifier+'.png')
  plt.close()

  for i,npv in enumerate(npv_keys):
    plt.plot(avgtruept,npv_sigmaRs[npv],color=colors[i],linestyle=linestyles[i],label=labels[i])
  plt.xlabel('$p_T^{true}$ [GeV]')
  plt.ylabel('$\sigma[p_T^{reco}/p_T^{true}]$')
  highlim = max([max(s) for s in npv_sigmaRs.values()])
  plt.ylim(0,highlim+0.1)
  plt.xlim(0,80)
  plt.legend(loc='upper left',frameon=False,numpoints=1)
  plt.savefig(options.plotDir+'/jetsigmaR_pttrue_'+jet+'_'+options.identifier+'.png')
  plt.close()


  return Ropt

j0fit = fitres(options.jet)
