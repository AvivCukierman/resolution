from numpy import load,log,linspace,digitize,array,mean,std,exp,all,average,sqrt,asarray,sign
import os
import numpy
from numpy import save
from scipy.optimize import curve_fit,fsolve
from scipy.stats import norm
from extend_scipy_stats import gaussian_kde
from operator import sub
from optparse import OptionParser
from quantile import quantile
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
parser.add_option("-m","--central",help="Choice of notion of central tendency (mean, mode, or median)",type='choice',choices=['mean','mode','median'],default='mean')
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
  etas = []
  mindrs = []
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
      eta = []
      mindr = []
      for i,(jpt,tjpt) in enumerate(zip(jpts,tjpts)):
          if has_eta:
            tjeta = tjetas[i]
            if fabs(tjeta)>options.maxeta or fabs(tjeta)<options.mineta: continue
          if has_mindr:
            tjmindr = tjmindrs[i]
            if tjmindr<options.mindr: continue
          truept.append(tjpt)
          recopt.append(jpt)
          eta.append(tjeta)
          mindr.append(tjmindr)
          if has_event_weight:
            weightjets.append(event_weight)
          else: weightjets.append(1) #set all events to have the same weight

      npv = [npv]*len(truept)
      npvs += npv
      truepts += truept
      recopts += recopt
      weights += weightjets
      etas += eta
      mindrs += mindr

  save(options.submitDir+'/truepts_'+finalmu,truepts)
  save(options.submitDir+'/recopts_'+finalmu,recopts)
  save(options.submitDir+'/npvs_'+finalmu,npvs)
  save(options.submitDir+'/etas_'+finalmu,etas)
  save(options.submitDir+'/mindrs_'+finalmu,mindrs)
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
      n,bins,patches = plt.hist(resdata,normed=True,bins=50,weights=weightdata,facecolor='b')
      # maximum likelihood estimates
      mean = average(resdata,weights=weightdata)
      var = average((resdata-mean)**2,weights=weightdata)
      std = sqrt(var)
      mean_err = std*sqrt(sum(weightdata**2))
      var_err = var*sqrt(2*sum(weightdata**2)) # from https://web.eecs.umich.edu/~fessler/papers/files/tr/stderr.pdf
      #var = sigma^2 -> var_err/var = 2*sigma_err/sigma
      std_err = 0.5*var_err/std
      if options.central == 'median':
        mu = quantile(resdata,weightdata,0.5)
        mu_err = 1.2533*mean_err #http://influentialpoints.com/Training/standard_error_of_median.htm
        upper_quantile = quantile(resdata,weightdata,0.8413) #CDF(1)
        lower_quantile = quantile(resdata,weightdata,0.1587) #CDF(-1)
        sigma = 0.5*(upper_quantile-lower_quantile)
        sigma_err = 1.573*std_err #http://stats.stackexchange.com/questions/110902/error-on-interquartile-range seems reasonable
        plt.plot((mu,mu),(0,plt.ylim()[1]),'r--',linewidth=2)
        height = 0.607*max(n) #height at x=1*sigma in normal distribution
        plt.plot((lower_quantile,upper_quantile),(height,height),'r--',linewidth=2)
        plt.plot((lower_quantile,lower_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
        plt.plot((upper_quantile,upper_quantile),(height-0.02,height+0.02),'r-',linewidth=2)
      if options.central == 'mean':
        mu = mean
        mu_err = mean_err
        sigma = std
        sigma_err = std_err 
        gfunc = norm
        y = gfunc.pdf( bins, mu, sigma)
        l = plt.plot(bins, y, 'r--', linewidth=2)
      if options.central == 'mode':
        '''mode_est = bins[numpy.argmax(n)]
        trimmed_indices = abs(resdata-mode_est)<1.5*std
        trimmed_resdata = resdata[trimmed_indices]
        trimmed_weightdata = weightdata[trimmed_indices]
        mode = average(trimmed_resdata,weights=trimmed_weightdata)
        mode_var = average((trimmed_resdata-mode)**2,weights=trimmed_weightdata)
        mode_std = sqrt(var)
        mode_err = mode_std*sqrt(sum(trimmed_weightdata**2))
        mode_var_err = mode_var*sqrt(2*sum(trimmed_weightdata**2)) # from https://web.eecs.umich.edu/~fessler/papers/files/tr/stderr.pdf
        #var = sigma^2 -> var_err/var = 2*sigma_err/sigma
        mode_std_err = 0.5*mode_var_err/mode_std'''
        '''trimmed_n,trimmed_bins,trimmed_patches = plt.hist(trimmed_resdata,normed=True,bins=50,weights=trimmed_weightdata,alpha=0)
        gfunc = norm
        y = gfunc.pdf(trimmed_bins,mu,sigma)
        l = plt.plot(trimmed_bins,y,'r--',linewidth=2)'''
        kernel = gaussian_kde(resdata,weights=weightdata)
        y = kernel(bins)
        plt.plot(bins,y,'r--',linewidth=2)
        mode_est = bins[numpy.argmax(kernel(bins))] 

        smallbins = numpy.linspace(mode_est-0.2,mode_est+0.2,400) 
        mode = smallbins[numpy.argmax(kernel(smallbins))] 

        mu = mode
        mu_err = mu 
        sigma = mu 
        sigma_err = mu 
      plt.xlabel('$p_T^{reco}/p_T^{true}$')
      plt.ylabel('a.u.')
      plt.savefig(options.plotDir+'/resbin%d'%ptbin+'_NPV'+str(npvedges[npvbin-1])+str(npvedges[npvbin])+'_'+options.central+'_'+options.identifier+'.png')
      plt.close()
      print mu,mu_err,sigma,sigma_err
      pdb.set_trace()
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

(fit,sigmas,sigma_errs,sigmaRs,sigmaR_errs,pttrue,ptedges) = fitres()
import pickle
pickle.dump(fit,open(options.submitDir+'/fit_'+options.identifier+'.p','wb'))
pickle.dump(sigmas,open(options.submitDir+'/sigmas_'+options.identifier+'.p','wb'))
pickle.dump(sigma_errs,open(options.submitDir+'/sigma_errs_'+options.identifier+'.p','wb'))
pickle.dump(sigmaRs,open(options.submitDir+'/sigmaRs_'+options.identifier+'.p','wb'))
pickle.dump(sigmaR_errs,open(options.submitDir+'/sigmaR_errs_'+options.identifier+'.p','wb'))
pickle.dump(pttrue,open(options.submitDir+'/avgpttrue_'+options.identifier+'.p','wb'))
pickle.dump(ptedges,open(options.submitDir+'/ptedges_'+options.identifier+'.p','wb'))
