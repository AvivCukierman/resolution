from quantile import quantile
from extend_scipy_stats import gaussian_kde
from scipy import stats 
import numpy
from numpy import linspace,array,mean,std,average,sqrt
import pdb

import ROOT as r
def mode_est(x,w,nSigmaA,nSigmaB):
  hist = r.TH1F("h1","h1",100,min(x),max(x))
  for xx,ww in zip(x,w): hist.Fill(xx,ww)

  mean = hist.GetMean()
  sigma = hist.GetRMS()
  max_val = hist.GetMaximum()

  lowestX = 0
  highestX = 0
  for i in range(hist.GetNbinsX()):
    if hist.GetBinContent(i+1) > max_val/10: highestX = hist.GetBinCenter(i+1)
  for i in reversed(range(hist.GetNbinsX())):
    if hist.GetBinContent(i+1) > max_val/10: lowestX = hist.GetBinCenter(i+1)

  #gfit = r.TF1("Gaussian","gaus", mean - nSigmaB * sigma, mean + nSigmaA * sigma) # Create the fit function
  #gfit.SetParameters(mean, sigma);
  #hist.Fit(gfit,"RQ0"); # Fit histogram h

  for nFit in range(2):
    minRange = mean - nSigmaB * sigma
    maxRange = mean + nSigmaA * sigma
    if minRange < lowestX: minRange = lowestX
    if maxRange > highestX: maxRange = highestX
    gfit = r.TF1("Gaussian","gaus", minRange, maxRange) # Create the fit function
    gfit.SetParLimits(1, minRange, maxRange)
    hist.Fit(gfit,"RQ0") # Fit histogram h
    mean=gfit.GetParameter(1)
    sigma=gfit.GetParameter(2)

  return mean,sigma,minRange,maxRange

def distribution_values(data,weights,central,eff=1):
      weights=weights/sum(weights) #normalize
      # maximum likelihood estimates
      mean = average(data,weights=weights)
      var = average((data-mean)**2,weights=weights)
      std = sqrt(var)
      mean_err = std*sqrt(sum(weights**2))
      var_err = var*sqrt(2*sum(weights**2)) # from https://web.eecs.umich.edu/~fessler/papers/files/tr/stderr.pdf
      #var = sigma^2 -> var_err/var = 2*sigma_err/sigma
      std_err = 0.5*var_err/std
      err = False
      if central == 'absolute_median':
        mu = quantile(data,weights,(0.5-(1-eff))/eff)
        mu_err = 1.2533*mean_err #http://influentialpoints.com/Training/standard_error_of_median.htm
        upper_quantile = quantile(data,weights,(0.8413-(1-eff))/eff) #CDF(1)
        if 0.1587 < (1-eff): lower_quantile = float('-inf')
        else: lower_quantile = quantile(data,weights,(0.1587-(1-eff))/eff)
        sigma = (upper_quantile-mu)
        sigma_err = 1.573*std_err #http://stats.stackexchange.com/questions/110902/error-on-interquartile-range seems reasonable
        return mu,mu_err,sigma,sigma_err,upper_quantile,lower_quantile,err
      if central == 'median':
        mu = quantile(data,weights,0.5)
        mu_err = 1.2533*mean_err #http://influentialpoints.com/Training/standard_error_of_median.htm
        upper_quantile = quantile(data,weights,0.8413) #CDF(1)
        lower_quantile = quantile(data,weights,0.1587) #CDF(-1)
        sigma = 0.5*(upper_quantile-lower_quantile)
        sigma_err = 1.573*std_err #http://stats.stackexchange.com/questions/110902/error-on-interquartile-range seems reasonable
        return mu,mu_err,sigma,sigma_err,upper_quantile,lower_quantile
      if central == 'mean':
        return mean,mean_err,std,std_err
      if central == 'trimmed' or central == 'mode' or central=='kde_mode':
        n,bins = numpy.histogram(data,weights=weights,bins=100)
        newmean_est,newstd_est,lower_val,upper_val = mode_est(data,weights,1.75,1.75)

        newweights = weights[numpy.all([data>lower_val,data<upper_val],axis=0)]
        newweights/=sum(newweights)
        new_mean_err = newstd_est*sqrt(sum(newweights**2))
        new_var_err = newstd_est**2*sqrt(2*sum(newweights**2)) # from https://web.eecs.umich.edu/~fessler/papers/files/tr/stderr.pdf
        #var = sigma^2 -> var_err/var = 2*sigma_err/sigma
        new_std_err = 0.5*new_var_err/newstd_est

        if central == 'trimmed': return newmean_est,new_mean_err,newstd_est,new_std_err,lower_val,upper_val 
        #else central == 'mode' or central=='kde_mode': use absolute IQR
        if central == 'kde_mode':
          #Calculate mode using KDE:
          bw = len(data)**(-1./5)  #scotts factor
          kernel = gaussian_kde(data,weights=weights,bw_method=bw*2.0)
          bins = numpy.histogram(data,weights=weights,bins=50)[1]
          y = kernel(bins)
          kde_mode_est = bins[numpy.argmax(y)] 

          binsize = bins[1]-bins[0]
          smallbins = numpy.linspace(kde_mode_est-2*binsize,kde_mode_est+2*binsize,400) 
          mode = smallbins[numpy.argmax(kernel(smallbins))] 
          mu = mode

        if central == 'mode': mu = newmean_est
        mu_quantile = sum(weights[data<mu]) 
        absolute_mu_quantile = mu_quantile*eff+(1-eff) #should be approximately 0.5
        err = False
        if abs(absolute_mu_quantile-0.5) > 0.1:
          print '<< Fitted mode is > .1 away from 50th percentile. Efficiency might be less than 50%. Calibration value might not be very useful. >>'
          err = True
        if absolute_mu_quantile > 1-.3413:
          print '<< Fitted mode is at > 65th percentile! Bad fit. Efficiency might be less than 50%. Returning max value - mode. >>'
          upper_quantile = max(data)
          err = True
        else: upper_quantile = quantile(data,weights,mu_quantile+.3413/eff) # = (absolute_mu_quantile+0.3413-(1-eff))/eff
        if absolute_mu_quantile<0.3413+1-eff: lower_quantile = float('-inf')
        else: lower_quantile = quantile(data,weights,mu_quantile-.3413/eff) # = (absolute_mu_quantile-0.3413-(1-eff))/eff
        sigma = (upper_quantile-mu)
        sigma_err = 1.573*std_err #http://stats.stackexchange.com/questions/110902/error-on-interquartile-range seems reasonable
        return mu,new_mean_err,sigma,sigma_err,upper_quantile,lower_quantile,err

  
  

