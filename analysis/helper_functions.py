from quantile import quantile
from extend_scipy_stats import gaussian_kde
from scipy import stats 
import numpy
from numpy import linspace,array,mean,std,average,sqrt
import pdb

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
      if central == 'absolute_median':
        mu = quantile(data,weights,(0.5-(1-eff))/eff)
        mu_err = 1.2533*mean_err #http://influentialpoints.com/Training/standard_error_of_median.htm
        upper_quantile = quantile(data,weights,(0.8413-(1-eff))/eff) #CDF(1)
        if 0.1587 < (1-eff): lower_quantile = float('-inf')
        else: lower_quantile = quantile(data,weights,(0.1587-(1-eff))/eff)
        sigma = (upper_quantile-mu)
        sigma_err = 1.573*std_err #http://stats.stackexchange.com/questions/110902/error-on-interquartile-range seems reasonable
        return mu,mu_err,sigma,sigma_err,upper_quantile,lower_quantile
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
      if central == 'mode':
        bw = len(data)**(-1./5)  #scotts factor
        kernel = gaussian_kde(data,weights=weights,bw_method=bw*2.0)
        bins = numpy.histogram(data,weights=weights,bins=50)[1]
        y = kernel(bins)
        mode_est = bins[numpy.argmax(y)] 

        binsize = bins[1]-bins[0]
        smallbins = numpy.linspace(mode_est-2*binsize,mode_est+2*binsize,400) 
        mode = smallbins[numpy.argmax(kernel(smallbins))] 
        return mode,mean_err,std,std_err,kernel
      if central == 'trimmed':
        n,bins = numpy.histogram(data,weights=weights,bins=50)
        max_val = max(n) 
        max_bin = numpy.argmax(n)

        upper = max_bin
        while(n[upper]>max_val/3.5 and bins[upper]<1.75*std+bins[max_bin] and upper<len(n)): upper+=1
        lower = max_bin
        while(n[lower]>max_val/3.5 and bins[lower]>-1.75*std+bins[max_bin] and lower>0): lower-=1

        upper_val = bins[upper]
        lower_val = bins[lower]
        newdata = data[numpy.all([data>lower_val,data<upper_val],axis=0)]
        newweights = data[numpy.all([data>lower_val,data<upper_val],axis=0)]
        newweights=newweights/sum(newweights) #normalize

        newmean_est = average(newdata,weights=newweights)
        newvar_est = average((newdata-newmean_est)**2,weights=newweights)
        newstd_est = sqrt(newvar_est)

        rv = stats.truncnorm
        for _ in range(2):
          a = (lower_val-newmean_est)/(newstd_est)
          b = (upper_val-newmean_est)/(newstd_est)
          params = rv.fit(newdata,a,b,loc=newmean_est,scale=newstd_est,weights=newweights)

          newmean_est = params[2]
          newstd_est = params[3]

        new_mean_err = newstd_est*sqrt(sum(newweights**2))
        new_var_err = newstd_est**2*sqrt(2*sum(newweights**2)) # from https://web.eecs.umich.edu/~fessler/papers/files/tr/stderr.pdf
        #var = sigma^2 -> var_err/var = 2*sigma_err/sigma
        new_std_err = 0.5*new_var_err/newstd_est

        return newmean_est,new_mean_err,newstd_est,new_std_err,lower_val,upper_val 
