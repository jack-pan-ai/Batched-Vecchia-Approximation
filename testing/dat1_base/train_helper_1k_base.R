library(GpGp)
n_replicates = 100
n = 1000

array_gpgp = function (n, m) 
{
  NNarray <- matrix(NA, n, m + 1)
  ordering = 1:n
  
  for (i in 1:n){
    NNarray[i, 1] = i
    if (i == 1){
    }else if(i <= (m+1)){
      NNarray[i, 2:i] = rev(ordering[1:(i-1)])
    }else{
      NNarray[i, 2:(m+1)] = rev(ordering[(i-m):(i-1)])
    }
  }
  return(NNarray)
}

for (i in 1:n_replicates) {
  locs = read.csv(paste0('./synthetic_ds/LOC_', n,'_univariate_matern_stationary_', i),
                  header=FALSE)
  locs = as.matrix(locs)
  y = read.csv(paste0('./synthetic_ds/Z1_', n, '_univariate_matern_stationary_', i),
               header=FALSE)
  y = as.matrix(y)
  n = length(y)
  
  #run 
  for(nn in c(2, 5, 10, 20, 40)){
      timing<-system.time({
        ot<-capture.output(
        fit<-fit_model_meanzero(y,locs, "matern_isotropic", max_iter=2000, 
                   fixed_parms=c(4), start_parms=c(0.01, 0.01, 0.01, 0),
                   NNarray = array_gpgp(n, nn), # comment it if your like to use knn
                   group=TRUE, m_seq=c(nn), 
                   reorder=TRUE, convtol = 1e-09)
      )
    })
    n_ot = length(ot)
    iter = as.numeric(gsub(".*Iter\\s+(\\d+).*", "\\1", ot[n_ot - 5]))
    # Subtract the values 
    data = data.frame(
      iterations = iter,
      time = timing["elapsed"],
      variance = fit$covparms[1],
      range = fit$covparms[2],
      smoothness = fit$covparms[3],
      log_likelihood = fit$loglik
    )
    write.csv(data, 
              file = paste0("./batchsize_", as.character(nn), 
                            "_", as.character(nn), "/sum_", 
                            as.character(n),"_", 
                            as.character(nn), 
                            "_", as.character(i) ,".csv"), 
              row.names = FALSE, quote = FALSE)
    }
}
