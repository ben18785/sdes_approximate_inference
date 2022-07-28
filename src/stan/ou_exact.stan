functions {
  matrix make_covariance(vector t, real theta, real kappa, real sigma_n, int N) {
    matrix[N, N] Sigma;
    real kappa2_o_2theta = kappa^2 / (2 * theta);
    for(i in 1:N)
      for(j in 1:N)
        Sigma[i, j] = kappa2_o_2theta * (exp(-theta * abs(t[i] - t[j])) - exp(-theta * (t[i] + t[j])));
    
    for(i in 1:N)
      Sigma[i, i] += sigma_n^2;
    return(Sigma);
  }
}

data {
  int nobs;
  vector[nobs] t;
  vector[nobs] Y;
  real X_0;
}

parameters {
  real theta;
  real<lower=0> kappa;
  real<lower=0> sigma_n;
}

model {
  vector[nobs] mu = X_0 * exp(-theta * t);

  Y ~ multi_normal(mu, make_covariance(t, theta, kappa, sigma_n, nobs));
    
  theta ~ normal(0, 10);
  kappa ~ normal(0, 10);
  sigma_n ~ normal(0, 10);
}
