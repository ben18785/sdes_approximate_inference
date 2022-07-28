functions {
#include /approximate_helpers.stan
  
  real[] random_rhs(real t,
                   real[] y,
                   real[] params,
                   real[] d_r,
                   int[] d_i
  ) {
    int N = d_i[1];
    vector[N] one_to_N = to_vector(d_r);
    real theta = params[1];
    real kappa = params[2];
    real T = params[3];
    vector[N] Zs = to_vector(params[4:]);
    real X = y[1];
    real dXdt[1];
    dXdt[1] = theta * X - 0.5 * kappa^2 * X + kappa * X * z_sum(Zs, t, T, one_to_N);
    return(dXdt);
  }
}

data {
  int nobs;
  real t[nobs];
  vector[nobs] X;
  real X_0;
  int N;
  real T;
  
  // prior on sigma
  real kappa_width;
  real kappa_mean;
}

transformed data {
  int d_i[1];
  real d_r[N];
  d_i[1] = N; 
  for(i in 1:N)
    d_r[i] = i;
}

parameters {
  real theta;
  real<lower=0> kappa;
  real Z[N];
  real<lower=0> sigma_n;
}

transformed parameters {
  real params[3 + N];
  real X_sim[nobs, 1];
  params[1] = theta;
  params[2] = kappa;
  params[3] = T;
  for(i in 1:N)
    params[3 + i] = Z[i];
  
  X_sim = integrate_ode_rk45(random_rhs, {X_0}, 0.0, t, params, d_r, d_i);
}

model {
  
  for(i in 1:nobs)
    X[i] ~ lognormal(log(X_sim[i,1]) - 0.5 * sigma_n^2, sigma_n);
  
  Z ~ normal(0, 1);
  theta ~ normal(0, 10);
  kappa ~ normal(kappa_mean, kappa_width);
  sigma_n ~ normal(0, 10);
}

generated quantities {
  vector[nobs] loglikelihood;
  for(i in 1:nobs)
    loglikelihood[i] = lognormal_lpdf(X[i]|log(X_sim[i,1]) - 0.5 * sigma_n^2, sigma_n);
}
