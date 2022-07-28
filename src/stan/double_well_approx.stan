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
    real alpha = params[1];
    real gamma = params[2];
    real kappa = params[3];
    real T = params[4];
    vector[N] Zs = to_vector(params[5:]);
    real X = y[1];
    real dXdt[1];
    // adapt this to match a given SDE
    dXdt[1] = alpha * X * (gamma^2 - X^2) + kappa * z_sum(Zs, t, T, one_to_N);
    return(dXdt);
  }
}

data {
  int nobs;
  real t[nobs];
  vector[nobs] Y;
  real X_0;
  int N;
  real T;
  
  // prior on kappa
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
  real<lower=0> alpha;
  real<lower=0> gamma;
  real<lower=0> kappa;
  real Z[N];
  real<lower=0> sigma_n;
}

transformed parameters {
  real params[4 + N];
  real X_sim[nobs, 1];
  params[1] = alpha;
  params[2] = gamma;
  params[3] = kappa;
  params[4] = T;
  for(i in 1:N)
    params[4 + i] = Z[i];
    
  X_sim = integrate_ode_rk45(random_rhs, {X_0}, 0.0, t, params, d_r, d_i);
}

model {
  for(i in 1:nobs)
    Y[i] ~ normal(X_sim[i], sigma_n);
  
  Z ~ normal(0, 1);
  alpha ~ normal(0, 10);
  gamma ~ normal(0, 10);
  kappa ~ normal(kappa_mean, kappa_width);
  sigma_n ~ normal(0, 10);
}

generated quantities {
  vector[nobs] loglikelihood;
  for(i in 1:nobs)
    loglikelihood[i] = normal_lpdf(Y[i]|X_sim[i], sigma_n);
}
