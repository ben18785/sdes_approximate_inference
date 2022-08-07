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
    // adapt this to match a given SDE
    dXdt[1] = -theta * X + kappa * z_sum(Zs, t, T, one_to_N);
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
  
  // posterior predictive sims
  int Nzsim;
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
    Y[i] ~ normal(X_sim[i], sigma_n);
  
  Z ~ normal(0, 1);
  theta ~ normal(0, 10);
  kappa ~ normal(kappa_mean, kappa_width);
  sigma_n ~ normal(0, 10);
}

generated quantities {
  int d_i_sim[1];
  real d_r_sim[Nzsim];
  real params_sim[3 + Nzsim];
  real X_sim_1[nobs, 1];
  real X_sim_n[nobs, 1];
  d_i_sim[1] = Nzsim; 
  for(i in 1:Nzsim)
    d_r_sim[i] = i;
  params_sim[1] = theta;
  params_sim[2] = kappa;
  params_sim[3] = T;
  for(i in 1:Nzsim)
    params_sim[3 + i] = normal_rng(0, 1);
  X_sim_1 = integrate_ode_rk45(random_rhs, {X_0}, 0.0, t, params_sim, d_r_sim, d_i_sim);
  for(i in 1:nobs) {
    X_sim_n[i, 1] = normal_rng(X_sim[i, 1], sigma_n);
  }
}
