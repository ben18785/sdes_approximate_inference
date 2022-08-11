functions {
vector phi(real t, vector i, real T) {
  return((2 / T)^0.5 * cos((2 * i - 1) * pi() * t / (2 * T)));
}
real z_sum(vector Zs, real t, real T, vector one_to_N) {
  return(sum(Zs .* phi(t, one_to_N, T)));
}
  
  real[] random_rhs(real t,
                   real[] y,
                   real[] params,
                   real[] d_r,
                   int[] d_i
  ) {
    int N = d_i[1];
    vector[N] one_to_N = to_vector(d_r);
    real kappa = params[1];
    real T = params[2];
    vector[N] Zs = to_vector(params[3:]);
    real X = y[1];
    real dXdt[1];
    // adapt this to match a given SDE
    dXdt[1] = (1.0 - 0.5 * kappa^2) + kappa * sqrt(X) * z_sum(Zs, t, T, one_to_N); // note have used transform to make into Stratonovich
    return(dXdt);
  }
  
  real [, ] int_x (int nobs, real X_0, real [] t, real [] params, real[] d_r,
                   int[] d_i) {
    real X_sim[nobs, 1];
    X_sim = integrate_ode_rk45(random_rhs, {X_0}, 0.0, t, params, d_r, d_i);
    return(X_sim);
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
  real<lower=0> kappa;
  real Z[N];
  real<lower=0> sigma_n;
}

transformed parameters {
  real params[2 + N];
  real X_sim[nobs, 1];
  params[1] = kappa;
  params[2] = T;
  for(i in 1:N)
    params[2 + i] = Z[i];
    
  X_sim = integrate_ode_rk45(random_rhs, {X_0}, 0.0, t, params, d_r, d_i);
}

model {
  for(i in 1:nobs)
    Y[i] ~ normal(X_sim[i], sigma_n);
  
  Z ~ normal(0, 1);
  kappa ~ normal(kappa_mean, kappa_width);
  sigma_n ~ normal(0, 10);
}

generated quantities {
  vector[nobs] loglikelihood;
  for(i in 1:nobs)
    loglikelihood[i] = normal_lpdf(Y[i]|X_sim[i], sigma_n);
}
