functions {
  
  vector phi(real t, vector i, real T) {
    return((2 / T)^0.5 * cos((2 * i - 1) * pi() * t / (2 * T)));
  }
  vector z_sum(matrix Zs, real t, real T, vector one_to_N) {
    return(Zs * phi(t, one_to_N, T));
  }
  
  real[] random_rhs(real t,
                   real[] y,
                   real[] params,
                   real[] d_r,
                   int[] d_i
  ) {
    int N = d_i[1];
    vector[N] one_to_N = to_vector(d_r);
    real q = params[1];
    real T = params[2];
    vector[N] Zs_all = to_vector(params[3:]);
    vector[2] X = [y[1], y[2]]';
    matrix[2, 2] L;
    real dXdt[2];
    vector[2] dXdt_vec;
    matrix[2, 2] F;
    matrix [2, N] Zs;
    Zs[1, :] = rep_row_vector(0.0, N); // redundant Z terms
    Zs[2, :] = to_row_vector(Zs_all);
    L[1, 1] = 0.0;
    L[1, 2] = 0.0;
    L[2, 1] = 0.0;
    L[2, 2] = sqrt(q);
    F[1, 1] = 0.0;
    F[1, 2] = 1.0;
    F[2, 1] = 0.0;
    F[2, 2] = 0.0;
    dXdt_vec = F * X + L * z_sum(Zs, t, T, one_to_N);
    dXdt = {dXdt_vec[1], dXdt_vec[2]};
    return(dXdt);
  }
}

data {
  int nobs;
  real t[nobs];
  vector[nobs] Y;
  real X_1_0;
  real X_2_0;
  int N;
  real T;
}

transformed data {
  int d_i[1];
  real d_r[N];
  d_i[1] = N; 
  for(i in 1:N)
    d_r[i] = i;
}

parameters {
  real<lower=0> q;
  real Z[N];
  real<lower=0> sigma_n;
}

transformed parameters {
  real params[2 + N];
  real X_sim[nobs, 2];
  params[1] = q;
  params[2] = T;
  for(i in 1:N)
    params[2 + i] = Z[i];
    
  X_sim = integrate_ode_rk45(random_rhs, {X_1_0, X_2_0}, 0.0, t, params, d_r, d_i);
}

model {
  for(i in 1:nobs)
    Y[i] ~ normal(X_sim[i, 1], sigma_n);
  
  Z ~ normal(0, 1);
  q ~ normal(0, 10);
  sigma_n ~ normal(0, 10);
}

generated quantities {
  vector[nobs] loglikelihood;
  for(i in 1:nobs)
    loglikelihood[i] = normal_lpdf(Y[i]|X_sim[i], sigma_n);
}
