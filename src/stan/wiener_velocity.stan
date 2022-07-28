functions {
  real v_k_f(real y_k, row_vector H, vector m_k_bar) {
    return(y_k - H * m_k_bar);
  }
  vector m_k_bar_f(matrix A, vector m_k) {
    return(A * m_k);
  }
  real S_k_f(row_vector H, matrix P_k_bar, real sigma) {
    return(H * P_k_bar * H' + sigma^2);
  }
  matrix P_k_bar_f(matrix A, matrix P_k, matrix Sigma) {
    return(A * P_k * A' + Sigma);
  }
  matrix P_k_f(matrix P_k_bar, vector K_k, real S_k) {
    return(P_k_bar - K_k * S_k * K_k');
  }
  vector K_k_f(matrix P_k_bar, row_vector H, real S_k) {
    return(P_k_bar * H' * S_k^-1);
  }
  vector m_k_f(vector m_k_bar, vector K_k, real v_k) {
    return(m_k_bar + K_k * v_k);
  }
  real mu_f(row_vector H, vector m_k_bar) {
    return(H * m_k_bar);
  }
}

data {
  int nobs;
  vector[nobs] t;
  vector[nobs] Y;
  real X_1_0;
  real X_2_0;
  real delta_t;
}

transformed data{
  matrix[2, 2] A;
  row_vector[2] H;
  matrix[2, 2] Sigma_const;
  A[1, 1] = 1;
  A[1, 2] = delta_t;
  A[2, 1] = 0.0;
  A[2, 2] = 1.0;
  H[1] = 1.0;
  H[2] = 0.0;
  Sigma_const[1, 1] = (1.0 / 3.0) * delta_t^3;
  Sigma_const[1, 2] = (1.0 / 2.0) * delta_t^2;
  Sigma_const[2, 1] = (1.0 / 2.0) * delta_t^2;
  Sigma_const[2, 2] = delta_t;
}

parameters {
  real<lower=0> q;
  real<lower=0> sigma_n;
}

model {
  vector[2] m_k = [X_1_0, X_2_0]';
  matrix[2, 2] P_k;
  vector[2] m_k_bar;
  matrix[2, 2] P_k_bar;
  real v_k;
  real S_k;
  vector[2] K_k;
  real mu;
  matrix[2, 2] Sigma = q * Sigma_const;
  P_k[1, 1] = 1.0;
  P_k[1, 2] = 0.0;
  P_k[2, 1] = 0.0;
  P_k[2, 2] = 1.0;
  
  for(i in 2:nobs) {
    
    // prediction step
    m_k_bar = m_k_bar_f(A, m_k);
    P_k_bar = P_k_bar_f(A, P_k, Sigma);
    
    // update step
    v_k = v_k_f(Y[i], H, m_k_bar);
    S_k = S_k_f(H, P_k_bar, sigma_n);
    K_k = K_k_f(P_k_bar, H, S_k);
    m_k = m_k_f(m_k_bar, K_k, v_k);
    P_k = P_k_f(P_k_bar, K_k, S_k);
    
    mu = mu_f(H, m_k_bar);
    Y[i] ~ normal(mu, S_k);
  }

  sigma_n ~ normal(0, 10);
  q ~ normal(0, 10);
}
