fit_model_any_z <- function(N_zs, model, init_fn_list, base_stan_data, refresh_val=0, iters=400, ndims=1) {
  init_fn_list$Z <- as.array(rep(1, ndims * N_zs))
  init_fn1 <- function() {
    init_fn_list
  }
  stan_data <- base_stan_data
  stan_data$N <- N_zs
  fit <- sampling(model,
                  data=stan_data,
                  iter=iters, chains=4, init=init_fn1,
                  refresh=refresh_val)
  fit
}
