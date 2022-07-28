fit_model_any_z <- function(N_zs, model, init_fn, refresh_val=0, iters=400) {
  stan_data <- list(
    nobs=length(X),
    t=t,
    X=X,
    X_0=X_0,
    N=N_zs,
    T=T,
    sigma_mean=0,
    sigma_width=10)
  fit <- sampling(model,
                  data=stan_data,
                  iter=iters, chains=4, init=init_fn,
                  refresh=refresh_val)
  fit
}
