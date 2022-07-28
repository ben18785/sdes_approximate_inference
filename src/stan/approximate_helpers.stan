vector phi(real t, vector i, real T) {
  return((2 / T)^0.5 * cos((2 * i - 1) * pi() * t / (2 * T)));
}
real z_sum(vector Zs, real t, real T, vector one_to_N) {
  return(sum(Zs .* phi(t, one_to_N, T)));
}