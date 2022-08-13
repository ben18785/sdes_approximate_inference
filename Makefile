
figures/gbm_fits_vs_actual.pdf: src/geometric_brownian_motion.Rmd
	Rscript -e "rmarkdown::render('src/geometric_brownian_motion.Rmd')"