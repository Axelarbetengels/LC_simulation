import numpy as np 


def get_significance_band(t_lag, DCFs, significance_level):

	band_up = []
	band_down = []

	for i in range(len(t_lag)):
		
		band_up.append(np.percentile(DCFs[:,i], significance_level))
		band_down.append(np.percentile(DCFs[:,i], 100-significance_level))


	return (t_lag, band_up, band_down)