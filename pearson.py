import numpy as np
from scipy.stats.stats import pearsonr


def pearson_coeff(LC1, LC2, delta_t):
	

	t_LC1 = LC1[0]
	f_LC1 = LC1[1]

	t_LC2 = LC2[0]
	f_LC2 = LC2[1]

	min_delta_LC1 = np.min(abs(t_LC1[1:]-t_LC1[:-1]))
	min_delta_LC2 = np.min(abs(t_LC2[1:]-t_LC2[:-1]))

	if delta_t>min_delta_LC1 and delta_t>min_delta_LC2:
		raise Exception('ERROR, simultaneous window larger than the minimal sampling rate of two LCs')		
		

	if min_delta_LC1<min_delta_LC2:
		raise Exception("LC2 sampling must be smaller than LC1")
		

	t_simult_1 = []
	f_simult_1 = []

	t_simult_2 = []
	f_simult_2 = []

	for i in range(len(t_LC1)):
		mjd = []
		flux = []

		for j in range(len(t_LC2)):
			if (abs(t_LC2[j]-t_LC1[i])<delta_t/2):
				
				mjd.append(t_LC2[j])
				flux.append(f_LC2[j])


		if len(flux):
			flux_2_averaged = np.average(flux)

			t_simult_1.append(t_LC1[i])
			f_simult_1.append(f_LC1[i])

			t_simult_2.append(np.mean(mjd))
			f_simult_2.append(flux_2_averaged)


	t_simult_1 = np.asarray(t_simult_1)
	f_simult_1 = np.asarray(f_simult_1)

	t_simult_2 = np.asarray(t_simult_2)
	f_simult_2 = np.asarray(f_simult_2)
	
	r = pearsonr(f_simult_1, f_simult_2)[0]


	return r

