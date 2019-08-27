import numpy as np



def DCF(t, mjd_1, mjd_2, flux_1, flux_2, delta_t):

	UDCF = []

	mean_a = np.mean(flux_1)
	mean_b = np.mean(flux_2)

	sigma_a = np.std(flux_1)
	sigma_b = np.std(flux_2)


	for i in range(len(flux_1)):
		
		for j in range(len(flux_2)):

			if (mjd_1[i]-mjd_2[j])<t+(delta_t/2) and (mjd_1[i]-mjd_2[j])>t-(delta_t/2) :
				
				a = flux_1[i]
				b = flux_2[j]

				#e_a = flux_1_errors[i]
				#e_b = flux_2_errors[j]
				
				UDCF.append((a-mean_a)*(b-mean_b)/np.sqrt((sigma_a**2)*(sigma_b**2)))

	UDCF = np.array(UDCF)
	DCF = np.sum(UDCF)/len(UDCF)


	return (DCF)



def calc_DCF_LC(LC1, LC2, t_min, t_max, delta_t):

	N_LC_simulated = len(LC1[0])
	
	if (len(LC1[0])!=len(LC2[0])):
		return 'ERROR, Number of LC simulated not equal'

		
	t_LC1 = LC1[0]
	f_LC1 = LC1[1]

	t_LC2 = LC2[0]
	f_LC2 = LC2[1]

	t_lag = np.arange(t_min, t_max, delta_t)
	DCFs = []

	for i in range(N_LC_simulated):
		dcfi = []
		
		for lag in t_lag:

			dcfi.append((DCF(lag, t_LC1[i], t_LC2[i], f_LC1[i], f_LC2[i], delta_t)))

		DCFs.append(dcfi)
		print ('DCF of LC pair ', i+1, ' out of' , N_LC_simulated, ' computed!')
	
	return (t_lag, np.array(DCFs))

