import numpy as np



def DCF(t, mjd_1, mjd_2, flux_1, flux_2, delta_t,sigma=False):

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

	sigma_DCF = np.sqrt(np.sum((UDCF-DCF)**2))/(len(UDCF)-1)

	if sigma:
		return (DCF, sigma_DCF)
	else:
		return (DCF)



def calc_DCF_LC(LC1, LC2, t_min, t_max, delta_t,sigma=False):

	N_LC_simulated = len(LC1[1])

	if (len(LC1[1])!=len(LC2[1])):
		return 'ERROR, Number of LC simulated not equal'

	
	t_LC1 = LC1[0]
	f_LC1 = LC1[1]

	t_LC2 = LC2[0]
	f_LC2 = LC2[1]

	t_lag = np.arange(t_min, t_max, delta_t)
	DCFs = []
	DCFs_err = []

	for i in range(N_LC_simulated):
		dcfi = []
		dcfi_err= []
		
		for lag in t_lag:

			if sigma:
				dcfi.append((DCF(lag, t_LC1, t_LC2, f_LC1[i], f_LC2[i], delta_t,sigma)[0]))
				dcfi_err.append((DCF(lag, t_LC1, t_LC2, f_LC1[i], f_LC2[i], delta_t,sigma)[1]))
			else:
				dcfi.append((DCF(lag, t_LC1, t_LC2, f_LC1[i], f_LC2[i], delta_t)))

		DCFs.append(dcfi)
		DCFs_err.append(dcfi_err)
		
		if (i+1)%100==0:
			print ('DCF of LC pair ', i+1, ' out of' , N_LC_simulated, ' computed!')
	
	if sigma:
		return (t_lag, np.array(DCFs), np.array(DCFs_err))
	else:
		return (t_lag, np.array(DCFs))