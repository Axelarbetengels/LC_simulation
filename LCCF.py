import numpy as np

def LCCF(t, mjd_1, mjd_2, flux_1, flux_2, delta_t):


	a = []
	b = []

	for i in range(len(flux_1)):
		
		for j in range(len(flux_2)):

			if (mjd_1[i]-mjd_2[j])<t+(delta_t/2) and (mjd_1[i]-mjd_2[j])>t-(delta_t/2) :
				
				a.append(flux_1[i])
				b.append(flux_2[j])

	a = np.array(a)
	b = np.array(b)
	
	mean_a = np.mean(a)		
	mean_b = np.mean(b)		

	sigma_a = np.std(a)
	sigma_b = np.std(b)

	ULCCF = ((a-mean_a)*(b-mean_b)/np.sqrt((sigma_a**2)*(sigma_b**2)))

	ULCCF = np.array(ULCCF)
	LCCF = np.sum(ULCCF)/len(ULCCF)


	return (LCCF)



def calc_LCCF_LC(LC1, LC2, t_min, t_max, delta_t):

	N_LC_simulated = len(LC1[1])
	
	if (len(LC1[0])!=len(LC2[0])):
		return 'ERROR, Number of LC simulated not equal'

		
	t_LC1 = LC1[0]
	f_LC1 = LC1[1]

	t_LC2 = LC2[0]
	f_LC2 = LC2[1]

	t_lag = np.arange(t_min, t_max, delta_t)
	LCCFs = []

	for i in range(N_LC_simulated):
		lccfi = []
		
		for lag in t_lag:

			lccfi.append((LCCF(lag, t_LC1, t_LC2, f_LC1[i], f_LC2[i], delta_t)))

		LCCFs.append(lccfi)
		print ('LCCF of LC pair ', i+1, ' out of' , N_LC_simulated, ' computed!')
	
	return (t_lag, np.array(LCCFs))

