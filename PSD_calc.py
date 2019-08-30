import numpy as np 




def calc_sim_PSD(LC):

	t = LC[0]
	T = max(t)-min(t)
	N = len(t)
	
	LCs_flux = LC[1]

	freq = np.arange(1,  N/2 ) / T
	PSD = []

	for i in range(len(LCs_flux)):

		flux = LCs_flux[i]
		PSDj = []
		for j in range(len(freq)):

			F = (np.sum(flux*np.cos(2*np.pi*freq[j]*t)))**2 + (np.sum(flux*np.sin(2*np.pi*freq[j]*t)))**2
			
			#normalize F
			PSDj.append(F * (2*T)/(np.mean(flux)**2 * N**2))

		PSD.append(PSDj)	

	
	return freq, np.array(PSD)



def calc_obs_PSD(obs_mjd, obs_flux):

	t = obs_mjd
	T = max(t)-min(t)
	N = len(t)
	
	LC_flux = obs_flux

	freq = np.arange(1,  N/2 ) / T
	PSD = []


	PSD = []
	for j in range(len(freq)):

		F = (np.sum(LC_flux*np.cos(2*np.pi*freq[j]*t)))**2 + (np.sum(LC_flux*np.sin(2*np.pi*freq[j]*t)))**2
			
		#normalize F
		PSD.append(F * (2*T)/(np.mean(LC_flux)**2 * N**2))

	
	return freq, np.array(PSD)






def calc_chisquare_PSD(PSD_1, PSD_sim):

	mean_PSD_sim = []
	std_PSD_sim = []
	
	for i in range(len(PSD_sim[0])):

		mean_PSD_sim.append(np.mean(PSD_sim[:,i]))
		std_PSD_sim.append(np.std(PSD_sim[:,i]))

	mean_PSD_sim = np.array(mean_PSD_sim)
	std_PSD_sim = np.array(mean_PSD_sim)

	chisquare = (PSD_1-mean_PSD_sim)**2/(std_PSD_sim)**2

	chisquare = np.sum(chisquare)

	return chisquare