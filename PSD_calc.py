import numpy as np 
from scipy import stats
import matplotlib.pyplot as plt



def calc_sim_PSD(LC, mjd_data):

	t = mjd_data
	t_f = 1#2.
	T = max(t)-min(t)
	N = len(t)
	
	LCs_flux = LC[1]

	#freq = np.arange(1,  N/2 ) / T
	freq = np.arange(1, T/(2*t_f)) / T

	PSD = []

	for i in range(len(LCs_flux)):

		flux = LCs_flux[i]
		PSDj = []
		for j in range(len(freq)):

			F = (np.sum(flux*np.cos(2*np.pi*freq[j]*t)))**2 + (np.sum(flux*np.sin(2*np.pi*freq[j]*t)))**2
			
			#normalize F
			PSDj.append(F * (2*T)/(N**2))
		
		#logbins
		#psd_log_binned, freq_log_edges, _ = stats.binned_statistic(freq, PSDj, 'mean', bins=np.logspace(np.log10(freq[0]), np.log10(freq[-1]), 1+(np.log10(freq[-1])-np.log10(freq[0]))/np.log10(1.5)))
		#freq_log = 10**((np.log10(freq_log_edges[1:])+np.log10(freq_log_edges[:-1]))/2.)
		#linbins
		psd_log_binned, freq_log_edges, _ = stats.binned_statistic(freq, PSDj, 'mean', bins=np.linspace(freq[0], freq[-1], 12))
		freq_log = ((freq_log_edges[1:]+freq_log_edges[:-1])/2.)

		#PSD.append(PSDj)	
		PSD.append(psd_log_binned)
	
	return freq_log, np.array(PSD)



def calc_obs_PSD(obs_mjd, obs_flux, PSD_bin_number):
	
	
	t = obs_mjd
	t_f = 1#2.

	T = max(t)-min(t)
	N = len(t)
	
	LC_flux = obs_flux

	#freq = np.arange(1,  N/2 ) / T
	freq = np.arange(1, N/(2*t_f)) / T

	PSD = []

	for j in range(len(freq)):

		F = (np.sum(LC_flux*np.cos(2*np.pi*freq[j]*t)))**2 + (np.sum(LC_flux*np.sin(2*np.pi*freq[j]*t)))**2
			
		#normalize F
		PSD.append(F * (2*T)/(N**2))
	#logbins
	#psd_log_binned, freq_log_edges, _ = stats.binned_statistic(freq, PSD, 'mean', bins=np.logspace(np.log10(freq[0]), np.log10(freq[-1]), 1+(np.log10(freq[-1])-np.log10(freq[0]))/np.log10(1.5)))
	#freq_log = 10**((np.log10(freq_log_edges[1:])+np.log10(freq_log_edges[:-1]))/2.)
	#linbins
	

	psd_log_binned, freq_log_edges, _ = stats.binned_statistic(freq, PSD, 'mean', bins=np.linspace(freq[0], freq[-1], PSD_bin_number))
	freq_log = ((freq_log_edges[1:]+freq_log_edges[:-1])/2.)


	#plt.loglog(freq_log, psd_log_binned)
	#plt.show()
	#return freq, np.array(PSD)
	return freq_log, psd_log_binned






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
