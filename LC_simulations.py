import numpy as np 
import matplotlib.pyplot as plt 
import random
import cmath
from numpy.fft import fft, ifft
from scipy import stats
import math
from .PSD_calc import *

def gen_random_numbers_normaldistr(N_numbers):

	mu = 0
	sigma = 1

	s = np.random.normal(mu, sigma, N_numbers)

	return s


def gen_fourier_coeff(freq, PSD_index, sum_flux=1e5):

		factor = (1/freq)**(PSD_index/2)

		a = gen_random_numbers_normaldistr(len(freq)) * factor
		b = gen_random_numbers_normaldistr(len(freq)) * factor

		coeff = a + 1j * b
		coeff = np.append(np.conj(coeff), np.flipud(coeff))#to make it real, f(-w)=f*(w)
		
		#add int to make it real
		coeff = np.append(sum_flux, coeff) 
		return coeff



class lightcurve:


	def __init__(self, data=[], mjd_column=0, flux_column=1, flux_error_column=2):

		if not len(data):
			self.data = data
			print ('Attention, no observational data was given!')
		
		else:

			self.data = data
			self.mjd_data = data[:,mjd_column]
			self.data_time_span = math.ceil(max(self.mjd_data)-min(self.mjd_data))

			self.flux_LC_data = data[:,flux_column]
			self.flux_error_LC_data = data[:,flux_error_column]
			
			self.mean_LC_data = np.mean(data[:,flux_column])
			self.std_LC_data = np.std(data[:,flux_column])



	def produce_sampling_pattern(self, LC_output_t_bin, LC_sim_time_span):

		if not len(self.data):
			print ('An observed lightcurve is needed to compute an sampled light curve')
			return 0


		mjd_data = np.sort(self.mjd_data)

		sim_T_bins = np.linspace(min(mjd_data), min(mjd_data)+LC_sim_time_span, self.sim_LC_Npoints)

		pattern = np.full(len(sim_T_bins), False, dtype=bool)

		for t_sim in range(len(sim_T_bins)):
			a = (sim_T_bins[t_sim]-mjd_data)
				
			if min(abs(a)) < LC_output_t_bin/2 :
				pattern[t_sim] = True

		return pattern




	def simulate_LC(self, N_sim_LC, PSD_index, LC_sim_time_span, N_LC_sim_length_mult, LC_sim_time_precision, LC_output_t_bin, normalize_sim_LC=False, sample_sim_LC=False):

		#Following Timmer & Koenig, 1995, Astronomy & Astrophysics, 300, 707
		#everything is in unit of [day]

		#check int value for N_LC_sim_length_mult
		if type(N_LC_sim_length_mult) != int:
			print ('Please enter a integer value for N_LC_sim_length_mult')
			return (0)

		LC_sim_length = LC_sim_time_span / LC_sim_time_precision
		LC_sim_length *= N_LC_sim_length_mult

		freq = np.arange(1, LC_sim_length/2 + 1) / (LC_sim_length * LC_sim_time_precision)

		LC_sim_flux = []

		for i in range(N_sim_LC):


			fourier_coeffs = gen_fourier_coeff(freq, PSD_index)

			#inverse fourier, produce simulated LC
			full_LC = ifft(fourier_coeffs).real
			
			#cut LC to desired length
			if N_LC_sim_length_mult == 1 :

				cut_LC = full_LC

				#bin LC to desired bin width
				sim_t_slices = np.arange(0, len(full_LC), 1) * LC_sim_time_precision

				cut_LC_binned = stats.binned_statistic(sim_t_slices, cut_LC, 'mean', bins=(len(cut_LC) * LC_sim_time_precision) / LC_output_t_bin)[0]

			else :

				cut = random.randint(0, LC_sim_length/2)

				cut_LC = full_LC[cut:int(cut+LC_sim_length/N_LC_sim_length_mult)+1]

				#bin LC to desired bin width
				sim_t_slices = np.arange(0, len(cut_LC), 1) * LC_sim_time_precision

				cut_LC_binned = stats.binned_statistic(sim_t_slices, cut_LC, 'mean', bins=(len(cut_LC) * LC_sim_time_precision) / LC_output_t_bin)[0]


			self.sim_LC_Npoints = len(cut_LC_binned)

			
			
			if not len(self.data):
				#simply add non-sampled, non normalized LC if no obs. data is given
				LC_sim_flux.append(cut_LC_binned)	

				print ('Lightcurve ', i+1, ' out of ', N_sim_LC, ' simulated!')
				
				if normalize_sim_LC==True:
					print ('An observed lightcurve is needed to normalize the simulated light curve')
					return 0



			else:
				#sample and normalize LC

				self.norm_factor = np.sqrt( (self.std_LC_data**2-np.mean(self.flux_error_LC_data)**2)/np.std(cut_LC_binned)**2 )


				if sample_sim_LC==False:
					
					T_bins_sim_LC_sampled = np.linspace(min(self.mjd_data), min(self.mjd_data)+LC_sim_time_span, self.sim_LC_Npoints)

					LC_sim_flux_sampled = cut_LC_binned


				if sample_sim_LC==True:
					
					#sample LC
					sampling_pattern = self.produce_sampling_pattern(LC_output_t_bin, LC_sim_time_span)

					T_bins_sim_LC_sampled = np.linspace(min(self.mjd_data), min(self.mjd_data)+LC_sim_time_span, self.sim_LC_Npoints)[sampling_pattern]
					
					LC_sim_flux_sampled = cut_LC_binned[sampling_pattern]

				#add Gaussian Noise, following errorbar of observations

				LC_sim_flux_sampled = np.random.normal(LC_sim_flux_sampled, self.norm_factor**-1 * self.flux_error_LC_data, len(LC_sim_flux_sampled))
				

				#normalize LC
				if normalize_sim_LC==True:

					LC_sim_flux_sampled = (LC_sim_flux_sampled-np.mean(cut_LC_binned))*self.norm_factor + self.mean_LC_data
					

				#append LC to LC list
				LC_sim_flux.append(LC_sim_flux_sampled)	

				print ('Lightcurve ', i+1, ' out of ', N_sim_LC, ' simulated!')
			
			
			


		if not len(self.data):
			return LC_sim_flux

		else:	
			return (T_bins_sim_LC_sampled, LC_sim_flux)
			





	def simulate_realistic_LC(self, N_sim_LC, PSD_index, N_LC_sim_length_mult, LC_sim_time_precision, LC_output_t_bin):

		if not len(self.data):
			print ('An observed lightcurve is needed to compute an sampled light curve')
			return 0

		LC_sim_flux_sampled = []
		T_bins_sim_LC_sampled = []

		T_bins_sim_LC_sampled, LC_sim_flux_sampled = self.simulate_LC(N_sim_LC, PSD_index, self.data_time_span, N_LC_sim_length_mult, LC_sim_time_precision, LC_output_t_bin, normalize_sim_LC=True, sample_sim_LC=True)


		return (T_bins_sim_LC_sampled, LC_sim_flux_sampled)





	def estimate_PSD(self, N_sim_LC, N_LC_sim_length_mult, LC_sim_time_precision, LC_output_t_bin, output_fig_name='SuF_vs_pwlindex.pdf', true_beta_LC_mjd=None, true_beta_LC_flux=None):

		#beta = np.arange(0.7,2.1,0.05)
		beta = np.arange(0.7,2.1,0.1)#for PSD uncertainty estimation
		suf_list = []

		
		for beta_i in beta:
			print ('Working on beta = ', beta_i, '...')

			sim_LCs = self.simulate_realistic_LC(N_sim_LC, beta_i, N_LC_sim_length_mult, LC_sim_time_precision, LC_output_t_bin)
			
			freq, sim_PSDs = calc_sim_PSD(sim_LCs)
			
			if np.shape(true_beta_LC_mjd) and np.shape(true_beta_LC_flux) :#in case one wants to estimate uncertainty
				
				freq, obs_PSD = calc_obs_PSD(true_beta_LC_mjd, true_beta_LC_flux)

			else:#in case one wants to estimate PSD from real data 

				freq, obs_PSD = calc_obs_PSD(self.mjd_data, self.flux_LC_data)

			chisquare_obs = calc_chisquare_PSD(obs_PSD, sim_PSDs)
			
			suf = 0
			for i in range(len(sim_PSDs)):

				chisquare_sim_1 = calc_chisquare_PSD(sim_PSDs[i], sim_PSDs)
				if chisquare_sim_1>chisquare_obs:
					suf+=1

			suf = suf/len(sim_PSDs)
			suf_list.append(suf)


		plt.figure()
		plt.plot(beta, suf_list, 'r-')

		plt.xlabel(r'$\beta$')
		plt.ylabel('SuF')
		plt.savefig(output_fig_name)
		
		best_beta = beta[np.argmax(suf_list)]
	
		return best_beta


	def estimate_PSD_uncertainty(self, true_beta, N_sim_LC, N_LC_sim_length_mult, LC_sim_time_precision, LC_output_t_bin, fig_name):
		#estimate the PSD uncertainties with a Neyman construction

		fitted_beta = []
		n_fit = 100

		
		#estimate PSD for a large number of fits
		for i in range(n_fit):
			
			#create LC with known PSD index
			true_beta_sim_LC = self.simulate_realistic_LC(1, true_beta, N_LC_sim_length_mult, LC_sim_time_precision, LC_output_t_bin)
			true_beta_mjd_sim = true_beta_sim_LC[0]
			true_beta_flux_sim = true_beta_sim_LC[1][0]

			best_fit_beta = self.estimate_PSD(N_sim_LC, N_LC_sim_length_mult, LC_sim_time_precision, LC_output_t_bin, output_fig_name=fig_name+str(i)+'.png', true_beta_LC_mjd=true_beta_mjd_sim, true_beta_LC_flux=true_beta_flux_sim)
			fitted_beta.append(best_fit_beta)
		
		#get uncertainty bands
		uncertainty_band = [np.percentile(fitted_beta, 100-68), np.percentile(fitted_beta, 68)]
		
		plt.figure()
		plt.hist(fitted_beta, bins=20)
		plt.xlabel(r'$\beta_{fit}$')
		plt.ylabel('Counts')
		plt.title(r'$\beta_{true}=$ '+str(true_beta))
		plt.savefig(str(true_beta)+'.pdf')

		
		return true_beta, uncertainty_band 