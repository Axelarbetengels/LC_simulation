import numpy as np 
import matplotlib.pyplot as plt 
import random
import cmath
from numpy.fft import fft, ifft



def gen_random_numbers_normaldistr(N_numbers):

	mu = 0
	sigma = 1

	s = np.random.normal(mu, sigma, N_numbers)

	return s


def gen_fourier_coeff(freq, PSD_index, sum_flux=1e3):

		factor = (1/freq)**(PSD_index/2)

		a = gen_random_numbers_normaldistr(len(freq)) * factor
		b = gen_random_numbers_normaldistr(len(freq)) * factor

		coeff = a + 1j * b
		coeff = np.append(np.conj(coeff), np.flipud(coeff))#to make it real, f(-w)=f*(w)
		
		#add int to make it real
		coeff = np.append(sum_flux, coeff) 
		return coeff



class lightcurve:


	def __init__(self, data):

		self.data = data
		self.mjd_data = data[:,0]
		self.data_time_span = round(max(self.mjd_data)-min(self.mjd_data))

	

	def simulate_LC(self, N_sim_LC, PSD_index, LC_sim_time_span, N_LC_sim_length_mult, LC_sim_time_precision):
		
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

			#inverse fourier
			LC = ifft(fourier_coeffs).real

			if N_LC_sim_length_mult == 1 :#cut LC to desired length

				LC_sim_flux.append(LC)
			
			else :

				cut = random.randint(0, LC_sim_length/2)

				LC = LC[cut:int(cut+LC_sim_length/N_LC_sim_length_mult)]
				LC_sim_flux.append(LC)

			print ('Lightcurve ', i+1, ' out of ', N_sim_LC, ' simulated!')
		

		return LC_sim_flux



	def produce_sampling_pattern(self, LC_sim_time_precision, sim_LC_Npoints):

		mjd_data = np.sort(self.mjd_data)

		sim_T_bins = np.linspace(min(mjd_data), max(mjd_data), sim_LC_Npoints)

		pattern = np.full(len(sim_T_bins), False, dtype=bool)

		for t_sim in range(len(sim_T_bins)):
			a = (sim_T_bins[t_sim]-mjd_data)
				
			if min(abs(a)) < LC_sim_time_precision/2 :
				pattern[t_sim] = True

		return pattern



	def simulate_LC_sampled(self, N_sim_LC, PSD_index, N_LC_sim_length_mult, LC_sim_time_precision):

		LC_sim_flux_sampled = []
		T_bins_sim_LC_sampled = []

		LC_sim_flux = self.simulate_LC(N_sim_LC, PSD_index, self.data_time_span, N_LC_sim_length_mult, LC_sim_time_precision)

		sim_LC_Npoints = len(LC_sim_flux[0])
		sampling_pattern = self.produce_sampling_pattern(LC_sim_time_precision, sim_LC_Npoints)

		for i in range(N_sim_LC):

			T_bins_sim_LC_sampled.append(np.linspace(min(self.mjd_data), max(self.mjd_data), sim_LC_Npoints)[sampling_pattern])
			LC_sim_flux_sampled.append((LC_sim_flux[i])[sampling_pattern])

		return (T_bins_sim_LC_sampled, LC_sim_flux_sampled)



