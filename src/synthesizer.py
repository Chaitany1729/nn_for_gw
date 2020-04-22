#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 02:48:45 2020

@author: chaitanya
"""

import h5py
import numpy as np
import random

from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries
from pycbc.types import real_same_precision_as
from pycbc.filter import matchedfilter
from pycbc.psd import interpolate

from constants import f_min, f_sample, time_duration, delta_f, delta_t
from generate_waveforms import generate_noise
        

class GWsynthesizer(object):
    

        
    def sample():
        
        m1 = random.randrange(10, 40, 1)
        m2 = random.randrange(10, 40, 1)
        SNR = random.randrange(10000, 100000, 1)/1000
        merger_time = random.randrange(650*time_duration, 850*time_duration, 1)/1000
        
        return m1, m2, SNR, merger_time
    
    def inject(merger_time, noise, template):
        
        merger_index = int(merger_time/delta_t) +1 
        start_index = merger_index + len(template)
        waveform = TimeSeries(np.zeros(len(noise)), delta_t=delta_t, \
        dtype=real_same_precision_as(noise))
        waveform[merger_index :start_index] = template
        injected_signal = noise + waveform
        
        return injected_signal
        
    def scale(template, noise, SNR):
        
        template_size = len(template)
        template.resize(time_duration*f_sample)
        psd = noise.psd(1)
        psd = interpolate(psd, template.delta_f)
        sigma = matchedfilter.sigma(template, psd = psd, low_frequency_cutoff=f_min)
        amplitude = SNR/(sigma)
        template *= amplitude
        template.resize(template_size)
        
        return template, amplitude
    
    def create_dataset(filename, samples):
        
        file = h5py.File(filename, "w")
        for i in range(samples):
            print(i)
            m1, m2, SNR, merger_time = GWsynthesizer.sample()
            hp, hc = get_td_waveform(approximant='SEOBNRv4_opt', mass1=m1, mass2=m2, delta_t= delta_t, f_lower = 25, distance=100)
            noise = generate_noise(time_duration, delta_f, delta_t, f_min)
            
            scaled_template, amplitude = GWsynthesizer.scale(hp, noise, SNR)
            injected_waveform = GWsynthesizer.inject(merger_time, noise, scaled_template) 
            injected_waveform = np.array(injected_waveform)
            
            dset = file.create_dataset('sample_'+ str(i), data=injected_waveform)
            
            dset.attrs['m1'] = m1
            dset.attrs['m2'] = m2
            dset.attrs['SNR'] = SNR
            dset.attrs['amplitude'] = amplitude
            dset.attrs['merger_time'] = merger_time
            
        return file


file = GWsynthesizer.create_dataset('filenam88.hdF5', 5)        

     


    
    
