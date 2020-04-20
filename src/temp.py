#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:24:58 2020

@author: chaitanya
"""



import pylab 
import numpy 
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries
from pycbc.types import real_same_precision_as
from generate_waveforms import generate_noise
from pycbc.filter import matchedfilter

mass = 36 
hp, hc = get_td_waveform(approximant='SEOBNRv4_opt', mass1=mass, mass2=mass, delta_t=1/4096,delta_f = 0.25, f_lower = 20, distance=100) 
merger_time = (1/4096) * hp.numpy().argmax() 

snr = 20



#hp = hp*(Amplitude/hp.numpy().argmax().max)

pylab.figure(figsize=(10,5)) 
pylab.plot(hp.sample_times,hp) 
pylab.title('Generated Waveform')
pylab.xlabel('Time (s)')
pylab.ylabel('Strain')

noise = generate_noise(128) 
pylab.figure(figsize=(10,5)) 
pylab.plot(noise.sample_times,noise) 
pylab.title('Noise')
pylab.xlabel('Time (s)')
pylab.ylabel('Strain')


sigma = matchedfilter.sigma(hp, psd = noise.psd(1/1.3536021150033046), low_frequency_cutoff=20)
Amplitude = snr/sigma

merger_time = 69
merger_index = int(69/noise.delta_t)
start_index = merger_index - len(hp)
waveform = TimeSeries(numpy.zeros(len(noise)), delta_t=noise.delta_t, \
        dtype=real_same_precision_as(noise))
waveform[start_index:merger_index] = hp

signal = noise + waveform

pylab.figure(figsize=(10,5)) 
pylab.plot(waveform.sample_times,waveform,label='Signal')
pylab.plot(signal.sample_times,signal,label='Signal + Noise')
pylab.legend() 
pylab.xlabel('Time (s)')
pylab.ylabel('Strain')
pylab.title('Injected Waveform')


zoom_signal = signal.time_slice(merger_time-0.1,merger_time+0.01)
zoom_waveform = waveform.time_slice(merger_time-0.1,merger_time+0.01)
pylab.figure(figsize=(10,5)) 
pylab.plot(zoom_signal.sample_times,zoom_signal,label='Signal')
pylab.plot(zoom_waveform.sample_times,zoom_waveform,label='Waveform')
pylab.xlabel('Time (s)')
pylab.ylabel('Strain')
pylab.title('Zoomed View') 
pylab.legend()



