#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:05:01 2020

@author: chaitanya
"""

import numpy as np
from scipy import signal
import pycbc.psd, pycbc.noise 
from pycbc.types import TimeSeries
from numpy import pi 
from pycbc.waveform import get_td_waveform



class waveformGenerator(object):
    
    #generates TimeSeries of 1s duration with delta_t = 1/4096
    
    def generate_SineGaussianBlip(fc, bw, duration): 
        
        #returns sine gaussian blip with bandwidth = bw, centre frequency = fc
        
        t = np.linspace(-duration/2, duration/2, duration*4096)
        sine_gaussian = signal.gausspulse(t, fc=fc, bw=bw)
        waveform = TimeSeries(sine_gaussian, delta_t=1/4096)
        return waveform
    
    def generate_FrequencyLine(freq, duration):
        
        #returns a sine wave with frequency = freq
        
        t = np.linspace(0, duration, duration*4096)
        freq_line = np.sin(t*2*pi*freq)
        waveform = TimeSeries(freq_line, delta_t=1/4096)
        return waveform
        
    def generate_GWChirp(m1, m2):
        
        #returns + polarization of the gw chirp
        hp, hc = get_td_waveform(approximant='SEOBNRv4_opt', mass1=m1, mass2=m2, delta_t=1/4096, f_lower = 20 )
        return hp
    
    def generate_Impulse(position, duration):
        
        #returns an impusle signal with duration = durationa and position = position 
        
        impulse = signal.impulse(duration,position)
        waveform = TimeSeries(impulse, delta_t=1/4096)
        return waveform
    
    def generate_Whistle(f_lower,bw, duration, tc =0.5):
        '''
        f_lower: lower frequency
        tc: fractional center time position (time at which fequency is min)
        bw: bandwidth in frequency domian'''
        
        t = np.linspace(0, duration, duration*4096)
        omega = f_lower + 2*pi*(bw/tc**2)*(t - tc*duration)**2
        whistle = np.sin(omega*t)
        waveform = TimeSeries(whistle, delta_t=1/4096)
        return waveform
    
    def generate_Noise(duration, delta_f, delta_t, f_min): 
        flen = int(2048/delta_f) +1 
        psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, f_min)
        tsamples = int(duration / delta_t) 
        ts = pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed=127) 
        return ts

        
        
