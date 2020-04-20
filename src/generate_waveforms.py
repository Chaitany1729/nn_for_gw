#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 16:01:24 2020

@author: chaitanya
"""

#The module contains the functions generating eaveforms of differnt types viz. GW Chirp, whistels, sine gaussian blip, Constant Frequency Line, impulse, etc.
#The waveforms are generated in the time domain
#Each waveform has a duration of 2s


import numpy as np
from scipy import signal
import pycbc.psd, pycbc.noise 
from pycbc.types import TimeSeries
from numpy import pi 
from pycbc.waveform import get_td_waveform
import matplotlib.pyplot as plt

def generate_SineGaussianBlip(fc, bw): 
    
    #returns sine gaussian blip with bandwidth = bw, centre frequency = fc
    
    t = np.linspace(-1, 1, 2*4096, endpoint=False)
    sine_gaussian = signal.gausspulse(t, fc=fc, bw=bw)
    waveform = TimeSeries(sine_gaussian, delta_t=1/4096)
    return waveform

def generate_FrequencyLine(freq):
    
    #returns a sine wave with frequency = freq
    
    t = np.linspace(-1, 1, 2*4096, endpoint=False)
    freq_line = np.sin(t*2*pi*freq)
    waveform = TimeSeries(freq_line, delta_t=1/4096)
    return waveform
    
def generate_GWChirp():
    
    #returns + polarization of the gw chirp
    
    mass = 10 
    hp, hc = get_td_waveform(approximant='SEOBNRv4_opt', mass1=mass, mass2=mass, delta_t=1/4096, f_lower = 20 )
    return hp

def generare_Impulse(duration, position):
    
    #returns an impusle signal with duration = durationa and position = position 
    
    impulse = signal.impulse(duration,position)
    waveform = TimeSeries(impulse, delta_t=1/4096)
    return waveform
 
def generate_noise(time): 
    flow = 30 
    delta_f = 1/16
    flen = int(2048/delta_f) +1 
    psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)
    delta_t = 1.0 / 4096 
    tsamples = int(time / delta_t) 
    ts = pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed=127) 
    return ts

def generate_Whistle(duration, min_position):
        t = np.linspace(0, duration, duration*4096)
        omega = f_lower + 2*pi*(bw/tc**2)*(t - tc*duration)**2
        whistle = np.sin(omega*t)
        waveform = TimeSeries(whistle, delta_t=1/4096)
        return waveform
    

    

