import pylab 
import numpy 
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries
from pycbc.types import real_same_precision_as
from generate_waveforms import generate_noise
from pycbc.filter import matchedfilter
from pycbc.psd import interpolate, inverse_spectrum_truncation

mass = 36 

f_min = 20
time_duration = 128
f_sample = 4096
delta_t = 1/f_sample
delta_f = 1/time_duration

hp, hc = get_td_waveform(approximant='SEOBNRv4_opt', mass1=mass, mass2=mass, delta_t= delta_t, f_lower = f_min, distance=100)

 
merger_time = (1/4096) * hp.numpy().argmax() 
snr = 20

pylab.figure(figsize=(10,5)) 
pylab.plot(hp.sample_times,hp) 
pylab.title('Generated Waveform')
pylab.xlabel('Time (s)')
pylab.ylabel('Strain')

noise = generate_noise(time_duration, delta_f, delta_t) 
pylab.figure(figsize=(10,5)) 
pylab.plot(noise.sample_times,noise) 
pylab.title('Noise')
pylab.xlabel('Time (s)')
pylab.ylabel('Strain')


psd = noise.psd(4)
psd = interpolate(psd, hp.delta_f)

sigma = matchedfilter.sigma(hp, psd = psd, low_frequency_cutoff=20)
Amplitude = snr/(sigma**2)


hp *= Amplitude

merger_time = 69
merger_index = int(69/noise.delta_t)
start_index = merger_index - len(hp)
waveform = TimeSeries(numpy.zeros(len(noise)), delta_t=noise.delta_t, \
        dtype=real_same_precision_as(noise))


waveform[start_index:merger_index] = hp

signal = noise + waveform

pylab.figure(figsize=(10,5)) 
pylab.plot(signal.sample_times,signal,label='Waveform + Noise')
pylab.plot(waveform.sample_times,waveform,label='Waveform')

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