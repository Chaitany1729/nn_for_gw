import pylab 
import numpy 
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries
from pycbc.types import real_same_precision_as
from generate_waveforms import generate_noise
from pycbc.filter import matchedfilter
from pycbc.psd import interpolate
from pycbc.filter import matched_filter

mass = 36 

f_min = 20
time_duration = 128
f_sample = 4096
f_max = f_sample/2
delta_t = 1/f_sample
delta_f = 1/time_duration

hp, hc = get_td_waveform(approximant='SEOBNRv4_opt', mass1=mass, mass2=mass, delta_t= delta_t, f_lower = 25, distance=100)
noise = generate_noise(time_duration, delta_f, delta_t, f_min) 
 
merger_time = (1/4096) * hp.numpy().argmax() 
snr = 20

hp_size = len(hp)
hp.resize(time_duration*f_sample)
psd = noise.psd(4)
psd = interpolate(psd, hp.delta_f)
sigma = matchedfilter.sigma(hp, psd = psd, low_frequency_cutoff=f_min)
Amplitude = snr/(sigma)


hp *= Amplitude

hp.resize(hp_size)
merger_time = 69
merger_index = int(69/delta_t) +1 
start_index = merger_index + len(hp)
waveform = TimeSeries(numpy.zeros(len(noise)), delta_t=delta_t, \
        dtype=real_same_precision_as(noise))


waveform[merger_index :start_index] = hp

signal = noise + waveform

pylab.figure(figsize=(10,5)) 
pylab.plot(signal.sample_times,signal,label='Waveform + Noise')
pylab.plot(waveform.sample_times,waveform,label='Waveform')

pylab.legend() 
pylab.xlabel('Time (s)')
pylab.ylabel('Strain')
pylab.title('Injected Waveform')


zoom_signal = signal.time_slice(merger_time-0.5,merger_time+0.5)
zoom_waveform = waveform.time_slice(merger_time-0.5,merger_time+0.5)
pylab.figure(figsize=(10,5)) 
pylab.plot(zoom_signal.sample_times,zoom_signal,label='Signal')
pylab.plot(zoom_waveform.sample_times,zoom_waveform,label='Waveform')
pylab.xlabel('Time (s)')
pylab.ylabel('Strain')
pylab.title('Zoomed View') 
pylab.legend()

hp.resize(time_duration*f_sample)
snr = matched_filter(hp, signal,
                     psd=psd, low_frequency_cutoff=20)



pylab.figure(figsize=[10, 4])
pylab.plot(snr.sample_times, abs(snr))
pylab.ylabel('Signal-to-noise')
pylab.xlabel('Time (s)')
pylab.show()

peak = abs(snr).numpy().argmax()
snrp = snr[peak]
time = snr.sample_times[peak]

print("We found a signal at {}s with SNR {}".format(time, 
                                                    abs(snrp)))
''''
from synthesizer import GWsynthesizer
hp, hc = get_td_waveform(approximant='SEOBNRv4_opt', mass1=mass, mass2=mass, delta_t= delta_t, f_lower = 25, distance=100)
scaled, amp = GWsynthesizer.scale(hp, noise, 20)'''

