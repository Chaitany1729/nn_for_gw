# Neural Network For Gravitational Waves
#### The project aims to build a Neural Network based model to classify the glitches in the GW data.

The gravitational wave (GW) signals are feeble and are deeply embedded in the noisy data recorded by the LIGO detectors. Thus, computationally extensive techniques like matched-filtering are required to identify them. We wish to make use of recent advancements in recurrent neural networks (RNNs) to build a tool which successfully identifies GW signals while distinguishing them from other noise transients.

## Preliminaries 

1. GW Data Analysis Tools

The data analysis tools which cover the plotting methods, spectrograms, matched filtering and significance testing. These tutorials are completed for PyCBC and GWpy python modules. The practised code can be found [here](https://github.com/Chaitany1729/trac2019/tree/master/ligo_tutorials). Also, you can find the tutorials link [here](https://github.com/gwastro/PyCBC-Tutorials).

2. Neural Networks

For Neural Networks, following tutorials are worked out:

    * Image Classification on MNIST Data Set: This tutorial is for basic hands-on with the neural network, roc plots and effect of noise on the classification.

    * Audio Classification using CNN: An underlying four-layer CNN model is built as per the tutorial. This model classifies the audio data set with ten classes. The results are roc plot for the class car_horn is shown as sample result. The overall testing accuracy was found to be 86.34%.
    
## Data Set Preparation

The built model is firstly trained on the synthesized dataset. The dataset contains five classes, namely, GW Chirp signal, Whistles, Sine Gaussian Blips, Constant frequency line and Hammering signal. Each data point(signal) in the set has a constant length of 2s. The signals are injected in the Gaussian noise within 1.30s to 1.70s(65% to 85% of the length of the signal, i.e. 2s). 

1. Class waveformGenerator:
This class contains methods to generate all five kinds of waveforms. To generate the GW chip and gaussian noise Pycbc module is used. Other waveforms are modelled mathematically.

2. Injection with Given SNR
The GW chirp is generated with the help of Pycbc module. The amplitude of the chirp signal is scaled as per the required signal to noise ratio(SNR). Other waveforms are also scaled according to the amplitude of the chirp signal.

3. Class synthesizer:
This class calls the generator methods and do the proper injections. After injecting the waveform at the given position, the waveforms are stored as directories in hDf5 files along with the waveform parameters like SNR, the position of injection, etc.


