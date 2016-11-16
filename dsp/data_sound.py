#!/bin/python

# This script implements a very crude version of Applidium Data over Sound iOS application 
# LICENSE...Free to use on your own responsibility
# Usage: Python <script-name> <wav-file path/name> <Message String to be Embedded>
# The script will produce an output.wav file which contains a short user message
# It would then attempt to decode the user message by demodulating the audio 
# and report in case demodulated output message does not match the user input message.
# Since this is just a crude approximation, there is no error detection/correction
# and perfect synchronization between transmitter and receiver is assumed. In particular, due to
# a lack of preamble we assume the demodulator knows the packet length of incoming message.

import argparse
import numpy as np
from scipy.io import wavfile as wav
from matplotlib.pyplot import *

CenterFreq = 19600
SamplesPerSymbol = 36
BitsPerChar = 8
SNR = 1024

OutFileName = 'output.wav' 
parser = argparse.ArgumentParser()
parser.add_argument("wav_file",   help="Sound File in which to embedd data, result in "+OutFileName)
parser.add_argument("input_data", help="String of characters to be embedded")

# Low pass filter co-efficients
lp_filt = [6.9771e-05,1.8302e-04,4.2854e-04,9.2715e-04,1.8452e-03,3.3797e-03,5.7336e-03,9.0838e-03,
           1.3547e-02,1.9147e-02,2.5792e-02,3.3262e-02,4.1214e-02,4.9205e-02,5.6728e-02,6.3264e-02,
           6.8331e-02,7.1540e-02,7.2639e-02,7.1540e-02,6.8331e-02,6.3264e-02,5.6728e-02,4.9205e-02,
           4.1214e-02,3.3262e-02,2.5792e-02,1.9147e-02,1.3547e-02,9.0838e-03]

# Bandpass filter co-efficients
# GNU Octave did not give me a sharp enough band pass filter
# Had to use the one in Applidium code
# https://github.com/applidium/AudioModem/blob/master/Classes/AMRecorder.m
bp_filt = [+0.0055491, -0.0060955, +0.0066066, -0.0061506, +0.0033972, +0.0028618,-0.0130922, 
           +0.0265188, -0.0409498, +0.0530505, -0.0590496, +0.0557252, -0.0414030, +0.0166718,
           +0.0154256, -0.0498328, +0.0804827, -0.1016295, +0.1091734, -0.1016295, +0.0804827,
           -0.0498328, +0.0154256, +0.0166718, -0.0414030, +0.0557252, -0.0590496, +0.0530505,
           -0.0409498, +0.0265188, -0.0130922, +0.0028618, +0.0033972, -0.0061506, +0.0066066,
           -0.0060955, +0.005549]

# Embedd given message in the sound data by DBPSK modulating it 
# in a non-audible range
def dbpsk_modulate(sound_data, in_msg, sample_rate):
    print "Modulating Data ...", in_msg
    # Time axis of sound samples
    time_seq = np.arange(0, len(sound_data), 1, dtype=float)/sample_rate
    two_pi_fc_t = 2 * np.pi * CenterFreq * time_seq
    current_phase = 0
    dbpsk_signal  = np.zeros(len(sound_data))

    # Modulation loops over each character of input message,
    # differentially modulates it with last characters and then
    # upconverts on a carrier signal
    coded_bit = 0 # Initial state of coded message
    for mod_idx in np.arange(len(in_msg)):
        # Binary representation of each character in one byte
        char_to_send = bin(ord(in_msg[mod_idx]))[2:].zfill(BitsPerChar)
        start_idx = mod_idx * 8 * BitsPerChar
        for sym_idx in np.arange(0, BitsPerChar, 1):
            coded_bit = int(char_to_send[sym_idx]) ^ coded_bit
            current_phase = np.pi if coded_bit else 0
            begin = start_idx + sym_idx * SamplesPerSymbol
            end   = start_idx + (sym_idx + 1)  * SamplesPerSymbol
            dbpsk_signal[begin:end] = np.round(np.cos(two_pi_fc_t[0:SamplesPerSymbol] + current_phase) * SNR)

    return sound_data + dbpsk_signal

# DPSK demodulation from J.G. Proakis Digital Communication
# Fourth Edition, Figure 5.4-11
# rx_data is the audio signal from given wave file including the user message
# L is the packet length of user data    
def dbpsk_demod(rx_data, sample_rate, L):
    print "Demodulating Data...@", sample_rate
    time_seq = np.arange(0, len(rx_data), 1, dtype=float)/sample_rate
    two_pi_fc_t = 2 * np.pi * CenterFreq * time_seq
    # Filter out-of-band noise
    rx_inband = np.convolve(rx_data, bp_filt)
    N = len(rx_inband)
    # Downconvert I/Q channels into baseband signals
    rx_bb_i = np.multiply(rx_inband[SamplesPerSymbol/2: N - SamplesPerSymbol/2], np.cos(two_pi_fc_t))
    rx_bb_q = np.multiply(rx_inband[SamplesPerSymbol/2: N - SamplesPerSymbol/2], np.sin(two_pi_fc_t))
    # Filter any high frequency remnants
    audio_bb_i = np.convolve(rx_bb_i, lp_filt)[:L*SamplesPerSymbol*BitsPerChar]
    audio_bb_q = np.convolve(rx_bb_q, lp_filt)[:L*SamplesPerSymbol*BitsPerChar]
    decoded_bits = np.zeros(L*BitsPerChar)
    # Previous Phase and decode bit
    pp = 0
    pb = 0
    detected_bitstream = np.zeros(L * BitsPerChar, dtype=int)
    T = SamplesPerSymbol
    # Matched filter is just a rectangular pulse
    rect_pulse = np.ones(T)
    for demod in np.arange(L*BitsPerChar):
        sym_i = np.correlate(audio_bb_i[demod*T:(demod+1)*T], rect_pulse, 'full')[T]
        sym_q = np.correlate(audio_bb_q[demod*T:(demod+1)*T], rect_pulse, 'full')[T]
        cp = np.arctan(sym_q/sym_i)
        #print "Phase Diff:", cp-pp
        if (np.abs(cp - pp) > 0.1):
            detected_bitstream[demod] = (pb ^ 1)
        else:
            detected_bitstream[demod] = detected_bitstream[demod-1]
            pb = detected_bitstream[demod]
            pp = cp

    return detected_bitstream

# Method to extract characters from decoded bit stream and
# compare with user message
# d_bitstream: DBPSK decoded bitstream
# input_msg  : User input message that was embedded in audio
def report_error(d_bitstream, input_msg):
    str_sym=''
    error_count = 0
    for d_idx in d_bitstream:
        str_sym += str(d_idx)
    print "Decode BitStream", str_sym
    for s_idx in np.arange(len(input_msg)):
        if (input_msg[s_idx] != chr(int(str_sym[s_idx*8:(s_idx+1)*8],2))):
            error_count += 1
            print "Error detected at index ", s_idx
    if (error_count == 0):
        print "Hurray! No error in detected message!!"

# Controller module
if __name__ == '__main__':
    # Parse command line parameters
    args = parser.parse_args()
    # Open WavFile and read the data
    print "Opening sound file: ", args.wav_file, "..."
    [sample_rate, stereo_samples] = wav.read(args.wav_file)
    print sample_rate, len(stereo_samples), type(stereo_samples)
    # Stereo to Mono conversion
    data = stereo_samples[:,1].reshape(-1)
    modulated_ = dbpsk_modulate(data, args.input_data, sample_rate)
    print "Writing Sound+Data in: ", OutFileName, "..."
    wav.write('output.wav', sample_rate, modulated_)
    # Demodulation and error detection calls
    demod_msg = dbpsk_demod(modulated_, sample_rate, len(args.input_data)) 
    report_error(demod_msg, args.input_data)        

