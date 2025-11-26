**Experiment No. 3: Performance of Modulation Techniques**

**Theory for Viva**

Purpose

* To study the performance of ASK, FSK, and PSK modulation techniques under noise.
* To evaluate how noise affects signal detection quality in wireless communication systems used in WSNs.

Modulation meaning

* Conversion of digital data (bits) into an analog signal suitable for wireless transmission.
* Ensures efficient bandwidth usage and reliable communication.

Techniques involved

* Amplitude Shift Keying (ASK)

  * Bit 1 → high amplitude
  * Bit 0 → low amplitude
  * Highly sensitive to noise because noise easily distorts amplitude.
* Frequency Shift Keying (FSK)

  * Bit 1 → high frequency
  * Bit 0 → low frequency
  * More robust compared to ASK; frequency is less affected by noise.
* Phase Shift Keying (PSK)

  * Bit information encoded in phase changes of a carrier signal.
  * Very robust to noise and widely used in modern digital communication.

Performance parameters

* Bit Error Rate (BER): number of wrong bits after demodulation.
* Signal-to-Noise Ratio (SNR): measure of noise intensity.
* Spectral efficiency: how well bandwidth is utilized.

Common observations

* ASK has highest BER due to amplitude distortion.
* FSK performs better because frequency is stable under noise.
* PSK has the best performance and lowest BER even with higher noise.
* PSK is widely used in IEEE 802.15.4 and many WSN radios.

Summary points for viva

* BER is the key measure of modulation performance.
* Noise affects amplitude more than frequency or phase.
* PSK > FSK > ASK in robustness and reliability.
* WSN radios prefer PSK for better energy and error performance.

---

**Implementation for Performance**

Objective
To simulate and compare ASK, FSK, and PSK modulation schemes and analyze their performance under noise.

Requirements

* Python with NumPy and Matplotlib or MATLAB.
* Basic signal generation and noise addition functions.

Procedure (conceptual steps)

1. Generate a random binary sequence (0s and 1s).
2. For each modulation technique:

   * Convert bits to ASK/FSK/PSK signals.
   * Add Gaussian noise of different levels (SNR values).
3. Demodulate each signal back to bits.
4. Compute Bit Error Rate for each technique.
5. Plot BER vs SNR or visually inspect noisy waveforms.
6. Compare which technique performs best in noisy conditions.

Python implementation (fast demonstration)

```
import numpy as np
import matplotlib.pyplot as plt

N = 1000
bits = np.random.randint(0, 2, N)

# ASK modulation
ask = 2*bits - 1

# FSK modulation
t = np.linspace(0, 1, N)
f1 = np.sin(2*np.pi*5*t)
f0 = np.sin(2*np.pi*2*t)
fsk = np.where(bits == 1, f1, f0)

# PSK modulation
psk = np.cos(2*np.pi*5*t + np.pi*bits)

noise = 0.5 * np.random.randn(N)

ask_noisy = ask + noise
fsk_noisy = fsk + noise
psk_noisy = psk + noise

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.plot(ask_noisy[:200]); plt.title("ASK Noisy")
plt.subplot(1,3,2); plt.plot(fsk_noisy[:200]); plt.title("FSK Noisy")
plt.subplot(1,3,3); plt.plot(psk_noisy[:200]); plt.title("PSK Noisy")
plt.show()
```

Alternate implementation

```
# Realistic-but-simple WSN lab demo for Experiment 3:
# Compare ASK, FSK, BPSK performance (BER vs Eb/N0)
# Run in Google Colab or local Python (numpy + matplotlib required)

import numpy as np
import matplotlib.pyplot as plt

# ------- parameters -------
np.random.seed(1)
num_bits = 5000           # number of bits to simulate (adjust for speed/accuracy)
samples_per_bit = 16      # oversampling factor (>=8 recommended)
fs = samples_per_bit      # sampling frequency in samples per bit (normalized)
t_bit = np.arange(0, samples_per_bit) / fs

# carrier frequencies (in cycles per bit)
fc_ask = 2.0              # ASK carrier cycles per bit
fc_bpsk = 2.0             # BPSK carrier cycles per bit
fc_fsk_0 = 1.0            # FSK carrier for bit 0
fc_fsk_1 = 3.0            # FSK carrier for bit 1

# SNR range (Eb/N0 in dB)
eb_n0_dbs = np.arange(0, 11, 2)  # 0,2,4,...,10 dB

# ------- helper functions -------

def bits_to_waveform(bits, mod_type):
    """
    Convert bit array to oversampled waveform for chosen modulation.
    mod_type: 'ask', 'fsk', 'bpsk'
    returns: waveform (1D numpy array)
    """
    wave = np.zeros(num_bits * samples_per_bit)
    for i, b in enumerate(bits):
        idx0 = i * samples_per_bit
        if mod_type == 'ask':
            # ASK: amplitude 1 for bit=1, 0 for bit=0 (on-off keying)
            carrier = np.cos(2*np.pi*fc_ask*t_bit)
            amp = 1.0 if b == 1 else 0.0
            wave[idx0:idx0+samples_per_bit] = amp * carrier
        elif mod_type == 'bpsk':
            # BPSK: phase 0 for bit=1, pi for bit=0
            phase = 0.0 if b == 1 else np.pi
            carrier = np.cos(2*np.pi*fc_bpsk*t_bit + phase)
            wave[idx0:idx0+samples_per_bit] = carrier
        elif mod_type == 'fsk':
            # BFSK: choose frequency based on bit
            f = fc_fsk_1 if b == 1 else fc_fsk_0
            carrier = np.cos(2*np.pi*f*t_bit)
            wave[idx0:idx0+samples_per_bit] = carrier
        else:
            raise ValueError("Unknown mod_type")
    return wave

def add_awgn(signal, eb_n0_db, bits_per_symbol=1):
    """
    Add AWGN to 'signal' so that Eb/N0 equals eb_n0_db.
    signal: modulated waveform samples (float)
    bits_per_symbol: number of bits per symbol (1 here)
    """
    # energy per bit (Eb) estimation: average signal energy per bit sample times samples_per_bit
    # compute signal power per sample:
    Ps = np.mean(signal**2)
    # bit energy Eb = Ps * (samples_per_bit) / (bits_per_symbol)  (since each bit spans samples_per_bit samples)
    Eb = Ps * samples_per_bit / bits_per_symbol
    # convert Eb/N0 dB to linear
    eb_n0_lin = 10**(eb_n0_db / 10.0)
    # noise spectral density N0 = Eb / (Eb/N0)
    N0 = Eb / eb_n0_lin
    # noise power per sample = N0 * fs / 2 ? (for single-sided conventions)
    # simpler and standard approach for discrete signals: compute noise variance per sample as:
    noise_var = N0 * fs / 2.0  # this keeps units consistent with energy normalization
    noise = np.sqrt(noise_var) * np.random.randn(len(signal))
    return signal + noise

def demodulate_correlator(rx, mod_type):
    """
    Demodulate rx waveform (with oversampling) by correlating with reference carriers per bit.
    Returns estimated bit array.
    """
    est = np.zeros(num_bits, dtype=int)
    for i in range(num_bits):
        idx0 = i * samples_per_bit
        segment = rx[idx0:idx0+samples_per_bit]
        if mod_type == 'ask':
            ref = np.cos(2*np.pi*fc_ask*t_bit)
            # correlate: energy on carrier indicates bit=1 (on-off)
            metric = np.dot(segment, ref)
            est[i] = 1 if metric > 0.2 * samples_per_bit else 0
        elif mod_type == 'bpsk':
            ref0 = np.cos(2*np.pi*fc_bpsk*t_bit)          # phase 0 => bit 1
            # correlate with reference; sign indicates bit
            metric = np.dot(segment, ref0)
            est[i] = 1 if metric >= 0 else 0
        elif mod_type == 'fsk':
            ref0 = np.cos(2*np.pi*fc_fsk_0*t_bit)
            ref1 = np.cos(2*np.pi*fc_fsk_1*t_bit)
            m0 = np.dot(segment, ref0)
            m1 = np.dot(segment, ref1)
            est[i] = 1 if m1 > m0 else 0
    return est

def ber(bits_tx, bits_rx):
    return np.mean(bits_tx != bits_rx)

# ------- run simulation across Eb/N0 for each modulation -------
bits = np.random.randint(0, 2, num_bits)

results = {'ask': [], 'fsk': [], 'bpsk': []}

for eb in eb_n0_dbs:
    # ASK
    tx_ask = bits_to_waveform(bits, 'ask')
    rx_ask = add_awgn(tx_ask, eb)
    est_ask = demodulate_correlator(rx_ask, 'ask')
    results['ask'].append(ber(bits, est_ask))

    # FSK
    tx_fsk = bits_to_waveform(bits, 'fsk')
    rx_fsk = add_awgn(tx_fsk, eb)
    est_fsk = demodulate_correlator(rx_fsk, 'fsk')
    results['fsk'].append(ber(bits, est_fsk))

    # BPSK
    tx_bpsk = bits_to_waveform(bits, 'bpsk')
    rx_bpsk = add_awgn(tx_bpsk, eb)
    est_bpsk = demodulate_correlator(rx_bpsk, 'bpsk')
    results['bpsk'].append(ber(bits, est_bpsk))

# ------- plot -------
plt.figure(figsize=(8,5))
plt.semilogy(eb_n0_dbs, results['ask'], marker='o', label='ASK (OOK)')
plt.semilogy(eb_n0_dbs, results['fsk'], marker='s', label='BFSK')
plt.semilogy(eb_n0_dbs, results['bpsk'], marker='^', label='BPSK')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER vs Eb/N0: ASK vs BFSK vs BPSK')
plt.legend()
plt.ylim(1e-4, 1)
plt.show()

# ------- quick visual of first 200 samples of each noisy modulated signal -------
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(tx_ask[:200], label='tx')
plt.plot(add_awgn(tx_ask, 4)[:200], alpha=0.7, label='rx (4 dB)')
plt.title('ASK (first 200 samples)')
plt.legend()
plt.subplot(1,3,2)
plt.plot(tx_fsk[:200])
plt.plot(add_awgn(tx_fsk, 4)[:200], alpha=0.7)
plt.title('BFSK (first 200 samples)')
plt.subplot(1,3,3)
plt.plot(tx_bpsk[:200])
plt.plot(add_awgn(tx_bpsk, 4)[:200], alpha=0.7)
plt.title('BPSK (first 200 samples)')
plt.tight_layout()
plt.show()
```

Key observations during performance

* ASK shows severe distortion and fluctuating amplitude.
* FSK waveforms retain distinguishable frequency patterns even under noise.
* PSK waveform remains most stable with noise, retaining clear phase changes.

Conclusion
PSK provides the best resistance to noise and lowest BER, followed by FSK and ASK. Therefore, PSK is preferred for reliable and energy-efficient WSN communication.