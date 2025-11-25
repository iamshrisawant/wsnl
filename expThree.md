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

MATLAB implementation (alternative)

```
N = 1000;
bits = randi([0 1], 1, N);

ask = 2*bits - 1;

t = linspace(0,1,N);
f1 = sin(2*pi*5*t);
f0 = sin(2*pi*2*t);
fsk = bits .* f1 + (~bits) .* f0;

psk = cos(2*pi*5*t + pi*bits);

noise = 0.5*randn(1,N);

ask_noisy = ask + noise;
fsk_noisy = fsk + noise;
psk_noisy = psk + noise;

subplot(1,3,1); plot(ask_noisy(1:200)); title("ASK Noisy");
subplot(1,3,2); plot(fsk_noisy(1:200)); title("FSK Noisy");
subplot(1,3,3); plot(psk_noisy(1:200)); title("PSK Noisy");
```

Key observations during performance

* ASK shows severe distortion and fluctuating amplitude.
* FSK waveforms retain distinguishable frequency patterns even under noise.
* PSK waveform remains most stable with noise, retaining clear phase changes.

Conclusion
PSK provides the best resistance to noise and lowest BER, followed by FSK and ASK. Therefore, PSK is preferred for reliable and energy-efficient WSN communication.