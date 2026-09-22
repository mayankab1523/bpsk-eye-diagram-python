import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Root Raised Cosine (RRC) Filter
# ============================================================
def rrc_filter(beta, span, sps):
    """
    beta : roll-off factor
    span : filter length in symbols
    sps  : samples per symbol
    """

    N = span * sps
    t = np.arange(-N / 2, N / 2 + 1) / sps

    h = np.zeros(len(t))

    for i, ti in enumerate(t):

        if abs(ti) < 1e-12:
            h[i] = 1 + beta * (4 / np.pi - 1)

        elif beta > 0 and abs(abs(ti) - 1 / (4 * beta)) < 1e-12:
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )

        else:
            numerator = (
                np.sin(np.pi * ti * (1 - beta))
                + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            )

            denominator = (
                np.pi * ti * (1 - (4 * beta * ti) ** 2)
            )

            h[i] = numerator / denominator

    # Normalize filter energy
    h = h / np.sqrt(np.sum(h ** 2))

    return h


# ============================================================
# Reusable Eye Diagram Function
# ============================================================
def eye_diagram(signal, sps, eye_symbols=2, traces=100):
    """
    signal      : input signal
    sps         : samples per symbol
    eye_symbols : width of eye diagram in symbols
    traces      : number of traces
    """

    samples_per_trace = eye_symbols * sps

    plt.figure(figsize=(8, 5))

    max_start = len(signal) - samples_per_trace

    count = 0

    for start in range(0, max_start, sps):

        if count >= traces:
            break

        segment = signal[start:start + samples_per_trace + 1]

        time = np.arange(len(segment)) / sps
        time = time - eye_symbols / 2

        plt.plot(time, segment, linewidth=0.7)

        count += 1

    plt.axvline(0, linestyle="--", linewidth=1.5)
    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xlabel("Time (Symbol Periods)")
    plt.ylabel("Amplitude")
    plt.title("BPSK Eye Diagram")
    plt.grid(True)
    plt.show()


# ============================================================
# PARAMETERS
# ============================================================

Nsym = 2000                  # Number of BPSK symbols
sps = 8                      # Samples per symbol
span = 6                     # RRC span in symbols
rolloff = 0.35               # RRC roll-off
SNRs = [0, 5, 10, 15]        # SNR values in dB


# ============================================================
# BPSK GENERATION
# ============================================================

np.random.seed(10)

bits = np.random.randint(0, 2, Nsym)

# BPSK:
# bit 0 -> -1
# bit 1 -> +1
symbols = 2 * bits - 1


# ============================================================
# UPSAMPLING
# ============================================================

tx_up = np.zeros(Nsym * sps)

tx_up[::sps] = symbols


# ============================================================
# TRANSMIT RRC FILTER
# ============================================================

h = rrc_filter(rolloff, span, sps)

tx = np.convolve(tx_up, h)


# ============================================================
# FILTER DELAY
# ============================================================

single_filter_delay = span * sps // 2

total_filter_delay = 2 * single_filter_delay

print("======================================")
print("FILTER DELAY")
print("======================================")

print("Transmit filter delay :", single_filter_delay, "samples")
print("Receive filter delay  :", single_filter_delay, "samples")
print("Total filter delay    :", total_filter_delay, "samples")

print()


# ============================================================
# EYE DIAGRAM FOR DIFFERENT SNRs
# ============================================================

for snr_db in SNRs:

    # Signal power
    signal_power = np.mean(tx ** 2)

    # Noise power
    noise_power = signal_power / (10 ** (snr_db / 10))

    # AWGN
    noise = np.sqrt(noise_power) * np.random.randn(len(tx))

    rx = tx + noise

    # ========================================================
    # MATCHED FILTER
    # ========================================================

    mf = np.convolve(rx, h)

    # ========================================================
    # COMPENSATE TOTAL FILTER DELAY
    # ========================================================

    mf_sync = mf[total_filter_delay:]

    # ========================================================
    # EYE DIAGRAM
    # ========================================================

    eye_diagram(
        mf_sync,
        sps,
        eye_symbols=2,
        traces=100
    )

    plt.title(
        f"BPSK Eye Diagram After Matched Filtering - SNR = {snr_db} dB"
    )



# ============================================================
# TIMING OFFSET ANALYSIS
# ============================================================

snr_db = 10

signal_power = np.mean(tx ** 2)

noise_power = signal_power / (10 ** (snr_db / 10))

noise = np.sqrt(noise_power) * np.random.randn(len(tx))

rx = tx + noise


# ============================================================
# MATCHED FILTER
# ============================================================

mf = np.convolve(rx, h)


# ============================================================
# REMOVE TOTAL FILTER DELAY
# ============================================================

mf_sync = mf[total_filter_delay:]


# ============================================================
# FIND OPTIMUM SAMPLING INSTANT
# ============================================================

timing_offsets = np.arange(sps)

eye_opening = np.zeros(sps)


for offset in timing_offsets:

    # Take samples at this timing offset
    samples = mf_sync[offset::sps]

    # Make same length as transmitted symbols
    L = min(len(samples), len(symbols))

    samples = samples[:L]
    reference = symbols[:L]

    # Positive symbol samples
    positive = samples[reference == 1]

    # Negative symbol samples
    negative = samples[reference == -1]

    if len(positive) > 0 and len(negative) > 0:

        # Eye opening
        eye_opening[offset] = (
            np.mean(positive) - np.mean(negative)
        )


# ============================================================
# BEST TIMING OFFSET
# ============================================================

best_offset = np.argmax(eye_opening)

maximum_eye_opening = eye_opening[best_offset]


print("======================================")
print("OPTIMUM SAMPLING RESULT")
print("======================================")

print("SNR =", snr_db, "dB")

print("Best timing offset :",
      best_offset, "samples")

print("Best timing offset :",
      best_offset / sps,
      "symbol period")

print("Maximum eye opening :",
      maximum_eye_opening)

print()


# ============================================================
# PLOT EYE OPENING VS TIMING OFFSET
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    timing_offsets,
    eye_opening,
    "o-",
    linewidth=2
)

plt.plot(
    best_offset,
    maximum_eye_opening,
    "ro",
    markersize=10
)

plt.xlabel("Timing Offset (Samples)")
plt.ylabel("Eye Opening")

plt.title("Eye Opening vs Timing Offset")

plt.grid(True)

plt.show()


# ============================================================
# SAMPLES AT OPTIMUM TIMING
# ============================================================

optimal_samples = mf_sync[best_offset::sps]


# ============================================================
# PLOT OPTIMUM SAMPLES
# ============================================================

Nplot = min(100, len(optimal_samples))

plt.figure(figsize=(10, 5))

plt.stem(
    np.arange(Nplot),
    optimal_samples[:Nplot]
)

plt.axhline(0, linestyle="--")

plt.xlabel("Symbol Number")
plt.ylabel("Matched Filter Output")

plt.title(
    f"BPSK Samples at Optimum Timing "
    f"(Offset = {best_offset} samples)"
)

plt.grid(True)

plt.show()
