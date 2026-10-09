# Experiment 11: M-PSK and M-QAM (first-principles Python)

Hand-written QPSK, 8-PSK, 16-QAM and 64-QAM: constellation generation, Gray and binary mapping,
average-energy normalisation, complex AWGN with Eb/N0 (Es = k*Eb), minimum-distance detection,
independent BER and SER counting, d_min, and spectral efficiency. Only NumPy and Matplotlib are used.

## Run
```bash
pip install -r requirements.txt
python exp11_mpsk_mqam.py          # saves figures to exp11_figures/ and prints all tables and checks
```
or open `exp11_mpsk_mqam.ipynb` (already executed, outputs included).
`EXP11_NSYM=2e5 python exp11_mpsk_mqam.py` runs a shorter simulation (default 500000 symbols per SNR point).

## Notes
* The script prints a 55-item validation checklist at the end (point counts, Es = 1 after normalisation,
  Gray adjacency, detector vs brute force, zero-noise run, noise variance, d_min, BER/SER consistency).
* Theory curves are textbook approximations for Gray-coded BER (QPSK is exact); they are labelled as theory.
* Spectral efficiency log2(M) bit/s/Hz is analytic (ideal Nyquist, no coding/overhead), not measured.
