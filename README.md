# RAINet – a prototype for automatic assignment of rotational spectra

A prototype of the method from Zaleski & Prozument, (2018):
a neural network recognises the molecule type and estimates the rotational constants from
the list of line frequencies alone, and a least-squares fit then refines the result.

## What it can do

| Molecule type       | Parameters | Supported transitions                                             |
|---------------------|---|-------------------------------------------------------------------|
| linear              | B, DJ | J &rarr; J+1                                                      |
| symmetric (prolate) | B, DJ, DJK (+ HJ, HJK, HKJ when there are many lines) | J &rarr; J+1, $\Delta$K=0                                         |
| asymmetric          | A, B, C, DJ, DJK, DK, δJ, δK | a-type, J &rarr; J+1 <br><sub>($\Delta$Ka=0, $\Delta$Kc=+1)</sub> |

For all parameters the program reports a 1$\sigma$ uncertainty and flags those that the data
hardly constrain (e.g. A and DK in a-type series).

**Not supported:** b- and c-type lines, oblate molecules (e.g. NH3), fine and hyperfine
structure, internal rotation, inversion, mixtures,
intensities, vibrationally excited states.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

1. **Train the model** (once):
   ```powershell
   python rainet_rotational_ml.py train --samples 20000
   ```
   This creates `rainet_model.joblib`. Changing the frequency range, the number of peaks
   or the features requires retraining.

2. **Download a spectrum from CDMS**, save it as `spectrum_NAME.csv`, then convert it:
   ```powershell
   python cdms_to_csv.py spectrum_NAME.csv --r-branch
   ```
   The output goes to `cdms_NAME\` (`transitions.csv`, `peak_list.csv`). The `--r-branch`
   option keeps only the lines the model describes. The program prints the $\Delta$J
   distribution, the frequency range and the number of lines.

3. **Analysis**:
   ```powershell
   python rainet_rotational_ml.py analyze cdms_NAME\peak_list.csv
   ```
   or graphically:
   ```powershell
   streamlit run rainet_dashboard.py
   ```

Other commands: `benchmark` (literature constants, idealised).

## How it works

1. **Synthetic data** – random physical parameters &rarr; lines (formulas for linear and
   symmetric molecules, diagonalisation of the Hamiltonian for asymmetric ones) &rarr;
   a “measurement” (missing lines, noise, merging of close peaks).
2. **Features** – from the 60 lowest peaks: log f1, ratios f_i/f1, logarithms of the gaps.
3. **Networks** – a molecule type classifier, then a separate parameter regression for each class.
4. **Fit** – assignment of peaks to the nearest model lines and least squares
   (linear for linear and symmetric molecules, non-linear for asymmetric ones);
   many starting points, lines added gradually.
5. **Quality check** – if the chosen class does not explain ≥50% of the peaks, the program
   tries the other classes; below 50% the dashboard shows an error instead of parameters.

## How to read the result

- **“Class confidence”** only says which class the spectrum most resembles.
  Correctness is decided by the **number of matched peaks** and the rms.
- **$\sigma$** is the uncertainty from the fit, assuming the assignment is correct.
- A small rms with a number of lines close to the number of parameters does not indicate quality.


## Files

| File | Role                              |
|---|-----------------------------------|
| `rainet_rotational_ml.py` | spectrum generator, networks, fit |
| `rainet_dashboard.py` | Streamlit dashboard               |
| `cdms_to_csv.py` | CDMS export &rarr; CSV converter        |


## Limits of the validation

Accuracy on synthetic data does not prove that the method works on real spectra.


