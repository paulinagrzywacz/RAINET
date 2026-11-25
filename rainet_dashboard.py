"""
RAINet - interactive dashboard for rotational peak lists.

Run:  streamlit run rainet_dashboard.py
Needs rainet_rotational_ml.py and a model trained with it
(python rainet_rotational_ml.py train) next to this script, or uploaded in
the sidebar.
"""

import io
import json
import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from rainet_rotational_ml import (
    PARAM_NAMES,
    asym_a_type_lines,
    linear_frequencies,
    labelled_lines,
    load_peak_list,
    predict_from_peaks,
    symmetric_top_frequencies,
)

TYPE_LABEL = {"linear": "Linear", "symmetric": "Symmetric",
              "asymmetric": "Asymmetric (type a)"}
PARAM_LABEL = {"A_MHz": "A [MHz]", "B_MHz": "B [MHz]", "C_MHz": "C [MHz]",
               "DJ_MHz": "DJ [MHz]", "DJK_MHz": "DJK [MHz]", "DK_MHz": "DK [MHz]",
               "dJ_MHz": "δJ [MHz]", "dK_MHz": "δK [MHz]",
               "HJ_MHz": "HJ [MHz]", "HJK_MHz": "HJK [MHz]", "HKJ_MHz": "HKJ [MHz]"}
MATCH_TOL_MHZ = 1.0

def st_plot(fig, **kwargs):
    try:
        st.plotly_chart(fig, width="stretch", **kwargs)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, **kwargs)

def st_table(df, **kwargs):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)

@st.cache_resource(show_spinner=False)
def load_bundle_from_path(path: str, mtime: float):
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_bundle_from_bytes(data: bytes):
    return joblib.load(io.BytesIO(data))


def model_lines_for(kind: str, params: dict, f_max: float):
    """
    Predicted transitions up to f_max with labels (includes sextic terms if fitted).
    """
    return labelled_lines(kind, params, f_max)


def residuals(peaks: np.ndarray, lines: np.ndarray, labels=None):
    """
    Nearest predicted line (and its quantum-number label) for each observed peak.
    """
    if len(lines) == 0 or len(peaks) == 0:
        return (np.full(len(peaks), np.nan), np.full(len(peaks), np.nan),
                ["" for _ in peaks])
    idx = np.abs(peaks[:, None] - lines[None, :]).argmin(axis=1)
    nearest = lines[idx]
    lab = [labels[i] for i in idx] if labels is not None else ["" for _ in idx]
    return nearest, peaks - nearest, lab


def stick_trace(freqs, height, name, color, dash=None):
    f = np.sort(np.asarray(freqs, dtype=float))
    x = np.empty(3 * len(f))
    x[0::3], x[1::3], x[2::3] = f, f, np.nan
    y = np.empty(3 * len(f))
    y[0::3], y[1::3], y[2::3] = 0.0, height, np.nan
    line = dict(color=color, width=2)
    if dash:
        line["dash"] = dash
    return go.Scatter(x=x, y=y, mode="lines", line=line, name=name,
                      hovertemplate=f"{name}<extra></extra>")


@st.cache_data(show_spinner=False)
def run_analysis(_bundle, peaks_bytes: bytes, refine: bool, model_key: str,
                 force_kind: str | None = None):

    peaks = np.frombuffer(peaks_bytes, dtype=float)
    return predict_from_peaks(_bundle, peaks, refine=refine, force_kind=force_kind)


# === PAGE ===

st.set_page_config(page_title="RAINet - rotational spectra", layout="wide")
st.title("RAINet - interactive analysis of rotational spectra")
st.caption("Spectrum-type classification and estimation of rotational constants by a "
           "neural network, optionally refined by least-squares fitting.")
st.warning("These are preliminary prototype results, not a full Hamiltonian fit. ")

with st.sidebar:
    st.header("Input data")
    peak_file = st.file_uploader("Peak list (CSV/TXT)", type=["csv", "txt"])
    model_upload = st.file_uploader("rainet_model.joblib (optional)", type=["joblib"])
    max_freq = st.number_input("Maximum frequency [MHz]", 1000.0, 300000.0, 100000.0, 1000.0)

    st.header("Analysis")
    refine = st.checkbox("Refine by fitting (least squares)", True,
                         help="Without this the raw network output is shown (error usually a few %).")
    min_conf = st.slider("Minimum classification confidence", 0.50, 0.999, 0.90, 0.01)
    kind_choice = st.selectbox(
        "Molecule type", ["Automatic", "Linear", "Symmetric", "Asymmetric (a-type)"],
        help="Automatic: chosen by the classifier, with a check of the fit. "
             "A manual choice forces that type without any correction.")
    force_kind = {"Linear": "linear", "Symmetric": "symmetric",
                  "Asymmetric (a-type)": "asymmetric"}.get(kind_choice)

    st.header("Plot")
    show_peaks = st.checkbox("Show peaks from the file", True)
    show_predictions = st.checkbox("Show lines predicted by the model", True)

if peak_file is None:
    st.markdown(
        "### Getting started\n"
        "1. Train a model: `python rainet_rotational_ml.py train` (creates `rainet_model.joblib`).\n"
        "2. Upload a peak list, e.g. `cdms_output/peak_list.csv` (frequencies in MHz).\n"
        "3. The model must be next to this script or uploaded in the sidebar."
    )
    st.stop()

# === PEAK LIST ===
with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
    f.write(peak_file.getvalue())
    peak_path = Path(f.name)
try:
    all_peaks = np.asarray(load_peak_list(peak_path), dtype=float)
except ValueError as exc:
    st.error(f"Cannot read the file: {exc}")
    st.stop()
finally:
    peak_path.unlink(missing_ok=True)

mask = np.isfinite(all_peaks) & (all_peaks > 0) & (all_peaks <= max_freq)
peaks = np.sort(all_peaks[mask])
n_cut = int(len(all_peaks) - len(peaks))
if len(peaks) == 0:
    st.error("No peaks in the selected frequency range.")
    st.stop()

# === MODEL ===
try:
    if model_upload:
        model_key = f"upload:{model_upload.name}:{model_upload.size}"
        bundle = load_bundle_from_bytes(model_upload.getvalue())
    else:
        model_path = Path("rainet_model.joblib")
        if not model_path.exists():
            st.error("rainet_model.joblib not found. Train it "
                     "(`python rainet_rotational_ml.py train`) or upload it in the sidebar.")
            st.stop()
        model_key = f"file:{model_path.stat().st_mtime}"
        bundle = load_bundle_from_path(str(model_path), model_path.stat().st_mtime)
    if bundle.get("version", 0) < 4:
        st.error("This model comes from an old version of the script. Train it again: "
                 "`python rainet_rotational_ml.py train`.")
        st.stop()
except Exception as exc:  # corrupted file, version mismatch, ...
    st.error(f"Cannot load the model: {exc}")
    st.stop()

with st.spinner("Analysing the spectrum..."):
    result = run_analysis(bundle, peaks.tobytes(), refine, model_key, force_kind)

kind = result["spectrum_type"]
conf = result["classification_probabilities"]
params = result["parameters"]
ml_params = result["ml_parameters"]
ref = result["refinement"]


if n_cut:
    st.info(f"Skipped {n_cut} peaks outside the range 0-{max_freq:,.0f} MHz.")
classifier_choice = result.get("classifier_choice", kind)
overridden = classifier_choice != kind and not result.get("forced_kind")
for w in result["warnings"]:
    if w.startswith("The classifier chose"):
        continue       # replaced by the notice below
    st.warning(w)
if overridden:
    st.warning(
        f"The classifier chose the type “{TYPE_LABEL.get(classifier_choice, classifier_choice)}” "
        f"({conf.get(classifier_choice, 0)*100:.0f}%), but the fit of that type does not explain "
        f"the data. The type “{TYPE_LABEL.get(kind, kind)}” was used instead; the classifier rated "
        f"it at {conf.get(kind, 0)*100:.2f}%. The choice rests only on the fit to the peaks."
    )
elif conf[kind] < min_conf and not result.get("forced_kind"):
    st.warning(f"Classification confidence ({conf[kind]*100:.1f}%) is below the threshold "
               f"{min_conf*100:.0f}%. The spectrum type may be unsupported "
               "(e.g. a mixture of species, b/c-type lines, hyperfine structure).")

lines_all, labels_all = model_lines_for(kind, params, max_freq)
nearest, res, assign = residuals(peaks, lines_all, labels_all)
lines = np.unique(np.round(lines_all, 6))
matched = np.abs(res) <= MATCH_TOL_MHZ
rms_all = float(np.sqrt(np.nanmean(res[matched] ** 2))) if matched.any() else float("nan")

# === RELIABILITY CHECK ===
MIN_MATCH_FRACTION = 0.5
GOOD_MATCH_FRACTION = 0.8
if ref.get("refined") and ref.get("n_peaks_used"):
    match_fraction = ref["n_matched"] / ref["n_peaks_used"]
else:
    match_fraction = float(matched.sum()) / max(len(peaks), 1)
reliable = match_fraction >= MIN_MATCH_FRACTION
if reliable and match_fraction < GOOD_MATCH_FRACTION:
    st.warning(
        f"**Doubtful fit:** the model explains {match_fraction*100:.0f}% of the peaks "
        f"(a good fit is at least {GOOD_MATCH_FRACTION*100:.0f}%). Some lines do not match, "
        "e.g. because of splittings (internal rotation, hyperfine structure), lines of "
        "another type, or impurities."
    )
if not reliable:
    st.error(
        f"**The data fit none of the models.** The model explains only "
        f"{match_fraction*100:.0f}% of the peaks (threshold {MIN_MATCH_FRACTION*100:.0f}%), "
        "so the parameters below are NOT reliable, regardless of the “class confidence”. "
        "Likely causes: the list contains b/c lines, vibrational states, hyperfine or fine "
        "structure, several species, or a molecule outside the model range "
        "(linear, symmetric, asymmetric a-type)."
    )

# === SUMMARY ===
c1, c2, c3, c4 = st.columns(4)
c1.metric("Classification", TYPE_LABEL.get(kind, kind) if reliable else "Not recognised")
c2.metric("Class confidence", f"{conf[kind]*100:.1f}%" if reliable else "n/a")
c3.metric("Peaks matched (|r| ≤ 1 MHz)", f"{int(matched.sum())} / {len(peaks)}")
c4.metric("rms of matched [MHz]", "-" if np.isnan(rms_all) else f"{rms_all:.4f}")

if not ref.get("refined", False) and refine:
    st.caption("The fit failed - showing the raw network result.")
elif ref.get("refined"):
    st.caption(f"The fit used the {ref['n_peaks_used']} lowest peaks "
               f"(matched {ref['n_matched']}, rms {ref['rms_mhz']:.4f} MHz).")

# === PARAMETERS ===
if result.get("forced_kind"):
    st.info(f"Molecule type chosen manually: {TYPE_LABEL.get(kind, kind)}. "
            f"The classifier indicated: {TYPE_LABEL.get(classifier_choice, classifier_choice)} "
            f"({conf.get(classifier_choice, 0)*100:.0f}%).")
st.subheader("Parameters")
unc = (result.get("uncertainties") or {})
sigmas = unc.get("sigma", {})
poor = set(unc.get("poorly_determined", []))
rows = []
for key in params:
    ml_v, fin_v = ml_params.get(key), params[key]
    rows.append({
        "Parameter": PARAM_LABEL[key],
        "Network (raw)": "-" if ml_v is None else f"{ml_v:.6f}",
        "Final result": f"{fin_v:.6f}",
        "Change": "-" if ml_v is None else f"{fin_v - ml_v:+.6f}",
        "Final result [kHz]": "" if key in ("A_MHz", "B_MHz", "C_MHz") else f"{fin_v * 1000.0:.3f}",
        "σ (1σ) [MHz]": f"{sigmas[key]:.6g}" if key in sigmas else "-",
        "Assessment": ("poorly determined" if key in poor else "OK") if key in sigmas else "-",
    })
if reliable:
    st_table(pd.DataFrame(rows))
    if poor:
        st.warning("Poorly determined by the data (uncertainty comparable to the value or to "
                   "the assumed prior): " + ", ".join(PARAM_LABEL[k] for k in params if k in poor)
                   + ". Do not treat these values as physical; report B, C and the "
                   "well-determined distortion constants, and mark the rest as undetermined.")
    if unc and unc.get("n_lines_used", 99) <= unc.get("n_parameters", 0) + 3:
        st.caption(f"Note: {unc['n_lines_used']} lines for {unc['n_parameters']} parameters, "
                   "so a small rms says nothing about the quality of the fit.")
    if not ref.get("refined", False):
        st.caption("DJ and DJK from the network alone are not reliable (the paper does not fit them either).")
else:
    st.caption("Parameters hidden because the fit is too weak.")
    with st.expander("Show anyway (unreliable)"):
        st_table(pd.DataFrame(rows))

# === PROBABILITIES ===
with st.expander("Classification probabilities"):
    pf = pd.DataFrame({"Type": [TYPE_LABEL.get(k, k) for k in conf],
                       "Probability [%]": [v * 100 for v in conf.values()]})
    figp = go.Figure(go.Bar(x=pf["Type"], y=pf["Probability [%]"],
                            text=pf["Probability [%]"].map(lambda v: f"{v:.2f}%"),
                            textposition="outside"))
    figp.update_layout(height=260, yaxis_range=[0, 115], margin=dict(l=20, r=20, t=20, b=20))
    st_plot(figp)

# === SPECTRUM ===
st.subheader("Spectrum: input data and model")
fig = go.Figure()
if show_peaks:
    fig.add_trace(stick_trace(peaks, 1.0, "Peaks from file", "#35B8E8"))
if show_predictions and len(lines):
    fig.add_trace(stick_trace(lines, 0.78, "Model lines", "#FF765E", dash="dash"))

shown = [a for a, on in ((peaks, show_peaks), (lines, show_predictions)) if on and len(a)]
if shown:
    allf = np.concatenate(shown)
    xmin, xmax = float(allf.min()), float(allf.max())
    pad = max((xmax - xmin) * 0.04, 100.0)
    x_range = [max(0.0, xmin - pad), xmax + pad]
else:
    x_range = [0, max_freq]

fig.update_layout(
    xaxis=dict(title="Frequency [MHz]", range=x_range, showgrid=True,
               gridcolor="rgba(150,150,150,0.18)", zeroline=False,
               rangeslider=dict(visible=True, thickness=0.08)),
    yaxis=dict(range=[-0.04, 1.12], showticklabels=False, showgrid=False, zeroline=False),
    height=520, hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=30, r=25, t=45, b=55),
)
st_plot(fig, config={"displaylogo": False, "scrollZoom": True})
st.caption("Stick heights are arbitrary: the file contains frequencies, not intensities.")
if len(lines):
    st.caption(f"Model lines are computed up to {lines.max():,.0f} MHz "
               f"(highest peak in the file: {peaks.max():,.0f} MHz). Peaks above the model "
               "range have no counterpart and will appear as unmatched.")

# === RESIDUALS ===
st.subheader("Residuals: peak − nearest model line")
figr = go.Figure()
figr.add_trace(go.Scatter(x=peaks, y=res, mode="markers", name="Residual",
                          marker=dict(size=7, color=np.where(matched, "#35B8E8", "#FF765E")),
                          hovertemplate="%{x:.3f} MHz<br>residual %{y:.4f} MHz<extra></extra>"))
figr.add_hline(y=0, line_color="gray")
figr.add_hrect(y0=-MATCH_TOL_MHZ, y1=MATCH_TOL_MHZ, fillcolor="rgba(150,150,150,0.12)", line_width=0)
figr.update_layout(height=300, xaxis_title="Frequency [MHz]", yaxis_title="Residual [MHz]",
                   margin=dict(l=30, r=25, t=20, b=45), showlegend=False)
st_plot(figr)
st.caption("Blue points: matched (|residual| ≤ 1 MHz). Red: peaks the model does not explain "
           "(other species, hyperfine structure, or a model error).")

# === TABLES ===
t1, t2, t3 = st.tabs(["Peaks from file", "Model lines", "Result (JSON)"])
with t1:
    st_table(pd.DataFrame({
        "No.": np.arange(1, len(peaks) + 1),
        "Frequency [MHz]": peaks,
        "Nearest model line [MHz]": nearest,
        "Residual [MHz]": res,
        "Matched": np.where(matched, "yes", "no"),
        "Assignment": np.where(matched, np.array(assign, dtype=object), ""),
    }))
with t2:
    st_table(pd.DataFrame({"No.": np.arange(1, len(lines_all) + 1),
                           "Predicted frequency [MHz]": lines_all,
                           "Transition": labels_all}))
with t3:
    st.json(result)


def build_report() -> str:
    """HTML report"""
    import datetime
    import html
    esc = html.escape
    pt = pd.DataFrame(rows).to_html(index=False, escape=True, border=0)
    fig_r = go.Figure(go.Scatter(x=peaks, y=res, mode="markers",
                                 marker=dict(color=np.where(matched, "#35B8E8", "#FF765E"))))
    fig_r.update_layout(title="Residuals (obs - model) [MHz]", height=320,
                        xaxis_title="Frequency [MHz]", yaxis_title="Residual [MHz]")
    peaks_tab = pd.DataFrame({"Frequency [MHz]": peaks, "Model [MHz]": nearest,
                              "Residual [MHz]": res, "Assignment": np.where(matched, assign, "")})
    warns = result.get("warnings") or []
    wl = "".join(f"<li>{esc(w)}</li>" for w in warns) or "<li>none</li>"
    probs = ", ".join(f"{TYPE_LABEL.get(k, k)}: {v*100:.1f}%" for k, v in conf.items())
    rms_txt = "-" if np.isnan(rms_all) else f"{rms_all:.4f} MHz"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>RAINet - report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse}}td,th{{border-bottom:1px solid #ccc;padding:4px 10px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}</style></head><body>
<h1>RAINet - spectrum analysis report</h1>
<p>Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}. Prototype result, to be verified in SPFIT/SPCAT.</p>
<h2>Summary</h2><ul>
<li>Molecule type: <b>{esc(TYPE_LABEL.get(kind, kind))}</b>{' (chosen manually)' if result.get('forced_kind') else ''}</li>
<li>Classifier probabilities: {esc(probs)}</li>
<li>Peaks matched (|r| ≤ {MATCH_TOL_MHZ:g} MHz): {int(matched.sum())} / {len(peaks)}; rms: {rms_txt}</li>
<li>Reliability: {'reliable' if reliable else 'NOT RELIABLE (fit too weak)'}</li></ul>
<h2>Parameters</h2>{pt}
<h2>Warnings</h2><ul>{wl}</ul>
<h2>Settings</h2><ul><li>Maximum frequency: {max_freq:g} MHz</li>
<li>Least-squares refinement: {'yes' if refine else 'no'}</li>
<li>Classification confidence threshold: {min_conf:.2f}</li>
<li>Molecule type: {esc(kind_choice)}</li></ul>
<h2>Spectrum and residuals</h2>{fig_r.to_html(full_html=False, include_plotlyjs="cdn")}
<h2>Peaks and assignments</h2>{peaks_tab.to_html(index=False, border=0, float_format=lambda x: f"{x:.4f}")}
<h2>Limitations</h2><p>Linear molecules, prolate symmetric tops (with sextic terms) and asymmetric tops
(a-type transitions only). No intensities, mixtures, hyperfine structure,
b/c-type transitions ora oblate molecules. The fit is a custom least-squares
fit, not SPFIT.</p>
</body></html>"""


d1, d2, d3 = st.columns(3)
d1.download_button("Download model lines (CSV)",
                   pd.DataFrame({"frequency_mhz": lines}).to_csv(index=False),
                   "rainet_predicted_frequencies.csv", "text/csv")
d2.download_button("Download result (JSON)", json.dumps(result, indent=2),
                   "rainet_result.json", "application/json")
d3.download_button("Download report (HTML)", build_report(), "rainet_report.html", "text/html")

with st.expander("Limitations"):
    st.write(
        "Supported: linear molecules, prolate symmetric tops (with sextic constants HJ, HJK, HKJ when "
        "they improve the fit) and asymmetric tops (a-type transitions only "
        "with quartic distortion constants), based on frequencies only. Not supported: "
        "intensities, mixtures, hyperfine and fine structure, b/c-type transitions and oblate. "
        "For lists containing several species the result may be "
        "meaningless, which shows as a small number of matched peaks and large residuals."
    )