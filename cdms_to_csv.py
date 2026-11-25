#!/usr/bin/env python3
"""
Convert the CDMS XSAMS/XML export into CSV files.

Outputs:
  1) transitions.csv  - experimental and predicted transition data
  2) peak_list.csv   - one experimental frequency per row, for rainet_rotational_ml.py

It does not parse a raw XSAMS document.
"""

import argparse
import csv
import re
from html.parser import HTMLParser
from pathlib import Path


class Node:
    def __init__(self, tag: str, attrs: dict[str, str | None] | None = None):
        self.tag = tag
        self.attrs = attrs or {}
        self.children: list[Node | str] = []

    def text(self) -> str:
        return " ".join(
            child if isinstance(child, str) else child.text()
            for child in self.children
        )

    def descendants(self, tag: str) -> list["Node"]:
        found = []
        for child in self.children:
            if isinstance(child, Node):
                if child.tag == tag:
                    found.append(child)
                found.extend(child.descendants(tag))
        return found


class TreeParser(HTMLParser):
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
            "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("root")
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = Node(tag, dict(attrs))
        self.stack[-1].children.append(node)
        if tag not in self.VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].children.append(Node(tag, dict(attrs)))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        if data:
            self.stack[-1].children.append(data)


def first_node(nodes, tag):
    for node in nodes:
        if node.tag == tag:
            return node
    return None

def numeric_values(node):
    if node is None:
        return []
    values = []
    for div in node.descendants("div"):
        if any(isinstance(c, Node) for c in div.children):
            continue  # tylko liście, bez zagnieżdżonych div
        value = re.sub(r"\s+", " ", div.text()).strip()
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?", value):
            values.append(value)
    return values

def extract_qn(text: str, label: str, qn: str):
    """
    Quantum number (J, K, Ka, Kc, vi) from an 'I.S.:' or 'F.S.:' block (up to the next block).
    """
    block = re.search(rf"{re.escape(label)}:(.*?)(?:[IF]\.S\.:|$)", text, re.S)
    if not block:
        return ""
    m = re.search(rf"\b{qn}\s*:\s*(\d+)", block.group(1))
    return int(m.group(1)) if m else ""


def parse_transition(li, molecule):
    anchors = li.descendants("a")
    if not anchors:
        return None

    # In the CDMS export the first anchor is the measured/observed entry;
    # the second anchor (when present) is the model-predicted entry.
    observed_anchor = anchors[0]
    predicted_anchor = anchors[1] if len(anchors) > 1 else None
    observed = numeric_values(observed_anchor)
    predicted = numeric_values(predicted_anchor)

    if not observed:
        return None

    full_text = li.text()
    title = observed_anchor.attrs.get("title") or ""
    source_match = re.search(r"Source:[ \t]*([^\n]*)", title)
    method_match = re.search(r"Method:[ \t]*([^\n]*)", title)

    return {
        "molecule": molecule,
        "frequency_mhz": observed[0],
        "uncertainty_mhz": observed[1] if len(observed) > 1 else "",
        "intensity": observed[2] if len(observed) > 2 else "",
        "frequency_predicted_mhz": predicted[0] if predicted else "",
        "uncertainty_predicted_mhz": predicted[1] if len(predicted) > 1 else "",
        "Ka_lower": extract_qn(full_text, "I.S.", "Ka"),
        "Kc_lower": extract_qn(full_text, "I.S.", "Kc"),
        "Ka_upper": extract_qn(full_text, "F.S.", "Ka"),
        "Kc_upper": extract_qn(full_text, "F.S.", "Kc"),
        "J_lower": extract_qn(full_text, "I.S.", "J"),
        "K_lower": extract_qn(full_text, "I.S.", "K"),
        "vi_lower": extract_qn(full_text, "I.S.", "vi"),
        "J_upper": extract_qn(full_text, "F.S.", "J"),
        "K_upper": extract_qn(full_text, "F.S.", "K"),
        "vi_upper": extract_qn(full_text, "F.S.", "vi"),
        "source": source_match.group(1).strip() if source_match else "",
        "method": method_match.group(1).strip() if method_match else "",
    }


def parse_cdms_export(input_path: Path):
    raw = input_path.read_text(encoding="utf-8-sig", errors="replace")
    parser = TreeParser()
    parser.feed(raw)

    molecule = ""
    m = re.search(r"XCDMS-\d+:\s*([^\s<]+)", raw)  # np. "XCDMS-199: CH3CCH"
    if m:
        molecule = m.group(1)

    radiative = next(
        (n for n in parser.root.descendants("fieldset")
         if n.attrs.get("id") == "list_radiative"),
        None,
    )
    if radiative is None:
        raise ValueError(
            "Section list_radiative not found. "
            "This tool expects an export of a CDMS page, not a raw XSAMS file."
        )

    rows = []
    # Only the outer <li> items of the transition list
    # the inner <ul> holding the frequency is not a separate transition.
    top = [u for u in radiative.children if isinstance(u, Node) and u.tag == "ul"]
    for ul in top:
        for child in ul.children:
            if isinstance(child, Node) and child.tag == "li":
                row = parse_transition(child, molecule)
                if row:
                    rows.append(row)

    if not rows:
        raise ValueError("Found the Transitions section but could not read any rows.")

    return rows


def write_csv(path: Path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Convert a CDMS export to transitions.csv and peak_list.csv"
    )
    ap.add_argument("input", type=Path, help="CDMS export file (.md/.html/.txt)")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="Output directory (default: cdms_NAME, where NAME comes from "
                         "the input file spectrum_NAME.csv)")
    ap.add_argument("--include-predicted-in-peaks", action="store_true",
                    help="Also add the predicted frequencies to peak_list.csv "
                         "(by default only the values from the first CDMS entry are used).")
    ap.add_argument("--max-freq", type=float, default=300000.0,
                    help="Upper limit [MHz] for peak_list.csv (default 300000 = range of the RAINet model)")
    ap.add_argument("--rot-only", action="store_true", default=True,
                    help="Only vi=0->vi=0 lines in peak_list.csv (enabled by default)")
    ap.add_argument("--r-branch", action="store_true",
                    help="Only a-type R-branch lines in peak_list.csv (J+1<-J, K unchanged or "
                         "Ka unchanged and Kc+1) - the only transitions the RAINet model describes")
    args = ap.parse_args()

    rows = parse_cdms_export(args.input)

    if args.outdir is None:
        stem = args.input.stem
        if stem.lower().startswith("spectrum_") and len(stem) > len("spectrum_"):
            name = stem[len("spectrum_"):]          # spectrum_H2CO -> H2CO
        else:
            name = rows[0]["molecule"] or stem      # otherwise: the name from the CDMS file
        name = re.sub(r'[\\/:*?"<>|\s]+', "_", name)  # safe directory name
        args.outdir = Path(f"cdms_{name}")
    columns = [
        "molecule", "frequency_mhz", "uncertainty_mhz", "intensity",
        "frequency_predicted_mhz", "uncertainty_predicted_mhz",
        "J_lower", "K_lower", "Ka_lower", "Kc_lower", "vi_lower",
        "J_upper", "K_upper", "Ka_upper", "Kc_upper", "vi_upper",
        "source", "method",
    ]
    transitions_path = args.outdir / "transitions.csv"
    write_csv(transitions_path, rows, columns)

    def is_r_a(r):
        if r["J_lower"] == "" or r["J_upper"] == "":
            return True               # no labels: cannot tell
        if r["J_upper"] - r["J_lower"] != 1:
            return False
        if r["Ka_lower"] != "" and r["Ka_upper"] != "":
            if r["Ka_lower"] != r["Ka_upper"]:
                return False
            if r["Kc_lower"] != "" and r["Kc_upper"] != "":
                return r["Kc_upper"] - r["Kc_lower"] == 1
            return True
        if r["K_lower"] != "" and r["K_upper"] != "":
            return r["K_lower"] == r["K_upper"]
        return True

    def ok(r):
        if args.r_branch and not is_r_a(r):
            return False
        if args.rot_only and (r["vi_lower"] not in ("", 0) or r["vi_upper"] not in ("", 0)):
            return False
        return True

    freqs = []
    for row in rows:
        if not ok(row):
            continue
        for key, use in (("frequency_mhz", True),
                         ("frequency_predicted_mhz", args.include_predicted_in_peaks)):
            if use and row[key] != "":
                f = float(row[key])
                if 0 < f <= args.max_freq:
                    freqs.append(f)
    uniq = sorted(set(freqs))
    peak_rows = [{"frequency_mhz": f"{f:.4f}"} for f in uniq]

    peak_path = args.outdir / "peak_list.csv"
    write_csv(peak_path, peak_rows, ["frequency_mhz"])

    print(f"Molecule: {rows[0]['molecule']}")
    print(f"Transitions read: {len(rows)}")
    allf = [float(r["frequency_mhz"]) for r in rows]
    from collections import Counter
    dj = Counter((r["J_upper"] - r["J_lower"]) if r["J_lower"] != "" and r["J_upper"] != "" else None
                 for r in rows)
    print("Distribution of DeltaJ (J_upper - J_lower):", dict(dj))
    print(f"Frequency range in the file: {min(allf):.4f} - {max(allf):.4f} MHz")
    print(f"To peak_list.csv: {len(peak_rows)} unique lines "
          f"(from {len(freqs)}; limit {args.max_freq:.0f} MHz)")
    if len(peak_rows) < 2:
        print("WARNING: too few lines in range for RAINet.")
    print(f"Saved transition table: {transitions_path}")
    print(f"Saved frequency list: {peak_path}")
    print("Note: the frequency_mhz header assumes the CDMS frequencies are in MHz.")


if __name__ == "__main__":
    main()
