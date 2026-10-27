"""
Field-level extraction scorer for ref_prompt_library_final_layout doctype_id=2
(Submission_Document_Commercial_Property).

Compares a Foundry extraction JSON against a ground-truth JSON.
Each field gets 1 / 0.5 / 0 using a comparator chosen by field TYPE,
not a single generic string match -- that is the main gap in the
"just diff the strings" version of this approach.

Usage: python3 score_extraction.py ground_truth.json foundry_output.json
"""

import json
import re
import sys
import difflib
from datetime import datetime

# ----------------------------------------------------------------------
# Field type classification -- this encodes the extraction_rules' intent
# ----------------------------------------------------------------------

ENUM_FIELDS = {
    # BRT-compatible controlled vocabulary fields: must match the category,
    # near-miss paraphrase is NOT a 1 for these -- they drive rating.
    "Fire Alarm", "Backup Power", "Sprinkler System", "Prior Carrier Departure",
    "Risk Management Program", "Claims Litigation / Disputes",
}

NUMERIC_FIELDS = {
    "HVAC System", "Plumbing System", "Electrical System", "Roof (Flat / TPO / EPDM)",
    "Overall Building Age", "Building Code Compliance", "Last Fire Inspection",
    "Vacancy Rate", "Insurance-to-Value (ITV)", "BI Period Adequacy",
    "5-Year Loss Count", "Number of Floors", "Number of Guest Rooms",
    "Years in Business", "Contents/FF&E Value", "Total Insured Value (TIV)",
}

DATE_FIELDS = {"Effective Date", "Expiration Date"}

NARRATIVE_FIELDS = {
    "Departure Reason", "Property Description", "Plumbing", "Fire Protection",
    "Name and Mailing Address",
}

# Everything else (City, State, Zip, Broker Name/Email/Phone, Class Code,
# Roof Type, Valuation, Security System, Organization Name/Type,
# Named Insured, construction_class, Primary Construction Type,
# occupancy_type, Primary Use, Line of Business, Address Type,
# Address Line 1, Property Email/Phone, Property Contact, Year Constructed,
# Prior Carrier, Limit of Liability, Underwriting Company, Country,
# Term Type, Coverage Currency, Settlement Currency, Producer Code,
# Site Area (acres), Building Area (sq ft)) falls through to LABEL match.


def normalize_text(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("–", "-").replace("—", "-")
    return s


def normalize_number(v):
    """Extract a single float from messy numeric-ish strings/ints."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v)
    s = s.replace(",", "")
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group()) if m else None


def normalize_date(v):
    if v is None:
        return None
    s = str(v).strip()
    fmts = ["%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"]
    for f in fmts:
        try:
            return datetime.strptime(s, f).date()
        except ValueError:
            continue
    return None  # unparseable


def flatten_for_compare(v):
    """If a field was returned as a nested object/list instead of a flat
    value, flatten it to a single string for narrative/label comparison
    rather than auto-failing or auto-passing it."""
    if isinstance(v, dict):
        return " ".join(f"{k}: {val}" for k, val in v.items())
    if isinstance(v, list):
        return " ".join(str(x) for x in v)
    return v


def score_label(gt, ex):
    """Exact-ish match for short categorical/text fields. 1 if normalized
    strings are equal or one contains the other meaningfully; 0.5 if they
    share the core term but differ in wording/formatting; else 0."""
    gt_n, ex_n = normalize_text(flatten_for_compare(gt)), normalize_text(flatten_for_compare(ex))
    if gt_n is None and ex_n is None:
        return 1.0, "both null"
    if gt_n is None or ex_n is None:
        return 0.0, "one null, other not"
    if gt_n == ex_n:
        return 1.0, "exact match"
    if gt_n in ex_n or ex_n in gt_n:
        return 0.5, "substring match (extra/missing wording)"
    ratio = difflib.SequenceMatcher(None, gt_n, ex_n).ratio()
    if ratio >= 0.75:
        return 0.5, f"fuzzy match ratio={ratio:.2f}"
    return 0.0, f"no match ratio={ratio:.2f}"


def score_enum(gt, ex):
    """Controlled-vocabulary fields: near-miss paraphrase that doesn't
    map to the SAME category is a hard 0, because these drive rating
    credits/surcharges -- a 'close' sprinkler label in the wrong bucket
    is operationally wrong, not partially right."""
    gt_n, ex_n = normalize_text(flatten_for_compare(gt)), normalize_text(flatten_for_compare(ex))
    if gt_n is None and ex_n is None:
        return 1.0, "both null"
    if gt_n is None or ex_n is None:
        return 0.0, "one null, other not"
    if gt_n == ex_n:
        return 1.0, "exact enum match"
    if gt_n in ex_n or ex_n in gt_n:
        return 0.5, "same category, extra/missing qualifier text"
    return 0.0, "different enum category / not a controlled label"


def score_numeric(gt, ex):
    gt_n, ex_n = normalize_number(gt), normalize_number(ex)
    if gt is None and ex is None:
        return 1.0, "both null"
    if gt_n is None or ex_n is None:
        return 0.0, "unparseable or one null"
    if gt_n == ex_n:
        return 1.0, "exact numeric match"
    if gt_n != 0 and abs(gt_n - ex_n) / abs(gt_n) <= 0.02:
        return 1.0, "numeric match within 2% rounding tolerance"
    # off by a small integer amount, e.g. age off-by-one from calendar edge
    if abs(gt_n - ex_n) <= 1 and gt_n < 200:
        return 0.5, f"off by {abs(gt_n - ex_n)} (e.g. not reduced to a plain integer, or calendar edge)"
    return 0.0, f"numeric mismatch gt={gt_n} ex={ex_n}"


def score_date(gt, ex):
    if gt is None and ex is None:
        return 1.0, "both null"
    if gt is None or ex is None:
        return 0.0, "one null, other not"
    gt_d, ex_d = normalize_date(gt), normalize_date(ex)
    if gt_d is None or ex_d is None:
        return 0.0, "unparseable date format"
    if gt_d == ex_d:
        if str(gt).strip() == str(ex).strip():
            return 1.0, "exact value and format match"
        return 0.5, "correct date, wrong format (rule specifies DD/MM/YYYY)"
    return 0.0, f"date mismatch gt={gt_d} ex={ex_d}"


def score_narrative(gt, ex):
    gt_n, ex_n = normalize_text(flatten_for_compare(gt)), normalize_text(flatten_for_compare(ex))
    if gt_n is None and ex_n is None:
        return 1.0, "both null"
    if gt_n is None or ex_n is None:
        return 0.0, "one null, other not"
    ratio = difflib.SequenceMatcher(None, gt_n, ex_n).ratio()
    if ratio >= 0.85:
        return 1.0, f"near-verbatim match ratio={ratio:.2f}"
    if ratio >= 0.40:
        return 0.5, f"same substance, condensed/reworded ratio={ratio:.2f}"
    return 0.0, f"substance differs ratio={ratio:.2f}"


def classify_and_score(field, gt, ex):
    if field in ENUM_FIELDS:
        return score_enum(gt, ex)
    if field in NUMERIC_FIELDS:
        return score_numeric(gt, ex)
    if field in DATE_FIELDS:
        return score_date(gt, ex)
    if field in NARRATIVE_FIELDS:
        return score_narrative(gt, ex)
    return score_label(gt, ex)


def main():
    gt_path, ex_path = sys.argv[1], sys.argv[2]
    with open(gt_path) as f:
        gt = json.load(f)
    with open(ex_path) as f:
        ex = json.load(f)

    gt_fields = {k: v for k, v in gt.items() if not k.startswith("_")}

    rows = []
    total = 0.0
    for field, gt_val in gt_fields.items():
        ex_val = ex.get(field, "<<MISSING KEY>>")
        if ex_val == "<<MISSING KEY>>":
            s, reason = 0.0, "field missing entirely from Foundry output"
        else:
            s, reason = classify_and_score(field, gt_val, ex_val)
        total += s
        rows.append((field, s, reason, gt_val, ex_val))

    n = len(gt_fields)
    pct = 100 * total / n

    rows.sort(key=lambda r: r[1])  # worst first

    print(f"{'FIELD':<32} {'SCORE':<6} REASON")
    print("-" * 110)
    for field, s, reason, gt_val, ex_val in rows:
        marker = "FAIL" if s == 0 else ("PART" if s == 0.5 else "PASS")
        print(f"{field:<32} {s:<6} [{marker}] {reason}")

    print("\n" + "=" * 60)
    print(f"TOTAL FIELDS:      {n}")
    print(f"SUM OF SCORES:     {total}")
    print(f"OVERALL ACCURACY:  {pct:.1f}%")
    exact = sum(1 for r in rows if r[1] == 1.0)
    partial = sum(1 for r in rows if r[1] == 0.5)
    fail = sum(1 for r in rows if r[1] == 0.0)
    print(f"  Exact (1.0):  {exact}")
    print(f"  Partial (0.5): {partial}")
    print(f"  Fail (0.0):    {fail}")

    # dump machine-readable results too
    out = {
        "total_fields": n,
        "sum_scores": total,
        "overall_accuracy_pct": round(pct, 2),
        "exact_count": exact,
        "partial_count": partial,
        "fail_count": fail,
        "field_results": [
            {"field": f, "score": s, "reason": r, "ground_truth": gv, "foundry_output": ev}
            for f, s, r, gv, ev in rows
        ],
    }
    out_path = ex_path.rsplit(".", 1)[0] + "_scored.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nDetailed results written to: {out_path}")


if __name__ == "__main__":
    main()
