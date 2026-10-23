"""DataDrift - Data Engineering module (Member 1).
Sample data, issue injection, detection (-> incident JSON) and safe cleaning."""
import difflib
import numpy as np
import pandas as pd

EXPECTED_SCHEMA = {
    "order_id": "int", "customer_id": "int", "product": "str", "quantity": "int",
    "unit_price": "float", "order_date": "str", "country": "str", "order_total": "float",
}
CATALOG = {"Wireless Mouse": 799, "USB-C Cable": 299, "Laptop Stand": 1499,
           "Headphones": 2499, "Keyboard": 1799, "Webcam": 2999}


# ---------- sample data ----------
def make_clean_data(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    product = rng.choice(list(CATALOG), n)
    qty = rng.integers(1, 5, n)
    price = np.array([CATALOG[p] for p in product], dtype=float)
    return pd.DataFrame({
        "order_id": np.arange(10001, 10001 + n),
        "customer_id": rng.integers(1, 400, n),
        "product": product,
        "quantity": qty,
        "unit_price": price,
        "order_date": pd.date_range("2026-10-01", periods=n, freq="25min").strftime("%Y-%m-%d %H:%M"),
        "country": rng.choice(["India", "UAE", "Singapore", "UK"], n, p=[.6, .15, .15, .1]),
        "order_total": qty * price,
    })


def inject_issues(df, seed=7, dup_n=40, spike_n=8):
    """Corrupt a clean frame: nulls in the late batch, x100 totals, renamed column, retry duplicates."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    tail = d.index[int(len(d) * .75):]
    d["customer_id"] = d["customer_id"].astype(float)
    d.loc[rng.choice(tail, int(len(tail) * .30), replace=False), "customer_id"] = np.nan
    d.loc[rng.choice(tail, int(len(tail) * .10), replace=False), "unit_price"] = np.nan
    d.loc[rng.choice(d.index, spike_n, replace=False), "order_total"] *= 100
    d = d.rename(columns={"country": "country_code"})
    pos = rng.choice(len(d), dup_n, replace=False)
    d = pd.concat([d, d.iloc[pos]]).sort_index(kind="stable").reset_index(drop=True)
    return d


# ---------- detection ----------
def _sev(pct):
    return "high" if pct >= 5 else "medium" if pct >= 1 else "low"


def detect_incidents(df):
    n, inc = len(df), []

    def add(type_, severity, title, column, affected, evidence):
        inc.append({"id": f"INC-{len(inc) + 1:03d}", "type": type_, "severity": severity,
                    "title": title, "column": column, "affected_rows": int(affected),
                    "total_rows": int(n), "evidence": evidence})

    # schema drift: missing / new columns, likely renames
    missing = [c for c in EXPECTED_SCHEMA if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_SCHEMA]
    if missing or extra:
        rename = {}
        for m in missing:
            hit = difflib.get_close_matches(m, extra, n=1, cutoff=0.5)
            if hit:
                rename[m] = hit[0]
        add("schema_drift", "high", "Schema changed vs expected contract", ", ".join(missing + extra), n,
            {"missing_columns": missing, "new_columns": extra, "likely_rename": rename,
             "renamed_column_values_match_dtype": bool(all(
                 pd.api.types.is_object_dtype(df[v]) == (EXPECTED_SCHEMA[k] == "str")
                 for k, v in rename.items()))})
    for col, kind in EXPECTED_SCHEMA.items():
        if col in df.columns and kind in ("int", "float") and not pd.api.types.is_numeric_dtype(df[col]):
            add("schema_drift", "high", f"Type drift in '{col}'", col, n,
                {"expected_type": kind, "actual_dtype": str(df[col].dtype)})

    # duplicates
    key = "order_id" if "order_id" in df.columns else df.columns[0]
    dup_mask = df.duplicated(keep="first")
    if dup_mask.any():
        d = df[dup_mask]
        prev_same = (df[key] == df[key].shift(1))[dup_mask].mean()
        ev = {"duplicate_rows": int(dup_mask.sum()), "duplicate_pct": round(100 * dup_mask.mean(), 2),
              "all_exact_copies": bool(df.duplicated(key, keep="first").sum() == dup_mask.sum()),
              "pct_directly_after_original": round(100 * float(prev_same), 1),
              "sample_ids": [int(x) for x in d[key].head(5)] if pd.api.types.is_numeric_dtype(d[key]) else list(d[key].head(5))}
        if "order_total" in df.columns and pd.api.types.is_numeric_dtype(df["order_total"]):
            ev["revenue_inflated"] = round(float(d["order_total"].sum()), 2)
        add("duplicate_records", _sev(ev["duplicate_pct"]), "Duplicate records", key, dup_mask.sum(), ev)

    # missing values
    for col in df.columns:
        k = int(df[col].isna().sum())
        if k:
            rows = np.flatnonzero(df[col].isna().to_numpy())
            add("missing_values", _sev(100 * k / n), f"Missing values in '{col}'", col, k,
                {"null_count": k, "null_pct": round(100 * k / n, 2),
                 "pct_in_last_quarter_of_file": round(100 * float((rows >= n * .75).mean()), 1),
                 "first_null_row": int(rows[0]), "last_null_row": int(rows[-1])})

    # value inconsistency: total != quantity * unit_price
    need = {"quantity", "unit_price", "order_total"}
    if need <= set(df.columns) and all(pd.api.types.is_numeric_dtype(df[c]) for c in need):
        exp = df["quantity"] * df["unit_price"]
        ok = exp.notna() & df["order_total"].notna()
        bad = ok & ~np.isclose(df["order_total"], exp, rtol=0.01)
        if bad.any():
            ratio = (df.loc[bad, "order_total"] / exp[bad])
            add("value_inconsistency", "high", "order_total does not match quantity x unit_price",
                "order_total", bad.sum(),
                {"mismatch_rows": int(bad.sum()), "mismatch_pct": round(100 * bad.mean(), 2),
                 "median_ratio_total_to_expected": round(float(ratio.median()), 2),
                 "min_ratio": round(float(ratio.min()), 2), "max_ratio": round(float(ratio.max()), 2),
                 "excess_revenue": round(float((df.loc[bad, "order_total"] - exp[bad]).sum()), 2),
                 "sample_ids": [int(x) for x in df.loc[bad, key].head(5)]})
    return inc


# ---------- cleaning ----------
def clean_data(df, incidents):
    """Apply only safe, reversible fixes. Returns (clean_df, fix_log)."""
    d, log = df.copy(), []
    for i in incidents:
        if i["type"] == "schema_drift":
            ren = i["evidence"].get("likely_rename", {})
            if ren:
                d = d.rename(columns={v: k for k, v in ren.items()})
                log.append(f"Renamed columns back to contract: {ren}")
    before = len(d)
    d = d.drop_duplicates().reset_index(drop=True)
    if before - len(d):
        log.append(f"Dropped {before - len(d)} exact duplicate rows")
    if "unit_price" in d and "product" in d:
        mode = d.groupby("product")["unit_price"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
        m = d["unit_price"].isna()
        d.loc[m, "unit_price"] = d.loc[m, "product"].map(mode)
        if m.sum():
            log.append(f"Filled {int(m.sum())} missing unit_price values from each product's usual price")
    if {"quantity", "unit_price", "order_total"} <= set(d.columns):
        exp = d["quantity"] * d["unit_price"]
        bad = ~np.isclose(d["order_total"], exp, rtol=0.01) & exp.notna()
        if bad.sum():
            d.loc[bad, "order_total"] = exp[bad]
            log.append(f"Recalculated order_total for {int(bad.sum())} rows (quantity x unit_price)")
    if "customer_id" in d and d["customer_id"].isna().any():
        d["needs_review"] = d["customer_id"].isna()
        log.append(f"Flagged {int(d['needs_review'].sum())} rows with missing customer_id as needs_review (not guessed)")
    return d, log
