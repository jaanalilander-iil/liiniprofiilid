"""
prepare_data.py
---------------
Teisendab Remix AVL ekspordi (4.0.3 Detailne liinide väljumine)
profiilide dashboardi jaoks sobivaks Parquet-failiks.

Kasutus:
    python prepare_data.py andmed.csv            # -> data/data_<periood>.parquet + manifest
    python prepare_data.py andmed.csv minu.parquet  # -> minu.parquet (manifest ei uuene)

Eeldab: pandas pyarrow  (pip install pandas pyarrow)
"""

import sys
import json
import os
import re
import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DATA_DIR = "data"  # alamkaust kuhu Parquet-failid kirjutatakse


def load_csv(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp1257", "latin-1"):
        try:
            df = pd.read_csv(path, skiprows=3, encoding=enc, low_memory=False)
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Ei suuda faili lugeda: {path}")


def extract_period_from_header(path: str) -> str:
    """Loe periood CSV esimesest reast (veerud 24-25)."""
    for enc in ("utf-8", "utf-8-sig", "cp1257", "latin-1"):
        try:
            header = pd.read_csv(path, nrows=1, header=None, encoding=enc)
            start = str(header.iloc[0, 24]).strip()[:10] if len(header.columns) > 24 else ""
            end   = str(header.iloc[0, 25]).strip()[:10] if len(header.columns) > 25 else ""
            if re.match(r"\d{4}-\d{2}-\d{2}", start) and re.match(r"\d{4}-\d{2}-\d{2}", end):
                return f"{start}_{end}"
            break
        except UnicodeDecodeError:
            continue
    # Kui päisest ei leia, tuleta failinimest
    m = re.search(r"(\d{4}-\d{2}-\d{2})[^\d]*(\d{4}-\d{2}-\d{2})", os.path.basename(path))
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return "periood_tundmatu"


def build_df(df: pd.DataFrame) -> pd.DataFrame:
    """Puhasta ja teisenda CSV -> tasane DataFrame ühe reaga peatuse kohta."""
    df = df.rename(columns={
        "Peatuse jrk":           "jrk",
        "Liin":                  "liin",
        "Veoots":                "veoots",
        "Peatus":                "peatus",
        "Valideerimisi":         "board",
        "Väljumisi":             "alight",
        "Pardal":                "pardal",
        "Planeeritud väljumine": "planned_dep",
    })

    df["jrk"] = pd.to_numeric(df["jrk"], errors="coerce")
    df = df[df["jrk"].notna()].copy()
    df["jrk"] = df["jrk"].astype(int)

    df["liin"]   = df["liin"].astype(str).str.strip()
    df["veoots"] = df["veoots"].astype(str).str.strip()
    df = df[df["liin"].str.len() > 0]
    df = df[df["liin"].str.lower() != "nan"]
    df = df[df["veoots"].str.len() > 0]
    df = df[df["veoots"].str.lower() != "nan"]

    df["board"]       = pd.to_numeric(df["board"],       errors="coerce").fillna(0)
    df["alight"]      = pd.to_numeric(df["alight"],      errors="coerce").fillna(0)
    df["pardal"]      = pd.to_numeric(df["pardal"],      errors="coerce").fillna(0)
    df["planned_dep"] = pd.to_datetime(df["planned_dep"], errors="coerce")

    print(f"  Kehtivaid ridu: {len(df):,}")

    # Unikaalne väljumine = veoots + kuupäev
    df["dep_date"] = df["planned_dep"].dt.normalize()

    # Tuleta kuupäev ja kellaaeg esimesest peatusest (jrk==min) iga väljumise kohta
    first = (
        df.sort_values("jrk")
        .groupby(["liin", "veoots", "dep_date"])["planned_dep"]
        .first()
        .reset_index()
        .rename(columns={"planned_dep": "dep_dt"})
    )
    df = df.merge(first, on=["liin", "veoots", "dep_date"], how="left")
    df["date"] = df["dep_dt"].dt.strftime("%d.%m.%Y").fillna("")
    df["time"] = df["dep_dt"].dt.strftime("%H:%M").fillna("")

    # Jäta ainult vajalikud veerud
    out = df[["liin", "veoots", "date", "time", "jrk", "peatus", "board", "alight", "pardal"]].copy()
    out["board"]  = out["board"].round(2)
    out["alight"] = out["alight"].round(2)
    out["pardal"] = out["pardal"].round(2)
    return out


def update_manifest(data_dir: str) -> None:
    """Skanni data/ kaust ja kirjuta manifest.json."""
    files = sorted(glob.glob(os.path.join(data_dir, "data_*.parquet")))
    entries = []
    for f in files:
        fname = os.path.basename(f)
        m = re.search(r"data_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.parquet", fname)
        if m:
            label = f"{m.group(1)} – {m.group(2)}"
        else:
            label = fname.replace("data_", "").replace(".parquet", "")
        entries.append({"file": f"data/{fname}", "label": label})

    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"  manifest.json uuendatud: {len(entries)} perioodi")
    for e in entries:
        print(f"    {e['label']}  ->  {e['file']}")


def main():
    if len(sys.argv) < 2:
        print("Kasutus: python prepare_data.py andmed.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    custom_out = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Loen: {csv_path}")
    df = load_csv(csv_path)
    print(f"  Ridu: {len(df):,}  |  Veerge: {len(df.columns)}")

    out = build_df(df)
    liine  = out["liin"].nunique()
    valja  = out.groupby(["liin", "veoots", "date"]).ngroups
    print(f"  Liine: {liine}  |  Väljumisi: {valja:,}  |  Ridu: {len(out):,}")

    if custom_out:
        parquet_path = custom_out
    else:
        os.makedirs(DATA_DIR, exist_ok=True)
        period       = extract_period_from_header(csv_path)
        parquet_path = os.path.join(DATA_DIR, f"data_{period}.parquet")

    table = pa.Table.from_pandas(out, preserve_index=False)
    pq.write_table(table, parquet_path, compression="snappy")

    size_mb = os.path.getsize(parquet_path) / 1024 / 1024
    print(f"Kirjutatud: {parquet_path}  ({size_mb:.1f} MB)")

    if not custom_out:
        print()
        update_manifest(DATA_DIR)


if __name__ == "__main__":
    main()
