"""
Data Preparation & Pipeline Project - Automobile dataset
=========================================================
Alur ETL:
    Extract   : baca CSV mentah dari data/raw/
    Transform : inspeksi -> cleaning -> transformasi
    Load      : simpan hasil ke data/processed/automobileEDA_processed.csv

Jalankan dari root project:
    python src/pipeline.py
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Konfigurasi
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "automobileEDA_dirty_training.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "automobileEDA_processed.csv"

DATE_COL = "transaction_date"

# Kolom kategorikal bertipe teks yang diseragamkan (strip + lowercase)
CATEGORICAL_COLS = [
    "make", "aspiration", "num-of-doors", "body-style", "drive-wheels",
    "engine-location", "engine-type", "num-of-cylinders", "fuel-system",
    "horsepower-binned",
]

# Sinonim label yang dipetakan ke label baku
SYNONYMS = {"drive-wheels": {"awd": "4wd"}}

# Kolom numerik kontinu yang dinormalisasi dengan Min-Max Scaling.
# 'price' (target), 'symboling' (skor ordinal) dan kolom biner tidak di-scale.
MINMAX_COLS = [
    "normalized-losses", "wheel-base", "length", "width", "height",
    "curb-weight", "engine-size", "bore", "stroke", "compression-ratio",
    "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "city-L/100km",
]

# Kolom nominal -> One-Hot Encoding
ONEHOT_COLS = [
    "aspiration", "body-style", "drive-wheels", "engine-location",
    "engine-type", "fuel-system",
]

# Kolom ordinal -> Ordinal Encoding (urutan bermakna)
ORDINAL_MAPS = {
    "num-of-cylinders": {"two": 2, "three": 3, "four": 4, "five": 5,
                         "six": 6, "eight": 8, "twelve": 12},
    "num-of-doors": {"two": 2, "four": 4},
    "horsepower-binned": {"low": 0, "medium": 1, "high": 2},
}


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def file_hash(path):
    """SHA-256 file, dipakai untuk membuktikan dataset mentah tidak berubah."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# EXTRACT
# --------------------------------------------------------------------------- #
def load_data(path=RAW_PATH):
    """Membaca dataset mentah (read-only) ke DataFrame."""
    banner("1. LOAD DATA")
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
    df = pd.read_csv(path)
    print(f"Membaca: {Path(path).relative_to(PROJECT_ROOT)}")
    print(f"Berhasil dimuat: {df.shape[0]} baris x {df.shape[1]} kolom")
    return df


# --------------------------------------------------------------------------- #
# TRANSFORM - 1. Inspeksi
# --------------------------------------------------------------------------- #
def inspect_data(df):
    """Pemeriksaan awal dataset; hasil ditampilkan ke terminal."""
    banner("2. DATA INSPECTION")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    print("\n[a] Lima baris pertama:")
    print(df.head())

    print(f"\n[b] Jumlah baris & kolom: {df.shape[0]} baris, {df.shape[1]} kolom")

    print("\n[c] Nama kolom:")
    print(list(df.columns))

    print("\n[d] Tipe data tiap kolom:")
    print(df.dtypes.astype(str).to_string())

    missing = df.isna().sum()
    print("\n[e] Missing values per kolom (hanya yang > 0):")
    print(missing[missing > 0].to_string())
    print(f"Total missing values: {int(missing.sum())}")

    print(f"\n[f] Duplicate records (identik persis): {int(df.duplicated().sum())}")

    print("\n[g] Nilai unik kolom kategorikal:")
    for col in CATEGORICAL_COLS:
        print(f"  - {col} ({df[col].nunique(dropna=True)} unik): "
              f"{sorted(df[col].dropna().astype(str).unique().tolist())}")

    # --- Temuan tambahan ---
    print("\n[h] Temuan tambahan:")
    ws = {c: int((df[c].dropna() != df[c].dropna().str.strip()).sum())
          for c in CATEGORICAL_COLS if df[c].dtype == "object" or str(df[c].dtype) == "str"}
    ws = {c: n for c, n in ws.items() if n}
    print(f"  - Kolom dengan spasi berlebih: {ws if ws else 'tidak ada'}")
    for c in CATEGORICAL_COLS:
        s = df[c].dropna().astype(str)
        n_raw = s.nunique()
        n_norm = s.str.strip().str.lower().nunique()
        if n_raw != n_norm:
            print(f"  - '{c}': {n_raw} label mentah -> {n_norm} label setelah "
                  f"strip+lowercase (penulisan tidak konsisten)")
    print(f"  - '{DATE_COL}' bertipe teks dengan 4 format campuran "
          f"(YYYY-MM-DD, DD/MM/YYYY, MM-DD-YYYY, DD-Mon-YYYY)")
    print("  - Kolom numerik berskala berbeda-beda (mis. length ~0.8, "
          "curb-weight ~2500, price ~13000)")
    return missing


# --------------------------------------------------------------------------- #
# TRANSFORM - 2. Cleaning
# --------------------------------------------------------------------------- #
def _parse_date(value):
    """Parse satu string tanggal berdasarkan polanya (format campuran)."""
    if pd.isna(value):
        return pd.NaT
    s = str(value).strip()
    if "/" in s:                                  # 02/01/2025 -> DD/MM/YYYY
        return pd.to_datetime(s, format="%d/%m/%Y")
    if len(s) == 10 and s[4] == "-":              # 2025-01-01 -> ISO
        return pd.to_datetime(s, format="%Y-%m-%d")
    if len(s) == 11 and s[3:6].isalpha():         # 04-Jan-2025 -> DD-Mon-YYYY
        return pd.to_datetime(s, format="%d-%b-%Y")
    return pd.to_datetime(s, format="%m-%d-%Y")   # 01-03-2025 -> MM-DD-YYYY


def _fill_group_median(df, col, group_levels):
    """Isi NaN dengan median grup; coba tiap level grup berurutan, lalu global."""
    for keys in group_levels:
        med = df.groupby(keys)[col].transform("median")
        df[col] = df[col].fillna(med)
    df[col] = df[col].fillna(df[col].median())


def clean_data(df_raw):
    """
    Membersihkan dataset. Mengembalikan (df_bersih, ringkasan).
    DataFrame input tidak diubah (bekerja pada salinan).
    """
    banner("3. DATA CLEANING")
    df = df_raw.copy()
    rows_before = len(df)
    missing_before = df.isna().sum()
    changed_cols = set()

    # --- 3.1 Strip spasi + lowercase + sinonim ------------------------------
    for col in CATEGORICAL_COLS:
        before = df[col].copy()
        df[col] = df[col].str.strip().str.lower()
        if col in SYNONYMS:
            df[col] = df[col].replace(SYNONYMS[col])
        if not before.equals(df[col]) and (before.fillna("") != df[col].fillna("")).any():
            changed_cols.add(col)
            n = int((before.fillna("") != df[col].fillna("")).sum())
            print(f"[standardisasi] '{col}': {n} nilai diseragamkan "
                  f"(strip/lowercase{'/sinonim' if col in SYNONYMS else ''})")

    # --- 3.2 Perbaiki tipe data tanggal --------------------------------------
    df[DATE_COL] = df[DATE_COL].map(_parse_date)
    changed_cols.add(DATE_COL)
    print(f"[tipe data] '{DATE_COL}': string campuran -> datetime64 "
          f"({int(df[DATE_COL].isna().sum())} nilai kosong)")

    # --- 3.3 Hapus duplikat (setelah standardisasi, agar duplikat tersembunyi terdeteksi)
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    print(f"[duplikat] {n_dup} baris duplikat dihapus (keep='first')")

    # --- 3.4 Missing values ---------------------------------------------------
    # a) transaction_date: data berupa deret harian berurutan -> interpolasi tetangga
    n_date = int(df[DATE_COL].isna().sum())
    if n_date:
        base = df[DATE_COL].min()
        days = (df[DATE_COL] - base).dt.days.astype("float")
        days = days.interpolate(method="linear", limit_direction="both").round()
        df[DATE_COL] = base + pd.to_timedelta(days, unit="D")
    print(f"[missing] '{DATE_COL}': {n_date} nilai diisi via interpolasi linear "
          f"antar tanggal tetangga (deret harian berurutan)")

    # b) price = variabel target -> baris tanpa harga dihapus (tidak diimputasi)
    n_price = int(df["price"].isna().sum())
    df = df.dropna(subset=["price"]).reset_index(drop=True)
    print(f"[missing] 'price': {n_price} baris dihapus (target tidak boleh diisi tebakan)")
    changed_cols.add("price")

    # c) make: tidak bisa diturunkan dari kolom lain -> label 'unknown'
    n_make = int(df["make"].isna().sum())
    df["make"] = df["make"].fillna("unknown")
    print(f"[missing] 'make': {n_make} nilai diisi 'unknown' (tidak dapat ditebak, baris tetap dipertahankan)")
    if n_make:
        changed_cols.add("make")

    # d) num-of-doors: modus per body-style
    n_doors = int(df["num-of-doors"].isna().sum())
    mode_by_body = df.groupby("body-style")["num-of-doors"].transform(
        lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
    df["num-of-doors"] = df["num-of-doors"].fillna(mode_by_body)
    print(f"[missing] 'num-of-doors': {n_doors} nilai diisi modus per body-style")
    if n_doors:
        changed_cols.add("num-of-doors")

    # e) horsepower & stroke: median grup (numerik, berkorelasi dg spesifikasi mesin)
    n_hp = int(df["horsepower"].isna().sum())
    _fill_group_median(df, "horsepower",
                       [["engine-size", "horsepower-binned"], ["horsepower-binned"],
                        ["engine-size"]])
    print(f"[missing] 'horsepower': {n_hp} nilai diisi median per engine-size + "
          f"horsepower-binned (fallback: horsepower-binned / engine-size / global), "
          f"sehingga konsisten dengan label bin yang sudah ada")
    n_stroke = int(df["stroke"].isna().sum())
    _fill_group_median(df, "stroke", [["make", "engine-type"], ["engine-type"]])
    print(f"[missing] 'stroke': {n_stroke} nilai diisi median per make+engine-type "
          f"(fallback: engine-type / global)")
    changed_cols.update(["horsepower", "stroke"])

    # f) horsepower-binned: hitung ulang dari horsepower memakai ambang yang
    #    teramati pada data (Low <= 101, Medium 102-155, High >= 156)
    n_bin = int(df["horsepower-binned"].isna().sum())
    known = df.dropna(subset=["horsepower-binned"])
    hp = known.groupby("horsepower-binned")["horsepower"].agg(["min", "max"])
    low_max, med_max = hp.loc["low", "max"], hp.loc["medium", "max"]
    derived = np.where(df["horsepower"] <= low_max, "low",
                       np.where(df["horsepower"] <= med_max, "medium", "high"))
    df["horsepower-binned"] = df["horsepower-binned"].fillna(pd.Series(derived, index=df.index))
    print(f"[missing] 'horsepower-binned': {n_bin} nilai diturunkan dari horsepower "
          f"(ambang: low<={low_max:g}, medium<={med_max:g}, high>{med_max:g})")
    if n_bin:
        changed_cols.add("horsepower-binned")

    # --- 3.5 Pemeriksaan nilai tidak wajar -----------------------------------
    print("\n[nilai tidak wajar] Pemeriksaan:")
    num_cols = df.select_dtypes(include="number").columns
    non_positive = [c for c in num_cols
                    if c not in ("symboling", "diesel", "gas") and (df[c] <= 0).any()]
    print(f"  - Kolom dengan nilai <= 0 (tidak wajar): {non_positive if non_positive else 'tidak ada'}")
    fuel_ok = int(((df["diesel"] + df["gas"]) == 1).all())
    print(f"  - diesel + gas selalu = 1: {'ya' if fuel_ok else 'TIDAK'}")
    for c in ["price", "horsepower", "engine-size", "compression-ratio"]:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        n_out = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
        print(f"  - Outlier IQR '{c}': {n_out} baris -> DIPERTAHANKAN "
              f"(mobil mewah/diesel yang valid, bukan error input)")

    # --- Ringkasan ---
    missing_after = df.isna().sum()
    print("\n[ringkasan cleaning]")
    print(f"  Jumlah baris  : sebelum={rows_before}, sesudah={len(df)} "
          f"(dihapus={rows_before - len(df)})")
    print(f"  Total missing : sebelum={int(missing_before.sum())}, sesudah={int(missing_after.sum())}")
    print("  Missing per kolom (sebelum -> sesudah):")
    for c in missing_before[missing_before > 0].index:
        print(f"      {c:<20}{int(missing_before[c]):>3} -> {int(missing_after[c])}")
    print(f"  Duplikat dihapus: {n_dup}")
    print(f"  Kolom yang berubah: {sorted(changed_cols)}")

    summary = {
        "rows_before": rows_before, "rows_after": len(df),
        "missing_before": int(missing_before.sum()), "missing_after": int(missing_after.sum()),
        "duplicates_removed": n_dup, "changed_columns": sorted(changed_cols),
    }
    return df, summary


# --------------------------------------------------------------------------- #
# TRANSFORM - 3. Transformasi
# --------------------------------------------------------------------------- #
def transform_data(df_clean):
    """Min-Max Scaling (numerik) + Encoding (kategorikal). Kolom asli dipertahankan."""
    banner("4. DATA TRANSFORMATION")
    df = df_clean.copy()
    new_cols = []

    # --- Min-Max Scaling: x' = (x - min) / (max - min) -----------------------
    for col in MINMAX_COLS:
        lo, hi = df[col].min(), df[col].max()
        df[f"{col}_scaled"] = 0.0 if hi == lo else (df[col] - lo) / (hi - lo)
        new_cols.append(f"{col}_scaled")
    print(f"[Min-Max] {len(MINMAX_COLS)} kolom numerik -> *_scaled (rentang 0-1)")

    # --- Ordinal Encoding ----------------------------------------------------
    for col, mapping in ORDINAL_MAPS.items():
        df[f"{col}_encoded"] = df[col].map(mapping)
        if df[f"{col}_encoded"].isna().any():
            raise ValueError(f"Label tak dikenal pada '{col}'")
        new_cols.append(f"{col}_encoded")
    print(f"[Ordinal] {list(ORDINAL_MAPS)} -> *_encoded")

    # --- Label encoding untuk make (kardinalitas tinggi, 20+ merek) ----------
    cats = sorted(df["make"].unique())
    df["make_encoded"] = df["make"].map({m: i for i, m in enumerate(cats)})
    new_cols.append("make_encoded")
    print(f"[Label] 'make' ({len(cats)} merek) -> make_encoded (kode alfabetis)")

    # --- One-Hot Encoding ----------------------------------------------------
    dummies = pd.get_dummies(df[ONEHOT_COLS], prefix=ONEHOT_COLS, dtype=int)
    df = pd.concat([df, dummies], axis=1)
    new_cols += list(dummies.columns)
    print(f"[One-Hot] {ONEHOT_COLS} -> {dummies.shape[1]} kolom biner")

    print(f"\nKolom baru: {len(new_cols)} | Total kolom: {df.shape[1]}")

    # --- Perbandingan sebelum vs sesudah ------------------------------------
    print("\n[Perbandingan sebelum vs sesudah - 5 baris pertama]")
    compare = pd.DataFrame({
        "horsepower": df["horsepower"],
        "horsepower_scaled": df["horsepower_scaled"].round(4),
        "curb-weight": df["curb-weight"],
        "curb-weight_scaled": df["curb-weight_scaled"].round(4),
        "body-style": df["body-style"],
        "body-style_sedan": df["body-style_sedan"],
        "num-of-cylinders": df["num-of-cylinders"],
        "cylinders_enc": df["num-of-cylinders_encoded"],
    }).head()
    print(compare.to_string())
    return df


# --------------------------------------------------------------------------- #
# LOAD
# --------------------------------------------------------------------------- #
def save_data(df, path=PROCESSED_PATH):
    """Menyimpan hasil ke file BARU di data/processed/ (tidak menyentuh data/raw/)."""
    banner("5. SAVE RESULT")
    path = Path(path)
    if RAW_PATH.parent in path.resolve().parents:
        raise ValueError("Output tidak boleh disimpan di folder data/raw/")
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out[DATE_COL] = out[DATE_COL].dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)
    print(f"Tersimpan: {path.relative_to(PROJECT_ROOT)}  "
          f"({out.shape[0]} baris x {out.shape[1]} kolom)")


# --------------------------------------------------------------------------- #
# Orkestrasi pipeline
# --------------------------------------------------------------------------- #
def run_pipeline():
    hash_before = file_hash(RAW_PATH)

    df_raw = load_data()                       # Extract
    inspect_data(df_raw)                       # Transform: inspeksi
    df_clean, _ = clean_data(df_raw)           # Transform: cleaning
    df_final = transform_data(df_clean)        # Transform: transformasi
    save_data(df_final)                        # Load

    banner("VERIFIKASI")
    same = file_hash(RAW_PATH) == hash_before
    print(f"Dataset mentah tidak berubah (SHA-256 identik): {'YA' if same else 'TIDAK'}")
    print(f"Missing values pada processed dataset: {int(df_final.isna().sum().sum())}")
    print("Pipeline selesai.")
    return df_final


if __name__ == "__main__":
    run_pipeline()
