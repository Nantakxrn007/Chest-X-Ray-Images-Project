# clean.py
import os
import imagehash
from PIL import Image
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

# ==========================================================
# LOG HELPERS
# ==========================================================
def log_title(text, style="bold cyan"):
    console.print(Panel(text, style=style, expand=True))

def log_info(text):
    console.print(f"[green][INFO][/green] {text}")

def log_skip(text):
    console.print(f"[yellow][SKIP][/yellow] {text}")

def log_run(text):
    console.print(f"[cyan][RUN][/cyan] {text}")

def log_done(text):
    console.print(f"[bold green][DONE][/bold green] {text}")

def log_table(before, after, removed, title):
    table = Table(title=title, box=box.SIMPLE_HEAVY)
    table.add_column("Metric", justify="left", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Before", str(before))
    table.add_row("After", str(after))
    table.add_row("Removed", str(removed))

    console.print(table)


# ==========================================================
# STEP 0 — LOAD DATAFRAME
# ==========================================================
def make_dataframe(base_dir):
    filepaths, labels = [], []

    for split in ["train", "val", "test"]:
        for label_dir in ["NORMAL", "PNEUMONIA"]:
            folder = os.path.join(base_dir, split, label_dir)
            for filename in os.listdir(folder):
                filepaths.append(os.path.join(folder, filename))
                labels.append(0 if label_dir == "NORMAL" else 1)

    df = pd.DataFrame({"filepath": filepaths, "label": labels})

    log_title("STEP 0 — Load DataFrame")
    log_info(f"Total files loaded: {len(df)}")

    return df


# ==========================================================
# STEP 1 — PHASH (DEDUP)
# ==========================================================
def phash_dedup_no_copy(df, cache_csv, split_name):

    if os.path.exists(cache_csv):
        log_title(f"{split_name.upper()} — SKIPPED (Cache Found)", style="bold yellow")
        df_clean = pd.read_csv(cache_csv)
        log_info(f"Using cached PHASH: {cache_csv}")
        log_info(f"Images: {len(df_clean)}")
        return df_clean

    log_title(f"{split_name.upper()} — RUNNING pHash Dedup", style="bold magenta")
    log_info(f"Cache path: {cache_csv}")

    seen = set()
    keep_rows = []
    before = len(df)

    console.print("[cyan]Computing pHash… (low RAM mode)[/cyan]")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            img = Image.open(row["filepath"])
            h = str(imagehash.phash(img))
        except:
            continue

        if h not in seen:
            seen.add(h)
            keep_rows.append(row)

    df_clean = pd.DataFrame(keep_rows)
    after = len(df_clean)
    removed = before - after

    log_table(before, after, removed, f"{split_name.upper()} — PHASH Summary")

    os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
    df_clean.to_csv(cache_csv, index=False)
    log_done(f"Saved: {cache_csv}")

    return df_clean


# ==========================================================
# STEP 2 — RGB CONVERSION (IN-PLACE)
# ==========================================================
def convert_to_rgb(df, cache_csv, split_name):

    # If RGB cache exists → skip
    if os.path.exists(cache_csv):
        log_title(f"{split_name.upper()} — SKIPPED (RGB Cache Found)", style="bold yellow")
        df_rgb = pd.read_csv(cache_csv)
        log_info(f"Using RGB cache: {cache_csv}")
        log_info(f"Images (RGB): {len(df_rgb)}")
        return df_rgb

    log_title(f"{split_name.upper()} — RUNNING RGB Conversion", style="bold cyan")
    log_info("Converting images to RGB (in-place)")
    log_info(f"Cache will be saved to: {cache_csv}")

    count = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = row["filepath"]

        try:
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
                img.save(path)
                count += 1
        except:
            continue

    log_info(f"Converted to RGB: {count} images")
    log_done("RGB conversion complete.")

    df.to_csv(cache_csv, index=False)
    log_done(f"Saved RGB cache → {cache_csv}")

    return df

# ==========================================================
# STEP 3 — CLAHE (IN-PLACE)
# ==========================================================
def apply_clahe(df, cache_csv, split_name):

    # ----- ถ้ามี cache แล้ว → SKIP -----
    if os.path.exists(cache_csv):
        log_title(f"{split_name.upper()} — SKIPPED (CLAHE Cache Found)", style="bold yellow")
        df_clahe = pd.read_csv(cache_csv)
        log_info(f"Using CLAHE cache: {cache_csv}")
        log_info(f"Images (CLAHE): {len(df_clahe)}")
        return df_clahe

    # ----- RUN CLAHE -----
    log_title(f"{split_name.upper()} — RUNNING CLAHE", style="bold blue")
    log_info(f"Applying CLAHE (in-place) to {len(df)} images")
    log_info(f"Cache will be saved to: {cache_csv}")

    # Prepare CLAHE algorithm
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="CLAHE"):
        path = row["filepath"]

        try:
            # read image as GRAY
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            # if grayscale
            if img is None:
                continue

            # apply CLAHE
            cl = clahe.apply(img)

            # save back (overwrite)
            cv2.imwrite(path, cl)

            count += 1

        except:
            continue

    log_info(f"CLAHE applied: {count} images")
    log_done("CLAHE conversion complete.")

    # save cache
    os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
    df.to_csv(cache_csv, index=False)
    log_done(f"Saved CLAHE cache → {cache_csv}")

    return df

# ==========================================================
# MAIN PIPELINE
# ==========================================================
def run_cleaning(base_dir="Data/chest_xray"):

    log_title("🚀 START CLEANING PIPELINE")

    df_all = make_dataframe(base_dir)

    # Split
    df_train = df_all[df_all["filepath"].str.contains("train")]
    df_val   = df_all[df_all["filepath"].str.contains("val")]
    df_test  = df_all[df_all["filepath"].str.contains("test")]

    cache_dir = "Data/cache"
    os.makedirs(cache_dir, exist_ok=True)

    # STEP 1: pHash Dedup
    df_train_ph = phash_dedup_no_copy(df_train, f"{cache_dir}/clean_phash_train.csv", "train")
    df_val_ph   = phash_dedup_no_copy(df_val,   f"{cache_dir}/clean_phash_val.csv",   "val")
    df_test_ph  = phash_dedup_no_copy(df_test,  f"{cache_dir}/clean_phash_test.csv",  "test")

    # STEP 2: RGB Convert
    df_train_rgb = convert_to_rgb(df_train_ph, f"{cache_dir}/clean_rgb_train.csv", "train")
    df_val_rgb   = convert_to_rgb(df_val_ph,   f"{cache_dir}/clean_rgb_val.csv",   "val")
    df_test_rgb  = convert_to_rgb(df_test_ph,  f"{cache_dir}/clean_rgb_test.csv",  "test")

    # STEP 3: CLAHE
    df_train_clahe = apply_clahe(df_train_rgb, f"{cache_dir}/clean_clahe_train.csv", "train")
    df_val_clahe   = apply_clahe(df_val_rgb,   f"{cache_dir}/clean_clahe_val.csv",   "val")
    df_test_clahe  = apply_clahe(df_test_rgb,  f"{cache_dir}/clean_clahe_test.csv",  "test")

    # Final
    log_title("🎉 CLEANING COMPLETE")
    log_info(f"Train images: {len(df_train_clahe)}")
    log_info(f"Val images:   {len(df_val_clahe)}")
    log_info(f"Test images:  {len(df_test_clahe)}")

    return df_train_clahe, df_val_clahe, df_test_clahe

