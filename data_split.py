# data_split.py
import pandas as pd
from sklearn.model_selection import train_test_split
from rich.console import Console
from rich import box
from rich.table import Table

console = Console()

# -----------------------------
# Log helpers (short & clear)
# -----------------------------
def info(msg):
    console.print(f"[cyan][INFO][/cyan] {msg}")

def good(msg):
    console.print(f"[bold green]✔ {msg}[/bold green]")

def warn(msg):
    console.print(f"[yellow][WARN][/yellow] {msg}")

def error(msg):
    console.print(f"[bold red]✘ {msg}[/bold red]")

def title(msg):
    console.print(f"\n[bold underline cyan]{msg}[/bold underline cyan]")

def summary(train, val, test):
    table = Table(title="SUMMARY", box=box.SIMPLE)
    table.add_column("Set", style="cyan")
    table.add_column("Count", justify="right", style="white")
    table.add_row("Train", str(train))
    table.add_row("Val", str(val))
    table.add_row("Test", str(test))
    table.add_row("Total", str(train + val + test))
    console.print(table)


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def perform_split(
    df_train_new,
    df_val_old,
    df_test_old,
    target_val=870,
    target_test=870
):

    title("STEP: Data Split")

    # --------------------------------------
    # 1) เอารูปที่ใช้ใน val/test ออกจาก train
    # --------------------------------------
    used = set(df_val_old['filepath']).union(df_test_old['filepath'])
    df_remaining = df_train_new[~df_train_new['filepath'].isin(used)]

    info(f"Remaining train images: {len(df_remaining)}")

    # --------------------------------------
    # 2) จำนวนที่ต้องเพิ่ม val/test
    # --------------------------------------
    need_val = target_val - len(df_val_old)
    need_test = target_test - len(df_test_old)
    need_more = need_val + need_test

    info(f"Need more = {need_more} (val {need_val}, test {need_test})")

    if need_more > len(df_remaining):
        error("Not enough images in df_remaining!")
        raise ValueError("Need more images than available in remaining train.")

    # --------------------------------------
    # 3) สุ่ม subset ที่จะไปเติม val/test (df_extra)
    # --------------------------------------
    df_extra, df_train_final = train_test_split(
        df_remaining,
        test_size=(len(df_remaining) - need_more) / len(df_remaining),
        stratify=df_remaining["label"],
        random_state=42
    )

    # --------------------------------------
    # 4) แบ่ง df_extra ให้ val/test
    # --------------------------------------
    df_val_add = df_extra.sample(n=need_val, random_state=42)
    df_test_add = df_extra.drop(df_val_add.index).sample(n=need_test, random_state=42)

    # รวมชุดใหม่
    df_val_new = pd.concat([df_val_old, df_val_add], ignore_index=True)
    df_test_new = pd.concat([df_test_old, df_test_add], ignore_index=True)

    # --------------------------------------
    # 5) Summary
    # --------------------------------------
    summary(len(df_train_final), len(df_val_new), len(df_test_new))

    # --------------------------------------
    # 6) Safety check
    # --------------------------------------
    title("CHECK OVERLAP")

    assert len(set(df_val_new['filepath']).intersection(df_test_new['filepath'])) == 0, \
        error("Overlap: val <-> test")

    assert len(set(df_val_new['filepath']).intersection(df_train_final['filepath'])) == 0, \
        error("Overlap: val <-> train")

    assert len(set(df_test_new['filepath']).intersection(df_train_final['filepath'])) == 0, \
        error("Overlap: test <-> train")

    good("No overlaps detected.")

    return df_train_final, df_val_new, df_test_new