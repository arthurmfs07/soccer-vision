# helper functions

from pathlib import Path

def load_abs_path():
    home = Path(__file__).resolve().parent.parent
    return home

def get_data_path() -> Path:
    return load_abs_path() / "data"

def get_csv_path() -> Path:
    return get_data_path() / "00--raw" / "df_padronizado.csv"

def get_actual_yolo() -> Path:
    return get_data_path() / "10--models" / "yolov8_finetuned.pt"


def parse_timestamp(ts: str) -> float:
    """
    HH:MM:SS.sss
    ->
    SS.sss
    """
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        h = int(h)
        m = int(m)
    elif len(parts) == 2:
        h = 0
        m, s = parts
        m = int(m)
    else:
        raise ValueError(f"Invalid timestamp format: {ts}. Should be: HH:MM:SS.sss")
    return h * 3600 + m * 60 + float(s)



rf2my = {
     0:  0,   1:  6,   2: 14,   3: 15,
     4:  7,   5:  2,   6: 16,   7: 17,
     8:  4,   9:  8,  10: 26,  11: 27,
    12:  9,  13: 22,  14: 24,  15: 25,
    16: 23,  17: 12,  18: 28,  19: 29,
    20: 13,  21:  5,  22: 20,  23: 21,
    24:  1,  25: 10,  26: 18,  27: 19,
    28: 11,  29:  3,  30: 30,  31: 31
} # conferido