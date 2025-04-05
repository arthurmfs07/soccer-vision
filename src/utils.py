# helper functions

from pathlib import Path

def load_abs_path():
    home = Path(__file__).resolve().parent.parent
    return home

def get_data_path() -> Path:
    return load_abs_path() / "data"

def get_csv_path() -> Path:
    return get_data_path() / "00--raw" / "df_on_360.csv"

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