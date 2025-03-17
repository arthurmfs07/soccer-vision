# helper functions

from pathlib import Path

def load_abs_path():
    home = Path(__file__).resolve().parent.parent
    return home

