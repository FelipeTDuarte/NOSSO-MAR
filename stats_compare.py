import os
import sys
from pathlib import Path

def dir_stats(p: Path):
    total_files = 0
    total_bytes = 0
    for root, dirs, files in os.walk(p):
        for f in files:
            try:
                fp = Path(root) / f
                total_files += 1
                total_bytes += fp.stat().st_size
            except Exception:
                pass
    return total_files, total_bytes

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python stats_compare.py <dir1> <dir2>')
        sys.exit(2)
    a = Path(sys.argv[1]).resolve()
    b = Path(sys.argv[2]).resolve()
    fa, ba = dir_stats(a)
    fb, bb = dir_stats(b)
    print(f"DIR_A:{a}")
    print(f"FILES_A:{fa}")
    print(f"BYTES_A:{ba}")
    print(f"DIR_B:{b}")
    print(f"FILES_B:{fb}")
    print(f"BYTES_B:{bb}")
    if fa>fb:
        print('BEST: A')
    elif fb>fa:
        print('BEST: B')
    else:
        if ba>bb:
            print('BEST: A')
        elif bb>ba:
            print('BEST: B')
        else:
            print('BEST: TIE')
