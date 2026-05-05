import filecmp
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python compare_remote.py <remote_path>")
    sys.exit(2)

local = os.path.abspath('.')
remote = os.path.abspath(sys.argv[1])

print(f"Comparing local: {local}\n        remote: {remote}\n")

dc = filecmp.dircmp(local, remote)

def report(dc, prefix=""):
    if dc.left_only:
        print(prefix + "Only in local:", dc.left_only)
    if dc.right_only:
        print(prefix + "Only in remote:", dc.right_only)
    if dc.diff_files:
        print(prefix + "Different files:", dc.diff_files)
    for name, sub in dc.subdirs.items():
        report(sub, prefix + name + '/ ')

report(dc)

# Exit code 0 means script ran; differences are printed above
sys.exit(0)
