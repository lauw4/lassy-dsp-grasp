import os
from pathlib import Path

def cleanup_old_results(folder, keep_last=5):
    methods = ["GRASP", "VND", "MNS", "TABU"]
    path = Path(folder)

    for method in methods:
        pattern = f"scores_{method}_*.csv"
        files = sorted(
            path.glob(pattern),
            key=os.path.getmtime,
            reverse=True
        )
        for f in files[keep_last:]:
            try:
                f.unlink()
                print(f"Deleted: {f.name}")
            except Exception as e:
                print(f"Error deleting {f.name}: {e}")
