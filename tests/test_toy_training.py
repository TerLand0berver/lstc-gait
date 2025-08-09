import subprocess, sys
from pathlib import Path


def test_toy_training_numeric_baseline(tmp_path: Path):
    data = tmp_path / "toy_data"
    # generate a tiny dataset (faster than default)
    subprocess.run([sys.executable, "examples/gen_toy_dataset.py", "--out", str(data), "--subjects", "4", "--seq-per-subject", "2", "--frames", "8"], check=True)
    # run a very short training (CPU)
    result = subprocess.run([sys.executable, "examples/train_real.py", "--data-root", str(data), "--epochs", "1", "--batch-size", "8", "--seq-len", "8", "--device", "cpu"], check=True, capture_output=True, text=True)
    # parse last printed line containing "Best val acc"
    out = result.stdout.strip().splitlines()
    last = "\n".join(out[-5:])
    # Expect accuracy to beat random baseline (1/num_classes). Our toy has 4 subjects.
    # Use a conservative threshold > 0.35 to allow fluctuation while verifying learning signal.
    # Extract any acc floats present
    import re
    nums = [float(x) for x in re.findall(r"acc[:=]\s*([0-9.]+)", last)]
    assert nums, f"No accuracy found in output tail: {last}"
    best = max(nums)
    assert best > 0.35, f"Expected acc > 0.35, got {best}. Output tail:\n{last}"
