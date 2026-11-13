# macOS reproducibility and release verification

This repository is already published. Do not reinitialise Git or recreate a
published tag merely to run the verification steps.

```bash
git clone https://github.com/pkd-45/space-mission-pipeline-vv-lab.git
cd space-mission-pipeline-vv-lab
bash run_tests_macos.sh
git status --short
gh release view v0.1.0
```

Run the reference demo and compare generated metrics with the checked-in
synthetic reference outputs before describing any numerical result. The
software is not TOLIMAN or GOES GLM mission software; no real mission
measurements or operational performance are claimed.
