# Publish on GitHub from macOS

```bash
cd "$HOME/Documents/GitHub/space-mission-pipeline-vv-lab"
rm -rf .venv

git init
git branch -M main
git add .
git diff --cached --check
git commit -m "Initial public release of space-mission-pipeline-vv-lab v0.1.0"

gh repo create pkd-45/space-mission-pipeline-vv-lab \
  --public \
  --description "Synthetic V&V benchmark for precision astrometry and rare-event satellite pipelines." \
  --source=. \
  --remote=origin \
  --push

gh run list --limit 3
RUN_ID=$(gh run list --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" --exit-status

git tag -a v0.1.0 -m "SETI Pipeline Scientist application methods release"
git push origin v0.1.0

gh release create v0.1.0 \
  --title "space-mission-pipeline-vv-lab v0.1.0" \
  --notes "Controlled synthetic verification-and-validation benchmark combining differential-astrometry injection-recovery with grouped rare-event satellite light-curve classification. Includes explicit nuisance variables, calibration and class-balanced metrics, leakage-resistant splits, release gates, product checksums, seven automated tests, and CI on Python 3.11-3.13. This is not TOLIMAN or GOES GLM mission software and the results are not mission-performance claims."
```
