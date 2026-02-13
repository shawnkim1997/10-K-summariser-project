#!/bin/bash
# Push current (English) version to GitHub
# Run from project root: bash push_to_github.sh

set -e
cd "/Users/seonpil/Documents/FQDC Project"

echo "--- Git status ---"
git status

echo ""
echo "--- Add all changes ---"
git add -A

echo ""
echo "--- Commit (README: 3-tab architecture + Project Origin & Vision) ---"
git commit -m "docs: Update README — 3-tab system (10-K Insights, DCF, Comps) + Project Origin & Vision (English)"

echo ""
echo "--- Push to origin main ---"
git push origin main

echo ""
echo "Done. Check https://github.com/shawnkim1997/10-K-summariser-project"
