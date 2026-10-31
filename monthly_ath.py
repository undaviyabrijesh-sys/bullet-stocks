name: ATH Monthly Scan

on:
  schedule:
    - cron: "30 5 * * 1-5"   # 11:00 AM IST
    - cron: "20 9 * * 1-5"   # 2:50 PM IST (GitHub delays runs, so this lands near 3 PM)
  workflow_dispatch:

permissions:
  contents: write

concurrency:
  group: ath-scan
  cancel-in-progress: false

jobs:
  scan:
    runs-on: ubuntu-latest
    timeout-minutes: 45
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install pandas requests yfinance
      - run: python monthly_ath.py
        env:
          DISCORD_WEBHOOK_URL: ${{ secrets.DISCORD_WEBHOOK_URL }}
      - name: Save state
        run: |
          git config user.name "github-actions"
          git config user.email "actions@github.com"
          git add state.json
          git diff --cached --quiet || (git commit -m "update scan state" && git push)
