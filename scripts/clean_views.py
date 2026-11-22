import re

with open("web/dashboard/views.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    # Keep the first _enrich_runs at line ~450, skip subsequent definitions
    if i > 500 and ("def _enrich_runs" in line):
        skip = True
    elif skip and (line.startswith("def ") or line.startswith("@csrf_exempt") or line.startswith("# ---")):
        if "api_backtest_history" not in line and "api_get_backtest_run" not in line and "_enrich_runs" not in line:
            skip = False
    
    if not skip:
        new_lines.append(line)

with open("web/dashboard/views.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Cleaned views.py! Line count:", len(new_lines))
