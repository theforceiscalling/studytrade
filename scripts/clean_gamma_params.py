with open("web/dashboard/views.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
in_gamma_view = False
for i, line in enumerate(lines):
    if "def gamma_squeeze_view" in line:
        in_gamma_view = True
    elif in_gamma_view and line.startswith("def "):
        in_gamma_view = False

    if "target_delta = float(request.GET.get" in line and not in_gamma_view:
        continue
    if "profit_target_pct = float(request.GET.get" in line and not in_gamma_view:
        continue
    if "stop_loss_pct = float(request.GET.get" in line and not in_gamma_view:
        continue
    if "min_dte = int(request.GET.get" in line and not in_gamma_view:
        continue
    if "max_dte = int(request.GET.get" in line and not in_gamma_view:
        continue

    new_lines.append(line)

with open("web/dashboard/views.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Updated views.py! Total lines:", len(new_lines))
