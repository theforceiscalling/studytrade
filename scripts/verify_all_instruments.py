import os
import sys
import django

sys.path.insert(0, os.path.abspath("web"))
sys.path.insert(0, os.path.abspath("src"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "studytrade_web.settings")
django.setup()

from django.test import RequestFactory
from dashboard.views import gamma_squeeze_view, paper_view

rf = RequestFactory()
for sym in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
    req = rf.get(f"/gamma-squeeze/?symbol={sym}")
    resp = gamma_squeeze_view(req)
    print(f"[{sym}] Gamma Squeeze status: {resp.status_code}")
    content = resp.content.decode("utf-8")
    if "Manual Trade Recommendation" in content:
        print(f"[{sym}] Found Manual Trade Recommendation section!")

req_p = rf.get("/paper/")
resp_p = paper_view(req_p)
print(f"[PAPER] Status code: {resp_p.status_code}")
