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
# 1. Test Gamma Squeeze View
req_gs = rf.get("/gamma-squeeze/?symbol=NIFTY")
resp_gs = gamma_squeeze_view(req_gs)
print("Gamma squeeze status code:", resp_gs.status_code)
content = resp_gs.content.decode("utf-8")
if "26100" in content:
    print("Found 26100 in gamma squeeze page!")
if "14.80" in content or "14.8" in content:
    print("Found 14.80 in gamma squeeze page!")
else:
    print("Check premium in content:")
    import re
    matches = re.findall(r"₹\s*([0-9\.]+)", content)
    print("Found rupee values:", matches[:10])

# 2. Test Paper View
req_p = rf.get("/paper/")
resp_p = paper_view(req_p)
print("Paper view status code:", resp_p.status_code)
content_p = resp_p.content.decode("utf-8")
if "NIFTY26NOV26100CE" in content_p:
    print("Found NIFTY26NOV26100CE in paper trading view!")
if "14.80" in content_p:
    print("Found 14.80 in paper trading view!")
