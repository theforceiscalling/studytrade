import os
import sys
import django

sys.path.insert(0, os.path.abspath("web"))
sys.path.insert(0, os.path.abspath("src"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "studytrade_web.settings")
django.setup()

from django.test import RequestFactory
from dashboard.views import gamma_squeeze_view

rf = RequestFactory()
req_gs = rf.get("/gamma-squeeze/?symbol=NIFTY")
resp = gamma_squeeze_view(req_gs)
# print recommendation from context or HTML
content = resp.content.decode("utf-8")
import re
print("Matches for strike / premium:")
for m in re.finditer(r"(Recommended Manual Strike|Target Strike|LTP / Entry Premium|26100).*?", content):
    start = max(0, m.start() - 50)
    end = min(len(content), m.end() + 200)
    print("--- SNIPPET ---")
    print(content[start:end])
