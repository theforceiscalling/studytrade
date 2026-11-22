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
print("Context signals:", resp.context_data.get("signals") if hasattr(resp, "context_data") else "No context_data on HttpResponse")
