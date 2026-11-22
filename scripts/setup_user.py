"""
StudyTrade — Create or Update Operator User
"""
import os
import sys

# Ensure web/ is on path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
web_path = os.path.join(BASE_DIR, "web")
if web_path not in sys.path:
    sys.path.insert(0, web_path)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "studytrade_web.settings")

import django
django.setup()

from django.contrib.auth import get_user_model

User = get_user_model()

EMAIL = "tusharpgtbiology@gmail.com"
USERNAME = "tushar"
PASSWORD = "StudyTrade@2026"

user = User.objects.filter(email__iexact=EMAIL).first()
if not user:
    user = User.objects.filter(username__iexact=USERNAME).first()

if not user:
    user = User.objects.create_superuser(
        username=USERNAME,
        email=EMAIL,
        password=PASSWORD,
    )
    print(f"Created new superuser: {USERNAME} ({EMAIL})")
else:
    user.username = USERNAME
    user.email = EMAIL
    user.set_password(PASSWORD)
    user.is_superuser = True
    user.is_staff = True
    user.save()
    print(f"Updated existing user: {USERNAME} ({EMAIL})")

print(f"Login Credentials:")
print(f"  Username/Email: {EMAIL} (or '{USERNAME}')")
print(f"  Password:       {PASSWORD}")
