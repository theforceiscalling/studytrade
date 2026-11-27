import os
import sys
from pathlib import Path

from django.apps import AppConfig


class DashboardConfig(AppConfig):
    name = 'dashboard'

    def ready(self) -> None:
        """Start the daily data refresher inside the actual server worker process."""
        if os.environ.get("STUDYTRADE_AUTO_SYNC", "1") == "0":
            return
        argv = sys.argv
        is_runserver = len(argv) > 1 and argv[1] == "runserver"
        is_gunicorn = any("gunicorn" in str(arg).lower() for arg in argv) or "gunicorn" in sys.modules

        # When under runserver, only start in the reloaded main process
        if is_runserver and not (os.environ.get("RUN_MAIN") == "true" or "--noreload" in argv):
            return

        if not (is_runserver or is_gunicorn):
            return

        from studytrade.data.auto_sync import start_auto_sync

        data_dir = Path(__file__).resolve().parent.parent.parent / "data"
        start_auto_sync(data_dir=data_dir)
