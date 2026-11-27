import os
import sys
from pathlib import Path

from django.apps import AppConfig


class DashboardConfig(AppConfig):
    name = 'dashboard'

    def ready(self) -> None:
        """Start the daily data refresher inside the actual runserver worker process."""
        if os.environ.get("STUDYTRADE_AUTO_SYNC", "1") == "0":
            return
        argv = sys.argv
        is_server = len(argv) > 1 and argv[1] == "runserver"
        if not is_server or not (os.environ.get("RUN_MAIN") == "true" or "--noreload" in argv):
            return

        from studytrade.data.auto_sync import start_auto_sync

        data_dir = Path(__file__).resolve().parent.parent.parent / "data"
        start_auto_sync(data_dir=data_dir)
