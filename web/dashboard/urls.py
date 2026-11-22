from django.urls import path

from . import views

app_name = "dashboard"

urlpatterns = [
    path("", views.overview_view, name="overview"),
    path("data-lab/", views.data_lab_view, name="data_lab"),
    path("features/", views.features_view, name="features"),
    path("gamma-squeeze/", views.gamma_squeeze_view, name="gamma_squeeze"),
    path("backtest/", views.backtest_view, name="backtest"),
    path("analytics/", views.analytics_view, name="analytics"),
    path("risk/", views.risk_view, name="risk"),
    path("paper/", views.paper_view, name="paper"),

    # AJAX endpoints
    path("api/generate-data/", views.api_generate_data, name="api_generate_data"),
    path("api/option-chain/", views.api_option_chain, name="api_option_chain"),
    path("api/gamma-profile/", views.api_gamma_profile, name="api_gamma_profile"),
    path("api/run-gamma-strategy/", views.api_run_gamma_strategy, name="api_run_gamma_strategy"),
    path("api/run-backtest/", views.api_run_backtest, name="api_run_backtest"),
    path("api/backtest-history/", views.api_backtest_history, name="api_backtest_history"),
    path("api/backtest-run/<int:run_id>/", views.api_get_backtest_run, name="api_get_backtest_run"),
    path("api/run-monte-carlo/", views.api_run_monte_carlo, name="api_run_monte_carlo"),
    path("api/calc-sizing/", views.api_calc_sizing, name="api_calc_sizing"),
    path("api/live-status/", views.api_live_scanner_status, name="api_live_status"),
    path("api/toggle-live-scanner/", views.api_toggle_live_scanner, name="api_toggle_live_scanner"),
    path("api/paper/buy-signal/", views.api_paper_buy_signal, name="api_paper_buy_signal"),
    path("api/paper/close-position/", views.api_paper_close_position, name="api_paper_close_position"),
    path("api/paper/auto-trader/status/", views.api_auto_trader_status, name="api_auto_trader_status"),
    path("api/paper/auto-trader/configure/", views.api_auto_trader_configure, name="api_auto_trader_configure"),
    path("api/paper/auto-trader/toggle/", views.api_auto_trader_toggle, name="api_auto_trader_toggle"),
    path("api/paper/auto-trader/run-cycle/", views.api_auto_trader_run_cycle, name="api_auto_trader_run_cycle"),
    path("api/paper/sync-nse-delayed/", views.api_sync_nse_delayed_prices, name="api_sync_nse_delayed"),
    path("api/gamma-squeeze/ranks/", views.api_gamma_ranks, name="api_gamma_ranks"),
    path("api/gamma-squeeze/calibrate/", views.api_gamma_calibrate, name="api_gamma_calibrate"),
    path("api/sync-bhavcopy/", views.api_sync_bhavcopy, name="api_sync_bhavcopy"),
]
