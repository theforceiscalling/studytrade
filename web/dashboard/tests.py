import json

from django.test import Client, TestCase
from django.urls import reverse


class DashboardViewsTestCase(TestCase):
    def setUp(self):
        self.client = Client()

    def test_overview_view(self):
        res = self.client.get(reverse("dashboard:overview"))
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "Systematic Option Buying Framework")
        self.assertContains(res, "NIFTY")

    def test_data_lab_view(self):
        res = self.client.get(reverse("dashboard:data_lab"))
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "Option Chain Ladder")
        self.assertContains(res, "Max Pain")

    def test_features_view(self):
        res = self.client.get(reverse("dashboard:features"))
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "Feature Engineering")
        self.assertContains(res, "GARCH")

    def test_backtest_view(self):
        res = self.client.get(reverse("dashboard:backtest"))
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "Strategy Configuration")
        self.assertContains(res, "Donchian Breakout")

    def test_analytics_view(self):
        res = self.client.get(reverse("dashboard:analytics"))
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "Monte Carlo")
        self.assertContains(res, "Deflated Sharpe Ratio")

    def test_risk_view(self):
        res = self.client.get(reverse("dashboard:risk"))
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "Position Sizing")
        self.assertContains(res, "Half-Kelly")

    def test_paper_view(self):
        res = self.client.get(reverse("dashboard:paper"))
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "Greeks Cockpit")
        self.assertContains(res, "90-Day")

    def test_api_calc_sizing(self):
        payload = {
            "capital": 200000.0,
            "risk_pct": 1.5,
            "symbol": "NIFTY",
            "premium": 150.0,
            "atr": 180.0,
            "base_atr": 150.0,
            "win_rate": 0.38,
            "payoff": 2.6,
        }
        res = self.client.post(
            reverse("dashboard:api_calc_sizing"),
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertIn("fixed_fractional", data)
        self.assertIn("volatility_adjusted", data)
        self.assertIn("half_kelly", data)

    def test_api_run_monte_carlo(self):
        payload = {
            "pnls": [1500, -800, 2400, -1100, 3200, -900, 1800, -750, 4200],
            "initial_capital": 200000.0,
            "n_simulations": 100,
            "num_trials": 10,
        }
        res = self.client.post(
            reverse("dashboard:api_run_monte_carlo"),
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertIn("p50_drawdown_pct", data)
        self.assertIn("dsr_probability", data)

    def test_api_run_backtest(self):
        payload = {
            "strategy": "breakout",
            "symbol": "NIFTY",
            "initial_capital": 200000.0,
            "risk_per_trade_pct": 1.5,
            "target_delta": 0.35,
            "slippage_pts": 0.5,
            "stop_loss_pct": 30.0,
            "profit_target_pct": 60.0,
            "min_dte": 2,
            "max_dte": 20,
        }
        res = self.client.post(
            reverse("dashboard:api_run_backtest"),
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertIn("metrics", data)
        self.assertIn("equity_curve", data)

    def test_gamma_squeeze_view(self):
        res = self.client.get(reverse("dashboard:gamma_squeeze"))
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "Gamma Squeeze")
        self.assertContains(res, "Call Wall")
        self.assertContains(res, "Put Wall")
        self.assertContains(res, "Gamma Flip")

    def test_api_gamma_profile(self):
        res = self.client.get(reverse("dashboard:gamma_squeeze"))
        self.assertEqual(res.status_code, 200)
        # Call api_gamma_profile with default symbol
        res = self.client.get(reverse("dashboard:api_gamma_profile"), {"symbol": "NIFTY"})
        # Even if date not provided or data empty, should return 200 or 404 cleanly
        self.assertIn(res.status_code, [200, 404])

    def test_api_run_backtest_with_gamma_squeeze(self):
        payload = {
            "strategy": "gamma_squeeze",
            "symbol": "NIFTY",
            "initial_capital": 200000.0,
            "risk_per_trade_pct": 1.5,
            "target_delta": 0.40,
            "slippage_pts": 0.5,
            "stop_loss_pct": 30.0,
            "profit_target_pct": 60.0,
            "min_dte": 0,
            "max_dte": 4,
        }
        res = self.client.post(
            reverse("dashboard:api_run_backtest"),
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertIn("metrics", data)
