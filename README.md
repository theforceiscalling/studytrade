# StudyTrade

> Systematic option buying research, backtesting, and paper-trading framework for NSE India (Nifty / Bank Nifty).

**⚠️ This is a research and learning tool. It does not promise profits. Option buying is a negative-expectancy game by default — this framework helps you rigorously test whether a strategy has a real edge after costs.**

## Quick Start

```bash
# 1. Clone and enter project
cd studytrade

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 3. Install in development mode
pip install -e ".[dev]"

# 4. Copy and edit local config (optional)
copy config\sample_local.yaml config\local.yaml

# 5. Run tests
pytest

# 6. Start exploring
jupyter lab notebooks/
```

## Project Structure

```
src/studytrade/
├── config.py          # YAML config loader with pydantic validation
├── logger.py          # Structured logging (console + file)
├── constants.py       # NSE market constants, lot sizes, fees
├── data/              # Module 2: Data fetching, cleaning, storage
├── math/              # Module 3: BSM, Greeks, IV solver
├── features/          # Module 4: Technical, volatility, regime features
├── strategy/          # Module 5: Strategy engine (plug-in design)
├── ml/                # Module 6: ML models with walk-forward CV
├── backtest/          # Module 7: Backtesting engine with full cost model
├── analytics/         # Module 8: Performance metrics, Monte Carlo, overfitting checks
├── risk/              # Module 9: Position sizing, loss limits, kill switch
└── paper/             # Module 10: Paper trading with Streamlit dashboard
```

## Configuration

All settings live in `config/default.yaml`. Override locally in `config/local.yaml` (gitignored).

Key settings:
- **Capital**: Starting capital and currency
- **Risk**: Per-trade risk, daily/weekly limits, kill switch
- **Costs**: Full NSE India fee model (STT, exchange, GST, stamp duty, slippage)
- **Data**: Source selection (synthetic/CSV/broker API)
- **Strategy**: Signal thresholds, stop-loss, targets, time stops

## Testing

```bash
pytest                        # Run all tests
pytest -v                     # Verbose
pytest -m "not slow"          # Skip slow tests
pytest --cov=studytrade       # With coverage
```

## License

MIT
