# Helios v2

Production-grade, modular quantitative trading framework with event-driven architecture, comprehensive tests, and System Sentinel monitoring.

- Event-driven core engine
- Strategies: Trend Following, Mean Reversion
- Risk Manager and Paper Broker
- System Sentinel monitoring with Slack alerts
- Centralized logging, .env-driven config

Quickstart

- Create .env from .env.example and set SLACK_WEBHOOK_URL if desired
- pip install -r requirements.txt
- Run tests: pytest -v
- Demo backtest: python run_backtest.py --symbol AAPL --strategy trend_following
