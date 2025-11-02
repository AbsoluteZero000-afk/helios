# Helios v3 - Backend Evolution

Production-grade quantitative trading framework with event-driven architecture, comprehensive self-healing capabilities, and full containerization.

## 🚀 Key Features

- **Event-Driven Architecture**: Modular, scalable core engine with async processing
- **Self-Healing System Sentinel**: Comprehensive monitoring and auto-repair capabilities
- **Full Containerization**: Docker and Kubernetes ready with multi-stage builds
- **Production Database**: PostgreSQL with SQLAlchemy ORM and automated migrations
- **Redis Integration**: Caching, pub/sub, and task queue management
- **Celery Task Queue**: Asynchronous processing for trades, analytics, and notifications
- **Alpaca Integration**: Full paper and live trading support
- **Advanced Risk Management**: Portfolio-level constraints and position sizing
- **Comprehensive Logging**: Structured logging with Loguru and optional JSON format
- **Slack Integration**: Real-time alerts with emoji and color-coded notifications
- **Extensive Testing**: Unit, integration, and end-to-end test coverage

## 📁 Architecture Overview

```
helios/
├── Dockerfile                    # Multi-stage production build
├── docker-compose.yml           # Full stack development environment
├── pyproject.toml               # Modern Python packaging
├── requirements.txt             # Pinned dependencies
├── .env.example                 # Comprehensive configuration template
├── config/
│   ├── settings.py             # Pydantic-based settings with validation
│   └── strategy.yaml           # Strategy activation and parameters
├── core/
│   ├── engine.py               # Event-driven trading engine
│   ├── events.py               # Event system and bus
│   ├── portfolio.py            # Portfolio management
│   └── performance.py          # Performance analytics
├── data/
│   ├── feeder.py              # Market data feeding
│   ├── processor.py           # Data cleaning and transformation
│   └── storage.py             # Data persistence layer
├── execution/
│   ├── broker_base.py         # Abstract broker interface
│   ├── paper_broker.py        # Paper trading simulation
│   └── alpaca_broker.py       # Alpaca Markets integration
├── risk/
│   └── manager.py             # Risk management and position sizing
├── strategies/
│   ├── base.py                # Abstract strategy framework
│   ├── trend_following.py     # SMA crossover strategy
│   ├── mean_reversion.py      # Bollinger Band strategy
│   └── momentum_breakout.py   # Breakout momentum strategy
├── utils/
│   ├── logger.py              # Centralized Loguru logging
│   ├── slack.py               # Enhanced Slack notifications
│   ├── sentinel.py            # System Sentinel (self-healing)
│   ├── redis_queue.py         # Redis operations and caching
│   ├── db.py                  # Database management and ORM
│   └── health.py              # Comprehensive health checks
├── workers/
│   └── tasks.py               # Celery task definitions
├── k8s/                       # Kubernetes manifests
│   ├── deployment.yaml        # Application deployments
│   ├── service.yaml           # Kubernetes services
│   └── configmap.yaml         # Configuration and PVC
├── tests/                     # Comprehensive test suite
└── .github/workflows/         # CI/CD pipeline
```

## 🛠️ Local Development Setup

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- PostgreSQL (optional for local dev)
- Redis (optional for local dev)

### Quick Start

1. **Clone and Setup**
   ```bash
   git clone https://github.com/AbsoluteZero000-afk/helios.git
   cd helios
   
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install dependencies
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **macOS TA-Lib Installation**
   ```bash
   # For Apple Silicon Macs
   brew install ta-lib
   pip install TA-Lib
   ```

3. **Configuration**
   ```bash
   # Copy and customize environment variables
   cp .env.example .env
   
   # Edit .env with your settings:
   # - SLACK_WEBHOOK_URL (optional)
   # - ALPACA_API_KEY and ALPACA_SECRET_KEY (for live trading)
   # - LOG_LEVEL, DATABASE_URL, etc.
   ```

4. **Run Tests**
   ```bash
   # Run full test suite
   pytest -v
   
   # Run with coverage
   pytest -v --cov=. --cov-report=html
   ```

5. **Demo Backtest**
   ```bash
   # Run simple backtest
   python run_backtest.py --symbol AAPL --strategy trend_following
   
   # System health check
   python utils/sentinel.py --status
   
   # Full integrity audit
   python utils/sentinel.py --audit
   ```

## 🐳 Docker Deployment

### Development with Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Run tests in container
docker-compose exec helios_app pytest -v

# View logs
docker-compose logs -f helios_app
docker-compose logs -f celery_worker

# Access services
# - pgAdmin: http://localhost:8080
# - Redis Commander: http://localhost:8081 (with dev profile)
# - Application: http://localhost:8000

# Stop services
docker-compose down
```

### Production Build

```bash
# Build production image
docker build -t helios:3.0.0 .

# Run with production compose
docker-compose -f docker-compose.yml --profile production up
```

## ☸️ Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace helios

# Apply configurations
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n helios
kubectl logs -f deployment/helios-app -n helios

# Scale workers
kubectl scale deployment helios-celery-worker --replicas=3 -n helios
```

## 🔧 System Sentinel

The System Sentinel provides comprehensive monitoring and self-healing:

- **Integrity Audits**: Syntax, imports, database schema, Redis connectivity
- **Auto-Repair**: Package installation, module reloading, cache clearing
- **Health Monitoring**: CPU, memory, disk space, service connectivity
- **Slack Alerts**: Real-time notifications with emoji and severity levels

### Sentinel Commands

```bash
# Check system status
python utils/sentinel.py --status

# Run dry-run audit (no repairs)
python utils/sentinel.py --dry-run

# Full audit with auto-repair
python utils/sentinel.py --audit

# View repair history
python utils/sentinel.py --history 20
```

## 📊 Monitoring & Alerts

### Slack Notifications

The system sends structured Slack messages with emoji indicators:

- ✅ **TRADE**: Order executions and fills
- ⚠️ **RISK**: Risk threshold breaches
- 💥 **SYSTEM**: Critical system events
- 📊 **PERFORMANCE**: Daily/weekly performance summaries
- 🔴 **ERROR**: System errors and failures
- ☀️ **SUCCESS**: System startup and milestones

### Health Endpoints

Health check endpoints for monitoring:

```python
# Programmatic health check
from utils.health import check_system_health
report = await check_system_health()

# Docker health check
python -c "from utils.health import health_check; health_check()"
```

## 🎯 Trading Strategies

### Available Strategies

1. **Trend Following**: SMA crossover with configurable periods
2. **Mean Reversion**: Bollinger Band-based entries and exits
3. **Momentum Breakout**: Volume-confirmed price breakouts

### Strategy Configuration

Edit `config/strategy.yaml` to:
- Enable/disable strategies
- Configure parameters
- Set position sizing rules
- Define risk limits

## 🔐 Security & Best Practices

- **Environment Variables**: All sensitive data via .env files
- **Database Security**: Connection pooling with timeout protection
- **API Rate Limiting**: Built-in throttling for broker APIs
- **Input Validation**: Pydantic models for configuration validation
- **Error Handling**: Comprehensive exception handling with circuit breakers

## 🧪 Testing

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Database, Redis, and broker integration
- **Strategy Tests**: Signal generation and risk management
- **End-to-End Tests**: Full system workflow validation

### Running Tests

```bash
# All tests
pytest -v

# Specific test categories
pytest -v -m "unit"          # Unit tests only
pytest -v -m "integration"   # Integration tests only
pytest -v tests/test_strategies/  # Strategy tests only

# Coverage report
pytest --cov=. --cov-report=html
```

## 📈 Performance & Scalability

- **Async Processing**: Full asyncio support for concurrent operations
- **Connection Pooling**: Efficient database and Redis connection management
- **Task Queues**: Celery for background processing and analytics
- **Caching**: Multi-layer caching for market data and computations
- **Horizontal Scaling**: Kubernetes-ready with multiple worker support

## 🚨 Troubleshooting

### Common Issues

1. **TA-Lib Installation on macOS**:
   ```bash
   brew install ta-lib
   pip install --upgrade --no-cache-dir TA-Lib
   ```

2. **Database Connection Issues**:
   ```bash
   # Check PostgreSQL status
   docker-compose ps postgres
   
   # Reset database
   docker-compose down -v
   docker-compose up postgres
   ```

3. **Redis Connection Issues**:
   ```bash
   # Test Redis connectivity
   python -c "from utils.redis_queue import test_redis_connection; print(test_redis_connection())"
   ```

4. **Celery Worker Issues**:
   ```bash
   # Check worker status
   celery -A workers.tasks inspect ping
   
   # Restart workers
   docker-compose restart celery_worker
   ```

### Logs and Diagnostics

```bash
# View application logs
tail -f logs/helios.log

# Check Sentinel repairs
cat logs/sentinel_repairs.json | jq .

# Database diagnostics
python -c "
    from utils.db import get_database_info
    import json
    print(json.dumps(get_database_info(), indent=2))
"
```

## 🎯 Next Steps

- **Expand Strategy Library**: Add more sophisticated algorithms
- **Enhanced Risk Management**: VAR calculations and correlation analysis
- **Real-time Dashboard**: Streamlit or FastAPI-based monitoring interface
- **Machine Learning Integration**: Feature engineering and model deployment
- **Multi-Asset Support**: Crypto, forex, and options trading
- **Advanced Order Types**: Bracket orders, trailing stops, conditional orders

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest -v`
5. Submit a pull request

---

**Helios v3** - Intelligent, resilient, production-ready quantitative trading.

*Built for reliability, designed for scale.*