# Derivatives Course 📊

Interactive educational material for learning financial derivatives.

## 📁 Repository Structure
```
derivatives/
├── data/              # Market data and examples
├── notebooks/         # Jupyter notebooks for learning
├── backend/           # Business logic and calculations
├── streamlit_apps/    # Interactive Streamlit applications
└── docs/              # Additional documentation
```

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone git@github.com:Thomas4390/derivatives.git
cd derivatives

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Launch Streamlit Applications

Each application can be launched independently:
```bash
# Option Pricer
streamlit run streamlit_apps/option_pricer/app.py

# Volatility Surface
streamlit run streamlit_apps/volatility_surface/app.py

# Strategy Builder
streamlit run streamlit_apps/strategy_builder/app.py

# Monte Carlo Simulator
streamlit run streamlit_apps/monte_carlo_simulator/app.py
```

## 📚 Course Content

### 1. Educational Notebooks (`notebooks/`)

Self-contained Jupyter notebooks covering:
- **introduction.ipynb**: Basic concepts and derivatives markets
- **forwards_futures.ipynb**: Forward and futures contracts
- **options_basics.ipynb**: European and American options
- **black_scholes.ipynb**: Black-Scholes model and implementation
- **greeks.ipynb**: Option sensitivities (Delta, Gamma, Vega, Theta, Rho)
- **volatility.ipynb**: Volatility modeling and implied volatility
- **swaps.ipynb**: Interest rate and currency swaps
- **trading_strategies.ipynb**: Advanced trading strategies

### 2. Interactive Applications (`streamlit_apps/`)

#### Option Pricer
Interactive tool for pricing options with:
- European and American options
- Black-Scholes and Binomial tree models
- Real-time Greeks calculation
- Payoff diagrams

#### Volatility Surface
Visualize and analyze volatility:
- 3D volatility surface plotting
- Implied volatility calculation
- Volatility smile/skew analysis

#### Strategy Builder
Build and visualize option strategies:
- Single leg positions
- Vertical and horizontal spreads
- Complex multi-leg strategies
- P&L analysis

#### Monte Carlo Simulator
Simulate asset prices and option pricing:
- Geometric Brownian Motion simulation
- Path visualization
- Option pricing via Monte Carlo

**Architecture**: Each app has its frontend (Streamlit UI) separated from backend (business logic)

### 3. Backend (`backend/`)

Reusable business logic:
- **models/**: Pricing models (Black-Scholes, Binomial, Monte Carlo)
- **calculators/**: Greeks calculator, implied volatility solver
- **data_handlers/**: Data loading and processing utilities
- **validators/**: Input parameter validation

### 4. Data (`data/`)

- **raw/**: Raw market data (should not be modified)
- **processed/**: Cleaned and prepared data
- **examples/**: Small datasets for tutorials

## 🛠️ Technologies

- **Python 3.9+**
- **NumPy/Pandas**: Numerical computing
- **Matplotlib/Plotly**: Visualizations
- **Streamlit**: Interactive web applications
- **Jupyter**: Educational notebooks
- **SciPy**: Scientific computing

## 📖 Usage Guide

### For Students

1. Start with **notebooks** in suggested order
2. Explore **Streamlit applications** to visualize concepts
3. Experiment with parameters to build intuition

### For Instructors

- Notebooks can be modified for lectures
- Streamlit apps are customizable in each `streamlit_apps/*/pages/` directory
- Backend is modular and extensible

## 📝 License

[MIT License](LICENSE) - Free for educational use
