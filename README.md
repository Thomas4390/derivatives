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

Self-contained Jupyter notebooks

### 2. Interactive Applications (`streamlit_apps/`)

**Architecture**: Each app has its frontend (Streamlit UI) separated from backend (business logic)

### 3. Backend (`backend/`)

Reusable business logic

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
