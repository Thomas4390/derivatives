# Plan d'Améliorations du Backend - Revue Critique

## Résumé Exécutif

Cette revue critique a analysé les **61 fichiers Python** du backend (~22,000 lignes de code). L'architecture globale est **bien conçue** avec une séparation claire des responsabilités (Three Pillars: Instrument, Model, Engine). Cependant, plusieurs problèmes ont été identifiés nécessitant des corrections.

---

## Table des Matières

1. [Problèmes Critiques (HIGH)](#1-problèmes-critiques-high)
2. [Problèmes Importants (MEDIUM)](#2-problèmes-importants-medium)
3. [Améliorations Mineures (LOW)](#3-améliorations-mineures-low)
4. [Résumé par Module](#4-résumé-par-module)
5. [Plan d'Exécution](#5-plan-dexécution)

---

## 1. Problèmes Critiques (HIGH)

### 1.1 Code Legacy à Supprimer

**Fichier:** `backend/old/option_pricing.py` (1,041 lignes)

**Problème:** Ce fichier monolithique contient du code entièrement redondant avec l'architecture moderne:
- Réimplémente `norm_cdf()`, `norm_pdf()` (déjà dans `utils/math.py`)
- Réimplémente le calcul des Greeks (déjà dans `greeks/analytic.py`)
- Classes `OptionPosition`, `StockPosition` redondantes avec `instruments/` et `portfolio/`
- N'est importé nulle part dans le codebase

**Action:**
```bash
rm backend/old/option_pricing.py
rmdir backend/old/  # si vide après suppression
```

**Impact:** Réduction de ~1,000 lignes de code mort, clarification de l'architecture.

---

### 1.2 Duplication des Formules Black-Scholes

**Problème:** Les formules BS (d1, d2, pricing, vega) sont dupliquées dans 3 fichiers:

| Fichier | Lignes | Contenu |
|---------|--------|---------|
| `engines/analytic_engine.py` | 250-300 | `_d1_d2()`, `_bs_price()`, `_bs_vega()` |
| `greeks/analytic.py` | 100-138 | `_d1_d2()`, calculs BS |
| `engines/vectorized_bs.py` | 62-73 | Formules BS réimplémentées |

**Risque:** Les corrections de bugs dans un fichier ne sont pas propagées aux autres.

**Action:**
1. Créer un module unique `backend/core/bs_formulas.py` contenant:
   - `d1_d2(s, k, t, r, q, sigma)` - Numba-optimisé
   - `bs_call_price()`, `bs_put_price()`
   - `bs_vega()`
2. Modifier les 3 fichiers pour importer depuis ce module unique
3. Supprimer les implémentations locales

**Fichiers à modifier:**
- `backend/core/bs_formulas.py` (nouveau)
- `backend/engines/analytic_engine.py`
- `backend/greeks/analytic.py`
- `backend/engines/vectorized_bs.py`

---

### 1.3 Modèles GARCH Sans Interface Model

**Fichier:** `backend/models/garch.py`

**Problème:** Les classes `GARCHModel`, `NGARCHModel`, `GJRGARCHModel` sont des frozen dataclasses mais **n'héritent pas** de la classe `Model` définie dans `core/interfaces.py`.

**Conséquence:**
- Violation du principe de Liskov (SOLID)
- Ces modèles ne peuvent pas être utilisés avec le système de pricing unifié
- `EngineRegistry` ne peut pas les gérer

**Action:**
```python
# backend/models/garch.py
from backend.core.interfaces import Model

@dataclass(frozen=True)
class GARCHModel(Model):  # Ajouter héritage
    ...

    @property
    def name(self) -> str:
        return "GARCH"

    @property
    def supported_engines(self) -> List[PricingCapability]:
        return [PricingCapability.MONTE_CARLO]

    def get_parameters(self) -> dict:
        return {"omega": self.omega, "alpha": self.alpha, "beta": self.beta}
```

**Fichiers à modifier:**
- `backend/models/garch.py` (3 classes à mettre à jour)

---

### 1.4 Registry Engine Instantiation Défaillante

**Fichier:** `backend/core/registry.py`

**Problème:** La méthode `get_engine()` instancie les engines avec `cls._engines[key]()` sans arguments:
```python
# Ligne problématique
engine = cls._engines[key]()  # Appelle MonteCarloEngine() sans args
```

**Conséquence:** `MonteCarloEngine(n_paths=50000, seed=42)` enregistré ne peut pas être instancié.

**Action:** Implémenter un pattern Factory:
```python
# Option 1: Stocker des factory functions
_engine_factories: Dict[Tuple[str, PricingCapability], Callable[[], PricingEngine]] = {}

@classmethod
def register(cls, model_name: str, capability: PricingCapability,
             engine_factory: Callable[[], PricingEngine]):
    cls._engine_factories[(model_name, capability)] = engine_factory

# Option 2: Stocker les paramètres avec la classe
_engines: Dict[Tuple[str, PricingCapability], Tuple[Type[PricingEngine], dict]] = {}
```

**Fichiers à modifier:**
- `backend/core/registry.py`

---

### 1.5 Calcul Vega Échoue Silencieusement

**Fichier:** `backend/portfolio/portfolio.py` (lignes 458-490)

**Problème:** `_compute_vega()` retourne `0.0` si le modèle n'a pas de paramètre `sigma` ou `v0`:
```python
if hasattr(model, 'sigma'):
    # calcul vega
elif hasattr(model, 'v0'):
    # calcul vega Heston
else:
    return 0.0  # ÉCHEC SILENCIEUX!
```

**Conséquence:** Les modèles GARCH retournent vega=0 sans avertissement.

**Action:**
```python
else:
    raise NotImplementedError(
        f"Vega computation not supported for model type {type(model).__name__}. "
        f"Model must have 'sigma' or 'v0' parameter."
    )
```

**Fichiers à modifier:**
- `backend/portfolio/portfolio.py`

---

### 1.6 Engine PDE Non Fonctionnel

**Fichier:** `backend/engines/pde/fd_engine.py` (186 lignes)

**Problème:** Ce fichier est un stub qui lève `NotImplementedError`:
```python
def price(self, instrument, model, market) -> PricingResult:
    raise NotImplementedError("FD pricing not yet implemented")
```

**Conséquence:** Si sélectionné par le registry, cause un crash runtime.

**Action:**
- Option A: Supprimer le fichier et retirer PDE de `PricingCapability`
- Option B: Implémenter le solveur (Crank-Nicolson)

**Recommandation:** Option A pour l'instant, documenter comme future feature.

**Fichiers à modifier:**
- `backend/engines/pde/fd_engine.py` (supprimer ou implémenter)
- `backend/engines/pde/__init__.py`
- `backend/core/result_types.py` (retirer PDE si suppression)

---

## 2. Problèmes Importants (MEDIUM)

### 2.1 Scaling Incohérent des Greeks

**Fichier:** `backend/greeks/analytic.py`

**Problème:** Les Greeks sont scalés différemment:

| Greek | Scaling | Ligne |
|-------|---------|-------|
| Vega | per 1% vol (`/100.0`) | 215 |
| Rho | per 1% rate (`/100.0`) | 233 |
| Volga | per 1%² (`/10000.0`) | 316 |
| Vanna | per 1% (`/100.0`) | 316 |

**Conséquence:** L'agrégation des Greeks de portefeuille peut être incorrecte.

**Action:**
1. Documenter clairement le scaling dans les docstrings
2. Ajouter une constante de scaling configurable
3. Créer une fonction `unscale_greeks()` pour obtenir les valeurs brutes

**Fichiers à modifier:**
- `backend/greeks/analytic.py`
- `backend/greeks/calculator.py`

---

### 2.2 Duplication dans les Modèles GARCH

**Fichier:** `backend/models/garch.py`

**Problème:** Les 3 classes GARCH répètent du code quasi-identique:
- Validation des paramètres (lignes 71-87, 225-241, 359-377)
- Properties `persistence`, `long_run_variance`, `long_run_volatility`
- Méthodes `create_simulator()`, `create_pricer()`

**Action:** Extraire une classe de base:
```python
@dataclass(frozen=True)
class BaseGARCHModel(Model):
    omega: float
    alpha: float
    beta: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run_variance(self) -> float:
        return self.omega / (1 - self.persistence)

    # ... méthodes communes

@dataclass(frozen=True)
class GARCHModel(BaseGARCHModel):
    # Spécificités GARCH(1,1)

@dataclass(frozen=True)
class NGARCHModel(BaseGARCHModel):
    theta: float  # Paramètre supplémentaire
```

**Fichiers à modifier:**
- `backend/models/garch.py`

---

### 2.3 Greeks Numériques Incomplets

**Fichier:** `backend/greeks/numerical.py`

**Problème:** Seuls `vanna` et `volga` sont implémentés pour les Greeks de second ordre. Les autres retournent `0.0`:
- `charm`, `veta`, `speed`, `zomma`, `color`, `ultima` manquants

**Action:** Implémenter les Greeks manquants par différences finies:
```python
def finite_difference_charm(engine, model, instrument, market, h_s=None, h_t=None):
    """Charm = ∂Delta/∂t = ∂²V/∂S∂t"""
    # Implémenter différence croisée spot-time
```

**Fichiers à modifier:**
- `backend/greeks/numerical.py`
- `backend/greeks/calculator.py`

---

### 2.4 Factories d'Options Dupliquées

**Fichier:** `backend/instruments/options.py` (lignes 255-403)

**Problème:** 6 fonctions factory quasi-identiques:
- `EuropeanCall()`, `EuropeanPut()`
- `AmericanCall()`, `AmericanPut()`
- `BermudanCall()`, `BermudanPut()`

**Action:** Créer une factory générique:
```python
def create_vanilla_option(
    strike: float,
    maturity: float,
    is_call: bool,
    exercise: ExerciseStyle = ExerciseStyle.EUROPEAN
) -> VanillaOption:
    return VanillaOption(strike, maturity, is_call, exercise)

# Aliases pour compatibilité
EuropeanCall = lambda s, t: create_vanilla_option(s, t, True, ExerciseStyle.EUROPEAN)
```

**Fichiers à modifier:**
- `backend/instruments/options.py`

---

### 2.5 Bumps Greeks Hard-Codés

**Fichier:** `backend/greeks/numerical.py` (lignes 357-360)

**Problème:** Les perturbations pour les différences finies sont hard-codées:
```python
h_s = spot * 0.01  # 1% du spot
h_v = 0.01         # 1% absolu
h_t = 1/365        # 1 jour
h_r = 0.0001       # 1bp
```

**Risque:** Pour des spots très petits, `h_s` peut causer du bruit numérique.

**Action:**
```python
@dataclass
class GreeksBumpConfig:
    spot_pct: float = 0.01
    vol_abs: float = 0.01
    time_days: float = 1/365
    rate_bps: float = 0.0001
    min_spot_bump: float = 0.01  # Plancher pour éviter bruit
```

**Fichiers à modifier:**
- `backend/greeks/numerical.py`
- `backend/greeks/calculator.py`

---

### 2.6 Characteristic Function Inconsistante

**Fichiers:** `backend/models/gbm.py`, `backend/models/heston.py`

**Problème:**
- GBM définit `characteristic_function()` inline avec Numba
- Heston délègue à `heston_cf.py`
- FFT engine appelle `model.characteristic_function_vectorized()` qui n'existe pas dans l'ABC

**Action:**
1. Ajouter `characteristic_function_vectorized()` à l'interface `Model`
2. Standardiser le pattern dans tous les modèles

**Fichiers à modifier:**
- `backend/core/interfaces.py`
- `backend/models/gbm.py`
- `backend/models/heston.py`
- `backend/models/merton.py`
- `backend/models/bates.py`

---

## 3. Améliorations Mineures (LOW)

### 3.1 Docstrings Manquantes dans vectorized_bs.py

**Fichier:** `backend/engines/vectorized_bs.py`

**Problème:** Les fonctions Numba n'ont pas de docstrings après la ligne 100.

**Action:** Ajouter des docstrings expliquant:
- Le format des paramètres (`option_type: int` → 1=call, 0=put)
- Les dimensions des arrays retournés
- Le scaling utilisé

---

### 3.2 Smoke Tests dans les Fichiers Source

**Fichiers:** Tous les fichiers de modèles (`gbm.py`, `heston.py`, etc.)

**Problème:** Tests embarqués avec `if __name__ == "__main__":`

**Action:** Migrer vers le dossier `tests/` avec pytest.

---

### 3.3 Type Hints Perdus dans Numba

**Problème:** Les fonctions `@njit` perdent leurs type hints pour l'IDE.

**Action:** Créer des wrappers Python avec type hints:
```python
def norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return _norm_cdf_numba(x)

@njit
def _norm_cdf_numba(x):
    # Implémentation
```

---

### 3.4 Validation Feller Optionnelle

**Fichier:** `backend/utils/validation.py`

**Problème:** `validate_heston_parameters()` a `check_feller=True` optionnel.

**Recommandation:** Forcer par défaut, avertir si désactivé.

---

### 3.5 math_kernels Non Intégré

**Dossier:** `backend/math_kernels/`

**Statut:** Intentionnellement standalone (implémentations de référence).

**Action:** Documenter ce choix architectural dans le README ou `__init__.py`.

---

## 4. Résumé par Module

| Module | Statut | Issues | Priorité Max |
|--------|--------|--------|--------------|
| `core/` | Bon | Registry instantiation | HIGH |
| `engines/` | Bon | Duplication BS, PDE stub | HIGH |
| `greeks/` | Moyen | Scaling, Greeks incomplets | MEDIUM |
| `instruments/` | Bon | Factories dupliquées | MEDIUM |
| `models/` | Moyen | GARCH sans interface, duplication | HIGH |
| `portfolio/` | Moyen | Vega silencieux | HIGH |
| `simulation/` | Excellent | Aucun | - |
| `utils/` | Excellent | Feller optionnel | LOW |
| `math_kernels/` | Bon | Non intégré (by design) | LOW |
| `old/` | À supprimer | Code mort | HIGH |

---

## 5. Plan d'Exécution

### Phase 1: Nettoyage (Estimé: 1h)

1. [ ] Supprimer `backend/old/option_pricing.py`
2. [ ] Supprimer ou implémenter `backend/engines/pde/fd_engine.py`
3. [ ] Migrer les smoke tests vers `tests/`

### Phase 2: Corrections Critiques (Estimé: 3h)

4. [ ] Créer `backend/core/bs_formulas.py` avec formules unifiées
5. [ ] Refactorer `analytic_engine.py`, `greeks/analytic.py`, `vectorized_bs.py`
6. [ ] Faire hériter les modèles GARCH de `Model`
7. [ ] Corriger l'échec silencieux de vega dans `portfolio.py`
8. [ ] Corriger le registry pour supporter les engines paramétrés

### Phase 3: Améliorations (Estimé: 2h)

9. [ ] Extraire `BaseGARCHModel` pour réduire duplication
10. [ ] Documenter le scaling des Greeks
11. [ ] Implémenter les Greeks numériques manquants
12. [ ] Refactorer les factories d'options

### Phase 4: Polissage (Estimé: 1h)

13. [ ] Ajouter docstrings à `vectorized_bs.py`
14. [ ] Ajouter type hints wrappers pour Numba
15. [ ] Documenter `math_kernels/` comme standalone
16. [ ] Forcer validation Feller par défaut

---

## Vérification

Après implémentation, vérifier avec:

```bash
# Tests unitaires
pytest tests/ -v

# Type checking
mypy backend/ --ignore-missing-imports

# Vérifier les imports
python -c "from backend import *; print('Imports OK')"

# Smoke test pricing
python -c "
from backend import GBMModel, VanillaOption, MarketEnvironment, BSAnalyticEngine
model = GBMModel(sigma=0.2)
option = VanillaOption(strike=100, maturity=0.25, is_call=True)
market = MarketEnvironment(spot=100, rate=0.05)
engine = BSAnalyticEngine()
result = engine.price(option, model, market)
print(f'Price: {result.price:.4f}')
"
```

---

## Statistiques Finales

- **Fichiers analysés:** 61
- **Lignes de code:** ~22,000
- **Issues HIGH:** 6
- **Issues MEDIUM:** 6
- **Issues LOW:** 5
- **Code à supprimer:** ~1,200 lignes
- **Temps estimé total:** 7 heures
