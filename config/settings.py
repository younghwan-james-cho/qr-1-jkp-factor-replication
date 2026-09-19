# config/settings.py
#
# Central configuration file for the factor replication project.
#

from pathlib import Path

# --- Project Structure ---
# Defines the root directory of the project for consistent path management.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Data Ingestion Settings ---
# The single source of truth for the analysis start date.
START_DATE_STR = "2020-01-01"  # setting 1
# START_DATE_STR = '1963-07-01' # setting 2
# START_DATE_STR = '2000-01-01' # setting 3

# --- Validation Settings ---
# The correlation threshold to determine a successful replication.
CORRELATION_THRESHOLD = 0.95

# --- Replication Settings ---
# A list of all factors to be replicated using the single-sort method.
# Format: (characteristic_column, benchmark_factor_name, long_short_direction)
FACTORS_TO_REPLICATE = [
    ("me", "market_equity", "short"),  # Size
    ("be_me", "be_me", "long"),  # Value
    ("ope_be", "ope_be", "long"),  # Profitability
    ("at_gr1", "at_gr1", "short"),  # Investment
    ("ret_12_1", "ret_12_1", "long"),  # Momentum
    ("ret_1_0", "ret_1_0", "short"),  # Short-Term Reversal
]

# Defines each weighting scheme and its corresponding benchmark file path.
SCHEMES = {
    "vw_cap": PROJECT_ROOT / "data/raw/factor/usa_all_factors_monthly_vw_cap.csv",
    "ew": PROJECT_ROOT / "data/raw/factor/usa_all_factors_monthly_ew.csv",
    "vw": PROJECT_ROOT / "data/raw/factor/usa_all_factors_monthly_vw.csv",
}
