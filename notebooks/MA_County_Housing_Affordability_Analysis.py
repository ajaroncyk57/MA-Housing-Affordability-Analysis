# Massachusetts County Housing Affordability Analysis
# Exported from notebook as a GitHub-render-safe fallback.


# ==============================================================================
# # Massachusetts County Housing Affordability Analysis
# 
# This notebook pulls official U.S. Census ACS 5-Year county-level data for Massachusetts and evaluates housing affordability using income, home value, rent, tenure, vacancy, cost-burden, and estimated ownership-cost metrics.
# 
# **Primary question:** How does housing affordability pressure vary across Massachusetts counties for renters and prospective homeowners?
# 
# **Official data source:** U.S. Census Bureau ACS 5-Year API.
# 
# **Note:** This notebook intentionally uses official public data sources and does not rely on a locally maintained raw CSV file.

# ==============================================================================
# ## 1. Setup
# 
# Import packages, configure display options, and define local output folders for processed data and charts.

# Cell 2
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

DATA_DIR = Path("../data/processed")
CHART_DIR = Path("../outputs/charts")

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# ## 2. Pull ACS County Data
# 
# This section pulls Massachusetts county-level ACS 5-Year data directly from the Census API.
# 
# The selected variables focus on population, households, income, housing units, tenure, median home value, median gross rent, housing age, bedroom mix, and housing cost burden.

# Cell 4
ACS_YEAR = "2024"
BASE_URL = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"

variables = {
    # Geography
    "NAME": "geography_name",

    # Population / households
    "B01003_001E": "population",
    "B11001_001E": "total_households",

    # Income
    "B19013_001E": "median_household_income",

    # Housing stock / tenure
    "B25001_001E": "total_housing_units",
    "B25002_001E": "total_occupancy_status_units",
    "B25002_002E": "occupied_housing_units",
    "B25002_003E": "vacant_housing_units",
    "B25003_001E": "occupied_tenure_units",
    "B25003_002E": "owner_occupied_units",
    "B25003_003E": "renter_occupied_units",

    # Home value / rent
    "B25077_001E": "median_home_value",
    "B25064_001E": "median_gross_rent",

    # Housing age / bedroom mix
    "B25035_001E": "median_year_structure_built",
    "B25041_001E": "total_bedroom_units",
    "B25041_002E": "no_bedroom_units",
    "B25041_003E": "one_bedroom_units",
    "B25041_004E": "two_bedroom_units",
    "B25041_005E": "three_bedroom_units",
    "B25041_006E": "four_bedroom_units",
    "B25041_007E": "five_plus_bedroom_units",

    # Owner costs as % of household income
    "B25091_001E": "owner_cost_ratio_total",
    "B25091_002E": "owner_cost_under_20_pct",
    "B25091_003E": "owner_cost_20_to_24_9_pct",
    "B25091_004E": "owner_cost_25_to_29_9_pct",
    "B25091_005E": "owner_cost_30_to_34_9_pct",
    "B25091_006E": "owner_cost_35_pct_or_more",

    # Gross rent as % of household income
    "B25070_001E": "rent_burden_total",
    "B25070_002E": "rent_under_10_pct",
    "B25070_003E": "rent_10_to_14_9_pct",
    "B25070_004E": "rent_15_to_19_9_pct",
    "B25070_005E": "rent_20_to_24_9_pct",
    "B25070_006E": "rent_25_to_29_9_pct",
    "B25070_007E": "rent_30_to_34_9_pct",
    "B25070_008E": "rent_35_to_39_9_pct",
    "B25070_009E": "rent_40_to_49_9_pct",
    "B25070_010E": "rent_50_pct_or_more",
    "B25070_011E": "rent_not_computed",
}

params = {
    "get": ",".join(variables.keys()),
    "for": "county:*",
    "in": "state:25",  # Massachusetts
}

api_url = f"{BASE_URL}?{urlencode(params)}"
api_url

# Cell 5
raw_data = pd.read_json(api_url)

ma_county_df = pd.DataFrame(
    raw_data.iloc[1:].values,
    columns=raw_data.iloc[0].values,
)

ma_county_df = ma_county_df.rename(columns=variables)

id_cols = ["state", "county", "geography_name"]

for col in ma_county_df.columns:
    if col not in id_cols:
        ma_county_df[col] = pd.to_numeric(ma_county_df[col], errors="coerce")

ma_county_df["county_name"] = (
    ma_county_df["geography_name"]
    .str.replace(", Massachusetts", "", regex=False)
)

print("Raw county data shape:", ma_county_df.shape)
display(ma_county_df.head())

# ==============================================================================
# ## 3. Create Base Affordability Metrics
# 
# This section creates the main renter, owner, tenure, and housing stock metrics used throughout the analysis.

# Cell 7
def safe_divide(numerator, denominator):
    """Safely divides two values and returns NaN when denominator is zero."""
    return np.where(denominator == 0, np.nan, numerator / denominator)


ma_county_df["monthly_median_income"] = (
    ma_county_df["median_household_income"] / 12
)

ma_county_df["affordable_monthly_housing_cost"] = (
    ma_county_df["monthly_median_income"] * 0.30
)

ma_county_df["home_value_to_income_ratio"] = safe_divide(
    ma_county_df["median_home_value"],
    ma_county_df["median_household_income"],
)

ma_county_df["rent_to_income_ratio"] = safe_divide(
    ma_county_df["median_gross_rent"],
    ma_county_df["monthly_median_income"],
)

ma_county_df["rent_affordability_gap"] = (
    ma_county_df["median_gross_rent"]
    - ma_county_df["affordable_monthly_housing_cost"]
)

ma_county_df["owner_occupied_share"] = safe_divide(
    ma_county_df["owner_occupied_units"],
    ma_county_df["occupied_housing_units"],
)

ma_county_df["renter_occupied_share"] = safe_divide(
    ma_county_df["renter_occupied_units"],
    ma_county_df["occupied_housing_units"],
)

ma_county_df["vacancy_rate"] = safe_divide(
    ma_county_df["vacant_housing_units"],
    ma_county_df["total_housing_units"],
)

ma_county_df["renter_cost_burdened_units"] = (
    ma_county_df["rent_30_to_34_9_pct"]
    + ma_county_df["rent_35_to_39_9_pct"]
    + ma_county_df["rent_40_to_49_9_pct"]
    + ma_county_df["rent_50_pct_or_more"]
)

ma_county_df["renter_cost_burden_rate"] = safe_divide(
    ma_county_df["renter_cost_burdened_units"],
    ma_county_df["rent_burden_total"],
)

ma_county_df["severely_rent_burdened_rate"] = safe_divide(
    ma_county_df["rent_50_pct_or_more"],
    ma_county_df["rent_burden_total"],
)

ma_county_df["owner_cost_burdened_units"] = (
    ma_county_df["owner_cost_30_to_34_9_pct"]
    + ma_county_df["owner_cost_35_pct_or_more"]
)

ma_county_df["owner_cost_burden_rate"] = safe_divide(
    ma_county_df["owner_cost_burdened_units"],
    ma_county_df["owner_cost_ratio_total"],
)

# ==============================================================================
# ## 4. Build Analysis Dataset
# 
# This creates the clean county-level analysis table used for rankings, charts, owner-cost estimates, and exports.

# Cell 9
analysis_cols = [
    "county_name",
    "state",
    "county",
    "population",
    "total_households",
    "median_household_income",
    "monthly_median_income",
    "affordable_monthly_housing_cost",
    "median_home_value",
    "median_gross_rent",
    "home_value_to_income_ratio",
    "rent_to_income_ratio",
    "rent_affordability_gap",
    "owner_occupied_share",
    "renter_occupied_share",
    "vacancy_rate",
    "owner_cost_burden_rate",
    "renter_cost_burden_rate",
    "severely_rent_burdened_rate",
]

ma_analysis_df = ma_county_df[analysis_cols].copy()

print("Analysis data shape:", ma_analysis_df.shape)
display(ma_analysis_df.head())

# Cell 10
missing_values = ma_analysis_df.isna().sum().sort_values(ascending=False)

display(missing_values)
display(ma_analysis_df.describe().T)

# ==============================================================================
# ## 5. County Rankings
# 
# These tables identify counties with the highest owner and renter affordability pressure using ACS income, home value, rent, and cost-burden metrics.

# Cell 12
owner_affordability_ranking = ma_analysis_df[
    [
        "county_name",
        "median_household_income",
        "median_home_value",
        "home_value_to_income_ratio",
        "owner_cost_burden_rate",
    ]
].sort_values("home_value_to_income_ratio", ascending=False)

display(owner_affordability_ranking)

# Cell 13
renter_affordability_ranking = ma_analysis_df[
    [
        "county_name",
        "median_household_income",
        "median_gross_rent",
        "rent_to_income_ratio",
        "rent_affordability_gap",
        "renter_cost_burden_rate",
        "severely_rent_burdened_rate",
    ]
].sort_values("rent_to_income_ratio", ascending=False)

display(renter_affordability_ranking)

# ==============================================================================
# ## 6. Affordability Pressure Labels
# 
# This simple classification helps summarize whether a county shows lower, moderate, or high affordability pressure based on selected owner and renter metrics.

# Cell 15
def classify_affordability_pressure(row):
    if (
        row["home_value_to_income_ratio"] >= 5
        or row["rent_to_income_ratio"] >= 0.35
        or row["renter_cost_burden_rate"] >= 0.50
    ):
        return "High Pressure"

    if (
        row["home_value_to_income_ratio"] >= 4
        or row["rent_to_income_ratio"] >= 0.30
        or row["renter_cost_burden_rate"] >= 0.40
    ):
        return "Moderate Pressure"

    return "Lower Pressure"


ma_analysis_df["affordability_pressure"] = ma_analysis_df.apply(
    classify_affordability_pressure,
    axis=1,
)

display(
    ma_analysis_df[
        [
            "county_name",
            "home_value_to_income_ratio",
            "rent_to_income_ratio",
            "renter_cost_burden_rate",
            "affordability_pressure",
        ]
    ].sort_values("home_value_to_income_ratio", ascending=False)
)

# ==============================================================================
# ## 7. Visual Analysis: Base Affordability Metrics
# 
# The following charts compare home value pressure, rent pressure, and renter cost burden across Massachusetts counties. Each chart is saved locally to the `outputs/charts` folder.

# Cell 17
plot_df = ma_analysis_df.sort_values("home_value_to_income_ratio")

plt.figure(figsize=(10, 7))
plt.barh(plot_df["county_name"], plot_df["home_value_to_income_ratio"])
plt.axvline(4, linestyle="--", label="4x Income")
plt.axvline(5, linestyle="--", label="5x Income")
plt.title("Median Home Value-to-Income Ratio by Massachusetts County")
plt.xlabel("Median Home Value / Median Household Income")
plt.ylabel("County")
plt.legend()
plt.tight_layout()

plt.savefig(
    CHART_DIR / "home_value_to_income_ratio_by_county.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

# Cell 18
plot_df = ma_analysis_df.sort_values("rent_to_income_ratio")

plt.figure(figsize=(10, 7))
plt.barh(plot_df["county_name"], plot_df["rent_to_income_ratio"])
plt.axvline(0.30, linestyle="--", label="30% Affordability Threshold")
plt.title("Median Gross Rent as a Share of Monthly Income by Massachusetts County")
plt.xlabel("Median Gross Rent / Monthly Median Household Income")
plt.ylabel("County")
plt.legend()
plt.tight_layout()

plt.savefig(
    CHART_DIR / "rent_to_income_ratio_by_county.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

# Cell 19
plot_df = ma_analysis_df.sort_values("renter_cost_burden_rate")

plt.figure(figsize=(10, 7))
plt.barh(plot_df["county_name"], plot_df["renter_cost_burden_rate"])
plt.axvline(0.30, linestyle="--", label="30% Cost-Burden Reference")
plt.title("Share of Renter Households Cost-Burdened by Massachusetts County")
plt.xlabel("Renter Cost-Burden Rate")
plt.ylabel("County")
plt.legend()
plt.tight_layout()

plt.savefig(
    CHART_DIR / "renter_cost_burden_rate_by_county.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

# Cell 20
plt.figure(figsize=(9, 6))
plt.scatter(
    ma_analysis_df["median_household_income"],
    ma_analysis_df["median_home_value"],
)

for _, row in ma_analysis_df.iterrows():
    plt.annotate(
        row["county_name"],
        (
            row["median_household_income"],
            row["median_home_value"],
        ),
        fontsize=8,
        alpha=0.8,
    )

plt.title("Median Household Income vs. Median Home Value by MA County")
plt.xlabel("Median Household Income")
plt.ylabel("Median Home Value")
plt.tight_layout()

plt.savefig(
    CHART_DIR / "income_vs_home_value_by_county.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

# ==============================================================================
# ## 8. Estimated Monthly Ownership Cost Analysis
# 
# To extend the affordability analysis beyond home value-to-income ratios, this section estimates a monthly owner cost for the median-valued home in each Massachusetts county.
# 
# The estimate uses a standard mortgage payment formula and assumes:
# - 20% down payment.
# - 30-year fixed-rate mortgage.
# - User-defined annual interest rate.
# - Estimated property tax rate.
# - Estimated annual homeowners insurance.
# - No HOA or PMI by default.
# 
# This allows county-level median home values to be translated into an estimated monthly housing cost and compared against the traditional 30% gross-income affordability benchmark.

# Cell 22
# Mortgage / ownership cost assumptions.
# These are intentionally centralized here so the scenario can be adjusted easily.

DOWN_PAYMENT_RATE = 0.20
ANNUAL_INTEREST_RATE = 0.065  # 6.5%
MORTGAGE_TERM_YEARS = 30
ANNUAL_PROPERTY_TAX_RATE = 0.011  # 1.1% estimate
ANNUAL_HOME_INSURANCE = 1800
MONTHLY_HOA = 0
MONTHLY_PMI = 0

MONTHS_PER_YEAR = 12


def calculate_monthly_principal_and_interest(
    home_value,
    down_payment_rate=DOWN_PAYMENT_RATE,
    annual_interest_rate=ANNUAL_INTEREST_RATE,
    mortgage_term_years=MORTGAGE_TERM_YEARS,
):
    """
    Estimate monthly principal and interest payment for a fixed-rate mortgage.
    """
    loan_amount = home_value * (1 - down_payment_rate)
    monthly_rate = annual_interest_rate / MONTHS_PER_YEAR
    number_of_payments = mortgage_term_years * MONTHS_PER_YEAR

    if pd.isna(home_value) or loan_amount <= 0:
        return np.nan

    if monthly_rate == 0:
        return loan_amount / number_of_payments

    payment = loan_amount * (
        monthly_rate * (1 + monthly_rate) ** number_of_payments
    ) / (
        (1 + monthly_rate) ** number_of_payments - 1
    )

    return payment


def calculate_estimated_monthly_owner_cost(
    home_value,
    down_payment_rate=DOWN_PAYMENT_RATE,
    annual_interest_rate=ANNUAL_INTEREST_RATE,
    mortgage_term_years=MORTGAGE_TERM_YEARS,
    annual_property_tax_rate=ANNUAL_PROPERTY_TAX_RATE,
    annual_home_insurance=ANNUAL_HOME_INSURANCE,
    monthly_hoa=MONTHLY_HOA,
    monthly_pmi=MONTHLY_PMI,
):
    """
    Estimate total monthly owner cost including principal, interest,
    property tax, insurance, HOA, and PMI.
    """
    if pd.isna(home_value):
        return np.nan

    principal_and_interest = calculate_monthly_principal_and_interest(
        home_value=home_value,
        down_payment_rate=down_payment_rate,
        annual_interest_rate=annual_interest_rate,
        mortgage_term_years=mortgage_term_years,
    )

    monthly_property_tax = (home_value * annual_property_tax_rate) / MONTHS_PER_YEAR
    monthly_insurance = annual_home_insurance / MONTHS_PER_YEAR

    total_owner_cost = (
        principal_and_interest
        + monthly_property_tax
        + monthly_insurance
        + monthly_hoa
        + monthly_pmi
    )

    return total_owner_cost

# Cell 23
ma_analysis_df["estimated_monthly_principal_interest"] = (
    ma_analysis_df["median_home_value"]
    .apply(calculate_monthly_principal_and_interest)
)

ma_analysis_df["estimated_monthly_owner_cost"] = (
    ma_analysis_df["median_home_value"]
    .apply(calculate_estimated_monthly_owner_cost)
)

ma_analysis_df["owner_affordability_gap"] = (
    ma_analysis_df["estimated_monthly_owner_cost"]
    - ma_analysis_df["affordable_monthly_housing_cost"]
)

ma_analysis_df["estimated_owner_cost_to_income_ratio"] = safe_divide(
    ma_analysis_df["estimated_monthly_owner_cost"],
    ma_analysis_df["monthly_median_income"],
)

owner_cost_analysis_cols = [
    "county_name",
    "median_household_income",
    "median_home_value",
    "affordable_monthly_housing_cost",
    "estimated_monthly_principal_interest",
    "estimated_monthly_owner_cost",
    "estimated_owner_cost_to_income_ratio",
    "owner_affordability_gap",
]

display(
    ma_analysis_df[owner_cost_analysis_cols]
    .sort_values("owner_affordability_gap", ascending=False)
)

# Cell 24
def classify_owner_affordability(row):
    if row["estimated_owner_cost_to_income_ratio"] >= 0.45:
        return "Severely Unaffordable"

    if row["estimated_owner_cost_to_income_ratio"] >= 0.35:
        return "Highly Unaffordable"

    if row["estimated_owner_cost_to_income_ratio"] >= 0.30:
        return "Moderately Unaffordable"

    return "Potentially Affordable"


ma_analysis_df["owner_affordability_class"] = ma_analysis_df.apply(
    classify_owner_affordability,
    axis=1,
)

display(
    ma_analysis_df[
        [
            "county_name",
            "estimated_owner_cost_to_income_ratio",
            "owner_affordability_gap",
            "owner_affordability_class",
        ]
    ].sort_values("estimated_owner_cost_to_income_ratio", ascending=False)
)

# Cell 25
mortgage_assumptions_df = pd.DataFrame(
    {
        "assumption": [
            "Down Payment Rate",
            "Annual Interest Rate",
            "Mortgage Term Years",
            "Annual Property Tax Rate",
            "Annual Home Insurance",
            "Monthly HOA",
            "Monthly PMI",
        ],
        "value": [
            DOWN_PAYMENT_RATE,
            ANNUAL_INTEREST_RATE,
            MORTGAGE_TERM_YEARS,
            ANNUAL_PROPERTY_TAX_RATE,
            ANNUAL_HOME_INSURANCE,
            MONTHLY_HOA,
            MONTHLY_PMI,
        ],
    }
)

display(mortgage_assumptions_df)

# ==============================================================================
# ## 9. Visual Analysis: Estimated Ownership Costs
# 
# These charts translate county-level median home values into estimated monthly ownership costs and compare those costs to local income capacity.

# Cell 27
plot_df = ma_analysis_df.sort_values("owner_affordability_gap")

plt.figure(figsize=(10, 7))
plt.barh(plot_df["county_name"], plot_df["owner_affordability_gap"])
plt.axvline(0, linestyle="--", label="Affordability Benchmark")
plt.title("Estimated Monthly Owner Cost Gap by Massachusetts County")
plt.xlabel("Estimated Owner Cost - 30% Monthly Income Benchmark")
plt.ylabel("County")
plt.legend()
plt.tight_layout()

plt.savefig(
    CHART_DIR / "estimated_owner_cost_gap_by_county.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

# Cell 28
plot_df = ma_analysis_df.sort_values("estimated_owner_cost_to_income_ratio")

plt.figure(figsize=(10, 7))
plt.barh(plot_df["county_name"], plot_df["estimated_owner_cost_to_income_ratio"])
plt.axvline(0.30, linestyle="--", label="30% Affordability Threshold")
plt.axvline(0.40, linestyle="--", label="40% High Pressure Reference")
plt.title("Estimated Monthly Owner Cost as a Share of Income by Massachusetts County")
plt.xlabel("Estimated Monthly Owner Cost / Monthly Median Income")
plt.ylabel("County")
plt.legend()
plt.tight_layout()

plt.savefig(
    CHART_DIR / "estimated_owner_cost_to_income_ratio_by_county.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

# ==============================================================================
# ## 10. Mortgage Rate Sensitivity Analysis
# 
# Because ownership affordability is highly sensitive to interest rates, this section estimates monthly owner costs under multiple mortgage-rate scenarios.

# Cell 30
interest_rate_scenarios = [0.055, 0.065, 0.075, 0.085]

scenario_results = []

for rate in interest_rate_scenarios:
    temp_df = ma_analysis_df.copy()

    temp_df["scenario_interest_rate"] = rate
    temp_df["scenario_monthly_owner_cost"] = temp_df["median_home_value"].apply(
        lambda home_value: calculate_estimated_monthly_owner_cost(
            home_value=home_value,
            annual_interest_rate=rate,
        )
    )
    temp_df["scenario_owner_cost_to_income_ratio"] = safe_divide(
        temp_df["scenario_monthly_owner_cost"],
        temp_df["monthly_median_income"],
    )
    temp_df["scenario_owner_affordability_gap"] = (
        temp_df["scenario_monthly_owner_cost"]
        - temp_df["affordable_monthly_housing_cost"]
    )

    scenario_results.append(
        temp_df[
            [
                "county_name",
                "scenario_interest_rate",
                "scenario_monthly_owner_cost",
                "scenario_owner_cost_to_income_ratio",
                "scenario_owner_affordability_gap",
            ]
        ]
    )

rate_sensitivity_df = pd.concat(scenario_results, ignore_index=True)

display(rate_sensitivity_df.head())

# Cell 31
rate_sensitivity_pivot = rate_sensitivity_df.pivot(
    index="county_name",
    columns="scenario_interest_rate",
    values="scenario_owner_affordability_gap",
).sort_values(0.065, ascending=False)

display(rate_sensitivity_pivot)

# Cell 32
plot_df = rate_sensitivity_df.copy()
plot_df["scenario_interest_rate_label"] = (
    plot_df["scenario_interest_rate"] * 100
).round(1).astype(str) + "%"

plt.figure(figsize=(10, 7))

for county_name, county_df in plot_df.groupby("county_name"):
    county_df = county_df.sort_values("scenario_interest_rate")
    plt.plot(
        county_df["scenario_interest_rate"] * 100,
        county_df["scenario_owner_affordability_gap"],
        marker="o",
        linewidth=1,
        alpha=0.8,
    )

plt.axhline(0, linestyle="--", label="Affordability Benchmark")
plt.title("Owner Affordability Gap by Mortgage Rate Scenario")
plt.xlabel("Mortgage Interest Rate (%)")
plt.ylabel("Estimated Owner Cost Gap")
plt.tight_layout()

plt.savefig(
    CHART_DIR / "owner_affordability_gap_by_rate_scenario.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

# ==============================================================================
# ## 11. Summary Statistics and Top Pressure Counties
# 
# This section creates compact summaries of statewide county-level affordability patterns and identifies the highest-pressure counties by selected metrics.

# Cell 34
statewide_summary = {
    "Median County Home Value-to-Income Ratio": ma_analysis_df["home_value_to_income_ratio"].median(),
    "Median County Rent-to-Income Ratio": ma_analysis_df["rent_to_income_ratio"].median(),
    "Median County Renter Cost-Burden Rate": ma_analysis_df["renter_cost_burden_rate"].median(),
    "Median County Severe Rent-Burden Rate": ma_analysis_df["severely_rent_burdened_rate"].median(),
    "Median County Estimated Owner Cost-to-Income Ratio": ma_analysis_df["estimated_owner_cost_to_income_ratio"].median(),
    "Median County Owner Affordability Gap": ma_analysis_df["owner_affordability_gap"].median(),
}

statewide_summary_df = pd.DataFrame(
    statewide_summary.items(),
    columns=["metric", "value"],
)

display(statewide_summary_df)

# Cell 35
least_affordable_owner_counties = ma_analysis_df[
    [
        "county_name",
        "median_household_income",
        "median_home_value",
        "estimated_monthly_owner_cost",
        "affordable_monthly_housing_cost",
        "owner_affordability_gap",
        "estimated_owner_cost_to_income_ratio",
        "owner_affordability_class",
    ]
].sort_values("owner_affordability_gap", ascending=False)

least_affordable_renter_counties = ma_analysis_df[
    [
        "county_name",
        "median_household_income",
        "median_gross_rent",
        "rent_to_income_ratio",
        "rent_affordability_gap",
        "renter_cost_burden_rate",
        "severely_rent_burdened_rate",
    ]
].sort_values("renter_cost_burden_rate", ascending=False)

print("Least affordable owner counties:")
display(least_affordable_owner_counties.head(10))

print("Least affordable renter counties:")
display(least_affordable_renter_counties.head(10))

# Cell 36
highest_home_value_pressure = ma_analysis_df.loc[
    ma_analysis_df["home_value_to_income_ratio"].idxmax()
]

highest_rent_pressure = ma_analysis_df.loc[
    ma_analysis_df["rent_to_income_ratio"].idxmax()
]

highest_renter_burden = ma_analysis_df.loc[
    ma_analysis_df["renter_cost_burden_rate"].idxmax()
]

highest_owner_gap = ma_analysis_df.loc[
    ma_analysis_df["owner_affordability_gap"].idxmax()
]

print("Highest home value-to-income pressure:")
display(highest_home_value_pressure)

print("Highest rent-to-income pressure:")
display(highest_rent_pressure)

print("Highest renter cost-burden rate:")
display(highest_renter_burden)

print("Highest owner affordability gap:")
display(highest_owner_gap)

# ==============================================================================
# ## 12. Initial Findings
# 
# This county-level ACS analysis suggests that housing affordability pressure in Massachusetts varies meaningfully by geography.
# 
# The highest owner affordability pressure appears in counties where median home values are largest relative to median household income. This indicates that even when incomes are relatively high, home prices may still exceed what local households can reasonably afford under traditional affordability benchmarks.
# 
# Renter affordability pressure is captured through both median rent-to-income ratios and renter cost-burden rates. Counties with high renter cost-burden rates indicate that a large share of renter households are paying at least 30% of income toward rent.
# 
# After translating median county home values into estimated monthly ownership costs, the affordability gap becomes more directly interpretable. Rather than only comparing home values to annual income, this analysis estimates the monthly cost of purchasing the median-valued home under a standard mortgage scenario.
# 
# Counties with the largest positive owner affordability gaps are areas where the estimated monthly owner cost exceeds the 30% gross-income benchmark by the widest margin. This indicates that median homeownership may be meaningfully out of reach for median-income households in those counties under the selected assumptions.
# 
# The mortgage-rate sensitivity analysis shows how quickly owner affordability can change when interest rates move. This is especially important because the same median home value can create very different monthly affordability outcomes under different mortgage-rate environments.

# ==============================================================================
# ## 13. Assumptions and Limitations
# 
# This analysis uses ACS 5-Year county-level estimates, which are useful for geographic comparison but may smooth over meaningful neighborhood-level variation within counties.
# 
# The ownership-cost analysis is based on a standardized mortgage scenario and does not represent every buyer situation. Results are sensitive to assumptions around down payment percentage, interest rate, property tax rate, insurance, HOA fees, and PMI.
# 
# Median household income and median home value are compared at the county level, but the median-income household is not necessarily the household purchasing the median-valued home. The analysis should therefore be interpreted as an affordability pressure indicator rather than a precise underwriting model.
# 
# Future versions could incorporate HUD Fair Market Rents, CHAS cost-burden data, municipal-level property tax rates, MSA-level comparisons, and tract-level geographic analysis.

# ==============================================================================
# ## 14. Export Processed Outputs
# 
# Save the cleaned county-level affordability table, mortgage assumptions, summary statistics, owner/renter ranking tables, and mortgage-rate sensitivity table locally.

# Cell 40
ma_analysis_output_path = DATA_DIR / "ma_county_housing_affordability_analysis.csv"
mortgage_assumptions_output_path = DATA_DIR / "mortgage_assumptions.csv"
rate_sensitivity_output_path = DATA_DIR / "ma_county_mortgage_rate_sensitivity.csv"
statewide_summary_output_path = DATA_DIR / "ma_county_affordability_summary.csv"
owner_ranking_output_path = DATA_DIR / "least_affordable_owner_counties.csv"
renter_ranking_output_path = DATA_DIR / "least_affordable_renter_counties.csv"

ma_analysis_df.to_csv(ma_analysis_output_path, index=False)
mortgage_assumptions_df.to_csv(mortgage_assumptions_output_path, index=False)
rate_sensitivity_df.to_csv(rate_sensitivity_output_path, index=False)
statewide_summary_df.to_csv(statewide_summary_output_path, index=False)
least_affordable_owner_counties.to_csv(owner_ranking_output_path, index=False)
least_affordable_renter_counties.to_csv(renter_ranking_output_path, index=False)

print("Saved outputs:")
print(f"- {ma_analysis_output_path}")
print(f"- {mortgage_assumptions_output_path}")
print(f"- {rate_sensitivity_output_path}")
print(f"- {statewide_summary_output_path}")
print(f"- {owner_ranking_output_path}")
print(f"- {renter_ranking_output_path}")
