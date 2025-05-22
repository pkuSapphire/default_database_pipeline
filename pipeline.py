# coding:utf-8

import subprocess
import sys
import os
import io
import pickle
import requests
import datetime as dt
import pandas as pd
import numpy as np
import statsmodels.api as sm

DATA_DIR = "data_pickle"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

#  We try to import WRDS; if not found, we install it
try:
    import wrds
except ImportError:
    print("Installing wrds package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wrds", "--quiet"])
    import wrds

#  We'll also check scikit-learn for AUC. If not found, we install it
try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    print("Installing scikit-learn package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "--quiet"])
    from sklearn.metrics import roc_auc_score

# ========================================================================
# --------------------- Global Variables and URLs ------------------------
# ========================================================================

WRDS_USERNAME = "bosen"
WRDS_PASSWORD = "Y@QC8f4qEjCz!eY"
WRDS_CONN = None

MAJOR_GROUPS_URL = "https://raw.githubusercontent.com/saintsjd/sic4-list/master/major-groups.csv"
DIVISIONS_URL = "https://raw.githubusercontent.com/saintsjd/sic4-list/master/divisions.csv"

#  These are the features we will eventually compute and examine
global target_vars
target_vars = [
    "ACTLCT",
    "APSALE",
    "CASHTA",
    "CHAT",
    "CHLCT",
    "EBITAT",
    "EBITSALE",
    "FAT",
    "FFOLT",
    "INVTSALES",
    "LCTLT",
    "LOGAT",
    "LOGSALE",
    "NIAT",
    "NIMTA",
    "NISALE",
    "TDEBT",
    "MKVAL",
    "LBTAT",
    "DBTAT",
    "DBTMKTEQ",
    "LBTMKTEQ",
]

# ========================================================================
# --------------------------- WRDS Connection ----------------------------
# ========================================================================


def get_wrds_conn(username=WRDS_USERNAME, password=WRDS_PASSWORD):

    global WRDS_CONN
    if WRDS_CONN is None:
        WRDS_CONN = wrds.Connection(wrds_username=username, wrds_password=password)
    return WRDS_CONN


def close_wrds_conn():
    """

    Safely close the global WRDS connection if it is open.
    """
    global WRDS_CONN
    if WRDS_CONN is not None:
        WRDS_CONN.close()
        WRDS_CONN = None


# ========================================================================
# ------------------- Part 1: Building the Base Dataset ------------------
# ========================================================================


def get_gvkey(filename=os.path.join(DATA_DIR, "gvkey_data.pkl")):
    """

    Fetch gvkey data from WRDS or load a local pickle file if it already exists.
    """
    if os.path.exists(filename):
        print(f"Loading data from {filename}...")
        with open(filename, "rb") as f:
            gvkey_data = pickle.load(f)
    else:
        conn = get_wrds_conn()
        query = "SELECT * FROM ciq.wrds_gvkey"
        gvkey_data = conn.raw_sql(query)
        with open(filename, "wb") as f:
            pickle.dump(gvkey_data, f)
        print(f"Data saved to {filename}")
    return gvkey_data


def get_ratings(filename=os.path.join(DATA_DIR, "ratings_data.pkl")):
    """

    Retrieve credit ratings from WRDS or load local pickle if available.
    Only keep long-term local currency ratings from 1990 onwards.
    """
    if os.path.exists(filename):
        print(f"Loading data from {filename}...")
        with open(filename, "rb") as f:
            ratings_data = pickle.load(f)
    else:
        conn = get_wrds_conn()
        query = """
        SELECT company_id as companyid, entity_pname, ratingdate, ratingsymbol, ratingactionword, unsol
        FROM ciq_ratings.wrds_erating
        WHERE longtermflag = 1 
          AND ratingtypename = 'Local Currency LT' 
          AND ratingdate >= '1990-01-01'
        """
        ratings_data = conn.raw_sql(query)
        symbols = [
            "AAA",
            "AA+",
            "AA",
            "AA-",
            "A+",
            "A",
            "A-",
            "BBB+",
            "BBB",
            "BBB-",
            "BB+",
            "BB",
            "BB-",
            "B+",
            "B",
            "B-",
            "CCC+",
            "CCC",
            "CCC-",
            "CC",
            "C",
            "D",
            "SD",
            "NR",
            "R",
        ]
        ratings_data = ratings_data[ratings_data.ratingsymbol.isin(symbols)]
        with open(filename, "wb") as f:
            pickle.dump(ratings_data, f)
        print(f"Data saved to {filename}")
    return ratings_data


def merge_ratings_with_gvkey(gvkey_df, ratings_df):
    """

    Merge gvkey and ratings on companyid, handle duplicates, and create a 'ratingenddate' field.
    """
    merged_df = pd.merge(
        gvkey_df[["gvkey", "companyid", "startdate", "enddate"]], ratings_df, on="companyid"
    )
    merged_df = merged_df.drop_duplicates(subset=["gvkey", "ratingdate"])
    merged_df = merged_df.sort_values(
        ["gvkey", "companyid", "ratingdate"], ascending=[True, True, False]
    )
    merged_df["ratingenddate"] = merged_df.ratingdate.shift()
    merged_df["gvkey_shift"] = merged_df.gvkey.shift()
    merged_df.loc[merged_df.gvkey != merged_df.gvkey_shift, "ratingenddate"] = str(
        dt.date(2100, 12, 31)
    )
    merged_df.drop(columns=["gvkey_shift"], inplace=True)
    return merged_df


def get_sector(ratings_merged, filename=os.path.join(DATA_DIR, "sector_data.pkl")):
    """

    Pull sector info from comp.company or load from a local file.
    Then keep only those with gvkey in 'ratings_merged'.
    """
    if os.path.exists(filename):
        print(f"Loading data from {filename}...")
        with open(filename, "rb") as f:
            info_data = pickle.load(f)
    else:
        conn = get_wrds_conn()
        sql_info = """
        SELECT
            gvkey,
            conm,
            fic,
            gsector,
            ggroup,
            gind,
            idbflag,
            incorp,
            loc,
            naics,
            sic,
            state
        FROM comp.company
        """
        info_data = conn.raw_sql(sql_info)
        info_data = info_data[info_data.gvkey.isin(ratings_merged.gvkey)]
        info_data = info_data.drop_duplicates(subset=["gvkey"])
        with open(filename, "wb") as f:
            pickle.dump(info_data, f)
        print(f"Data saved to {filename}")
    return info_data


def get_or_download_csv(pkl_filename, csv_url):
    """

    Download a CSV from GitHub and store it as a pickle if it doesn't already exist locally.
    """
    if os.path.exists(pkl_filename):
        print(f"Loading data from {pkl_filename}...")
        df = pd.read_pickle(pkl_filename)
    else:
        print(f"{pkl_filename} not found, downloading from GitHub...")
        response = requests.get(csv_url)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            df.to_pickle(pkl_filename)
            print(f"Data saved to {pkl_filename}")
        else:
            raise Exception(f"Download failed with status code: {response.status_code}")
    return df


def get_sector_info(ratings_merged):
    """

    Merge sector data (SIC-based) and GICS info, then create an overall 'sector' column.
    """
    info = get_sector(ratings_merged)
    major_groups = get_or_download_csv(os.path.join(DATA_DIR, "major_groups.pkl"), MAJOR_GROUPS_URL)
    divisions = get_or_download_csv(os.path.join(DATA_DIR, "divisions.pkl"), DIVISIONS_URL)

    info["Major Group"] = info["sic"].astype(str).str[:2].str.zfill(2)
    major_groups["Major Group"] = major_groups["Major Group"].astype(str).str.zfill(2)

    info_with_div = info.merge(
        major_groups[["Major Group", "Division"]], on="Major Group", how="left"
    ).merge(divisions[["Division", "Description"]], on="Division", how="left")

    info_with_div.rename(columns={"Description": "SIC Division Name"}, inplace=True)
    info_with_div.drop(columns=["Major Group", "Division"], inplace=True)

    gics_data = {
        "gsector": ["10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60"],
        "GIC Sector Name": [
            "Energy",
            "Materials",
            "Industrials",
            "Consumer Discretionary",
            "Consumer Staples",
            "Health Care",
            "Financials",
            "Information Technology",
            "Communication Services",
            "Utilities",
            "Real Estate",
        ],
    }
    gics_df = pd.DataFrame(gics_data)

    info_with_gic = info_with_div.merge(gics_df, on="gsector", how="left")
    info_with_gic["sector"] = info_with_gic["GIC Sector Name"]
    info_with_gic.loc[info_with_gic["sector"].isnull(), "sector"] = info_with_gic[
        "SIC Division Name"
    ]
    info_1 = info_with_gic.drop(columns=["fic", "gind", "idbflag", "incorp", "state"])

    # Adjust sector manually for known edge cases
    info_1.loc[info_1["conm"] == "ARGO GROUP INTL 6.5 SR NT 42", "sector"] = "Insurance"
    info_1.loc[info_1["conm"] == "HILFIGER (TOMMY) U S A INC", "sector"] = "Manufacturing"
    info_1.loc[info_1["conm"] == "NOVA SCOTIA POWER INC", "sector"] = "Utilities"

    # For consumer discretionary/staples, revert to original SIC division name
    def replace_sector(row):
        if row["sector"] in ["Consumer Discretionary", "Consumer Staples"]:
            return row["SIC Division Name"]
        else:
            return row["sector"]

    info_2 = info_1.copy()
    info_2["sector"] = info_2.apply(replace_sector, axis=1)
    info_2["ggroup"] = info_2["ggroup"].fillna("")

    # Map sectors
    sector_mapping = {
        "Industrials": "Manufacturing",
        "Health Care": "Health",
        "Energy": "Utilities",
        "Information Technology": "Information Technology",
        "Wholesale Trade": "Wholesale",
        "Utilities": "Utilities",
        "Financials": "Financials",
        "Materials": "Manufacturing",
        "Transportation, Communications, Electric, Gas, And Sanitary Services": "Transportation, Communications, Electric, Gas, And Sanitary Services",
        "Communication Services": "Services",
        "Retail Trade": "Retail",
        "Manufacturing": "Manufacturing",
        "Construction": "Construction",
        "Finance, Insurance, And Real Estate": "Financials",
        "Services": "Services",
        "Agriculture, Forestry, And Fishing": "Agriculture",
        "Public Administration": "Utilities",
        "Real Estate": "Financials",
        "Mining": "Manufacturing",
        "Insurance": "Financials",
    }
    info_3 = info_2.copy()
    info_3["sector"] = info_3["sector"].map(sector_mapping)

    # Further refine transportation vs. utilities vs. services
    def map_sic_to_sector(row):
        if row["sector"] == "Transportation, Communications, Electric, Gas, And Sanitary Services":
            sic_major_group = int(str(row["sic"])[:2])
            if 40 <= sic_major_group < 48:
                return "Transportation"
            elif sic_major_group == 48:
                return "Services"
            else:
                return "Utilities"
        else:
            return row["sector"]

    def map_gic_transportation(row):
        if row["ggroup"] == "2030":
            return "Transportation"
        else:
            return row["sector"]

    info_3["sector"] = info_3.apply(map_sic_to_sector, axis=1)
    info_3["sector"] = info_3.apply(map_gic_transportation, axis=1)

    return info_3


def prepare_ratings(info_3, ratings_merged):
    """

    Merge sector data into ratings, and flag defaults (D, SD, R).
    """
    merged_data = pd.merge(ratings_merged, info_3[["gvkey", "sector"]], on="gvkey", how="left")
    defaults_all = merged_data[merged_data.ratingsymbol.isin(["D", "SD", "R"])].copy()
    defaults_all2 = defaults_all[["gvkey", "ratingdate"]].drop_duplicates("gvkey")
    defaults_all2["default_flag"] = 1

    # Merge default flag
    ratings_prepared = pd.merge(merged_data, defaults_all2, on=["gvkey", "ratingdate"], how="left")
    ratings_prepared.loc[pd.isnull(ratings_prepared.default_flag), "default_flag"] = 0
    return ratings_prepared


def get_financials(filename=os.path.join(DATA_DIR, "financials_data.pkl")):
    """

    Pull fundamental annual data from comp.funda or load local pickle if available.
    """
    if os.path.exists(filename):
        print(f"Loading data from {filename}...")
        with open(filename, "rb") as f:
            fin_data = pickle.load(f)
    else:
        conn = get_wrds_conn()
        sql_financials = """
        SELECT
            gvkey,
            datadate,
            fyear,
            fyr,
            at,
            lt,
            ceq,
            act,
            lct,
            invt,
            rect,
            ap,
            dlc,
            dltt,
            dltis,
            dvt,
            che,
            xint,
            xrd,
            xsga,
            oibdp,
            ebit,
            sale,
            cogs,
            ni,
            oancf,
            fincf,
            csho,
            prcc_f,
            'Annual' AS freq
        FROM comp.funda
        WHERE
            indfmt = 'INDL'
            AND datafmt = 'STD'
            AND popsrc = 'D'
            AND consol = 'C'
            AND fyear >= 1990
        """
        fin_data = conn.raw_sql(sql_financials)
        with open(filename, "wb") as f:
            pickle.dump(fin_data, f)
        print(f"Data saved to {filename}")
    return fin_data


def prepare_financials():
    """

    Sort the financials by date descending for each gvkey.
    """
    fin = get_financials()
    fin = fin.sort_values(by=["gvkey", "datadate"], ascending=[True, False])
    return fin


def override_by_exact_fyear(mfinancials_df, ratings_df):
    """

    If a default occurs in a specific fiscal year, override the rating/date with that default info.
    """
    defaults = ratings_df[ratings_df["ratingsymbol"].isin(["D", "SD", "R"])].copy()
    defaults["ratingdate"] = pd.to_datetime(defaults["ratingdate"], errors="coerce")
    defaults["fyear"] = defaults["ratingdate"].dt.year

    overrides = (
        defaults.groupby(["gvkey", "fyear"])
            .agg(
            ratingsymbol_override=(
                "ratingsymbol",
                lambda x: x[x.isin(["D", "SD", "R"])].iloc[0] if x.isin(["D", "SD", "R"]).any() else x.iloc[-1]
            ),
            ratingdate_override=("ratingdate", "min")
        )
            .reset_index()
    )

    merged = pd.merge(mfinancials_df, overrides, on=["gvkey", "fyear"], how="left")

    merged["ratingsymbol"] = merged["ratingsymbol_override"].combine_first(merged["ratingsymbol"])
    merged["ratingdate"] = merged["ratingdate_override"].combine_first(merged["ratingdate"])
    merged.drop(columns=["ratingsymbol_override", "ratingdate_override"], inplace=True)

    return merged


def merge_financials_ratings(financials_df, ratings_df):
    """

    Merge financials with rating intervals (ratingdate to ratingenddate),
    such that datadate is within [ratingdate, ratingenddate].
    """
    common_gvkeys = set(financials_df["gvkey"]).intersection(set(ratings_df["gvkey"]))
    fin_sub = financials_df[financials_df["gvkey"].isin(common_gvkeys)].copy()
    rat_sub = ratings_df[ratings_df["gvkey"].isin(common_gvkeys)].copy()

    fin_sub["datadate"] = pd.to_datetime(fin_sub["datadate"], errors="coerce")
    rat_sub["ratingdate"] = pd.to_datetime(rat_sub["ratingdate"], errors="coerce")
    rat_sub["ratingenddate"] = pd.to_datetime(rat_sub["ratingenddate"], errors="coerce")
    rat_sub.loc[rat_sub["ratingenddate"] > "2100-12-31", "ratingenddate"] = pd.Timestamp(
        "2100-12-31"
    )

    merged_df = fin_sub.merge(rat_sub, how="left", on="gvkey")
    merged_df["datadate"] = pd.to_datetime(merged_df["datadate"], errors="coerce")
    merged_df["ratingdate"] = pd.to_datetime(merged_df["ratingdate"], errors="coerce")
    merged_df["ratingenddate"] = pd.to_datetime(merged_df["ratingenddate"], errors="coerce")

    # We keep only rows where datadate falls between ratingdate and ratingenddate
    merged_df = merged_df[
        (merged_df["ratingdate"].notna())
        & (merged_df["ratingenddate"].notna())
        & (merged_df["ratingdate"] <= merged_df["datadate"])
        & (merged_df["datadate"] <= merged_df["ratingenddate"])
    ]

    merged_df = merged_df.sort_values(by=["gvkey", "datadate"], ascending=[True, True])
    merged_df.reset_index(drop=True, inplace=True)

    # Keep a relevant subset of columns from financials + rating columns
    columns_to_keep = list(fin_sub.columns) + [
        "entity_pname",
        "ratingdate",
        "ratingsymbol",
        "ratingactionword",
        "unsol",
        "ratingenddate",
        "sector",
    ]
    merged_fin_df = merged_df[columns_to_keep]
    merged_fin_df = override_by_exact_fyear(merged_fin_df, ratings_df)

    return merged_fin_df


def compute_default_dates(merged_fin_df):
    """

    Find the earliest default date for each gvkey. If no default, set 2100-12-31.
    """
    default_df = merged_fin_df.filter(["gvkey", "ratingsymbol", "ratingdate"]).copy()

    def find_default_date(row):
        if row["ratingsymbol"] in ["D", "SD", "R"]:
            return row["ratingdate"]
        else:
            return pd.NaT

    default_df["dflt_date"] = default_df.apply(find_default_date, axis=1)
    default_df["dflt_date"] = pd.to_datetime(default_df["dflt_date"])
    default_df = default_df.sort_values(by=["gvkey", "dflt_date"], ascending=[True, True])
    default_df = default_df.groupby("gvkey", as_index=False).first()  # earliest default
    default_df["dflt_date"] = default_df["dflt_date"].fillna(pd.Timestamp("2100-12-31"))
    default_df.drop(["ratingdate", "ratingsymbol"], axis=1, inplace=True)
    default_df["dflt_date"] = pd.to_datetime(default_df["dflt_date"])
    return default_df


def merge_default_dates(merged_fin_df, default_df):
    """

    Merge the earliest default date and compute days to default for each row.
    Then set a dflt_flag if days2dflt is within [90,455].
    """
    df1 = pd.merge(merged_fin_df, default_df, on=["gvkey"], how="left")
    df2 = df1.copy()
    df2["dflt_date"] = pd.to_datetime(df2["dflt_date"], errors="coerce")
    df2["datadate"] = pd.to_datetime(df2["datadate"], errors="coerce")

    df2["days2dflt"] = (df2["dflt_date"] - df2["datadate"]).dt.days
    df2["dflt_flag"] = 0
    df2.loc[(df2["days2dflt"] >= 90) & (df2["days2dflt"] <= 455), "dflt_flag"] = 1

    return df2


def clean_dataset(df):
    """

    Exclude financials sector and any records with days2dflt < 90 (to avoid future leakage).
    """
    df_clean = df.loc[df["sector"] != "Financials"]
    df_clean = df_clean.loc[df_clean["days2dflt"] >= 90]
    return df_clean


# ========================================================================
# ---------------- Part 2: Feature Engineering & AUC ---------------------
# ========================================================================


def impute_data(df):
    """

    Impute missing current assets (act) and liabilities (lct) with estimates.
    Fill xrd/invt with 0 where missing.
    """
    df["act_est"] = df["che"] + df["rect"] + df["invt"]
    df["lct_est"] = df["ap"] + df["dlc"]

    df["act"] = np.where(df["act"].isna(), df["act_est"], df["act"])
    df["lct"] = np.where(df["lct"].isna(), df["lct_est"], df["lct"])

    df.drop(columns=["act_est", "lct_est"], inplace=True)

    df["xrd"] = df["xrd"].fillna(0)
    df["invt"] = df["invt"].fillna(0)
    return df


def build_features(df):
    """

    Create various financial ratio features.
    """
    # Current Assets / Current Liabilities
    df["ACTLCT"] = df["act"] / df["lct"]
    # Accounts Payable / Sales
    df["APSALE"] = df["ap"] / df["sale"]
    # Cash / Total Assets
    df["CASHTA"] = df["che"] / df["at"]
    # Duplicate for consistency
    df["CHAT"] = df["che"] / df["at"]
    # Cash / Current Liabilities
    df["CHLCT"] = df["che"] / df["lct"]
    # EBIT / Total Assets
    df["EBITAT"] = df["ebit"] / df["at"]
    # EBIT / Sales
    df["EBITSALE"] = df["ebit"] / df["sale"]
    # (Short-term Debt + 0.5 × Long-term Debt) / Total Assets
    df["FAT"] = (df["dlc"] + 0.5 * df["dltt"]) / df["at"]
    # Operating Cash Flow / Total Liabilities
    df["FFOLT"] = df["oancf"] / df["lt"]
    # Inventory / Sales
    df["INVTSALES"] = df["invt"] / df["sale"]
    # Current Liabilities / Total Liabilities
    df["LCTLT"] = df["lct"] / df["lt"]
    # log(Total Assets)
    df["LOGAT"] = df["at"].apply(lambda x: None if x <= 0 else np.log(x))
    # log(Sales)
    df["LOGSALE"] = df["sale"].apply(lambda x: None if x <= 0 else np.log(x))
    # Net Income / Total Assets
    df["NIAT"] = df["ni"] / df["at"]
    # Net Income / (Market Cap + Total Liabilities)
    df["NIMTA"] = df["ni"] / (df["prcc_f"] * df["csho"] + df["lt"])
    # Net Income / Sales
    df["NISALE"] = df["ni"] / df["sale"]
    # Short-term + Long-term Debt
    df["TDEBT"] = df["dlc"] + df["dltt"]
    # Market Cap = Price * Shares
    df["MKVAL"] = df["prcc_f"] * df["csho"]
    # Total Liabilities / Total Assets
    df["LBTAT"] = df["lt"] / df["at"]
    # Total Debt / Total Assets
    df["DBTAT"] = df["TDEBT"] / df["at"]
    # Total Debt / (Total Debt + Market Cap or Equity)
    df["DBTMKTEQ"] = np.where(
        df["MKVAL"].notna(),
        df["TDEBT"] / (df["TDEBT"] + df["MKVAL"]),
        df["TDEBT"] / (df["TDEBT"] + df["ceq"]),
    )
    # Total Liabilities / (Total Liabilities + Market Cap or Equity)
    df["LBTMKTEQ"] = np.where(
        df["MKVAL"].notna(),
        df["lt"] / (df["lt"] + df["MKVAL"]),
        df["TDEBT"] / (df["TDEBT"] + df["ceq"]),
    )
    return df


def tobins_q_n_Altman_Z(df):
    """

    Compute Tobin's Q and Altman Z-Score, fill numeric NaNs with 0.
    """
    df["datadate"] = pd.to_datetime(df["datadate"])

    df["ME"] = df["prcc_f"] * df["csho"]  # Market Value of Equity
    df["PREF"] = df[["ceq"]].fillna(0)  # Using ceq as a proxy for pref
    df["BE"] = df["ceq"] - df["PREF"]  # Book Value of Equity

    # Tobin's Q
    df["Tobin_Q"] = (df["at"] + df["ME"] - df["BE"]) / df["at"]
    df.loc[df["at"] <= 0, "Tobin_Q"] = None

    # Altman Z-Score
    df["Altman_Z"] = (
        3.3 * (df["ebit"] / df["at"])
        + 0.99 * (df["sale"] / df["at"])
        + 0.6 * (df["ME"] / df["lt"])
        + 1.2 * (df["act"] / df["at"])
        + 1.4 * (df["ni"] / df["at"])
    )

    # Fill numeric NaNs with 0
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)


def calculate_auc(df):
    """

    For each variable in target_vars plus Tobin_Q and Altman_Z, compute absolute contribution (AC).
    AC = abs(AUC - 0.5) * 200 on a 0–100 scale.
    """
    ac_scores = {}
    these_vars = target_vars + ["Tobin_Q", "Altman_Z"]

    for var in these_vars:
        valid_rows = df[["dflt_flag", var]].dropna()
        if valid_rows["dflt_flag"].nunique() < 2:
            print(f"Skipping {var}: only one class in dflt_flag after dropna.")
            continue

        auc_val = roc_auc_score(valid_rows["dflt_flag"], valid_rows[var])
        ac = abs(auc_val - 0.5) * 200  # 0–100 scale
        ac_scores[var] = ac

    for var, ac in sorted(ac_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{ac:6.2f}  \t {var}")


def get_final_dataframe():
    """
    Load the base dataset pickle, then do cleaning, imputation, feature building, etc.
    Finally return the processed DataFrame.
    """
    import os
    import pickle
    import numpy as np
    import pandas as pd

    with open(os.path.join(DATA_DIR, "base_dataset.pkl"), "rb") as f:
        df = pickle.load(f)
    df = clean_dataset(df)
    df = impute_data(df)
    df = build_features(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=target_vars)
    tobins_q_n_Altman_Z(df)
    return df


def transformation(df):
    """
    Apply candidate transformations to each target, fit Logit models,
    select best by AIC, compute Akaike weights, and add '_trf' columns.
    Returns transformed DataFrame and a summary DataFrame.
    """
    data = df.copy()
    models = {
        "linear": ["linear"],
        "polynomial": ["linear", "square"],
        "exponential": ["exp"],
        "hyperbolic": ["inv", "inv2"],
        "poly+exp": ["linear", "square", "exp"],
        "poly+hyper": ["linear", "square", "inv", "inv2"],
        "kitchen_sink": ["linear", "square", "exp", "inv", "inv2"],
    }

    def safe_logit_fit(y, X):
        """Fit logistic regression safely; return model or None."""
        try:
            return sm.Logit(y, X).fit(disp=0)
        except:
            return None

    results = []
    for var in target_vars:
        x = data[var]
        y = data["dflt_flag"]

        # build transformation matrix
        df_forms = pd.DataFrame(
            {
                "linear": x,
                "square": x**2,
                "exp": np.exp(-x),
                "inv": 1 / (x + 1),
                "inv2": 1 / (x + 1) ** 2,
            }
        )

        fits, aics, lls = {}, {}, {}
        for name, cols in models.items():
            Xm = df_forms[cols].dropna()
            y_clean = y.loc[Xm.index]
            if len(Xm) < 2 or y_clean.nunique() < 2:
                fits[name] = None
                aics[name] = np.nan
                lls[name] = np.nan
                continue
            Xc = sm.add_constant(Xm, has_constant="add")
            m = safe_logit_fit(y_clean, Xc)
            fits[name] = m
            if m is not None:
                aics[name] = m.aic
                lls[name] = m.llf
            else:
                aics[name] = np.nan
                lls[name] = np.nan

        # select best model by AIC and compute weights
        valid = {k: v for k, v in aics.items() if not pd.isna(v)}
        if valid:
            best = min(valid, key=valid.get)
            aic_min = valid[best]
            denom = sum(np.exp(-0.5 * (v - aic_min)) for v in valid.values())
            weights = {
                m: (np.exp(-0.5 * (aics[m] - aic_min)) / denom) if m in valid else np.nan
                for m in models
            }
        else:
            best = None
            weights = {m: np.nan for m in models}

        # add transformed column
        if best:
            p = fits[best].params
            data[f"{var}_trf"] = p["const"] + df_forms[models[best]].dot(p[models[best]])
        else:
            data[f"{var}_trf"] = np.nan

        # record summary
        rec = {
            "Variable": var,
            "Selected Model": best,
            "Best_AIC": aics.get(best, np.nan),
            "Best_Weight": weights.get(best, np.nan),
        }
        for m in models:
            rec.update({f"LL_{m}": lls[m], f"AIC_{m}": aics[m], f"W_{m}": weights[m]})
        results.append(rec)

    summary_df = pd.DataFrame(results)
    return data, summary_df


def reduce_correlation(df, threshold=0.5, target_vars=None):
    all_factors = df[target_vars]
    a = all_factors.corr()
    listCol = a.columns.tolist()
    listRecord = []
    for i in listCol:
        for j in listCol:
            if i != j:
                if abs(a[i][j]) > 0.5:
                    listTemp = [i, j]
                    listTemp1 = [j, i]
                    if (listTemp not in listRecord) and (listTemp1 not in listRecord):
                        listRecord.append(listTemp)
    results = []

    for var in target_vars:
        # Drop NaNs only for the 'dflt_flag' and current variable
        temp_df = df[["dflt_flag", var]].dropna()

        if not temp_df.empty:
            auc = roc_auc_score(temp_df["dflt_flag"], temp_df[var])
            ar = 2 * auc - 1
            results.append({"Variable": var, "AUC": auc, "Accuracy Ratio": ar})

    auc_ar_df = pd.DataFrame(results)
    auc_ar_df = auc_ar_df.sort_values(by="AUC", ascending=False).reset_index(drop=True)


# ========================================================================
# ---------------------- Single Integrated Main --------------------------
# ========================================================================


def main():
    """

    1) Build the base dataset from WRDS (gvkey, ratings, sector, financials, etc.)
    2) Merge everything, compute default flags, save base_dataset.pkl
    3) Load and transform (clean, impute, build features, etc.)
    4) Compute and display AUC for each feature.
    """

    # ============ PART A: Build Base Dataset ============
    print("\n[1] Building base dataset...")

    # Step 1: Load or create gvkey and ratings
    gvkey_data = get_gvkey()
    ratings_data = get_ratings()
    ratings_merged = merge_ratings_with_gvkey(gvkey_data, ratings_data)

    # Step 2: Sector info, refine ratings
    info_3 = get_sector_info(ratings_merged)
    ratings_prepared = prepare_ratings(info_3, ratings_merged)

    # Step 3: Load financials, merge with ratings
    fin = prepare_financials()
    merged_fin = merge_financials_ratings(fin, ratings_prepared)
    close_wrds_conn()

    # Step 4: Compute earliest default date, merge default flags
    default_date_df = compute_default_dates(merged_fin)
    base_df = merge_default_dates(merged_fin, default_date_df)

    # Step 5: Save to local pickle
    base_df.to_pickle(os.path.join(DATA_DIR, "base_dataset.pkl"))
    print("Base dataset saved to base_dataset.pkl")

    print(sum(base_df["dflt_flag"]))

    # ============ PART B: Feature Engineering & AUC ============
    print("\n[2] Loading the base dataset and generating features...")
    # Load from pickle
    with open(os.path.join(DATA_DIR, "base_dataset.pkl"), "rb") as f:
        df = pickle.load(f)

    # Clean data to exclude financials sector and remove future info
    df = clean_dataset(df)
    # Impute missing values
    df = impute_data(df)
    # Build ratio features
    df = build_features(df)
    # Replace infinities, drop if feature is NaN
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=target_vars)
    # Compute Tobin's Q & Altman Z
    tobins_q_n_Altman_Z(df)
    # transform
    df, transformation_summary = transformation(df)
    target_vars2 = [col + "_trf" for col in target_vars]
    # compute AUC for each feature
    # calculate_auc(df)




#  Run if called directly
if __name__ == "__main__":
    main()
