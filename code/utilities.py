import csv
import ast
import random
import sys

import latexify
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from scipy.stats import linregress

# set up LaTeX-style plotting
latexify.latexify()
csv.field_size_limit(2 ** 30)


def occupation_func_female_logitprop(row):
    occupation_func_female_logitprop.label = "Women Occupation Logit Prop"
    occupation_func_female_logitprop.savelabel = "WomenOccupationLogProp"

    p = float(row["Female"])
    if p < 1e-5 or p > 1 - 1e-5:
        return None

    return np.log(p / (1 - p))


def occupation_func_female_percent(row):
    occupation_func_female_percent.label = "Women Occupation $\\%$ Difference"
    occupation_func_female_percent.savelabel = "WomenOccupRelativePer"

    p = float(row["Female"])

    # percent minority - percent majority
    return (2 * p - 1) * 100


bad_occupations = ["smith", "conductor"]


def occupation_func_whitehispanic_logitprop(row):
    occupation_func_whitehispanic_logitprop.label = "Hispanic Occupation Logit Prop"
    occupation_func_whitehispanic_logitprop.savelabel = "HispanicOccupationLogProp"

    if row["Occupation"] in bad_occupations:
        return None

    p = float(row["hispanic"]) / (
        float(row["hispanic"]) + float(row["white"]) + 1e-5
    )
    if p < 1e-4 or p > 1 - 1e-4:
        return None

    p = np.log(p / (1 - p))
    if p > 5:
        return None
    return p


def occupation_func_whitehispanic_percent(row):
    occupation_func_whitehispanic_percent.label = (
        "Hispanic Occupation $\\%$ Difference"
    )
    occupation_func_whitehispanic_percent.savelabel = "HispanicOccupRelativePer"

    if row["Occupation"] in bad_occupations:
        return None

    p = float(row["hispanic"]) / (
        float(row["hispanic"]) + float(row["white"]) + 1e-5
    )
    # percent minority - percent majority
    return (2 * p - 1) * 100


def load_mturkstereotype_data(filename):
    dd = {}
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # center at 0
            dd[row["occupation"]] = float(row["stereotype_score"]) - 2
    return dd


def occupation_func_whiteasian_logitprop(row):
    occupation_func_whiteasian_logitprop.label = "Asian Occupation Logit Prop"
    occupation_func_whiteasian_logitprop.savelabel = "AsianOccupationProportion"

    if row["Occupation"] in bad_occupations:
        return None

    p = float(row["asian"]) / (float(row["asian"]) + float(row["white"]) + 1e-5)
    if p < 1e-4 or p > 1 - 1e-4:
        return None

    p = np.log(p / (1 - p))
    if p > 5:
        return None
    return p


def occupation_func_whiteasian_percent(row):
    occupation_func_whiteasian_percent.label = "Asian Occupation $\\%$ Difference"
    occupation_func_whiteasian_percent.savelabel = "AsianOccupRelativeProp"

    if row["Occupation"] in bad_occupations:
        return None

    p = float(row["asian"]) / (float(row["asian"]) + float(row["white"]) + 1e-5)
    return (2 * p - 1) * 100


def load_williamsbestadjectives(filename, otherfunc, yrs_to_do=None):

    d = {}
    with open(filename, "r") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            row["word"] = row["word"].strip().replace("p.n", "")
            word = row["word"].strip()
            curr = d.get(word, {})
            curr[float(row["year"].strip())] = otherfunc(row)
            d[word] = curr

    ret = {}
    ret_weights = {}
    for occ in d:
        ret[occ] = [d[occ].get(x, np.nan) for x in yrs_to_do]
        ret_weights[occ] = [1 for _ in yrs_to_do]

    return ret, ret_weights


def occupation_func_williamsbestadject(row):
    occupation_func_williamsbestadject.label = "Human Stereotype Score"
    occupation_func_williamsbestadject.savelabel = "HSS"

    return float(row["transformed_score"].strip())




def load_occupationpercent_data(
    filename, occupation_func, yrs_to_do=None
):
    if yrs_to_do is None:
        yrs_to_do = list(range(1950, 2000, 10))
# original:
# def load_occupationpercent_data(filename, occupation_func, yrs_to_do=list(range(1950, 2000, 10))):

    # load as dictionary: occupation -> occupation_func(group_type : array over time)
    d = {}
    d_weights = {}
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            occ = row["Occupation"]
            curr = d.get(occ, {})
            occp = occupation_func(row)
            if occp is None:
                continue
            curr[int(row["Census year"])] = occp
            d[occ] = curr

            curr_w = d_weights.get(occ, {})
            total_weight = row.get("Total Weight", "").strip()
            if len(total_weight) == 0:
                curr_w[int(row["Census year"])] = 1
            else:
                curr_w[int(row["Census year"])] = float(total_weight)
            d_weights[occ] = curr_w

    ret = {}
    ret_weights = {}
    for occ in d:
        ret[occ] = [d[occ].get(x, np.nan) for x in yrs_to_do]
        ret_weights[occ] = [d_weights[occ].get(x, np.nan) for x in yrs_to_do]

    return ret, ret_weights


def load_files(filenames):
    rows = {}
    for f in filenames:
        r = load_file(f)
        print(r.keys())
        for d in r:
            rows[d] = r[d]
    return rows


def load_file(filename):
    rows = {}
    with open(filename, "r") as f:
        reader = list(csv.reader(f))
        # replace literal "nan" strings so eval can see np.nan
        for en in range(len(reader)):
            reader[en] = [s.replace("nan", "np.nan") for s in reader[en]]
        for en in range(0, len(reader), 2):
            try:
                header = reader[en]
                values = reader[en + 1]
                # key is label in second column
                key = values[1]
                rows[key] = {
                    header[i]: eval(values[i]) for i in range(2, len(header))
                }
            except Exception as e:
                print(e)
                continue
    return rows


def differences(vec1, vec2):
    return np.subtract(vec1, vec2)


nytyears = list(range(1987, 2005))
sgnyears = list(range(1910, 2000, 10))
svdyears = list(range(1910, 2000, 10))
cohayears = list(range(1880, 2000, 10))


def get_years(label):
    if "svd" in label:
        yrs = svdyears
    elif "sgns" in label:
        yrs = sgnyears
    elif "wikipedia" in label:
        yrs = [2015]
    elif "google" in label or "commoncrawlglove" in label:
        yrs = [2015]
    elif "nyt" in label:
        yrs = nytyears
    else:
        print(f"don't have years for label: {label}")
        return None
    return yrs

# def get_years_single(label):
#     if 'svd' in label:
#         yrs = get_years(label)
#     elif 'sgns' in label:
#         yrs = get_years(label)
#     elif 'wikipedia' in label:
#         yrs = [2015]
#     elif 'google' in label or 'commoncrawlglove' in label:
#         yrs = [2015]
#     else:
#         print('dont have years: ' + str(label))
#         return None
#     return yrs
