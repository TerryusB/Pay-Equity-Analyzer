# ⚖️ Pay Equity Analyzer

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-OLS_Regression-orange?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## 📋 Overview

**Pay Equity Analyzer** is a People Analytics tool designed to audit corporate payroll data with scientific rigor. Bridging the gap between **Executive Search methodologies** and **Data Science**, this project allows HR professionals to move from intuition to evidence-based decision-making.

[cite_start]It automates the diagnosis of salary structures, identifies retention risks due to underpayment, and statistically tests for **Gender Pay Gaps** using OLS (Ordinary Least Squares) regression models to meet **ESG standards**[cite: 29, 30].

## ✨ Key Features

* **🔄 Automated Data Standardization:**
    [cite_start]Cleans raw payroll data and maps diverse job titles to a standardized Level Architecture (N1 to N8), automatically assigning market salary midpoints[cite: 15, 16].

* **🚦 Risk Analysis Engine:**
    [cite_start]Calculates **Compa-Ratios** for every employee and classifies them into a "Traffic Light" risk system (e.g., *Critical Underpayment*, *Market Aligned*, *Overpaid*) to detect flight risks[cite: 20, 21].

* **Correction & Cost Modeling:**
    Suggests immediate salary adjustments to bring underpaid employees to the minimum market range and calculates the total budget impact.

* **📉 Statistical Auditing (OLS Regression):**
    Uses `statsmodels` to run a multivariate regression. [cite_start]It isolates variables to determine if a specific Gender Pay Gap exists **after controlling for job level and performance**, validating results with P-values[cite: 23, 24, 25].

## 🛠️ Tech Stack

* **Language:** Python 3
* **Core Libraries:**
    * `pandas`: For data manipulation and structure mapping.
    * `numpy`: For numerical calculations and array handling.
    * `statsmodels`: For statistical modeling (OLS Regression).
    * `tabulate`: For formatting console output tables.
    * `pytest`: For unit testing business logic.

## 📂 Project Structure

```text
📦 Pay-Equity-Analyzer
 ┣ 📜 project.py           # Main script with business logic and execution flow
 ┣ 📜 test_project.py      # Unit tests (Pytest)
 ┣ 📜 requirements.txt     # Dependency list
 ┣ 📜 data.csv             # Synthetic dataset (300 employees) for demo purposes
 ┗ 📜 README.md            # Project documentation
