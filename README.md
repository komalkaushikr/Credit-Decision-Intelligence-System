# Credit Decision Intelligence System

### Profit-Optimized Loan Approval Engine

## The Business Question

How should a fintech lender approve loans to **maximize expected portfolio profit** while controlling default risk?

Traditional credit systems use fixed probability cutoffs (often 0.5 or 0.2).
This project challenges that assumption.

Instead of optimizing accuracy, we optimize **capital efficiency**.

---

## Executive Summary

This project builds a **profit-driven credit approval system** that converts predicted Probability of Default (PD) into portfolio-level capital allocation decisions.

The system:

* Models default risk using supervised learning
* Calculates expected profit per loan
* Identifies the **profit-maximizing approval threshold**
* Simulates portfolio performance under different economic scenarios

Core finding:

> A fixed risk threshold leaves money on the table.
> Profit optimization changes which loans should be approved.

---

## Dataset

LendingClub Accepted Loans dataset
(~20% historical default rate)

The dataset includes borrower characteristics such as:

* Interest rate
* Credit grade
* Employment length
* Revolving utilization
* Loan amount
* Credit history length

---

## Methodology

### 1. Data Engineering

* Cleaned categorical financial fields
* Converted credit history into credit age
* Ordinal encoded risk grades (A → G)
* Constructed financial ratios relevant to default behavior

### 2. Risk Modeling

* Built a Probability of Default (PD) model using Scikit-learn
* Evaluated model calibration and predictive power
* Generated PD scores for all applicants

### 3. Profit-Based Decision Logic

For each loan:

Expected Profit =
(Interest Income × (1 − PD)) − (Loss Given Default × PD)

Approval decision is made based on maximizing total portfolio profit.

This replaces arbitrary 0.5 cutoffs with economically rational thresholds.

---

## Key Results

Historical default rate: ~20%

Optimal fixed PD threshold: **0.497**

Maximum portfolio profit: **$125.51M**

### Performance Comparison

| Strategy                | Approval Rate | Portfolio Profit |
| ----------------------- | ------------- | ---------------- |
| Conservative Rule-Based | 55%           | ~$77M            |
| Profit-Optimized System | 94.8%         | ~$126M           |

The optimized system generated **$49M more profit**.

---

## Core Insight

A 25% interest loan can remain profitable even with a 30% probability of default.

A fixed PD threshold cannot capture this nuance.

Risk and return must be evaluated together.

---

## Why Threshold ≠ 0.5

0.5 assumes symmetric cost of error.

In lending, costs are asymmetric:

* Rejecting a good borrower loses interest income
* Approving a bad borrower incurs principal loss

Profit-aware thresholds reflect this imbalance.

---

## Why Optimize Profit Instead of Accuracy?

Accuracy treats all mistakes equally.

Banks do not.

A model with lower accuracy can produce higher profit if it makes economically rational decisions.

In financial systems, **money is the metric**.

---

## Stress Testing

The model simulates recession scenarios by increasing default probabilities.

This evaluates:

* Capital resilience
* Sensitivity of profit to macroeconomic shocks
* Portfolio robustness under stress

---

## Technologies

Python
Pandas
Scikit-learn
NumPy
Matplotlib

---

## What This Project Demonstrates

* End-to-end ML pipeline development
* Business metric optimization
* Credit risk modeling
* Threshold calibration under asymmetric loss
* Portfolio-level decision intelligence

##Some images from results:

<img width="2196" height="1729" alt="phase9_dashboard" src="https://github.com/user-attachments/assets/3d3525fe-24e3-495b-b747-337215c726de" />

<img width="2384" height="1523" alt="phase2_eda" src="https://github.com/user-attachments/assets/5c314bf7-840f-40fc-be66-b55b11f71704" />

<img width="2044" height="743" alt="phase5_decision" src="https://github.com/user-attachments/assets/243fd1b2-b0a6-4c91-8726-b53ed9878d64" />
