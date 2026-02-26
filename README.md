Credit Decision Intelligence System
Business Problem

How should a fintech company approve loans to maximize expected profit while controlling default risk?

Project Objective

This project builds a profit-optimized credit approval system that converts predicted default probabilities into capital allocation decisions.

Dataset

LendingClub Accepted Loans dataset.

Key Components

Data cleaning and preprocessing

Risk modeling (Probability of Default)

Profit-based decision threshold

Portfolio optimization

Stress testing under economic shifts

Core Insight

Approval threshold is not 0.5.
It is determined by expected profit under default risk.

Results

Overall default rate: ~20%

Optimal PD threshold: (fill this)

Expected portfolio profit: (fill this)

Technologies Used

Python

Pandas

Scikit-learn


Matplotlib

RESULT:
Optimal fixed threshold0.497
Max profit (fixed threshold)$125.51M

Our profit-optimised system approved 94.8% of applicants and generated $126M in portfolio profit — $49M more than the conservative rule-based approach, which only approved 55% of loans. The key insight is that a 0.20 fixed threshold is leaving $49M on the table by rejecting profitable high-rate loans. Our per-loan break-even calculation correctly identifies that a 25% rate loan can absorb a 30% default probability and still generate positive expected profit — something a fixed threshold can never capture

<img width="2196" height="1729" alt="phase9_dashboard" src="https://github.com/user-attachments/assets/3d3525fe-24e3-495b-b747-337215c726de" />

