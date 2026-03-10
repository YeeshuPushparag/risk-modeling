# Financial Risk Model Training & Data Pipeline

This repository contains the **data collection, data transformation, feature engineering, and machine learning training pipeline** used to build the models for the production risk analytics platform.

The models trained here are later used in the production system for portfolio analytics, risk estimation, and financial insights.

Production platform repository:  
https://github.com/YeeshuPushparag/risk-engine/

---

## Purpose

This repository focuses on the **research and training layer** of the risk analytics platform.

It is responsible for:

- collecting financial datasets
- preparing training datasets
- performing feature engineering
- training machine learning models
- evaluating model performance

The trained models are later integrated into the production analytics system.

---

## Data Sources

The training datasets are built using multiple financial and macroeconomic sources:

- yFinance – market price data
- FRED – macroeconomic indicators
- 13F filings – institutional holdings data

These datasets are combined to create model training datasets.

The models are trained on approximately **1.5 years of historical financial data covering ~3200 instruments across multiple asset classes**.

---

## Data Pipeline

The model training workflow follows these steps:

Data Collection  
↓  
Data Cleaning & Transformation  
↓  
Feature Engineering  
↓  
Model Training  
↓  
Model Evaluation  
↓  
Export Trained Models for Production

---

## Feature Engineering

Features are generated from financial time-series data and macroeconomic indicators.

Examples include:

- asset returns
- rolling volatility
- drawdowns
- moving averages
- macroeconomic variables
- liquidity metrics

These features are used as inputs for machine learning models.

---

## Machine Learning Models

The repository contains training pipelines for models used in financial risk analysis.

Examples include:

- Probability of Default (PD)
- Value at Risk (VaR)
- volatility forecasting
- asset clustering using K-Means

Models are trained on historical datasets and evaluated before being integrated into the production platform.