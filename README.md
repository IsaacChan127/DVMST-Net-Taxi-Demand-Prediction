# DVMST-Net Taxi Demand Prediction (NYC)

## Overview

This project implements a deep learning–based taxi demand forecasting system inspired by the DVMST-Net spatio-temporal architecture. The model predicts short-term hourly taxi pickup demand for individual New York City taxi zones using historical demand sequences and temporal contextual features.

The implementation focuses on zone-level modelling, where each taxi zone is trained independently using historical demand data.

---

## Dataset

The model uses aggregated taxi pickup demand derived from the NYC Taxi and Limousine Commission (TLC) trip record dataset.

Raw trip-level taxi data is transformed into a pivot matrix where:

- Rows represent hourly timestamps.
- Columns represent taxi pickup zones.

Input data is loaded from:
