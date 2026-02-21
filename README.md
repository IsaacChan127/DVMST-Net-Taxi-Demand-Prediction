# NYC Taxi Demand Prediction using DVMST-Net Inspired Model

## Overview

This project implements a deep learning model for short-term taxi demand prediction using historical New York City taxi trip data. The system forecasts hourly taxi pickup demand for individual taxi zones by learning temporal demand patterns and spatial relationships between neighbouring regions.

The model is inspired by the DVMST-Net spatio-temporal architecture and combines custom spatial feature learning with LSTM-based sequence modelling.

---

## How the Model Works

The workflow follows a structured machine learning pipeline:

1. Raw taxi trip data is aggregated into hourly pickup demand counts.
2. Demand is organised into a pivot matrix where each column represents a taxi zone.
3. Historical demand sequences are constructed for supervised learning.
4. Temporal features are added using one-hot encoding.
5. A neural network learns spatial and temporal demand patterns to predict future demand.

---

