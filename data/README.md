# Pivot Matrix Dataset
Dataset derived from NYC TLC Open Data (June–August 2024).
## Overview

The pivot matrix used in this project is constructed from three months of New York City yellow taxi trip record data obtained from the NYC Taxi and Limousine Commission (TLC) open data repository.

The dataset covers taxi trip activity during:

- June 2024
- July 2024
- August 2024

Raw trip-level GPS records were aggregated into hourly taxi pickup demand counts across geographic taxi zones.

---

## Construction Methodology

The pivot matrix was generated through the following preprocessing pipeline:

1. Raw taxi trip records were filtered to retain valid pickup timestamps and pickup location identifiers.
2. Trips were grouped by pickup hour using the `tpep_pickup_datetime` field.
3. Taxi demand was aggregated as the total number of pickups per zone per hour.
4. Missing time-zone combinations were filled with zero values to preserve a consistent temporal structure.

This transformation converts sparse trip-level data into a structured time-series representation suitable for deep learning models.

---

## Data Structure

The pivot matrix is stored as:

[pivot_matrix.csv](https://github.com/user-attachments/files/25455701/pivot_matrix.csv)



Structure:

- Rows represent hourly timestamps.
- Columns represent NYC Taxi Zone IDs (`PULocationID`).
- Cell values represent the number of taxi pickups recorded in that zone during the corresponding hour.

Example:

| Timestamp | Zone 132 | Zone 140 | Zone 164 |
|---|---|---|---|
| 2024-06-01 00:00 | 12 | 5 | 2 |
| 2024-06-01 01:00 | 9 | 3 | 1 |

---

## Purpose

The pivot matrix serves as the primary input to the spatio-temporal deep learning model.

It enables:

- temporal sequence modelling using LSTM layers,
- spatial demand analysis across neighbouring taxi zones,
- integration of contextual temporal features such as hour-of-day and day-of-week.

---

## Data Source

NYC Taxi and Limousine Commission (TLC) Trip Record Data:

https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

All copyrights belong to the NYC TLC Open Data programme.
