# CombAIns_LOT

main.ipynb is the most important - final file here

**Optimizing Pre-Flight Communications to Enhance Customer Experience**

CombAIns_LOT is a machine learning project aimed at optimizing pre-flight communications by predicting which email layouts will generate the highest click rates. The project leverages innovative feature engineering techniques and custom loss function designs to improve prediction performance in a 10-class classification task.

---

## Table of Contents

- [Motivation](#motivation)
- [Project Description](#project-description)
  - [Approach I – Customized Loss Function](#approach-i--customized-loss-function)
  - [Approach II – Decomposing the Target Column](#approach-ii--decomposing-the-target-column)
- [Feature Engineering](#feature-engineering)
- [Repository Structure](#repository-structure)
- [Installation and Running](#installation-and-running)
- [Example Code](#example-code)
- [Authors](#authors)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Motivation

The goal of this project is to improve customer communication by sending optimized emails that have a higher likelihood of receiving clicks. By utilizing apriori features along with supplementary information from the `clicked` column (which is not used as a training feature), we build a model capable of optimizing a custom loss function that drives improved label predictions.

---

## Project Description

Our project employs two main approaches:

### Approach I – Customized Loss Function
In this approach, we map the original target column to 10 unique values.
- **Key Assumptions:**
  - The apriori features do not fully explain the randomness in label selection.
  - Although the `clicked` column is not used as a training feature, it is integrated into our custom loss function to improve model predictions.
- **Loss Function Strategy:**
  - When the predicted click probability based on the email layout is low, the standard cross entropy loss is augmented with a penalty inversely proportional to the predicted probability.

### Approach II – Decomposing the Target Column
We extend the initial solution by breaking the target column into individual components.
- **Key Assumptions:**
  - Instead of treating the target as a single entity, we decompose it into its individual parts.
  - We analyze the contribution of each component when predicting if a user will click.
  - A prediction with one mismatched component is treated less harshly than one where all components are incorrect.
- **Results:**
  - Experimental results indicate that the best performance is achieved with the Random Forest model using this decomposed target approach.

---

## Feature Engineering

Robust feature engineering is crucial to our project. Below are some examples of the engineered features:
- **adult_ratio** – The ratio of adults to the total number of passengers in a booking.
- **is_international** – A flag indicating whether the booking involves more than one country.
- **booking_window_category** – Categories representing the booking time window (e.g., same_day, 1-2_days, 3-7_days, up to 4 weeks).
- **is_online_booking** – Indicates whether the booking was made online.

These features are processed through a pipeline that includes feature extraction, one-hot/target encoding, and scaling to prepare them for model training.

---

## Repository Structure

Below is an example of the repository structure. Adjust it as necessary:

