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
- [Data Processing and Augmentation](#data-processing-and-augmentation)
- [Installation and Running](#installation-and-running)
- [Authors](#authors)
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

## Data Processing and Augmentation

In order to improve model training and robustness, we apply custom data augmentation strategies:

- **Label Creation**:  
  We concatenate multiple categorical features (`graphic_design`, `top_1_section`, `top_2_section`, `top_3_section`) to create a compound target label, which is later encoded into integer class labels.

- **Negative Sample Generation**:  
  For each original data point, we generate synthetic “negative” samples by modifying the layout label to other possible values (0–9), setting the `clicked` column to 0 for these rows. This approach balances the dataset and enhances the model's ability to distinguish effective layouts from ineffective ones.

This augmented dataset forms the basis for training our classification models under both Approach I and Approach II described earlier.

---

## Installation and Running

To run the project locally:

```python
# Clone the repository
git clone https://github.com/yourusername/CombAIns_LOT.git
cd CombAIns_LOT

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the main notebook
jupyter notebook main.ipynb
```

## Authors

* Filip Langiewicz – [@FilipLangiewicz](https://github.com/FilipLangiewicz)
* Mikołaj Rowicki - [@MikolajRowicki](https://github.com/MikolajRowicki)
* Jakub Półtorak - [@JakubPoltorak147](https://github.com/JakubPoltorak147)
* Igor Rudolf - [@Igor](https://github.com/IgorRudolf)
* Maciej Momot - [@MaciejMomot](https://github.com/MaciejMomot)
* Michał Pytel - [@Michael-Pytel](https://github.com/Michael-Pytel)

---


## Acknowledgements

This project was developed during the HackCarpathia2025 and is inspired by real-world challenges in airline customer communication optimization.  
Special thanks to the organizers, mentors, and all the fellow participants who made the event engaging and insightful.
