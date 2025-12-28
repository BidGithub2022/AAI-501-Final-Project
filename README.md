# Project Title: CAR VALUE & RECOMMENDATION SYSTEM
This project is a part of the AAI-501 course in the Applied Artificial Intelligence Program at the University of San Diego (USD). 

Project Status: Completed

## Installation
To use this project, first clone the repo on your device using the command below:

```
git init
git clone https://github.com/BidGithub2022/AAI-501-Final-Project.git
```

## Project Intro/Objective
The main purpose of this project is to develop an integrated deep learning framework that provides consumers with objective, data-driven insights into used car market values. In the wake of post-pandemic supply shocks, traditional pricing intuition has become unreliable; therefore, this project aims to deploy a DNN Regressor for price estimation and a DNN Classifier to automatically flag listings as "Good Deals" or "Overpriced." By establishing a statistical market benchmark, the system empowers buyers to navigate high-volatility environments with greater transparency and confidence.

Beyond simple valuation, the project utilizes an unsupervised Autoencoder to generate deep feature embeddings, enabling a robust similarity-matching engine. This application addresses the critical need for "fair-value alternatives" by surfacing vehicles that are structurally similar in performance and condition, even if they differ by brand name. The potential impact is a more efficient marketplace where consumers can avoid price premiums and discover high-value substitutes through latent-space analysis, effectively reducing information asymmetry and helping users make more informed financial decisions in a complex automotive sector.

## Partner(s)/Contributor(s)  
Bidyut Prabha Sahu

## Methods Used
EDA
Deep Neural Network (DNN)
Linear Regression
Deep Neural Network (DNN) Regressor
Rule-Based Engine
DNN Classifier
Unsupervised Autoencoder
t-Distributed Stochastic Neighbor Embedding (t-SNE)
Supervised Benchmarking
Correlation Heatmaps
Price Distribution Histograms
Correlation Matrices

## Technologies
1. Development & Runtime Environment

Google Colab: Used as the primary cloud-based development environment, allowing for accelerated computing and collaborative notebook-based development.

Jupyter Notebook (.ipynb): The standard format for documenting and executing the Python code blocks in a logical, phased sequence.

2. Core Programming & Numerical Computing

Python: The foundational language for the entire pipeline.

NumPy: Utilized for high-performance multidimensional array processing and linear algebra operations required by the neural networks.

3. Data Manipulation & Analysis

Pandas: The core library used in Phase 1 for data ingestion, cleaning, handling categorical variables, and structured data analysis.

4. Machine Learning & Preprocessing (Scikit-Learn)

Scikit-Learn (sklearn): Used extensively for the traditional machine learning components, including:

Preprocessing: StandardScaler for feature normalization and OneHotEncoder for categorical encoding.

Pipelines: ColumnTransformer to automate the data transformation workflow.

Dimensionality Reduction: TSNE (t-Distributed Stochastic Neighbor Embedding) to visualize the Autoencoder's latent space.

Similarity Metrics: cosine_similarity to power the recommendation engine.

5. Deep Learning Framework (TensorFlow/Keras)

TensorFlow: The backend engine for building and training the deep neural networks.

Keras: The high-level API used to design the architectures:

  * DNN Regressor: For non-linear price estimation.

  * DNN Classifier: For market-value flagging.

  * Autoencoder: For unsupervised feature extraction and generating "vehicle fingerprints."

6. Data Visualization

Matplotlib: Used for generating standard plots, histograms, and regression line visualizations.

Seaborn: Used for advanced statistical visualizations, such as correlation heatmaps and t-SNE cluster plots.




<img width="695" height="306" alt="Screenshot 2025-12-27 at 7 08 24 PM" src="https://github.com/user-attachments/assets/441ef6fb-ba8b-4a07-a995-d66f3aac756a" />

## Project Description
### 1. Project Overview
This project addresses the transparency gap in the volatile post-pandemic automotive market. Using a multi-stage artificial intelligence pipeline, the system analyzes vehicle listings to provide three core services: precise Fair Value Estimation, automated Market Deal Classification, and Deep Similarity Matching. By moving beyond traditional linear pricing models, this project leverages deep learning to identify non-linear relationships between vehicle attributes and market value, empowering consumers to make data-driven financial decisions.

### 2. Dataset Description
The project utilizes a comprehensive Car Price Prediction Dataset sourced from Kaggle, representing a diverse cross-section of the secondary automotive market.

Size of Dataset: Approximately 2,500 unique vehicle listings.

Variables: 9 primary features covering structural, mechanical, and usage attributes.

Data Source: Aggregated market data reflecting late-pandemic pricing trends.

### 3. Questions and Hypotheses
Core Research Question: Can a deep learning architecture accurately identify "fair market value" in a noisy, high-volatility environment where traditional linear models fail?

Hypotheses:

Non-Linearity Hypothesis: Vehicle depreciation is not a simple linear function of age and mileage; rather, it is a complex interaction between brand prestige, engine type, and wear.

Outlier Hypothesis: Statistical deviations from the predicted "Market Mean" can be used to programmatically flag "Good Deals" and "Overpriced" listings with higher accuracy than manual human review.

Latent Identity Hypothesis: Vehicles possess a structural "DNA" that can be captured in a low-dimensional latent space to suggest functional alternatives across different brands.

### 4. Technical Work: Analysis, Visualization, and Modeling
To solve the problem, the project implements a modular technical stack:

Data Analysis & Manipulation: Using Pandas and NumPy for feature engineering (e.g., converting "Year" to "Car Age") and handling high-cardinality categorical variables through one-hot encoding.

Visualization: Using Seaborn and Matplotlib to generate correlation heatmaps, price distribution histograms, and t-SNE cluster plots to validate model embeddings.

Modeling:

Supervised Learning: A multi-layer DNN Regressor (TensorFlow/Keras) for price estimation and a DNN Classifier for deal categorization.

Unsupervised Learning: An Autoencoder architecture to extract latent features for the similarity-matching engine.

### 5. Roadblocks and Challenges
The Correlation Gap: Initial analysis revealed that traditional features (Mileage/Year) had a lower-than-expected linear correlation with price ($R^2 < 0.10$). This required a pivot from high-precision regression to a "Market Baselining" approach.

High Cardinality: The "Model" variable contained numerous unique entries, creating a sparse matrix after encoding. This was mitigated by using dense layers and Dropout in the DNN to prevent overfitting.

Data Noise: Extreme price variance for identical models (likely due to missing variables like "Trim Level" or "Location") made training a perfect predictor challenging, necessitating the shift toward Unsupervised Similarity as a primary value-add.

## License
See the LICENSE file.

## Acknowledgments
Thank you Professor Dave Friesen!
