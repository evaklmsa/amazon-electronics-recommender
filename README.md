# Project Title
Amazon Electronic Product Recommendation System

This project develops a recommendation system for Amazon’s electronics category, aiming to deliver personalized product suggestions that enhance the user experience while supporting business outcomes. Several approaches including Neural Collaborative Filtering, Collaborative Filtering are implemented and compared in order to identify the most effective strategy for large-scale, sparse user–item data.


## Problem Statement
The primary objective is to build a recommendation system that accurately predicts user preferences for electronic products. The challenge lies in efficiently processing a massive dataset with over 7 million rows, where the user-item interaction matrix is sparse. The project aims to:
1. Preprocess and reduce the dataset to a manageable size while preserving meaningful user-item interactions.
2. Develop and evaluate several recommendation models, including rank-based, user-user similarity, item-item similarity, and matrix factorization (SVD).
3. Optimize the best-performing models using hyperparameter tuning to improve predictive accuracy.
4. Identify and justify the selection of the most suitable model based on a balance of performance metrics (RMSE, Precision, Recall, and F1-score) and computational efficiency.

## Dataset Details
The dataset, sourced from Kaggle’s [Amazon Electronics Rating Notebook](https://www.kaggle.com/code/anasmjali/recommendation-system-amazon-electronics-rating), contains user ratings for various Amazon electronics products.
* **Original Size**: The dataset initially contains over 7.8 million rows.
* **Columns**: The data includes `user_id`, `prod_id`, `rating`, and `timestamp`. The `timestamp` column was dropped during preprocessing.
* **Preprocessing**: Due to the dataset's size and sparsity, a subset was created by filtering for users with at least 50 ratings and products with at least 5 ratings. This reduced the dataset to 65,290 observations, making it computationally feasible for model training while retaining active users and popular products. 

## Methods and Algorithms
This project employs a diverse set of methods to build and evaluate the recommendation system:
1. **Rank-Based Recommendation**: A non-personalized baseline model that recommends products based on their average rating and the number of reviews.
2. **Collaborative Filtering (Similarity-Based)**:
    * **User-User Similarity**: Recommends items to a user based on the preferences of similar users, calculated using **cosine** similarity.
    * **Item-Item Similarity**: Recommends items similar to those a user has already liked, also using **cosine** similarity.
3. **Collaborative Filtering (Model-Based)**:
    * **Singular Value Decomposition (SVD)**: A matrix factorization technique that decomposes the user-item interaction matrix into latent feature matrices for users and items, which are then used to predict ratings.
4. **Neural Collaborative Filtering**: A deep learning approach that replaces linear matrix factorization with a neural network to model user-item interactions.
**Evaluation Metrics**: Models are evaluated using **Root Mean Squared Error (RMSE)** to measure prediction accuracy, and **Precision@k**, **Recall@k**, and **F1-score@k** to assess the quality of the top-N recommendations. Hyperparameter tuning was performed using **`GridSearchCV`** for the similarity-based models and **`Optuna`** for the NCF model.

## Key Results
* **Optimized Models**: After extensive hyperparameter tuning:
    * The optimized user-user similarity-based collaborative filtering model achieved the highest F1 score (~0.829), indicating superior performance.
    * Matrix factorization exhibited a lower RMSE (~0.8980), demonstrating its ability to capture underlying patterns and nuances in user-item interactions.
* **Conclusion**: After weighing these factors, user-user similarity-based collaborative filtering was selected as the recommendation model. This decision was driven by the need to optimize precision and recall, ensuring relevant products are recommended and recommended products are relevant, while noting that the RMSE of this method was only 0.06 higher than that of matrix factorization.

## Tech Stack
* **Libraries**:
    * `pandas` and `numpy` for data manipulation.
    * `matplotlib` and `seaborn` for data visualization.
    * `surprise` for building and evaluating collaborative filtering models.
    * `scikit-learn` for data splitting.
    * `optuna` and `GridSearchCV` for hyperparameter optimization.
    * `torch` for implementing the Neural Collaborative Filtering model.