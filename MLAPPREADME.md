# Interactive Machine Learning Explorer 🧠


## Project Overview 🎯

The **Interactive Machine Learning Explorer** is a Streamlit web application designed to provide a hands-on experience with fundamental machine learning concepts. The primary goal is to allow users, regardless of their technical background, to easily:

* **Explore different datasets:** Utilize built-in examples or upload their own data in CSV format.
* **Understand machine learning tasks:** Experiment with both classification and regression problems.
* **Experiment with various models:** Train and evaluate popular machine learning algorithms.
* **Visualize model performance:** Gain insights through intuitive metrics and plots.

With this app, users can quickly load data, select a machine learning model, adjust its hyperparameters, and immediately see the results through performance metrics and visualizations, fostering a better understanding of how these models work in practice.

## Instructions ⚙️

### Running the App Locally

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/jthomp33/THOMPSON-Data-Science-Portfolio.git](https://github.com/jthomp33/THOMPSON-Data-Science-Portfolio.git)
    cd your-repository-name
    ```
  

2.  **Install Dependencies:**
    Ensure you have Python (version 3.8 or higher is recommended) and pip installed.
    ```
    Install the necessary packages individually:
    ```bash
    pip install streamlit
    pip install pandas
    pip install scikit-learn
    pip install matplotlib
    pip install seaborn
    pip install numpy
    ```
    

3.  **Run the Streamlit App:**
    Navigate to the directory containing the Python script (`your_script_name.py`) and execute:
    ```bash
    streamlit run your_script_name.py
    ```
    This command will automatically open the application in your default web browser.

### Deployed Version 🌐

You can access the deployed version of this app online at:

[https://your-streamlit-app-url.streamlit.app](https://your-streamlit-app-url.streamlit.app)

[This will be updated pending github resolution]

## App Features 💡

This app provides a range of features to explore machine learning models:

* **Dataset Loading:** Supports loading built-in datasets (Iris for classification, California Housing for regression) and uploading custom CSV files.
* **Task Type Handling:** Automatically attempts to detect the task (classification or regression) based on the target variable's characteristics, with an option for the user to override.
* **Data Preprocessing:**
    * **Label Encoding:** Non-numeric categorical features are automatically converted into numerical representations using `LabelEncoder` from scikit-learn.
    * **Standard Scaling:** Numerical features are scaled using `StandardScaler` from scikit-learn to standardize their range, which can improve the performance of some models.
* **Model Selection and Hyperparameter Tuning:**
    * **Classification Models:**
        * **Logistic Regression:** Implemented using `LogisticRegression` from scikit-learn. The **Inverse Regularization Strength (C)** is a hyperparameter that can be adjusted using a slider in the sidebar. Lower values of C imply stronger regularization.
        * **Decision Tree:** Implemented using `DecisionTreeClassifier` from scikit-learn. Tunable hyperparameters include **Maximum Depth**, **Minimum Samples Split**, **Minimum Samples Leaf**, and the **Criterion** for splitting nodes (Gini impurity or entropy), all controlled via sidebar sliders and a dropdown.
        * **K-Nearest Neighbors:** Implemented using `KNeighborsClassifier` from scikit-learn. Users can tune the **Number of Neighbors (k)**, the **Weighting** of neighbors (uniform or by distance), and the **Distance Metric** (Minkowski, Euclidean, or Manhattan) through sidebar controls.
    * **Regression Models:**
        * **Linear Regression:** Implemented using `LinearRegression` from scikit-learn. This basic implementation does not expose any tunable hyperparameters in the app.
        * **Decision Tree:** Implemented using `DecisionTreeRegressor` from scikit-learn. Similar to the classification tree, users can adjust **Maximum Depth**, **Minimum Samples Split**, **Minimum Samples Leaf**, and the **Criterion** for splitting (squared error, Friedman MSE, or absolute error) via the sidebar.
* **Performance Evaluation:**
    * **Classification:** Displays accuracy, precision, recall, F1-score, and a confusion matrix generated using `confusion_matrix` from scikit-learn. For binary classification, an ROC curve and AUC score (calculated using `roc_curve` and `roc_auc_score` from scikit-learn) are also shown.
    * **Regression:** Presents the R-squared score (`r2_score`), Mean Squared Error (`mean_squared_error`), and Mean Absolute Error (`mean_absolute_error`) from scikit-learn. Additionally, it visualizes predictions against actual values and displays a residual plot.
* **Visualizations:** Utilizes `matplotlib` and `seaborn` for creating informative plots such as confusion matrices, ROC curves, scatter plots of predictions vs. actuals, residual plots, and visualizations of the decision tree structure (using `plot_tree` from scikit-learn).

## References 📚

The implementation of this interactive machine learning explorer was informed by the following resources:

* **Streamlit Documentation:** [https://streamlit.io/docs/](https://streamlit.io/docs/) - The official documentation for the Streamlit library, which provides the framework for building the web application.
* **scikit-learn Documentation:** [https://scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html) - The user guide for the scikit-learn library, which provides the machine learning models, preprocessing tools, and evaluation metrics used in the app.
* **matplotlib Documentation:** [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html) - The documentation for the matplotlib library, used for creating basic plots and visualizations.
* **seaborn Documentation:** [https://seaborn.pydata.org/](https://seaborn.pydata.org/) - The documentation for the seaborn library, which provides a higher-level interface for creating statistical graphics.
* **Various online tutorials and examples** for using Streamlit with scikit-learn for interactive machine learning applications.
