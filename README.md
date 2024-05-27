# Loan Classification

This project aims to classify loan applications based on various features using different machine learning algorithms. The project includes data preprocessing, model training, validation, and comparison of various classification models.

## Project Structure

The repository contains the following files:

- `train_loan.csv`: The training dataset used for building the classification models.
- `test_loan.csv`: The testing dataset used for evaluating the performance of the models.
- `loan_class.ipynb`: Jupyter notebook containing the code for data preprocessing, model training, validation, and comparison.
- `loan_class-checkpoint.ipynb`: Checkpoint file for the Jupyter notebook.

## Getting Started

To get started with this project, follow the instructions below:

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- Jupyter Notebook
- Required Python packages (listed in `requirements.txt`)

### Installing

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/loan-classification.git
    cd loan-classification
    ```

2. Install the required packages:

    ```bash
    pip install pandas
    pip install seaborn
    ```

### Running the Jupyter Notebook

1. Start the Jupyter Notebook server:

    ```bash
    jupyter notebook
    ```

2. Open the `loan_class.ipynb` notebook in your browser and run the cells to execute the code.

## Project Workflow

1. **Data Preprocessing**: The dataset is loaded, and preprocessing steps such as handling missing values, encoding categorical features, and feature scaling are performed.

2. **Model Training**: Different machine learning models (e.g., Logistic Regression, Decision Tree, Random Forest, XGBoost) are trained on the training dataset.

3. **Model Validation**: The models are validated using cross-validation techniques, and performance metrics such as accuracy, ROC-AUC scores are calculated.

4. **Model Comparison**: The performance of different models is compared, and the best-performing model is selected.

5. **Prediction**: The best model is used to make predictions on the test dataset.

## Results

The performance of different models is compared using validation scores and leaderboard scores. I settled on the final model being a Random Forest Classifier with tuned hyperparameters using GridSearch. The mean accuracy of this model was 81%.

## Conclusion

This project demonstrates the process of building and evaluating different regression and machine learning models for loan classification. The Random Forest and XGBoost models showed promising results in terms of accuracy and ROC-AUC scores.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Kaggle](https://www.kaggle.com/) for providing the dataset.
- [Scikit-learn](https://scikit-learn.org/) for the machine learning tools and libraries.
