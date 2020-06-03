This project does exploratory data analysis on a given dataset for binary classification task, perform data cleaning and feature engineering, and apply multiple supervised machine learning techniques on the preprocessed dataset to evaluate the performances of these algorithms and also a write-up is attached which explains why certain algorithms worked well and why others failed to work. 

This project also includes results.csv files which gives the predictions on the unseen test data using each of the two best performing trained models [in this case, neural network and SVC performed well].  The predictions should be the class probabilities for belonging to the positive class (labeled '1').  

Learning outcome:
1.Structuring a machine learning project. 
2.Selection of a ML algorithm for a given task.
3.End-to-end of data science project pipeline from data cleaning till model evaluation.
4.Innovative methods for data cleaning and feature augmentation.

Project CODE files:
The html files of the 2 jupyter notebooks used are:

1. CODE_exploratory_data_analysis_and_cleaning.html which shows how the data was read, cleaned, preprocessed etc. The data analysis, outlier detection and feature engineering are also performed in this notebook.
The executable .ipynb file of this notebook is present under executable_jupyter_notebooks folder as exploratory_data_analysis_and_cleaning.ipynb.

2. CODE_building_tuning_model.html file shows the model selection, hyperparameter tuning using grid search and 10-fold stratified cross validation and evaluation of the models.
The executable .ipynb file of this notebook is present under executable_jupyter_notebooks folder as building_tuning_model.ipynb 

Code description:

Test data preprocessing is performed in the last cell of exploratory_data_analysis_and_cleaning.ipynb:
    
#read test data
df_test = pd.read_csv("exercise_04_test.csv")

#process test data
x_test = prepare_test_data(df_test)

#write processed test data to csv to use in building_tuning_models.ipynb.
x_test.to_csv("processed_test.csv")

Assumptions for hold-out test data:

1. Except for columns x34, x35, x68, x41 , x45 and x93, all other columns will be of numeric datatype, as in given data.
2. x41 column will only have $ sign or spaces other than the numeric (positive/negative) value, as in train data.
3. x45 column will only have % sign or spaces other than numeric value.
4. For out-of-vocab strings in the 4 string columns (x34, x35, x68,x93), the value is substituted as unk token.
