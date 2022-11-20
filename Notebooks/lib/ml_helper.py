# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

def cross_validate(pipeline, X_train, y_train, metric, folds=5):
    """
    The function return results of cross validation given provided metric.

    Parameters
    ----------
    pipeline : sklearn pipeline
        Machine learning pipeline to be cross validated
    metric : str
        Metric to be calculated.
    X_train: DataFrame
       Dataframe with independent variables.
    y_train: array
       Dependent variable. 
    folds : int, optional
        Number of folds to be used. The default is 5.

    Returns
    -------
    metric_scores: list of floats
        List of provided metric results per each fold

    """
        
    metric_scores = cross_val_score(pipeline, X_train, y_train.values.reshape(-1, 1),                                                           
                              n_jobs=-1, scoring=metric)  
    msg = f"The mean Roc AUC score is {metric_scores.mean()} and the standard deviation is {metric_scores.std()}"
    print(msg)

    return metric_scores
        

def fit_pipeline(pipeline, X_train, y_train):
    """
    The function fits the  provided machine learning pipeline.
    
    Parameters
    ----------
    pipeline : sklearn pipeline
        Machine learning pipeline to be fitted.
    X_train: DataFrame
       Dataframe with independent variables.
    y_train: array
       Dependent variable. 
       
    Returns
    -------
    fitted_pipeline: sklearn pipeline
        Fitted machine learning pipeline.
       
    """
    
    fitted_pipeline = pipeline.fit(X_train, y_train.values.reshape(-1, 1))    

    return fitted_pipeline  

def get_features_coefficients(fitted_pipeline, X_train_columns):
    """
    The function creates a dataframe of coefficients of the features.
    
    Parameters
    ----------
    fitted_pipeline: sklearn pipeline
        Fitted machine learning pipeline. 
    X_train_columns: list of str
        Column names of the independent variables used for the training.
       
    Returns
    -------
    feature_coefficients: DataFrame
        Dataframe with the coefficients.
    """
    features_coefficients = pd.DataFrame({'Features': X_train_columns, 'Feature_coefficient': fitted_pipeline[1].coef_[0]})
    features_coefficients = features_coefficients[features_coefficients['Feature_coefficient'] != 0]
    features_coefficients.reset_index(drop=True, inplace=True)
    features_coefficients = features_coefficients.sort_values('Feature_coefficient', ascending=False)
    
    return features_coefficients
            
def plot_confusion_matrix(fitted_pipeline, X, y):
    
    """
    The function evaluates predictions against provided values.
    The results are displayed as roc_auc score, f1 score and confusion matrix.
    
    Parameters
    ----------
    fitted_pipeline : sklearn pipeline
        Machine learning pipeline to be fitted.
    X: DataFrame
       Dataframe with independent variables.
    y: array
       Dependent variable. 
       
    """
           
    predictions = fitted_pipeline.predict(X)
    
    ConfusionMatrix = confusion_matrix(y, predictions)
    RocAuc = roc_auc_score(y, predictions) 
    f1score = f1_score(y, predictions)
    ConfusionMatrixDisplay(ConfusionMatrix, display_labels = [0,1]).plot(values_format='d')
    plt.show()
    print('ROC AUC on provided set is: ', RocAuc)
    print('F1 score on provided set is: ', f1score)
    print()
    
def compare_pipelines(pipeline_names, cv_results):
    """
    Plots results of cross validation for each provided pipeline for comparisson.
    
    pipeline_names: list of str
        List of name(s) of the provided pipeline(s)
    cv_results: list of lists of floats
        List of cross validation results of each pipeline
        
    """
    
    # if the pipeline name provided is string make it a list
    if type(pipeline_names) != list:
        pipeline_names = [pipeline_names]

    # if only one list of floats is provided make it item of list  
    if type(cv_results[0]) != list:
        cv_results = [cv_results]

    for pipeline_name, cv_result in zip(pipeline_names, cv_results):
        msg = f"{pipeline_name}: The mean Roc AUC score is {cv_result.mean()} and the standard deviation is {cv_result.std()}"
    
        print(msg)
        
    # Compare pipelines
    fig = plt.figure(figsize = (12, 8))
    fig.suptitle('Pipelines Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(cv_results)
    ax.set_xticklabels(pipeline_names)
    plt.show()   







            
            