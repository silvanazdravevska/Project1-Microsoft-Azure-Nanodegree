# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using Python SDK.
For optimization with HyperDrive are being used the Scikit-learn Logistic Regression—the hyperparameters.
Then an AutoML model is build and optimized on the same dataset, for comparison of the results of the two methods


## Summary
This dataset contains marketing data of a bank about certain individuals. The aim is to predict if the customer subscribes to a fixed term deposit or not.
The better performing model was the AutoML model It had an accuracy of 0.9161. In contrast, the HyperDrive model has accuracy of 0.90966.

## Scikit-learn Pipeline
The dataset was imported from specifies URL of Bank Marketing Data. It was pre-processed in the clean_data function of train.py file and split into training and testing. After that Logistic Regression Model was used for training with tuning hyperparameters such as C and max_iter using HyperDrive.
Hyperparameter tuning is the process of finding the configuration of hyperparameters that results in the best performance. The process is typically computationally expensive and manual.
The HyperDriveConfig:

hyperdrive_config = HyperDriveConfig(estimator=est,
                                     hyperparameter_sampling=ps,
                                     policy=policy,
                                     primary_metric_name='Accuracy',
                                     primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                     max_total_runs=20,
                                     max_concurrent_runs=4)

Parameter Sampler used is RandomParameterSampling, it defines random sampling over a hyperparameter search space.
Benefits from RandomParameterSampling: easy method to extract a research sample from a larger population, it allows the search space to include both discrete and continuous hyperparameters, it supports early termination of low-performance runs. 
C is the Regularization while max_iter is the maximum number of iterations.

ps = RandomParameterSampling(
    {
        '--C': uniform(0.0, 1.0), 
        '--max_iter': choice(50, 100, 150, 200, 250)
    }
)

Early Stopping Policy used is BanditPolicy. It defines an early termination policy based on slack criteria, and a frequency and delay interval for evaluation. 
Benefits from BanditPolicy: it is conceptually simple, it improves computational efficiency, it terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run.

policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)

## AutoML
Using the AutoML, VotingEnsemble model performed the best with the accuracy of 91.61%. 
Compute target is remote. The experiment type to run is classification. 
The primary metric that Automated Machine Learning will optimize for model selection is accuracy.
label_column_name is the training labels to use when fitting pipelines during an experiment. This is the value the model predicts. 
Maximum amount of time in minutes that all iterations combined can take before the experiment terminates is 30 minutes.
(ONNX) can help optimize the inference of the machine learning model. Inference, or model scoring, is the phase where the deployed model is used for prediction
Cross-validation=2; the metrics are calculated with the average of the 2 validation metrics.
The Automl Config:

automl_config = AutoMLConfig(
    compute_target=cpu_cluster,
    experiment_timeout_minutes=30,
    task='classification',
    primary_metric='accuracy',
    training_data=ds,
    label_column_name='y',
    enable_onnx_compatible_models=True,
    n_cross_validations=2)

## Pipeline comparison
The Hyperparameter Model had accuracy of 90.97%, and the AutoML was with accuracy of 91.61% i.e. difference is 0.64%.
This difference was because in HyperDrive we specified a fixed model (Logistic Regression) and could only improve the hyperparameters, whereas AutoML gave us the flexibility to use various models and get the best result.

Hyperparameter Model Best Run: Accuracy = 0.9096611026808296, Duration = 47.84s, Max iterations = 200, --C = 0.16980643557917052, Regularization Strenght = 0.16980643557917052
AutoML Best Run: Accuracy = 0.9161153262518968, Duration = 1m 36.31s, Algorithm Name = Voting Ensemble, AUC Macro = 0.94673, AUC Micro = 0.98052

## Future work
Class imbalance is a very common issue in classification problems in machine learning. Imbalanced data negatively impact the model's accuracy because it is easy for the model to be very accurate just by predicting the majority class, while the accuracy for the minority class can fail.
As the existing data has class imbalance problem we can do better pre-processing of data or get more data to balance it and to explore the important features which can result in better performance with quality data.
Usage of different metric, like AUC, better fit for imbalanced data.
Using different combinations of hyperparameters like C and max_iter with HyperDrive and also try using different loss algorithm parameters.
In AutoML we can try other values for cross validation to improve accuracy.  cross-validation is the process of taking many subsets of the full training data and training a model on each subset, the higher the number of cross validations is, the higher the accuracy achieved is, but with caution to cost.


## Proof of cluster clean up

See attached Screenshot