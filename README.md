# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains marketing data of a bank about certain individuals. We seek to predict weather an individual would consider a bank deposit.

The better performing model was the AutoML model It had an accuracy of 0.9161. In contrast, the HyperDrive model has accuracy of 0.90966.

## Scikit-learn Pipeline
The HyperDriveConfig:

hyperdrive_config = HyperDriveConfig(estimator=est,
                                     hyperparameter_sampling=ps,
                                     policy=policy,
                                     primary_metric_name='Accuracy',
                                     primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                     max_total_runs=20,
                                     max_concurrent_runs=4)

Parameter Sampler used is RandomParameterSampling, it defines random sampling over a hyperparameter search space.

ps = RandomParameterSampling(
    {
        '--C': uniform(0.0, 1.0), 
        '--max_iter': choice(50, 100, 150, 200, 250)
    }
)

Early Stopping Policy used is BanditPolicy. It defines an early termination policy based on slack criteria, and a frequency and delay interval for evaluation.

policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)

## AutoML
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

Hyperparameter Model Best Run: Accuracy = 0.9096611026808296, Duration = 47.84s, Max iterations = 200, --C = 0.16980643557917052, Regularization Strenght = 0.16980643557917052
AutoML Best Run: Accuracy = 0.9161153262518968, Duration = 1m 36.31s, Algorithm Name = Voting Ensemble, AUC Macro = 0.94673, AUC Micro = 0.98052

## Future work
As the existing data has class imbalance problem we can do better pre-processing of data or get more data to balance it and to explore the important features which can result in better performance with quality data

Using different combinations of hyperparameters like C and max_iter with HyperDrive and also try using different loss algorithm parameters.

In AutoML we can try other values for cross validation to improve accuracy


## Proof of cluster clean up

See attached Screenshot