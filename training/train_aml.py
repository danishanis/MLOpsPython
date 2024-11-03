# Model Training Code

import os
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from azureml.core.run import Run
from azureml.core import Dataset, Datastore, Workspace
import argparse
import joblib
import json
from train import split_data, train_model, get_model_metrics

def register_dataset(
    aml_workspace: Workspace,
    dataset_name: str,
    datastore_name: str,
    file_path: str
) -> Dataset:
    datastore = Datastore.get(aml_workspace, datastore_name)
    dataset = Dataset.Tabular.from_delimited_files(path=(datastore, file_path))
    dataset = dataset.register(workspace=aml_workspace,
                               name=dataset_name,
                               create_new_version=True)

    return dataset

def main():
    """
    Defining arguments which we will pass in the run command while we call
    the script from some workflow script
    """
    print("Running train_aml.py...")

    parser = argparse.ArgumentParser("train")

    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model",
        default="diabetes_model.pkl"
    )
    
    # passing data onto the next step
    parser.add_argument(
        "--step_output",
        type=str,
        help=("output for passing data to the next step")
    )

    parser.add_argument(
        "--caller-run-id",
        type=str,
        help=("caller run id, for example ADF pipeline run id")
    )

    # dataset name to train the model
    parser.add_argument(
        "--dataset_name",
        type=str,
        help=("Dataset name. Dataset must be passed by name\
              to always get the desired dataset version\
              rather than the one used while the pipeline creation")
    )

    # for dataset registry
    parser.add_argument(
        "--dataset_version",
        type=str,
        help=("dataset version")
    )

    parser.add_argument(
        "--data_file_path",
        type=str,
        help=("data file path, if specified,\
               a new version of the dataset will be registered")
    )

    args = parser.parse_args()
    
    print("Argument [model_name]: %s" % args.model_name)
    print("Argument [step_output]: %s" % args.step_output)
    print("Argument [dataset_version]: %s" % args.dataset_version)
    print("Argument [data_file_path]: %s" % args.data_file_path)
    print("Argument [caller_run_id]: %s" % args.caller_run_id)
    print("Argument [dataset_name]: %s" % args.dataset_name)

    # Assigning variables for each argument captured in the rrun command
    model_name = args.model_name
    step_output_path = args.step_output
    dataset_version = args.dataset_version
    data_file_path = args.data_file_path
    dataset_name = args.dataset_name

    # Setting pipenv configurations from get context method (so the /
    # script can run from start to finish)
    run = Run.get_context()

    print("Getting training parameters")

    # Load the training parameters from the parameters file instead of /
    # hardcoding them
    with open("../utils/parameters.json") as f:
        pars = json.load(f)
    try:
        train_args = pars["training"]
    except KeyError:
        print("Could not load training values from file")
        train_args = {}

    # Logging the training parameters so one can check later what /
    # parameters were used in previous runs
    print(f"Parameters: {train_args}")
    for (k, v) in train_args.items():
        run.log(k, v)
        # run.parent.log(k, v)    # logging metrics to an optional parent run

    # Get the dataset
    dataset = Dataset.get_by_name(run.experiment.workspace, dataset_name,
                                  dataset_version)  # NOQA: E402, E501


    # if (dataset_name):
    #     if (data_file_path == 'none'):
    #         dataset = Dataset.get_by_name(run.experiment.workspace, dataset_name, dataset_version)  # NOQA: E402, E501
    #     else:
    #         dataset = register_dataset(run.experiment.workspace,
    #                                    dataset_name,
    #                                    os.environ.get("DATASTORE_NAME"),
    #                                    data_file_path)
    # else:
    #     e = ("No dataset provided")
    #     print(e)
    #     raise Exception(e)

    # Linking dataset to the step run so it is trackable in the UI
    # (Basically linking the pipeline run id with the dataset id)
    run.input_datasets['training_data'] = dataset
    run.parent.tag("dataset_id", value=dataset.id)

    # Split the data into test/train
    df = dataset.to_pandas_dataframe()
    data = split_data(df)

    # Train the model
    model = train_model(data, train_args)

    # Evaluate and log the metrics returned from the train function
    metrics = get_model_metrics(model, data)
    for (k, v) in metrics.items():
        run.log(k, v)
        run.parent.log(k, v)

    # Pass model file to next step/an output path
    os.makedirs(step_output_path, exist_ok=True)
    model_output_path = os.path.join(step_output_path, model_name)
    joblib.dump(value=model, filename=model_output_path)

    # Also upload model file to run outputs for history
    os.makedirs('outputs', exist_ok=True)
    output_path = os.path.join('outputs', model_name)
    joblib.dump(value=model, filename=output_path)

    run.tag("run_type", value="train")
    print(f"tags now present for run: {run.tags}")

    run.complete()

if __name__ == '__main__':
    main()