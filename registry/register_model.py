import json
import os
import sys
import argparse
import traceback
import joblib
from azureml.core import Run, Experiment, Workspace, Dataset
from azureml.core.model import Model as AMLModel

def main():
    """
    Registering model file, training parameters, evaluation metrics,
    experiment names etc.
    """
    print("Registering model...")

    run = Run.get_context()
    
    if (run.id.startswith('OfflineRun')):
        from dotenv import load_dotenv
        # For local development, set values in this section
        load_dotenv()
        workspace_name = os.environ.get("WORKSPACE_NAME")
        experiment_name = os.environ.get("EXPERIMENT_NAME")
        resource_group = os.environ.get("RESOURCE_GROUP")
        subscription_id = os.environ.get("SUBSCRIPTION_ID")
        # run_id useful to query previous runs
        run_id = "bd184a18-2ac8-4951-8e78-e290bef3b012"
        aml_workspace = Workspace.get(
            name=workspace_name,
            subscription_id=subscription_id,
            resource_group=resource_group
        )
        ws = aml_workspace
        exp = Experiment(ws, experiment_name)
    else:
        ws = run.experiment.workspace
        exp = run.experiment
        run_id = 'amlcompute'

    parser = argparse.ArgumentParser("register")

    parser.add_argument(
        "--run_id",
        type=str,
        help="Training run ID",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the Model",
        default="diabetes_model.pkl",
    )

    parser.add_argument(
        "--step_input",
        type=str,
        help=("input from previous steps")
    )

    args = parser.parse_args()
    if (args.run_id is not None):
        run_id = args.run_id
    if (run_id == 'amlcompute'):
        run_id = run.parent.id
    model_name = args.model_name
    model_path = args.step_input

    print("Getting registration parameters")

    # Load the registration parameters from the parameters file
    with open("parameters.json") as f:
        pars = json.load(f)
    try:
        register_args = pars["registration"]
    except KeyError:
        print("Could not load registration values from file")
        register_args = {"tags": []}
    
    # dictionary of metrics and their corresponding values from /
    # this run 
    model_tags = {}

    # taking each element from the metric list, getting its /
    # corresponding metric value for the pipeline's parent run and /
    # save it 
    for tag in register_args["tags"]:
        try:
            mtag = run.parent.get_metrics()[tag]
            model_tags[tag] = mtag
        except KeyError:
            print(f"Could not find {tag} metric on parent run.")

    # load the model
    print("Loading model from " + model_path)
    
    # Loading the model file from where the training code /
    # has placed it (eg. in training script, the model file was /
    # dumped in an output path)
    model_file = os.path.join(model_path, model_name)
    model = joblib.load(model_file)
    parent_tags = run.parent.get_tags()

    try:
        build_id = parent_tags["BuildId"]
    except KeyError:
        build_id = None
        print("BuildId tag not found on parent run.")
        print(f"Tags present: {parent_tags}")
    
    try:
        build_uri = parent_tags["BuildUri"]
    except KeyError:
        build_uri = None
        print("BuildUri tag not found on parent run.")
        print(f"Tags present: {parent_tags}")

    if (model is not None):
        # If model found, we fetch the dataset id that the model was /
        # trained on from the appearence tag and proceed to registering /
        # the model
        dataset_id = parent_tags["dataset_id"]
        if (build_id is None):
            register_aml_model(
                model_file,
                model_name,
                model_tags,
                exp,
                run_id,
                dataset_id)
        elif (build_uri is None):
            register_aml_model(
                model_file,
                model_name,
                model_tags,
                exp,
                run_id,
                dataset_id,
                build_id)
        else:
            register_aml_model(
                model_file,
                model_name,
                model_tags,
                exp,
                run_id,
                dataset_id,
                build_id,
                build_uri)
    else:
        print("Model not found. Skipping model registration.")
        sys.exit(0)


def model_already_registered(model_name, exp, run_id):
    """
    Method checks if there is already a model with the same name, 
    experiment name, model id, run id etc. If so, then the model is 
    not registered and an exception is raised.
    """
    model_list = AMLModel.list(exp.workspace, name=model_name, run_id=run_id)
    if len(model_list) >= 1:
        e = ("Model name:", model_name, "in workspace",
             exp.workspace, "with run_id ", run_id, "is already registered.")
        print(e)
        raise Exception(e)
    else:
        print("Model is not registered for this run.")


def register_aml_model(
    model_path,
    model_name,
    model_tags,
    exp,
    run_id,
    dataset_id,
    build_id: str = 'none',
    build_uri=None
):
    """
    Method updates the model tags dictionary with some more tags,
    other than just 'mse' associated with the model being registered.

    Next, it checks whether the model has been registered with the 
    same build id or not.  
    """
    try:
        tagsValue = {"area": "diabetes_regression",
                     "run_id": run_id,
                     "experiment_name": exp.name}
        tagsValue.update(model_tags)
        if (build_id != 'none'):
            model_already_registered(model_name, exp, run_id)
            tagsValue["BuildId"] = build_id
            if (build_uri is not None):
                tagsValue["BuildUri"] = build_uri

        model = AMLModel.register(
            workspace=exp.workspace,
            model_name=model_name,
            model_path=model_path,
            tags=tagsValue,
            datasets=[('training data',
                       Dataset.get_by_id(exp.workspace, dataset_id))])
        os.chdir("..")
        print(
            "Model registered: {} \nModel Description: {} "
            "\nModel Version: {}".format(
                model.name, model.description, model.version
            )
        )
    except Exception:
        traceback.print_exc(limit=None, file=None, chain=True)
        print("Model registration failed")
        raise

if __name__ == '__main__':
    main()