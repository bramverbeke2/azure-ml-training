# Set up your handle to the workspace

# enter details of your AML workspace
subscription_id = "59a62e46-b799-4da2-8314-f56ef5acf82b"
resource_group = "rg-azuremltraining"
workspace = "dummy-workspace"


import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

from azure.ai.ml.entities import Environment

from azure.ai.ml import command
from azure.ai.ml import Input, Output

# importing the Component Package
from azure.ai.ml import load_component


# the dsl decorator tells the sdk that we are defining an Azure ML pipeline
from azure.ai.ml import dsl, Input, Output

import webbrowser
import argparse



# get a handle to the workspace
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)



def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, help="Environment")
    args = parser.parse_args()

    env = args.env

    
    # The URL below is where the data is located, create a Data asset (URI_FILE) named
    # "<yourname>_creditcard_defaults" (so as to avoid conflicts with others)
    # Also tag it with the field "creator" with your name as the value
    web_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"



    credit_data = Data(
        path=web_path,
        type=AssetTypes.URI_FILE,
        description="Credit Data Description",
        name="credit_data",
        version="1"
    )


    # Register your dataset so others can see it!

    ml_client.data.create_or_update(credit_data)


    print(
        f"Dataset with name {credit_data.name} was registered to workspace, the dataset version is {credit_data.version}"
    )
    # See if you can find the Data asset you just created in the Azure ML workspace!



    # Get a reference to our compute cluster!
    # You'll need to know its name. You can list the available computes using the ml_client, or simply go see in the AML studio.
    # This variable should be a string holding the name of the compute cluster.
    cpu_compute_target = "aml-cluster"

    dependencies_dir = "./dependencies"
    os.makedirs(dependencies_dir, exist_ok=True)



    custom_env_name = "aml-scikit-learn"
    # Create an Environment based on the conda yaml specification above
    # As a base image, use "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
    # Name it <your_name>_environment
    # Also tag it with a tag "creator" with your name as a value

    pipeline_job_env = Environment(
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        conda_file="dependencies/conda.yaml",
        name=custom_env_name,
        description="Environment created from a Docker image plus Conda environment.",
    )
    ml_client.environments.create_or_update(pipeline_job_env)



    data_prep_src_dir = "ml/src/components/data_prep"
    os.makedirs(data_prep_src_dir, exist_ok=True)


    data_prep_component = command(
        name="data_prep_credit_defaults",
        display_name="Data preparation for training",
        description="reads a .xl input, split the input to train and test",
        inputs={
            "data": Input(type="uri_folder"),
            "test_train_ratio": Input(type="number"),
        },
        outputs=dict(
            train_data=Output(type="uri_folder", mode="rw_mount"),
            test_data=Output(type="uri_folder", mode="rw_mount"),
        ),
        # The source folder of the component
        code=data_prep_src_dir,
        command="""python data_prep.py \
                --data ${{inputs.data}} --test_train_ratio ${{inputs.test_train_ratio}} \
                --train_data ${{outputs.train_data}} --test_data ${{outputs.test_data}} \
                """,
        environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
    )



    # Now we register the component to the workspace
    data_prep_component = ml_client.create_or_update(data_prep_component.component)

    # Create (register) the component in your workspace
    print(
        f"Component {data_prep_component.name} with Version {data_prep_component.version} is registered"
    )



    train_src_dir = "ml/src/components/train"
    os.makedirs(train_src_dir, exist_ok=True)


    # Loading the component from the yml file
    train_component = load_component(source="ml/src/components/train/train.yml")



    # Now we register the component to the workspace
    train_component = ml_client.create_or_update(train_component)

    # Create (register) the component in your workspace
    print(
        f"Component {train_component.name} with Version {train_component.version} is registered"
    )





    @dsl.pipeline(
        compute=cpu_compute_target,
        description="E2E data_perp-train pipeline",
    )
    def credit_defaults_pipeline(
        pipeline_job_data_input,
        pipeline_job_test_train_ratio,
        pipeline_job_learning_rate,
        pipeline_job_registered_model_name,
    ):
        # using data_prep_function like a python call with its own inputs
        data_prep_job = data_prep_component(
            data=pipeline_job_data_input,
            test_train_ratio=pipeline_job_test_train_ratio,
        )

        train_job = train_component(
            train_data=data_prep_job.outputs.train_data,
            test_data=data_prep_job.outputs.test_data,
            learning_rate=pipeline_job_learning_rate,
            registered_model_name=pipeline_job_registered_model_name
        )

        # Add the data prep and training component

        # a pipeline returns a dictionary of outputs
        # keys will code for the pipeline output identifier
        return {
            "pipeline_job_train_data": data_prep_job.outputs.train_data,
            "pipeline_job_test_data": data_prep_job.outputs.test_data,
        }



    registered_model_name = "credit_defaults_model"

    # Let's instantiate the pipeline with the parameters of our choice
    pipeline = credit_defaults_pipeline(
        pipeline_job_data_input=Input(type="uri_file", path=credit_data.path),
        pipeline_job_test_train_ratio=0.25,
        pipeline_job_learning_rate=0.05,
        pipeline_job_registered_model_name=registered_model_name,
    )



    # submit the pipeline job
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline,
        experiment_name="e2e_registered_components_bram_verbeke",
    )
    # open the pipeline in web browser
    webbrowser.open(pipeline_job.studio_url)




    return env


if __name__ == "__main__":
    main()

