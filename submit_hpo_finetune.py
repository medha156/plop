import sys

import yaml
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

def convert_yaml_to_spec(yaml_params):
    """Converts a dictionary from YAML into Vertex AI Spec objects."""
    spec_map = {}
    
    for config in yaml_params:
        param_id = config["parameterId"]
        # Identify the type and map it to the SDK class
        if 'doubleValueSpec' in config:
            spec_map[param_id] = hpt.DoubleParameterSpec(
                min=float(config['doubleValueSpec']['minValue']),
                max=float(config['doubleValueSpec']['maxValue']),
                scale=config.get('scale_type', 'linear').lower().replace('unit_', '').replace('_scale', '')
            )
        elif 'integerValueSpec' in config:
            spec_map[param_id] = hpt.IntegerParameterSpec(
                min=int(config['integerValueSpec']['minValue']),
                max=int(config['integerValueSpec']['maxValue']),
                scale=config.get('scale_type', 'linear').lower().replace('unit_', '').replace('_scale', '')
            )
        elif 'categoricalValueSpec' in config:
            spec_map[param_id] = hpt.CategoricalParameterSpec(
                values=config['categoricalValueSpec']['values']
            )
        elif 'discreteValueSpec' in config:
            spec_map[param_id] = hpt.DiscreteParameterSpec(
                values=[float(val) for val in config['discreteValueSpec']['values']],
                scale=config.get('scale_type', 'linear').lower().replace('unit_', '').replace('_scale', '')
            )
        else:
            raise ValueError("Bad data: " + str(config))
    return spec_map

# Load your existing YAML file
with open("hp_finetune_config.yml", "r") as f:
    config = yaml.safe_load(f)

aiplatform.init(
    project="plop-486317",
    location="us-central1",
    staging_bucket="gs://protein-ligand-outcome-prediction"
)

# Extract specs directly from the YAML dictionary
study_spec = config['studySpec']
max_trial_count = config['maxTrialCount']
parallel_trial_count = config['parallelTrialCount']

metric_spec = {
    m['metricId']: m['goal'].lower() 
    for m in study_spec['metrics']
}

parameter_spec = convert_yaml_to_spec(study_spec['parameters'])

# Define your worker pool (Hardware)
worker_pool_specs = [{
    "machine_spec": {
        "machine_type": "g2-standard-16",
        "accelerator_type": "NVIDIA_L4",
        "accelerator_count": 1,
    },
    "replica_count": 1,
    "container_spec": {
        "image_uri": "us-central1-docker.pkg.dev/plop-486317/training-repo/" + sys.argv[1],
        "args": [
            "--project_id", "plop-486317",
            "--dataset_id", "binding_data",
            "--protein_dir", "/gcs/protein-ligand-outcome-prediction/proteins_smol/proteins-{000000..000011}.tar",
            "--ligand_dir", "/gcs/protein-ligand-outcome-prediction/shardmmen/ligands-{000000..000032}.tar",
            "--downsample", "1000000",
            "--epochs", "5"
        ]
    },
}]

# Create the Job using the YAML data
hpo_job = aiplatform.HyperparameterTuningJob(
    display_name="Finetuning",
    custom_job=aiplatform.CustomJob(
        display_name="finetune_job",
        worker_pool_specs=worker_pool_specs
    ),
    metric_spec=metric_spec,
    parameter_spec=parameter_spec,
    max_trial_count=max_trial_count,
    parallel_trial_count=parallel_trial_count
)

hpo_job.run(service_account="637619776534-compute@developer.gserviceaccount.com")