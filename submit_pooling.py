from google.cloud import aiplatform
import sys
import copy

PROJECT_ID = "plop-486317"
REGION = "us-central1"
BUCKET = "gs://protein-ligand-outcome-prediction"

aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=BUCKET
)

worker_pool_specs_default = [{
    "machine_spec": {
        "machine_type": "g2-standard-32",
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
        ],
    },
}]

specs = {
    'mean': copy.deepcopy(worker_pool_specs_default),
    'sum': copy.deepcopy(worker_pool_specs_default),
}
specs['mean'][0]["container_spec"]["args"] += ["--pool", "mean", "--model_dir", "/gcs/protein-ligand-outcome-prediction/mean_pool"]
specs['sum'][0]["container_spec"]["args"] += ["--pool", "sum", "--model_dir", "/gcs/protein-ligand-outcome-prediction/sum_pool"]

for layer, spec in specs.items():
    job = aiplatform.CustomJob(
        display_name=f"pooling_{layer}",
        worker_pool_specs=spec,
    )

    job.run(
        service_account="637619776534-compute@developer.gserviceaccount.com",
        sync=False
    )
