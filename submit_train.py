from google.cloud import aiplatform
import sys

PROJECT_ID = "plop-486317"
REGION = "us-central1"
BUCKET = "gs://protein-ligand-outcome-prediction"

aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=BUCKET
)

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
        ],
    },
}]

job = aiplatform.CustomJob(
    display_name="gnn_training_downsample_50k",
    worker_pool_specs=worker_pool_specs,
)

job.run(
    service_account="637619776534-compute@developer.gserviceaccount.com"
)
