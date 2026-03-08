from google.cloud import aiplatform

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
        "machine_type": "g2-standard-8",
        "accelerator_type": "NVIDIA_L4",
        "accelerator_count": 1,
    },
    "replica_count": 1,
    "container_spec": {
        "image_uri": "us-central1-docker.pkg.dev/plop-486317/training-repo/gnn-hpo:v3",
        "args": [
            "--project_id", "plop-486317",
            "--dataset_id", "binding_data",
            "--protein_dir", "/gcs/protein-ligand-outcome-prediction/protein-smol",
            "--ligand_dir", "/gcs/protein-ligand-outcome-prediction/ligand-shards/ligands-shards/",
            "--downsample", "25000",
            "--epochs", "2",
            "--lr", "0.0001",
            "--gnn_layers", "2",
            "--atn_layers", "2",
            "--mlp_layers", "2",
            "--atn_protein_heads", "4",
            "--atn_ligand_heads", "4",
            "--node_embed", "128",
            "--edge_embed", "64",
            "--batch_size", "64",
            "--dropout_rate", "0.1"
        ],
    },
}]

job = aiplatform.CustomJob(
    display_name="gnn_training_downsample_25k",
    worker_pool_specs=worker_pool_specs,
)

job.run(
    service_account="637619776534-compute@developer.gserviceaccount.com"
)