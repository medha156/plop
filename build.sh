docker buildx build --platform linux/amd64 \
  -t us-central1-docker.pkg.dev/plop-486317/training-repo/"$1" \
  --push \
  .
