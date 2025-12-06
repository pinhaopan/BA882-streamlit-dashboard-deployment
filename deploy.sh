gcloud config set project baratz00-ba882-fall25

echo "======================================================"
echo "build (no cache)"
echo "======================================================"

docker build --no-cache -t gcr.io/baratz00-ba882-fall25/streamlit-poc .

echo "======================================================"
echo "push"
echo "======================================================"

docker push gcr.io/baratz00-ba882-fall25/streamlit-poc

echo "======================================================"
echo "deploy run"
echo "======================================================"

gcloud run deploy streamlit-poc \
    --image gcr.io/baratz00-ba882-fall25/streamlit-poc \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --service-account cfb-pipeline-sa@baratz00-ba882-fall25.iam.gserviceaccount.com \
    --memory 1Gi