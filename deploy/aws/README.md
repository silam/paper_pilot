# Deploying PaperPilot on AWS

Two options, depending on how much you want to show.

## Option A - EKS (Kubernetes, matches `k8s/`)
| Component | AWS service |
|---|---|
| Container registry | ECR (or keep GHCR) |
| API + workers | EKS deployments from `k8s/` |
| Postgres | RDS for PostgreSQL (replace the in-cluster StatefulSet) |
| Redis (cache, Celery broker, online features) | ElastiCache for Redis |
| Vector DB | Chroma StatefulSet on EBS, or Pinecone / OpenSearch Serverless vector |
| Streaming | Amazon MSK (Kafka) **or** Kinesis Data Streams |
| Secrets | Secrets Manager + External Secrets Operator |
| LLM | Anthropic API, or Claude via Amazon Bedrock, or vLLM on a GPU node group |
| Metrics / logs | Amazon Managed Prometheus + Managed Grafana, CloudWatch Logs |
| Raw data + DVC remote | S3 |

```bash
eksctl create cluster --name paperpilot --region us-east-1 --nodes 2 --node-type t3.large
kubectl apply -f k8s/
kubectl -n paperpilot get pods
```

## Option B - cheapest demo: one EC2 box
```bash
# Amazon Linux 2023, t3.large
sudo dnf install -y docker git && sudo systemctl start docker
git clone https://github.com/<you>/paperpilot && cd paperpilot
cp .env.example .env   # add ANTHROPIC_API_KEY, LLM_PROVIDER=anthropic
docker compose up -d --build
```

## Kinesis instead of Kafka
The consumer logic in `app/streaming/kafka_pipeline.py` maps 1:1:
topic -> stream, partition -> shard, consumer group offsets -> KCL checkpoints in DynamoDB.
