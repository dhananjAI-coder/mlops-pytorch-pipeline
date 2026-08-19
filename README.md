
# MLOps PyTorch Pipeline

End-to-end deployment of a PyTorch CIFAR-10 image classification workload using Docker and Kubernetes.

The project demonstrates the complete ML deployment lifecycle:

- PyTorch model training
- Configuration through YAML and Kubernetes ConfigMaps
- Docker containerization
- Kubernetes Job-based training
- Persistent storage for datasets and model checkpoints
- Kubernetes model serving
- Health and readiness probes
- ClusterIP service
- Horizontal Pod Autoscaling
- End-to-end prediction validation
- Git feature branches and Pull Requests

---

## Architecture

```mermaid
flowchart TD
    A[CIFAR-10 Dataset] --> B[Kubernetes Training Job]

    C[ConfigMap<br/>training-config] --> B
    D[data-pvc] --> B
    B --> E[checkpoint-pvc]

    E --> F[Model Serving Deployment]

    F --> G[Replica 1]
    F --> H[Replica 2]

    G --> I[ClusterIP Service<br/>port 80 -> 8080]
    H --> I

    I --> J[POST /predict]

    K[HPA<br/>CPU target 70%<br/>min 2 / max 5] --> F

    F --> L[GET /health]
