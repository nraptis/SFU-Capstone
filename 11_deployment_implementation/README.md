# BloodyFlask

This project serves a blood-cell inference UI with Flask using pre-trained model weights from `ml/state_dicts`.

## Notebook

`example_usage.ipynb` is the inference-only walkthrough notebook.  
It demonstrates how to:

- load holdout data from `hold_out_test_data`
- reuse project pipeline code from `ml/` (preprocess, classify, pretty print)
- run evaluation/stat summaries on holdout samples
- inspect concrete prediction examples with side-by-side original vs normalized images

The notebook does not train models; it only runs inference with existing weights.

## Deployment

The app is deployed on AWS using:

- Elastic Beanstalk (application/environment management)
- Application Load Balancer (traffic entry)
- Route 53 (DNS routing)
- one EC2 instance currently (single-instance runtime)

The EC2 layer can scale up by changing Elastic Beanstalk capacity settings (instance count and size) as traffic grows.

![](https://raw.githubusercontent.com/nraptis/BloodyFlask/refs/heads/main/you_might_be_super_sick.jpg)

## Docker / Kubernetes Next Steps

### Deploy with Docker

1. Build image:
   `docker build -t bloodyflask:latest .`
2. Run locally:
   `docker run --rm -p 8080:8080 bloodyflask:latest`
3. Tag and push to a registry (example: ECR, Docker Hub):
   `docker tag bloodyflask:latest <registry>/<repo>:<tag>`
   `docker push <registry>/<repo>:<tag>`
4. Point Elastic Beanstalk (Docker platform) to that image tag.

### Move to Kubernetes (basic path)

1. Create an image and push to a registry.
2. Create Kubernetes manifests for:
   - `Deployment` (replicas + container image)
   - `Service` (ClusterIP or LoadBalancer)
   - `Ingress` (optional, for host/path routing)
3. Add env vars/secrets and resource requests/limits.
4. Apply manifests:
   `kubectl apply -f k8s/`
5. Scale replicas as needed:
   `kubectl scale deployment bloodyflask --replicas=3`
