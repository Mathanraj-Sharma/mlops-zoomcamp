```bash
pip install zenml zenml[server]
```

```bash
zenml up
```

```bash
zenml experiment-tracker register mlflow_tracker \
    --flavor=mlflow \
    --tracking_uri=http://localhost:5000 \
    --tracking_username=zenml \
    --tracking_password=zenml
```

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./local_artifacts 
```

```bash
 zenml stack register my_stack \
    -o default \
    -a default \
    -e mlflow_tracker
```

```bash
zenml stack set my_stack
```