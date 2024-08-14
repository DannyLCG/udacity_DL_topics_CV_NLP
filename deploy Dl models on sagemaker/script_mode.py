from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role

role = get_execution_role()
hyperparameters = {"epochs": "2", "batch-size": "32", "test-batch-size": "100", "lr": "0.01"}

estimator = PyTorch(
    entry_point="scripts/pytorch_cifar.py",
    base_job_name="sagemaker-script-mode",
    role=role,
    instance_type="ml.m5.large",
    instane_count=1,
    hyperparameters=hyperparameters,
    framework_version="1.8", #Pytorch version List of supported versions: https://github.com/aws/deep-learning-containers/blob/master/available_images.md.
    py_version="py36"
)

# Train the model
estimator.fit(wait=True)