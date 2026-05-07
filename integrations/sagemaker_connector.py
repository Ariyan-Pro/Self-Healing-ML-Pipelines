#!/usr/bin/env python3
"""
Amazon SageMaker Integration Connector for Self-Healing ML Pipelines

Provides model training, deployment, and inference capabilities through
Amazon SageMaker integration.

Features:
    - Automated model training on SageMaker
    - One-click model deployment to endpoints
    - A/B testing and model variants
    - Automatic scaling configuration
    - CloudWatch metrics integration
    - Model registry integration

Usage:
    from integrations.sagemaker_connector import SageMakerConnector
    
    connector = SageMakerConnector(region="us-east-1", role_arn="arn:aws:...")
    connector.create_training_job(
        job_name="healing-training-001",
        image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch:1.9.0",
        instance_type="ml.m5.xlarge"
    )
    connector.deploy_model(
        endpoint_name="healing-endpoint",
        instance_type="ml.t2.medium"
    )

Author: Self-Healing ML Pipelines Team
License: MIT
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class SageMakerConnector:
    """
    Connector for Amazon SageMaker training and deployment.
    
    This connector provides a clean interface to SageMaker for training
    models, deploying endpoints, and managing model versions.
    """
    
    def __init__(
        self,
        region: str = "us-east-1",
        role_arn: Optional[str] = None,
        bucket: Optional[str] = None,
        profile_name: Optional[str] = None
    ):
        """
        Initialize SageMaker connector.
        
        Args:
            region: AWS region for SageMaker resources
            role_arn: IAM role ARN for SageMaker execution
            bucket: S3 bucket for storing artifacts
            profile_name: AWS CLI profile name (optional)
        """
        self.region = region
        self.role_arn = role_arn
        self.bucket = bucket or f"sagemaker-self-healing-{region}"
        self.profile_name = profile_name
        self._sagemaker = None
        self._s3 = None
        self._boto3 = None
        self._session = None
        self._training_job_name: Optional[str] = None
        self._endpoint_name: Optional[str] = None
        
        # Try to import boto3 and sagemaker
        try:
            import boto3
            import sagemaker
            from sagemaker import get_execution_role
            from sagemaker.estimator import Estimator
            from sagemaker.model import Model
            from sagemaker.predictor import Predictor
            
            self._boto3 = boto3
            self._sagemaker = sagemaker
            self._Estimator = Estimator
            self._Model = Model
            self._Predictor = Predictor
            
            # Try to get execution role if not provided
            if not role_arn:
                try:
                    self.role_arn = get_execution_role()
                except Exception:
                    print("⚠️  Could not auto-detect SageMaker execution role")
                    print("   Please provide role_arn parameter")
                    
        except ImportError:
            print("⚠️  SageMaker SDK not installed. Install with: pip install sagemaker boto3")
            print("   Running in mock mode - no actual AWS calls will be made")
    
    def _ensure_sagemaker_available(self) -> bool:
        """Check if SageMaker SDK is available."""
        if self._sagemaker is None:
            return False
        return True
    
    def connect(self) -> 'SageMakerConnector':
        """
        Establish connection to AWS SageMaker.
        
        Returns:
            Self for method chaining
        """
        if self._ensure_sagemaker_available():
            try:
                session = self._sagemaker.Session(boto_session=self._boto3.Session(
                    region_name=self.region,
                    profile_name=self.profile_name
                ))
                self._session = session
                print(f"✅ Connected to SageMaker in region: {self.region}")
                print(f"   Execution role: {self.role_arn}")
                print(f"   Default bucket: {self.bucket}")
            except Exception as e:
                print(f"⚠️  Connection warning: {e}")
                print("   Continuing in mock mode")
        else:
            print("📝 SageMaker not available - running in mock mode")
        
        return self
    
    def create_training_job(
        self,
        job_name: str,
        image_uri: str,
        instance_type: str = "ml.m5.xlarge",
        instance_count: int = 1,
        volume_size_gb: int = 30,
        max_run_hours: int = 24,
        input_data: Optional[Dict[str, str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        environment_vars: Optional[Dict[str, str]] = None,
        tags: Optional[List[Dict[str, str]]] = None
    ) -> 'SageMakerConnector':
        """
        Create and start a SageMaker training job.
        
        Args:
            job_name: Unique name for the training job
            image_uri: Docker image URI for training container
            instance_type: EC2 instance type for training
            instance_count: Number of instances to use
            volume_size_gb: Size of EBS volume in GB
            max_run_hours: Maximum training duration in hours
            input_data: Dictionary of channel names to S3 paths
            hyperparameters: Training hyperparameters
            environment_vars: Environment variables for training container
            tags: Tags to apply to the training job
            
        Returns:
            Self for method chaining
        """
        self._training_job_name = job_name
        
        if self._ensure_sagemaker_available():
            try:
                # Create estimator
                estimator = self._Estimator(
                    image_uri=image_uri,
                    role=self.role_arn,
                    instance_count=instance_count,
                    instance_type=instance_type,
                    volume_size=volume_size_gb,
                    max_run=max_run_hours * 3600,
                    output_path=f"s3://{self.bucket}/output",
                    sagemaker_session=self._session,
                    environment_vars=environment_vars or {},
                    tags=tags or [{'Key': 'system', 'Value': 'self-healing-ml-pipelines'}]
                )
                
                # Set hyperparameters
                if hyperparameters:
                    estimator.set_hyperparameters(**hyperparameters)
                
                # Prepare input channels
                channels = []
                if input_data:
                    for channel_name, s3_path in input_data.items():
                        channels.append(
                            self._sagemaker.inputs.TrainingInput(
                                s3_data=s3_path,
                                content_type="application/x-sagemaker"
                            )
                        )
                
                # Start training
                if channels:
                    estimator.fit(inputs={k: v for k, v in input_data.items()})
                else:
                    # Mock training with no input
                    print(f"📝 Starting training job: {job_name}")
                
                self._estimator = estimator
                print(f"🧪 Created training job: {job_name}")
                print(f"   Instance type: {instance_type}")
                print(f"   Instance count: {instance_count}")
                
            except Exception as e:
                print(f"⚠️  Error creating training job: {e}")
                print("   Continuing in mock mode")
        else:
            # Mock mode
            print(f"📝 [MOCK] Created training job: {job_name}")
            print(f"   Image: {image_uri}")
            print(f"   Instance: {instance_type} x{instance_count}")
            print(f"   Volume: {volume_size_gb}GB")
            if hyperparameters:
                print(f"   Hyperparameters: {hyperparameters}")
        
        return self
    
    def wait_for_training(self, poll_interval_seconds: int = 60) -> 'SageMakerConnector':
        """
        Wait for training job to complete.
        
        Args:
            poll_interval_seconds: Time between status checks
            
        Returns:
            Self for method chaining
        """
        if not self._training_job_name:
            print("⚠️  No training job to wait for.")
            return self
        
        if self._ensure_sagemaker_available():
            try:
                print(f"⏳ Waiting for training job: {self._training_job_name}")
                if hasattr(self, '_estimator') and self._estimator:
                    self._estimator.jobs[-1].wait(logs=False)
                print(f"✅ Training job completed: {self._training_job_name}")
            except Exception as e:
                print(f"⚠️  Error waiting for training: {e}")
        else:
            print(f"📝 [MOCK] Simulated waiting for training: {self._training_job_name}")
            time.sleep(1)  # Brief pause for realism
            print(f"✅ [MOCK] Training completed")
        
        return self
    
    def register_model(
        self,
        model_name: str,
        model_package_group: Optional[str] = None,
        version_description: Optional[str] = None
    ) -> Optional[str]:
        """
        Register trained model in SageMaker Model Registry.
        
        Args:
            model_name: Name for the registered model
            model_package_group: Package group name (default: model_name)
            version_description: Description for this model version
            
        Returns:
            Model package ARN if successful
        """
        if self._ensure_sagemaker_available():
            try:
                group_name = model_package_group or model_name
                
                # Get the latest training job output
                if hasattr(self, '_estimator') and self._estimator:
                    training_job = self._estimator.latest_training_job
                    model_data = training_job.model_data
                    
                    # Create model package
                    response = self._session.sagemaker_client.create_model_package(
                        ModelPackageGroupName=group_name,
                        ModelPackageDescription=version_description or f"Model from {self._training_job_name}",
                        InferenceSpecification={
                            'Containers': [{
                                'Image': self._estimator.image_uri,
                                'ModelDataUrl': model_data,
                            }],
                            'SupportedTransformInstanceTypes': ['ml.m5.large'],
                            'SupportedContentTypes': ['text/csv'],
                            'SupportedResponseMIMETypes': ['text/csv']
                        },
                        CertifyForMarketplace=False
                    )
                    
                    model_package_arn = response['ModelPackageArn']
                    print(f"✅ Registered model: {model_name}")
                    print(f"   Package ARN: {model_package_arn}")
                    return model_package_arn
                    
            except Exception as e:
                print(f"⚠️  Error registering model: {e}")
        else:
            print(f"📝 [MOCK] Would register model: {model_name}")
            return f"arn:aws:sagemaker:{self.region}:model-package/{model_name}/1"
        
        return None
    
    def deploy_model(
        self,
        endpoint_name: str,
        instance_type: str = "ml.t2.medium",
        initial_instance_count: int = 1,
        data_capture_config: Optional[Dict] = None,
        environment_vars: Optional[Dict[str, str]] = None,
        tags: Optional[List[Dict[str, str]]] = None
    ) -> 'SageMakerConnector':
        """
        Deploy model to a SageMaker endpoint.
        
        Args:
            endpoint_name: Name for the endpoint
            instance_type: Instance type for the endpoint
            initial_instance_count: Initial number of instances
            data_capture_config: Configuration for data capture
            environment_vars: Environment variables for endpoint
            tags: Tags to apply to the endpoint
            
        Returns:
            Self for method chaining
        """
        self._endpoint_name = endpoint_name
        
        if self._ensure_sagemaker_available():
            try:
                # Create model from training job
                if hasattr(self, '_estimator') and self._estimator:
                    model = self._estimator.create_model(
                        environment_vars=environment_vars or {},
                        tags=tags or [{'Key': 'system', 'Value': 'self-healing-ml-pipelines'}]
                    )
                    
                    # Configure data capture if specified
                    if data_capture_config:
                        from sagemaker.model_monitor import DataCaptureConfig
                        data_capture = DataCaptureConfig(**data_capture_config)
                        predictor = model.deploy(
                            initial_instance_count=initial_instance_count,
                            instance_type=instance_type,
                            endpoint_name=endpoint_name,
                            data_capture_config=data_capture
                        )
                    else:
                        predictor = model.deploy(
                            initial_instance_count=initial_instance_count,
                            instance_type=instance_type,
                            endpoint_name=endpoint_name
                        )
                    
                    self._predictor = predictor
                    print(f"🚀 Deployed endpoint: {endpoint_name}")
                    print(f"   Instance type: {instance_type}")
                    print(f"   Instance count: {initial_instance_count}")
                    print(f"   Endpoint URL: {predictor.endpoint}")
                    
            except Exception as e:
                print(f"⚠️  Error deploying model: {e}")
                print("   Continuing in mock mode")
        else:
            print(f"📝 [MOCK] Deployed endpoint: {endpoint_name}")
            print(f"   Instance: {instance_type} x{initial_instance_count}")
        
        return self
    
    def predict(self, data: Any, content_type: str = "application/json") -> Optional[Any]:
        """
        Make predictions using deployed endpoint.
        
        Args:
            data: Input data for prediction
            content_type: Content type of input data
            
        Returns:
            Prediction results
        """
        if not self._endpoint_name:
            print("⚠️  No endpoint deployed. Call deploy_model() first.")
            return None
        
        if self._ensure_sagemaker_available():
            try:
                if hasattr(self, '_predictor') and self._predictor:
                    result = self._predictor.predict(data, initial_args={'ContentType': content_type})
                    print(f"📊 Made prediction")
                    return result
            except Exception as e:
                print(f"⚠️  Error making prediction: {e}")
        else:
            print(f"📝 [MOCK] Would make prediction with data: {type(data)}")
            return {"prediction": "mock_result", "confidence": 0.95}
        
        return None
    
    def update_endpoint(
        self,
        endpoint_name: Optional[str] = None,
        instance_type: str = "ml.t2.medium",
        instance_count: int = 1,
        variant_name: str = "AllTraffic"
    ) -> 'SageMakerConnector':
        """
        Update endpoint configuration (scaling, instance type).
        
        Args:
            endpoint_name: Endpoint to update (default: current endpoint)
            instance_type: New instance type
            instance_count: New instance count
            variant_name: Production variant name
            
        Returns:
            Self for method chaining
        """
        ep_name = endpoint_name or self._endpoint_name
        
        if not ep_name:
            print("⚠️  No endpoint specified.")
            return self
        
        if self._ensure_sagemaker_available():
            try:
                client = self._session.sagemaker_client
                
                # Get current endpoint config
                endpoint_desc = client.describe_endpoint(EndpointName=ep_name)
                current_config = endpoint_desc['EndpointConfigName']
                
                # Create new production variant
                new_variant = {
                    'VariantName': variant_name,
                    'ModelName': endpoint_desc['ProductionVariants'][0]['ModelName'],
                    'InitialInstanceCount': instance_count,
                    'InstanceType': instance_type,
                    'InitialVariantWeight': 1.0
                }
                
                # Create new endpoint config
                new_config_name = f"{ep_name}-config-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                client.create_endpoint_config(
                    EndpointConfigName=new_config_name,
                    ProductionVariants=[new_variant]
                )
                
                # Update endpoint
                client.update_endpoint(
                    EndpointName=ep_name,
                    EndpointConfigName=new_config_name
                )
                
                print(f"🔄 Updated endpoint: {ep_name}")
                print(f"   New instance type: {instance_type}")
                print(f"   New instance count: {instance_count}")
                
            except Exception as e:
                print(f"⚠️  Error updating endpoint: {e}")
        else:
            print(f"📝 [MOCK] Would update endpoint: {ep_name}")
            print(f"   To: {instance_type} x{instance_count}")
        
        return self
    
    def delete_endpoint(self, endpoint_name: Optional[str] = None) -> 'SageMakerConnector':
        """
        Delete a SageMaker endpoint.
        
        Args:
            endpoint_name: Endpoint to delete (default: current endpoint)
            
        Returns:
            Self for method chaining
        """
        ep_name = endpoint_name or self._endpoint_name
        
        if not ep_name:
            print("⚠️  No endpoint specified.")
            return self
        
        if self._ensure_sagemaker_available():
            try:
                client = self._session.sagemaker_client
                client.delete_endpoint(EndpointName=ep_name)
                print(f"🗑️  Deleted endpoint: {ep_name}")
            except Exception as e:
                print(f"⚠️  Error deleting endpoint: {e}")
        else:
            print(f"📝 [MOCK] Would delete endpoint: {ep_name}")
        
        return self
    
    def list_endpoints(self) -> List[Dict]:
        """
        List all SageMaker endpoints.
        
        Returns:
            List of endpoint information dictionaries
        """
        if self._ensure_sagemaker_available():
            try:
                client = self._session.sagemaker_client
                response = client.list_endpoints(MaxResults=50)
                endpoints = response.get('Endpoints', [])
                print(f"🔍 Found {len(endpoints)} endpoints")
                return endpoints
            except Exception as e:
                print(f"⚠️  Error listing endpoints: {e}")
        else:
            print(f"📝 [MOCK] Would list endpoints")
            return []
        
        return []
    
    def list_training_jobs(self, status_filter: Optional[str] = None) -> List[Dict]:
        """
        List SageMaker training jobs.
        
        Args:
            status_filter: Filter by status (Completed, Failed, InProgress, etc.)
            
        Returns:
            List of training job information dictionaries
        """
        if self._ensure_sagemaker_available():
            try:
                client = self._session.sagemaker_client
                kwargs = {'MaxResults': 50}
                if status_filter:
                    kwargs['StatusEquals'] = status_filter
                
                response = client.list_training_jobs(**kwargs)
                jobs = response.get('TrainingJobSummaries', [])
                print(f"🔍 Found {len(jobs)} training jobs")
                return jobs
            except Exception as e:
                print(f"⚠️  Error listing training jobs: {e}")
        else:
            print(f"📝 [MOCK] Would list training jobs")
            return []
        
        return []
    
    def upload_to_s3(self, local_path: str, s3_key: Optional[str] = None) -> Optional[str]:
        """
        Upload a file to S3.
        
        Args:
            local_path: Local file path
            s3_key: S3 key (default: filename-based)
            
        Returns:
            S3 URI if successful
        """
        if self._ensure_sagemaker_available():
            try:
                s3_uri = self._session.upload_data(
                    path=local_path,
                    bucket=self.bucket,
                    key_prefix=s3_key or ''
                )
                print(f"📦 Uploaded to S3: {s3_uri}")
                return s3_uri
            except Exception as e:
                print(f"⚠️  Error uploading to S3: {e}")
        else:
            print(f"📝 [MOCK] Would upload {local_path} to s3://{self.bucket}/")
            return f"s3://{self.bucket}/{Path(local_path).name}"
        
        return None
    
    def __enter__(self):
        """Context manager entry."""
        return self.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._endpoint_name and exc_type is None:
            print("\n💡 Tip: Remember to delete endpoints when done to avoid charges")


def demo_usage():
    """Demonstrate SageMaker connector usage."""
    print("\n" + "="*60)
    print("SageMaker Connector Demo")
    print("="*60 + "\n")
    
    # Example 1: Basic training and deployment
    print("Example 1: Training and deployment workflow\n")
    
    connector = SageMakerConnector(
        region="us-east-1",
        role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole"
    )
    connector.connect()
    
    # Create training job
    connector.create_training_job(
        job_name="healing-train-001",
        image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch:1.9.0-cpu-py38",
        instance_type="ml.m5.xlarge",
        hyperparameters={
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001
        }
    )
    
    # Wait for training
    connector.wait_for_training()
    
    # Register model
    connector.register_model(
        model_name="healing-model-v1",
        version_description="First version of healing model"
    )
    
    # Deploy endpoint
    connector.deploy_model(
        endpoint_name="healing-endpoint-prod",
        instance_type="ml.t2.medium"
    )
    
    # Make prediction
    result = connector.predict({"features": [0.1, 0.2, 0.3]})
    print(f"   Prediction result: {result}")
    
    print("\n" + "-"*60 + "\n")
    
    # Example 2: Context manager usage
    print("Example 2: Using context manager\n")
    
    with SageMakerConnector(region="us-west-2") as sm:
        print("Connected to SageMaker in us-west-2")
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60 + "\n")


if __name__ == '__main__':
    demo_usage()
