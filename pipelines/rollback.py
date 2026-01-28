"""Model rollback pipeline for self-healing ML system."""
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import joblib

logger = logging.getLogger(__name__)


class RollbackPipeline:
    """Manages model versioning and rollback operations."""
    
    def __init__(self):
        """Initialize rollback pipeline."""
        self.models_dir = Path("models")
        self.checkpoints_dir = self.models_dir / "checkpoints"
        self.registry_dir = self.models_dir / "registry"
        
        # Ensure directories exist
        self.models_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.registry_dir.mkdir(exist_ok=True)
        
        logger.info("RollbackPipeline initialized")
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available model versions.
        
        Returns:
            List of model metadata dictionaries
        """
        models = []
        
        # Find all model files
        model_files = list(self.checkpoints_dir.glob("*.joblib"))
        
        for model_file in model_files:
            model_name = model_file.stem
            metadata_file = self.registry_dir / f"{model_name}.json"
            
            metadata = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {model_name}: {e}")
            
            models.append({
                "version": model_name,
                "model_path": str(model_file),
                "metadata_path": str(metadata_file),
                "exists": model_file.exists(),
                "metadata": metadata,
                "last_modified": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
            })
        
        # Sort by modification time (newest first)
        models.sort(key=lambda x: x["last_modified"], reverse=True)
        
        return models
    
    def get_current_model(self) -> Optional[Dict[str, Any]]:
        """
        Get current active model information.
        
        Returns:
            Current model metadata or None
        """
        current_model_path = self.models_dir / "current_model.joblib"
        current_metadata_path = self.models_dir / "current_metadata.json"
        
        if not current_model_path.exists():
            return None
        
        metadata = {}
        if current_metadata_path.exists():
            try:
                with open(current_metadata_path, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load current model metadata: {e}")
        
        return {
            "version": metadata.get("version", "unknown"),
            "model_path": str(current_model_path),
            "metadata": metadata,
            "last_modified": datetime.fromtimestamp(
                current_model_path.stat().st_mtime
            ).isoformat() if current_model_path.exists() else None
        }
    
    def rollback_to_version(
        self,
        target_version: str,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Rollback to a specific model version.
        
        Args:
            target_version: Target model version name
            validate: Whether to validate the model before activation
            
        Returns:
            Dictionary with rollback results
        """
        logger.info(f"Attempting rollback to version: {target_version}")
        
        try:
            # Check if target version exists
            target_model_path = self.checkpoints_dir / f"{target_version}.joblib"
            target_metadata_path = self.registry_dir / f"{target_version}.json"
            
            if not target_model_path.exists():
                return {
                    "status": "failed",
                    "error": f"Model version {target_version} not found",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Load target model for validation
            if validate:
                try:
                    target_model = joblib.load(target_model_path)
                    logger.debug(f"Successfully loaded model {target_version} for validation")
                except Exception as e:
                    return {
                        "status": "failed",
                        "error": f"Failed to load model {target_version}: {e}",
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Load target metadata
            target_metadata = {}
            if target_metadata_path.exists():
                try:
                    with open(target_metadata_path, 'r') as f:
                        target_metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {target_version}: {e}")
            
            # Activate target model
            current_model_path = self.models_dir / "current_model.joblib"
            current_metadata_path = self.models_dir / "current_metadata.json"
            
            # Copy model
            import shutil
            shutil.copy2(target_model_path, current_model_path)
            
            # Save metadata
            with open(current_metadata_path, 'w') as f:
                json.dump(target_metadata, f, indent=2)
            
            result = {
                "status": "success",
                "action": "rollback",
                "timestamp": datetime.now().isoformat(),
                "from_version": self.get_current_model()["version"] 
                if self.get_current_model() else "unknown",
                "to_version": target_version,
                "model_path": str(target_model_path),
                "metadata": target_metadata,
                "message": f"Successfully rolled back to version {target_version}"
            }
            
            logger.info(f"Rollback successful to {target_version}")
            
            return result
            
        except Exception as e:
            error_msg = f"Rollback failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "status": "failed",
                "action": "rollback",
                "timestamp": datetime.now().isoformat(),
                "error": error_msg
            }
    
    def rollback_to_previous(
        self,
        skip_versions: int = 1,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Rollback to a previous version.
        
        Args:
            skip_versions: Number of versions to skip back
            validate: Whether to validate the model before activation
            
        Returns:
            Dictionary with rollback results
        """
        # Get available models sorted by date (newest first)
        available_models = self.list_available_models()
        
        if len(available_models) <= skip_versions:
            return {
                "status": "failed",
                "error": f"Cannot rollback {skip_versions} versions, "
                        f"only {len(available_models)} available",
                "timestamp": datetime.now().isoformat()
            }
        
        # Skip the specified number of versions
        target_version = available_models[skip_versions]["version"]
        
        return self.rollback_to_version(target_version, validate)
    
    def cleanup_old_models(
        self,
        keep_last_n: int = 5
    ) -> Dict[str, Any]:
        """
        Clean up old model versions, keeping only the most recent N.
        
        Args:
            keep_last_n: Number of most recent models to keep
            
        Returns:
            Dictionary with cleanup results
        """
        logger.info(f"Cleaning up old models, keeping last {keep_last_n}")
        
        try:
            # Get all models sorted by modification time
            all_models = self.list_available_models()
            
            if len(all_models) <= keep_last_n:
                return {
                    "status": "skipped",
                    "message": f"Only {len(all_models)} models found, "
                              f"keeping all (threshold: {keep_last_n})",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Identify models to delete
            models_to_keep = all_models[:keep_last_n]
            models_to_delete = all_models[keep_last_n:]
            
            current_model = self.get_current_model()
            current_version = current_model["version"] if current_model else None
            
            deleted = []
            errors = []
            
            # Delete old models (skip current model)
            for model_info in models_to_delete:
                if model_info["version"] == current_version:
                    logger.info(f"Skipping current model {current_version}")
                    continue
                
                try:
                    # Delete model file
                    model_path = Path(model_info["model_path"])
                    if model_path.exists():
                        model_path.unlink()
                    
                    # Delete metadata file
                    metadata_path = Path(model_info["metadata_path"])
                    if metadata_path.exists():
                        metadata_path.unlink()
                    
                    deleted.append(model_info["version"])
                    logger.debug(f"Deleted model: {model_info['version']}")
                    
                except Exception as e:
                    errors.append({
                        "version": model_info["version"],
                        "error": str(e)
                    })
                    logger.warning(f"Failed to delete model {model_info['version']}: {e}")
            
            result = {
                "status": "success" if not errors else "partial",
                "action": "cleanup",
                "timestamp": datetime.now().isoformat(),
                "kept_models": [m["version"] for m in models_to_keep],
                "deleted_models": deleted,
                "total_deleted": len(deleted),
                "errors": errors if errors else None
            }
            
            logger.info(f"Cleanup completed: kept {len(models_to_keep)}, deleted {len(deleted)}")
            
            return result
            
        except Exception as e:
            error_msg = f"Cleanup failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "status": "failed",
                "action": "cleanup",
                "timestamp": datetime.now().isoformat(),
                "error": error_msg
            }