from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import uvicorn
from datetime import datetime, timedelta
import random
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import joblib
import io
import os
from enum import Enum
import uuid
import asyncio

app = FastAPI(title="ML Training Dashboard API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enums
class ModelType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

class TrainingStatus(str, Enum):
    PENDING = "pending"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"

class ModelAlgorithm(str, Enum):
    RANDOM_FOREST = "random_forest"
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_REGRESSION = "linear_regression"
    SVM = "svm"

# Data Models
class User(BaseModel):
    id: int
    name: str
    email: str
    status: str

class Dataset(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    file_path: str
    columns: List[str]
    rows: int
    created_at: datetime
    target_column: Optional[str] = None

class TrainingRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    dataset_id: str
    model_name: str
    algorithm: ModelAlgorithm
    model_type: ModelType
    target_column: str
    feature_columns: List[str]
    test_size: float = 0.2
    hyperparameters: Optional[Dict[str, Any]] = None

class TrainingJob(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    id: str
    model_name: str
    dataset_id: str
    algorithm: ModelAlgorithm
    model_type: ModelType
    status: TrainingStatus
    progress: float = 0.0
    created_at: datetime
    completed_at: Optional[datetime] = None
    metrics: Optional[Dict[str, Any]] = None
    model_path: Optional[str] = None
    error_message: Optional[str] = None

class ModelInfo(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    id: str
    name: str
    algorithm: ModelAlgorithm
    model_type: ModelType
    dataset_used: str
    metrics: Dict[str, Any]
    created_at: datetime
    model_path: str
    feature_columns: List[str]
    target_column: str

class PredictionRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    model_id: str
    data: Dict[str, Any]

class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    prediction: Union[float, int, str]
    confidence: Optional[float] = None
    model_used: str

# Global storage
users_db = [
    User(id=1, name="John Doe", email="john@example.com", status="active"),
    User(id=2, name="Jane Smith", email="jane@example.com", status="inactive"),
    User(id=3, name="Bob Johnson", email="bob@example.com", status="active"),
    User(id=4, name="Alice Brown", email="alice@example.com", status="active"),
]

datasets_db: Dict[str, Dataset] = {}
training_jobs_db: Dict[str, TrainingJob] = {}
models_db: Dict[str, ModelInfo] = {}

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ML Training Agent
class MLTrainingAgent:
    def __init__(self):
        self.training_queue = []
        self.active_trainings = {}
    
    async def train_model(self, job_id: str, request: TrainingRequest):
        """Background task to train a model"""
        job = training_jobs_db[job_id]
        
        try:
            job.status = TrainingStatus.TRAINING
            job.progress = 0.1
            
            # Load dataset
            dataset = datasets_db[request.dataset_id]
            df = pd.read_csv(dataset.file_path)
            
            job.progress = 0.2
            
            # Prepare features and target
            X = df[request.feature_columns]
            y = df[request.target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=request.test_size, random_state=42
            )
            
            job.progress = 0.4
            
            # Initialize model based on algorithm and type
            model = self._get_model(request.algorithm, request.model_type, request.hyperparameters)
            
            job.progress = 0.6
            
            # Train model
            model.fit(X_train, y_train)
            
            job.progress = 0.8
            
            # Make predictions and calculate metrics
            y_pred = model.predict(X_test)
            metrics = self._calculate_metrics(y_test, y_pred, request.model_type)
            
            # Save model
            model_path = f"models/{job_id}.joblib"
            joblib.dump(model, model_path)
            
            job.progress = 1.0
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.now()
            job.metrics = metrics
            job.model_path = model_path
            
            # Create model entry
            model_info = ModelInfo(
                id=job_id,
                name=request.model_name,
                algorithm=request.algorithm,
                model_type=request.model_type,
                dataset_used=dataset.name,
                metrics=metrics,
                created_at=job.created_at,
                model_path=model_path,
                feature_columns=request.feature_columns,
                target_column=request.target_column
            )
            models_db[job_id] = model_info
            
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
    
    def _get_model(self, algorithm: ModelAlgorithm, model_type: ModelType, hyperparameters: Dict = None):
        """Get the appropriate model based on algorithm and type"""
        if hyperparameters is None:
            hyperparameters = {}
        
        if algorithm == ModelAlgorithm.RANDOM_FOREST:
            if model_type == ModelType.CLASSIFICATION:
                return RandomForestClassifier(**hyperparameters)
            else:
                return RandomForestRegressor(**hyperparameters)
        
        elif algorithm == ModelAlgorithm.LOGISTIC_REGRESSION:
            if model_type == ModelType.CLASSIFICATION:
                return LogisticRegression(**hyperparameters)
            else:
                raise ValueError("Logistic regression is only for classification")
        
        elif algorithm == ModelAlgorithm.LINEAR_REGRESSION:
            if model_type == ModelType.REGRESSION:
                return LinearRegression(**hyperparameters)
            else:
                raise ValueError("Linear regression is only for regression")
        
        elif algorithm == ModelAlgorithm.SVM:
            if model_type == ModelType.CLASSIFICATION:
                return SVC(**hyperparameters)
            else:
                return SVR(**hyperparameters)
        
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def _calculate_metrics(self, y_true, y_pred, model_type: ModelType):
        """Calculate appropriate metrics based on model type"""
        if model_type == ModelType.CLASSIFICATION:
            accuracy = accuracy_score(y_true, y_pred)
            return {
                "accuracy": float(accuracy),
                "classification_report": classification_report(y_true, y_pred, output_dict=True)
            }
        else:  # Regression
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            return {
                "mse": float(mse),
                "rmse": float(rmse)
            }

# Initialize agent
ml_agent = MLTrainingAgent()

# Original endpoints
@app.get("/")
async def root():
    return {"message": "ML Training Dashboard API is running!"}

@app.get("/dashboard")
async def get_dashboard():
    """Serve the ML training dashboard"""
    return FileResponse("dashboard.html")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/api/users", response_model=List[User])
async def get_users():
    return users_db

# Dataset Management Endpoints
@app.post("/api/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = None,
    description: str = None
):
    """Upload a CSV dataset"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    dataset_id = str(uuid.uuid4())
    file_path = f"uploads/{dataset_id}_{file.filename}"
    
    # Save file
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Analyze dataset
    df = pd.read_csv(file_path)
    
    dataset = Dataset(
        id=dataset_id,
        name=name or file.filename,
        description=description,
        file_path=file_path,
        columns=list(df.columns),
        rows=len(df),
        created_at=datetime.now()
    )
    
    datasets_db[dataset_id] = dataset
    
    return {"dataset_id": dataset_id, "message": "Dataset uploaded successfully", "dataset": dataset}

@app.get("/api/datasets", response_model=List[Dataset])
async def get_datasets():
    """Get all uploaded datasets"""
    return list(datasets_db.values())

@app.get("/api/datasets/{dataset_id}")
async def get_dataset_preview(dataset_id: str, rows: int = 5):
    """Get a preview of the dataset"""
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = datasets_db[dataset_id]
    df = pd.read_csv(dataset.file_path)
    
    return {
        "dataset": dataset,
        "preview": df.head(rows).to_dict(orient="records"),
        "statistics": df.describe().to_dict()
    }

# Training Endpoints
@app.post("/api/training/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start a new model training job"""
    if request.dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = datasets_db[request.dataset_id]
    
    # Validate columns
    if request.target_column not in dataset.columns:
        raise HTTPException(status_code=400, detail="Target column not found in dataset")
    
    for col in request.feature_columns:
        if col not in dataset.columns:
            raise HTTPException(status_code=400, detail=f"Feature column '{col}' not found in dataset")
    
    job_id = str(uuid.uuid4())
    training_job = TrainingJob(
        id=job_id,
        model_name=request.model_name,
        dataset_id=request.dataset_id,
        algorithm=request.algorithm,
        model_type=request.model_type,
        status=TrainingStatus.PENDING,
        created_at=datetime.now()
    )
    
    training_jobs_db[job_id] = training_job
    
    # Start training in background
    background_tasks.add_task(ml_agent.train_model, job_id, request)
    
    return {"job_id": job_id, "message": "Training started", "job": training_job}

@app.get("/api/training/jobs", response_model=List[TrainingJob])
async def get_training_jobs():
    """Get all training jobs"""
    return list(training_jobs_db.values())

@app.get("/api/training/jobs/{job_id}", response_model=TrainingJob)
async def get_training_job(job_id: str):
    """Get specific training job status"""
    if job_id not in training_jobs_db:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return training_jobs_db[job_id]

# Model Management Endpoints
@app.get("/api/models", response_model=List[ModelInfo])
async def get_models():
    """Get all trained models"""
    return list(models_db.values())

@app.get("/api/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """Get specific model information"""
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return models_db[model_id]

@app.post("/api/models/{model_id}/predict", response_model=PredictionResponse)
async def predict(model_id: str, request: PredictionRequest):
    """Make prediction using a trained model"""
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = models_db[model_id]
    
    try:
        # Load model
        model = joblib.load(model_info.model_path)
        
        # Prepare input data
        input_df = pd.DataFrame([request.data])
        
        # Ensure we have the right columns
        missing_cols = set(model_info.feature_columns) - set(input_df.columns)
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {list(missing_cols)}"
            )
        
        # Make prediction
        prediction = model.predict(input_df[model_info.feature_columns])[0]
        
        # Get confidence if available (for some classifiers)
        confidence = None
        if hasattr(model, 'predict_proba') and model_info.model_type == ModelType.CLASSIFICATION:
            proba = model.predict_proba(input_df[model_info.feature_columns])[0]
            confidence = float(max(proba))
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_used=model_info.name
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a trained model"""
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = models_db[model_id]
    
    # Delete model file
    if os.path.exists(model_info.model_path):
        os.remove(model_info.model_path)
    
    # Remove from database
    del models_db[model_id]
    
    # Also remove training job if exists
    if model_id in training_jobs_db:
        del training_jobs_db[model_id]
    
    return {"message": f"Model {model_info.name} deleted successfully"}

# Analytics Endpoints
@app.get("/api/analytics/dashboard")
async def get_ml_dashboard():
    """Get ML dashboard analytics"""
    total_datasets = len(datasets_db)
    total_models = len(models_db)
    active_trainings = len([j for j in training_jobs_db.values() if j.status == TrainingStatus.TRAINING])
    
    # Recent training jobs
    recent_jobs = sorted(training_jobs_db.values(), key=lambda x: x.created_at, reverse=True)[:5]
    
    # Model performance summary
    model_performance = []
    for model in models_db.values():
        if model.model_type == ModelType.CLASSIFICATION:
            accuracy = model.metrics.get('accuracy', 0)
            model_performance.append({
                'name': model.name,
                'type': model.model_type,
                'accuracy': accuracy
            })
        else:
            rmse = model.metrics.get('rmse', 0)
            model_performance.append({
                'name': model.name,
                'type': model.model_type,
                'rmse': rmse
            })
    
    return {
        "total_datasets": total_datasets,
        "total_models": total_models,
        "active_trainings": active_trainings,
        "recent_jobs": recent_jobs,
        "model_performance": model_performance
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)