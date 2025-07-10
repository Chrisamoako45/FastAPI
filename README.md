# 🤖 ML Training Dashboard

A complete full-stack machine learning training platform built with FastAPI and vanilla HTML/CSS/JavaScript. Upload datasets, train models, and make predictions through an intuitive web interface.

![Dashboard Preview](https://img.shields.io/badge/Status-Ready%20to%20Use-green)
![Python](https://img.shields.io/badge/Python-3.7+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-red)
![Frontend](https://img.shields.io/badge/Frontend-Vanilla%20JS-yellow)

## ✨ Features

### 🔧 Backend Capabilities
- **Multiple ML Algorithms**: Random Forest, Logistic Regression, Linear Regression, SVM
- **Both Classification & Regression**: Automatic model type detection
- **Background Training**: Non-blocking model training with real-time progress tracking
- **RESTful API**: Complete CRUD operations for datasets, models, and training jobs
- **Model Management**: Save, load, and delete trained models
- **Real-time Predictions**: Make predictions with trained models via API
- **Data Validation**: Comprehensive input validation and error handling

### 🎨 Frontend Features
- **Drag & Drop Upload**: Intuitive CSV file upload with preview
- **Real-time Monitoring**: Live training progress bars and status updates
- **Model Dashboard**: View all trained models with performance metrics
- **Prediction Interface**: Interactive prediction form with model selection
- **Debug Mode**: Built-in debugging tools for troubleshooting
- **Responsive Design**: Works on desktop and mobile devices
- **Auto-refresh**: Automatic data updates every 10 seconds

### 🛠️ Supported Algorithms

| Algorithm | Classification | Regression | Hyperparameters |
|-----------|----------------|------------|-----------------|
| Random Forest | ✅ | ✅ | Configurable via API |
| Logistic Regression | ✅ | ❌ | Configurable via API |
| Linear Regression | ❌ | ✅ | Configurable via API |
| SVM | ✅ | ✅ | Configurable via API |

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone or download the project files**:
   ```bash
   mkdir ml-dashboard
   cd ml-dashboard
   # Copy main.py, dashboard.html, and debug scripts to this directory
   ```

2. **Install Python dependencies**:
   ```bash
   pip install fastapi uvicorn pandas numpy scikit-learn joblib python-multipart
   ```

3. **Start the FastAPI server**:
   ```bash
   python main.py
   ```
   
   The server will start on `http://localhost:8000`

4. **Open the dashboard**:
   - Navigate to `http://localhost:8000/dashboard` in your browser
   - Or serve the HTML file directly

### Verify Installation

Run the debug script to test all functionality:
```bash
python debug_script.py
```

## 📊 Usage Guide

### 1. Upload Dataset
- Click the upload area or drag & drop a CSV file
- Enter optional name and description
- Click "Upload Dataset"
- Preview dataset columns and statistics

### 2. Train Model
- Select uploaded dataset from dropdown
- Choose model name and algorithm
- Select target column (what you want to predict)
- Click "Start Training"
- Monitor progress in real-time

### 3. Monitor Training
- View all training jobs in the "Training Jobs" section
- See real-time progress bars for active training
- Check completion status and metrics
- Debug failed training jobs

### 4. Use Trained Models
- View all completed models in "Trained Models"
- See performance metrics (accuracy for classification, RMSE for regression)
- Delete models if needed

### 5. Make Predictions
- Select a trained model
- Enter feature values in the form
- Click "Predict" to get results
- View prediction confidence (for classification)

## 🔌 API Documentation

### Dataset Endpoints
```http
POST /api/datasets/upload          # Upload CSV dataset
GET  /api/datasets                 # List all datasets
GET  /api/datasets/{id}            # Get dataset info and preview
```

### Training Endpoints
```http
POST /api/training/start           # Start model training
GET  /api/training/jobs            # List training jobs
GET  /api/training/jobs/{id}       # Get training job status
```

### Model Endpoints
```http
GET    /api/models                 # List trained models
GET    /api/models/{id}            # Get model info
POST   /api/models/{id}/predict    # Make prediction
DELETE /api/models/{id}            # Delete model
```

### Analytics Endpoints
```http
GET /api/analytics/dashboard       # Get dashboard analytics
GET /health                        # Health check
```

### Example API Usage

**Start Training**:
```json
POST /api/training/start
{
    "dataset_id": "uuid-string",
    "model_name": "My Classification Model",
    "algorithm": "random_forest",
    "model_type": "classification",
    "target_column": "approved",
    "feature_columns": ["age", "income", "score"],
    "test_size": 0.2,
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 10
    }
}
```

**Make Prediction**:
```json
POST /api/models/{model_id}/predict
{
    "model_id": "uuid-string",
    "data": {
        "age": 35,
        "income": 60000,
        "score": 85
    }
}
```

## 🏗️ Project Structure

```
ml-dashboard/
├── main.py                 # FastAPI backend server
├── dashboard.html          # Frontend web interface
├── debug_script.py         # API testing script
├── uploads/               # Uploaded CSV files (auto-created)
├── models/                # Trained model files (auto-created)
└── README.md              # This file
```

## 🔧 Configuration

### Backend Configuration
Edit `main.py` to modify:
- **CORS settings**: Currently allows all origins (`allow_origins=["*"]`)
- **File upload limits**: Add size restrictions if needed
- **Database backend**: Currently uses in-memory storage
- **Model algorithms**: Add new algorithms to `MLTrainingAgent`

### Frontend Configuration
Edit `dashboard.html` to modify:
- **API base URL**: Change `API_BASE` constant
- **Auto-refresh interval**: Currently 10 seconds
- **UI styling**: Modify CSS section
- **Debug mode**: Toggle debug information display

## 🐛 Troubleshooting

### Common Issues

**1. API Connection Failed**
```
Error: Cannot connect to API
Solution: Ensure FastAPI server is running on port 8000
Command: python main.py
```

**2. CORS Errors**
```
Error: CORS policy blocked request
Solution: Check CORS configuration in main.py
Current setting: allow_origins=["*"]
```

**3. File Upload Fails**
```
Error: Only CSV files are supported
Solution: Ensure file has .csv extension and proper format
```

**4. Training Fails**
```
Error: Target column not found
Solution: Verify CSV has column headers and correct column names
```

### Debug Mode
Enable debug mode in the frontend to see:
- API request/response details
- File upload progress
- Training status updates
- Error messages and stack traces

### Testing with Debug Script
Run the comprehensive test script:
```bash
python debug_script.py
```

This will test:
- API endpoint connectivity
- CSV upload functionality
- Model training workflow
- CORS configuration
- Real-time monitoring

## 📈 Performance Notes

### Memory Usage
- **In-memory storage**: All data stored in RAM
- **Model files**: Saved to disk using joblib
- **Scalability**: Consider database for production use

### Training Speed
- **Background processing**: Training runs in separate threads
- **Progress tracking**: Real-time updates via polling
- **Resource usage**: Depends on dataset size and algorithm

### Frontend Performance
- **Auto-refresh**: 10-second intervals to avoid overwhelming server
- **Caching**: Form selections preserved during refresh
- **Responsive**: Optimized for various screen sizes

## 🔒 Security Considerations

### Current Implementation
- **No authentication**: Open access to all endpoints
- **File uploads**: Only CSV files accepted
- **CORS**: Currently allows all origins
- **Data validation**: Input validation on both frontend and backend

### Production Recommendations
- Implement user authentication
- Add rate limiting for API endpoints
- Restrict CORS to specific domains
- Add file size limits for uploads
- Use HTTPS in production
- Implement proper logging and monitoring

## 🚀 Deployment

### Local Development
```bash
# Development server with auto-reload
python main.py
```

### Production Deployment
```bash
# Production server with Gunicorn
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📋 Requirements

### Python Dependencies
```
fastapi>=0.68.0
uvicorn>=0.15.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.0.0
python-multipart>=0.0.5
```

### Browser Support
- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## 🤝 Contributing

1. **Bug Reports**: Use debug mode to gather information
2. **Feature Requests**: Consider API compatibility
3. **Pull Requests**: Follow existing code style
4. **Testing**: Run debug script before submitting

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **FastAPI**: For the excellent web framework
- **Scikit-learn**: For comprehensive ML algorithms
- **Pandas**: For data manipulation capabilities
- **Uvicorn**: For ASGI server implementation

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Run the debug script
3. Enable debug mode in the frontend
4. Check browser console for errors
5. Verify FastAPI server logs

---

**Made with ❤️ for the ML community**