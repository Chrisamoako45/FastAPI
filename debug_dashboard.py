#!/usr/bin/env python3
"""
Complete dashboard troubleshooting script
Run this to test CSV upload and model training functionality
"""

import requests
import json
import pandas as pd
import numpy as np
import io

API_BASE = "http://localhost:8000"

def test_csv_upload():
    """Test CSV file upload functionality"""
    print("🔧 Testing CSV Upload Functionality...")
    print("-" * 40)
    
    # Create a simple test dataset
    np.random.seed(42)
    data = {
        'age': np.random.randint(20, 60, 100),
        'income': np.random.randint(30000, 100000, 100),
        'score': np.random.randint(1, 100, 100)
    }
    # Create target variable
    data['approved'] = (data['income'] > 50000).astype(int)
    
    df = pd.DataFrame(data)
    
    # Save to CSV string
    csv_string = df.to_csv(index=False)
    
    print(f"✅ Created test dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Sample data:")
    print(df.head(3))
    
    # Test upload via API
    try:
        files = {'file': ('test_data.csv', csv_string, 'text/csv')}
        data_form = {
            'name': 'Test Upload Dataset',
            'description': 'Testing CSV upload functionality'
        }
        
        print(f"\n📤 Uploading to: {API_BASE}/api/datasets/upload")
        response = requests.post(f"{API_BASE}/api/datasets/upload", 
                               files=files, data=data_form)
        
        print(f"   Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload successful!")
            print(f"   Dataset ID: {result.get('dataset_id')}")
            print(f"   Response: {result}")
            return result.get('dataset_id')
        else:
            print(f"❌ Upload failed!")
            print(f"   Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Upload failed with exception: {e}")
        return None

def test_model_training(dataset_id):
    """Test model training functionality"""
    print(f"\n🤖 Testing Model Training...")
    print("-" * 40)
    
    if not dataset_id:
        print("❌ No dataset ID provided, skipping training test")
        return None
    
    training_request = {
        "dataset_id": dataset_id,
        "model_name": "Test Classification Model",
        "algorithm": "random_forest",
        "model_type": "classification",
        "target_column": "approved",
        "feature_columns": ["age", "income", "score"],
        "test_size": 0.2
    }
    
    try:
        print(f"📤 Starting training...")
        print(f"   Request: {json.dumps(training_request, indent=2)}")
        
        response = requests.post(f"{API_BASE}/api/training/start",
                               json=training_request,
                               headers={'Content-Type': 'application/json'})
        
        print(f"   Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Training started!")
            print(f"   Job ID: {result.get('job_id')}")
            return result.get('job_id')
        else:
            print(f"❌ Training failed!")
            print(f"   Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Training failed with exception: {e}")
        return None

def monitor_training(job_id):
    """Monitor training progress"""
    print(f"\n⏱️ Monitoring Training Progress...")
    print("-" * 40)
    
    if not job_id:
        print("❌ No job ID provided, skipping monitoring")
        return
    
    import time
    
    for i in range(10):  # Check up to 10 times
        try:
            response = requests.get(f"{API_BASE}/api/training/jobs/{job_id}")
            
            if response.status_code == 200:
                job = response.json()
                status = job.get('status', 'unknown')
                progress = job.get('progress', 0)
                
                print(f"   Attempt {i+1}: Status = {status}, Progress = {progress:.1%}")
                
                if status == 'completed':
                    print(f"✅ Training completed!")
                    print(f"   Metrics: {job.get('metrics', {})}")
                    break
                elif status == 'failed':
                    print(f"❌ Training failed!")
                    print(f"   Error: {job.get('error_message', 'Unknown error')}")
                    break
                    
                time.sleep(2)  # Wait 2 seconds before next check
            else:
                print(f"❌ Failed to check status: {response.text}")
                break
                
        except Exception as e:
            print(f"❌ Monitoring failed: {e}")
            break

def check_api_endpoints():
    """Check if all required endpoints are working"""
    print("🔍 Checking API Endpoints...")
    print("-" * 40)
    
    endpoints = [
        ("GET", "/health", "Health Check"),
        ("GET", "/api/datasets", "List Datasets"),
        ("GET", "/api/models", "List Models"),
        ("GET", "/api/training/jobs", "List Training Jobs"),
        ("GET", "/dashboard", "Dashboard Page")
    ]
    
    results = {}
    
    for method, endpoint, name in endpoints:
        try:
            if method == "GET":
                response = requests.get(f"{API_BASE}{endpoint}")
            
            status = "✅ OK" if response.status_code in [200, 404] else "❌ ERROR"
            results[endpoint] = response.status_code
            
            print(f"   {status} {name} ({endpoint}) - Status: {response.status_code}")
            
        except Exception as e:
            print(f"   ❌ ERROR {name} ({endpoint}) - Exception: {e}")
            results[endpoint] = "ERROR"
    
    return results

def test_dashboard_connection():
    """Test if dashboard can connect to API"""
    print(f"\n🌐 Testing Dashboard Connection...")
    print("-" * 40)
    
    try:
        # Test CORS by making a request from browser perspective
        response = requests.get(f"{API_BASE}/health", 
                              headers={'Origin': 'http://localhost:8000'})
        
        print(f"   CORS Test: {response.status_code}")
        print(f"   Headers: {dict(response.headers)}")
        
        if 'access-control-allow-origin' in response.headers:
            print("✅ CORS is properly configured")
        else:
            print("❌ CORS might not be configured properly")
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")

def main():
    """Run complete dashboard test"""
    print("🚀 ML Dashboard Troubleshooting")
    print("=" * 50)
    
    # Step 1: Check API endpoints
    endpoint_results = check_api_endpoints()
    
    # Step 2: Test dashboard connection
    test_dashboard_connection()
    
    # Step 3: Test CSV upload
    dataset_id = test_csv_upload()
    
    # Step 4: Test model training
    if dataset_id:
        job_id = test_model_training(dataset_id)
        
        # Step 5: Monitor training
        if job_id:
            monitor_training(job_id)
    
    print("\n" + "=" * 50)
    print("🎯 TROUBLESHOOTING SUMMARY")
    print("=" * 50)
    
    print("\n💡 If you're still having issues:")
    print("1. Make sure FastAPI server is running: python3 main.py")
    print("2. Check browser console (F12) for JavaScript errors")
    print("3. Try uploading a simple CSV file (not too large)")
    print("4. Make sure CSV has proper column headers")
    print("5. Check CORS configuration in main.py")
    
    print(f"\n🔗 URLs to test:")
    print(f"   Dashboard: {API_BASE}/dashboard")
    print(f"   API Docs: {API_BASE}/docs")
    print(f"   Health: {API_BASE}/health")

if __name__ == "__main__":
    main()