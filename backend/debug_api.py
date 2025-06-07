#!/usr/bin/env python3
"""
Debug script to check if your ML agent API is working
Run this while your FastAPI server is running
"""

import requests
import json

API_BASE = "http://localhost:8000"

def check_endpoints():
    """Check all API endpoints"""
    print("🔍 Debugging ML Agent API...")
    print("=" * 50)
    
    endpoints = [
        ("/health", "Health Check"),
        ("/api/datasets", "Datasets"),
        ("/api/models", "Models"), 
        ("/api/training/jobs", "Training Jobs"),
        ("/api/analytics/dashboard", "Analytics")
    ]
    
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{API_BASE}{endpoint}")
            print(f"\n✅ {name} ({endpoint})")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    print(f"   Items: {len(data)}")
                    if len(data) > 0:
                        print(f"   Sample: {list(data[0].keys()) if data else 'None'}")
                else:
                    print(f"   Data: {data}")
            else:
                print(f"   Error: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"\n❌ {name} ({endpoint})")
            print("   Error: Cannot connect to API")
            print("   Make sure FastAPI server is running: python3 main.py")
        except Exception as e:
            print(f"\n❌ {name} ({endpoint})")
            print(f"   Error: {e}")
    
    print("\n" + "=" * 50)
    print("🌐 Dashboard URL: http://localhost:8000/dashboard")
    print("📚 API Docs: http://localhost:8000/docs")

if __name__ == "__main__":
    check_endpoints()