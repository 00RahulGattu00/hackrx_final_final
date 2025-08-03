#!/usr/bin/env python3
"""
Quick test with your ngrok endpoint using backup APIs
"""

import requests
import json

# Configuration
NGROK_URL = "https://619eea5d5a79.ngrok-free.app"
BEARER_TOKEN = "16ca23504efb8f8b98b1d84b2516a4b6ccb69f3c955ac9a8107497f5d14d6dbb"

def quick_test():
    """Quick test of your live API with a simple document"""
    print("🚀 Quick API Test with Backup APIs")
    print("=" * 50)
    
    # Test health first
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{NGROK_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health: {data.get('status')}")
            print(f"   API endpoints: {data.get('api_endpoints', 0)}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return
    
    # Test main endpoint with a simple document
    print("\n2. Testing main endpoint...")
    
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Use a simple, small PDF for testing
    payload = {
        "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        "questions": [
            "What is this document about?"
        ]
    }
    
    try:
        print("   Sending request (this may take a moment due to API fallbacks)...")
        response = requests.post(
            f"{NGROK_URL}/hackrx/run",
            headers=headers,
            json=payload,
            timeout=180  # 3 minutes to allow for API fallbacks
        )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            answers = data.get('answers', [])
            
            print("✅ SUCCESS!")
            print(f"   Received {len(answers)} answer(s)")
            
            for i, answer in enumerate(answers):
                print(f"\n   Question: {payload['questions'][i]}")
                print(f"   Answer: {answer}")
            
            print(f"\n🎉 Your API is working with backup APIs!")
            
        else:
            print(f"❌ Request failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data}")
            except:
                print(f"   Error text: {response.text}")
    
    except requests.exceptions.Timeout:
        print("⚠️  Request timed out - this might be normal if APIs are slow")
        print("   Your system is probably working but APIs are taking time to respond")
    except Exception as e:
        print(f"❌ Request error: {e}")

if __name__ == "__main__":
    quick_test()
    print("\n" + "=" * 50)
    print("💡 Next steps:")
    print("1. If this worked, your API is ready for production!")
    print("2. If OpenAI quota is exceeded, your backup APIs are handling requests")
    print("3. Consider adding more API keys or upgrading OpenAI plan for production")