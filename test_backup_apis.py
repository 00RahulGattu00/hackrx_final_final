#!/usr/bin/env python3
"""
Test script specifically for backup APIs (non-OpenAI)
This will test Groq, Hugging Face, and Cohere APIs directly
"""

import os
import requests
from dotenv import load_dotenv
import time

load_dotenv()

def test_groq_api():
    """Test Groq API directly"""
    print("🚀 Testing Groq API...")
    
    groq_key = os.getenv('GROQ_API_KEY')
    if not groq_key:
        print("❌ Groq API key not found")
        return False
    
    headers = {
        'Authorization': f'Bearer {groq_key}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': 'llama3-8b-8192',
        'messages': [
            {"role": "user", "content": "Say 'Groq API test successful' in exactly those words."}
        ],
        'max_tokens': 50,
        'temperature': 0.2
    }
    
    try:
        response = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content']
            print(f"✅ Groq API working!")
            print(f"   Response: {answer}")
            return True
        else:
            print(f"❌ Groq API failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Groq API error: {e}")
        return False

def test_huggingface_api():
    """Test Hugging Face API directly"""
    print("\n🤗 Testing Hugging Face API...")
    
    hf_key = os.getenv('HUGGINGFACE_API_KEY')
    if not hf_key:
        print("❌ Hugging Face API key not found")
        return False
    
    headers = {
        'Authorization': f'Bearer {hf_key}',
        'Content-Type': 'application/json'
    }
    
    # Test with DialoGPT model
    data = {
        'inputs': 'Human: Say hello in a friendly way.\n\nAssistant:',
        'parameters': {
            'max_new_tokens': 50,
            'temperature': 0.2,
            'return_full_text': False
        },
        'options': {
            'wait_for_model': True,
            'use_cache': False
        }
    }
    
    try:
        print("   Trying DialoGPT model...")
        response = requests.post(
            'https://api-inference.huggingface.co/models/microsoft/DialoGPT-large',
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Hugging Face API working!")
            print(f"   Response: {result}")
            return True
        else:
            print(f"⚠️  DialoGPT failed ({response.status_code}), trying fallback...")
            return test_huggingface_fallback(headers)
            
    except Exception as e:
        print(f"⚠️  DialoGPT error: {e}, trying fallback...")
        return test_huggingface_fallback(headers)

def test_huggingface_fallback(headers):
    """Test Hugging Face with a simpler model"""
    try:
        # Try with a simpler text generation model
        data = {
            'inputs': 'The weather today is',
            'parameters': {
                'max_new_tokens': 20,
                'temperature': 0.5
            },
            'options': {
                'wait_for_model': True
            }
        }
        
        response = requests.post(
            'https://api-inference.huggingface.co/models/gpt2',
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Hugging Face fallback working!")
            print(f"   Model: gpt2")
            print(f"   Response: {result}")
            return True
        else:
            print(f"❌ Hugging Face fallback failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Hugging Face fallback error: {e}")
        return False

def test_cohere_api():
    """Test Cohere API directly"""
    print("\n🔗 Testing Cohere API...")
    
    cohere_key = os.getenv('COHERE_API_KEY')
    if not cohere_key:
        print("❌ Cohere API key not found")
        return False
    
    headers = {
        'Authorization': f'Bearer {cohere_key}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': 'command-light',
        'prompt': 'Say "Cohere API test successful" in exactly those words.',
        'max_tokens': 50,
        'temperature': 0.2
    }
    
    try:
        response = requests.post(
            'https://api.cohere.ai/v1/generate',
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result['generations'][0]['text'].strip()
            print(f"✅ Cohere API working!")
            print(f"   Response: {answer}")
            return True
        else:
            print(f"❌ Cohere API failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Cohere API error: {e}")
        return False

def test_integrated_processor():
    """Test the integrated processor with fallback"""
    print("\n🧠 Testing Integrated LLM Processor (skipping OpenAI)...")
    
    try:
        from llm_processor import LLMProcessor
        
        processor = LLMProcessor()
        
        # Remove OpenAI APIs temporarily for this test
        non_openai_apis = [api for api in processor.apis if api['type'] != 'openai']
        original_apis = processor.apis
        processor.apis = non_openai_apis
        
        print(f"   Testing with {len(processor.apis)} non-OpenAI APIs")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Integrated test successful' in exactly those words."}
        ]
        
        response = processor.generate_response(messages)
        print(f"✅ Integrated processor working!")
        print(f"   Response: {response[:100]}...")
        
        # Restore original APIs
        processor.apis = original_apis
        return True
        
    except Exception as e:
        print(f"❌ Integrated processor error: {e}")
        return False

def main():
    """Run all backup API tests"""
    print("🔄 Backup API Testing (Non-OpenAI)")
    print("=" * 50)
    print("Since OpenAI quota is exceeded, testing backup APIs...")
    print()
    
    tests = [
        ("Groq API", test_groq_api),
        ("Hugging Face API", test_huggingface_api),
        ("Cohere API", test_cohere_api),
        ("Integrated Processor", test_integrated_processor)
    ]
    
    results = {}
    working_apis = 0
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
            if results[test_name]:
                working_apis += 1
        except Exception as e:
            print(f"❌ Unexpected error in {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 BACKUP API TEST RESULTS:")
    
    for test_name, result in results.items():
        status = "✅ WORKING" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nWorking APIs: {working_apis}/{len(tests)}")
    
    if working_apis > 0:
        print(f"\n🎉 You have {working_apis} working backup API(s)!")
        print("Your system will work even without OpenAI.")
        print("\nRecommendations:")
        if results.get("Groq API"):
            print("✅ Groq is your best backup - fast and reliable")
        if results.get("Hugging Face API"):
            print("✅ Hugging Face is working - good for free tier")
        if results.get("Cohere API"):
            print("✅ Cohere is working - good backup option")
    else:
        print("\n⚠️  No backup APIs are working!")
        print("Check your API keys and internet connection.")

if __name__ == "__main__":
    main()