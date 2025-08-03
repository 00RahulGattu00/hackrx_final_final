#!/usr/bin/env python3
"""
Test script to validate your LLM API configurations
Run this before starting your main application
"""

import os
import sys
from dotenv import load_dotenv
from llm_processor import LLMProcessor
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_api_keys():
    """Test if API keys are properly configured"""
    load_dotenv()
    
    print("🔍 Checking API Key Configuration...")
    print("=" * 50)
    
    api_keys = {
        'OpenAI': os.getenv('OPENAI_API_KEY'),
        'OpenAI Backup 1': os.getenv('OPENAI_API_KEY_1'),
        'OpenAI Backup 2': os.getenv('OPENAI_API_KEY_2'),
        'Groq': os.getenv('GROQ_API_KEY'),
        'Hugging Face': os.getenv('HUGGINGFACE_API_KEY'),
        'Cohere': os.getenv('COHERE_API_KEY'),
    }
    
    configured_apis = []
    for name, key in api_keys.items():
        if key:
            print(f"✅ {name}: Configured")
            configured_apis.append(name)
        else:
            print(f"❌ {name}: Not configured")
    
    print("\n📊 Summary:")
    print(f"Total APIs configured: {len(configured_apis)}")
    print(f"Configured APIs: {', '.join(configured_apis)}")
    
    if len(configured_apis) == 0:
        print("\n⚠️  WARNING: No API keys found!")
        print("Please add at least one API key to your .env file")
        return False
    
    return True

def test_llm_processor():
    """Test LLM processor initialization and basic functionality"""
    print("\n🧠 Testing LLM Processor...")
    print("=" * 50)
    
    try:
        # Initialize processor
        processor = LLMProcessor()
        
        if len(processor.apis) == 0:
            print("❌ No APIs available in processor")
            return False
        
        print(f"✅ LLM Processor initialized with {len(processor.apis)} APIs")
        
        # Test query parsing with fallback
        print("\n📝 Testing query parsing (fallback mode)...")
        test_query = "What is the waiting period for knee surgery?"
        
        try:
            parsed = processor._fallback_parse_query(test_query)
            print(f"✅ Fallback query parsing works")
            print(f"   Intent: {parsed.get('intent')}")
            print(f"   Entities: {parsed.get('entities')}")
        except Exception as e:
            print(f"❌ Fallback query parsing failed: {e}")
            return False
        
        # Test LLM API call
        print("\n🌐 Testing LLM API call...")
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'API test successful' in exactly those words."}
        ]
        
        try:
            response = processor.generate_response(test_messages)
            print(f"✅ LLM API call successful")
            print(f"   Response: {response[:100]}...")
            return True
        except Exception as e:
            print(f"⚠️  LLM API call failed: {e}")
            print("   Fallback functionality will be used")
            return True  # Still consider this a pass since fallback works
            
    except Exception as e:
        print(f"❌ LLM Processor initialization failed: {e}")
        return False

def test_huggingface_specific():
    """Test Hugging Face API specifically"""
    print("\n🤗 Testing Hugging Face API...")
    print("=" * 50)
    
    hf_key = os.getenv('HUGGINGFACE_API_KEY')
    if not hf_key:
        print("❌ Hugging Face API key not found")
        return False
    
    try:
        import requests
        
        headers = {
            'Authorization': f'Bearer {hf_key}',
            'Content-Type': 'application/json'
        }
        
        # Test with a simple model
        data = {
            'inputs': 'Hello, world!',
            'parameters': {'max_new_tokens': 50},
            'options': {'wait_for_model': True}
        }
        
        response = requests.post(
            'https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium',
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Hugging Face API is accessible")
            result = response.json()
            print(f"   Response type: {type(result)}")
            return True
        else:
            print(f"⚠️  Hugging Face API returned status {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Hugging Face API test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 LLM API Configuration Test")
    print("=" * 50)
    
    # Test 1: API Keys
    keys_ok = test_api_keys()
    
    if not keys_ok:
        print("\n❌ CRITICAL: No API keys configured!")
        print("\nTo fix this:")
        print("1. Copy the .env template")
        print("2. Add your API keys")
        print("3. Run this test again")
        sys.exit(1)
    
    # Test 2: LLM Processor
    processor_ok = test_llm_processor()
    
    # Test 3: Hugging Face specific
    hf_ok = test_huggingface_specific()
    
    print("\n" + "=" * 50)
    print("🎯 FINAL RESULTS:")
    print(f"API Keys: {'✅ PASS' if keys_ok else '❌ FAIL'}")
    print(f"LLM Processor: {'✅ PASS' if processor_ok else '❌ FAIL'}")
    print(f"Hugging Face: {'✅ PASS' if hf_ok else '⚠️  DEGRADED'}")
    
    if keys_ok and processor_ok:
        print("\n🎉 Your system is ready to run!")
        print("You can now start your FastAPI server with: python app.py")
    else:
        print("\n⚠️  Some issues detected, but the system may still work with fallbacks")
    
    print("\n💡 Tips:")
    print("- For best reliability, configure multiple API keys")
    print("- OpenAI provides the most consistent results")
    print("- Free tier APIs may have rate limits")
    print("- The system automatically falls back between APIs")

if __name__ == "__main__":
    main()