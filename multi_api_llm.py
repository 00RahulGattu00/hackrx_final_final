import os
import random
from typing import List, Dict, Any, Optional
import openai
from openai import OpenAI
import requests
import json
import time

class MultiAPILLMProcessor:
    def __init__(self):
        # Initialize multiple API clients
        self.apis = []
        self.current_api_index = 0
        
        # OpenAI API Keys (you can add multiple)
        openai_keys = [
            os.getenv('OPENAI_API_KEY_1'),
            os.getenv('OPENAI_API_KEY_2'),
            os.getenv('OPENAI_API_KEY_3'),
            # Add more keys as needed
        ]
        
        # Add OpenAI clients
        for key in openai_keys:
            if key:
                self.apis.append({
                    'type': 'openai',
                    'client': OpenAI(api_key=key),
                    'model': 'gpt-3.5-turbo',
                    'key': key
                })
        
        # Add Groq API (Free tier: 15 req/min, 6000 req/day)
        groq_key = os.getenv('GROQ_API_KEY')
        if groq_key:
            self.apis.append({
                'type': 'groq',
                'key': groq_key,
                'model': 'llama3-8b-8192',  # or 'mixtral-8x7b-32768'
                'base_url': 'https://api.groq.com/openai/v1'
            })
        
        # Add Anthropic Claude API
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key:
            self.apis.append({
                'type': 'anthropic',
                'key': anthropic_key,
                'model': 'claude-3-haiku-20240307'
            })
        
        # Add Hugging Face API (Free tier available)
        hf_key = os.getenv('HUGGINGFACE_API_KEY')
        if hf_key:
            self.apis.append({
                'type': 'huggingface',
                'key': hf_key,
                'model': 'microsoft/DialoGPT-medium'
            })
        
        # Add Cohere API (Free tier: 100 req/min)
        cohere_key = os.getenv('COHERE_API_KEY')
        if cohere_key:
            self.apis.append({
                'type': 'cohere',
                'key': cohere_key,
                'model': 'command-light'
            })
        
        print(f"Initialized with {len(self.apis)} API endpoints")
    
    def call_openai(self, api_config: Dict, messages: List[Dict]) -> str:
        """Call OpenAI API"""
        try:
            response = api_config['client'].chat.completions.create(
                model=api_config['model'],
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def call_groq(self, api_config: Dict, messages: List[Dict]) -> str:
        """Call Groq API (OpenAI-compatible)"""
        try:
            headers = {
                'Authorization': f'Bearer {api_config["key"]}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': api_config['model'],
                'messages': messages,
                'max_tokens': 1000,
                'temperature': 0.7
            }
            
            response = requests.post(
                f"{api_config['base_url']}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                raise Exception(f"Groq API error: {response.status_code} - {response.text}")
        except Exception as e:
            raise Exception(f"Groq API error: {str(e)}")
    
    def call_anthropic(self, api_config: Dict, messages: List[Dict]) -> str:
        """Call Anthropic Claude API"""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=api_config['key'])
            
            # Convert OpenAI format to Anthropic format
            system_message = ""
            user_messages = []
            
            for msg in messages:
                if msg['role'] == 'system':
                    system_message = msg['content']
                else:
                    user_messages.append(msg)
            
            # Combine messages for Anthropic
            combined_message = ""
            for msg in user_messages:
                combined_message += f"{msg['role']}: {msg['content']}\n"
            
            response = client.messages.create(
                model=api_config['model'],
                max_tokens=1000,
                system=system_message,
                messages=[{"role": "user", "content": combined_message}]
            )
            
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")
    
    def call_huggingface(self, api_config: Dict, messages: List[Dict]) -> str:
        """Call Hugging Face API"""
        try:
            headers = {
                'Authorization': f'Bearer {api_config["key"]}',
                'Content-Type': 'application/json'
            }
            
            # Extract the last user message
            user_input = ""
            for msg in messages:
                if msg['role'] == 'user':
                    user_input = msg['content']
            
            data = {
                'inputs': user_input,
                'parameters': {
                    'max_length': 1000,
                    'temperature': 0.7
                }
            }
            
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{api_config['model']}",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '').replace(user_input, '').strip()
                return str(result)
            else:
                raise Exception(f"HuggingFace API error: {response.status_code} - {response.text}")
        except Exception as e:
            raise Exception(f"HuggingFace API error: {str(e)}")
    
    def call_cohere(self, api_config: Dict, messages: List[Dict]) -> str:
        """Call Cohere API"""
        try:
            headers = {
                'Authorization': f'Bearer {api_config["key"]}',
                'Content-Type': 'application/json'
            }
            
            # Extract the last user message
            user_input = ""
            for msg in messages:
                if msg['role'] == 'user':
                    user_input = msg['content']
            
            data = {
                'model': api_config['model'],
                'prompt': user_input,
                'max_tokens': 1000,
                'temperature': 0.7
            }
            
            response = requests.post(
                'https://api.cohere.ai/v1/generate',
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['generations'][0]['text'].strip()
            else:
                raise Exception(f"Cohere API error: {response.status_code} - {response.text}")
        except Exception as e:
            raise Exception(f"Cohere API error: {str(e)}")
    
    def generate_response(self, messages: List[Dict], max_retries: int = 3) -> str:
        """
        Generate response using available APIs with fallback mechanism
        """
        if not self.apis:
            raise Exception("No API endpoints configured")
        
        # Shuffle APIs to distribute load
        apis_to_try = self.apis.copy()
        random.shuffle(apis_to_try)
        
        last_error = None
        
        for api_config in apis_to_try:
            for attempt in range(max_retries):
                try:
                    print(f"Trying {api_config['type']} API (attempt {attempt + 1})")
                    
                    if api_config['type'] == 'openai':
                        return self.call_openai(api_config, messages)
                    elif api_config['type'] == 'groq':
                        return self.call_groq(api_config, messages)
                    elif api_config['type'] == 'anthropic':
                        return self.call_anthropic(api_config, messages)
                    elif api_config['type'] == 'huggingface':
                        return self.call_huggingface(api_config, messages)
                    elif api_config['type'] == 'cohere':
                        return self.call_cohere(api_config, messages)
                    
                except Exception as e:
                    last_error = e
                    print(f"Error with {api_config['type']}: {str(e)}")
                    
                    # Wait before retry
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    continue
            
            print(f"Failed all attempts for {api_config['type']}")
        
        raise Exception(f"All API endpoints failed. Last error: {str(last_error)}")

# Usage example
def main():
    # Set your API keys as environment variables
    os.environ['OPENAI_API_KEY_1'] = 'your-openai-key-1'
    os.environ['OPENAI_API_KEY_2'] = 'your-openai-key-2'
    os.environ['GROQ_API_KEY'] = 'your-groq-key'
    os.environ['ANTHROPIC_API_KEY'] = 'your-anthropic-key'
    os.environ['HUGGINGFACE_API_KEY'] = 'your-hf-key'
    os.environ['COHERE_API_KEY'] = 'your-cohere-key'
    
    processor = MultiAPILLMProcessor()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    try:
        response = processor.generate_response(messages)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
