import os
import random
import json
from typing import List, Dict, Any, Optional
import openai
from openai import OpenAI
import httpx
import time
import logging
from functools import lru_cache
import hashlib
from dotenv import load_dotenv
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class LLMProcessor:
    """
    Multi-API LLM Processor that can use multiple LLM providers with fallback
    Handles OpenAI, Groq, Hugging Face, and Cohere APIs with improved reliability
    """
    
    def __init__(self):
        # Initialize attributes first
        self.apis = []
        self.current_api_index = 0
        self.api_stats = {}  # Initialize this BEFORE calling _setup_apis()
        
        # Response cache for reducing duplicate API calls
        self._response_cache = {}
        self._cache_max_size = 100
        
        # HTTP client for non-OpenAI APIs
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        
        # Now initialize APIs (which uses api_stats)
        self._setup_apis()
        
        logger.info(f"Initialized LLMProcessor with {len(self.apis)} API endpoints")
        if len(self.apis) == 0:
            logger.warning("No API keys found! Please add API keys to your .env file")
    
    def _setup_apis(self):
        """Setup all available API clients in priority order with better error handling"""
        
        # OpenAI API Keys (highest priority)
        openai_keys = [
            os.getenv('OPENAI_API_KEY'),
            os.getenv('OPENAI_API_KEY_1'),
            os.getenv('OPENAI_API_KEY_2'),
            os.getenv('OPENAI_API_KEY_3'),
        ]
        
        for i, key in enumerate(openai_keys):
            if key and key.strip() and not key.startswith('your_'):
                try:
                    # Test key format
                    if key.startswith('sk-'):
                        self.apis.append({
                            'type': 'openai',
                            'client': OpenAI(api_key=key),
                            'model': 'gpt-3.5-turbo',
                            'key': key[:10] + '...',  # Hide full key in logs
                            'priority': 1,
                            'id': f'openai_{i+1}',
                            'rate_limit': {'requests_per_minute': 3500, 'tokens_per_minute': 90000}
                        })
                        logger.info(f"Added OpenAI API client #{i+1}")
                    else:
                        logger.warning(f"Invalid OpenAI API key format for key #{i+1}")
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAI client #{i+1}: {e}")
        
        # Groq API (second priority - fast and reliable)
        groq_key = os.getenv('GROQ_API_KEY')
        if groq_key and groq_key.strip() and not groq_key.startswith('your_'):
            try:
                self.apis.append({
                    'type': 'groq',
                    'key': groq_key,
                    'model': 'llama3-8b-8192',
                    'base_url': 'https://api.groq.com/openai/v1',
                    'priority': 2,
                    'id': 'groq_1',
                    'rate_limit': {'requests_per_minute': 30, 'requests_per_day': 14400}
                })
                logger.info("Added Groq API client")
            except Exception as e:
                logger.warning(f"Failed to configure Groq API: {e}")
        
        # Cohere API (third priority - reliable)
        cohere_key = os.getenv('COHERE_API_KEY')
        if cohere_key and cohere_key.strip() and not cohere_key.startswith('your_'):
            try:
                self.apis.append({
                    'type': 'cohere',
                    'key': cohere_key,
                    'model': 'command-light',
                    'priority': 3,
                    'id': 'cohere_1',
                    'rate_limit': {'requests_per_minute': 100}
                })
                logger.info("Added Cohere API client")
            except Exception as e:
                logger.warning(f"Failed to configure Cohere API: {e}")
        
        # Hugging Face API (lowest priority - can be slow)
        hf_key = os.getenv('HUGGINGFACE_API_KEY')
        if hf_key and hf_key.strip() and not hf_key.startswith('your_'):
            try:
                self.apis.append({
                    'type': 'huggingface',
                    'key': hf_key,
                    'model': 'gpt2',
                    'priority': 4,
                    'id': 'huggingface_1',
                    'rate_limit': {'requests_per_hour': 1000}
                })
                logger.info("Added Hugging Face API client")
            except Exception as e:
                logger.warning(f"Failed to configure Hugging Face API: {e}")
        
        # Sort APIs by priority
        self.apis.sort(key=lambda x: x['priority'])
        
        # Initialize stats for each API
        for api in self.apis:
            self.api_stats[api['id']] = {
                'calls': 0,
                'successes': 0,
                'failures': 0,
                'total_time': 0.0,
                'avg_response_time': 0.0
            }
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language query to extract structured information with caching
        """
        # Check cache first
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self._response_cache:
            logger.debug("Using cached query parsing result")
            return self._response_cache[cache_key]
        
        system_prompt = """You are an expert query parser for insurance, legal, HR, and compliance documents. 
        Parse the given query and extract structured information.
        
        Return a JSON object with:
        - intent: The main intent (e.g., "coverage_check", "eligibility", "claim_amount", "waiting_period")
        - entities: Extracted entities like procedures, amounts, time periods, conditions
        - keywords: Key terms for semantic search
        - domain: Document domain (insurance, legal, hr, compliance)
        - complexity: Query complexity (simple, medium, complex)
        
        Be precise and comprehensive in your extraction."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Parse this query: '{query}'"}
        ]
        
        try:
            response = self.generate_response(messages, max_tokens=300)
            
            # Try to parse JSON from response
            try:
                # First try direct JSON parsing
                result = json.loads(response)
            except json.JSONDecodeError:
                # Extract JSON from text if wrapped in markdown or other text
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    logger.warning("No valid JSON found in query parsing response, using fallback")
                    result = self._fallback_parse_query(query)
            
            # Cache result
            self._cache_response(cache_key, result)
            
            logger.info(f"Successfully parsed query with intent: {result.get('intent', 'unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing query with LLM: {str(e)}")
            result = self._fallback_parse_query(query)
            self._cache_response(cache_key, result)
            return result
    
    def generate_answer(self, question: str, context: str, retrieved_chunks: List[Dict]) -> Dict[str, Any]:
        """
        Generate comprehensive answer based on question and context
        """
        # Check cache first
        context_summary = self._create_context_summary(retrieved_chunks)
        cache_key = hashlib.md5(f"{question}_{context_summary}".encode()).hexdigest()
        
        if cache_key in self._response_cache:
            logger.debug("Using cached answer generation result")
            return self._response_cache[cache_key]
        
        # Prepare context from retrieved chunks
        context_text = self._prepare_context(retrieved_chunks)
        
        system_prompt = """You are an expert document analyzer specializing in insurance, legal, HR, and compliance documents.
        
        Given a question and relevant document excerpts, provide:
        1. A clear, accurate answer
        2. Specific clause references that support your answer
        3. Step-by-step reasoning
        4. Confidence level (high/medium/low)
        5. Any important conditions or limitations
        
        Be precise, cite specific clauses, and explain your reasoning clearly.
        If information is insufficient, state what additional information would be needed."""
        
        user_prompt = f"""Question: {question}
        
        Relevant Document Excerpts:
        {context_text}
        
        Provide a comprehensive answer with supporting evidence and reasoning."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            answer_text = self.generate_response(messages, max_tokens=800)
            
            result = {
                "answer": answer_text,
                "reasoning": self._extract_reasoning(answer_text),
                "confidence": self._assess_confidence(retrieved_chunks),
                "supporting_chunks": [chunk.get('id', i) for i, chunk in enumerate(retrieved_chunks[:3])],
                "token_usage": len(answer_text) // 4  # Rough estimate
            }
            
            # Cache result
            self._cache_response(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer with LLM: {str(e)}")
            result = self._fallback_generate_answer(question, retrieved_chunks)
            self._cache_response(cache_key, result)
            return result
    
    def _cache_response(self, key: str, response: Any):
        """Cache response with size management"""
        if len(self._response_cache) >= self._cache_max_size:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self._response_cache))
            del self._response_cache[oldest_key]
        
        self._response_cache[key] = response
    
    def _create_context_summary(self, chunks: List[Dict]) -> str:
        """Create a short summary of context for caching"""
        if not chunks:
            return "empty"
        
        # Use first chunk's text snippet and chunk count
        first_chunk_snippet = chunks[0].get('text', '')[:50]
        return f"{len(chunks)}_{first_chunk_snippet}"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def generate_response_async(self, messages: List[Dict], max_tokens: int = 1000) -> str:
        """
        Async version of generate_response with retry logic
        """
        return self.generate_response(messages, max_retries=1, max_tokens=max_tokens)
    
    def generate_response(self, messages: List[Dict], max_retries: int = 2, max_tokens: int = 1000) -> str:
        """
        Generate response using available APIs with improved fallback mechanism
        """
        if not self.apis:
            raise Exception("No API endpoints configured. Please add API keys to your .env file")
        
        # Track API usage
        start_time = time.time()
        last_error = None
        
        # Sort APIs by success rate and priority
        sorted_apis = self._get_sorted_apis_by_performance()
        
        # Try APIs in performance order
        for api_config in sorted_apis:
            api_id = api_config['id']
            
            for attempt in range(max_retries):
                try:
                    logger.debug(f"Trying {api_config['type']} API (attempt {attempt + 1})")
                    
                    # Update stats
                    self.api_stats[api_id]['calls'] += 1
                    call_start = time.time()
                    
                    # Call appropriate API
                    if api_config['type'] == 'openai':
                        response = self._call_openai(api_config, messages, max_tokens)
                    elif api_config['type'] == 'groq':
                        response = self._call_groq(api_config, messages, max_tokens)
                    elif api_config['type'] == 'huggingface':
                        response = self._call_huggingface(api_config, messages, max_tokens)
                    elif api_config['type'] == 'cohere':
                        response = self._call_cohere(api_config, messages, max_tokens)
                    else:
                        continue
                    
                    # Success - update stats
                    call_time = time.time() - call_start
                    self.api_stats[api_id]['successes'] += 1
                    self.api_stats[api_id]['total_time'] += call_time
                    self.api_stats[api_id]['avg_response_time'] = (
                        self.api_stats[api_id]['total_time'] / self.api_stats[api_id]['successes']
                    )
                    
                    total_time = time.time() - start_time
                    logger.info(f"Successfully used {api_config['type']} API (response time: {total_time:.2f}s)")
                    return response
                    
                except Exception as e:
                    last_error = e
                    self.api_stats[api_id]['failures'] += 1
                    logger.warning(f"Error with {api_config['type']} (attempt {attempt + 1}): {str(e)}")
                    
                    # Wait before retry with exponential backoff
                    if attempt < max_retries - 1:
                        wait_time = min(2 ** attempt, 10)  # Cap at 10 seconds
                        time.sleep(wait_time)
                    continue
            
            logger.info(f"All attempts failed for {api_config['type']}, trying next API")
        
        # All APIs failed
        total_time = time.time() - start_time
        logger.error(f"All API endpoints failed after {total_time:.2f}s")
        raise Exception(f"All API endpoints failed. Last error: {str(last_error)}")
    
    def _get_sorted_apis_by_performance(self) -> List[Dict]:
        """Sort APIs by performance metrics and priority"""
        def api_score(api):
            api_id = api['id']
            stats = self.api_stats.get(api_id, {})
            
            # Calculate success rate
            total_calls = stats.get('calls', 0)
            successes = stats.get('successes', 0)
            success_rate = successes / total_calls if total_calls > 0 else 1.0
            
            # Calculate response time score (lower is better)
            avg_time = stats.get('avg_response_time', 1.0)
            time_score = 1.0 / (1.0 + avg_time)  # Normalize to 0-1 range
            
            # Combine with priority (lower priority number is better)
            priority_score = 1.0 / api['priority']
            
            # Final score (higher is better)
            return (success_rate * 0.5) + (time_score * 0.3) + (priority_score * 0.2)
        
        return sorted(self.apis, key=api_score, reverse=True)
    
    def _call_openai(self, api_config: Dict, messages: List[Dict], max_tokens: int) -> str:
        """Call OpenAI API with improved error handling"""
        try:
            response = api_config['client'].chat.completions.create(
                model=api_config['model'],
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
                timeout=30
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Log specific OpenAI errors
            if "rate_limit" in str(e).lower():
                raise Exception(f"OpenAI rate limit exceeded: {str(e)}")
            elif "quota" in str(e).lower():
                raise Exception(f"OpenAI quota exceeded: {str(e)}")
            else:
                raise Exception(f"OpenAI API error: {str(e)}")
    
    def _call_groq(self, api_config: Dict, messages: List[Dict], max_tokens: int) -> str:
        """Call Groq API with improved error handling"""
        headers = {
            'Authorization': f'Bearer {api_config["key"]}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': api_config['model'],
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': 0.2
        }
        
        try:
            with httpx.Client(timeout=30) as client:
                response = client.post(
                    f"{api_config['base_url']}/chat/completions",
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content'].strip()
                elif response.status_code == 429:
                    raise Exception("Groq rate limit exceeded")
                else:
                    raise Exception(f"Groq API error: {response.status_code} - {response.text}")
        except httpx.TimeoutException:
            raise Exception("Groq API timeout")
        except Exception as e:
            if "rate limit" not in str(e):
                raise Exception(f"Groq API error: {str(e)}")
            raise
    
    def _call_huggingface(self, api_config: Dict, messages: List[Dict], max_tokens: int) -> str:
        """Call Hugging Face API with improved reliability"""
        headers = {
            'Authorization': f'Bearer {api_config["key"]}',
            'Content-Type': 'application/json'
        }
        
        # Convert messages to simple prompt for better compatibility
        prompt = self._messages_to_prompt(messages)
        
        # Try with GPT-2 first (more reliable)
        models_to_try = ['gpt2', 'microsoft/DialoGPT-medium']
        
        for model in models_to_try:
            try:
                data = {
                    'inputs': prompt,
                    'parameters': {
                        'max_new_tokens': min(max_tokens, 200),  # HF has limits
                        'temperature': 0.3,
                        'return_full_text': False,
                        'do_sample': True,
                        'top_p': 0.9
                    },
                    'options': {
                        'wait_for_model': True,
                        'use_cache': False
                    }
                }
                
                with httpx.Client(timeout=60) as client:
                    response = client.post(
                        f"https://api-inference.huggingface.co/models/{model}",
                        headers=headers,
                        json=data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if isinstance(result, list) and len(result) > 0:
                            generated = result[0].get('generated_text', '')
                            return generated.strip()
                        elif isinstance(result, dict) and 'generated_text' in result:
                            return result['generated_text'].strip()
                        else:
                            continue  # Try next model
                    elif response.status_code == 503:
                        # Model loading, try next
                        continue
                    else:
                        continue  # Try next model
                        
            except Exception as e:
                logger.debug(f"HuggingFace model {model} failed: {e}")
                continue
        
        raise Exception("All HuggingFace models failed")
    
    def _call_cohere(self, api_config: Dict, messages: List[Dict], max_tokens: int) -> str:
        """Call Cohere API with improved error handling"""
        headers = {
            'Authorization': f'Bearer {api_config["key"]}',
            'Content-Type': 'application/json'
        }
        
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        data = {
            'model': api_config['model'],
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': 0.2,
            'stop_sequences': []
        }
        
        try:
            with httpx.Client(timeout=30) as client:
                response = client.post(
                    'https://api.cohere.ai/v1/generate',
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['generations'][0]['text'].strip()
                elif response.status_code == 429:
                    raise Exception("Cohere rate limit exceeded")
                else:
                    raise Exception(f"Cohere API error: {response.status_code} - {response.text}")
        except httpx.TimeoutException:
            raise Exception("Cohere API timeout")
        except Exception as e:
            raise Exception(f"Cohere API error: {str(e)}")
    
    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert OpenAI-style messages to a single prompt"""
        prompt_parts = []
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"Human: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    def _prepare_context(self, chunks: List[Dict]) -> str:
        """Prepare context from retrieved chunks for LLM input"""
        if not chunks:
            return "No relevant document excerpts found."
        
        context_parts = []
        for i, chunk in enumerate(chunks[:5]):  # Limit to top 5 chunks
            relevance = chunk.get('relevance_score', 0)
            context_parts.append(f"[Excerpt {i+1}] (Relevance: {relevance:.3f})")
            context_parts.append(chunk.get('text', ''))
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _extract_reasoning(self, answer_text: str) -> str:
        """Extract reasoning section from LLM response"""
        lines = answer_text.split('\n')
        reasoning_lines = []
        
        for line in lines:
            if any(keyword in line.lower() for keyword in 
                   ['because', 'since', 'according to', 'based on', 'reasoning', 'therefore']):
                reasoning_lines.append(line)
        
        return ' '.join(reasoning_lines) if reasoning_lines else "Reasoning not explicitly provided."
    
    def _assess_confidence(self, chunks: List[Dict]) -> str:
        """Assess confidence based on retrieved chunks quality"""
        if not chunks:
            return "low"
        
        avg_score = sum(chunk.get('relevance_score', 0) for chunk in chunks) / len(chunks)
        
        if avg_score > 0.8:
            return "high"
        elif avg_score > 0.6:
            return "medium"
        else:
            return "low"
    
    def _fallback_parse_query(self, query: str) -> Dict[str, Any]:
        """Fallback query parsing without LLM with improved pattern matching"""
        import re
        
        entities = {}
        keywords = []
        
        # Extract common patterns
        query_lower = query.lower()
        
        # Medical/insurance procedures
        if re.search(r'surgery|operation|procedure|treatment', query_lower):
            entities['procedure_type'] = 'medical_procedure'
            keywords.extend(['surgery', 'procedure', 'treatment'])
        
        # Insurance concepts
        if re.search(r'waiting period|wait', query_lower):
            entities['concern'] = 'waiting_period'
            keywords.append('waiting_period')
        
        if re.search(r'cover|coverage', query_lower):
            entities['concern'] = 'coverage'
            keywords.append('coverage')
        
        if re.search(r'claim|reimbursement', query_lower):
            entities['concern'] = 'claim'
            keywords.append('claim')
        
        # Extract numbers (amounts, ages, periods)
        numbers = re.findall(r'\d+', query)
        if numbers:
            entities['numbers'] = numbers
        
        # Extract age
        age_match = re.search(r'(\d+)\s*year', query_lower)
        if age_match:
            entities['age'] = int(age_match.group(1))
        
        # Add meaningful words as keywords
        words = [word for word in query_lower.split() 
                if len(word) > 3 and word not in {'what', 'when', 'where', 'how', 'does', 'have'}]
        keywords.extend(words[:10])  # Limit keywords
        
        # Determine complexity
        complexity = "simple"
        if len(query.split()) > 15:
            complexity = "complex"
        elif len(query.split()) > 8:
            complexity = "medium"
        
        return {
            "intent": "coverage_check",
            "entities": entities,
            "keywords": list(set(keywords)),
            "domain": "insurance",
            "complexity": complexity
        }
    
    def _fallback_generate_answer(self, question: str, chunks: List[Dict]) -> Dict[str, Any]:
        """Fallback answer generation without LLM"""
        if not chunks:
            return {
                "answer": "Unable to find relevant information in the document to answer this question.",
                "reasoning": "No relevant chunks retrieved from the document.",
                "confidence": "low",
                "supporting_chunks": [],
                "token_usage": 0
            }
        
        # Use the best chunk with improved formatting
        best_chunk = chunks[0]
        chunk_text = best_chunk.get('text', '')
        
        # Create a more natural answer
        if len(chunk_text) > 300:
            answer = f"Based on the document content: {chunk_text[:300]}..."
        else:
            answer = f"According to the document: {chunk_text}"
        
        return {
            "answer": answer,
            "reasoning": f"Answer derived from highest relevance document section (score: {best_chunk.get('relevance_score', 0):.3f})",
            "confidence": self._assess_confidence(chunks),
            "supporting_chunks": [chunk.get('id', i) for i, chunk in enumerate(chunks[:3])],
            "token_usage": 0
        }
    
    def get_api_stats(self) -> Dict[str, Any]:
        """Get comprehensive API usage statistics"""
        return {
            "total_apis": len(self.apis),
            "api_stats": self.api_stats,
            "available_apis": [api['id'] for api in self.apis],
            "cache_size": len(self._response_cache),
            "cache_hit_rate": self._calculate_cache_hit_rate()
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (placeholder - would need hit tracking)"""
        # This is a simplified implementation
        # In a real system, you'd track cache hits vs misses
        return 0.0
    
    def clear_cache(self):
        """Clear response cache"""
        self._response_cache.clear()
        logger.info("Response cache cleared")
    
    def get_best_performing_api(self) -> Optional[str]:
        """Get the ID of the best performing API"""
        if not self.apis:
            return None
        
        best_api = None
        best_score = -1
        
        for api in self.apis:
            api_id = api['id']
            stats = self.api_stats.get(api_id, {})
            
            if stats.get('calls', 0) > 0:
                success_rate = stats.get('successes', 0) / stats.get('calls', 1)
                avg_time = stats.get('avg_response_time', float('inf'))
                
                # Score based on success rate and speed
                score = success_rate * (1.0 / (1.0 + avg_time))
                
                if score > best_score:
                    best_score = score
                    best_api = api_id
        
        return best_api or self.apis[0]['id']
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources"""
        if hasattr(self, 'http_client'):
            await self.http_client.aclose()