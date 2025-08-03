from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
import logging
import warnings
import pickle
import os
from functools import lru_cache
import hashlib
import time

# Suppress some warnings from transformers
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handles document chunking and preprocessing with improved algorithms and caching
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Cache for processed chunks
        self._chunk_cache = {}
        self._cache_max_size = 50
        
        logger.info(f"DocumentProcessor initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def create_chunks(self, text: str, use_cache: bool = True) -> List[Dict[str, any]]:
        """
        Create semantic chunks from document text with metadata and caching
        
        Args:
            text (str): Input document text
            use_cache (bool): Whether to use caching for repeated text
            
        Returns:
            List[Dict]: List of chunk dictionaries with metadata
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to create_chunks")
            return []
        
        # Check cache first
        if use_cache:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self._chunk_cache:
                logger.debug("Using cached chunks")
                return self._chunk_cache[text_hash]
        
        try:
            # Clean and preprocess text
            text = self._clean_text(text)
            
            # Split into sentences first
            sentences = self._split_into_sentences(text)
            
            if not sentences:
                logger.warning("No sentences found after text processing")
                return []
            
            # Create overlapping chunks with improved algorithm
            chunks = self._create_semantic_chunks(sentences)
            
            # Cache result if enabled
            if use_cache and chunks:
                self._cache_chunks(text_hash, chunks)
            
            logger.info(f"Created {len(chunks)} chunks from {len(sentences)} sentences")
            
            # Log chunk statistics
            if chunks:
                avg_length = sum(c['length'] for c in chunks) / len(chunks)
                logger.info(f"Average chunk length: {avg_length:.0f} characters")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}")
            # Return simple fallback chunks
            return self._create_simple_chunks(text)
    
    def _create_semantic_chunks(self, sentences: List[str]) -> List[Dict[str, any]]:
        """
        Create chunks with improved semantic awareness
        """
        chunks = []
        current_chunk = ""
        chunk_id = 0
        sentence_start = 0
        
        for i, sentence in enumerate(sentences):
            sentence_clean = sentence.strip()
            if not sentence_clean:
                continue
            
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence_clean if current_chunk else sentence_clean
            
            if len(potential_chunk) > self.chunk_size and current_chunk:
                # Finalize current chunk
                chunk_data = self._create_chunk_metadata(
                    chunk_id, current_chunk.strip(), sentence_start, i
                )
                chunks.append(chunk_data)
                
                # Start new chunk with overlap
                overlap_text = self._create_smart_overlap(current_chunk, sentence_clean)
                current_chunk = overlap_text
                sentence_start = max(0, i - 1)
                chunk_id += 1
            else:
                current_chunk = potential_chunk
        
        # Add final chunk
        if current_chunk.strip():
            chunk_data = self._create_chunk_metadata(
                chunk_id, current_chunk.strip(), sentence_start, len(sentences)
            )
            chunks.append(chunk_data)
        
        return chunks
    
    def _create_chunk_metadata(self, chunk_id: int, text: str, start_sentence: int, end_sentence: int) -> Dict[str, any]:
        """Create comprehensive metadata for a chunk"""
        return {
            'id': chunk_id,
            'text': text,
            'start_sentence': start_sentence,
            'end_sentence': end_sentence,
            'length': len(text),
            'word_count': len(text.split()),
            'chunk_type': self._classify_chunk_type(text),
            'key_terms': self._extract_key_terms(text),
            'sentence_count': end_sentence - start_sentence,
            'has_numbers': bool(re.search(r'\d+', text)),
            'has_monetary': bool(re.search(r'\$|USD|dollars?|cents?', text, re.IGNORECASE)),
            'has_dates': bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}', text))
        }
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from chunk text"""
        # Simple keyword extraction
        words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'that', 'this', 'with', 'from', 'they', 'been', 'have', 'were', 'said', 'each', 
            'which', 'their', 'time', 'will', 'about', 'would', 'there', 'could', 'other',
            'more', 'very', 'what', 'know', 'just', 'first', 'into', 'over', 'think', 'also'
        }
        
        key_terms = [word for word in words if word not in stop_words]
        
        # Return top 10 most frequent terms
        from collections import Counter
        term_counts = Counter(key_terms)
        return [term for term, count in term_counts.most_common(10)]
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text with improved preprocessing"""
        try:
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Fix common PDF extraction issues
            text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # Fix hyphenated words
            text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)  # Ensure space after sentences
            
            # Clean up special characters but preserve important punctuation
            text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\/\%\$\'\"]', ' ', text)
            
            # Clean up multiple periods and standardize ellipses
            text = re.sub(r'\.{3,}', '...', text)
            text = re.sub(r'\.{2}', '.', text)
            
            # Fix spacing around punctuation
            text = re.sub(r'\s+([,.;:!?])', r'\1', text)
            text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with improved sentence boundary detection"""
        try:
            sentences = []
            
            # Enhanced sentence splitting with multiple patterns
            # Split on sentence endings followed by space and capital letter
            parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                
                # Further split very long parts on other punctuation
                if len(part) > self.chunk_size * 0.8:
                    # Split on semicolons and colons followed by space
                    sub_parts = re.split(r'(?<=[;:])\s+', part)
                    for sub_part in sub_parts:
                        if len(sub_part.strip()) > 15:  # Minimum sentence length
                            sentences.append(sub_part.strip())
                else:
                    if len(part) > 15:  # Minimum sentence length
                        sentences.append(part)
            
            # Final cleanup - remove very short or empty sentences
            sentences = [s for s in sentences if len(s.strip()) > 20 and not s.strip().isdigit()]
            
            return sentences
            
        except Exception as e:
            logger.error(f"Error splitting sentences: {str(e)}")
            # Simple fallback
            return [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    
    def _create_smart_overlap(self, current_chunk: str, next_sentence: str) -> str:
        """Create intelligent overlap based on content"""
        try:
            # Find the last sentence in current chunk
            sentences_in_chunk = re.split(r'[.!?]+', current_chunk)
            sentences_in_chunk = [s.strip() for s in sentences_in_chunk if s.strip()]
            
            if not sentences_in_chunk:
                return next_sentence
            
            # Take last sentence as overlap if it's not too long
            last_sentence = sentences_in_chunk[-1]
            if len(last_sentence) <= self.chunk_overlap:
                return last_sentence + " " + next_sentence
            else:
                # Take last few words instead
                words = last_sentence.split()
                overlap_words = words[-(self.chunk_overlap // 10):]  # Roughly 10 chars per word
                return " ".join(overlap_words) + " " + next_sentence
                
        except Exception as e:
            logger.debug(f"Error creating smart overlap: {e}")
            # Fallback to simple word-based overlap
            words = current_chunk.split()
            if len(words) <= self.chunk_overlap // 10:
                return current_chunk + " " + next_sentence
            
            overlap_words = words[-(self.chunk_overlap // 10):]
            return " ".join(overlap_words) + " " + next_sentence
    
    def _classify_chunk_type(self, text: str) -> str:
        """Classify the type of content in a chunk with improved patterns"""
        try:
            text_lower = text.lower()
            
            # Insurance-related keywords
            insurance_keywords = ['claim', 'coverage', 'premium', 'deductible', 'policy', 'benefit', 'copay']
            medical_keywords = ['procedure', 'surgery', 'treatment', 'medical', 'doctor', 'hospital', 'diagnosis']
            legal_keywords = ['legal', 'contract', 'agreement', 'terms', 'clause', 'liability', 'obligations']
            hr_keywords = ['employee', 'hr', 'human resources', 'benefits', 'workplace', 'personnel']
            financial_keywords = ['amount', 'cost', 'price', 'fee', 'payment', 'reimbursement', 'expense']
            
            # Count keyword matches
            keyword_counts = {
                'insurance': sum(1 for kw in insurance_keywords if kw in text_lower),
                'medical': sum(1 for kw in medical_keywords if kw in text_lower),
                'legal': sum(1 for kw in legal_keywords if kw in text_lower),
                'hr': sum(1 for kw in hr_keywords if kw in text_lower),
                'financial': sum(1 for kw in financial_keywords if kw in text_lower)
            }
            
            # Return category with highest count
            if max(keyword_counts.values()) > 0:
                return max(keyword_counts, key=keyword_counts.get)
            else:
                return 'general'
                
        except Exception:
            return 'general'
    
    def _create_simple_chunks(self, text: str) -> List[Dict[str, any]]:
        """Fallback method to create simple chunks if main method fails"""
        try:
            chunks = []
            words = text.split()
            
            # Simple word-based chunking
            words_per_chunk = self.chunk_size // 6  # Roughly 6 chars per word
            
            for i in range(0, len(words), words_per_chunk):
                chunk_words = words[i:i + words_per_chunk]
                chunk_text = " ".join(chunk_words)
                
                chunks.append({
                    'id': i // words_per_chunk,
                    'text': chunk_text,
                    'start_sentence': i,
                    'end_sentence': i + len(chunk_words),
                    'length': len(chunk_text),
                    'word_count': len(chunk_words),
                    'chunk_type': 'general',
                    'key_terms': [],
                    'sentence_count': 1,
                    'has_numbers': bool(re.search(r'\d+', chunk_text)),
                    'has_monetary': bool(re.search(r'\$|USD|dollars?|cents?', chunk_text, re.IGNORECASE)),
                    'has_dates': False
                })
            
            logger.info(f"Created {len(chunks)} simple fallback chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Even fallback chunking failed: {str(e)}")
            return []
    
    def _cache_chunks(self, text_hash: str, chunks: List[Dict]):
        """Cache chunks with size management"""
        if len(self._chunk_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._chunk_cache))
            del self._chunk_cache[oldest_key]
        
        self._chunk_cache[text_hash] = chunks
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get caching statistics"""
        return {
            'cache_size': len(self._chunk_cache),
            'cache_max_size': self._cache_max_size
        }
    
    def clear_cache(self):
        """Clear chunk cache"""
        self._chunk_cache.clear()
        logger.info("Chunk cache cleared")

class SemanticSearch:
    """
    Handles semantic search using sentence transformers and FAISS with improved performance and caching
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", enable_persistence: bool = False):
        self.enable_persistence = enable_persistence
        self.index_cache_dir = "search_indexes" if enable_persistence else None
        
        if self.enable_persistence and not os.path.exists(self.index_cache_dir):
            os.makedirs(self.index_cache_dir)
        
        try:
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            # Try fallback models in order of preference
            fallback_models = [
                "paraphrase-MiniLM-L3-v2", 
                "all-mpnet-base-v2",
                "paraphrase-albert-small-v2"
            ]
            
            for fallback_model in fallback_models:
                try:
                    self.model = SentenceTransformer(fallback_model)
                    self.model_name = fallback_model
                    logger.info(f"Loaded fallback embedding model: {fallback_model}")
                    break
                except Exception as e2:
                    logger.warning(f"Failed to load fallback model {fallback_model}: {str(e2)}")
                    continue
            else:
                raise Exception("Could not load any embedding model")
        
        # Search result cache
        self._search_cache = {}
        self._search_cache_max_size = 100
    
    def build_index(self, chunks: List[Dict[str, any]], index_name: Optional[str] = None) -> Tuple[faiss.Index, np.ndarray]:
        """
        Build FAISS index from document chunks with caching and persistence
        
        Args:
            chunks: List of chunk dictionaries
            index_name: Optional name for persistent storage
            
        Returns:
            Tuple of (FAISS index, embeddings array)
        """
        if not chunks:
            raise ValueError("No chunks provided to build index")
        
        # Check for cached index if persistence is enabled
        if self.enable_persistence and index_name:
            cached_index = self._load_cached_index(index_name, len(chunks))
            if cached_index:
                return cached_index
        
        try:
            # Extract text from chunks
            texts = [chunk.get('text', '') for chunk in chunks]
            texts = [t for t in texts if t.strip()]  # Remove empty texts
            
            if not texts:
                raise ValueError("No valid text found in chunks")
            
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            
            # Generate embeddings with optimized parameters
            embeddings = self.model.encode(
                texts, 
                show_progress_bar=len(texts) > 20,
                batch_size=min(32, len(texts)),  # Dynamic batch size
                convert_to_numpy=True,
                normalize_embeddings=True  # Pre-normalize for cosine similarity
            )
            
            # Validate embeddings
            if embeddings is None or embeddings.size == 0:
                raise ValueError("Failed to generate embeddings")
            
            # Build optimized FAISS index
            index = self._build_optimized_index(embeddings)
            
            # Save index if persistence is enabled
            if self.enable_persistence and index_name:
                self._save_index(index_name, index, embeddings, len(chunks))
            
            logger.info(f"Built FAISS index with {index.ntotal} vectors of dimension {embeddings.shape[1]}")
            return index, embeddings
            
        except Exception as e:
            logger.error(f"Error building search index: {str(e)}")
            raise Exception(f"Failed to build search index: {str(e)}")
    
    def _build_optimized_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build optimized FAISS index based on data size"""
        dimension = embeddings.shape[1]
        n_vectors = len(embeddings)
        
        # Choose index type based on dataset size and requirements
        if n_vectors < 1000:
            # Small dataset: use flat index (exact search)
            index = faiss.IndexFlatIP(dimension)
        elif n_vectors < 10000:
            # Medium dataset: use IVF with reasonable number of clusters
            nlist = min(100, n_vectors // 10)
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            index.nprobe = min(10, nlist)
        else:
            # Large dataset: use more sophisticated indexing
            nlist = min(1000, n_vectors // 50)
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            index.nprobe = min(20, nlist)
        
        # Train index if needed
        if hasattr(index, 'train'):
            logger.info("Training FAISS index...")
            index.train(embeddings.astype('float32'))
        
        # Add vectors to index
        index.add(embeddings.astype('float32'))
        
        return index
    
    def search(self, index_data: Tuple[faiss.Index, np.ndarray], query: str, 
               chunks: List[Dict[str, any]], top_k: int = 5) -> List[Dict[str, any]]:
        """
        Search for most relevant chunks with improved ranking and caching
        
        Args:
            index_data: Tuple of (FAISS index, embeddings)
            query: Search query string
            chunks: Original chunks list
            top_k: Number of top results to return
            
        Returns:
            List of chunks with relevance scores
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to search")
            return []
        
        # Check cache first
        cache_key = self._create_search_cache_key(query, top_k, len(chunks))
        if cache_key in self._search_cache:
            logger.debug("Using cached search results")
            return self._search_cache[cache_key]
        
        try:
            index, embeddings = index_data
            
            if index.ntotal == 0:
                logger.warning("Search index is empty")
                return []
            
            # Encode query with same normalization
            query_embedding = self.model.encode([query.strip()], convert_to_numpy=True, normalize_embeddings=True)
            
            # Ensure we don't ask for more results than we have
            search_k = min(top_k * 2, index.ntotal, len(chunks))  # Get more candidates for reranking
            
            # Search
            scores, indices = index.search(query_embedding.astype('float32'), search_k)
            
            # Process and rerank results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(chunks) and idx >= 0:  # Valid index
                    chunk = chunks[idx].copy()
                    chunk['relevance_score'] = float(score)
                    chunk['rank'] = i + 1
                    
                    # Enhanced scoring with multiple factors
                    enhanced_score = self._calculate_enhanced_score(query, chunk, float(score))
                    chunk['enhanced_score'] = enhanced_score
                    
                    results.append(chunk)
            
            # Sort by enhanced score and limit to top_k
            results.sort(key=lambda x: x['enhanced_score'], reverse=True)
            results = results[:top_k]
            
            # Update final ranks
            for i, result in enumerate(results):
                result['final_rank'] = i + 1
            
            # Cache results
            self._cache_search_results(cache_key, results)
            
            logger.info(f"Retrieved {len(results)} relevant chunks for query")
            
            if results:
                logger.debug(f"Top result enhanced score: {results[0]['enhanced_score']:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return []
    
    def _calculate_enhanced_score(self, query: str, chunk: Dict, semantic_score: float) -> float:
        """Calculate enhanced score using multiple ranking factors"""
        try:
            # Start with semantic similarity score
            enhanced_score = semantic_score
            
            # Factor 1: Keyword matching bonus
            keyword_bonus = self._calculate_keyword_bonus(query, chunk['text'])
            enhanced_score += keyword_bonus * 0.1
            
            # Factor 2: Chunk type relevance
            type_bonus = self._calculate_type_bonus(query, chunk.get('chunk_type', 'general'))
            enhanced_score += type_bonus * 0.05
            
            # Factor 3: Content richness (longer chunks with more information)
            length_bonus = min(chunk.get('word_count', 0) / 100, 0.1)  # Cap at 0.1
            enhanced_score += length_bonus * 0.02
            
            # Factor 4: Special content indicators
            if chunk.get('has_numbers') and any(char.isdigit() for char in query):
                enhanced_score += 0.02
            
            if chunk.get('has_monetary') and ('cost' in query.lower() or 'price' in query.lower()):
                enhanced_score += 0.02
            
            # Factor 5: Key terms match
            query_terms = set(query.lower().split())
            chunk_key_terms = set(chunk.get('key_terms', []))
            term_overlap = len(query_terms.intersection(chunk_key_terms))
            if term_overlap > 0:
                enhanced_score += (term_overlap / len(query_terms)) * 0.05
            
            return enhanced_score
            
        except Exception as e:
            logger.debug(f"Error calculating enhanced score: {e}")
            return semantic_score
    
    def _calculate_keyword_bonus(self, query: str, text: str) -> float:
        """Calculate bonus score based on exact keyword matches"""
        try:
            query_words = set(word.lower() for word in query.split() if len(word) > 3)
            text_words = set(word.lower() for word in re.findall(r'\b\w+\b', text))
            
            if not query_words:
                return 0.0
            
            # Calculate overlap
            common_words = query_words.intersection(text_words)
            overlap_ratio = len(common_words) / len(query_words)
            
            return overlap_ratio
            
        except Exception:
            return 0.0
    
    def _calculate_type_bonus(self, query: str, chunk_type: str) -> float:
        """Calculate bonus based on query-chunk type alignment"""
        try:
            query_lower = query.lower()
            
            type_indicators = {
                'insurance': ['coverage', 'policy', 'claim', 'premium', 'deductible'],
                'medical': ['surgery', 'treatment', 'procedure', 'medical', 'doctor'],
                'legal': ['contract', 'agreement', 'legal', 'terms', 'clause'],
                'hr': ['employee', 'benefits', 'workplace', 'hr'],
                'financial': ['cost', 'price', 'amount', 'fee', 'payment']
            }
            
            # Check if query indicates specific domain
            for domain, indicators in type_indicators.items():
                if any(indicator in query_lower for indicator in indicators):
                    if chunk_type == domain:
                        return 1.0
                    elif chunk_type == 'general':
                        return 0.5
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _create_search_cache_key(self, query: str, top_k: int, chunk_count: int) -> str:
        """Create cache key for search results"""
        key_data = f"{query}_{top_k}_{chunk_count}_{self.model_name}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _cache_search_results(self, cache_key: str, results: List[Dict]):
        """Cache search results with size management"""
        if len(self._search_cache) >= self._search_cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._search_cache))
            del self._search_cache[oldest_key]
        
        self._search_cache[cache_key] = results
    
    def _load_cached_index(self, index_name: str, chunk_count: int) -> Optional[Tuple[faiss.Index, np.ndarray]]:
        """Load cached index if available and valid"""
        try:
            index_path = os.path.join(self.index_cache_dir, f"{index_name}.index")
            embeddings_path = os.path.join(self.index_cache_dir, f"{index_name}.embeddings")
            metadata_path = os.path.join(self.index_cache_dir, f"{index_name}.metadata")
            
            if not all(os.path.exists(p) for p in [index_path, embeddings_path, metadata_path]):
                return None
            
            # Check metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            if metadata.get('chunk_count') != chunk_count or metadata.get('model_name') != self.model_name:
                return None
            
            # Load index and embeddings
            index = faiss.read_index(index_path)
            with open(embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)
            
            logger.info(f"Loaded cached index: {index_name}")
            return index, embeddings
            
        except Exception as e:
            logger.debug(f"Failed to load cached index {index_name}: {e}")
            return None
    
    def _save_index(self, index_name: str, index: faiss.Index, embeddings: np.ndarray, chunk_count: int):
        """Save index to cache"""
        try:
            index_path = os.path.join(self.index_cache_dir, f"{index_name}.index")
            embeddings_path = os.path.join(self.index_cache_dir, f"{index_name}.embeddings")
            metadata_path = os.path.join(self.index_cache_dir, f"{index_name}.metadata")
            
            # Save index
            faiss.write_index(index, index_path)
            
            # Save embeddings
            with open(embeddings_path, 'wb') as f:
                pickle.dump(embeddings, f)
            
            # Save metadata
            metadata = {
                'chunk_count': chunk_count,
                'model_name': self.model_name,
                'created_at': time.time()
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Saved index to cache: {index_name}")
            
        except Exception as e:
            logger.warning(f"Failed to save index {index_name}: {e}")
    
    def get_model_info(self) -> Dict[str, any]:
        """Get comprehensive information about the loaded model"""
        try:
            return {
                'model_name': self.model_name,
                'max_seq_length': getattr(self.model, 'max_seq_length', 'unknown'),
                'embedding_dimension': self.model.get_sentence_embedding_dimension(),
                'device': str(self.model.device) if hasattr(self.model, 'device') else 'unknown',
                'search_cache_size': len(self._search_cache),
                'persistence_enabled': self.enable_persistence,
                'cache_directory': self.index_cache_dir
            }
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {'error': str(e)}
    
    def clear_caches(self):
        """Clear all caches"""
        self._search_cache.clear()
        logger.info("Search caches cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get caching statistics"""
        return {
            'search_cache_size': len(self._search_cache),
            'search_cache_max_size': self._search_cache_max_size
        }