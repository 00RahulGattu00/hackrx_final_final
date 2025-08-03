from typing import Dict, List, Any
import logging
import time
from llm_processor import LLMProcessor

logger = logging.getLogger(__name__)

class DecisionEngine:
    """
    AI-powered decision engine that combines semantic search results 
    with LLM reasoning for explainable decisions
    """
    
    def __init__(self, llm_processor: LLMProcessor):
        self.llm_processor = llm_processor
        self.processing_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'fallback_queries': 0,
            'avg_processing_time': 0.0
        }
        logger.info("Initialized Decision Engine with LLM processor")
    
    def generate_answer(self, question: str, parsed_query: Dict[str, Any], 
                       context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive answer with explainable reasoning
        """
        start_time = time.time()
        self.processing_stats['total_queries'] += 1
        
        try:
            logger.info(f"Generating answer for question: {question[:50]}...")
            
            # Step 1: Analyze question intent and context quality
            analysis = self._analyze_context_quality(question, context_chunks)
            logger.debug(f"Context analysis: {analysis['quality']} quality, {analysis['relevance']:.3f} relevance")
            
            # Step 2: Generate answer using LLM with structured reasoning
            llm_result = self.llm_processor.generate_answer(
                question=question, 
                context="", 
                retrieved_chunks=context_chunks
            )
            
            # Step 3: Post-process and enhance the answer
            enhanced_answer = self._enhance_answer(
                question=question,
                llm_result=llm_result,
                parsed_query=parsed_query,
                context_chunks=context_chunks,
                analysis=analysis
            )
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats['successful_queries'] += 1
            self._update_avg_processing_time(processing_time)
            
            logger.info(f"Successfully generated answer in {processing_time:.2f}s")
            return enhanced_answer
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error in decision engine after {processing_time:.2f}s: {str(e)}")
            
            # Use fallback
            self.processing_stats['fallback_queries'] += 1
            fallback_result = self._fallback_answer(question, context_chunks)
            
            self._update_avg_processing_time(processing_time)
            return fallback_result
    
    def _analyze_context_quality(self, question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the quality and relevance of retrieved context with improved metrics
        """
        if not chunks:
            return {
                "quality": "poor",
                "relevance": 0.0,
                "coverage": "incomplete",
                "missing_info": ["No relevant document sections found"],
                "chunk_count": 0,
                "total_context_length": 0,
                "confidence_factors": []
            }
        
        try:
            # Calculate metrics
            relevance_scores = [chunk.get('relevance_score', 0) for chunk in chunks]
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            max_relevance = max(relevance_scores) if relevance_scores else 0
            min_relevance = min(relevance_scores) if relevance_scores else 0
            
            total_length = sum(len(chunk.get('text', '')) for chunk in chunks)
            avg_chunk_length = total_length / len(chunks)
            
            # Analyze chunk types for domain coverage
            chunk_types = [chunk.get('chunk_type', 'general') for chunk in chunks]
            unique_types = set(chunk_types)
            
            # Determine quality rating with improved logic
            confidence_factors = []
            
            if avg_relevance > 0.8:
                quality = "excellent"
                confidence_factors.append("high_semantic_similarity")
            elif avg_relevance > 0.6:
                quality = "good"
                confidence_factors.append("good_semantic_similarity")
            elif avg_relevance > 0.4:
                quality = "fair"
                confidence_factors.append("moderate_semantic_similarity")
            else:
                quality = "poor"
                confidence_factors.append("low_semantic_similarity")
            
            # Adjust based on content volume
            if total_length > 1500:
                confidence_factors.append("sufficient_content_volume")
            elif total_length > 800:
                confidence_factors.append("adequate_content_volume")
            else:
                confidence_factors.append("limited_content_volume")
                if quality == "excellent":
                    quality = "good"
                elif quality == "good":
                    quality = "fair"
            
            # Check for domain diversity
            if len(unique_types) > 1:
                confidence_factors.append("diverse_content_types")
            
            # Check for keyword matches (if available)
            question_words = set(question.lower().split())
            keyword_matches = 0
            for chunk in chunks:
                chunk_words = set(chunk.get('text', '').lower().split())
                keyword_matches += len(question_words.intersection(chunk_words))
            
            if keyword_matches > len(question_words):
                confidence_factors.append("strong_keyword_overlap")
            elif keyword_matches > 0:
                confidence_factors.append("some_keyword_overlap")
            
            # Determine coverage
            if avg_relevance > 0.7 and total_length > 1000:
                coverage = "comprehensive"
            elif avg_relevance > 0.5 and total_length > 500:
                coverage = "adequate"
            elif total_length > 200:
                coverage = "partial"
            else:
                coverage = "minimal"
            
            # Identify potential missing information
            missing_info = []
            if avg_relevance < 0.5:
                missing_info.append("Low relevance suggests important information may be missing")
            if total_length < 300:
                missing_info.append("Limited content available for comprehensive analysis")
            if len(chunks) < 3:
                missing_info.append("Few relevant sections found - answer may lack depth")
            
            return {
                "quality": quality,
                "relevance": avg_relevance,
                "max_relevance": max_relevance,
                "min_relevance": min_relevance,
                "coverage": coverage,
                "chunk_count": len(chunks),
                "total_context_length": total_length,
                "avg_chunk_length": avg_chunk_length,
                "content_types": list(unique_types),
                "confidence_factors": confidence_factors,
                "missing_info": missing_info,
                "keyword_matches": keyword_matches
            }
            
        except Exception as e:
            logger.error(f"Error analyzing context quality: {str(e)}")
            return {
                "quality": "unknown",
                "relevance": 0.0,
                "coverage": "unknown",
                "missing_info": [f"Analysis error: {str(e)}"],
                "chunk_count": len(chunks),
                "total_context_length": 0,
                "confidence_factors": ["analysis_error"]
            }
    
    def _enhance_answer(self, question: str, llm_result: Dict[str, Any], 
                       parsed_query: Dict[str, Any], context_chunks: List[Dict[str, Any]],
                       analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance the LLM-generated answer with additional structured information
        """
        try:
            # Extract key information from the answer
            answer_text = llm_result.get("answer", "")
            
            # Determine confidence level based on multiple factors
            confidence_level = self._determine_confidence_level(llm_result, analysis, context_chunks)
            
            # Add structured metadata
            enhanced_result = {
                "answer": answer_text,
                "metadata": {
                    "question_intent": parsed_query.get("intent", "unknown"),
                    "question_domain": parsed_query.get("domain", "unknown"),
                    "question_complexity": parsed_query.get("complexity", "unknown"),
                    "context_quality": analysis["quality"],
                    "confidence_level": confidence_level,
                    "processing_stats": {
                        "chunks_analyzed": len(context_chunks),
                        "avg_relevance": analysis["relevance"],
                        "context_coverage": analysis["coverage"],
                        "token_usage": llm_result.get("token_usage", 0),
                        "content_types": analysis.get("content_types", [])
                    }
                },
                "supporting_evidence": self._extract_supporting_evidence(context_chunks),
                "reasoning_chain": self._build_reasoning_chain(question, context_chunks, answer_text, analysis),
                "confidence_factors": analysis.get("confidence_factors", []),
                "limitations": self._identify_limitations(analysis, parsed_query, llm_result),
                "recommendations": self._generate_recommendations(analysis, parsed_query)
            }
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error enhancing answer: {str(e)}")
            # Return basic structure if enhancement fails
            return {
                "answer": llm_result.get("answer", "Error processing answer"),
                "metadata": {
                    "question_intent": parsed_query.get("intent", "unknown"),
                    "context_quality": analysis.get("quality", "unknown"),
                    "confidence_level": "low",
                    "processing_stats": {
                        "chunks_analyzed": len(context_chunks),
                        "error": str(e)
                    }
                },
                "supporting_evidence": [],
                "reasoning_chain": [{"step": "Error", "description": f"Enhancement failed: {str(e)}"}],
                "limitations": ["Answer enhancement failed"],
                "recommendations": []
            }
    
    def _determine_confidence_level(self, llm_result: Dict[str, Any], 
                                   analysis: Dict[str, Any], 
                                   context_chunks: List[Dict[str, Any]]) -> str:
        """
        Determine overall confidence level based on multiple factors
        """
        try:
            confidence_score = 0
            
            # Factor 1: Context quality (0-4 points)
            quality = analysis.get("quality", "poor")
            if quality == "excellent":
                confidence_score += 4
            elif quality == "good":
                confidence_score += 3
            elif quality == "fair":
                confidence_score += 2
            elif quality == "poor":
                confidence_score += 1
            
            # Factor 2: Relevance score (0-3 points)
            relevance = analysis.get("relevance", 0)
            if relevance > 0.8:
                confidence_score += 3
            elif relevance > 0.6:
                confidence_score += 2
            elif relevance > 0.4:
                confidence_score += 1
            
            # Factor 3: Content volume (0-2 points)
            content_length = analysis.get("total_context_length", 0)
            if content_length > 1000:
                confidence_score += 2
            elif content_length > 500:
                confidence_score += 1
            
            # Factor 4: LLM confidence (0-1 point)
            llm_confidence = llm_result.get("confidence", "low")
            if llm_confidence == "high":
                confidence_score += 1
            
            # Convert score to level (out of 10 points)
            if confidence_score >= 8:
                return "high"
            elif confidence_score >= 5:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            logger.error(f"Error determining confidence level: {str(e)}")
            return "low"
    
    def _extract_supporting_evidence(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract and structure supporting evidence from context chunks
        """
        try:
            evidence = []
            
            # Sort chunks by relevance score
            sorted_chunks = sorted(chunks, key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            for i, chunk in enumerate(sorted_chunks[:3]):  # Top 3 most relevant
                # Create excerpt
                text = chunk.get("text", "")
                if len(text) > 300:
                    excerpt = text[:297] + "..."
                else:
                    excerpt = text
                
                evidence.append({
                    "chunk_id": chunk.get("id", i),
                    "relevance_score": chunk.get("relevance_score", 0),
                    "combined_score": chunk.get("combined_score", chunk.get("relevance_score", 0)),
                    "excerpt": excerpt,
                    "full_length": len(text),
                    "chunk_type": chunk.get("chunk_type", "general"),
                    "rank": i + 1,
                    "word_count": chunk.get("word_count", len(text.split()))
                })
            
            return evidence
            
        except Exception as e:
            logger.error(f"Error extracting supporting evidence: {str(e)}")
            return []
    
    def _build_reasoning_chain(self, question: str, chunks: List[Dict[str, Any]], 
                              answer: str, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Build a step-by-step reasoning chain for explainability
        """
        try:
            reasoning_steps = []
            
            # Step 1: Question analysis
            reasoning_steps.append({
                "step": "Question Analysis",
                "description": f"Analyzed question intent and extracted key concepts from: '{question[:100]}{'...' if len(question) > 100 else ''}'"
            })
            
            # Step 2: Context retrieval
            if chunks:
                reasoning_steps.append({
                    "step": "Context Retrieval",
                    "description": f"Retrieved {len(chunks)} relevant document sections using semantic search with average relevance score of {analysis.get('relevance', 0):.3f}"
                })
            
            # Step 3: Content analysis
            if analysis.get("content_types"):
                types_str = ", ".join(analysis["content_types"])
                reasoning_steps.append({
                    "step": "Content Analysis",
                    "description": f"Analyzed content types: {types_str}. Context quality assessed as '{analysis.get('quality', 'unknown')}'"
                })
            
            # Step 4: Evidence evaluation
            if chunks:
                best_chunk = max(chunks, key=lambda x: x.get('relevance_score', 0))
                reasoning_steps.append({
                    "step": "Evidence Evaluation",
                    "description": f"Identified most relevant evidence with {best_chunk.get('relevance_score', 0):.3f} confidence match from {analysis.get('chunk_count', 0)} sections"
                })
            
            # Step 5: Answer synthesis
            reasoning_steps.append({
                "step": "Answer Synthesis",
                "description": f"Synthesized comprehensive answer from {analysis.get('total_context_length', 0)} characters of context with {analysis.get('coverage', 'unknown')} coverage"
            })
            
            # Step 6: Quality assessment
            confidence_factors = analysis.get("confidence_factors", [])
            if confidence_factors:
                reasoning_steps.append({
                    "step": "Quality Assessment",
                    "description": f"Evaluated answer quality based on: {', '.join(confidence_factors[:3])}"
                })
            
            return reasoning_steps
            
        except Exception as e:
            logger.error(f"Error building reasoning chain: {str(e)}")
            return [
                {
                    "step": "Error in Reasoning Chain",
                    "description": f"Could not build detailed reasoning: {str(e)}"
                }
            ]
    
    def _identify_limitations(self, analysis: Dict[str, Any], 
                            parsed_query: Dict[str, Any],
                            llm_result: Dict[str, Any]) -> List[str]:
        """
        Identify potential limitations in the answer
        """
        try:
            limitations = []
            
            # Context quality limitations
            if analysis.get("quality") == "poor":
                limitations.append("Limited relevant information found in the document")
            
            if analysis.get("relevance", 0) < 0.5:
                limitations.append("Low confidence in document relevance to the question")
            
            if analysis.get("chunk_count", 0) < 2:
                limitations.append("Answer based on limited document sections")
            
            if analysis.get("total_context_length", 0) < 300:
                limitations.append("Insufficient context available for comprehensive analysis")
            
            # Query complexity limitations
            if parsed_query.get("complexity") == "complex" and analysis.get("quality") != "excellent":
                limitations.append("Complex question may require additional context not available in the document")
            
            # Domain-specific limitations
            question_domain = parsed_query.get("domain", "unknown")
            context_types = analysis.get("content_types", [])
            
            if question_domain != "unknown" and question_domain not in context_types:
                limitations.append(f"Question appears to be about {question_domain} but available content may not fully address this domain")
            
            # LLM-specific limitations
            if llm_result.get("confidence") == "low":
                limitations.append("Language model expressed low confidence in the generated answer")
            
            # Missing information
            missing_info = analysis.get("missing_info", [])
            limitations.extend(missing_info[:2])  # Add top 2 missing info items
            
            return limitations[:5]  # Limit to top 5 limitations
            
        except Exception as e:
            logger.error(f"Error identifying limitations: {str(e)}")
            return ["Could not assess answer limitations due to processing error"]
    
    def _generate_recommendations(self, analysis: Dict[str, Any], 
                                 parsed_query: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations for improving answer quality
        """
        try:
            recommendations = []
            
            # Based on context quality
            if analysis.get("quality") == "poor":
                recommendations.append("Consider providing additional or more specific documents related to your question")
            
            if analysis.get("chunk_count", 0) < 3:
                recommendations.append("A longer or more comprehensive document might provide better answers")
            
            # Based on question complexity
            if parsed_query.get("complexity") == "complex":
                recommendations.append("For complex questions, consider breaking them into smaller, more specific queries")
            
            # Based on domain match
            question_domain = parsed_query.get("domain", "unknown")
            if question_domain != "unknown":
                recommendations.append(f"Ensure the document contains sufficient {question_domain}-related content")
            
            # Based on missing information
            if analysis.get("relevance", 0) < 0.6:
                recommendations.append("Try rephrasing your question or using different keywords")
            
            return recommendations[:3]  # Limit to top 3 recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []
    
    def _fallback_answer(self, question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate fallback answer when LLM processing fails
        """
        try:
            if not chunks:
                answer = "I could not find relevant information in the document to answer your question."
                evidence = []
            else:
                # Use the most relevant chunk
                best_chunk = chunks[0]
                chunk_text = best_chunk.get('text', '')
                
                if len(chunk_text) > 200:
                    answer = f"Based on the available information: {chunk_text[:200]}..."
                else:
                    answer = f"Based on the available information: {chunk_text}"
                
                evidence = [{
                    "chunk_id": best_chunk.get("id", 0),
                    "relevance_score": best_chunk.get("relevance_score", 0),
                    "excerpt": chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text,
                    "rank": 1
                }]
            
            return {
                "answer": answer,
                "metadata": {
                    "question_intent": "unknown",
                    "question_domain": "unknown",
                    "context_quality": "limited",
                    "confidence_level": "low",
                    "processing_stats": {
                        "chunks_analyzed": len(chunks),
                        "avg_relevance": sum(c.get('relevance_score', 0) for c in chunks) / len(chunks) if chunks else 0,
                        "token_usage": 0,
                        "fallback_used": True
                    }
                },
                "supporting_evidence": evidence,
                "reasoning_chain": [
                    {
                        "step": "Fallback Processing",
                        "description": "Used simplified processing due to system limitations - answer may be incomplete"
                    }
                ],
                "confidence_factors": ["fallback_processing"],
                "limitations": ["Processing limitations - answer may be incomplete", "LLM processing unavailable"],
                "recommendations": ["Try again later when full processing is available"]
            }
            
        except Exception as e:
            logger.error(f"Error in fallback answer generation: {str(e)}")
            return {
                "answer": "Unable to process question due to system error.",
                "metadata": {
                    "confidence_level": "none",
                    "processing_stats": {"error": str(e)}
                },
                "supporting_evidence": [],
                "reasoning_chain": [],
                "limitations": ["System error prevented processing"],
                "recommendations": []
            }
    
    def _update_avg_processing_time(self, processing_time: float):
        """Update average processing time statistics"""
        try:
            current_avg = self.processing_stats['avg_processing_time']
            total_queries = self.processing_stats['total_queries']
            
            # Calculate new average
            self.processing_stats['avg_processing_time'] = (
                (current_avg * (total_queries - 1)) + processing_time
            ) / total_queries
            
        except Exception:
            pass  # Don't fail the main process for stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics for monitoring"""
        try:
            total = self.processing_stats['total_queries']
            successful = self.processing_stats['successful_queries']
            
            return {
                "total_queries": total,
                "successful_queries": successful,
                "fallback_queries": self.processing_stats['fallback_queries'],
                "success_rate": (successful / total * 100) if total > 0 else 0,
                "avg_processing_time": round(self.processing_stats['avg_processing_time'], 3)
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"error": str(e)}