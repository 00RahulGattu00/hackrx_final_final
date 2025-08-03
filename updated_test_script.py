#!/usr/bin/env python3
"""
Comprehensive system test script for the updated LLM Query-Retrieval System
Run this to validate all components work properly together
"""

import os
import sys
import time
from dotenv import load_dotenv
import logging

# Setup clean logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise during testing
    format='%(levelname)s: %(message)s'
)

def test_environment_setup():
    """Test environment variables and basic setup"""
    print("🔧 Testing Environment Setup...")
    print("=" * 50)
    
    load_dotenv()
    
    # Check for .env file
    if os.path.exists('.env'):
        print("✅ .env file found")
    else:
        print("⚠️  .env file not found (using environment variables)")
    
    # Check API keys
    api_keys = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'GROQ_API_KEY': os.getenv('GROQ_API_KEY'),
        'HUGGINGFACE_API_KEY': os.getenv('HUGGINGFACE_API_KEY'),
        'COHERE_API_KEY': os.getenv('COHERE_API_KEY'),
        'BEARER_TOKEN': os.getenv('BEARER_TOKEN')
    }
    
    configured_keys = []
    for name, key in api_keys.items():
        if key and key.strip():
            print(f"✅ {name}: Configured")
            configured_keys.append(name)
        else:
            print(f"❌ {name}: Not configured")
    
    print(f"\n📊 Summary: {len(configured_keys)}/5 keys configured")
    return len(configured_keys) >= 2  # Need at least 2 keys (1 API key + bearer token)

def test_imports():
    """Test that all modules can be imported"""
    print("\n📦 Testing Module Imports...")
    print("=" * 50)
    
    modules_to_test = [
        ('extract', 'extract_text_from_pdf'),
        ('search', 'DocumentProcessor'),
        ('search', 'SemanticSearch'),
        ('llm_processor', 'LLMProcessor'),
        ('decision_engine', 'DecisionEngine')
    ]
    
    failed_imports = []
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name)
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
        except ImportError as e:
            print(f"❌ {module_name}.{class_name}: Import Error - {e}")
            failed_imports.append((module_name, str(e)))
        except AttributeError as e:
            print(f"❌ {module_name}.{class_name}: Attribute Error - {e}")
            failed_imports.append((module_name, str(e)))
        except Exception as e:
            print(f"⚠️  {module_name}.{class_name}: Other Error - {e}")
            failed_imports.append((module_name, str(e)))
    
    if failed_imports:
        print(f"\n❌ {len(failed_imports)} import failures detected")
        for module, error in failed_imports:
            print(f"   - {module}: {error}")
        return False
    else:
        print(f"\n✅ All {len(modules_to_test)} modules imported successfully")
        return True

def test_component_initialization():
    """Test that all components can be initialized"""
    print("\n🏗️  Testing Component Initialization...")
    print("=" * 50)
    
    components = {}
    
    # Test DocumentProcessor
    try:
        from search import DocumentProcessor
        components['document_processor'] = DocumentProcessor()
        print("✅ DocumentProcessor initialized")
    except Exception as e:
        print(f"❌ DocumentProcessor failed: {e}")
        return False, components
    
    # Test SemanticSearch
    try:
        from search import SemanticSearch
        print("   Loading embedding model (this may take a moment)...")
        components['semantic_search'] = SemanticSearch()
        print("✅ SemanticSearch initialized")
    except Exception as e:
        print(f"❌ SemanticSearch failed: {e}")
        return False, components
    
    # Test LLMProcessor
    try:
        from llm_processor import LLMProcessor
        components['llm_processor'] = LLMProcessor()
        api_count = len(components['llm_processor'].apis)
        print(f"✅ LLMProcessor initialized with {api_count} APIs")
        
        if api_count == 0:
            print("⚠️  Warning: No API keys configured for LLM processing")
    except Exception as e:
        print(f"❌ LLMProcessor failed: {e}")
        return False, components
    
    # Test DecisionEngine
    try:
        from decision_engine import DecisionEngine
        components['decision_engine'] = DecisionEngine(components['llm_processor'])
        print("✅ DecisionEngine initialized")
    except Exception as e:
        print(f"❌ DecisionEngine failed: {e}")
        return False, components
    
    print(f"\n✅ All {len(components)} components initialized successfully")
    return True, components

def test_pdf_processing():
    """Test PDF processing with a simple test"""
    print("\n📄 Testing PDF Processing...")
    print("=" * 50)
    
    try:
        from extract import extract_text_from_pdf, validate_pdf
        
        # Create a simple test
        print("   Note: PDF processing requires a real PDF file")
        print("   Testing validation functions...")
        
        # Test with non-existent file
        if not validate_pdf("nonexistent.pdf"):
            print("✅ PDF validation correctly rejects non-existent file")
        else:
            print("⚠️  PDF validation should reject non-existent file")
        
        # Check if there are any PDF files in current directory for testing
        pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
        if pdf_files:
            test_pdf = pdf_files[0]
            print(f"   Found test PDF: {test_pdf}")
            if validate_pdf(test_pdf):
                print(f"✅ PDF validation passed for {test_pdf}")
                # Try to extract a small sample
                try:
                    text = extract_text_from_pdf(test_pdf, max_pages=1)
                    if text and len(text.strip()) > 0:
                        print(f"✅ Successfully extracted {len(text)} characters")
                    else:
                        print("⚠️  PDF text extraction returned empty result")
                except Exception as e:
                    print(f"⚠️  PDF text extraction failed: {e}")
            else:
                print(f"❌ PDF validation failed for {test_pdf}")
        else:
            print("   No PDF files found for testing")
        
        print("✅ PDF processing functions available")
        return True
        
    except Exception as e:
        print(f"❌ PDF processing test failed: {e}")
        return False

def test_semantic_search(components):
    """Test semantic search functionality"""
    print("\n🔍 Testing Semantic Search...")
    print("=" * 50)
    
    try:
        doc_processor = components['document_processor']
        semantic_search = components['semantic_search']
        
        # Test with sample text
        test_text = """
        This is a test document about insurance policies. 
        The policy covers medical procedures including surgery.
        There is a waiting period of 2 years for pre-existing conditions.
        Claims must be submitted within 30 days of treatment.
        The premium is $200 per month for comprehensive coverage.
        Emergency procedures are covered immediately without waiting periods.
        Dental coverage requires a separate premium of $50 per month.
        """
        
        print("   Creating chunks from test text...")
        chunks = doc_processor.create_chunks(test_text)
        print(f"✅ Created {len(chunks)} chunks")
        
        if not chunks:
            print("❌ No chunks created from test text")
            return False
        
        print("   Building search index...")
        search_index = semantic_search.build_index(chunks)
        print("✅ Search index built successfully")
        
        print("   Testing search functionality...")
        test_queries = [
            "What is the waiting period for surgery?",
            "How much is the monthly premium?",
            "What about dental coverage?"
        ]
        
        all_searches_passed = True
        for query in test_queries:
            print(f"   Searching: '{query}'")
            results = semantic_search.search(search_index, query, chunks, top_k=3)
            
            if results:
                print(f"     ✅ Found {len(results)} results")
                print(f"     Top result relevance: {results[0].get('relevance_score', 0):.3f}")
            else:
                print(f"     ❌ No results for query: {query}")
                all_searches_passed = False
        
        return all_searches_passed
            
    except Exception as e:
        print(f"❌ Semantic search test failed: {e}")
        return False

def test_llm_processing(components):
    """Test LLM processing capabilities"""
    print("\n🧠 Testing LLM Processing...")
    print("=" * 50)
    
    try:
        llm_processor = components['llm_processor']
        
        # Test query parsing (fallback mode)
        print("   Testing query parsing (fallback mode)...")
        test_query = "What is the waiting period for knee surgery?"
        parsed_query = llm_processor._fallback_parse_query(test_query)
        
        if parsed_query and isinstance(parsed_query, dict):
            print("✅ Query parsing (fallback) works")
            print(f"   Parsed components: {list(parsed_query.keys())}")
        else:
            print("❌ Query parsing (fallback) failed")
            return False
        
        # Test LLM API availability
        print("   Testing LLM API availability...")
        available_apis = llm_processor.get_available_apis()
        if available_apis:
            print(f"✅ {len(available_apis)} APIs available: {', '.join(available_apis)}")
            
            # Test a simple query with the first available API
            test_context = "The insurance policy has a 2-year waiting period for pre-existing conditions."
            test_question = "How long is the waiting period?"
            
            print("   Testing LLM query processing...")
            try:
                response = llm_processor.process_query(
                    query=test_question,
                    context=test_context,
                    preferred_api=available_apis[0]
                )
                
                if response and len(response.strip()) > 0:
                    print("✅ LLM processing successful")
                    print(f"   Response length: {len(response)} characters")
                else:
                    print("⚠️  LLM processing returned empty response")
                    
            except Exception as e:
                print(f"⚠️  LLM processing failed: {e}")
                
        else:
            print("❌ No LLM APIs available")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ LLM processing test failed: {e}")
        return False

def test_decision_engine(components):
    """Test decision engine functionality"""
    print("\n🎯 Testing Decision Engine...")
    print("=" * 50)
    
    try:
        decision_engine = components['decision_engine']
        
        # Test query classification
        print("   Testing query classification...")
        test_queries = [
            "What is the waiting period for surgery?",
            "Calculate the monthly premium for me",
            "Hello, how are you today?",
            "Find all policies covering dental work"
        ]
        
        classifications_passed = 0
        for query in test_queries:
            try:
                classification = decision_engine.classify_query(query)
                if classification and isinstance(classification, dict):
                    print(f"   ✅ '{query}' -> {classification.get('type', 'unknown')}")
                    classifications_passed += 1
                else:
                    print(f"   ❌ Failed to classify: '{query}'")
            except Exception as e:
                print(f"   ⚠️  Classification error for '{query}': {e}")
        
        if classifications_passed >= len(test_queries) // 2:
            print(f"✅ Query classification working ({classifications_passed}/{len(test_queries)} passed)")
        else:
            print(f"❌ Query classification needs improvement ({classifications_passed}/{len(test_queries)} passed)")
            return False
        
        # Test decision making
        print("   Testing decision making...")
        test_context = {
            'query': 'What is the waiting period for surgery?',
            'has_context': True,
            'context_relevance': 0.8
        }
        
        try:
            decision = decision_engine.make_decision(test_context)
            if decision and isinstance(decision, dict):
                print(f"✅ Decision making successful")
                print(f"   Decision: {decision.get('action', 'unknown')}")
            else:
                print("❌ Decision making failed")
                return False
        except Exception as e:
            print(f"❌ Decision making error: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Decision engine test failed: {e}")
        return False

def test_integration(components):
    """Test full system integration"""
    print("\n🔗 Testing System Integration...")
    print("=" * 50)
    
    try:
        # Create a complete test scenario
        doc_processor = components['document_processor']
        semantic_search = components['semantic_search']
        decision_engine = components['decision_engine']
        
        # Sample document
        test_document = """
        HEALTH INSURANCE POLICY DOCUMENT
        
        Coverage Details:
        - Medical procedures: Covered after 6 months
        - Surgery: Covered after 2 years for pre-existing conditions
        - Emergency surgery: Covered immediately
        - Dental work: Requires separate dental plan
        - Vision care: 50% coverage after 1 year
        
        Premium Information:
        - Basic plan: $150/month
        - Comprehensive plan: $250/month
        - Family plan: $400/month
        
        Claims Process:
        - Submit claims within 30 days
        - Required documents: receipts, medical reports
        - Processing time: 7-14 business days
        """
        
        print("   Processing test document...")
        chunks = doc_processor.create_chunks(test_document)
        search_index = semantic_search.build_index(chunks)
        
        # Test end-to-end query processing
        test_queries = [
            "What is the waiting period for surgery?",
            "How much does the family plan cost?",
            "How long do I have to submit a claim?"
        ]
        
        successful_queries = 0
        for query in test_queries:
            print(f"   Processing query: '{query}'")
            
            try:
                # Search for relevant context
                search_results = semantic_search.search(search_index, query, chunks, top_k=3)
                
                if search_results:
                    context = search_results[0]['text']
                    
                    # Classify query
                    classification = decision_engine.classify_query(query)
                    
                    # Make decision
                    decision_context = {
                        'query': query,
                        'has_context': True,
                        'context_relevance': search_results[0].get('relevance_score', 0)
                    }
                    decision = decision_engine.make_decision(decision_context)
                    
                    print(f"     ✅ End-to-end processing successful")
                    print(f"     Query type: {classification.get('type', 'unknown')}")
                    print(f"     Action: {decision.get('action', 'unknown')}")
                    successful_queries += 1
                    
                else:
                    print(f"     ❌ No search results found")
                    
            except Exception as e:
                print(f"     ❌ Integration test failed: {e}")
        
        if successful_queries == len(test_queries):
            print(f"✅ All integration tests passed ({successful_queries}/{len(test_queries)})")
            return True
        else:
            print(f"⚠️  Some integration tests failed ({successful_queries}/{len(test_queries)})")
            return successful_queries > 0
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def run_performance_tests(components):
    """Run basic performance tests"""
    print("\n⚡ Running Performance Tests...")
    print("=" * 50)
    
    try:
        semantic_search = components['semantic_search']
        doc_processor = components['document_processor']
        
        # Create a larger test document
        large_text = """
        This is a large test document for performance testing. """ * 100
        
        # Time chunk creation
        start_time = time.time()
        chunks = doc_processor.create_chunks(large_text)
        chunk_time = time.time() - start_time
        print(f"✅ Chunk creation: {chunk_time:.3f}s for {len(chunks)} chunks")
        
        # Time index building
        start_time = time.time()
        search_index = semantic_search.build_index(chunks)
        index_time = time.time() - start_time
        print(f"✅ Index building: {index_time:.3f}s")
        
        # Time search operations
        search_times = []
        for i in range(5):
            start_time = time.time()
            results = semantic_search.search(search_index, "test query", chunks, top_k=5)
            search_time = time.time() - start_time
            search_times.append(search_time)
        
        avg_search_time = sum(search_times) / len(search_times)
        print(f"✅ Average search time: {avg_search_time:.3f}s")
        
        # Performance thresholds (adjust as needed)
        if chunk_time > 5.0:
            print("⚠️  Chunk creation slower than expected")
        if index_time > 10.0:
            print("⚠️  Index building slower than expected")
        if avg_search_time > 1.0:
            print("⚠️  Search operations slower than expected")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 LLM Query-Retrieval System Test Suite")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print("=" * 60)
    
    # Track test results
    test_results = {}
    
    # Run tests in sequence
    test_results['environment'] = test_environment_setup()
    
    if not test_results['environment']:
        print("\n❌ Environment setup failed. Please check your configuration.")
        return False
    
    test_results['imports'] = test_imports()
    
    if not test_results['imports']:
        print("\n❌ Module imports failed. Please check your installation.")
        return False
    
    success, components = test_component_initialization()
    test_results['components'] = success
    
    if not success:
        print("\n❌ Component initialization failed.")
        return False
    
    # Continue with other tests
    test_results['pdf'] = test_pdf_processing()
    test_results['search'] = test_semantic_search(components)
    test_results['llm'] = test_llm_processing(components)
    test_results['decision'] = test_decision_engine(components)
    test_results['integration'] = test_integration(components)
    test_results['performance'] = run_performance_tests(components)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.upper():.<20} {status}")
        if result:
            passed_tests += 1
    
    print("=" * 60)
    print(f"OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Your system is ready to use.")
        return True
    elif passed_tests >= total_tests * 0.8:
        print("⚠️  Most tests passed. System should work with minor issues.")
        return True
    else:
        print("❌ Many tests failed. Please review your configuration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)