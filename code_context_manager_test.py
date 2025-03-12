import asyncio
import os
import time
from code_context_manager import CodeContextManager

async def test_code_context_manager():
    """Test the CodeContextManager functionality"""
    # Get the repository path (current directory)
    repo_path = os.getcwd()
    
    print(f"Testing CodeContextManager on repository: {repo_path}")
    
    # Create the code context manager
    manager = CodeContextManager(repo_path)
    
    # Start the manager
    manager.start()
    
    try:
        # Wait for initial scanning to complete
        print("Waiting for initial repository scan...")
        await asyncio.sleep(2)
        
        # Get statistics
        stats = manager.get_token_stats()
        print("\nRepository Statistics:")
        print(f"Total tokens: {stats['total_tokens']}")
        print(f"Total files: {stats['total_files']}")
        print("Token types:")
        for token_type, count in stats['token_types'].items():
            print(f"  {token_type}: {count}")
        
        # Test queries
        queries = [
            "Task management",
            "Agent class",
            "process function",
            "code generation",
            "neural network"
        ]
        
        for query in queries:
            print(f"\n\nGetting context for query: '{query}'")
            context = manager.get_context_for_query(query)
            
            # Print a preview of the context
            preview = context[:500] + "..." if len(context) > 500 else context
            print(f"Context preview:\n{preview}")
            
            # Search for code related to the query
            print(f"\nSearching for code related to: '{query}'")
            results = manager.search_code(query, limit=3)
            
            for i, result in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"  File: {result['file_path']}")
                print(f"  Type: {result['token_type']}")
                print(f"  Lines: {result['line_start']}-{result['line_end']}")
                print(f"  Score: {result['score']}")
                
                # Print a preview of the content
                content_preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                print(f"  Content preview:\n{content_preview}")
        
        # Test function lookup
        print("\n\nLooking up function definitions:")
        functions_to_lookup = ["main", "process_task", "generate_response"]
        
        for func_name in functions_to_lookup:
            func_def = manager.get_function_definition(func_name)
            if func_def:
                preview = func_def[:200] + "..." if len(func_def) > 200 else func_def
                print(f"\nFunction '{func_name}':\n{preview}")
            else:
                print(f"\nFunction '{func_name}' not found")
        
        # Test class lookup
        print("\n\nLooking up class definitions:")
        classes_to_lookup = ["Task", "Agent", "CodeManager"]
        
        for class_name in classes_to_lookup:
            class_def = manager.get_class_definition(class_name)
            if class_def:
                preview = class_def[:200] + "..." if len(class_def) > 200 else class_def
                print(f"\nClass '{class_name}':\n{preview}")
            else:
                print(f"\nClass '{class_name}' not found")
    
    finally:
        # Stop the manager
        print("\nStopping CodeContextManager...")
        manager.stop()
        print("CodeContextManager stopped")

if __name__ == "__main__":
    asyncio.run(test_code_context_manager())
