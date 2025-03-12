import os
import asyncio
import logging
from typing import Dict, List, Any, Optional
from code_context_manager import CodeContextManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AgentCodeIntegration")

class AgentCodeAssistant:
    """
    Integrates the CodeContextManager with an agent to provide code-aware capabilities.
    This class serves as middleware between agents and the code repository.
    """
    
    def __init__(self, repo_path: Optional[str] = None):
        """
        Initialize the agent code assistant.
        
        Args:
            repo_path: Path to the repository (defaults to current directory)
        """
        self.repo_path = repo_path or os.getcwd()
        self.code_manager = CodeContextManager(self.repo_path)
        self.running = False
        
        # Cache for recent queries to avoid redundant processing
        self.context_cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps = {}
    
    async def start(self):
        """Start the code assistant and initialize the code manager"""
        if self.running:
            return
        
        logger.info("Starting AgentCodeAssistant")
        self.code_manager.start()
        self.running = True
    
    async def stop(self):
        """Stop the code assistant and clean up resources"""
        if not self.running:
            return
        
        logger.info("Stopping AgentCodeAssistant")
        self.code_manager.stop()
        self.running = False
        self.context_cache.clear()
        self.cache_timestamps.clear()
    
    async def get_code_context(self, query: str, max_tokens: int = None) -> str:
        """
        Get code context for a query, with caching for efficiency.
        
        Args:
            query: The query to get context for
            max_tokens: Maximum number of tokens to include
            
        Returns:
            String containing the relevant code context
        """
        # Check cache first
        current_time = asyncio.get_event_loop().time()
        if query in self.context_cache:
            cache_time = self.cache_timestamps.get(query, 0)
            if current_time - cache_time < self.cache_ttl:
                logger.debug(f"Cache hit for query: {query}")
                return self.context_cache[query]
        
        # Get fresh context
        context = self.code_manager.get_context_for_query(query, max_tokens)
        
        # Update cache
        self.context_cache[query] = context
        self.cache_timestamps[query] = current_time
        
        # Clean old cache entries
        self._clean_cache(current_time)
        
        return context
    
    def _clean_cache(self, current_time: float):
        """Clean expired entries from the cache"""
        expired_queries = [
            query for query, timestamp in self.cache_timestamps.items()
            if current_time - timestamp >= self.cache_ttl
        ]
        
        for query in expired_queries:
            if query in self.context_cache:
                del self.context_cache[query]
            if query in self.cache_timestamps:
                del self.cache_timestamps[query]
    
    async def search_code(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for code matching the query.
        
        Args:
            query: The search query
            limit: Maximum number of results
            
        Returns:
            List of matching code snippets with metadata
        """
        return self.code_manager.search_code(query, limit)
    
    async def get_function_definition(self, function_name: str) -> Optional[str]:
        """Get the definition of a function by name"""
        return self.code_manager.get_function_definition(function_name)
    
    async def get_class_definition(self, class_name: str) -> Optional[str]:
        """Get the definition of a class by name"""
        return self.code_manager.get_class_definition(class_name)
    
    async def get_file_content(self, file_path: str) -> str:
        """Get the content of a file from the repository"""
        return self.code_manager.get_file_content(file_path)
    
    async def provide_feedback(self, token_id: str, was_helpful: bool):
        """
        Provide feedback on whether a token was helpful.
        This helps improve future context building.
        
        Args:
            token_id: ID of the token to provide feedback for
            was_helpful: Whether the token was helpful
        """
        importance_delta = 0.5 if was_helpful else -0.3
        self.code_manager.update_token_importance(token_id, importance_delta)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the code manager"""
        return self.code_manager.get_token_stats()

async def demo():
    """Demonstrate the AgentCodeAssistant functionality"""
    assistant = AgentCodeAssistant()
    await assistant.start()
    
    try:
        # Get code context for a task
        print("Getting code context for 'task processing'...")
        context = await assistant.get_code_context("task processing")
        print(f"Context preview: {context[:300]}...")
        
        # Search for code
        print("\nSearching for 'agent class'...")
        results = await assistant.search_code("agent class", limit=3)
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}: {result['token_type']} in {result['file_path']}")
            print(f"Preview: {result['content'][:100]}...")
        
        # Get function definition
        print("\nLooking up 'process_task' function...")
        func_def = await assistant.get_function_definition("process_task")
        if func_def:
            print(f"Function definition preview: {func_def[:200]}...")
        else:
            print("Function not found")
        
        # Get stats
        print("\nCode manager statistics:")
        stats = await assistant.get_stats()
        print(f"Total tokens: {stats['total_tokens']}")
        print(f"Total files: {stats['total_files']}")
    
    finally:
        await assistant.stop()

if __name__ == "__main__":
    asyncio.run(demo())
