import os
import ast
import re
import inspect
import importlib
import importlib.util
import sys
import tokenize
import io
import logging
import json
import hashlib
import threading
import time
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import asyncio
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CodeContextManager")

@dataclass
class CodeToken:
    """Represents a token from the codebase with metadata"""
    content: str
    file_path: str
    line_start: int
    line_end: int
    token_type: str
    importance: float = 1.0
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    dependencies: Set[str] = field(default_factory=set)
    
    def update_access(self):
        """Update access metadata when this token is used"""
        self.last_accessed = time.time()
        self.access_count += 1

@dataclass
class CodeContext:
    """A collection of code tokens forming a context window"""
    tokens: List[CodeToken] = field(default_factory=list)
    max_tokens: int = 8192
    current_token_count: int = 0
    
    def add_token(self, token: CodeToken) -> bool:
        """Add a token to the context if it fits, return success"""
        token_size = len(token.content.split())
        if self.current_token_count + token_size <= self.max_tokens:
            self.tokens.append(token)
            self.current_token_count += token_size
            return True
        return False
    
    def clear(self):
        """Clear all tokens from context"""
        self.tokens = []
        self.current_token_count = 0
    
    def to_string(self) -> str:
        """Convert context to a string representation"""
        return "\n\n".join([f"# File: {t.file_path}, Lines: {t.line_start}-{t.line_end}\n{t.content}" 
                           for t in self.tokens])

class CodeDependencyGraph:
    """Tracks dependencies between code elements"""
    def __init__(self):
        self.dependencies = defaultdict(set)
        self.reverse_dependencies = defaultdict(set)
        self.lock = threading.RLock()
    
    def add_dependency(self, source: str, target: str):
        """Add a dependency from source to target"""
        with self.lock:
            self.dependencies[source].add(target)
            self.reverse_dependencies[target].add(source)
    
    def get_dependencies(self, source: str) -> Set[str]:
        """Get all dependencies of a source"""
        with self.lock:
            return self.dependencies.get(source, set())
    
    def get_dependents(self, target: str) -> Set[str]:
        """Get all elements that depend on target"""
        with self.lock:
            return self.reverse_dependencies.get(target, set())
    
    def remove_element(self, element: str):
        """Remove an element and all its dependencies"""
        with self.lock:
            # Remove from dependencies
            if element in self.dependencies:
                del self.dependencies[element]
            
            # Remove from reverse dependencies
            for source, targets in self.dependencies.items():
                if element in targets:
                    targets.remove(element)
            
            # Clean up reverse dependencies
            if element in self.reverse_dependencies:
                del self.reverse_dependencies[element]

class ASTAnalyzer:
    """Analyzes Python code using AST to extract tokens and dependencies"""
    
    def __init__(self):
        self.imports = {}
        self.classes = {}
        self.functions = {}
        self.dependencies = CodeDependencyGraph()
    
    def parse_file(self, file_path: str) -> Dict[str, CodeToken]:
        """Parse a Python file and extract tokens with their dependencies"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            tokens = {}
            
            # Process imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        import_name = name.name
                        line_start = node.lineno
                        line_end = node.lineno
                        token_content = f"import {import_name}"
                        token_id = f"{file_path}:{import_name}"
                        tokens[token_id] = CodeToken(
                            content=token_content,
                            file_path=file_path,
                            line_start=line_start,
                            line_end=line_end,
                            token_type="import"
                        )
                        self.imports[import_name] = token_id
                
                elif isinstance(node, ast.ImportFrom):
                    module = node.module
                    for name in node.names:
                        import_name = f"{module}.{name.name}" if module else name.name
                        line_start = node.lineno
                        line_end = node.lineno
                        token_content = f"from {module or '.'} import {name.name}"
                        token_id = f"{file_path}:{import_name}"
                        tokens[token_id] = CodeToken(
                            content=token_content,
                            file_path=file_path,
                            line_start=line_start,
                            line_end=line_end,
                            token_type="import"
                        )
                        self.imports[import_name] = token_id
            
            # Process classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    line_start = node.lineno
                    line_end = max(node.lineno, max([n.lineno for n in ast.walk(node) if hasattr(n, 'lineno')], default=node.lineno))
                    
                    # Extract the class definition including docstring
                    class_lines = content.splitlines()[line_start-1:line_end]
                    class_content = "\n".join(class_lines)
                    
                    token_id = f"{file_path}:{class_name}"
                    tokens[token_id] = CodeToken(
                        content=class_content,
                        file_path=file_path,
                        line_start=line_start,
                        line_end=line_end,
                        token_type="class"
                    )
                    self.classes[class_name] = token_id
                    
                    # Analyze class dependencies
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            self.dependencies.add_dependency(token_id, base.id)
                    
                    # Process methods
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):
                            method_name = child.name
                            method_line_start = child.lineno
                            method_line_end = max(child.lineno, max([n.lineno for n in ast.walk(child) if hasattr(n, 'lineno')], default=child.lineno))
                            
                            # Extract the method definition including docstring
                            method_lines = content.splitlines()[method_line_start-1:method_line_end]
                            method_content = "\n".join(method_lines)
                            
                            method_token_id = f"{file_path}:{class_name}.{method_name}"
                            tokens[method_token_id] = CodeToken(
                                content=method_content,
                                file_path=file_path,
                                line_start=method_line_start,
                                line_end=method_line_end,
                                token_type="method",
                                dependencies={token_id}  # Method depends on its class
                            )
                            self.functions[f"{class_name}.{method_name}"] = method_token_id
                            self.dependencies.add_dependency(method_token_id, token_id)
                
                elif isinstance(node, ast.FunctionDef) and not isinstance(node.parent, ast.ClassDef):
                    func_name = node.name
                    line_start = node.lineno
                    line_end = max(node.lineno, max([n.lineno for n in ast.walk(node) if hasattr(n, 'lineno')], default=node.lineno))
                    
                    # Extract the function definition including docstring
                    func_lines = content.splitlines()[line_start-1:line_end]
                    func_content = "\n".join(func_lines)
                    
                    token_id = f"{file_path}:{func_name}"
                    tokens[token_id] = CodeToken(
                        content=func_content,
                        file_path=file_path,
                        line_start=line_start,
                        line_end=line_end,
                        token_type="function"
                    )
                    self.functions[func_name] = token_id
            
            return tokens
        
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {str(e)}")
            return {}

class CodeContextManager:
    """
    Advanced code context manager that loads and manages code tokens from the repository.
    Features:
    - Dynamic loading of code based on relevance and dependencies
    - Token importance scoring based on usage patterns
    - Context window management with token eviction policies
    - Dependency tracking for coherent context building
    - Asynchronous preloading of likely-to-be-needed code
    - Memory-efficient token storage with LRU caching
    """
    
    def __init__(self, repo_path: str, max_context_size: int = 8192, 
                 preload_patterns: List[str] = None, 
                 excluded_dirs: List[str] = None):
        """
        Initialize the code context manager.
        
        Args:
            repo_path: Root path of the repository
            max_context_size: Maximum number of tokens to keep in context
            preload_patterns: Glob patterns for files to preload
            excluded_dirs: Directories to exclude from scanning
        """
        self.repo_path = os.path.abspath(repo_path)
        self.max_context_size = max_context_size
        self.preload_patterns = preload_patterns or ["*.py"]
        self.excluded_dirs = excluded_dirs or [".git", "__pycache__", "venv", "env", ".venv", ".env"]
        
        # Token storage
        self.tokens: Dict[str, CodeToken] = {}
        self.file_tokens: Dict[str, List[str]] = defaultdict(list)
        self.token_lock = threading.RLock()
        
        # Context management
        self.current_context = CodeContext(max_tokens=max_context_size)
        self.context_history = deque(maxlen=10)  # Keep track of recent contexts
        
        # Dependency tracking
        self.ast_analyzer = ASTAnalyzer()
        self.dependency_graph = CodeDependencyGraph()
        
        # File monitoring
        self.file_mtimes: Dict[str, float] = {}
        self.file_hashes: Dict[str, str] = {}
        
        # Async task management
        self.preload_task = None
        self.file_watch_task = None
        self.running = False
        
        # Initialize token importance scoring
        self.token_importance: Dict[str, float] = {}
        
        logger.info(f"Initialized CodeContextManager for repository: {self.repo_path}")
    
    def start(self):
        """Start the code context manager and background tasks"""
        self.running = True
        self.scan_repository()
        
        # Start background tasks if asyncio is running
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self.preload_task = asyncio.create_task(self._preload_important_tokens())
                self.file_watch_task = asyncio.create_task(self._watch_for_file_changes())
        except RuntimeError:
            logger.warning("No running event loop detected. Background tasks not started.")
        
        logger.info("CodeContextManager started")
    
    def stop(self):
        """Stop the code context manager and background tasks"""
        self.running = False
        
        # Cancel background tasks
        if self.preload_task:
            self.preload_task.cancel()
        
        if self.file_watch_task:
            self.file_watch_task.cancel()
        
        logger.info("CodeContextManager stopped")
    
    def scan_repository(self):
        """Scan the repository and index all code files"""
        logger.info(f"Scanning repository: {self.repo_path}")
        
        for root, dirs, files in os.walk(self.repo_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            for file in files:
                if any(file.endswith(pattern.replace("*", "")) for pattern in self.preload_patterns):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.repo_path)
                    
                    # Skip files in excluded directories
                    if any(excluded in rel_path.split(os.sep) for excluded in self.excluded_dirs):
                        continue
                    
                    # Track file modification time
                    self.file_mtimes[file_path] = os.path.getmtime(file_path)
                    
                    # Parse Python files with AST
                    if file.endswith('.py'):
                        self._parse_python_file(file_path)
                    else:
                        # For non-Python files, just store the content as a single token
                        self._parse_generic_file(file_path)
        
        logger.info(f"Repository scan complete. Indexed {len(self.tokens)} tokens from {len(self.file_tokens)} files")
    
    def _parse_python_file(self, file_path: str):
        """Parse a Python file and extract tokens"""
        tokens = self.ast_analyzer.parse_file(file_path)
        
        with self.token_lock:
            for token_id, token in tokens.items():
                self.tokens[token_id] = token
                self.file_tokens[file_path].append(token_id)
                
                # Initialize importance score
                self.token_importance[token_id] = 1.0
    
    def _parse_generic_file(self, file_path: str):
        """Parse a non-Python file as a single token"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create a hash of the file content
            file_hash = hashlib.md5(content.encode()).hexdigest()
            self.file_hashes[file_path] = file_hash
            
            # Create a token for the entire file
            token_id = f"{file_path}:content"
            token = CodeToken(
                content=content,
                file_path=file_path,
                line_start=1,
                line_end=content.count('\n') + 1,
                token_type="file_content"
            )
            
            with self.token_lock:
                self.tokens[token_id] = token
                self.file_tokens[file_path].append(token_id)
                self.token_importance[token_id] = 1.0
        
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {str(e)}")
    
    def get_context_for_query(self, query: str, max_tokens: int = None) -> str:
        """
        Build a context based on a query, retrieving the most relevant code.
        
        Args:
            query: The query to build context for
            max_tokens: Maximum number of tokens to include (defaults to self.max_context_size)
            
        Returns:
            A string containing the relevant code context
        """
        if max_tokens is None:
            max_tokens = self.max_context_size
        
        # Create a new context
        context = CodeContext(max_tokens=max_tokens)
        
        # Find relevant tokens based on the query
        relevant_tokens = self._find_relevant_tokens(query)
        
        # Add tokens to the context
        for token_id, relevance in relevant_tokens:
            token = self.tokens.get(token_id)
            if token:
                # Update token access metadata
                token.update_access()
                
                # Try to add to context
                if not context.add_token(token):
                    # Context is full
                    break
        
        # Save this context to history
        self.context_history.append(context)
        
        # Return the context as a string
        return context.to_string()
    
    def _find_relevant_tokens(self, query: str) -> List[Tuple[str, float]]:
        """
        Find tokens relevant to a query, sorted by relevance.
        
        Args:
            query: The query to find relevant tokens for
            
        Returns:
            List of (token_id, relevance_score) tuples
        """
        # Simple relevance scoring based on token content matching query terms
        query_terms = set(query.lower().split())
        relevant_tokens = []
        
        with self.token_lock:
            for token_id, token in self.tokens.items():
                # Calculate relevance score
                token_content = token.content.lower()
                
                # Count matching terms
                matching_terms = sum(1 for term in query_terms if term in token_content)
                
                # Calculate relevance score based on:
                # 1. Number of matching terms
                # 2. Token importance (based on past usage)
                # 3. Recency of access
                recency_factor = 1.0 / (1.0 + (time.time() - token.last_accessed) / 3600)  # Decay over hours
                importance = self.token_importance.get(token_id, 1.0)
                
                relevance = (matching_terms * 10.0) * importance * recency_factor
                
                if relevance > 0:
                    relevant_tokens.append((token_id, relevance))
        
        # Sort by relevance (highest first)
        relevant_tokens.sort(key=lambda x: x[1], reverse=True)
        
        # Add dependencies for top tokens
        enhanced_tokens = []
        seen_tokens = set()
        
        for token_id, relevance in relevant_tokens[:20]:  # Consider top 20 for dependencies
            if token_id not in seen_tokens:
                enhanced_tokens.append((token_id, relevance))
                seen_tokens.add(token_id)
            
            # Add direct dependencies with slightly lower relevance
            for dep_id in self.dependency_graph.get_dependencies(token_id):
                if dep_id not in seen_tokens:
                    enhanced_tokens.append((dep_id, relevance * 0.9))
                    seen_tokens.add(dep_id)
        
        # Re-sort the enhanced list
        enhanced_tokens.sort(key=lambda x: x[1], reverse=True)
        
        return enhanced_tokens
    
    def get_file_content(self, file_path: str) -> str:
        """Get the content of a file from the repository"""
        abs_path = file_path if os.path.isabs(file_path) else os.path.join(self.repo_path, file_path)
        
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {abs_path}: {str(e)}")
            return f"Error: Could not read file {file_path}"
    
    def get_function_definition(self, function_name: str) -> Optional[str]:
        """Get the definition of a function by name"""
        # Check if we have this function indexed
        if function_name in self.ast_analyzer.functions:
            token_id = self.ast_analyzer.functions[function_name]
            token = self.tokens.get(token_id)
            if token:
                token.update_access()
                return token.content
        
        # If not found, try to find it by searching all tokens
        for token_id, token in self.tokens.items():
            if token.token_type in ("function", "method") and token_id.endswith(f":{function_name}"):
                token.update_access()
                return token.content
        
        return None
    
    def get_class_definition(self, class_name: str) -> Optional[str]:
        """Get the definition of a class by name"""
        # Check if we have this class indexed
        if class_name in self.ast_analyzer.classes:
            token_id = self.ast_analyzer.classes[class_name]
            token = self.tokens.get(token_id)
            if token:
                token.update_access()
                return token.content
        
        # If not found, try to find it by searching all tokens
        for token_id, token in self.tokens.items():
            if token.token_type == "class" and token_id.endswith(f":{class_name}"):
                token.update_access()
                return token.content
        
        return None
    
    def update_token_importance(self, token_id: str, importance_delta: float):
        """Update the importance score of a token based on feedback"""
        with self.token_lock:
            current_importance = self.token_importance.get(token_id, 1.0)
            new_importance = max(0.1, min(10.0, current_importance + importance_delta))
            self.token_importance[token_id] = new_importance
    
    async def _preload_important_tokens(self):
        """Background task to preload important tokens into memory"""
        while self.running:
            try:
                # Find the most important tokens that aren't loaded
                important_tokens = sorted(
                    [(tid, imp) for tid, imp in self.token_importance.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:100]  # Top 100 important tokens
                
                # Ensure they're loaded
                for token_id, _ in important_tokens:
                    if token_id not in self.tokens:
                        # This would happen if tokens were evicted from memory
                        file_path = token_id.split(':')[0]
                        if file_path.endswith('.py'):
                            self._parse_python_file(file_path)
                        else:
                            self._parse_generic_file(file_path)
                
                # Sleep before next check
                await asyncio.sleep(60)  # Check every minute
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in preload task: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _watch_for_file_changes(self):
        """Background task to watch for file changes"""
        while self.running:
            try:
                changed_files = []
                
                # Check all tracked files for changes
                for file_path, mtime in list(self.file_mtimes.items()):
                    try:
                        current_mtime = os.path.getmtime(file_path)
                        if current_mtime > mtime:
                            changed_files.append(file_path)
                            self.file_mtimes[file_path] = current_mtime
                    except FileNotFoundError:
                        # File was deleted
                        logger.info(f"File was deleted: {file_path}")
                        with self.token_lock:
                            # Remove all tokens for this file
                            for token_id in self.file_tokens.get(file_path, []):
                                if token_id in self.tokens:
                                    del self.tokens[token_id]
                                if token_id in self.token_importance:
                                    del self.token_importance[token_id]
                            
                            # Remove file from tracking
                            if file_path in self.file_tokens:
                                del self.file_tokens[file_path]
                            if file_path in self.file_mtimes:
                                del self.file_mtimes[file_path]
                            if file_path in self.file_hashes:
                                del self.file_hashes[file_path]
                
                # Reparse changed files
                for file_path in changed_files:
                    logger.info(f"File changed: {file_path}")
                    
                    # Remove old tokens for this file
                    with self.token_lock:
                        for token_id in self.file_tokens.get(file_path, []):
                            if token_id in self.tokens:
                                del self.tokens[token_id]
                        
                        # Clear file tokens list
                        self.file_tokens[file_path] = []
                    
                    # Reparse the file
                    if file_path.endswith('.py'):
                        self._parse_python_file(file_path)
                    else:
                        self._parse_generic_file(file_path)
                
                # Sleep before next check
                await asyncio.sleep(5)  # Check every 5 seconds
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in file watch task: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying
    
    def search_code(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for code matching the query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of matching code snippets with metadata
        """
        query_terms = query.lower().split()
        results = []
        
        with self.token_lock:
            for token_id, token in self.tokens.items():
                content = token.content.lower()
                
                # Calculate match score
                score = sum(10 for term in query_terms if term in content)
                
                # Boost score for exact matches
                if query.lower() in content:
                    score += 20
                
                if score > 0:
                    results.append({
                        "token_id": token_id,
                        "file_path": token.file_path,
                        "line_start": token.line_start,
                        "line_end": token.line_end,
                        "token_type": token.token_type,
                        "content": token.content,
                        "score": score
                    })
        
        # Sort by score and limit results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def get_dependencies(self, token_id: str) -> List[str]:
        """Get all dependencies for a token"""
        deps = self.dependency_graph.get_dependencies(token_id)
        return [self.tokens[dep_id].content for dep_id in deps if dep_id in self.tokens]
    
    def get_dependents(self, token_id: str) -> List[str]:
        """Get all tokens that depend on this token"""
        deps = self.dependency_graph.get_dependents(token_id)
        return [self.tokens[dep_id].content for dep_id in deps if dep_id in self.tokens]
    
    def get_token_stats(self) -> Dict[str, Any]:
        """Get statistics about the tokens in the system"""
        with self.token_lock:
            token_types = {}
            for token in self.tokens.values():
                token_types[token.token_type] = token_types.get(token.token_type, 0) + 1
            
            return {
                "total_tokens": len(self.tokens),
                "total_files": len(self.file_tokens),
                "token_types": token_types,
                "current_context_size": self.current_context.current_token_count,
                "max_context_size": self.max_context_size
            }

# Example usage
if __name__ == "__main__":
    # Get the repository path (current directory by default)
    repo_path = os.getcwd()
    
    # Create the code context manager
    manager = CodeContextManager(repo_path)
    
    # Start the manager
    manager.start()
    
    try:
        # Example: Get context for a query
        context = manager.get_context_for_query("Task management and processing")
        print("Context for 'Task management and processing':")
        print(context[:500] + "..." if len(context) > 500 else context)
        
        # Example: Search for code
        results = manager.search_code("process task")
        print("\nSearch results for 'process task':")
        for i, result in enumerate(results[:3], 1):
            print(f"{i}. {result['token_type']} in {result['file_path']} (Lines {result['line_start']}-{result['line_end']})")
            print(f"   Score: {result['score']}")
            print(f"   Content: {result['content'][:100]}..." if len(result['content']) > 100 else result['content'])
        
        # Example: Get token stats
        stats = manager.get_token_stats()
        print("\nToken statistics:")
        print(f"Total tokens: {stats['total_tokens']}")
        print(f"Total files: {stats['total_files']}")
        print("Token types:")
        for token_type, count in stats['token_types'].items():
            print(f"  {token_type}: {count}")
    
    finally:
        # Stop the manager
        manager.stop()
