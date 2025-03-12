"""
Semantic Code Search

This module implements semantic code search with natural language queries.
It allows developers to search for code using natural language descriptions
rather than exact keyword matches.

Features:
- Natural language query processing
- Code embedding and indexing
- Semantic similarity search
- Context-aware results ranking
- Integration with existing code repositories
"""

import os
import re
import ast
import json
import logging
import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI
    HAVE_OPENAI = True
except ImportError:
    HAVE_OPENAI = False
    logger.warning("OpenAI package not found. Using dummy embeddings.")

try:
    import faiss
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False
    logger.warning("FAISS package not found. Using numpy for vector search.")

@dataclass
class CodeSnippet:
    """Represents a code snippet with metadata"""
    id: str
    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    type: str  # function, class, method, block, etc.
    name: Optional[str] = None
    docstring: Optional[str] = None
    embedding: Optional[List[float]] = None
    imports: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    complexity: Optional[int] = None
    last_modified: Optional[str] = None
    tags: List[str] = field(default_factory=list)

class CodeParser:
    """
    Parses code files into searchable snippets.
    Supports multiple languages with language-specific parsing.
    """
    def __init__(self):
        self.language_handlers = {
            "python": self._parse_python,
            "javascript": self._parse_javascript,
            "typescript": self._parse_typescript,
            "java": self._parse_java,
            "go": self._parse_go,
            "rust": self._parse_rust,
            "c": self._parse_c,
            "cpp": self._parse_cpp,
            "csharp": self._parse_csharp,
        }
        
    async def parse_file(self, file_path: str) -> List[CodeSnippet]:
        """Parse a file into code snippets"""
        try:
            # Determine language from file extension
            language = self._get_language_from_extension(file_path)
            if not language:
                logger.warning(f"Unsupported file type: {file_path}")
                return []
                
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Use language-specific parser
            if language in self.language_handlers:
                snippets = await self.language_handlers[language](file_path, content)
                return snippets
            else:
                # Fallback to generic parsing
                return await self._parse_generic(file_path, content, language)
                
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return []
            
    def _get_language_from_extension(self, file_path: str) -> Optional[str]:
        """Determine language from file extension"""
        ext = os.path.splitext(file_path)[1].lower()
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".c": "c",
            ".h": "c",
            ".cpp": "cpp",
            ".hpp": "cpp",
            ".cc": "cpp",
            ".cs": "csharp",
        }
        return extension_map.get(ext)
        
    async def _parse_python(self, file_path: str, content: str) -> List[CodeSnippet]:
        """Parse Python code into snippets using AST"""
        snippets = []
        try:
            tree = ast.parse(content)
            lines = content.splitlines()
            file_name = os.path.basename(file_path)
            
            # Track imports
            imports = []
            
            # Process each node in the AST
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Extract import information
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            imports.append(name.name)
                    else:  # ImportFrom
                        module = node.module or ""
                        for name in node.names:
                            imports.append(f"{module}.{name.name}")
                
                elif isinstance(node, ast.FunctionDef):
                    # Extract function information
                    start_line = node.lineno
                    end_line = node.end_lineno
                    func_lines = lines[start_line-1:end_line]
                    func_content = "\n".join(func_lines)
                    
                    # Extract docstring if present
                    docstring = ast.get_docstring(node)
                    
                    # Calculate cyclomatic complexity (simplified)
                    complexity = 1  # Base complexity
                    for subnode in ast.walk(node):
                        if isinstance(subnode, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler)):
                            complexity += 1
                    
                    snippet = CodeSnippet(
                        id=f"{file_name}:{node.name}:{start_line}",
                        content=func_content,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        language="python",
                        type="function",
                        name=node.name,
                        docstring=docstring,
                        imports=imports.copy(),
                        complexity=complexity
                    )
                    snippets.append(snippet)
                    
                elif isinstance(node, ast.ClassDef):
                    # Extract class information
                    start_line = node.lineno
                    end_line = node.end_lineno
                    class_lines = lines[start_line-1:end_line]
                    class_content = "\n".join(class_lines)
                    
                    # Extract docstring if present
                    docstring = ast.get_docstring(node)
                    
                    snippet = CodeSnippet(
                        id=f"{file_name}:{node.name}:{start_line}",
                        content=class_content,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        language="python",
                        type="class",
                        name=node.name,
                        docstring=docstring,
                        imports=imports.copy()
                    )
                    snippets.append(snippet)
                    
                    # Also extract methods within the class
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_start = item.lineno
                            method_end = item.end_lineno
                            method_lines = lines[method_start-1:method_end]
                            method_content = "\n".join(method_lines)
                            
                            # Extract docstring if present
                            method_docstring = ast.get_docstring(item)
                            
                            # Calculate cyclomatic complexity (simplified)
                            method_complexity = 1  # Base complexity
                            for subnode in ast.walk(item):
                                if isinstance(subnode, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler)):
                                    method_complexity += 1
                            
                            method_snippet = CodeSnippet(
                                id=f"{file_name}:{node.name}.{item.name}:{method_start}",
                                content=method_content,
                                file_path=file_path,
                                start_line=method_start,
                                end_line=method_end,
                                language="python",
                                type="method",
                                name=f"{node.name}.{item.name}",
                                docstring=method_docstring,
                                imports=imports.copy(),
                                complexity=method_complexity
                            )
                            snippets.append(method_snippet)
            
            return snippets
            
        except SyntaxError as e:
            logger.error(f"Syntax error in Python file {file_path}: {e}")
            # Fall back to generic parsing for files with syntax errors
            return await self._parse_generic(file_path, content, "python")
            
    async def _parse_javascript(self, file_path: str, content: str) -> List[CodeSnippet]:
        """Parse JavaScript code into snippets"""
        # For now, use generic parsing
        # In a real implementation, we would use a JavaScript parser like esprima
        return await self._parse_generic(file_path, content, "javascript")
        
    async def _parse_typescript(self, file_path: str, content: str) -> List[CodeSnippet]:
        """Parse TypeScript code into snippets"""
        # For now, use generic parsing
        # In a real implementation, we would use a TypeScript parser
        return await self._parse_generic(file_path, content, "typescript")
        
    async def _parse_java(self, file_path: str, content: str) -> List[CodeSnippet]:
        """Parse Java code into snippets"""
        # For now, use generic parsing
        # In a real implementation, we would use a Java parser
        return await self._parse_generic(file_path, content, "java")
        
    async def _parse_go(self, file_path: str, content: str) -> List[CodeSnippet]:
        """Parse Go code into snippets"""
        # For now, use generic parsing
        # In a real implementation, we would use a Go parser
        return await self._parse_generic(file_path, content, "go")
        
    async def _parse_rust(self, file_path: str, content: str) -> List[CodeSnippet]:
        """Parse Rust code into snippets"""
        # For now, use generic parsing
        # In a real implementation, we would use a Rust parser
        return await self._parse_generic(file_path, content, "rust")
        
    async def _parse_c(self, file_path: str, content: str) -> List[CodeSnippet]:
        """Parse C code into snippets"""
        # For now, use generic parsing
        # In a real implementation, we would use a C parser
        return await self._parse_generic(file_path, content, "c")
        
    async def _parse_cpp(self, file_path: str, content: str) -> List[CodeSnippet]:
        """Parse C++ code into snippets"""
        # For now, use generic parsing
        # In a real implementation, we would use a C++ parser
        return await self._parse_generic(file_path, content, "cpp")
        
    async def _parse_csharp(self, file_path: str, content: str) -> List[CodeSnippet]:
        """Parse C# code into snippets"""
        # For now, use generic parsing
        # In a real implementation, we would use a C# parser
        return await self._parse_generic(file_path, content, "csharp")
        
    async def _parse_generic(self, file_path: str, content: str, language: str) -> List[CodeSnippet]:
        """Generic parsing for any language using regex patterns"""
        snippets = []
        file_name = os.path.basename(file_path)
        lines = content.splitlines()
        
        # Define regex patterns for common code structures
        patterns = {
            "function": r"(?:function|def|func)\s+(\w+)\s*\([^)]*\)\s*(?:\{|:)",
            "class": r"(?:class)\s+(\w+)(?:\s+extends|\s+implements|\s*:|\s*\{)",
            "method": r"(?:function|def|func)\s+(\w+)\s*\([^)]*\)\s*(?:\{|:)",
        }
        
        # Find potential code blocks
        for block_type, pattern in patterns.items():
            matches = re.finditer(pattern, content, re.MULTILINE)
            
            for match in matches:
                # Get the start position of the match
                start_pos = match.start()
                
                # Find the line number
                start_line = content[:start_pos].count('\n') + 1
                
                # Find the end of the block
                if '{' in match.group():
                    # For languages with braces, find matching closing brace
                    open_count = 1
                    end_pos = start_pos + match.group().find('{') + 1
                    
                    while open_count > 0 and end_pos < len(content):
                        if content[end_pos] == '{':
                            open_count += 1
                        elif content[end_pos] == '}':
                            open_count -= 1
                        end_pos += 1
                        
                    end_line = content[:end_pos].count('\n') + 1
                else:
                    # For languages with indentation, find the end of the indented block
                    # This is a simplified approach and might not work for all cases
                    current_line = start_line
                    indent_level = None
                    
                    while current_line < len(lines):
                        if current_line >= len(lines):
                            break
                            
                        line = lines[current_line]
                        
                        # Skip empty lines
                        if not line.strip():
                            current_line += 1
                            continue
                            
                        # Determine indent level if not set
                        if indent_level is None:
                            indent_level = len(line) - len(line.lstrip())
                            current_line += 1
                            continue
                            
                        # Check if we're back to the original indent level or less
                        current_indent = len(line) - len(line.lstrip())
                        if current_indent <= indent_level and line.strip():
                            break
                            
                        current_line += 1
                        
                    end_line = current_line
                
                # Extract the block content
                block_lines = lines[start_line-1:end_line]
                block_content = "\n".join(block_lines)
                
                # Extract name from the match
                name = match.group(1) if match.groups() else None
                
                # Create snippet
                snippet = CodeSnippet(
                    id=f"{file_name}:{name or 'unnamed'}:{start_line}",
                    content=block_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    language=language,
                    type=block_type,
                    name=name
                )
                snippets.append(snippet)
        
        # If no snippets were found, create one for the entire file
        if not snippets:
            snippet = CodeSnippet(
                id=f"{file_name}:entire_file:1",
                content=content,
                file_path=file_path,
                start_line=1,
                end_line=len(lines),
                language=language,
                type="file",
                name=file_name
            )
            snippets.append(snippet)
            
        return snippets

class EmbeddingProvider:
    """
    Provides embeddings for code snippets and queries.
    Supports multiple embedding models and fallbacks.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-large"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.client = None
        self.embedding_dim = 3072  # Default for text-embedding-3-large
        
        if HAVE_OPENAI and self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key)
            logger.info(f"Using OpenAI for embeddings with model {model}")
        else:
            logger.warning("Using dummy embeddings (random vectors)")
            
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        if self.client:
            try:
                response = await self.client.embeddings.create(
                    input=text,
                    model=self.model
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error(f"Error getting embedding from OpenAI: {e}")
                return self._get_dummy_embedding()
        else:
            return self._get_dummy_embedding()
            
    async def get_embeddings_batch(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """Get embeddings for a batch of texts"""
        results = []
        
        if self.client:
            # Process in batches to avoid rate limits
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                try:
                    response = await self.client.embeddings.create(
                        input=batch,
                        model=self.model
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    results.extend(batch_embeddings)
                except Exception as e:
                    logger.error(f"Error getting batch embeddings from OpenAI: {e}")
                    # Fall back to dummy embeddings for this batch
                    dummy_embeddings = [self._get_dummy_embedding() for _ in batch]
                    results.extend(dummy_embeddings)
                
                # Sleep to avoid rate limits
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.5)
        else:
            # Use dummy embeddings
            results = [self._get_dummy_embedding() for _ in texts]
            
        return results
        
    def _get_dummy_embedding(self) -> List[float]:
        """Generate a dummy embedding (random vector)"""
        # Use a fixed seed for deterministic results
        np.random.seed(42)
        return list(np.random.uniform(-1, 1, self.embedding_dim))
        
    def set_model(self, model: str) -> None:
        """Change the embedding model"""
        self.model = model
        # Update embedding dimension based on model
        if model == "text-embedding-3-small":
            self.embedding_dim = 1536
        elif model == "text-embedding-3-large":
            self.embedding_dim = 3072
        elif model == "text-embedding-ada-002":
            self.embedding_dim = 1536
        logger.info(f"Changed embedding model to {model} with dimension {self.embedding_dim}")

class CodeSearchIndex:
    """
    Manages the index of code snippets for efficient semantic search.
    Supports both in-memory and disk-based indices.
    """
    def __init__(self, db_path: str = "code_search.db", use_faiss: bool = True):
        self.db_path = db_path
        self.use_faiss = use_faiss and HAVE_FAISS
        self.embedding_provider = EmbeddingProvider()
        self.index = None
        self.snippet_ids = []
        self.conn = None
        
        # Initialize database
        self._init_db()
        
        # Initialize index
        self._init_index()
        
    def _init_db(self) -> None:
        """Initialize the SQLite database"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
        # Create tables if they don't exist
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS snippets (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                file_path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                language TEXT NOT NULL,
                type TEXT NOT NULL,
                name TEXT,
                docstring TEXT,
                imports TEXT,
                references TEXT,
                complexity INTEGER,
                last_modified TEXT,
                tags TEXT,
                embedding BLOB
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                file_path TEXT PRIMARY KEY,
                language TEXT NOT NULL,
                last_indexed TEXT NOT NULL,
                last_modified TEXT,
                size INTEGER
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                results TEXT,
                user_feedback INTEGER
            )
        """)
        
        self.conn.commit()
        
    def _init_index(self) -> None:
        """Initialize the vector index"""
        if self.use_faiss:
            # Load existing snippets to build the index
            cursor = self.conn.cursor()
            cursor.execute("SELECT id, embedding FROM snippets WHERE embedding IS NOT NULL")
            rows = cursor.fetchall()
            
            if rows:
                # Extract embeddings and IDs
                embeddings = []
                self.snippet_ids = []
                
                for row in rows:
                    embedding_blob = row['embedding']
                    if embedding_blob:
                        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                        embeddings.append(embedding)
                        self.snippet_ids.append(row['id'])
                
                if embeddings:
                    # Create FAISS index
                    dimension = len(embeddings[0])
                    self.index = faiss.IndexFlatL2(dimension)
                    self.index.add(np.array(embeddings, dtype=np.float32))
                    logger.info(f"Loaded {len(embeddings)} embeddings into FAISS index")
        else:
            # For non-FAISS, we'll load embeddings on demand during search
            logger.info("Using numpy for vector search")
            
    async def add_snippet(self, snippet: CodeSnippet) -> None:
        """Add a code snippet to the index"""
        # Generate embedding if not present
        if snippet.embedding is None:
            # Prepare text for embedding
            text_for_embedding = self._prepare_text_for_embedding(snippet)
            snippet.embedding = await self.embedding_provider.get_embedding(text_for_embedding)
            
        # Convert embedding to bytes for storage
        embedding_bytes = None
        if snippet.embedding:
            embedding_bytes = np.array(snippet.embedding, dtype=np.float32).tobytes()
            
        # Store in database
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO snippets
            (id, content, file_path, start_line, end_line, language, type, name, 
             docstring, imports, references, complexity, last_modified, tags, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snippet.id,
            snippet.content,
            snippet.file_path,
            snippet.start_line,
            snippet.end_line,
            snippet.language,
            snippet.type,
            snippet.name,
            snippet.docstring,
            json.dumps(snippet.imports),
            json.dumps(snippet.references),
            snippet.complexity,
            snippet.last_modified,
            json.dumps(snippet.tags),
            embedding_bytes
        ))
        
        # Update file record
        cursor.execute("""
            INSERT OR REPLACE INTO files
            (file_path, language, last_indexed, last_modified, size)
            VALUES (?, ?, ?, ?, ?)
        """, (
            snippet.file_path,
            snippet.language,
            datetime.now().isoformat(),
            snippet.last_modified,
            len(snippet.content)
        ))
        
        self.conn.commit()
        
        # Update in-memory index if using FAISS
        if self.use_faiss and self.index is not None and snippet.embedding:
            embedding_np = np.array([snippet.embedding], dtype=np.float32)
            self.index.add(embedding_np)
            self.snippet_ids.append(snippet.id)
            
    async def add_snippets_batch(self, snippets: List[CodeSnippet]) -> None:
        """Add multiple code snippets to the index in batch"""
        # Generate embeddings for snippets without them
        snippets_without_embedding = [s for s in snippets if s.embedding is None]
        if snippets_without_embedding:
            texts = [self._prepare_text_for_embedding(s) for s in snippets_without_embedding]
            embeddings = await self.embedding_provider.get_embeddings_batch(texts)
            
            # Assign embeddings
            for snippet, embedding in zip(snippets_without_embedding, embeddings):
                snippet.embedding = embedding
                
        # Store in database
        cursor = self.conn.cursor()
        
        # Begin transaction
        self.conn.execute("BEGIN TRANSACTION")
        
        try:
            # Insert snippets
            for snippet in snippets:
                # Convert embedding to bytes for storage
                embedding_bytes = None
                if snippet.embedding:
                    embedding_bytes = np.array(snippet.embedding, dtype=np.float32).tobytes()
                    
                cursor.execute("""
                    INSERT OR REPLACE INTO snippets
                    (id, content, file_path, start_line, end_line, language, type, name, 
                     docstring, imports, references, complexity, last_modified, tags, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    snippet.id,
                    snippet.content,
                    snippet.file_path,
                    snippet.start_line,
                    snippet.end_line,
                    snippet.language,
                    snippet.type,
                    snippet.name,
                    snippet.docstring,
                    json.dumps(snippet.imports),
                    json.dumps(snippet.references),
                    snippet.complexity,
                    snippet.last_modified,
                    json.dumps(snippet.tags),
                    embedding_bytes
                ))
                
                # Update file record
                cursor.execute("""
                    INSERT OR REPLACE INTO files
                    (file_path, language, last_indexed, last_modified, size)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    snippet.file_path,
                    snippet.language,
                    datetime.now().isoformat(),
                    snippet.last_modified,
                    len(snippet.content)
                ))
                
            # Commit transaction
            self.conn.commit()
            
            # Update in-memory index if using FAISS
            if self.use_faiss and self.index is not None:
                embeddings = [s.embedding for s in snippets if s.embedding]
                if embeddings:
                    embeddings_np = np.array(embeddings, dtype=np.float32)
                    self.index.add(embeddings_np)
                    self.snippet_ids.extend([s.id for s in snippets if s.embedding])
                    
        except Exception as e:
            # Rollback on error
            self.conn.rollback()
            logger.error(f"Error adding snippets batch: {e}")
            raise
            
    def _prepare_text_for_embedding(self, snippet: CodeSnippet) -> str:
        """Prepare text for embedding by combining relevant fields"""
        parts = []
        
        # Add name if available
        if snippet.name:
            parts.append(f"Name: {snippet.name}")
            
        # Add type
        parts.append(f"Type: {snippet.type}")
        
        # Add docstring if available
        if snippet.docstring:
            parts.append(f"Documentation: {snippet.docstring}")
            
        # Add content
        parts.append(f"Code: {snippet.content}")
        
        return "\n".join(parts)
        
    async def search(self, query: str, top_k: int = 10, 
                   filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for code snippets matching the query
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            filters: Optional filters (language, type, etc.)
            
        Returns:
            List of matching snippets with similarity scores
        """
        # Generate query embedding
        query_embedding = await self.embedding_provider.get_embedding(query)
        
        # Search using the appropriate method
        if self.use_faiss and self.index is not None:
            results = self._search_faiss(query_embedding, top_k, filters)
        else:
            results = await self._search_numpy(query_embedding, top_k, filters)
            
        # Log search for analytics
        self._log_search(query, results)
        
        return results
        
    def _search_faiss(self, query_embedding: List[float], top_k: int, 
                     filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search using FAISS index"""
        # Convert query embedding to numpy array
        query_np = np.array([query_embedding], dtype=np.float32)
        
        # Adjust top_k if we have filters
        search_k = top_k
        if filters:
            # Request more results since we'll filter some out
            search_k = min(top_k * 5, len(self.snippet_ids))
            
        # Search the index
        distances, indices = self.index.search(query_np, search_k)
        
        # Get the snippet IDs for the results
        result_ids = [self.snippet_ids[idx] for idx in indices[0]]
        
        # Fetch the snippets from the database
        cursor = self.conn.cursor()
        results = []
        
        for i, snippet_id in enumerate(result_ids):
            cursor.execute("""
                SELECT * FROM snippets WHERE id = ?
            """, (snippet_id,))
            row = cursor.fetchone()
            
            if row:
                # Convert row to dict
                snippet_dict = dict(row)
                
                # Parse JSON fields
                for field in ['imports', 'references', 'tags']:
                    if snippet_dict[field]:
                        snippet_dict[field] = json.loads(snippet_dict[field])
                    else:
                        snippet_dict[field] = []
                        
                # Remove embedding blob
                if 'embedding' in snippet_dict:
                    del snippet_dict['embedding']
                    
                # Add distance (similarity score)
                similarity = 1.0 / (1.0 + distances[0][i])
                snippet_dict['similarity'] = similarity
                
                # Apply filters if provided
                if filters:
                    if not self._apply_filters(snippet_dict, filters):
                        continue
                        
                results.append(snippet_dict)
                
                # Stop if we have enough results
                if len(results) >= top_k:
                    break
                    
        return results
        
    async def _search_numpy(self, query_embedding: List[float], top_k: int, 
                          filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search using numpy for vector similarity"""
        # Fetch all snippets from the database
        cursor = self.conn.cursor()
        
        # Build query based on filters
        query_parts = ["SELECT * FROM snippets WHERE embedding IS NOT NULL"]
        query_params = []
        
        if filters:
            if 'language' in filters:
                query_parts.append("language = ?")
                query_params.append(filters['language'])
                
            if 'type' in filters:
                query_parts.append("type = ?")
                query_params.append(filters['type'])
                
        # Execute query
        cursor.execute(" AND ".join(query_parts), query_params)
        rows = cursor.fetchall()
        
        # Calculate similarities
        results = []
        query_np = np.array(query_embedding, dtype=np.float32)
        
        for row in rows:
            # Convert row to dict
            snippet_dict = dict(row)
            
            # Parse JSON fields
            for field in ['imports', 'references', 'tags']:
                if snippet_dict[field]:
                    snippet_dict[field] = json.loads(snippet_dict[field])
                else:
                    snippet_dict[field] = []
                    
            # Calculate similarity
            if snippet_dict['embedding']:
                embedding = np.frombuffer(snippet_dict['embedding'], dtype=np.float32)
                # Cosine similarity
                similarity = np.dot(query_np, embedding) / (np.linalg.norm(query_np) * np.linalg.norm(embedding))
                snippet_dict['similarity'] = float(similarity)
                
                # Remove embedding blob
                del snippet_dict['embedding']
                
                # Apply additional filters
                if filters and not self._apply_filters(snippet_dict, filters):
                    continue
                    
                results.append(snippet_dict)
                
        # Sort by similarity (descending)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top k results
        return results[:top_k]
        
    def _apply_filters(self, snippet: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Apply filters to a snippet"""
        # Language filter
        if 'language' in filters and filters['language'] and snippet['language'] != filters['language']:
            return False
            
        # Type filter
        if 'type' in filters and filters['type'] and snippet['type'] != filters['type']:
            return False
            
        # File path filter
        if 'file_path' in filters and filters['file_path']:
            if not snippet['file_path'].startswith(filters['file_path']):
                return False
                
        # Tag filter
        if 'tag' in filters and filters['tag']:
            if filters['tag'] not in snippet['tags']:
                return False
                
        # Complexity filter
        if 'max_complexity' in filters and filters['max_complexity'] is not None:
            if snippet['complexity'] and snippet['complexity'] > filters['max_complexity']:
                return False
                
        return True
        
    def _log_search(self, query: str, results: List[Dict[str, Any]]) -> None:
        """Log search query and results for analytics"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO search_history
                (query, timestamp, results)
                VALUES (?, ?, ?)
            """, (
                query,
                datetime.now().isoformat(),
                json.dumps([r['id'] for r in results])
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error logging search: {e}")
            
    def record_feedback(self, search_id: int, feedback: int) -> None:
        """Record user feedback on search results"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE search_history
                SET user_feedback = ?
                WHERE id = ?
            """, (feedback, search_id))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            
    def get_search_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent search history"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM search_history
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
        
    def close(self) -> None:
        """Close the database connection"""
        if self.conn:
            self.conn.close()

class CodeSearchEngine:
    """
    Main engine for semantic code search.
    Coordinates parsing, indexing, and searching.
    """
    def __init__(self, db_path: str = "code_search.db", use_faiss: bool = True):
        self.parser = CodeParser()
        self.index = CodeSearchIndex(db_path, use_faiss)
        
    async def index_file(self, file_path: str) -> int:
        """
        Index a single file
        
        Args:
            file_path: Path to the file to index
            
        Returns:
            Number of snippets indexed
        """
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return 0
            
        # Parse file into snippets
        snippets = await self.parser.parse_file(file_path)
        
        if not snippets:
            logger.warning(f"No snippets found in {file_path}")
            return 0
            
        # Add snippets to index
        await self.index.add_snippets_batch(snippets)
        
        logger.info(f"Indexed {len(snippets)} snippets from {file_path}")
        return len(snippets)
        
    async def index_directory(self, directory: str, recursive: bool = True, 
                            file_extensions: List[str] = None) -> int:
        """
        Index all files in a directory
        
        Args:
            directory: Directory to index
            recursive: Whether to recursively index subdirectories
            file_extensions: List of file extensions to index (e.g., ['.py', '.js'])
            
        Returns:
            Number of snippets indexed
        """
        # Check if directory exists
        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            return 0
            
        # Get list of files to index
        files_to_index = []
        
        if recursive:
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file_extensions:
                        ext = os.path.splitext(file)[1].lower()
                        if ext in file_extensions:
                            files_to_index.append(file_path)
                    else:
                        files_to_index.append(file_path)
        else:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path):
                    if file_extensions:
                        ext = os.path.splitext(file)[1].lower()
                        if ext in file_extensions:
                            files_to_index.append(file_path)
                    else:
                        files_to_index.append(file_path)
                        
        # Index files
        total_snippets = 0
        for file_path in files_to_index:
            try:
                snippets_count = await self.index_file(file_path)
                total_snippets += snippets_count
            except Exception as e:
                logger.error(f"Error indexing {file_path}: {e}")
                
        logger.info(f"Indexed {total_snippets} snippets from {len(files_to_index)} files in {directory}")
        return total_snippets
        
    async def search(self, query: str, top_k: int = 10, 
                   filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for code snippets matching the query
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            filters: Optional filters (language, type, etc.)
            
        Returns:
            List of matching snippets with similarity scores
        """
        return await self.index.search(query, top_k, filters)
        
    def close(self) -> None:
        """Close the search engine and release resources"""
        self.index.close()

# Example usage
async def example_usage():
    # Create search engine
    search_engine = CodeSearchEngine()
    
    # Index a directory
    await search_engine.index_directory("./", recursive=True, file_extensions=['.py'])
    
    # Search for code
    results = await search_engine.search("function to parse Python code", top_k=5)
    
    # Print results
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['name']} ({result['type']}) - Similarity: {result['similarity']:.4f}")
        print(f"   File: {result['file_path']} (Lines {result['start_line']}-{result['end_line']})")
        print(f"   Language: {result['language']}")
        if result['docstring']:
            print(f"   Docstring: {result['docstring'][:100]}...")
        print(f"   Content snippet: {result['content'][:100]}...")
        
    # Close the search engine
    search_engine.close()

if __name__ == "__main__":
    asyncio.run(example_usage())
