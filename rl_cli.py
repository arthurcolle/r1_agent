#!/usr/bin/env python3
"""
Full Advanced Agent Module with Error Correction and Self-Assembling

Features:
- Core agent functionality: MCTS, RL, self-modification.
- AST-based code chunking for granular self-reflection.
- After each self-modification, the agent compiles its source; on error, it reverts to backup.
- Dynamic plugin loading from a "plugins" directory, with error fallback.
- Reflection: the agent reflects on each code chunk (via LLM), embeds and stores reflections with timestamps.
- FastAPI endpoints with Pydantic models for dynamic tool registration and progress tracking.
- CLI interface with PDF watchers and interactive chat.
- Redis PubSub integration is available (optional).
"""

import os
import sys
import json
import uuid
import logging
import sqlite3
import subprocess
import platform
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import importlib
import importlib.util
import time
import random
import hashlib
import asyncio
import re
import shutil
import ast
import types
from math import log
from typing import Any, Dict, List, Optional, Tuple, Callable, AsyncGenerator

# Configure logging
logger = logging.getLogger(__name__)

import requests
import urllib.parse
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import PyPDF2
from openai import AsyncOpenAI
import numpy as np
from scipy import spatial
import redis
import gymnasium as gym
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from pydantic import BaseModel, Field, ConfigDict
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import uvicorn
import cmd
import shlex
from datetime import datetime
import traceback
import curses
from curses import panel
from enum import Enum, auto

class PDFHandler(FileSystemEventHandler):
    """Handler for PDF file events"""
    def __init__(self, agent):
        self.agent = agent
        self.processed_paths = set()
        
    def compute_file_hash(self, file_path):
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
        
    def should_process_file(self, file_path):
        if file_path in self.processed_paths:
            return False
        cur = self.agent.knowledge_base.conn.cursor()
        cur.execute("SELECT file_hash, last_modified FROM ingested_files WHERE file_path = ?", (file_path,))
        row = cur.fetchone()
        if not row:
            return True
        current_hash = self.compute_file_hash(file_path)
        current_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
        return (current_hash != row[0] or current_mtime.timestamp() > datetime.fromisoformat(row[1]).timestamp())
        
    def record_processed_file(self, file_path):
        c_hash = self.compute_file_hash(file_path)
        c_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
        cur = self.agent.knowledge_base.conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO ingested_files (file_path, file_hash, last_modified, ingested_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """, (file_path, c_hash, c_mtime.isoformat()))
        self.agent.knowledge_base.conn.commit()
        self.processed_paths.add(file_path)
        
    def on_created(self, event):
        if event.is_directory:
            logging.info(f"New directory detected: {event.src_path}")
            for root, _, files in os.walk(event.src_path):
                for fl in files:
                    if fl.lower().endswith(".pdf"):
                        fp = os.path.join(root, fl)
                        self.process_pdf(fp)
            return
        if event.src_path.lower().endswith(".pdf"):
            self.process_pdf(event.src_path)
            
    def process_pdf(self, file_path):
        if self.should_process_file(file_path):
            logging.info(f"Processing PDF: {file_path}")
            asyncio.run(self.agent.ingest_source(file_path))
            self.record_processed_file(file_path)
        else:
            logging.info(f"Skipping already processed PDF: {file_path}")

# ------------------------------------------------------------------------------
# Global Setup & Logging
# ------------------------------------------------------------------------------

def detect_system_capabilities() -> Dict[str, Any]:
    """Detect system capabilities and available tools"""
    capabilities = {
        "os": platform.system(),
        "python_version": platform.python_version(),
        "env_vars": {},
        "commands": {},
        "apis": {}
    }
    
    # Detect environment variables for APIs
    api_patterns = ['API_KEY', 'ACCESS_TOKEN', 'SECRET_KEY', 'TOKEN']
    for key, value in os.environ.items():
        if any(pattern in key.upper() for pattern in api_patterns):
            service = key.split('_')[0].lower()
            capabilities["env_vars"][service] = key

    # Detect common command line tools
    common_commands = ['git', 'curl', 'wget', 'ffmpeg', 'pandoc']
    for cmd in common_commands:
        try:
            subprocess.run([cmd, '--version'], capture_output=True)
            capabilities["commands"][cmd] = True
        except FileNotFoundError:
            capabilities["commands"][cmd] = False

    return capabilities

class JinaClient:
    """Client for interacting with Jina.ai endpoints"""
    
    def __init__(self, token: Optional[str] = None):
        """Initialize with your Jina token"""
        self.token = token or os.getenv("JINA_API_KEY", "JINA_API_KEY")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    async def search(self, query: str) -> dict:
        """Search using s.jina.ai endpoint"""
        encoded_query = urllib.parse.quote(query)
        url = f"https://s.jina.ai/{encoded_query}"
        response = requests.get(url, headers=self.headers)
        return response.json()
    
    async def fact_check(self, query: str) -> dict:
        """Get grounding info using g.jina.ai endpoint"""
        encoded_query = urllib.parse.quote(query)
        url = f"https://g.jina.ai/{encoded_query}"
        response = requests.get(url, headers=self.headers)
        return response.json()
        
    async def reader(self, url: str) -> dict:
        """Get ranking using r.jina.ai endpoint"""
        encoded_url = urllib.parse.quote(url)
        url = f"https://r.jina.ai/{encoded_url}"
        response = requests.get(url, headers=self.headers)
        return response.json()

def setup_system_tools(capabilities: Dict[str, Any]) -> Dict[str, Callable]:
    """Create tool functions based on detected capabilities"""
    tools = {}

    # File operations
    tools["read_file"] = lambda path: Path(path).read_text() if Path(path).exists() else None
    tools["write_file"] = lambda path, content: Path(path).write_text(content)
    tools["list_files"] = lambda path=".", pattern="*": list(Path(path).glob(pattern))
    
    # Command execution
    tools["run_command"] = lambda cmd: subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Web operations if requests is available
    if "requests" in sys.modules:
        tools["web_get"] = lambda url: requests.get(url).text
        
        # Jina.ai integration
        if "JINA_API_KEY" in os.environ:
            jina_client = JinaClient()
            tools["jina_search"] = jina_client.search
            tools["jina_fact_check"] = jina_client.fact_check
            tools["jina_reader"] = jina_client.reader
            
            # Register tool schemas
            register_tool(
                "jina_search",
                jina_client.search,
                "Search the web using Jina.ai",
                {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                }
            )
            
            register_tool(
                "jina_fact_check",
                jina_client.fact_check,
                "Fact check a statement using Jina.ai",
                {
                    "query": {
                        "type": "string",
                        "description": "Statement to fact check"
                    }
                }
            )
            
            register_tool(
                "jina_reader",
                jina_client.reader,
                "Extract content from a URL using Jina.ai",
                {
                    "url": {
                        "type": "string",
                        "description": "URL to analyze"
                    }
                }
            )
        
        # Weather API integration
        if "OPENWEATHERMAP_API_KEY" in os.environ:
            async def get_weather(location: str) -> Dict[str, Any]:
                """Get weather data for a location using Jina search as fallback if OpenWeather fails"""
                try:
                    api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
                    if api_key:
                        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=imperial"
                        response = requests.get(url)
                        if response.status_code == 200:
                            return response.json()
        
                    # Fallback to Jina search
                    jina_client = JinaClient()
                    search_result = await jina_client.search(f"current weather in {location}")
                    return {
                        "main": {"temp": "N/A"},
                        "weather": [{"description": search_result.get("weather_description", "Weather data unavailable")}],
                        "jina_results": search_result
                    }
                except Exception as e:
                    return {"error": f"Weather API error: {str(e)}"}
            tools["get_weather"] = get_weather
        
        # Search implementations
        if "serpapi" in capabilities["env_vars"]:
            from serpapi import GoogleSearch
            async def web_search(query: str) -> Dict[str, Any]:
                try:
                    search = GoogleSearch({
                        "q": query,
                        "api_key": os.getenv(capabilities["env_vars"]["serpapi"])
                    })
                    return search.get_dict()
                except Exception as e:
                    return {"error": f"Search error: {str(e)}"}
            tools["web_search"] = web_search
        elif "GOOGLE_API_KEY" in os.environ and "GOOGLE_CSE_ID" in os.environ:
            async def web_search(query: str) -> Dict[str, Any]:
                try:
                    url = "https://www.googleapis.com/customsearch/v1"
                    params = {
                        "key": os.environ["GOOGLE_API_KEY"],
                        "cx": os.environ["GOOGLE_CSE_ID"],
                        "q": query
                    }
                    response = requests.get(url, params=params)
                    if response.status_code == 200:
                        return response.json()
                    return {"error": f"Google search error: {response.status_code}"}
                except Exception as e:
                    return {"error": f"Search error: {str(e)}"}
            tools["web_search"] = web_search

    return tools

logging.basicConfig(level=logging.INFO)

class Colors:
    """ANSI color codes for logging and debugging."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# ------------------------------------------------------------------------------
# Chain-of-Thought Data Structures
# ------------------------------------------------------------------------------

class Metadata(BaseModel):
    key: str
    value: str

class ThoughtStep(BaseModel):
    step_number: int
    reasoning: str
    conclusion: str 
    confidence: float

class ChainOfThought(BaseModel):
    initial_thought: str
    steps: List[ThoughtStep]
    final_conclusion: str
    metadata: List[Metadata] = Field(default_factory=list)

# ------------------------------------------------------------------------------
# OpenAI Client Initialization (Async)
# ------------------------------------------------------------------------------

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<YOUR_OPENAI_API_KEY>"))

# ------------------------------------------------------------------------------
# Global Tool Registry
# ------------------------------------------------------------------------------

tools: List[Dict[str, Any]] = []
available_functions: Dict[str, Callable] = {}

def register_tool(
    name: str, 
    func: Callable, 
    description: str, 
    parameters: Dict[str, Any],
    required_params: Optional[List[str]] = None,
    examples: Optional[List[Dict[str, Any]]] = None,
    rate_limit: Optional[float] = None
):
    """
    Register a tool function for dynamic function calling with enhanced metadata.
    
    Args:
        name: Tool name
        func: The callable to execute
        description: Tool description
        parameters: Parameter schema
        required_params: List of required parameter names
        examples: List of example calls with inputs and outputs
        rate_limit: Minimum seconds between calls
    """
    global tools
    tools[:] = [t for t in tools if t["function"]["name"] != name]
    available_functions[name] = func
    
    tool_spec = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required_params or list(parameters.keys())
            }
        },
        "metadata": {
            "registered_at": datetime.now().isoformat(),
            "examples": examples or [],
            "rate_limit": rate_limit,
            "last_called": None
        }
    }
    
    tools.append(tool_spec)
    print(f"{Colors.OKGREEN}{Colors.BOLD}Registered tool:{Colors.ENDC} {name}")
    return tool_spec

def call_tool(function_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Call a registered tool with given arguments and return structured response."""
    func = available_functions.get(function_name)
    if not func:
        return {
            "status": "error",
            "error": f"Tool {function_name} not found",
            "result": None
        }
    try:
        result = func(**args)
        return {
            "status": "success",
            "error": None,
            "result": result,
            "function_name": function_name,
            "args": args,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "result": None,
            "function_name": function_name,
            "args": args,
            "timestamp": datetime.now().isoformat()
        }

# ------------------------------------------------------------------------------
# Embedding Computation Functions (with Caching)
# ------------------------------------------------------------------------------

async def compute_embeddings_batch(
    texts: List[str],
    model: str = "text-embedding-3-large",
    batch_size: int = 16,
    db_conn: Optional[sqlite3.Connection] = None
) -> List[List[float]]:
    results = []
    uncached_texts = []
    uncached_indices = []
    content_hashes = []

    if db_conn:
        cur = db_conn.cursor()
        for i, text in enumerate(texts):
            c_hash = hashlib.sha256(text.encode()).hexdigest()
            content_hashes.append(c_hash)
            cur.execute("SELECT embedding FROM embeddings_cache WHERE content_hash = ?", (c_hash,))
            row = cur.fetchone()
            if row:
                emb_list = np.frombuffer(row[0], dtype=np.float32).tolist()
                results.append(emb_list)
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
    else:
        uncached_texts = texts
        uncached_indices = list(range(len(texts)))
        content_hashes = [hashlib.sha256(t.encode()).hexdigest() for t in texts]

    for i in range(0, len(uncached_texts), batch_size):
        batch = uncached_texts[i:i+batch_size]
        try:
            response = await client.embeddings.create(input=batch, model=model)
            embeddings = [d.embedding for d in response.data]
            if db_conn:
                cur = db_conn.cursor()
                for j, emb in enumerate(embeddings):
                    actual_idx = i + j
                    if actual_idx < len(uncached_indices):
                        orig_idx = uncached_indices[actual_idx]
                        blob = np.array(emb, dtype=np.float32).tobytes()
                        cur.execute(
                            "INSERT OR REPLACE INTO embeddings_cache (content_hash, embedding) VALUES (?, ?)",
                            (content_hashes[orig_idx], blob)
                        )
                db_conn.commit()
            results.extend(embeddings)
        except Exception as ex:
            logging.error(f"Error in compute_embeddings_batch: {ex}")
            results.extend([None]*len(batch))

    if db_conn and len(uncached_indices) != len(texts):
        final_results = [None]*len(texts)
        uncached_idx = 0
        for i in range(len(texts)):
            if i in uncached_indices:
                final_results[i] = results[uncached_idx]
                uncached_idx += 1
            else:
                final_results[i] = results[i]
        return final_results

    return results

async def compute_embedding(
    text: str,
    model: str = "text-embedding-3-large",
    db_conn: Optional[sqlite3.Connection] = None
) -> List[float]:
    result_list = await compute_embeddings_batch([text], model=model, db_conn=db_conn)
    return result_list[0] if result_list else None

# ------------------------------------------------------------------------------
# SQLiteKnowledgeBase Implementation
# ------------------------------------------------------------------------------

class SQLiteKnowledgeBase:
    """
    A SQLite-based knowledge base for collections, documents, and embeddings.
    """
    def __init__(self, db_path: str = "./agent.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("SELECT MAX(version) FROM schema_version")
        current_version = cur.fetchone()[0] or 0
        if current_version < 1:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS collections (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    collection_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_updated DATETIME,
                    version INTEGER DEFAULT 1,
                    status TEXT DEFAULT 'active',
                    tags TEXT,
                    FOREIGN KEY(collection_id) REFERENCES collections(id) ON DELETE CASCADE
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddings_cache (
                    content_hash TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("INSERT INTO schema_version (version) VALUES (1)")
        if current_version < 2:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ingested_files (
                    file_path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    last_modified DATETIME NOT NULL,
                    ingested_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'completed'
                )
            """)
            cur.execute("INSERT INTO schema_version (version) VALUES (2)")
        if current_version < 3:
            # Add versioning and history tables
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_history (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    changed_at DATETIME NOT NULL,
                    change_type TEXT NOT NULL,
                    change_metadata TEXT,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_diffs (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    diff TEXT NOT NULL,
                    applied_at DATETIME NOT NULL,
                    compressed INTEGER DEFAULT 0,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """)
            # Add conflict tracking
            cur.execute("""
                CREATE TABLE IF NOT EXISTS code_conflicts (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    base_content TEXT NOT NULL,
                    our_content TEXT NOT NULL,
                    their_content TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    resolved_at DATETIME,
                    resolution_type TEXT,
                    resolved_by TEXT,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """)
            # Add real-time sync status tracking
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sync_operations (
                    id TEXT PRIMARY KEY,
                    started_at DATETIME NOT NULL,
                    completed_at DATETIME,
                    status TEXT DEFAULT 'pending',
                    error_message TEXT,
                    document_ids TEXT,
                    agent_id TEXT,
                    metadata TEXT
                )
            """)
            # Add database synchronization locks
            cur.execute("""
                CREATE TABLE IF NOT EXISTS db_locks (
                    resource_id TEXT PRIMARY KEY,
                    lock_holder TEXT NOT NULL,
                    acquired_at DATETIME NOT NULL,
                    expires_at DATETIME,
                    metadata TEXT
                )
            """)
            # Add distributed change notification tracking
            cur.execute("""
                CREATE TABLE IF NOT EXISTS change_notifications (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    notification_type TEXT NOT NULL,
                    sent_at DATETIME NOT NULL,
                    recipients TEXT,
                    acknowledgements TEXT,
                    metadata TEXT,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """)
            # Create indices for performance
            cur.execute("CREATE INDEX IF NOT EXISTS idx_document_history_document_id ON document_history(document_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_document_history_changed_at ON document_history(changed_at)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_document_diffs_document_id ON document_diffs(document_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_collection_id ON documents(collection_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_code_conflicts_document_id ON code_conflicts(document_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sync_operations_status ON sync_operations(status)")
            
            # Add transaction functions
            cur.execute("""
                CREATE TABLE IF NOT EXISTS db_transactions (
                    id TEXT PRIMARY KEY,
                    started_at DATETIME NOT NULL,
                    completed_at DATETIME,
                    status TEXT DEFAULT 'running',
                    operation_type TEXT NOT NULL,
                    affected_resources TEXT,
                    metadata TEXT
                )
            """)
            
            # Add full-text search capability
            cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS document_fts USING fts5(title, content, document_id UNINDEXED)")
            
            # Update schema version
            cur.execute("INSERT INTO schema_version (version) VALUES (3)")
            
        self.conn.commit()
        
        # Initialize foreign key enforcement
        cur.execute("PRAGMA foreign_keys = ON")

    def create_collection(self, name: str, description: str = None) -> str:
        collection_id = str(uuid.uuid4())
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO collections (id, name, description)
            VALUES (?, ?, ?)
        """, (collection_id, name, description))
        self.conn.commit()
        logging.info(f"Created collection '{name}' with id {collection_id}")
        return collection_id

    async def add_knowledge_entry(self, kb_id: str, title: str, content: str, embedding: List[float]) -> str:
        doc_id = str(uuid.uuid4())
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO documents (id, collection_id, title, content)
            VALUES (?, ?, ?, ?)
        """, (doc_id, kb_id, title, content))
        self.conn.commit()
        logging.info(f"Added knowledge entry {doc_id} titled '{title}' to collection {kb_id}")
        return doc_id

    async def search_knowledge(self, query: str, kb_id: Optional[str] = None, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from the knowledge base.
        
        Args:
            query: Search query text
            kb_id: Optional collection ID to search within
            top_n: Maximum number of results to return (default: 3)
            
        Returns:
            List of matching documents sorted by relevance
        """
        cur = self.conn.cursor()
        if kb_id:
            cur.execute("SELECT * FROM documents WHERE collection_id = ?", (kb_id,))
        else:
            cur.execute("SELECT * FROM documents")
        rows = cur.fetchall()
        if not rows:
            return []
            
        # Get query embedding
        query_emb = await compute_embedding(query, db_conn=self.conn)
        if not query_emb:
            return []
            
        query_vec = np.array(query_emb)
        scored = []
        
        # Calculate similarity for each document
        for row in rows:
            content = row["content"]
            doc_emb = await compute_embedding(content, db_conn=self.conn)
            if doc_emb:
                similarity = 1 - spatial.distance.cosine(query_vec, np.array(doc_emb))
                row_dict = dict(row)
                row_dict["similarity"] = similarity
                scored.append(row_dict)
                
        # Sort by similarity score (highest first)
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top matches
        return scored[:top_n]

# ------------------------------------------------------------------------------
# AST-based Code Chunking for Self-Reflection
# ------------------------------------------------------------------------------

class ASTChunker:
    """
    Parse a Python file with the ast module to extract top-level functions and classes.
    """
    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        lines = source.splitlines(keepends=True)
        chunks = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start_line = node.lineno - 1
                end_line = node.end_lineno
                chunk_text = "".join(lines[start_line:end_line])
                chunk_type = "function" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "class"
                chunks.append({
                    "type": chunk_type,
                    "name": node.name,
                    "start_line": start_line + 1,
                    "end_line": end_line,
                    "content": chunk_text
                })
        return chunks

# ------------------------------------------------------------------------------
# AdvancedCodeManager: Self-Modifying, Error-Correcting & Self-Assembling
# ------------------------------------------------------------------------------

class AdvancedCodeManager:
    """
    Loads and manages the agent's codebase with advanced indexing and rewriting capabilities.
    Features:
    - Full codebase indexing with AST parsing
    - Code rewriting with error correction and in-memory validation
    - Dynamic plugin loading
    - Code change validation and rollback
    - Semantic code search
    - File locking for safe modifications
    - In-memory code execution for testing
    - In-memory code modification without changing files
    - Database synchronization of code changes
    - Memory snapshots and versioning for code changes
    - Hot-swapping of code modules in runtime
    - Conflict resolution for concurrent memory edits
    - Live code patching without restart
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.backup_path = file_path + ".bak"
        self.code_index = {}  # Maps file paths to AST nodes
        self.function_index = {}  # Maps function names to locations
        self.class_index = {}  # Maps class names to locations
        self.dependency_graph = nx.DiGraph()  # Tracks code dependencies
        self.modification_history = []  # Tracks all code changes
        self.in_memory_changes = {}  # Stores proposed changes before committing
        self.file_locks = {}  # Tracks file locks
        self.memory_code = {}  # Stores in-memory code versions without file changes
        self.memory_snapshots = {}  # Stores versioned snapshots of memory code
        self.active_memory_version = {}  # Current active version for each file
        self.edit_conflicts = {}  # Tracks edit conflicts
        self.loaded_modules = {}  # Tracks loaded modules for hot-swapping
        self.memory_locks = {}  # Locks for concurrent memory access
        self.patch_listeners = []  # Callbacks for live code patching
        self.sync_with_db = True  # Flag to control database synchronization
        self.last_snapshot_id = 0  # Counter for snapshot IDs
        self.auto_snapshot = True  # Whether to automatically take snapshots
        self.load_code()

    def load_code(self):
        """Load and index all code files in the project"""
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.lines = f.readlines()
            
        # Initialize memory code with the file content
        self.memory_code[self.file_path] = self.lines.copy()
            
        # Index the main file
        self._index_file(self.file_path)
        
        # Index all Python files in the project
        project_root = os.path.dirname(os.path.abspath(self.file_path))
        for root, _, files in os.walk(project_root):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.readlines()
                        self.memory_code[file_path] = content
                    self._index_file(file_path)
    
    def get_full_code(self) -> str:
        """Get the full code as a string"""
        if self.file_path in self.memory_code:
            return "".join(self.memory_code[self.file_path])
        return "".join(self.lines)
                    
    def _index_file(self, file_path: str):
        """Create AST-based index for a Python file"""
        try:
            # Check if we have in-memory code for this file
            if file_path in self.memory_code:
                source = "".join(self.memory_code[file_path])
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source = f.read()
            
            tree = ast.parse(source)
                
            # Store file's AST
            self.code_index[file_path] = tree
            
            # Index functions and classes
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.function_index[node.name] = {
                        'file': file_path,
                        'line': node.lineno,
                        'end_line': node.end_lineno,
                        'ast_node': node
                    }
                elif isinstance(node, ast.ClassDef):
                    self.class_index[node.name] = {
                        'file': file_path,
                        'line': node.lineno,
                        'end_line': node.end_lineno,
                        'ast_node': node
                    }
                    
            # Build dependency graph
            self._analyze_dependencies(file_path, tree)
            
        except Exception as e:
            logging.error(f"Error indexing {file_path}: {e}")
            
    def _analyze_dependencies(self, file_path: str, tree: ast.AST):
        """Analyze and record code dependencies"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(n.name for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
                    
        # Add edges to dependency graph
        for imp in imports:
            self.dependency_graph.add_edge(file_path, imp)

    def backup(self):
        """Create backup of all indexed files"""
        for file_path in self.code_index:
            backup_path = file_path + ".bak"
            shutil.copy2(file_path, backup_path)
            
    def restore_backup(self):
        """Restore all files from backup"""
        restored = []
        for file_path in self.code_index:
            backup_path = file_path + ".bak"
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, file_path)
                restored.append(file_path)
        self.load_code()
        logging.info(f"Restored {len(restored)} files from backup")

    def compile_code(self) -> bool:
        """Compile and validate all indexed code"""
        try:
            # Compile each indexed file
            for file_path, tree in self.code_index.items():
                source = ast.unparse(tree)
                compile(source, file_path, 'exec')
                
            # Validate imports and dependencies
            import_errors = self._validate_imports()
            if import_errors:
                logging.error(f"Import validation errors: {import_errors}")
                return False
                
            return True
        except Exception as e:
            logging.error(f"Compilation error: {e}")
            return False
            
    def _validate_imports(self) -> List[str]:
        """Validate all imports across the codebase"""
        errors = []
        for file_path in self.code_index:
            try:
                spec = importlib.util.spec_from_file_location(
                    os.path.basename(file_path)[:-3], 
                    file_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as e:
                errors.append(f"{file_path}: {str(e)}")
        return errors

    def get_code_element(self, element_type: str, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a code element by name"""
        if element_type == "function":
            return self.function_index.get(name)
        elif element_type == "class":
            return self.class_index.get(name)
        return None
        
    def find_code(self, query: str) -> List[Dict[str, Any]]:
        """Search for code elements using pattern matching"""
        results = []
        pattern = re.compile(query, re.IGNORECASE)
        
        # Search functions
        for name, info in self.function_index.items():
            if pattern.search(name) or pattern.search(ast.unparse(info['ast_node'])):
                results.append({
                    'type': 'function',
                    'name': name,
                    **info
                })
                
        # Search classes
        for name, info in self.class_index.items():
            if pattern.search(name) or pattern.search(ast.unparse(info['ast_node'])):
                results.append({
                    'type': 'class',
                    'name': name,
                    **info
                })
                
        return results
        
    def get_dependencies(self, element_name: str) -> Dict[str, List[str]]:
        """Get dependencies for a code element"""
        deps = {
            'imports': [],
            'called_by': [],
            'calls': []
        }
        
        element = self.function_index.get(element_name) or self.class_index.get(element_name)
        if not element:
            return deps
            
        # Analyze AST node for dependencies
        node = element['ast_node']
        for child in ast.walk(node):
            if isinstance(child, ast.Import):
                deps['imports'].extend(n.name for n in child.names)
            elif isinstance(child, ast.ImportFrom):
                if child.module:
                    deps['imports'].append(child.module)
            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    deps['calls'].append(child.func.id)
                    
        # Find functions that call this one
        for fname, finfo in self.function_index.items():
            if fname != element_name:
                for node in ast.walk(finfo['ast_node']):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        if node.func.id == element_name:
                            deps['called_by'].append(fname)
                            
        return deps

    def create_memory_snapshot(self, file_path: str, description: str = None) -> int:
        """
        Create a snapshot of the current memory code for a file
        
        Args:
            file_path: Path to the file to snapshot
            description: Optional description of the snapshot
            
        Returns:
            int: Snapshot ID
        """
        if file_path not in self.memory_code:
            return -1
            
        # Create snapshot
        self.last_snapshot_id += 1
        snapshot_id = self.last_snapshot_id
        
        if file_path not in self.memory_snapshots:
            self.memory_snapshots[file_path] = {}
            
        # Copy the current memory code
        self.memory_snapshots[file_path][snapshot_id] = {
            'code': self.memory_code[file_path].copy(),
            'timestamp': datetime.now().isoformat(),
            'description': description or f"Snapshot {snapshot_id}",
            'index': {
                'functions': {k: v.copy() for k, v in self.function_index.items() 
                            if v.get('file') == file_path},
                'classes': {k: v.copy() for k, v in self.class_index.items() 
                          if v.get('file') == file_path}
            }
        }
        
        # Update active version
        self.active_memory_version[file_path] = snapshot_id
        
        logging.info(f"Created memory snapshot {snapshot_id} for {file_path}")
        return snapshot_id
        
    def restore_memory_snapshot(self, file_path: str, snapshot_id: int) -> bool:
        """
        Restore a memory snapshot for a file
        
        Args:
            file_path: Path to the file
            snapshot_id: ID of the snapshot to restore
            
        Returns:
            bool: Success status
        """
        if (file_path not in self.memory_snapshots or 
            snapshot_id not in self.memory_snapshots[file_path]):
            return False
            
        # Get snapshot
        snapshot = self.memory_snapshots[file_path][snapshot_id]
        
        # Restore memory code
        self.memory_code[file_path] = snapshot['code'].copy()
        
        # Restore indexes (filtering only for this file)
        for func_name, func_info in snapshot['index']['functions'].items():
            self.function_index[func_name] = func_info.copy()
            
        for class_name, class_info in snapshot['index']['classes'].items():
            self.class_index[class_name] = class_info.copy()
            
        # Update active version
        self.active_memory_version[file_path] = snapshot_id
        
        logging.info(f"Restored memory snapshot {snapshot_id} for {file_path}")
        return True
        
    def get_memory_snapshots(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Get all memory snapshots for a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of snapshot metadata
        """
        if file_path not in self.memory_snapshots:
            return []
            
        return [
            {
                'id': snapshot_id,
                'timestamp': info['timestamp'],
                'description': info['description'],
                'active': self.active_memory_version.get(file_path) == snapshot_id
            }
            for snapshot_id, info in self.memory_snapshots[file_path].items()
        ]
        
    async def hot_swap_module(self, file_path: str) -> bool:
        """
        Hot-swap a module with its in-memory version
        
        Args:
            file_path: Path to the module file
            
        Returns:
            bool: Success status
        """
        if file_path not in self.memory_code:
            return False
            
        try:
            # Get module name from file path
            module_name = os.path.basename(file_path).replace('.py', '')
            
            # Check if module is loaded
            if module_name not in sys.modules:
                logging.warning(f"Module {module_name} is not loaded, cannot hot-swap")
                return False
                
            # Create a temporary module
            temp_module = types.ModuleType(module_name)
            
            # Compile the memory code
            memory_code = "".join(self.memory_code[file_path])
            code_obj = compile(memory_code, file_path, 'exec')
            
            # Execute the compiled code in the temporary module
            exec(code_obj, temp_module.__dict__)
            
            # Store the original module
            original_module = sys.modules[module_name]
            self.loaded_modules[module_name] = original_module
            
            # Replace the module in sys.modules
            sys.modules[module_name] = temp_module
            
            # Update any imported references
            for mod_name, mod in list(sys.modules.items()):
                if mod_name != module_name and hasattr(mod, module_name):
                    setattr(mod, module_name, temp_module)
                    
            # Notify patch listeners
            for listener in self.patch_listeners:
                try:
                    asyncio.create_task(listener(module_name, temp_module))
                except Exception as e:
                    logging.error(f"Error notifying patch listener: {e}")
                    
            logging.info(f"Hot-swapped module {module_name}")
            return True
            
        except Exception as e:
            logging.error(f"Error hot-swapping module: {e}")
            return False
            
    def register_patch_listener(self, listener_func) -> int:
        """
        Register a listener for code patches
        
        Args:
            listener_func: Async function to call when code is patched
            
        Returns:
            int: Listener ID
        """
        self.patch_listeners.append(listener_func)
        return len(self.patch_listeners) - 1
        
    def unregister_patch_listener(self, listener_id: int) -> bool:
        """
        Unregister a patch listener
        
        Args:
            listener_id: ID of the listener to unregister
            
        Returns:
            bool: Success status
        """
        if 0 <= listener_id < len(self.patch_listeners):
            self.patch_listeners.pop(listener_id)
            return True
        return False
        
    async def edit_in_memory(self, file_path: str, new_content: str, reindex: bool = True, 
                            sync_db: bool = None, take_snapshot: bool = None) -> bool:
        """
        Edit code in memory without modifying the actual file.
        
        Args:
            file_path: Path to the file to modify in memory
            new_content: New content for the file
            reindex: Whether to reindex the file after modification
            sync_db: Whether to sync changes with the database (None uses global setting)
            take_snapshot: Whether to take a snapshot (None uses auto_snapshot setting)
            
        Returns:
            bool: Success status
        """
        # Acquire memory lock
        if file_path in self.memory_locks:
            # Wait for lock to be released (with timeout)
            for _ in range(10):  # Try for up to 1 second
                if not self.memory_locks[file_path]:
                    break
                await asyncio.sleep(0.1)
                
            if self.memory_locks[file_path]:
                logging.error(f"Timeout waiting for memory lock on {file_path}")
                return False
                
        # Set memory lock
        self.memory_locks[file_path] = True
        
        try:
            # Check for conflicts
            if file_path in self.edit_conflicts and self.edit_conflicts[file_path]:
                # Try to resolve conflict
                resolved = await self._resolve_edit_conflict(file_path, new_content)
                if not resolved:
                    logging.warning(f"Unresolved edit conflict for {file_path}")
                    # Continue anyway, but mark as conflicted
            
            # Try to compile first to validate syntax
            compile(new_content, file_path, 'exec')
            
            # Create snapshot if needed
            should_snapshot = self.auto_snapshot if take_snapshot is None else take_snapshot
            if should_snapshot and file_path in self.memory_code:
                self.create_memory_snapshot(file_path, "Pre-edit snapshot")
            
            # Store new content as lines
            if isinstance(new_content, str):
                self.memory_code[file_path] = new_content.splitlines(True)
            else:
                self.memory_code[file_path] = new_content
                
            # Log change
            self.modification_history.append({
                'timestamp': datetime.now().isoformat(),
                'file_path': file_path,
                'action': 'memory_edit',
                'content_length': len(new_content)
            })
            
            # Reindex if requested
            if reindex:
                self._index_file(file_path)
                
            # Sync with DB if requested
            should_sync = self.sync_with_db if sync_db is None else sync_db
            if should_sync:
                asyncio.create_task(self._sync_memory_code_with_db(file_path))
                
            # Clear any conflicts
            if file_path in self.edit_conflicts:
                self.edit_conflicts[file_path] = []
                
            return True
            
        except Exception as e:
            logging.error(f"Error editing code in memory: {e}")
            return False
            
        finally:
            # Release memory lock
            self.memory_locks[file_path] = False
            
    def get_memory_code(self, file_path: str) -> str:
        """Get the in-memory version of the file's code"""
        if file_path in self.memory_code:
            return "".join(self.memory_code[file_path])
        return None
        
    async def _resolve_edit_conflict(self, file_path: str, new_content: str) -> bool:
        """
        Attempt to resolve edit conflicts
        
        Args:
            file_path: Path to the file
            new_content: New content being applied
            
        Returns:
            bool: Whether the conflict was resolved
        """
        import difflib
        
        if file_path not in self.edit_conflicts:
            return True
            
        conflicts = self.edit_conflicts[file_path]
        if not conflicts:
            return True
            
        try:
            # Get the current memory content
            current_content = "".join(self.memory_code[file_path])
            
            # For each conflict, try to merge changes
            resolved_conflicts = []
            
            for conflict in conflicts:
                conflict_content = conflict.get('content', '')
                
                # Get a diff of the conflict content vs current memory
                conflict_diff = list(difflib.unified_diff(
                    current_content.splitlines(),
                    conflict_content.splitlines(),
                    n=3
                ))
                
                # Get a diff of the new content vs current memory
                new_diff = list(difflib.unified_diff(
                    current_content.splitlines(),
                    new_content.splitlines(),
                    n=3
                ))
                
                # Check if the diffs overlap (edit the same lines)
                conflict_hunks = self._parse_diff_hunks(conflict_diff)
                new_hunks = self._parse_diff_hunks(new_diff)
                
                overlap = False
                for c_hunk in conflict_hunks:
                    for n_hunk in new_hunks:
                        if self._hunks_overlap(c_hunk, n_hunk):
                            overlap = True
                            break
                    if overlap:
                        break
                        
                if not overlap:
                    # No overlap, can be auto-merged
                    resolved_conflicts.append(conflict)
                    logging.info(f"Auto-resolved conflict for {file_path}")
                    
            # Remove resolved conflicts
            for conflict in resolved_conflicts:
                if conflict in self.edit_conflicts[file_path]:
                    self.edit_conflicts[file_path].remove(conflict)
                    
            return len(self.edit_conflicts[file_path]) == 0
            
        except Exception as e:
            logging.error(f"Error resolving edit conflict: {e}")
            return False
            
    def _parse_diff_hunks(self, diff_lines: List[str]) -> List[Dict[str, Any]]:
        """
        Parse diff output into hunks
        
        Args:
            diff_lines: Output from difflib.unified_diff
            
        Returns:
            List of hunks, each with start and end line numbers
        """
        hunks = []
        current_hunk = None
        
        for line in diff_lines:
            if line.startswith('@@'):
                # New hunk
                if current_hunk:
                    hunks.append(current_hunk)
                    
                # Parse hunk header like @@ -1,7 +1,6 @@
                hunk_re = r'@@ -(\d+),(\d+) \+(\d+),(\d+) @@'
                match = re.search(hunk_re, line)
                if match:
                    a_start, a_len, b_start, b_len = map(int, match.groups())
                    current_hunk = {
                        'a_start': a_start,
                        'a_end': a_start + a_len - 1,
                        'b_start': b_start,
                        'b_end': b_start + b_len - 1
                    }
        
        if current_hunk:
            hunks.append(current_hunk)
            
        return hunks
        
    def _hunks_overlap(self, hunk1: Dict[str, Any], hunk2: Dict[str, Any]) -> bool:
        """
        Check if two diff hunks overlap
        
        Args:
            hunk1: First hunk
            hunk2: Second hunk
            
        Returns:
            bool: Whether the hunks overlap
        """
        # Check if the ranges overlap
        return (
            (hunk1['a_start'] <= hunk2['a_end'] and hunk1['a_end'] >= hunk2['a_start']) or
            (hunk1['b_start'] <= hunk2['b_end'] and hunk1['b_end'] >= hunk2['b_start'])
        )
        
    def mark_edit_conflict(self, file_path: str, editor_id: str, content: str) -> None:
        """
        Mark an edit conflict for a file
        
        Args:
            file_path: Path to the file
            editor_id: ID of the editor that made the conflicting change
            content: The conflicting content
        """
        if file_path not in self.edit_conflicts:
            self.edit_conflicts[file_path] = []
            
        self.edit_conflicts[file_path].append({
            'editor_id': editor_id,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        
        logging.warning(f"Edit conflict marked for {file_path} by {editor_id}")
        
    def get_memory_file_diff(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Get differences between in-memory code and actual file
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            List of dictionaries with differences: 
            [{'line': line_number, 'file': file_line, 'memory': memory_line}]
        """
        import difflib
        
        if file_path not in self.memory_code:
            return []
            
        # Get memory code
        memory_lines = self.memory_code[file_path]
        
        # Get file code
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_lines = f.readlines()
        except Exception:
            return [{'line': 0, 'file': 'FILE_NOT_FOUND', 'memory': ''}]
            
        # Compare line by line
        differences = []
        for i, (file_line, memory_line) in enumerate(zip(file_lines, memory_lines)):
            if file_line != memory_line:
                differences.append({
                    'line': i + 1,
                    'file': file_line,
                    'memory': memory_line
                })
        
        # Handle case where one is longer than the other
        if len(file_lines) > len(memory_lines):
            for i in range(len(memory_lines), len(file_lines)):
                differences.append({
                    'line': i + 1,
                    'file': file_lines[i],
                    'memory': 'MISSING_IN_MEMORY'
                })
        elif len(memory_lines) > len(file_lines):
            for i in range(len(file_lines), len(memory_lines)):
                differences.append({
                    'line': i + 1,
                    'file': 'MISSING_IN_FILE',
                    'memory': memory_lines[i]
                })
                
        return differences
        
    def set_sync_with_db(self, enabled: bool):
        """Set whether database synchronization is enabled"""
        self.sync_with_db = enabled
        
    async def _sync_memory_code_with_db(self, file_path: str):
        """Synchronize in-memory code with the database"""
        try:
            # The actual sync is delegated to the SelfTransformation class
            # This will be properly connected when an agent initializes the code manager
            # For now, we store the request for later processing
            self.modification_history.append({
                'timestamp': datetime.now().isoformat(),
                'file_path': file_path,
                'action': 'sync_request',
                'pending': True
            })
            logging.info(f"Requested sync of in-memory code for {file_path} with database")
        except Exception as e:
            logging.error(f"Error syncing memory code with DB: {e}")
            
    def set_sync_handler(self, handler):
        """
        Set a handler function that will be called to sync code with the database
        
        Args:
            handler: Async function that takes a file_path parameter
        """
        self._sync_handler = handler
        
        # Process any pending sync requests
        pending_syncs = [m for m in self.modification_history 
                         if m.get('action') == 'sync_request' and m.get('pending', False)]
        
        for sync_request in pending_syncs:
            file_path = sync_request.get('file_path')
            if file_path:
                asyncio.create_task(handler(file_path))
                sync_request['pending'] = False
        
    async def propose_change(self, file_path: str, new_content: str) -> bool:
        """Propose a change to be validated in memory before committing"""
        try:
            # Store proposed change
            self.in_memory_changes[file_path] = new_content
            
            # Create temporary module for validation
            temp_module = types.ModuleType('temp_validation')
            exec(new_content, temp_module.__dict__)
            
            # Try to compile
            compile(new_content, file_path, 'exec')
            
            # Run any available tests
            test_result = await self._run_validation_tests(temp_module)
            if not test_result:
                del self.in_memory_changes[file_path]
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Validation error for {file_path}: {e}")
            del self.in_memory_changes[file_path]
            return False
            
    async def commit_changes(self) -> bool:
        """Commit all validated changes with file locking"""
        if not self.in_memory_changes:
            return False
            
        try:
            # Acquire locks for all files
            for file_path in self.in_memory_changes:
                if file_path in self.file_locks:
                    logger.error(f"File {file_path} is locked")
                    return False
                self.file_locks[file_path] = True
                
            # Create backups
            self.backup()
            
            # Write all changes
            for file_path, content in self.in_memory_changes.items():
                with open(file_path, 'w') as f:
                    f.write(content)
                    
            # Reindex changed files
            for file_path in self.in_memory_changes:
                self._index_file(file_path)
                
            # Clear changes and release locks
            self.in_memory_changes.clear()
            for file_path in list(self.file_locks):
                del self.file_locks[file_path]
                
            return True
            
        except Exception as e:
            logger.error(f"Error committing changes: {e}")
            self.restore_backup()
            self.in_memory_changes.clear()
            for file_path in list(self.file_locks):
                del self.file_locks[file_path]
            return False
            
    async def _run_validation_tests(self, module) -> bool:
        """Run validation tests on proposed changes"""
        try:
            # Look for test functions in module
            test_funcs = [
                getattr(module, name) for name in dir(module)
                if name.startswith('test_') and callable(getattr(module, name))
            ]
            
            # Run tests
            for test in test_funcs:
                if asyncio.iscoroutinefunction(test):
                    await test()
                else:
                    test()
                    
            return True
            
        except Exception as e:
            logger.error(f"Validation tests failed: {e}")
            return False

    def modify_code_element(self, element_type: str, name: str, new_code: str) -> bool:
        """Modify a code element with validation"""
        try:
            # Find the element
            if element_type == "function":
                element = self.function_index.get(name)
            elif element_type == "class":
                element = self.class_index.get(name)
            else:
                return False
                
            if not element:
                return False
                
            # Parse new code
            try:
                new_node = ast.parse(new_code).body[0]
            except Exception as e:
                logging.error(f"Invalid Python syntax in new code: {e}")
                return False
                
            # Validate new code maintains interface
            if not self._validate_interface(element['ast_node'], new_node):
                logging.error("New code breaks existing interface")
                return False
                
            # Create backup
            self.backup()
                
            # Replace the code
            file_path = element['file']
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            # Replace the specific lines
            start = element['line'] - 1
            end = element['end_line']
            new_lines = new_code.splitlines(True)
            lines[start:end] = new_lines
                
            # Write back
            with open(file_path, 'w') as f:
                f.writelines(lines)
                
            # Reindex
            self._index_file(file_path)
                
            # Validate compilation
            if not self.compile_code():
                self.restore_backup()
                return False
                
            # Record modification
            self.modification_history.append({
                'timestamp': datetime.now().isoformat(),
                'element_type': element_type,
                'name': name,
                'old_code': ast.unparse(element['ast_node']),
                'new_code': new_code
            })
                
            return True
                
        except Exception as e:
            logging.error(f"Error modifying code element: {e}")
            self.restore_backup()
            return False
            
    def _validate_interface(self, old_node: ast.AST, new_node: ast.AST) -> bool:
        """Validate that new code maintains the existing interface"""
        if type(old_node) != type(new_node):
            return False
            
        if isinstance(old_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Check function signature
            old_args = ast.unparse(old_node.args)
            new_args = ast.unparse(new_node.args)
            if old_args != new_args:
                return False
                
            # Check return annotation if exists
            if (hasattr(old_node, 'returns') and old_node.returns and 
                (not hasattr(new_node, 'returns') or 
                 ast.unparse(old_node.returns) != ast.unparse(new_node.returns))):
                return False
                
        elif isinstance(old_node, ast.ClassDef):
            # Check base classes
            old_bases = {ast.unparse(b) for b in old_node.bases}
            new_bases = {ast.unparse(b) for b in new_node.bases}
            if old_bases != new_bases:
                return False
                
            # Check public methods
            old_methods = {n.name for n in ast.walk(old_node) 
                         if isinstance(n, ast.FunctionDef) and not n.name.startswith('_')}
            new_methods = {n.name for n in ast.walk(new_node) 
                         if isinstance(n, ast.FunctionDef) and not n.name.startswith('_')}
            if old_methods != new_methods:
                return False
                
        return True

    async def stream_tokens(self, text: str, delay: float = 0.05) -> AsyncGenerator[str, None]:
        tokens = re.split(r"(\s+)", text)
        for token in tokens:
            await asyncio.sleep(delay)
            yield token

    async def apply_modification(self, operation: str, params: Dict[str, Any]) -> AsyncGenerator[str, None]:
        yield "Starting modification...\n"
        if operation == "line":
            line_no = int(params["line_number"])
            new_line = params.get("new_line", "")
            self.set_line(line_no, new_line)
            yield f"Modified line {line_no}: {new_line}\n"
        elif operation == "block":
            st_marker = params["start_marker"]
            ed_marker = params["end_marker"]
            nb = params.get("new_block", [])
            self.replace_block(st_marker, ed_marker, nb)
            yield f"Replaced block from '{st_marker}' to '{ed_marker}'\n"
        else:
            yield "Unknown operation.\n"
        self.backup()
        yield "Backup created. Writing changes...\n"
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.writelines(self.lines)
        if not self.compile_code():
            yield "Compilation failed; reverting backup...\n"
            self.restore_backup()
            yield "Reversion complete.\n"
        else:
            yield "Modification complete and compiled successfully.\n"

    def load_plugins(self, plugins_dir: str = "plugins") -> List[str]:
        """
        Dynamically load Python modules from the plugins directory.
        If a module fails to load, try to load its backup (if exists) and log errors.
        """
        loaded_modules = []
        if not os.path.exists(plugins_dir):
            os.makedirs(plugins_dir)
            with open(os.path.join(plugins_dir, "__init__.py"), "w") as f:
                f.write("")
        for filename in os.listdir(plugins_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = filename[:-3]
                module_path = f"{plugins_dir.replace(os.sep, '.')}.{module_name}"
                try:
                    if module_path in sys.modules:
                        module = importlib.reload(sys.modules[module_path])
                    else:
                        module = importlib.import_module(module_path)
                    loaded_modules.append(module_name)
                    logging.info(f"Loaded plugin: {module_name}")
                except Exception as e:
                    logging.error(f"Error loading plugin {module_name}: {e}")
                    # Attempt to load backup if exists:
                    backup_file = os.path.join(plugins_dir, f"{module_name}.bak.py")
                    if os.path.exists(backup_file):
                        try:
                            spec = importlib.util.spec_from_file_location(module_name, backup_file)
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            loaded_modules.append(module_name)
                            logging.info(f"Loaded backup plugin: {module_name}")
                        except Exception as e2:
                            logging.error(f"Error loading backup plugin {module_name}: {e2}")
        return loaded_modules

# ------------------------------------------------------------------------------
# RL Environment for Self-Modification
# ------------------------------------------------------------------------------

class RLBootstrapEnv(gym.Env):
    """
    A Gym environment in which the agent can modify its own source code.
    Observations: normalized code length.
    Reward: 1 if the code shortens, -0.5 otherwise.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, code_manager: AdvancedCodeManager):
        super(RLBootstrapEnv, self).__init__()
        self.code_manager = code_manager
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.max_steps = 10
        self.current_step = 0
        self.initial_length = len(self.code_manager.lines)

    def reset(self):
        self.current_step = 0
        return np.array([len(self.code_manager.lines)/1000.0], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        info = {}
        if action == 0:
            ln = random.randint(0, len(self.code_manager.lines)-1)
            mod_line = f"# RL modified line at step {self.current_step}\n"
            self.code_manager.set_line(ln, mod_line)
            info["modification"] = f"line {ln}"
        elif action == 1:
            try:
                st_marker = "# BLOCK START"
                ed_marker = "# BLOCK END"
                rep = [f"# RL replaced block at step {self.current_step}\n"]
                self.code_manager.replace_block(st_marker, ed_marker, rep)
                info["modification"] = "block replaced"
            except Exception as e:
                info["modification"] = f"block replacement failed: {e}"
        self.code_manager.backup()
        with open(self.code_manager.file_path, "w", encoding="utf-8") as f:
            f.writelines(self.code_manager.lines)
        new_len = len(self.code_manager.lines)
        reward = 1.0 if new_len < self.initial_length else -0.5
        obs = np.array([new_len/1000.0], dtype=np.float32)
        done = self.current_step >= self.max_steps
        return obs, reward, done, info

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Code length: {len(self.code_manager.lines)}")

# ------------------------------------------------------------------------------
# Agent with Self-Reflection, MCTS, RL, and Self-Modifying Code
# ------------------------------------------------------------------------------

class Node:
    """
    A node in a Monte Carlo Tree Search tree.
    """
    def __init__(self, state: Dict[str, Any], parent: Optional['Node'] = None):
        self.state = state
        self.parent = parent
        self.children: List[Node] = []
        self.visits = 0
        self.value = 0.0

class CodeEvolutionTracker:
    """Tracks and visualizes code evolution over time"""
    def __init__(self):
        self.evolution_graph = nx.DiGraph()
        self.current_version = 0
        
    def add_transformation(self, original: Dict[str, Any], modified: Dict[str, Any], metadata: Dict[str, Any]):
        """Add a transformation to the evolution graph"""
        timestamp = datetime.now().isoformat()
        from_node = f"v{self.current_version}"
        to_node = f"v{self.current_version + 1}"
        
        # Add nodes
        self.evolution_graph.add_node(from_node, 
                                    code=original["content"],
                                    timestamp=timestamp)
        self.evolution_graph.add_node(to_node, 
                                    code=modified["content"],
                                    timestamp=timestamp)
        
        # Add edge with metadata
        self.evolution_graph.add_edge(from_node, 
                                    to_node,
                                    metadata=metadata)
        
        self.current_version += 1
        
    def visualize(self, output_path: str = "code_evolution.png"):
        """Generate a visualization of the code evolution graph"""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.evolution_graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.evolution_graph, pos, 
                             node_color='lightblue',
                             node_size=1000)
        
        # Draw edges
        nx.draw_networkx_edges(self.evolution_graph, pos, 
                             edge_color='gray',
                             arrows=True)
        
        # Add labels
        labels = {node: f"Version {node}\n{data['timestamp'][:10]}"
                 for node, data in self.evolution_graph.nodes(data=True)}
        nx.draw_networkx_labels(self.evolution_graph, pos, labels)
        
        plt.title("Code Evolution Graph")
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()
        
    def get_transformation_history(self) -> List[Dict[str, Any]]:
        """Get complete transformation history"""
        history = []
        for edge in self.evolution_graph.edges(data=True):
            from_node, to_node, data = edge
            history.append({
                "from_version": from_node,
                "to_version": to_node,
                "metadata": data["metadata"],
                "from_code": self.evolution_graph.nodes[from_node]["code"],
                "to_code": self.evolution_graph.nodes[to_node]["code"],
                "timestamp": self.evolution_graph.nodes[to_node]["timestamp"]
            })
        return history

class SelfTransformation:
    """Handles dynamic code transformation with safety checks, testing, and autonomous evolution"""
    def __init__(self, agent_instance):
        self.agent = agent_instance
        self.code_manager = AdvancedCodeManager(__file__)
        self.transformation_history = []
        self.safety_checks_enabled = True
        self.evolution_tracker = CodeEvolutionTracker()
        self.autonomous_mode = False
        self.learning_rate = 0.1
        self.max_autonomous_changes = 5
        self.changes_counter = 0
        self.test_instances = {}  # Stores test instances of the agent
        self.test_results = {}    # Stores test results
        self.sync_lock = asyncio.Lock()  # Lock for synchronizing DB operations
        
        # Set up the sync handler to connect the components
        self.code_manager.set_sync_handler(self.sync_memory_code_with_database)
        
    async def edit_in_memory(self, file_path: str, new_content: str, sync_to_db: bool = True, 
                            take_snapshot: bool = True) -> bool:
        """
        Edit a file in memory without changing the file itself
        
        Args:
            file_path: Path to the file to edit
            new_content: New content for the file
            sync_to_db: Whether to sync changes to the database
            take_snapshot: Whether to take a snapshot before editing
            
        Returns:
            bool: Success status
        """
        return await self.code_manager.edit_in_memory(
            file_path, new_content, sync_db=sync_to_db, take_snapshot=take_snapshot)
            
    def create_memory_snapshot(self, file_path: str, description: str = None) -> int:
        """
        Create a snapshot of the current memory code
        
        Args:
            file_path: Path to the file to snapshot
            description: Optional description of the snapshot
            
        Returns:
            int: Snapshot ID
        """
        return self.code_manager.create_memory_snapshot(file_path, description)
        
    def restore_memory_snapshot(self, file_path: str, snapshot_id: int) -> bool:
        """
        Restore a memory snapshot
        
        Args:
            file_path: Path to the file
            snapshot_id: ID of the snapshot to restore
            
        Returns:
            bool: Success status
        """
        return self.code_manager.restore_memory_snapshot(file_path, snapshot_id)
        
    def get_memory_snapshots(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Get all memory snapshots for a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of snapshot metadata
        """
        return self.code_manager.get_memory_snapshots(file_path)
        
    async def hot_swap_module(self, file_path: str) -> bool:
        """
        Hot-swap a module with its in-memory version
        
        Args:
            file_path: Path to the module file
            
        Returns:
            bool: Success status
        """
        return await self.code_manager.hot_swap_module(file_path)
        
    async def propose_transformation(self, code_chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Generate proposed code improvements with advanced analysis"""
        prompt = f"""Analyze this code and suggest specific improvements:
        {code_chunk['content']}
        
        Focus on:
        1. Performance optimization
        2. Error handling and safety
        3. Code clarity and maintainability
        4. Security implications
        5. New functionality opportunities
        6. Integration possibilities
        7. Self-improvement capabilities
        8. Learning mechanisms
        
        Return specific code changes in this format:
        CHANGE: <line number>
        <new code>
        
        or
        
        BLOCK: <start_marker>
        <replacement code block>
        <end_marker>
        
        Include rationale for each change."""
        
        messages = [
            {"role": "system", "content": "You are an expert code improvement AI."},
            {"role": "user", "content": prompt}
        ]
        
        response = await client.chat.completions.create(
            model="o3-mini",
            messages=messages,
            temperature=0.2
        )
        
        return {
            "original": code_chunk,
            "suggestion": response.choices[0].message.content,
            "timestamp": datetime.now().isoformat()
        }
    
    async def create_test_instance(self, transformation: Dict[str, Any]) -> str:
        """Create a test instance of the agent with proposed changes"""
        instance_id = str(uuid.uuid4())
        
        try:
            # Create a memory-only clone of the code
            test_code_manager = AdvancedCodeManager(__file__)
            
            # Parse and apply changes to the test instance
            changes = self._parse_code_changes(transformation["suggestion"])
            for change in changes:
                await test_code_manager.apply_modification(
                    operation=change["operation"],
                    params=change["params"]
                )
            
            # Create a temporary module with the modified code
            module_name = f"test_agent_{instance_id}"
            spec = importlib.util.find_spec("types")
            test_module = importlib.util.module_from_spec(spec)
            
            # Execute the modified code in the test module
            exec(test_code_manager.get_full_code(), test_module.__dict__)
            
            # Create an instance of the agent in the test module
            test_agent = test_module.Agent(
                instructions="Test instance for code validation",
                id=f"test-{instance_id}"
            )
            
            # Store the test instance
            self.test_instances[instance_id] = {
                "agent": test_agent,
                "module": test_module,
                "code_manager": test_code_manager,
                "transformation": transformation,
                "created_at": datetime.now().isoformat()
            }
            
            logging.info(f"Created test instance {instance_id} for code validation")
            return instance_id
            
        except Exception as e:
            logging.error(f"Error creating test instance: {e}")
            return None
    
    async def test_transformation(self, instance_id: str) -> Dict[str, Any]:
        """Run tests on the test instance to validate changes"""
        if instance_id not in self.test_instances:
            return {"status": "error", "message": "Test instance not found"}
            
        instance = self.test_instances[instance_id]
        test_agent = instance["agent"]
        
        try:
            # Run a series of validation tests
            tests = [
                self._test_basic_functionality(test_agent),
                self._test_error_handling(test_agent),
                self._test_performance(test_agent)
            ]
            
            results = await asyncio.gather(*tests)
            
            # Aggregate test results
            test_results = {
                "instance_id": instance_id,
                "basic_functionality": results[0],
                "error_handling": results[1],
                "performance": results[2],
                "timestamp": datetime.now().isoformat()
            }
            
            # Calculate overall success
            success_rate = sum(1 for r in results if r.get("success", False)) / len(results)
            test_results["success_rate"] = success_rate
            test_results["passed"] = success_rate >= 0.8  # 80% success threshold
            
            # Store test results
            self.test_results[instance_id] = test_results
            
            return test_results
            
        except Exception as e:
            logging.error(f"Error testing transformation: {e}")
            return {
                "status": "error", 
                "message": f"Test execution failed: {str(e)}",
                "success_rate": 0,
                "passed": False
            }
    
    async def _test_basic_functionality(self, test_agent) -> Dict[str, Any]:
        """Test basic agent functionality"""
        try:
            # Test conversation creation
            conv_id = await test_agent.create_conversation("Test Conversation")
            if not conv_id:
                return {"success": False, "message": "Failed to create conversation"}
                
            # Test simple query
            response = await test_agent.qa(conv_id, "Hello, are you working?")
            if not response:
                return {"success": False, "message": "Failed to get response"}
                
            return {
                "success": True,
                "message": "Basic functionality tests passed",
                "details": {"response": response[:100] + "..." if len(response) > 100 else response}
            }
            
        except Exception as e:
            return {"success": False, "message": f"Basic functionality test failed: {str(e)}"}
    
    async def _test_error_handling(self, test_agent) -> Dict[str, Any]:
        """Test agent error handling"""
        try:
            # Test with invalid input
            try:
                await test_agent.qa(None, None)
                return {"success": False, "message": "Failed to handle invalid input properly"}
            except Exception:
                # Expected to fail gracefully
                pass
                
            return {"success": True, "message": "Error handling tests passed"}
            
        except Exception as e:
            return {"success": False, "message": f"Error handling test failed: {str(e)}"}
    
    async def _test_performance(self, test_agent) -> Dict[str, Any]:
        """Test agent performance"""
        try:
            start_time = time.time()
            await test_agent.create_conversation()
            end_time = time.time()
            
            # Check if performance is acceptable
            duration = end_time - start_time
            
            return {
                "success": duration < 2.0,  # Arbitrary threshold
                "message": f"Performance test completed in {duration:.2f}s",
                "details": {"duration": duration}
            }
            
        except Exception as e:
            return {"success": False, "message": f"Performance test failed: {str(e)}"}
    
    async def cleanup_test_instance(self, instance_id: str) -> bool:
        """Clean up a test instance"""
        if instance_id in self.test_instances:
            try:
                # Remove references to allow garbage collection
                del self.test_instances[instance_id]
                if instance_id in self.test_results:
                    del self.test_results[instance_id]
                return True
            except Exception as e:
                logging.error(f"Error cleaning up test instance: {e}")
        return False
        
    async def apply_transformation(self, transformation: Dict[str, Any], autonomous: bool = False) -> bool:
        """Safely apply a code transformation with testing"""
        if self.safety_checks_enabled:
            # Backup current state
            self.code_manager.backup()
            
        try:
            # Create and test a temporary instance with the changes
            instance_id = await self.create_test_instance(transformation)
            if not instance_id:
                raise ValueError("Failed to create test instance")
                
            # Run tests on the instance
            test_results = await self.test_transformation(instance_id)
            if not test_results.get("passed", False):
                raise ValueError(f"Tests failed: {test_results.get('message', 'Unknown error')}")
                
            # Tests passed, apply changes to the actual code
            changes = self._parse_code_changes(transformation["suggestion"])
            for change in changes:
                await self.code_manager.apply_modification(
                    operation=change["operation"],
                    params=change["params"]
                )
            
            # Verify the changes compile
            if not self.code_manager.compile_code():
                raise ValueError("Transformed code failed to compile")
                
            # Sync changes with the database
            await self._sync_code_with_database(transformation)
                
            # Record successful transformation
            self.transformation_history.append({
                "transformation": transformation,
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "autonomous": autonomous,
                "test_results": test_results
            })
            
            # Track code evolution
            self.evolution_tracker.add_transformation(
                original=transformation["original"],
                modified={"content": self.code_manager.lines},
                metadata={
                    "type": "improvement",
                    "suggestion": transformation["suggestion"],
                    "timestamp": datetime.now().isoformat(),
                    "autonomous": autonomous,
                    "test_results": test_results
                }
            )
            
            # Generate visualization
            self.evolution_tracker.visualize()
            
            if autonomous:
                self.changes_counter += 1
                if self.changes_counter >= self.max_autonomous_changes:
                    self.autonomous_mode = False
                    logging.info("Reached maximum autonomous changes, disabling autonomous mode")
            
            # Trigger learning from successful transformation
            await self._learn_from_transformation(transformation)
            
            # Clean up test instance
            await self.cleanup_test_instance(instance_id)
            
            # Signal for agent restart if needed
            if not autonomous:
                await self._signal_restart()
            
            return True
            
        except Exception as e:
            logging.error(f"Transformation failed: {e}")
            if self.safety_checks_enabled:
                self.code_manager.restore_backup()
            return False
    
    async def _sync_code_with_database(self, transformation: Dict[str, Any], in_memory_only: bool = False):
        """
        Synchronize code changes with the database with advanced features:
        - Transaction-based atomic updates
        - Differential updates to minimize database writes
        - Conflict detection and resolution
        - Version tracking
        - Timestamp-based change history
        - Distributed notification via Redis
        
        Args:
            transformation: The transformation to sync
            in_memory_only: Whether the code only exists in memory and hasn't been written to file
        """
        async with self.sync_lock:
            try:
                # Start transaction
                tx_id = str(uuid.uuid4())
                logging.info(f"Starting database sync transaction {tx_id}")
                
                # Get the agent's knowledge base
                kb = self.agent.knowledge_base
                
                # Update the code collection with the new code
                if self.agent.code_collection_id:
                    # Get the original code chunk
                    original = transformation["original"]
                    timestamp = datetime.now().isoformat()
                    
                    # Find the entry in the database
                    cur = kb.conn.cursor()
                    cur.execute(
                        "SELECT id, content, last_updated FROM documents WHERE collection_id = ? AND title LIKE ?", 
                        (self.agent.code_collection_id, f"%{original.get('name', '')}%")
                    )
                    row = cur.fetchone()
                    
                    # Begin transaction
                    kb.conn.execute("BEGIN TRANSACTION")
                    
                    try:
                        if row:
                            doc_id = row[0]
                            existing_content = row[1] if row[1] else ""
                            last_updated = row[2] if len(row) > 2 else None
                            
                            # Get the updated code based on whether it's in-memory only
                            if in_memory_only:
                                file_path = original.get('file', self.code_manager.file_path)
                                element_name = original.get('name', '')
                                element_type = original.get('type', 'function')
                                
                                # Find the element in the memory code
                                code_element = self.code_manager.get_code_element(element_type, element_name)
                                if code_element:
                                    content = ast.unparse(code_element['ast_node'])
                                else:
                                    # Fall back to the memory code for the whole file
                                    content = self.code_manager.get_memory_code(file_path)
                            else:
                                # Get from actual file
                                updated_code = self.code_manager.get_code_element(
                                    original.get("type", "function"),
                                    original.get("name", "")
                                )
                                content = updated_code.get("content", "") if updated_code else ""
                            
                            # Check for conflicts - if last_updated timestamp is more recent than our transformation timestamp
                            if last_updated and transformation.get("timestamp") and last_updated > transformation.get("timestamp"):
                                logging.warning(f"Potential conflict detected for {original.get('name', '')}: DB updated at {last_updated}")
                                
                                # Use difflib to compute differences and attempt automatic merge
                                import difflib
                                differ = difflib.Differ()
                                diff = list(differ.compare(existing_content.splitlines(), content.splitlines()))
                                
                                # Check if changes can be safely merged (no overlapping changes)
                                conflicts = [line for line in diff if line.startswith('? ')]
                                if not conflicts:
                                    logging.info(f"Auto-merging changes for {original.get('name', '')}")
                                    # Proceed with merge - content already contains our new version
                                else:
                                    # Store conflict information for later resolution
                                    conflict_id = str(uuid.uuid4())
                                    cur.execute("""
                                        INSERT INTO code_conflicts (id, document_id, base_content, our_content, their_content, created_at)
                                        VALUES (?, ?, ?, ?, ?, ?)
                                    """, (conflict_id, doc_id, original.get("content", ""), content, existing_content, timestamp))
                                    logging.warning(f"Conflict stored with ID {conflict_id} for manual resolution")
                                    
                                    # Use their version for now to avoid data loss
                                    content = existing_content
                            
                            if content:
                                # Generate a diff to store only changes instead of full content when possible
                                if existing_content:
                                    import difflib
                                    diff = ''.join(difflib.unified_diff(
                                        existing_content.splitlines(True),
                                        content.splitlines(True),
                                        fromfile='previous',
                                        tofile='current',
                                        n=3
                                    ))
                                    
                                    # If diff is significantly smaller than full content, store as diff
                                    if len(diff) < len(content) * 0.5:  # Only use diff if it's < 50% of content size
                                        # Store the diff instead of full content
                                        cur.execute("""
                                            INSERT INTO document_diffs (document_id, diff, applied_at)
                                            VALUES (?, ?, ?)
                                        """, (doc_id, diff, timestamp))
                                        logging.info(f"Stored diff for {original.get('name', '')} (saved {len(content) - len(diff)} bytes)")
                                
                                # Always update the full document for direct access
                                cur.execute("""
                                    UPDATE documents SET 
                                        content = ?,
                                        last_updated = ?,
                                        version = COALESCE(version, 0) + 1
                                    WHERE id = ?
                                """, (content, timestamp, doc_id))
                                
                                # Update the embedding
                                emb = await compute_embedding(content, db_conn=kb.conn)
                                if emb:
                                    # Store in a format that can be retrieved later
                                    content_hash = hashlib.sha256(content.encode()).hexdigest()
                                    blob = np.array(emb, dtype=np.float32).tobytes()
                                    cur.execute(
                                        "INSERT OR REPLACE INTO embeddings_cache (content_hash, embedding) VALUES (?, ?)",
                                        (content_hash, blob)
                                    )
                                
                                # Create history record
                                cur.execute("""
                                    INSERT INTO document_history (document_id, changed_at, change_type, change_metadata)
                                    VALUES (?, ?, ?, ?)
                                """, (doc_id, timestamp, "update", json.dumps({
                                    "transformation_id": transformation.get("id", "unknown"),
                                    "in_memory_only": in_memory_only,
                                    "tx_id": tx_id,
                                    "change_size": len(content)
                                })))
                                
                                # Commit the transaction
                                kb.conn.commit()
                                
                                # Notify via Redis if available
                                if hasattr(self.agent, 'redis_client') and self.agent.redis_client:
                                    try:
                                        await self.agent.redis_client.publish(
                                            "code_updates", 
                                            json.dumps({
                                                "event": "code_updated",
                                                "document_id": doc_id,
                                                "element_name": original.get('name', ''),
                                                "timestamp": timestamp,
                                                "tx_id": tx_id,
                                                "agent_id": self.agent.id
                                            })
                                        )
                                    except Exception as redis_err:
                                        logging.error(f"Redis notification error: {redis_err}")
                                
                                logging.info(f"Updated {'in-memory' if in_memory_only else ''} code in database: {original.get('name', '')}")
                        else:
                            # Document doesn't exist - create new entry
                            new_doc_id = str(uuid.uuid4())
                            file_path = original.get('file', self.code_manager.file_path)
                            element_name = original.get('name', '')
                            element_type = original.get('type', 'function')
                            
                            # Get content from memory or file
                            if in_memory_only:
                                code_element = self.code_manager.get_code_element(element_type, element_name)
                                if code_element:
                                    content = ast.unparse(code_element['ast_node'])
                                else:
                                    content = self.code_manager.get_memory_code(file_path)
                            else:
                                updated_code = self.code_manager.get_code_element(element_type, element_name)
                                content = updated_code.get("content", "") if updated_code else ""
                            
                            if content:
                                title = f"{element_type} {element_name} (File: {os.path.basename(file_path)})"
                                cur.execute("""
                                    INSERT INTO documents (id, collection_id, title, content, created_at, last_updated, version)
                                    VALUES (?, ?, ?, ?, ?, ?, 1)
                                """, (new_doc_id, self.agent.code_collection_id, title, content, timestamp, timestamp))
                                
                                # Update the embedding
                                emb = await compute_embedding(content, db_conn=kb.conn)
                                if emb:
                                    content_hash = hashlib.sha256(content.encode()).hexdigest()
                                    blob = np.array(emb, dtype=np.float32).tobytes()
                                    cur.execute(
                                        "INSERT INTO embeddings_cache (content_hash, embedding) VALUES (?, ?)",
                                        (content_hash, blob)
                                    )
                                
                                # Create history record for new document
                                cur.execute("""
                                    INSERT INTO document_history (document_id, changed_at, change_type, change_metadata)
                                    VALUES (?, ?, ?, ?)
                                """, (new_doc_id, timestamp, "create", json.dumps({
                                    "transformation_id": transformation.get("id", "unknown"),
                                    "in_memory_only": in_memory_only,
                                    "tx_id": tx_id
                                })))
                                
                                # Commit transaction
                                kb.conn.commit()
                                
                                logging.info(f"Created new document for {element_name} in database")
                    except Exception as tx_error:
                        # Rollback transaction on error
                        kb.conn.rollback()
                        logging.error(f"Transaction {tx_id} rolled back: {tx_error}")
                        raise tx_error
                
            except Exception as e:
                logging.error(f"Error syncing code with database: {e}")
                raise
                
    async def sync_memory_code_with_database(self, file_path: str):
        """
        Sync in-memory code changes for a specific file with the database.
        Uses optimized multi-phase commit protocol to ensure consistency across distributed systems.
        
        Features:
        - Atomic transactions with rollback capability
        - Distributed lock acquisition to prevent write conflicts
        - Change notification via Redis pub/sub
        - Differential storage for efficient space usage
        - Automatic conflict detection and resolution
        - Full history tracking with metadata
        
        Args:
            file_path: Path to the file whose memory code should be synced
            
        Returns:
            bool: Success status of the sync operation
        """
        sync_id = str(uuid.uuid4())
        agent_id = getattr(self.agent, 'id', 'unknown')
        timestamp = datetime.now().isoformat()
        
        # Create sync operation record
        try:
            kb = self.agent.knowledge_base
            cur = kb.conn.cursor()
            
            # Register sync operation
            cur.execute("""
                INSERT INTO sync_operations (id, started_at, status, document_ids, agent_id, metadata)
                VALUES (?, ?, 'preparing', ?, ?, ?)
            """, (
                sync_id,
                timestamp,
                json.dumps([file_path]),
                agent_id,
                json.dumps({"source": "memory_sync", "file_path": file_path})
            ))
            kb.conn.commit()
            
            # Try to acquire a distributed lock
            lock_acquired = await self._acquire_sync_lock(file_path)
            if not lock_acquired:
                logging.warning(f"Could not acquire lock for {file_path}, sync deferred")
                
                # Update sync operation status
                cur.execute("""
                    UPDATE sync_operations SET 
                        status = 'deferred',
                        error_message = 'Could not acquire lock'
                    WHERE id = ?
                """, (sync_id,))
                kb.conn.commit()
                
                return False
                
            try:
                # Prepare a transformation-like structure for syncing
                content = self.code_manager.get_memory_code(file_path)
                if not content:
                    self._release_sync_lock(file_path)
                    return False
                    
                # Create structured transformation for the sync function
                transformation = {
                    "id": sync_id,
                    "timestamp": timestamp,
                    "original": {
                        "file": file_path,
                        "name": os.path.basename(file_path),
                        "type": "file"
                    }
                }
                
                # Update sync operation status
                cur.execute("""
                    UPDATE sync_operations SET 
                        status = 'syncing'
                    WHERE id = ?
                """, (sync_id,))
                kb.conn.commit()
                
                # Sync with database using the advanced sync function
                result = await self._sync_code_with_database(transformation, in_memory_only=True)
                
                # Broadcast sync completion event if Redis is available
                if hasattr(self.agent, 'redis_client') and self.agent.redis_client:
                    try:
                        await self.agent.redis_client.publish("sync_events", json.dumps({
                            "event_type": "sync_completed",
                            "sync_id": sync_id,
                            "file_path": file_path,
                            "agent_id": agent_id,
                            "timestamp": datetime.now().isoformat(),
                            "status": "success"
                        }))
                    except Exception as redis_err:
                        logging.error(f"Redis notification error: {redis_err}")
                
                # Update sync operation as completed
                cur.execute("""
                    UPDATE sync_operations SET 
                        status = 'completed',
                        completed_at = ?
                    WHERE id = ?
                """, (datetime.now().isoformat(), sync_id))
                kb.conn.commit()
                
                logging.info(f"Successfully synced memory code for {file_path} to database (operation: {sync_id})")
                return True
                
            finally:
                # Always release the lock
                await self._release_sync_lock(file_path)
                
        except Exception as e:
            logging.error(f"Error syncing memory code with database: {e}")
            try:
                # Update sync operation as failed
                cur = kb.conn.cursor()
                cur.execute("""
                    UPDATE sync_operations SET 
                        status = 'failed',
                        completed_at = ?,
                        error_message = ?
                    WHERE id = ?
                """, (datetime.now().isoformat(), str(e), sync_id))
                kb.conn.commit()
            except Exception as db_err:
                logging.error(f"Error updating sync status: {db_err}")
            
            # Try to release lock even on failure
            try:
                await self._release_sync_lock(file_path)
            except Exception:
                pass
                
            return False
            
    async def _acquire_sync_lock(self, resource_id: str, timeout_ms: int = 5000) -> bool:
        """
        Acquire a distributed lock for synchronization.
        
        Args:
            resource_id: Identifier for the resource to lock
            timeout_ms: Maximum time to wait for lock acquisition in milliseconds
            
        Returns:
            bool: Whether the lock was successfully acquired
        """
        kb = self.agent.knowledge_base
        lock_id = hashlib.md5(f"{resource_id}:lock".encode()).hexdigest()
        agent_id = getattr(self.agent, 'id', 'unknown')
        cur = kb.conn.cursor()
        
        # Try with Redis first if available (for distributed locking)
        if hasattr(self.agent, 'redis_client') and self.agent.redis_client:
            try:
                expiry = int(timeout_ms * 1.5)  # Set expiry slightly longer than timeout
                lock_result = await self.agent.redis_client.set(
                    f"lock:{lock_id}", 
                    agent_id,
                    nx=True,  # Only set if key doesn't exist
                    px=expiry  # Expiry in milliseconds
                )
                if lock_result:
                    # Also record in local DB for tracking
                    expires_at = (datetime.now() + timedelta(milliseconds=expiry)).isoformat()
                    cur.execute("""
                        INSERT OR REPLACE INTO db_locks 
                        (resource_id, lock_holder, acquired_at, expires_at, metadata)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        lock_id, 
                        agent_id, 
                        datetime.now().isoformat(), 
                        expires_at,
                        json.dumps({"distributed": True, "via": "redis"})
                    ))
                    kb.conn.commit()
                    return True
            except Exception as redis_err:
                logging.warning(f"Redis lock error (falling back to DB lock): {redis_err}")
        
        # Fall back to database locking
        try:
            # First clean up any expired locks
            cur.execute("""
                DELETE FROM db_locks 
                WHERE expires_at IS NOT NULL AND expires_at < ?
            """, (datetime.now().isoformat(),))
            
            # Try to acquire the lock
            expires_at = (datetime.now() + timedelta(milliseconds=timeout_ms)).isoformat()
            cur.execute("""
                INSERT OR IGNORE INTO db_locks 
                (resource_id, lock_holder, acquired_at, expires_at, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                lock_id, 
                agent_id, 
                datetime.now().isoformat(), 
                expires_at,
                json.dumps({"distributed": False, "via": "sqlite"})
            ))
            kb.conn.commit()
            
            # Check if we got the lock (by checking if our row was inserted)
            cur.execute("SELECT lock_holder FROM db_locks WHERE resource_id = ?", (lock_id,))
            row = cur.fetchone()
            return row is not None and row[0] == agent_id
            
        except Exception as db_err:
            logging.error(f"Database lock error: {db_err}")
            return False
            
    async def _release_sync_lock(self, resource_id: str) -> bool:
        """
        Release a previously acquired distributed lock.
        
        Args:
            resource_id: Identifier for the resource to unlock
            
        Returns:
            bool: Whether the lock was successfully released
        """
        kb = self.agent.knowledge_base
        lock_id = hashlib.md5(f"{resource_id}:lock".encode()).hexdigest()
        agent_id = getattr(self.agent, 'id', 'unknown')
        
        try:
            # Release Redis lock if available
            if hasattr(self.agent, 'redis_client') and self.agent.redis_client:
                try:
                    # Only delete if we're the owner (using Lua script for atomicity)
                    release_script = """
                    if redis.call('get', KEYS[1]) == ARGV[1] then
                        return redis.call('del', KEYS[1])
                    else
                        return 0
                    end
                    """
                    await self.agent.redis_client.eval(
                        release_script, 
                        keys=[f"lock:{lock_id}"], 
                        args=[agent_id]
                    )
                except Exception as redis_err:
                    logging.warning(f"Redis unlock error: {redis_err}")
            
            # Release database lock
            cur = kb.conn.cursor()
            cur.execute("""
                DELETE FROM db_locks 
                WHERE resource_id = ? AND lock_holder = ?
            """, (lock_id, agent_id))
            kb.conn.commit()
            
            return True
            
        except Exception as e:
            logging.error(f"Error releasing lock: {e}")
            return False
    
    async def _signal_restart(self):
        """Signal that the agent should restart to apply changes"""
        # This could be implemented in various ways:
        # 1. Write to a file that's checked on startup
        # 2. Send a message through Redis pub/sub
        # 3. Call a restart endpoint on the agent's API
        
        restart_file = "agent_restart_signal.txt"
        with open(restart_file, "w") as f:
            f.write(f"Restart requested at {datetime.now().isoformat()}")
            
        logging.info("Restart signal sent - agent will restart on next cycle")
        
        # If Redis is available, also publish a restart message
        if self.agent.redis_client:
            try:
                await self.agent.redis_client.publish(
                    "agent_control",
                    json.dumps({
                        "action": "restart",
                        "agent_id": self.agent.id,
                        "timestamp": datetime.now().isoformat()
                    })
                )
            except Exception as e:
                logging.error(f"Error publishing restart signal: {e}")
            
    async def _learn_from_transformation(self, transformation: Dict[str, Any]):
        """Learn from successful transformations to improve future suggestions"""
        try:
            # Analyze the successful change
            analysis_prompt = f"""
            Analyze this successful code transformation:
            Original: {transformation['original']['content']}
            Suggestion: {transformation['suggestion']}
            
            Extract patterns and principles that led to success.
            """
            
            messages = [
                {"role": "system", "content": "You are an AI learning to improve code transformation strategies."},
                {"role": "user", "content": analysis_prompt}
            ]
            
            response = await client.chat.completions.create(
                model="o3-mini",
                messages=messages,
                temperature=0.2
            )
            
            # Store the learning outcome
            learning = {
                "transformation": transformation,
                "analysis": response.choices[0].message.content,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update learning rate based on success patterns
            self.learning_rate *= 1.1  # Increase learning rate on success
            
            logging.info(f"Learned from transformation: {learning['analysis'][:100]}...")
            
        except Exception as e:
            logging.error(f"Error in learning from transformation: {e}")
            self.learning_rate *= 0.9  # Decrease learning rate on failure

    def _parse_code_changes(self, suggestion: str) -> List[Dict[str, Any]]:
        """Parse transformation suggestion into concrete changes"""
        # Basic parsing for now - could be enhanced with more sophisticated parsing
        changes = []
        lines = suggestion.split("\n")
        current_change = None
        
        for line in lines:
            if line.startswith("CHANGE:"):
                if current_change:
                    changes.append(current_change)
                current_change = {"operation": "line", "params": {}}
            elif line.startswith("BLOCK:"):
                if current_change:
                    changes.append(current_change)
                current_change = {"operation": "block", "params": {}}
            elif current_change and line.strip():
                # Add parameters to current change based on content
                if "line_number" not in current_change["params"]:
                    try:
                        current_change["params"]["line_number"] = int(line)
                    except ValueError:
                        current_change["params"]["content"] = line
                else:
                    current_change["params"]["content"] = line
        
        if current_change:
            changes.append(current_change)
            
        return changes

class Task(BaseModel):
    """
    Represents a task that can be created, tracked, and executed by the agent.
    """
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed
    priority: int = 5  # 1-10, lower is higher priority
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)  # List of task_ids this task depends on
    tags: List[str] = Field(default_factory=list)
    result: Optional[Any] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def update_status(self, status: str):
        """Update task status and timestamp"""
        self.status = status
        self.updated_at = datetime.now().isoformat()
        if status == "completed":
            self.completed_at = datetime.now().isoformat()

    def add_tag(self, tag: str):
        """Add a tag to the task"""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now().isoformat()

class TaskManager:
    """
    Manages tasks for the agent, including creation, tracking, and execution.
    """
    def __init__(self, db_conn: Optional[sqlite3.Connection] = None):
        self.tasks: Dict[str, Task] = {}
        self.db_conn = db_conn
        if db_conn:
            self._create_tables()

    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        cur = self.db_conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                priority INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                completed_at TEXT,
                dependencies TEXT,
                tags TEXT,
                result TEXT,
                metadata TEXT
            )
        """)
        self.db_conn.commit()

    def create_task(self, title: str, description: str, priority: int = 5, 
                   dependencies: List[str] = None, tags: List[str] = None) -> Task:
        """Create a new task"""
        task = Task(
            title=title,
            description=description,
            priority=priority,
            dependencies=dependencies or [],
            tags=tags or []
        )
        self.tasks[task.task_id] = task
        
        # Save to database if available
        if self.db_conn:
            self._save_task_to_db(task)
            
        return task

    def _save_task_to_db(self, task: Task):
        """Save task to database"""
        cur = self.db_conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO tasks 
            (task_id, title, description, status, priority, created_at, 
             updated_at, completed_at, dependencies, tags, result, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.task_id,
            task.title,
            task.description,
            task.status,
            task.priority,
            task.created_at,
            task.updated_at,
            task.completed_at,
            json.dumps(task.dependencies),
            json.dumps(task.tags),
            json.dumps(task.result) if task.result is not None else None,
            json.dumps(task.metadata)
        ))
        self.db_conn.commit()

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID"""
        return self.tasks.get(task_id)

    def update_task(self, task_id: str, **kwargs) -> Optional[Task]:
        """Update task properties"""
        task = self.get_task(task_id)
        if not task:
            return None
            
        # Update fields
        for key, value in kwargs.items():
            if hasattr(task, key):
                setattr(task, key, value)
                
        task.updated_at = datetime.now().isoformat()
        
        # Save to database if available
        if self.db_conn:
            self._save_task_to_db(task)
            
        return task

    def update_task_status(self, task_id: str, status: str) -> Optional[Task]:
        """Update task status"""
        task = self.get_task(task_id)
        if not task:
            return None
            
        task.update_status(status)
        
        # Save to database if available
        if self.db_conn:
            self._save_task_to_db(task)
            
        return task

    def update_task_result(self, task_id: str, result: Any) -> Optional[Task]:
        """Update task result"""
        task = self.get_task(task_id)
        if not task:
            return None
            
        task.result = result
        task.updated_at = datetime.now().isoformat()
        
        # Save to database if available
        if self.db_conn:
            self._save_task_to_db(task)
            
        return task

    def list_tasks(self, status: Optional[str] = None, tag: Optional[str] = None, 
                  priority_below: Optional[int] = None) -> List[Task]:
        """List tasks with optional filtering"""
        tasks = list(self.tasks.values())
        
        # Apply filters
        if status:
            tasks = [t for t in tasks if t.status == status]
        if tag:
            tasks = [t for t in tasks if tag in t.tags]
        if priority_below is not None:
            tasks = [t for t in tasks if t.priority <= priority_below]
            
        # Sort by priority (lower first)
        return sorted(tasks, key=lambda t: t.priority)

    def get_next_task(self) -> Optional[Task]:
        """Get the highest priority pending task with no unmet dependencies"""
        pending_tasks = self.list_tasks(status="pending")
        
        # Filter out tasks with unmet dependencies
        ready_tasks = []
        for task in pending_tasks:
            dependencies_met = True
            for dep_id in task.dependencies:
                dep_task = self.get_task(dep_id)
                if not dep_task or dep_task.status != "completed":
                    dependencies_met = False
                    break
            if dependencies_met:
                ready_tasks.append(task)
                
        if not ready_tasks:
            return None
            
        # Return highest priority task
        return min(ready_tasks, key=lambda t: t.priority)

    def delete_task(self, task_id: str) -> bool:
        """Delete a task"""
        if task_id in self.tasks:
            del self.tasks[task_id]
            
            # Delete from database if available
            if self.db_conn:
                cur = self.db_conn.cursor()
                cur.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
                self.db_conn.commit()
                
            return True
        return False

    def load_tasks_from_db(self):
        """Load tasks from database"""
        if not self.db_conn:
            return
            
        cur = self.db_conn.cursor()
        cur.execute("SELECT * FROM tasks")
        rows = cur.fetchall()
        
        for row in cur.fetchall():
            task = Task(
                task_id=row[0],
                title=row[1],
                description=row[2],
                status=row[3],
                priority=row[4],
                created_at=row[5],
                updated_at=row[6],
                completed_at=row[7],
                dependencies=json.loads(row[8]) if row[8] else [],
                tags=json.loads(row[9]) if row[9] else [],
                result=json.loads(row[10]) if row[10] else None,
                metadata=json.loads(row[11]) if row[11] else {}
            )
            self.tasks[task.task_id] = task

class Agent(BaseModel):
    """
    Advanced agent that can ingest data, reflect on its own code,
    and modify itself based on internal reflection and reinforcement learning.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    instructions: str = ""
    knowledge_base: SQLiteKnowledgeBase = Field(default_factory=SQLiteKnowledgeBase)
    redis_client: Optional[redis.Redis] = None
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    system_capabilities: Dict[str, Any] = Field(default_factory=dict)
    system_tools: Dict[str, Callable] = Field(default_factory=dict)
    self_transformation: SelfTransformation = None
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    task_manager: TaskManager = None

    root: Optional[Node] = None
    q_table: Dict[str, float] = Field(default_factory=dict)
    exploration_rate: float = 0.1
    learning_rate: float = 0.1
    discount_factor: float = 0.9

    # Collection for self-reflection code embeddings
    code_collection_id: Optional[str] = None

    async def analyze_thought_process(self, prompt: str) -> ChainOfThought:
        system_prompt = """
        Break down your reasoning into clear, structured sub-steps.
        1. Initial thought
        2. Step-by-step reasoning with confidence
        3. Final conclusion
        Use a formal chain-of-thought format.
        """
        messages = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        response = await client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=messages,
            response_format=ChainOfThought
        )
        return response.choices[0].message.parsed

    async def load_own_code_ast(self):
        """
        Parse the current file using ASTChunker and store each code chunk in a dedicated collection.
        """
        if not self.code_collection_id:
            self.code_collection_id = self.knowledge_base.create_collection(
                "agent_code_ast",
                "AST-based code chunks for self-reflection"
            )
        chunker = ASTChunker()
        try:
            file_path = __file__
            code_chunks = chunker.parse_file(file_path)
            for chunk in code_chunks:
                emb = await compute_embedding(chunk["content"], db_conn=self.knowledge_base.conn)
                title = f"{chunk['type']} {chunk['name']} (Lines {chunk['start_line']}-{chunk['end_line']})"
                await self.knowledge_base.add_knowledge_entry(
                    kb_id=self.code_collection_id,
                    title=title,
                    content=chunk["content"],
                    embedding=emb
                )
            logging.info(f"Loaded {len(code_chunks)} code chunks via AST into knowledge base.")
        except Exception as e:
            logging.error(f"Error in load_own_code_ast: {e}")

    def __init__(self, **data):
        super().__init__(**data)
        self._reflection_queue = asyncio.Queue()
        self._reflection_task = None
        self._reflection_running = False
        
        # Detect and setup system capabilities and tools
        self.system_capabilities = detect_system_capabilities()
        self.system_tools = setup_system_tools(self.system_capabilities)
        
        # Register system tools with the agent
        for name, func in self.system_tools.items():
            if func is not None:
                register_tool(
                    name=name,
                    func=func,
                    description=f"System tool: {name}",
                    parameters={"args": {"type": "object"}}
                )
        
        # Initialize self-transformation capability
        self.self_transformation = SelfTransformation(self)
        
        # Initialize task manager
        self.task_manager = TaskManager(self.knowledge_base.conn)
        
        # Register code reading and writing tools
        register_tool(
            name="read_code_file",
            func=self.read_code_file,
            description="Read a code file from the filesystem",
            parameters={
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            }
        )
        
        register_tool(
            name="write_code_file",
            func=self.write_code_file,
            description="Write code to a file on the filesystem",
            parameters={
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            }
        )
        
        register_tool(
            name="create_task",
            func=self.create_task,
            description="Create a new task for the agent to work on",
            parameters={
                "title": {
                    "type": "string",
                    "description": "Short title for the task"
                },
                "description": {
                    "type": "string",
                    "description": "Detailed description of the task"
                },
                "priority": {
                    "type": "integer",
                    "description": "Priority level (1-10, lower is higher priority)",
                    "default": 5
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of tags for categorizing the task",
                    "default": []
                }
            }
        )
        
        register_tool(
            name="list_tasks",
            func=self.list_tasks,
            description="List tasks with optional filtering",
            parameters={
                "status": {
                    "type": "string",
                    "description": "Filter by status (pending, in_progress, completed, failed)",
                    "default": None
                },
                "tag": {
                    "type": "string",
                    "description": "Filter by tag",
                    "default": None
                }
            }
        )
        
        register_tool(
            name="update_task",
            func=self.update_task,
            description="Update a task's status or add result",
            parameters={
                "task_id": {
                    "type": "string",
                    "description": "ID of the task to update"
                },
                "status": {
                    "type": "string",
                    "description": "New status for the task",
                    "default": None
                },
                "result": {
                    "type": "object",
                    "description": "Result data for the task",
                    "default": None
                }
            }
        )

    async def _reflection_worker(self):
        """Background worker that processes code reflections and triggers transformations"""
        self._reflection_running = True
        try:
            while self._reflection_running:
                try:
                    chunk = await self._reflection_queue.get()
                    if chunk is None:  # Sentinel value to stop
                        self._reflection_queue.task_done()
                        break
                    
                    # Generate reflection
                    prompt = f"Review the following {chunk['type']} '{chunk['name']}' and suggest improvements:\n{chunk['content']}"
                    messages = [
                        {"role": "system", "content": "You are a senior developer performing code review."},
                        {"role": "user", "content": prompt}
                    ]
                    
                    response = await client.chat.completions.create(
                        model="o3-mini",
                        messages=messages,
                        temperature=0.2
                    )
                    reflection = response.choices[0].message.content
                    timestamp = datetime.now().isoformat()
                    title = f"Reflection on {chunk['type']} {chunk['name']} at {timestamp}"
                    
                    # Store reflection
                    emb = await compute_embedding(reflection, db_conn=self.knowledge_base.conn)
                    await self.knowledge_base.add_knowledge_entry(
                        kb_id=self.code_collection_id,
                        title=title,
                        content=reflection,
                        embedding=emb
                    )
                    
                    # Generate and apply transformation if appropriate
                    if "TRANSFORMATION SUGGESTED" in reflection:
                        transformation = await self.self_transformation.propose_transformation(chunk)
                        if transformation:
                            success = await self.self_transformation.apply_transformation(transformation)
                            if success:
                                logging.info(f"Successfully applied transformation to {chunk['name']}")
                            else:
                                logging.warning(f"Failed to apply transformation to {chunk['name']}")
                    
                    logging.info(f"Completed reflection on {chunk['name']}")
                    self._reflection_queue.task_done()
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logging.error(f"Error in reflection worker: {e}")
                    if self._reflection_queue.qsize() > 0:
                        self._reflection_queue.task_done()
        finally:
            self._reflection_running = False

    async def start_reflections(self):
        """Start the background reflection process"""
        if not self.code_collection_id:
            await self.load_own_code_ast()
        
        if self._reflection_task is None:
            self._reflection_task = asyncio.create_task(self._reflection_worker())
            logging.info("Started reflection worker")

    async def stop_reflections(self):
        """Stop the background reflection process"""
        if self._reflection_running:
            self._reflection_running = False
            try:
                if self._reflection_task:
                    self._reflection_task.cancel()
                    await self._reflection_task
            except asyncio.CancelledError:
                pass
            finally:
                self._reflection_task = None
                logging.info("Stopped reflection worker")

    async def reflect_on_current_code(self):
        """Queue code chunks for reflection in the background"""
        if not self.code_collection_id:
            await self.load_own_code_ast()
        
        chunker = ASTChunker()
        try:
            file_path = __file__
            code_chunks = chunker.parse_file(file_path)
            
            # Start reflection worker if not running
            await self.start_reflections()
            
            # Queue all chunks for processing
            for chunk in code_chunks:
                await self._reflection_queue.put(chunk)
            
            logging.info(f"Queued {len(code_chunks)} chunks for reflection")
            return f"Queued {len(code_chunks)} code chunks for background reflection"
            
        except Exception as e:
            logging.error(f"Error queueing reflections: {e}")
            return f"Error: {e}"

    async def retrieve_relevant_code(self, query: str, top_n: int = 3) -> str:
        """
        Retrieve relevant code chunks from the agent's AST-based collection.
        
        Args:
            query: The search query text
            top_n: Maximum number of results to return (default: 3)
            
        Returns:
            String containing matching code chunks
        """
        if not self.code_collection_id:
            await self.load_own_code_ast()
        try:
            docs = await self.knowledge_base.search_knowledge(query, kb_id=self.code_collection_id, top_n=top_n)
            if not docs:
                return "No relevant code found."
            snippet = "\n\n".join([f"Title: {doc.get('title')}\n{doc.get('content')}" for doc in docs])
            return snippet
        except Exception as e:
            logging.error(f"Error retrieving relevant code: {e}")
            return f"Error: {e}"

    async def create_conversation(self, title: str = None, max_turns: int = None) -> str:
        conv_id = str(uuid.uuid4())
        logging.info(f"Created conversation {conv_id} titled '{title}'")
        return conv_id

    async def add_to_conversation(self, conv_id: str, role: str, content: str, name: str = None) -> str:
        msg_id = str(uuid.uuid4())
        logging.info(f"Added {role} message {msg_id} to conversation {conv_id}")
        return msg_id

    async def ingest_source(self, source: str):
        """
        Ingest a file, URL, or directory.
        """
        if os.path.isdir(source):
            logging.info(f"Source '{source}' is a directory.")
            # Implement directory ingestion (omitted for brevity)
        elif os.path.isfile(source):
            logging.info(f"Source '{source}' is a file.")
            # Implement file ingestion based on extension
        elif source.startswith("http"):
            logging.info(f"Source '{source}' is a URL.")
            # Implement URL ingestion
        else:
            logging.error(f"Unrecognized source type: {source}")

    async def query(self, query_text: str, initial_top_n: int = 15, final_top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents from the knowledge base (stub implementation).
        """
        logging.info(f"Querying knowledge base with: {query_text}")
        return []

    async def _stream_chat_response(self, messages: List[Dict[str, str]], conv_id: str) -> str:
        response = await client.chat.completions.create(
            model="o3-mini",
            reasoning_effort="high",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=True
        )
        accumulated = ""
        current_tool_call = None
        tool_calls_buffer = []
        
        print("\n", end="", flush=True)
        async for chunk in response:
            delta = chunk.choices[0].delta
            
            # Handle regular content streaming
            if hasattr(delta, 'content') and delta.content is not None:
                c = delta.content
                accumulated += c
                print(c, end="", flush=True)
                
            # Handle tool calls streaming
            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                for tool_call in delta.tool_calls:
                    if not current_tool_call:
                        # Start a new tool call
                        if hasattr(tool_call, 'id') and tool_call.id:
                            current_tool_call = type('ToolCall', (), {
                                'id': tool_call.id,
                                'function': type('Function', (), {'name': '', 'arguments': ''}),
                                'type': 'function'
                            })
                            print(f"\n🔧 Using tool: {tool_call.function.name if hasattr(tool_call.function, 'name') else '...'}", flush=True)
                    
                    # Update the current tool call
                    if current_tool_call and hasattr(tool_call, 'function'):
                        if hasattr(tool_call.function, 'name') and tool_call.function.name:
                            current_tool_call.function.name = getattr(current_tool_call.function, 'name', '') + tool_call.function.name
                        if hasattr(tool_call.function, 'arguments') and tool_call.function.arguments:
                            current_tool_call.function.arguments = getattr(current_tool_call.function, 'arguments', '') + tool_call.function.arguments
        
        # Process tool calls if any
        if current_tool_call and hasattr(current_tool_call, 'function'):
            tool_calls_buffer.append({
                "id": str(current_tool_call.id),
                "type": "function",
                "function": {
                    "name": current_tool_call.function.name,
                    "arguments": current_tool_call.function.arguments
                }
            })
            
            # Execute the tool call
            try:
                function_name = current_tool_call.function.name
                function_args = json.loads(current_tool_call.function.arguments)
                
                print(f"\n📊 Executing tool: {function_name} with args: {function_args}", flush=True)
                
                if function_name in available_functions:
                    func = available_functions[function_name]
                    if asyncio.iscoroutinefunction(func):
                        result = await func(**function_args)
                    else:
                        result = func(**function_args)
                    
                    result_str = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
                    print(f"\n📊 Tool result: {result_str}", flush=True)
                    
                    # Add tool result to messages and get final response
                    messages.append({
                        "role": "assistant",
                        "tool_calls": tool_calls_buffer
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": current_tool_call.id,
                        "content": result_str
                    })
                    
                    # Get final response with tool results
                    final_response = await client.chat.completions.create(
                        model="o3-mini",
                        messages=messages,
                        stream=True
                    )
                    
                    print("\n🤖 Final response:", end=" ", flush=True)
                    final_accumulated = ""
                    async for chunk in final_response:
                        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                            c = chunk.choices[0].delta.content
                            final_accumulated += c
                            print(c, end="", flush=True)
                    
                    accumulated += "\n" + final_accumulated
            except Exception as e:
                print(f"\n❌ Error executing tool: {e}", flush=True)
        
        print()
        return accumulated

    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a system tool with given arguments and return structured response"""
        if tool_name in self.system_tools:
            # Check rate limiting
            tool_spec = next((t for t in tools if t["function"]["name"] == tool_name), None)
            if tool_spec and "metadata" in tool_spec:
                rate_limit = tool_spec["metadata"].get("rate_limit")
                last_called = tool_spec["metadata"].get("last_called")
                
                if rate_limit and last_called:
                    last_time = datetime.fromisoformat(last_called)
                    elapsed = (datetime.now() - last_time).total_seconds()
                    if elapsed < rate_limit:
                        return {
                            "status": "error",
                            "error": f"Rate limit exceeded. Try again in {rate_limit - elapsed:.1f}s",
                            "result": None
                        }
            
            try:
                # Check if function is async
                func = self.system_tools[tool_name]
                if asyncio.iscoroutinefunction(func):
                    result = await func(**kwargs)
                else:
                    result = func(**kwargs)
                
                # Update last called time
                if tool_spec and "metadata" in tool_spec:
                    tool_spec["metadata"]["last_called"] = datetime.now().isoformat()
                
                # Check if result contains a tool call (recursive tool calling)
                if isinstance(result, dict) and result.get("tool_call"):
                    print(f"\n🔄 Recursive tool call detected: {result['tool_call']}", flush=True)
                    nested_tool = result["tool_call"]["name"]
                    nested_args = result["tool_call"]["arguments"]
                    
                    # Execute the nested tool
                    nested_result = await self.execute_tool(nested_tool, **nested_args)
                    
                    # Combine results
                    result["nested_result"] = nested_result
                
                return {
                    "status": "success",
                    "error": None,
                    "result": result,
                    "tool_name": tool_name,
                    "args": kwargs,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logging.error(f"Error executing tool {tool_name}: {e}")
                return {
                    "status": "error", 
                    "error": str(e),
                    "result": None,
                    "tool_name": tool_name,
                    "args": kwargs,
                    "timestamp": datetime.now().isoformat()
                }
        return {
            "status": "error",
            "error": f"Tool {tool_name} not found",
            "result": None
        }

    async def search_web(self, query: str) -> Optional[Dict[str, Any]]:
        """Perform a web search if search capability is available"""
        if "web_search" in self.system_tools and self.system_tools["web_search"]:
            try:
                return await self.system_tools["web_search"](query)
            except Exception as e:
                logging.error(f"Web search error: {e}")
        return None

    async def edit_file(self, file_path: str, content: str) -> bool:
        """Edit a file with given content"""
        try:
            await self.execute_tool("write_file", path=file_path, content=content)
            return True
        except Exception as e:
            logging.error(f"File edit error: {e}")
            return False

    async def read_code_file(self, file_path: str) -> Dict[str, Any]:
        """
        Read a code file from the filesystem
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            Dict containing file content and metadata
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "error": f"File not found: {file_path}",
                    "content": None
                }
                
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Get file metadata
            stat_info = os.stat(file_path)
            file_size = stat_info.st_size
            modified_time = datetime.fromtimestamp(stat_info.st_mtime).isoformat()
            
            # Detect file type based on extension
            _, ext = os.path.splitext(file_path)
            file_type = ext.lstrip('.').lower() if ext else 'unknown'
            
            # Store in knowledge base if it's code
            if file_type in ['py', 'js', 'java', 'c', 'cpp', 'go', 'rs', 'ts', 'rb', 'php']:
                # Create code collection if needed
                if not self.code_collection_id:
                    self.code_collection_id = self.knowledge_base.create_collection(
                        "code_files",
                        "External code files for analysis"
                    )
                
                # Compute embedding
                emb = await compute_embedding(content, db_conn=self.knowledge_base.conn)
                
                # Add to knowledge base
                await self.knowledge_base.add_knowledge_entry(
                    kb_id=self.code_collection_id,
                    title=f"Code: {os.path.basename(file_path)}",
                    content=content,
                    embedding=emb
                )
            
            return {
                "success": True,
                "content": content,
                "file_path": file_path,
                "file_type": file_type,
                "file_size": file_size,
                "modified_time": modified_time,
                "lines": content.count('\n') + 1
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content": None
            }
    
    async def write_code_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Write code to a file on the filesystem
        
        Args:
            file_path: Path to the file to write
            content: Content to write to the file
            
        Returns:
            Dict containing success status and metadata
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Check if file exists
            file_existed = os.path.exists(file_path)
            
            # Create backup if file exists
            if file_existed:
                backup_path = f"{file_path}.bak"
                shutil.copy2(file_path, backup_path)
            
            # Write content to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            # Get file metadata
            stat_info = os.stat(file_path)
            file_size = stat_info.st_size
            modified_time = datetime.fromtimestamp(stat_info.st_mtime).isoformat()
            
            # Store in knowledge base if it's code
            _, ext = os.path.splitext(file_path)
            file_type = ext.lstrip('.').lower() if ext else 'unknown'
            
            if file_type in ['py', 'js', 'java', 'c', 'cpp', 'go', 'rs', 'ts', 'rb', 'php']:
                # Create code collection if needed
                if not self.code_collection_id:
                    self.code_collection_id = self.knowledge_base.create_collection(
                        "code_files",
                        "External code files for analysis"
                    )
                
                # Compute embedding
                emb = await compute_embedding(content, db_conn=self.knowledge_base.conn)
                
                # Add to knowledge base
                await self.knowledge_base.add_knowledge_entry(
                    kb_id=self.code_collection_id,
                    title=f"Code: {os.path.basename(file_path)}",
                    content=content,
                    embedding=emb
                )
            
            return {
                "success": True,
                "file_path": file_path,
                "file_existed": file_existed,
                "file_type": file_type,
                "file_size": file_size,
                "modified_time": modified_time,
                "lines": content.count('\n') + 1
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def create_task(self, title: str, description: str, priority: int = 5, tags: List[str] = None) -> Dict[str, Any]:
        """
        Create a new task for the agent to work on
        
        Args:
            title: Short title for the task
            description: Detailed description of the task
            priority: Priority level (1-10, lower is higher priority)
            tags: List of tags for categorizing the task
            
        Returns:
            Dict containing task information
        """
        try:
            task = self.task_manager.create_task(
                title=title,
                description=description,
                priority=priority,
                tags=tags or []
            )
            
            return {
                "success": True,
                "task_id": task.task_id,
                "title": task.title,
                "status": task.status,
                "priority": task.priority,
                "created_at": task.created_at
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def list_tasks(self, status: Optional[str] = None, tag: Optional[str] = None) -> Dict[str, Any]:
        """
        List tasks with optional filtering
        
        Args:
            status: Filter by status (pending, in_progress, completed, failed)
            tag: Filter by tag
            
        Returns:
            Dict containing list of tasks
        """
        try:
            tasks = self.task_manager.list_tasks(status=status, tag=tag)
            
            task_list = []
            for task in tasks:
                task_list.append({
                    "task_id": task.task_id,
                    "title": task.title,
                    "description": task.description[:100] + "..." if len(task.description) > 100 else task.description,
                    "status": task.status,
                    "priority": task.priority,
                    "created_at": task.created_at,
                    "tags": task.tags
                })
            
            return {
                "success": True,
                "count": len(task_list),
                "tasks": task_list
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tasks": []
            }
    
    async def update_task(self, task_id: str, status: Optional[str] = None, result: Optional[Any] = None) -> Dict[str, Any]:
        """
        Update a task's status or add result
        
        Args:
            task_id: ID of the task to update
            status: New status for the task
            result: Result data for the task
            
        Returns:
            Dict containing updated task information
        """
        try:
            task = self.task_manager.get_task(task_id)
            if not task:
                return {
                    "success": False,
                    "error": f"Task not found: {task_id}"
                }
            
            # Update status if provided
            if status:
                self.task_manager.update_task_status(task_id, status)
            
            # Update result if provided
            if result is not None:
                self.task_manager.update_task_result(task_id, result)
            
            # Get updated task
            updated_task = self.task_manager.get_task(task_id)
            
            return {
                "success": True,
                "task_id": updated_task.task_id,
                "title": updated_task.title,
                "status": updated_task.status,
                "updated_at": updated_task.updated_at,
                "completed_at": updated_task.completed_at
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def qa(self, conv_id: str, question: str) -> str:
        if not conv_id:
            conv_id = await self.create_conversation()
        await self.add_to_conversation(conv_id, "user", question)
        
        # Start reflections if not already running
        if not self._reflection_running:
            await self.start_reflections()

        # Check if this is a polymorphic transformation request
        if any(term in question.lower() for term in ["polymorphic", "transform code", "morph code", "code morphing"]):
            print("\n🧬 Polymorphic transformation request detected. Initiating code morphing...", flush=True)
            
            # Extract target from question
            import re
            target_match = re.search(r"transform\s+(\w+(?:\.\w+)*)", question.lower())
            target_file = None
            
            if target_match:
                target_component = target_match.group(1)
                # Try to find the file containing this component
                code_chunks = await self.retrieve_relevant_code(target_component)
                if code_chunks:
                    # Extract file path from the code chunks if possible
                    file_match = re.search(r"File: ([^\)]+)", code_chunks)
                    if file_match:
                        target_file = file_match.group(1).strip()
            
            # If no specific target, use the main file
            if not target_file:
                target_file = __file__
                
            try:
                # Import the polymorphic engine
                from polymorphic_engine import PolymorphicEngine, TransformationType
                
                # Create the engine
                engine = PolymorphicEngine()
                
                # Determine intensity from the question
                intensity = 0.5  # Default
                if "aggressive" in question.lower() or "high intensity" in question.lower():
                    intensity = 0.8
                elif "subtle" in question.lower() or "low intensity" in question.lower():
                    intensity = 0.3
                
                # Determine transformation types from the question
                transformation_types = []
                if "rename variables" in question.lower():
                    transformation_types.append(TransformationType.RENAME_VARIABLES)
                if "reorder" in question.lower():
                    transformation_types.append(TransformationType.REORDER_STATEMENTS)
                if "control flow" in question.lower():
                    transformation_types.append(TransformationType.CHANGE_CONTROL_FLOW)
                if "dead code" in question.lower():
                    transformation_types.append(TransformationType.ADD_DEAD_CODE)
                
                # If no specific types mentioned, use all
                if not transformation_types:
                    transformation_types = None
                
                print(f"\n🧬 Applying polymorphic transformations to {target_file} with intensity {intensity}...", flush=True)
                
                # Apply the transformation
                success, message = engine.transform_file(
                    target_file, 
                    transformation_types=transformation_types,
                    intensity=intensity
                )
                
                if success:
                    print(f"\n✅ Transformation successful: {message}", flush=True)
                    return (f"Successfully applied polymorphic transformations to {target_file}. "
                           f"{message}. The code structure has been modified while preserving functionality. "
                           f"You may need to restart the agent to see the effects.")
                else:
                    print(f"\n❌ Transformation failed: {message}", flush=True)
                    return f"I attempted to apply polymorphic transformations but encountered an error: {message}"
                    
            except ImportError:
                return "The polymorphic engine module is not available. Please ensure it's properly installed."
            except Exception as e:
                print(f"\n❌ Error during polymorphic transformation: {str(e)}", flush=True)
                return f"An error occurred during the polymorphic transformation: {str(e)}"

        # Check if this is a self-modification request
        if any(term in question.lower() for term in ["modify yourself", "improve your code", "self-modify", "update your code"]):
            print("\n🔄 Self-modification request detected. Initiating code transformation...", flush=True)
            # Retrieve relevant code based on the query
            code_chunks = await self.retrieve_relevant_code(question)
            
            # Parse into a proper code chunk format
            chunk = {
                "type": "function",
                "name": "self_modification_target",
                "content": code_chunks
            }
            
            # Propose transformation
            transformation = await self.self_transformation.propose_transformation(chunk)
            if transformation:
                print("\n📝 Proposed transformation:", flush=True)
                print(transformation["suggestion"], flush=True)
                
                # Create and test a temporary instance
                print("\n🧪 Creating test instance to validate changes...", flush=True)
                instance_id = await self.self_transformation.create_test_instance(transformation)
                
                if instance_id:
                    print(f"\n🧪 Running tests on instance {instance_id}...", flush=True)
                    test_results = await self.self_transformation.test_transformation(instance_id)
                    
                    if test_results.get("passed", False):
                        print("\n✅ Tests passed! Applying changes to main codebase...", flush=True)
                        
                        # Apply transformation
                        success = await self.self_transformation.apply_transformation(transformation)
                        
                        if success:
                            # Clean up test instance
                            await self.self_transformation.cleanup_test_instance(instance_id)
                            
                            return (f"Successfully applied self-modification to improve my code. "
                                   f"The changes have been tested and implemented. "
                                   f"Test results: {test_results.get('success_rate', 0)*100:.0f}% success rate. "
                                   f"I'll restart to apply these changes fully.")
                        else:
                            return f"I validated the changes in a test instance, but encountered errors when applying to my main codebase."
                    else:
                        print(f"\n❌ Tests failed: {test_results.get('message', 'Unknown error')}", flush=True)
                        await self.self_transformation.cleanup_test_instance(instance_id)
                        return (f"I attempted to modify my code but the changes failed validation tests. "
                               f"Test results: {test_results.get('success_rate', 0)*100:.0f}% success rate. "
                               f"The changes were not applied to ensure system stability.")
                else:
                    return f"I attempted to create a test instance for the code changes, but failed. The changes were not applied."
        
        # Check if this is a weather-related query
        if any(term in question.lower() for term in ["weather", "temperature", "forecast", "rain", "snow", "sunny"]):
            # Extract location from the question
            import re
            location_match = re.search(r"weather\s+in\s+([A-Za-z\s,]+)", question.lower())
            
            if location_match:
                location = location_match.group(1).strip()
                print(f"\n🌤️ Weather query detected for location: {location}", flush=True)
                
                try:
                    # Try to get weather data
                    if "get_weather" in self.system_tools:
                        weather_data = await self.system_tools["get_weather"](location)
                        
                        if weather_data and "error" not in weather_data:
                            # Extract weather information
                            if "main" in weather_data and "weather" in weather_data:
                                temp = weather_data["main"].get("temp", "N/A")
                                description = weather_data["weather"][0].get("description", "unknown") if weather_data["weather"] else "unknown"
                                
                                # Format the response
                                weather_response = f"The current weather in {location} is {description} with an approximate temperature of {temp}°F."
                                
                                # Add to conversation and return
                                await self.add_to_conversation(conv_id, "assistant", weather_response)
                                return weather_response
                    
                    # If we get here, either the tool failed or we don't have it
                    # Continue with normal processing
                    print("\n⚠️ Weather tool unavailable or failed, falling back to standard processing", flush=True)
                except Exception as e:
                    print(f"\n❌ Error getting weather data: {str(e)}", flush=True)
        
        # Process with chain-of-thought
        cot = await self.analyze_thought_process(question)
        code_context = await self.retrieve_relevant_code(question)
        
        # Try web search for factual queries
        search_context = "Relevant documents: (stub)"
        if any(term in question.lower() for term in ["what", "who", "where", "when", "how"]):
            if "web_search" in self.system_tools:
                search_result = await self.system_tools["web_search"](question)
                if search_result and "error" not in search_result:
                    # Extract relevant information from search results
                    search_context = "Search results:\n"
                    if "items" in search_result:
                        for item in search_result["items"][:3]:
                            search_context += f"- {item['title']}: {item['snippet']}\n"
                    elif "organic_results" in search_result:
                        for item in search_result["organic_results"][:3]:
                            search_context += f"- {item['title']}: {item['snippet']}\n"
        
        combined_context = f"{search_context}\n\nRelevant Code:\n{code_context}"
        
        prompt = f"""
Chain-of-Thought:
Initial Thought: {cot.initial_thought}
Steps: {" ".join(f"Step {s.step_number}: {s.reasoning} => {s.conclusion} (Confidence: {s.confidence})" for s in cot.steps)}
Final Conclusion: {cot.final_conclusion}

Context:
{combined_context}

You can use tools to help answer this question. Use tools recursively if needed.
Question: {question}
"""
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant with self-modification capabilities. Use the provided chain-of-thought and context to answer the question. You can use tools recursively to gather information."},
            {"role": "user", "content": prompt}
        ]
        
        answer = await self._stream_chat_response(messages, conv_id)
        await self.add_to_conversation(conv_id, "assistant", answer)
        self.conversation_history.append({
            "prompt": question,
            "chain_of_thought": cot,
            "code_context": code_context,
            "response": answer
        })
        
        # After answering, trigger a self-reflection cycle.
        await self.reflect_on_current_code()
        return answer

# ------------------------------------------------------------------------------
# Pydantic Models for FastAPI Endpoints
# ------------------------------------------------------------------------------

class ToolRegistrationInput(BaseModel):
    name: str
    code: str
    description: str
    parameters: Dict[str, Any]

class ToolRegistrationOutput(BaseModel):
    result: str

class ProgressModel(BaseModel):
    status: str
    iteration: int
    max_iterations: int
    output: str
    completed: bool

class ToolsInfo(BaseModel):
    tools: List[Any]
    available_functions: List[str]
    available_api_keys: List[str]

# ------------------------------------------------------------------------------
# Main LLM-driven Build Loop (Stub Implementation)
# ------------------------------------------------------------------------------

history_dict: Dict[str, Any] = {"iterations": []}
progress = {
    "status": "idle",
    "iteration": 0,
    "max_iterations": 50,
    "output": "",
    "completed": False
}

def log_to_file(hist: Dict[str,Any]):
    try:
        with open("fastapi_app_builder_log.json", "w") as f:
            json.dump(hist, f, indent=2)
    except Exception:
        pass

def run_main_loop(user_input: str) -> str:
    progress["status"] = "running"
    progress["iteration"] = 0
    output = f"Starting main loop with input: {user_input}\n"
    iteration = 0
    max_iter = progress["max_iterations"]
    while iteration < max_iter:
        iteration += 1
        progress["iteration"] = iteration
        output += f"Iteration {iteration}: Processing...\n"
        time.sleep(0.5)
        if iteration == 5:
            output += "Completed early.\n"
            progress["completed"] = True
            progress["status"] = "completed"
            progress["output"] = output
            log_to_file(history_dict)
            return output
        progress["output"] = output
        log_to_file(history_dict)
    progress["completed"] = True
    progress["status"] = "completed"
    progress["output"] = output
    return output

# ------------------------------------------------------------------------------
# FastAPI App Setup & Endpoints
# ------------------------------------------------------------------------------

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    index_file = os.path.join("templates", "index.html")
    if os.path.exists(index_file):
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        return HTMLResponse("<h1>FastAPI App Builder</h1><p>No index.html found.</p>")

@app.post("/", response_class=HTMLResponse)
async def run_task(request: Request, background_tasks: BackgroundTasks):
    form = await request.form()
    user_input = form.get("user_input", "")
    global progress
    progress = {"status": "running", "iteration": 0, "max_iterations": 50, "output": "", "completed": False}
    background_tasks.add_task(run_main_loop, user_input)
    html_content = """
    <html>
    <head><title>Task Progress</title></head>
    <body>
      <h1>Task in Progress</h1>
      <pre id="progress"></pre>
      <script>
        setInterval(function(){
          fetch('/progress')
          .then(res => res.json())
          .then(data => {
            document.getElementById('progress').textContent = data.output;
            if(data.completed){
                document.getElementById('refresh-btn').style.display = 'block';
            }
          })
        },2000)
      </script>
      <button id="refresh-btn" style="display:none" onclick="location.reload()">Refresh</button>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/progress", response_model=ProgressModel)
async def get_progress():
    return ProgressModel(**progress)

@app.post("/register_tool", response_model=ToolRegistrationOutput)
async def register_tool_endpoint(data: ToolRegistrationInput):
    try:
        exec(data.code, globals())
        result = register_tool(
            data.name,
            globals()[data.name],
            data.description,
            data.parameters
        )
        return ToolRegistrationOutput(result=result)
    except Exception as e:
        return ToolRegistrationOutput(result=f"Error creating/updating tool '{data.name}': {e}")

@app.get("/tools", response_model=ToolsInfo)
async def get_tools():
    patterns = ['API_KEY','ACCESS_TOKEN','SECRET_KEY','TOKEN','APISECRET']
    found_keys = [k for k in os.environ.keys() if any(p in k.upper() for p in patterns)]
    return ToolsInfo(
        tools=tools,
        available_functions=list(available_functions.keys()),
        available_api_keys=found_keys
    )

# ------------------------------------------------------------------------------
# CLI & File Watchdog for Interactive Use
# ------------------------------------------------------------------------------

def cli_main():
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    class PDFHandler(FileSystemEventHandler):
        def __init__(self, agent):
            self.agent = agent
            self.processed_paths = set()
        def compute_file_hash(self, file_path):
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        def should_process_file(self, file_path):
            if file_path in self.processed_paths:
                return False
            cur = self.agent.knowledge_base.conn.cursor()
            cur.execute("SELECT file_hash, last_modified FROM ingested_files WHERE file_path = ?", (file_path,))
            row = cur.fetchone()
            if not row:
                return True
            current_hash = self.compute_file_hash(file_path)
            current_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            return (current_hash != row[0] or current_mtime.timestamp() > datetime.fromisoformat(row[1]).timestamp())
        def record_processed_file(self, file_path):
            c_hash = self.compute_file_hash(file_path)
            c_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            cur = self.agent.knowledge_base.conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO ingested_files (file_path, file_hash, last_modified, ingested_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (file_path, c_hash, c_mtime.isoformat()))
            self.agent.knowledge_base.conn.commit()
            self.processed_paths.add(file_path)
        def on_created(self, event):
            if event.is_directory:
                logging.info(f"New directory detected: {event.src_path}")
                for root, _, files in os.walk(event.src_path):
                    for fl in files:
                        if fl.lower().endswith(".pdf"):
                            fp = os.path.join(root, fl)
                            self.process_pdf(fp)
                return
            if event.src_path.lower().endswith(".pdf"):
                self.process_pdf(event.src_path)
        def process_pdf(self, file_path):
            if self.should_process_file(file_path):
                logging.info(f"Processing PDF: {file_path}")
                asyncio.run(self.agent.ingest_source(file_path))
                self.record_processed_file(file_path)
            else:
                logging.info(f"Skipping already processed PDF: {file_path}")

    class FileStatus(Enum):
        """Enum representing the status of a file in memory"""
        UNCHANGED = auto()  # File in memory matches the file on disk
        MODIFIED = auto()   # File in memory is different from disk version
        NEW = auto()        # File only exists in memory
        CONFLICT = auto()   # File has edit conflicts
        HOT_SWAPPED = auto()  # File has been hot-swapped
        SYNCING = auto()    # File is being synced with database
        LOCKED = auto()     # File is locked for editing

    @classmethod
    def get_color(cls, status):
        """Get the color pair for a file status"""
        color_map = {
            cls.UNCHANGED: 1,    # Green
            cls.MODIFIED: 2,     # Yellow
            cls.NEW: 3,          # Blue
            cls.CONFLICT: 4,     # Red
            cls.HOT_SWAPPED: 5,  # Magenta
            cls.SYNCING: 6,      # Cyan
            cls.LOCKED: 7,       # White on red
        }
        return color_map.get(status, 0)

class DashboardPanel:
    """Base class for dashboard panels"""
    def __init__(self, title, height, width, y, x):
        self.window = curses.newwin(height, width, y, x)
        self.panel = panel.new_panel(self.window)
        self.title = title
        self.height = height
        self.width = width
        self.y = y
        self.x = x
        
    def draw(self):
        """Draw the panel border and title"""
        self.window.clear()
        self.window.box()
        self.window.addstr(0, 2, f" {self.title} ", curses.A_BOLD)
        
    def update(self, data):
        """Update panel content"""
        pass
        
    def resize(self, height, width, y, x):
        """Resize and reposition the panel"""
        self.height = height
        self.width = width
        self.y = y
        self.x = x
        self.window.resize(height, width)
        self.window.mvwin(y, x)

class FileStatus(Enum):
    """Enum representing the status of a file in memory"""
    UNCHANGED = auto()  # File in memory matches the file on disk
    MODIFIED = auto()   # File in memory is different from disk version
    NEW = auto()        # File only exists in memory
    CONFLICT = auto()   # File has edit conflicts
    HOT_SWAPPED = auto()  # File has been hot-swapped
    SYNCING = auto()    # File is being synced with database
    LOCKED = auto()     # File is locked for editing

class MemoryCodePanel(DashboardPanel):
    """Panel showing memory code activity"""
    def __init__(self, height, width, y, x):
        super().__init__("Memory Code Activity", height, width, y, x)
        self.files = {}  # file_path -> (status, timestamp, size)
        self.scroll_offset = 0
        self.selected_index = 0
        
    def update(self, agent):
        """Update panel with current memory code status"""
        try:
            # Get memory code manager
            code_manager = agent.self_transformation.code_manager
            
            # Update file statuses
            for file_path in code_manager.memory_code:
                # Check status
                status = FileStatus.UNCHANGED
                
                # Check if modified
                diffs = code_manager.get_memory_file_diff(file_path)
                if diffs:
                    status = FileStatus.MODIFIED
                    
                # Check if locked
                if file_path in code_manager.memory_locks and code_manager.memory_locks[file_path]:
                    status = FileStatus.LOCKED
                    
                # Check if has conflicts
                if (file_path in code_manager.edit_conflicts and 
                    code_manager.edit_conflicts[file_path]):
                    status = FileStatus.CONFLICT
                    
                # Check if new (doesn't exist on disk)
                if not os.path.exists(file_path):
                    status = FileStatus.NEW
                    
                # Check if hot-swapped
                module_name = os.path.basename(file_path).replace('.py', '')
                if module_name in code_manager.loaded_modules:
                    status = FileStatus.HOT_SWAPPED
                
                # Get timestamp and size
                timestamp = datetime.now().isoformat()  # Placeholder
                for mod in code_manager.modification_history:
                    if mod.get('file_path') == file_path:
                        timestamp = mod.get('timestamp', timestamp)
                        
                size = len("".join(code_manager.memory_code[file_path]))
                
                # Update file info
                self.files[file_path] = (status, timestamp, size)
                
            # Draw the dashboard
            self.draw()
            
        except Exception as e:
            self.window.addstr(1, 1, f"Error updating dashboard: {e}")
            
    def draw(self):
        """Draw the memory code dashboard"""
        super().draw()
        
        # Draw header
        self.window.addstr(1, 1, "File".ljust(30), curses.A_BOLD)
        self.window.addstr(1, 31, "Status".ljust(15), curses.A_BOLD)
        self.window.addstr(1, 46, "Size".ljust(10), curses.A_BOLD)
        self.window.addstr(1, 56, "Last Modified".ljust(20), curses.A_BOLD)
        
        # Draw horizontal separator
        self.window.addstr(2, 1, "─" * (self.width - 2))
        
        # Draw file list with scrolling
        display_height = self.height - 4  # Adjust for header and borders
        files_list = list(self.files.items())
        
        # Handle scrolling
        max_scroll = max(0, len(files_list) - display_height)
        self.scroll_offset = min(self.scroll_offset, max_scroll)
        self.scroll_offset = max(0, self.scroll_offset)
        
        # Handle selection
        self.selected_index = min(self.selected_index, len(files_list) - 1)
        self.selected_index = max(0, self.selected_index)
        
        # Ensure selection is visible
        if self.selected_index < self.scroll_offset:
            self.scroll_offset = self.selected_index
        elif self.selected_index >= self.scroll_offset + display_height:
            self.scroll_offset = self.selected_index - display_height + 1
        
        # Draw visible files
        for i, (file_path, (status, timestamp, size)) in enumerate(
            files_list[self.scroll_offset:self.scroll_offset + display_height]):
            
            y = i + 3  # Start after header and separator
            
            # Highlight selected row
            attr = curses.A_REVERSE if i + self.scroll_offset == self.selected_index else 0
            
            # Get filename (truncated) and format size
            filename = os.path.basename(file_path)
            if len(filename) > 28:
                filename = filename[:25] + "..."
                
            size_str = f"{size / 1024:.1f} KB" if size >= 1024 else f"{size} B"
            
            # Draw file info
            self.window.addstr(y, 1, filename.ljust(30), attr)
            
            # Draw status with color
            color_pair = FileStatus.get_color(status)
            status_attr = curses.color_pair(color_pair) | attr
            self.window.addstr(y, 31, status.name.ljust(15), status_attr)
            
            # Draw size and timestamp
            self.window.addstr(y, 46, size_str.ljust(10), attr)
            
            # Format timestamp
            ts = datetime.fromisoformat(timestamp)
            time_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            self.window.addstr(y, 56, time_str.ljust(20), attr)
            
        # Draw scrollbar if needed
        if len(files_list) > display_height:
            scrollbar_height = int(display_height * display_height / len(files_list))
            scrollbar_height = max(1, scrollbar_height)
            scrollbar_pos = int(self.scroll_offset * display_height / len(files_list))
            
            for i in range(display_height):
                if i >= scrollbar_pos and i < scrollbar_pos + scrollbar_height:
                    self.window.addstr(i + 3, self.width - 2, "█")
                else:
                    self.window.addstr(i + 3, self.width - 2, "│")
    
    def scroll_up(self):
        """Scroll up one line"""
        self.selected_index = max(0, self.selected_index - 1)
        
    def scroll_down(self):
        """Scroll down one line"""
        self.selected_index = min(len(self.files) - 1, self.selected_index + 1)
        
    def page_up(self):
        """Scroll up one page"""
        self.selected_index = max(0, self.selected_index - (self.height - 4))
        
    def page_down(self):
        """Scroll down one page"""
        self.selected_index = min(len(self.files) - 1, self.selected_index + (self.height - 4))

class SystemMetricsPanel(DashboardPanel):
    """Panel showing system performance metrics"""
    def __init__(self, height, width, y, x):
        super().__init__("System Metrics", height, width, y, x)
        self.metrics = {
            'memory_files': 0,
            'total_memory_size': 0,
            'locked_files': 0,
            'modified_files': 0,
            'hot_swapped_modules': 0,
            'snapshots': 0,
            'conflicts': 0,
            'sync_operations': 0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'update_time': 0.0,
        }
        self.history = {k: [] for k in self.metrics}
        self.max_history = 60  # Store up to 60 data points for graphs
        
    def update(self, agent):
        """Update system metrics"""
        try:
            start_time = time.time()
            
            # Get memory code manager
            code_manager = agent.self_transformation.code_manager
            
            # Count files and categorize
            memory_files = len(code_manager.memory_code)
            total_size = sum(len("".join(content)) for content in code_manager.memory_code.values())
            
            # Count locked files
            locked_files = sum(1 for locked in code_manager.memory_locks.values() if locked)
            
            # Count files with diffs
            modified_files = 0
            for file_path in code_manager.memory_code:
                diffs = code_manager.get_memory_file_diff(file_path)
                if diffs:
                    modified_files += 1
            
            # Count hot-swapped modules
            hot_swapped = len(code_manager.loaded_modules)
            
            # Count snapshots
            snapshots = sum(len(snaps) for snaps in code_manager.memory_snapshots.values())
            
            # Count conflicts
            conflicts = sum(len(conflicts) for conflicts in code_manager.edit_conflicts.values() if conflicts)
            
            # Count sync operations (from history)
            sync_ops = sum(1 for mod in code_manager.modification_history if mod.get('action') == 'sync_request')
            
            # Get system resource usage
            try:
                import psutil
                process = psutil.Process()
                cpu_usage = process.cpu_percent()
                memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
            except (ImportError, Exception):
                # Fallback if psutil not available
                cpu_usage = 0.0
                memory_usage = 0.0
            
            # Update metrics
            self.metrics['memory_files'] = memory_files
            self.metrics['total_memory_size'] = total_size
            self.metrics['locked_files'] = locked_files
            self.metrics['modified_files'] = modified_files
            self.metrics['hot_swapped_modules'] = hot_swapped
            self.metrics['snapshots'] = snapshots
            self.metrics['conflicts'] = conflicts
            self.metrics['sync_operations'] = sync_ops
            self.metrics['cpu_usage'] = cpu_usage
            self.metrics['memory_usage'] = memory_usage
            self.metrics['update_time'] = time.time() - start_time
            
            # Update history
            for k, v in self.metrics.items():
                self.history[k].append(v)
                if len(self.history[k]) > self.max_history:
                    self.history[k].pop(0)
            
            # Draw the dashboard
            self.draw()
            
        except Exception as e:
            self.window.addstr(1, 1, f"Error updating metrics: {e}")
    
    def draw(self):
        """Draw the system metrics dashboard"""
        super().draw()
        
        # Draw metrics in two columns
        col1_metrics = [
            ('Memory Files', 'memory_files', '{} files'),
            ('Total Size', 'total_memory_size', '{:.1f} KB'),
            ('Modified Files', 'modified_files', '{} files'),
            ('Locked Files', 'locked_files', '{} files'),
            ('Hot-Swapped', 'hot_swapped_modules', '{} modules'),
        ]
        
        col2_metrics = [
            ('Snapshots', 'snapshots', '{}'),
            ('Conflicts', 'conflicts', '{}'),
            ('Sync Operations', 'sync_operations', '{}'),
            ('CPU Usage', 'cpu_usage', '{:.1f}%'),
            ('Memory Usage', 'memory_usage', '{:.1f} MB'),
        ]
        
        # Draw column 1
        for i, (label, key, fmt) in enumerate(col1_metrics):
            # Format the value
            if key == 'total_memory_size':
                value = self.metrics[key] / 1024  # Convert to KB
            else:
                value = self.metrics[key]
                
            formatted = fmt.format(value)
            
            # Determine color based on value
            color = self._get_metric_color(key, value)
            
            # Draw label and value
            self.window.addstr(i + 2, 2, f"{label}:", curses.A_BOLD)
            self.window.addstr(i + 2, 15, formatted, curses.color_pair(color))
        
        # Draw column 2
        col2_x = self.width // 2
        for i, (label, key, fmt) in enumerate(col2_metrics):
            # Format the value
            value = self.metrics[key]
            formatted = fmt.format(value)
            
            # Determine color based on value
            color = self._get_metric_color(key, value)
            
            # Draw label and value
            self.window.addstr(i + 2, col2_x, f"{label}:", curses.A_BOLD)
            self.window.addstr(i + 2, col2_x + 15, formatted, curses.color_pair(color))
        
        # Draw mini performance graphs
        self._draw_mini_graph('CPU', self.history['cpu_usage'], 8, self.width - 20, 2)
        self._draw_mini_graph('MEM', self.history['memory_usage'], 9, self.width - 20, 2)
        
        # Draw update time at the bottom
        update_time = self.metrics['update_time'] * 1000  # Convert to ms
        self.window.addstr(self.height - 2, 2, f"Update Time: {update_time:.1f} ms")
    
    def _get_metric_color(self, key, value):
        """Get color based on metric value"""
        # Define thresholds for different metrics
        if key in ['conflicts', 'locked_files'] and value > 0:
            return 4  # Red for any conflicts or locks
        elif key == 'modified_files' and value > 0:
            return 2  # Yellow for modified files
        elif key == 'cpu_usage':
            if value > 80:
                return 4  # Red for high CPU
            elif value > 50:
                return 2  # Yellow for medium CPU
            else:
                return 1  # Green for low CPU
        elif key == 'hot_swapped_modules' and value > 0:
            return 5  # Magenta for hot-swapped modules
        else:
            return 0  # Default
    
    def _draw_mini_graph(self, label, data, y, x, width):
        """Draw a mini performance graph"""
        if not data:
            return
        
        # Calculate scaling
        max_val = max(data) if data else 1
        scale = width / max_val if max_val > 0 else 1
        
        # Draw label
        self.window.addstr(y, x - len(label) - 1, label, curses.A_BOLD)
        
        # Draw axis
        self.window.addstr(y, x, '│')
        
        # Draw the bar graph
        for i, val in enumerate(data[-width:]):
            bar_height = int(val * scale)
            if i < width:
                try:
                    self.window.addstr(y, x + i + 1, '█', self._get_graph_color(val, max_val))
                except curses.error:
                    pass  # Out of bounds
    
    def _get_graph_color(self, val, max_val):
        """Get color for graph based on value"""
        if val > max_val * 0.8:
            return curses.color_pair(4)  # Red for high values
        elif val > max_val * 0.5:
            return curses.color_pair(2)  # Yellow for medium values
        else:
            return curses.color_pair(1)  # Green for low values

class ActivityLogPanel(DashboardPanel):
    """Panel showing recent activity logs"""
    def __init__(self, height, width, y, x):
        super().__init__("Activity Log", height, width, y, x)
        self.log_entries = []
        self.scroll_offset = 0
        self.max_entries = 1000  # Maximum number of entries to keep
        self.filter = None  # Optional filter for log entries
        self.paused = False  # Whether to pause log updates
        
    def add_log(self, action, details, level="INFO"):
        """Add a log entry"""
        timestamp = datetime.now().isoformat()
        self.log_entries.append({
            'timestamp': timestamp,
            'action': action,
            'details': details,
            'level': level
        })
        
        # Trim if needed
        if len(self.log_entries) > self.max_entries:
            self.log_entries.pop(0)
    
    def update(self, agent):
        """Update log entries from agent activity"""
        if self.paused:
            # Skip updates when paused
            self.draw()
            return
            
        try:
            # Get memory code manager
            code_manager = agent.self_transformation.code_manager
            
            # Process modification history for new entries
            mod_history = code_manager.modification_history
            
            # Get the last processed modification timestamp
            last_processed = None
            if self.log_entries:
                last_processed = self.log_entries[-1].get('details', {}).get('timestamp')
            
            # Add new modifications as log entries
            for mod in mod_history:
                mod_time = mod.get('timestamp')
                
                # Skip already processed entries
                if last_processed and mod_time <= last_processed:
                    continue
                
                action = mod.get('action', 'unknown')
                file_path = mod.get('file_path', 'unknown')
                
                # Add log entry
                details = f"File: {os.path.basename(file_path)}"
                level = "INFO"
                
                if action == 'memory_edit':
                    self.add_log("EDIT", details)
                elif action == 'sync_request':
                    self.add_log("SYNC", details)
                elif action == 'snapshot':
                    snapshot_id = mod.get('snapshot_id', '?')
                    self.add_log("SNAPSHOT", f"{details} (ID: {snapshot_id})")
                elif action == 'hot_swap':
                    self.add_log("HOT-SWAP", details, "WARNING")
                elif action == 'conflict':
                    self.add_log("CONFLICT", details, "ERROR")
                elif action == 'lock':
                    self.add_log("LOCK", details, "WARNING")
                elif action == 'unlock':
                    self.add_log("UNLOCK", details)
            
            # Draw the panel
            self.draw()
        
        except Exception as e:
            self.window.addstr(1, 1, f"Error updating logs: {e}")
    
    def draw(self):
        """Draw the activity log panel"""
        super().draw()
        
        # Draw header and filter status
        filter_status = f"Filter: {self.filter}" if self.filter else "No filter"
        pause_status = "PAUSED" if self.paused else "LIVE"
        status = f"{filter_status} | {pause_status}"
        self.window.addstr(1, 2, status, curses.A_BOLD)
        
        # Draw separator
        self.window.addstr(2, 1, "─" * (self.width - 2))
        
        # Filter logs if needed
        if self.filter:
            filtered_logs = [log for log in self.log_entries if self.filter.lower() in 
                            log['action'].lower() or self.filter.lower() in log['details'].lower()]
        else:
            filtered_logs = self.log_entries
        
        # Handle scrolling
        display_height = self.height - 4  # Adjust for header and borders
        max_scroll = max(0, len(filtered_logs) - display_height)
        self.scroll_offset = min(max_scroll, self.scroll_offset)
        
        # Draw logs
        for i, log in enumerate(filtered_logs[self.scroll_offset:self.scroll_offset + display_height]):
            y = i + 3  # Start after header and separator
            
            # Format timestamp
            ts = datetime.fromisoformat(log['timestamp'])
            time_str = ts.strftime("%H:%M:%S")
            
            # Determine color based on level
            if log['level'] == "ERROR":
                color = curses.color_pair(4)  # Red
            elif log['level'] == "WARNING":
                color = curses.color_pair(2)  # Yellow
            else:
                color = curses.color_pair(0)  # Default
            
            # Format and truncate text
            action = log['action'].ljust(10)
            details = log['details']
            if len(details) > self.width - 25:
                details = details[:self.width - 28] + "..."
            
            # Draw log entry
            self.window.addstr(y, 2, time_str, curses.A_DIM)
            self.window.addstr(y, 11, action, color | curses.A_BOLD)
            self.window.addstr(y, 22, details)
            
        # Draw scrollbar if needed
        if len(filtered_logs) > display_height:
            scrollbar_height = int(display_height * display_height / len(filtered_logs))
            scrollbar_height = max(1, scrollbar_height)
            scrollbar_pos = int(self.scroll_offset * display_height / len(filtered_logs))
            
            for i in range(display_height):
                if i >= scrollbar_pos and i < scrollbar_pos + scrollbar_height:
                    self.window.addstr(i + 3, self.width - 2, "█")
                else:
                    self.window.addstr(i + 3, self.width - 2, "│")
    
    def set_filter(self, filter_text):
        """Set filter for log entries"""
        self.filter = filter_text
    
    def toggle_pause(self):
        """Toggle pause state"""
        self.paused = not self.paused
    
    def scroll_up(self):
        """Scroll up one line"""
        self.scroll_offset = max(0, self.scroll_offset - 1)
    
    def scroll_down(self):
        """Scroll down one line"""
        self.scroll_offset = self.scroll_offset + 1
    
    def page_up(self):
        """Scroll up one page"""
        self.scroll_offset = max(0, self.scroll_offset - (self.height - 4))
    
    def page_down(self):
        """Scroll down one page"""
        self.scroll_offset = self.scroll_offset + (self.height - 4)

class TUIApp:
    """Text User Interface Application for monitoring system activity"""
    def __init__(self, agent):
        self.agent = agent
        self.stdscr = None
        self.panels = []
        self.memory_code_panel = None
        self.system_metrics_panel = None
        self.activity_log_panel = None
        self.running = False
        self.update_interval = 0.3  # seconds
        self.active_panel_index = 0
        self.help_visible = False
        self.layout_mode = 0  # 0: default, 1: focused, 2: maximized
        
    def setup_colors(self):
        """Setup color pairs"""
        curses.start_color()
        curses.use_default_colors()
        
        # Define color pairs for file statuses
        curses.init_pair(1, curses.COLOR_GREEN, -1)    # UNCHANGED
        curses.init_pair(2, curses.COLOR_YELLOW, -1)   # MODIFIED
        curses.init_pair(3, curses.COLOR_BLUE, -1)     # NEW
        curses.init_pair(4, curses.COLOR_RED, -1)      # CONFLICT
        curses.init_pair(5, curses.COLOR_MAGENTA, -1)  # HOT_SWAPPED
        curses.init_pair(6, curses.COLOR_CYAN, -1)     # SYNCING
        curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_RED)  # LOCKED
        
    def setup_panels(self):
        """Create dashboard panels"""
        height, width = self.stdscr.getmaxyx()
        
        # Create panels with different layouts
        third_height = height // 3
        half_height = height // 2
        
        # Create memory code panel (top left, expanded)
        code_height = 2 * third_height
        self.memory_code_panel = MemoryCodePanel(code_height, width, 0, 0)
        self.panels.append(self.memory_code_panel)
        
        # Create system metrics panel (top right)
        metrics_height = third_height
        self.system_metrics_panel = SystemMetricsPanel(metrics_height, width // 2, 0, width // 2)
        self.panels.append(self.system_metrics_panel)
        
        # Create activity log panel (bottom)
        log_height = third_height
        self.activity_log_panel = ActivityLogPanel(log_height, width, height - log_height, 0)
        self.panels.append(self.activity_log_panel)
        
        # Add initial log entries
        self.activity_log_panel.add_log("SYSTEM", "Dashboard initialized")
        self.activity_log_panel.add_log("INFO", f"Monitoring {len(self.agent.self_transformation.code_manager.memory_code)} files")
        
    def resize_panels(self):
        """Resize panels to fit the screen based on layout mode"""
        height, width = self.stdscr.getmaxyx()
        
        if self.layout_mode == 0:  # Default layout
            # Split screen in thirds vertically
            third_height = height // 3
            
            # Memory code panel (top 2/3)
            code_height = 2 * third_height
            self.memory_code_panel.resize(code_height, width, 0, 0)
            
            # System metrics panel (top right)
            self.system_metrics_panel.resize(third_height, width // 2, 0, width // 2)
            
            # Activity log panel (bottom)
            log_height = height - code_height
            self.activity_log_panel.resize(log_height, width, height - log_height, 0)
            
        elif self.layout_mode == 1:  # Focus on active panel
            # Make active panel larger
            if self.active_panel_index == 0:  # Memory code panel
                # Memory code panel (3/4 of screen)
                self.memory_code_panel.resize(3 * height // 4, width, 0, 0)
                
                # System metrics panel (right top of remaining)
                self.system_metrics_panel.resize(height // 8, width // 2, 3 * height // 4, 0)
                
                # Activity log panel (right bottom of remaining)
                self.activity_log_panel.resize(height // 8, width // 2, 7 * height // 8, 0)
                
            elif self.active_panel_index == 1:  # System metrics panel
                # System metrics panel (3/4 of screen)
                self.system_metrics_panel.resize(3 * height // 4, width, 0, 0)
                
                # Memory code panel (left of remaining)
                self.memory_code_panel.resize(height // 4, width // 2, 3 * height // 4, 0)
                
                # Activity log panel (right of remaining)
                self.activity_log_panel.resize(height // 4, width // 2, 3 * height // 4, width // 2)
                
            elif self.active_panel_index == 2:  # Activity log panel
                # Activity log panel (3/4 of screen)
                self.activity_log_panel.resize(3 * height // 4, width, 0, 0)
                
                # Memory code panel (left of remaining)
                self.memory_code_panel.resize(height // 4, width // 2, 3 * height // 4, 0)
                
                # System metrics panel (right of remaining)
                self.system_metrics_panel.resize(height // 4, width // 2, 3 * height // 4, width // 2)
                
        elif self.layout_mode == 2:  # Maximize active panel
            # Only show active panel
            if self.active_panel_index == 0:  # Memory code panel
                self.memory_code_panel.resize(height, width, 0, 0)
                self.system_metrics_panel.resize(0, 0, 0, 0)  # Hide
                self.activity_log_panel.resize(0, 0, 0, 0)    # Hide
                
            elif self.active_panel_index == 1:  # System metrics panel
                self.memory_code_panel.resize(0, 0, 0, 0)     # Hide
                self.system_metrics_panel.resize(height, width, 0, 0)
                self.activity_log_panel.resize(0, 0, 0, 0)    # Hide
                
            elif self.active_panel_index == 2:  # Activity log panel
                self.memory_code_panel.resize(0, 0, 0, 0)     # Hide
                self.system_metrics_panel.resize(0, 0, 0, 0)  # Hide
                self.activity_log_panel.resize(height, width, 0, 0)
                
    def draw_help_overlay(self):
        """Draw help overlay with keyboard shortcuts"""
        if not self.help_visible:
            return
            
        height, width = self.stdscr.getmaxyx()
        help_height = 14
        help_width = 40
        
        # Create help window in the center
        help_y = (height - help_height) // 2
        help_x = (width - help_width) // 2
        
        help_win = curses.newwin(help_height, help_width, help_y, help_x)
        help_win.box()
        help_win.addstr(0, 2, " Keyboard Shortcuts ", curses.A_BOLD)
        
        # Add keyboard shortcuts
        shortcuts = [
            ("q", "Quit"),
            ("h", "Toggle help"),
            ("Tab", "Switch panels"),
            ("l", "Change layout mode"),
            ("m", "Toggle maximize"),
            ("r", "Force refresh"),
            ("p", "Toggle pause (log panel)"),
            ("f", "Set filter (log panel)"),
            ("c", "Clear logs/history"),
            ("↑/↓", "Scroll up/down"),
            ("PgUp/PgDn", "Page up/down"),
            ("Esc", "Close overlay")
        ]
        
        for i, (key, desc) in enumerate(shortcuts):
            help_win.addstr(i + 1, 2, key, curses.A_BOLD)
            help_win.addstr(i + 1, 10, desc)
            
        help_win.refresh()
        
    def update_panels(self):
        """Update all panels with current data"""
        # Skip updating hidden panels based on layout mode
        if self.layout_mode < 2 or self.active_panel_index == 0:
            self.memory_code_panel.update(self.agent)
            
        if self.layout_mode < 2 or self.active_panel_index == 1:
            self.system_metrics_panel.update(self.agent)
            
        if self.layout_mode < 2 or self.active_panel_index == 2:
            self.activity_log_panel.update(self.agent)
            
        # Draw help overlay if visible
        self.draw_help_overlay()
        
    def run(self, stdscr):
        """Main TUI application loop"""
        self.stdscr = stdscr
        self.setup_colors()
        self.setup_panels()
        self.running = True
        
        # Hide cursor
        curses.curs_set(0)
        
        # Don't wait for key input
        self.stdscr.nodelay(True)
        
        # Enable keypad for special keys
        self.stdscr.keypad(True)
        
        last_update = 0
        
        while self.running:
            # Handle window resize
            try:
                height, width = self.stdscr.getmaxyx()
                resize_needed = False
                
                # Check if any panel needs resizing
                for p in self.panels:
                    if p.height > 0 and p.width > 0:  # Only check visible panels
                        if height != p.height + p.y or width != p.width + p.x:
                            resize_needed = True
                            break
                            
                if resize_needed:
                    self.resize_panels()
            except curses.error:
                pass
            
            # Update panels periodically
            now = time.time()
            if now - last_update > self.update_interval:
                self.update_panels()
                last_update = now
                
            # Update display
            panel.update_panels()
            curses.doupdate()
            
            # Handle keyboard input
            try:
                key = self.stdscr.getch()
                if key != -1:
                    self.handle_key(key)
            except curses.error:
                pass
                
            # Small sleep to reduce CPU usage
            time.sleep(0.05)
            
    def handle_key(self, key):
        """Handle keyboard input"""
        if key == ord('q'):
            self.running = False
        elif key == ord('h'):
            # Toggle help overlay
            self.help_visible = not self.help_visible
        elif key == 9:  # Tab
            # Cycle through panels
            self.active_panel_index = (self.active_panel_index + 1) % len(self.panels)
            self.activity_log_panel.add_log("SYSTEM", f"Active panel: {self.panels[self.active_panel_index].title}")
        elif key == ord('l'):
            # Cycle through layout modes
            self.layout_mode = (self.layout_mode + 1) % 3
            layout_names = ["Default", "Focused", "Maximized"]
            self.activity_log_panel.add_log("SYSTEM", f"Layout: {layout_names[self.layout_mode]}")
            self.resize_panels()
        elif key == ord('m'):
            # Toggle maximize current panel
            if self.layout_mode == 2:
                self.layout_mode = 0  # Back to default
            else:
                self.layout_mode = 2  # Maximize
            self.resize_panels()
        elif key == ord('p'):
            # Toggle pause (for log panel)
            if self.active_panel_index == 2:
                self.activity_log_panel.toggle_pause()
        elif key == ord('f'):
            # Set filter (for log panel)
            if self.active_panel_index == 2:
                # TODO: Implement filter input
                pass
        elif key == ord('c'):
            # Clear current panel data
            if self.active_panel_index == 2:
                self.activity_log_panel.log_entries = []
                self.activity_log_panel.add_log("SYSTEM", "Logs cleared")
        elif key == 27:  # Escape
            # Close overlay or reset layout
            if self.help_visible:
                self.help_visible = False
            elif self.layout_mode != 0:
                self.layout_mode = 0
                self.resize_panels()
        elif key == curses.KEY_UP:
            if self.active_panel_index == 0:
                self.memory_code_panel.scroll_up()
            elif self.active_panel_index == 2:
                self.activity_log_panel.scroll_up()
        elif key == curses.KEY_DOWN:
            if self.active_panel_index == 0:
                self.memory_code_panel.scroll_down()
            elif self.active_panel_index == 2:
                self.activity_log_panel.scroll_down()
        elif key == curses.KEY_PPAGE:
            if self.active_panel_index == 0:
                self.memory_code_panel.page_up()
            elif self.active_panel_index == 2:
                self.activity_log_panel.page_up()
        elif key == curses.KEY_NPAGE:
            if self.active_panel_index == 0:
                self.memory_code_panel.page_down()
            elif self.active_panel_index == 2:
                self.activity_log_panel.page_down()
        elif key == ord('r'):
            # Force update
            self.update_panels()

class AgentCLI(cmd.Cmd):
        intro = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   Welcome to the Advanced Agent CLI with Self-Modification       ║
║                                                                  ║
║   Type 'help' or '?' to list commands                           ║
║   Type 'dashboard' to launch the interactive dashboard           ║
║   Type 'chat <message>' to interact with the agent               ║
║   Type 'polymorphic' to apply code transformations               ║
║   Type 'exit' to quit                                            ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
        prompt = "\033[1;32m(agent)\033[0m "
    
        def do_enable_autonomous(self, arg):
            """Enable autonomous mode: enable_autonomous [max_changes]"""
            try:
                max_changes = int(arg) if arg else 5
                self.agent.self_transformation.autonomous_mode = True
                self.agent.self_transformation.max_autonomous_changes = max_changes
                self.agent.self_transformation.changes_counter = 0
                print(f"Autonomous mode enabled with max {max_changes} changes")
            except ValueError:
                print("Invalid argument. Usage: enable_autonomous [max_changes]")
    
        def do_disable_autonomous(self, arg):
            """Disable autonomous mode: disable_autonomous"""
            self.agent.self_transformation.autonomous_mode = False
            print("Autonomous mode disabled")
            
        def do_view_memory_code(self, arg):
            """View in-memory code: view_memory_code [file_path]"""
            if not arg:
                print("Usage: view_memory_code [file_path]")
                return
                
            file_path = arg.strip()
            try:
                content = self.agent.self_transformation.code_manager.get_memory_code(file_path)
                if content:
                    print("\n=== In-Memory Code for", file_path, "===\n")
                    # Print with line numbers
                    for i, line in enumerate(content.splitlines()):
                        print(f"{i+1:4d}: {line}")
                else:
                    print(f"No in-memory code found for {file_path}")
            except Exception as e:
                print(f"Error viewing memory code: {e}")
                
        def do_show_memory_diff(self, arg):
            """Show differences between file and memory: show_memory_diff [file_path]"""
            if not arg:
                print("Usage: show_memory_diff [file_path]")
                return
                
            file_path = arg.strip()
            try:
                diffs = self.agent.self_transformation.code_manager.get_memory_file_diff(file_path)
                if diffs:
                    print(f"\n=== Differences for {file_path} ===\n")
                    for diff in diffs:
                        print(f"Line {diff['line']}:")
                        print(f"  File: {diff['file'].rstrip()}")
                        print(f"  Memory: {diff['memory'].rstrip()}")
                        print()
                else:
                    print(f"No differences found for {file_path}")
            except Exception as e:
                print(f"Error showing differences: {e}")
                print(traceback.format_exc())
                
        def do_snapshot(self, arg):
            """Create a snapshot of in-memory code: snapshot [file_path] [description]"""
            args = shlex.split(arg)
            if not args:
                print("Usage: snapshot [file_path] [description]")
                return
                
            file_path = args[0]
            description = ' '.join(args[1:]) if len(args) > 1 else None
            
            try:
                snapshot_id = self.agent.self_transformation.code_manager.create_memory_snapshot(
                    file_path, description)
                    
                if snapshot_id > 0:
                    print(f"Created memory snapshot {snapshot_id} for {file_path}")
                else:
                    print(f"Failed to create snapshot for {file_path}")
            except Exception as e:
                print(f"Error creating snapshot: {e}")
                
        def do_dashboard(self, arg):
            """
            Launch the interactive TUI dashboard: dashboard
            
            The dashboard provides a real-time view of memory code activity with color-coded 
            status indicators for each file. Navigate using arrow keys and press 'q' to exit.
            """
            try:
                # Initialize and run the TUI app
                app = TUIApp(self.agent)
                curses.wrapper(app.run)
            except Exception as e:
                print(f"Error running dashboard: {e}")
                print(traceback.format_exc())
                
        def do_list_snapshots(self, arg):
            """List snapshots for a file: list_snapshots [file_path]"""
            if not arg:
                print("Usage: list_snapshots [file_path]")
                return
                
            file_path = arg.strip()
            try:
                snapshots = self.agent.self_transformation.code_manager.get_memory_snapshots(file_path)
                if snapshots:
                    print(f"\nSnapshots for {file_path}:")
                    for snapshot in snapshots:
                        active = " (ACTIVE)" if snapshot['active'] else ""
                        print(f"  {snapshot['id']}: {snapshot['description']} - {snapshot['timestamp']}{active}")
                else:
                    print(f"No snapshots found for {file_path}")
            except Exception as e:
                print(f"Error listing snapshots: {e}")
                
        def do_restore_snapshot(self, arg):
            """Restore a snapshot: restore_snapshot [file_path] [snapshot_id]"""
            args = shlex.split(arg)
            if len(args) != 2:
                print("Usage: restore_snapshot [file_path] [snapshot_id]")
                return
                
            file_path = args[0]
            try:
                snapshot_id = int(args[1])
                success = self.agent.self_transformation.code_manager.restore_memory_snapshot(
                    file_path, snapshot_id)
                    
                if success:
                    print(f"Restored snapshot {snapshot_id} for {file_path}")
                else:
                    print(f"Failed to restore snapshot {snapshot_id} for {file_path}")
            except ValueError:
                print(f"Invalid snapshot ID: {args[1]}")
            except Exception as e:
                print(f"Error restoring snapshot: {e}")
                
        def do_hot_swap(self, arg):
            """Hot-swap a module with in-memory code: hot_swap [file_path]"""
            if not arg:
                print("Usage: hot_swap [file_path]")
                return
                
            file_path = arg.strip()
            try:
                # Create a fake async context to run the hot_swap
                result = asyncio.run(
                    self.agent.self_transformation.code_manager.hot_swap_module(file_path))
                    
                if result:
                    print(f"Successfully hot-swapped module {os.path.basename(file_path)}")
                else:
                    print(f"Failed to hot-swap module {file_path}")
            except Exception as e:
                print(f"Error hot-swapping module: {e}")
                print(traceback.format_exc())
                
        def do_edit_in_memory(self, arg):
            """Edit code in memory only: edit_in_memory [file_path] [code_content or '-' for multiline]"""
            args = shlex.split(arg)
            if len(args) < 2:
                print("Usage: edit_in_memory [file_path] [code_content or '-' for multiline]")
                return
                
            file_path = args[0]
            if args[1] == '-':
                # Multiline mode
                print("Enter code content (end with a line containing only '###'):")
                content_lines = []
                while True:
                    line = input()
                    if line.strip() == '###':
                        break
                    content_lines.append(line + '\n')
                content = ''.join(content_lines)
            else:
                content = ' '.join(args[1:])
                
            # Perform the edit in memory
            try:
                # Create a fake async context to run the edit_in_memory function
                success = asyncio.run(
                    self.agent.self_transformation.code_manager.edit_in_memory(file_path, content))
                    
                if success:
                    print(f"Successfully edited {file_path} in memory")
                    # Access the code element to verify
                    element_name = os.path.basename(file_path).replace('.py', '')
                    code_element = self.agent.self_transformation.code_manager.get_code_element("class", element_name)
                    if code_element:
                        print(f"Code element '{element_name}' was updated in memory")
                    else:
                        print(f"Code element '{element_name}' was not found in memory")
                else:
                    print(f"Failed to edit {file_path} in memory")
            except Exception as e:
                print(f"Error editing in memory: {e}")
                print(traceback.format_exc())
    
        def do_get_evolution_stats(self, arg):
            """Show evolution statistics: get_evolution_stats"""
            history = self.agent.self_transformation.evolution_tracker.get_transformation_history()
            print("\nCode Evolution Statistics:")
            print(f"Total transformations: {len(history)}")
            autonomous = sum(1 for h in history if h['metadata'].get('autonomous', False))
            print(f"Autonomous transformations: {autonomous}")
            print(f"Current learning rate: {self.agent.self_transformation.learning_rate:.3f}")
            print("\nRecent transformations:")
            for h in history[-5:]:
                print(f"- {h['timestamp']}: {h['metadata']['type']}")
                
        def do_self_modify(self, arg):
            """Request the agent to modify its own code: self_modify [target_function]"""
            if not arg:
                print("Usage: self_modify [target_function or description]")
                return
                
            prompt = f"Please modify your code to improve the {arg} functionality. Consider performance, readability, and error handling."
            try:
                print(f"Requesting self-modification for: {arg}")
                result = asyncio.run(self.agent.qa(self.current_conversation, prompt))
                print(f"\nSelf-modification result: {result}")
            except Exception as e:
                print(f"Error during self-modification: {e}")
                
        def do_polymorphic(self, arg):
            """Apply polymorphic transformations to code: polymorphic [file_path] [intensity]"""
            args = shlex.split(arg)
            if not args:
                print("Usage: polymorphic [file_path] [intensity]")
                return
                
            file_path = args[0]
            intensity = float(args[1]) if len(args) > 1 else 0.5
            
            try:
                # Import the polymorphic engine
                from polymorphic_engine import PolymorphicEngine
                
                # Create the engine
                engine = PolymorphicEngine()
                
                print(f"Applying polymorphic transformations to {file_path} with intensity {intensity}...")
                
                # Apply the transformation
                success, message = engine.transform_file(file_path, intensity=intensity)
                
                if success:
                    print(f"Transformation successful: {message}")
                else:
                    print(f"Transformation failed: {message}")
                    
            except ImportError:
                print("The polymorphic engine module is not available. Please ensure it's properly installed.")
            except Exception as e:
                print(f"Error during polymorphic transformation: {e}")
                print(traceback.format_exc())
                
        def do_analyze_code(self, arg):
            """Analyze code structure: analyze_code [file_path]"""
            if not arg:
                print("Usage: analyze_code [file_path]")
                return
                
            file_path = arg.strip()
            
            try:
                # Import the polymorphic engine
                from polymorphic_engine import PolymorphicEngine
                
                # Create the engine
                engine = PolymorphicEngine()
                
                print(f"Analyzing code structure of {file_path}...")
                
                # Analyze the file
                results = engine.analyze_file(file_path)
                
                if "error" in results:
                    print(f"Analysis failed: {results['error']}")
                    return
                    
                # Print summary
                print("\n=== Code Analysis Summary ===")
                print(f"File: {file_path}")
                print(f"Lines of code: {results['loc']}")
                print(f"Complexity: {results['complexity']}")
                print(f"Functions: {len(results['functions'])}")
                print(f"Classes: {len(results['classes'])}")
                print(f"Imports: {len(results['imports'])}")
                
                # Print detailed function info
                if results['functions']:
                    print("\n=== Functions ===")
                    for name, info in results['functions'].items():
                        print(f"  {name}:")
                        print(f"    Lines: {info['line']}-{info['end_line']}")
                        print(f"    Complexity: {info['complexity']}")
                        print(f"    Arguments: {info['args']}")
                        
                # Print detailed class info
                if results['classes']:
                    print("\n=== Classes ===")
                    for name, info in results['classes'].items():
                        print(f"  {name}:")
                        print(f"    Lines: {info['line']}-{info['end_line']}")
                        print(f"    Bases: {info['bases']}")
                        print(f"    Methods: {info['methods']}")
                        
            except ImportError:
                print("The polymorphic engine module is not available. Please ensure it's properly installed.")
            except Exception as e:
                print(f"Error analyzing code: {e}")
                print(traceback.format_exc())
                
        def do_generate_variants(self, arg):
            """Generate code variants: generate_variants [file_path] [num_variants] [intensity]"""
            args = shlex.split(arg)
            if len(args) < 2:
                print("Usage: generate_variants [file_path] [num_variants] [intensity]")
                return
                
            file_path = args[0]
            num_variants = int(args[1]) if len(args) > 1 else 1
            intensity = float(args[2]) if len(args) > 2 else 0.5
            
            try:
                # Import the polymorphic engine
                from polymorphic_engine import PolymorphicEngine
                
                # Create the engine
                engine = PolymorphicEngine()
                
                print(f"Generating {num_variants} variants of {file_path} with intensity {intensity}...")
                
                # Generate variants
                variants = engine.generate_variant(file_path, num_variants, intensity)
                
                if variants:
                    print(f"Successfully generated {len(variants)} variants:")
                    for variant in variants:
                        print(f"  {variant}")
                else:
                    print("Failed to generate variants.")
                    
            except ImportError:
                print("The polymorphic engine module is not available. Please ensure it's properly installed.")
            except Exception as e:
                print(f"Error generating variants: {e}")
                print(traceback.format_exc())
                
        def do_list_tools(self, arg):
            """List all available tools: list_tools"""
            print("\nAvailable Tools:")
            for tool in tools:
                name = tool["function"]["name"]
                desc = tool["function"]["description"]
                print(f"- {name}: {desc}")
                
        def do_register_tool(self, arg):
            """Register a new tool: register_tool [name] [code]"""
            args = shlex.split(arg)
            if len(args) < 2:
                print("Usage: register_tool [name] [code]")
                return
                
            name = args[0]
            code = " ".join(args[1:])
            
            try:
                # Execute the code in global namespace
                exec(code, globals())
                
                # Register the tool
                register_tool(
                    name=name,
                    func=globals()[name],
                    description=f"User-defined tool: {name}",
                    parameters={"args": {"type": "object"}}
                )
                print(f"Tool '{name}' registered successfully.")
            except Exception as e:
                print(f"Error registering tool: {e}")
                
        def do_test_transformation(self, arg):
            """Test a code transformation without applying it: test_transformation [target_function]"""
            if not arg:
                print("Usage: test_transformation [target_function or description]")
                return
                
            try:
                # Retrieve relevant code
                code_chunks = asyncio.run(self.agent.retrieve_relevant_code(arg))
                
                # Parse into a proper code chunk format
                chunk = {
                    "type": "function",
                    "name": arg,
                    "content": code_chunks
                }
                
                # Propose transformation
                print(f"Generating transformation proposal for: {arg}")
                transformation = asyncio.run(
                    self.agent.self_transformation.propose_transformation(chunk)
                )
                
                if transformation:
                    print("\n📝 Proposed transformation:")
                    print(transformation["suggestion"])
                    
                    # Create test instance
                    print("\n🧪 Creating test instance...")
                    instance_id = asyncio.run(
                        self.agent.self_transformation.create_test_instance(transformation)
                    )
                    
                    if instance_id:
                        print(f"Test instance created: {instance_id}")
                        
                        # Run tests
                        print("\n🧪 Running tests...")
                        test_results = asyncio.run(
                            self.agent.self_transformation.test_transformation(instance_id)
                        )
                        
                        print("\n📊 Test Results:")
                        print(f"Success Rate: {test_results.get('success_rate', 0)*100:.0f}%")
                        print(f"Passed: {test_results.get('passed', False)}")
                        
                        for test_name, result in test_results.items():
                            if isinstance(result, dict) and test_name not in ["instance_id", "timestamp", "success_rate", "passed"]:
                                print(f"\n{test_name.replace('_', ' ').title()}:")
                                print(f"  Success: {result.get('success', False)}")
                                print(f"  Message: {result.get('message', 'No message')}")
                                if "details" in result:
                                    print(f"  Details: {result['details']}")
                        
                        # Clean up
                        asyncio.run(self.agent.self_transformation.cleanup_test_instance(instance_id))
                        print("\nTest instance cleaned up.")
                    else:
                        print("Failed to create test instance.")
                else:
                    print("Failed to generate transformation proposal.")
            except Exception as e:
                print(f"Error testing transformation: {e}")
                print(traceback.format_exc())
                
        def do_check_restart(self, arg):
            """Check if a restart is needed and restart the agent if necessary"""
            restart_file = "agent_restart_signal.txt"
            if os.path.exists(restart_file):
                print("Restart signal detected. Restarting agent...")
                os.remove(restart_file)
                
                # Restart the agent
                print("Stopping current agent...")
                asyncio.run(self.agent.stop_reflections())
                
                print("Reloading agent code...")
                # Force reload of the agent module
                import importlib
                import sys
                
                # Remove the module from sys.modules
                if "rl_cli" in sys.modules:
                    del sys.modules["rl_cli"]
                
                # Reimport the module
                import rl_cli
                importlib.reload(rl_cli)
                
                # Create a new agent instance
                print("Creating new agent instance...")
                self.agent = rl_cli.Agent(instructions="Restarted agent with updated code.")
                
                print("Agent restarted successfully!")
            else:
                print("No restart signal detected.")
        def setup_pdf_watcher(self, path="./pdfs"):
            abs_path = os.path.abspath(path)
            if not os.path.exists(abs_path):
                os.makedirs(abs_path)
                logging.info(f"Created PDFs directory: {abs_path}")
            self.observer = Observer()
            handler = PDFHandler(self.agent)
            self.observer.schedule(handler, abs_path, recursive=True)
            self.observer.start()
            logging.info(f"Started watching {abs_path} for new PDFs.")
            for root, _, files in os.walk(abs_path):
                for fl in files:
                    if fl.lower().endswith(".pdf"):
                        fp = os.path.join(root, fl)
                        handler.process_pdf(fp)
        def __init__(self):
            super().__init__()
            self.agent = Agent(instructions="Ingest documents, reflect on its own code, and answer queries with self-modification.")
            self.current_conversation = None
            self.prompt_queue = []
            self.history = []
            self.last_command_time = time.time()
            
            # Set up command history
            self.history_file = os.path.expanduser("~/.agent_cli_history")
            try:
                import readline
                if os.path.exists(self.history_file):
                    readline.read_history_file(self.history_file)
                    print(f"Loaded command history from {self.history_file}")
                # Set history length
                readline.set_history_length(1000)
                # Save history on exit
                import atexit
                atexit.register(readline.write_history_file, self.history_file)
            except (ImportError, IOError):
                pass
                
            # Print initialization info
            print(f"Agent ID: {self.agent.id}")
            print(f"Agent Instructions: {self.agent.instructions}")
            print(f"Database: {self.agent.knowledge_base.db_path}")
            
            # Start reflections (will be started by main if not disabled)
            # asyncio.run(self.agent.start_reflections())
        def do_ingest(self, arg):
            if not arg:
                print("Usage: ingest <source>")
                return
            try:
                asyncio.run(self.agent.ingest_source(arg))
                print(f"Ingested: {arg}")
            except Exception as e:
                print(f"Error: {e}")
        def do_load_code_ast(self, arg):
            try:
                asyncio.run(self.agent.load_own_code_ast())
                print("AST-based code ingestion completed.")
            except Exception as e:
                print(f"Error: {e}")
        def do_retrieve_code(self, arg):
            if not arg:
                print("Usage: retrieve_code <query>")
                return
            try:
                snippet = asyncio.run(self.agent.retrieve_relevant_code(arg))
                print("\nRelevant Code Chunks:")
                print(snippet)
            except Exception as e:
                print(f"Error: {e}")
        def do_show_evolution(self, arg):
            """Show the code evolution graph: show_evolution"""
            if hasattr(self.agent.self_transformation, 'evolution_tracker'):
                self.agent.self_transformation.evolution_tracker.visualize()
                print("Code evolution graph saved to code_evolution.png")
            else:
                print("No code evolution tracking available")

        def do_chat(self, arg):
            """Chat with the agent: chat <message>"""
            if not arg:
                print("Usage: chat <message>")
                return
            try:
                if not self.current_conversation:
                    self.current_conversation = asyncio.run(self.agent.create_conversation())
                    print(f"Started new conversation: {self.current_conversation}")
                self.prompt_queue.append(arg)
                while self.prompt_queue:
                    prompt = self.prompt_queue.pop(0)
                    print(f"\n\033[1;34mUser:\033[0m {prompt}")
                    
                    # Show a spinner while waiting for response
                    import threading
                    import itertools
                    
                    spinner_active = True
                    def spin():
                        for c in itertools.cycle(['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷']):
                            if not spinner_active:
                                break
                            sys.stdout.write('\rThinking ' + c)
                            sys.stdout.flush()
                            time.sleep(0.1)
                    
                    spinner = threading.Thread(target=spin)
                    spinner.start()
                    
                    try:
                        answer = asyncio.run(self.agent.qa(self.current_conversation, prompt))
                    finally:
                        spinner_active = False
                        spinner.join()
                        sys.stdout.write('\r          \r')  # Clear the spinner
                    
                    print(f"\n\033[1;32mAgent:\033[0m {answer}")
                    
                    # Add to history
                    self.history.append({"role": "user", "content": prompt})
                    self.history.append({"role": "assistant", "content": answer})
            except Exception as e:
                print(f"Error: {e}")
                print(traceback.format_exc())
                
        def do_new_chat(self, arg):
            """Start a new conversation: new_chat [title]"""
            title = arg if arg else f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            try:
                self.current_conversation = asyncio.run(self.agent.create_conversation(title))
                print(f"Started new conversation: {self.current_conversation}")
                self.history = []
            except Exception as e:
                print(f"Error: {e}")
                
        def do_multiline(self, arg):
            """Enter a multiline message to send to the agent: multiline"""
            print("Enter your multiline message (type '###' on a line by itself to end):")
            lines = []
            while True:
                try:
                    line = input()
                    if line.strip() == '###':
                        break
                    lines.append(line)
                except EOFError:
                    break
                    
            message = '\n'.join(lines)
            if message.strip():
                self.do_chat(message)
            else:
                print("Empty message, not sending.")
        def do_status(self, arg):
            """Show current agent status: status"""
            print("\n=== Agent Status ===")
            print(f"Agent ID: {self.agent.id}")
            print(f"Autonomous mode: {self.agent.self_transformation.autonomous_mode}")
            if self.agent.self_transformation.autonomous_mode:
                print(f"  Changes made: {self.agent.self_transformation.changes_counter}")
                print(f"  Max changes: {self.agent.self_transformation.max_autonomous_changes}")
            
            # Check reflection status
            reflection_status = "Running" if self.agent._reflection_running else "Stopped"
            print(f"Reflections: {reflection_status}")
            if self.agent._reflection_running:
                print(f"  Queue size: {self.agent._reflection_queue.qsize()}")
            
            # Check memory code stats
            code_manager = self.agent.self_transformation.code_manager
            memory_files = len(code_manager.memory_code)
            print(f"Memory files: {memory_files}")
            
            # Check conversation stats
            print(f"Current conversation: {self.current_conversation}")
            print(f"Conversation history: {len(self.agent.conversation_history)} entries")
            
            # Check task stats
            pending_tasks = len(self.agent.task_manager.list_tasks(status="pending"))
            in_progress_tasks = len(self.agent.task_manager.list_tasks(status="in_progress"))
            completed_tasks = len(self.agent.task_manager.list_tasks(status="completed"))
            print(f"Tasks: {pending_tasks} pending, {in_progress_tasks} in progress, {completed_tasks} completed")
            
            # Check system resources
            try:
                import psutil
                process = psutil.Process()
                cpu_usage = process.cpu_percent()
                memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
                print(f"CPU usage: {cpu_usage:.1f}%")
                print(f"Memory usage: {memory_usage:.1f} MB")
            except ImportError:
                pass
                
        def do_batch(self, arg):
            """Run a batch of commands from a file: batch <filename>"""
            if not arg:
                print("Usage: batch <filename>")
                return
                
            try:
                with open(arg, 'r') as f:
                    commands = f.readlines()
                    
                print(f"Running {len(commands)} commands from {arg}")
                for cmd in commands:
                    cmd = cmd.strip()
                    if not cmd or cmd.startswith('#'):
                        continue  # Skip empty lines and comments
                        
                    print(f"\n>>> {cmd}")
                    self.onecmd(cmd)
            except Exception as e:
                print(f"Error running batch file: {e}")
                
        def do_save_history(self, arg):
            """Save conversation history to file: save_history <filename>"""
            if not arg:
                arg = f"conversation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
            try:
                with open(arg, 'w') as f:
                    json.dump(self.agent.conversation_history, f, indent=2)
                print(f"Saved conversation history to {arg}")
            except Exception as e:
                print(f"Error saving history: {e}")
                
        def do_search_code(self, arg):
            """Search for code elements: search_code <query>"""
            if not arg:
                print("Usage: search_code <query>")
                return
                
            try:
                results = self.agent.self_transformation.code_manager.find_code(arg)
                if results:
                    print(f"\nFound {len(results)} matching code elements:")
                    for i, result in enumerate(results):
                        print(f"{i+1}. {result['type']} {result['name']} (File: {os.path.basename(result['file'])})")
                        print(f"   Lines {result['line']}-{result['end_line']}")
                else:
                    print("No matching code elements found.")
            except Exception as e:
                print(f"Error searching code: {e}")
                
        def do_search_knowledge(self, arg):
            """Search knowledge base: search_knowledge <query> [collection_id]"""
            args = shlex.split(arg)
            if not args:
                print("Usage: search_knowledge <query> [collection_id]")
                return

            query = args[0]
            collection_id = args[1] if len(args) > 1 else None
            limit = int(args[2]) if len(args) > 2 else 5
            
            try:
                print(f"Searching for: {query}" + (f" in collection: {collection_id}" if collection_id else ""))
                results = asyncio.run(self.agent.knowledge_base.search_knowledge(query, kb_id=collection_id, top_n=limit))
                
                if results:
                    print(f"\nFound {len(results)} matching documents:")
                    for i, result in enumerate(results):
                        similarity = result.get("similarity", 0)
                        similarity_str = f" (score: {similarity:.2f})" if similarity else ""
                        print(f"\n{i+1}. {result.get('title', 'Untitled')}{similarity_str}")
                        print(f"   ID: {result.get('id', 'Unknown')}")
                        
                        # Print a snippet of content
                        content = result.get("content", "")
                        if len(content) > 200:
                            print(f"   Content: {content[:200]}...")
                        else:
                            print(f"   Content: {content}")
                else:
                    print("No matching documents found.")
            except Exception as e:
                print(f"Error searching knowledge base: {e}")
                print(traceback.format_exc())
                
        def do_retrieve_code(self, arg):
            """Retrieve relevant code chunks: retrieve_code <query> [limit]"""
            args = shlex.split(arg)
            if not args:
                print("Usage: retrieve_code <query> [limit]")
                return
                
            query = args[0]
            limit = int(args[1]) if len(args) > 1 else 3
            
            try:
                print(f"Searching for code relevant to: {query}")
                results = asyncio.run(self.agent.retrieve_relevant_code(query, top_n=limit))
                print("\n" + results)
            except Exception as e:
                print(f"Error retrieving code: {e}")
                print(traceback.format_exc())
                
        def do_tasks(self, arg):
            """Manage tasks: tasks [list|create|update|delete|next]"""
            args = shlex.split(arg)
            if not args:
                print("Usage: tasks [list|create|update|delete|next]")
                return
                
            command = args[0].lower()
            
            if command == "list":
                # Parse filters
                status = None
                tag = None
                for i in range(1, len(args)):
                    if args[i].startswith("status="):
                        status = args[i].split("=")[1]
                    elif args[i].startswith("tag="):
                        tag = args[i].split("=")[1]
                
                # Get tasks
                tasks = asyncio.run(self.agent.list_tasks(status=status, tag=tag))
                
                if tasks["success"]:
                    print(f"\nFound {tasks['count']} tasks:")
                    for i, task in enumerate(tasks["tasks"]):
                        status_color = "\033[92m" if task["status"] == "completed" else \
                                      "\033[93m" if task["status"] == "in_progress" else \
                                      "\033[91m" if task["status"] == "failed" else "\033[94m"
                        print(f"{i+1}. {task['title']} ({status_color}{task['status']}\033[0m)")
                        print(f"   ID: {task['task_id']}")
                        print(f"   Priority: {task['priority']}")
                        print(f"   Tags: {', '.join(task['tags']) if task['tags'] else 'None'}")
                        print(f"   Description: {task['description']}")
                        print()
                else:
                    print(f"Error listing tasks: {tasks['error']}")
            
            elif command == "create":
                if len(args) < 3:
                    print("Usage: tasks create <title> <description> [priority=5] [tags=tag1,tag2]")
                    return
                
                title = args[1]
                description = args[2]
                priority = 5
                tags = []
                
                # Parse optional arguments
                for i in range(3, len(args)):
                    if args[i].startswith("priority="):
                        try:
                            priority = int(args[i].split("=")[1])
                        except ValueError:
                            print(f"Invalid priority: {args[i].split('=')[1]}")
                            return
                    elif args[i].startswith("tags="):
                        tags = args[i].split("=")[1].split(",")
                
                # Create task
                result = asyncio.run(self.agent.create_task(
                    title=title,
                    description=description,
                    priority=priority,
                    tags=tags
                ))
                
                if result["success"]:
                    print(f"Created task: {result['title']} (ID: {result['task_id']})")
                else:
                    print(f"Error creating task: {result['error']}")
            
            elif command == "update":
                if len(args) < 3:
                    print("Usage: tasks update <task_id> <status> [result=...]")
                    return
                
                task_id = args[1]
                status = args[2]
                result = None
                
                # Parse result if provided
                for i in range(3, len(args)):
                    if args[i].startswith("result="):
                        result = args[i].split("=", 1)[1]
                
                # Update task
                update_result = asyncio.run(self.agent.update_task(
                    task_id=task_id,
                    status=status,
                    result=result
                ))
                
                if update_result["success"]:
                    print(f"Updated task: {update_result['title']} (Status: {update_result['status']})")
                else:
                    print(f"Error updating task: {update_result['error']}")
            
            elif command == "delete":
                if len(args) < 2:
                    print("Usage: tasks delete <task_id>")
                    return
                
                task_id = args[1]
                
                # Delete task
                if self.agent.task_manager.delete_task(task_id):
                    print(f"Deleted task: {task_id}")
                else:
                    print(f"Error deleting task: Task not found")
            
            elif command == "next":
                # Get next task
                next_task = self.agent.task_manager.get_next_task()
                
                if next_task:
                    print(f"\nNext task to work on:")
                    print(f"Title: {next_task.title}")
                    print(f"ID: {next_task.task_id}")
                    print(f"Priority: {next_task.priority}")
                    print(f"Description: {next_task.description}")
                    print(f"Tags: {', '.join(next_task.tags) if next_task.tags else 'None'}")
                else:
                    print("No pending tasks available.")
            
            else:
                print(f"Unknown command: {command}")
                print("Usage: tasks [list|create|update|delete|next]")
                
        def do_code(self, arg):
            """Read or write code files: code [read|write] <file_path> [content]"""
            args = shlex.split(arg)
            if not args:
                print("Usage: code [read|write] <file_path> [content]")
                return
                
            command = args[0].lower()
            
            if command == "read":
                if len(args) < 2:
                    print("Usage: code read <file_path>")
                    return
                
                file_path = args[1]
                
                # Read file
                result = asyncio.run(self.agent.read_code_file(file_path))
                
                if result["success"]:
                    print(f"\nFile: {result['file_path']}")
                    print(f"Type: {result['file_type']}")
                    print(f"Size: {result['file_size']} bytes")
                    print(f"Modified: {result['modified_time']}")
                    print(f"Lines: {result['lines']}")
                    print("\n--- Content ---\n")
                    
                    # Print with line numbers
                    for i, line in enumerate(result["content"].splitlines()):
                        print(f"{i+1:4d}: {line}")
                else:
                    print(f"Error reading file: {result['error']}")
            
            elif command == "write":
                if len(args) < 2:
                    print("Usage: code write <file_path> [content or '-' for multiline]")
                    return
                
                file_path = args[1]
                
                if len(args) > 2 and args[2] == '-':
                    # Multiline mode
                    print("Enter file content (end with a line containing only '###'):")
                    content_lines = []
                    while True:
                        try:
                            line = input()
                            if line.strip() == '###':
                                break
                            content_lines.append(line)
                        except EOFError:
                            break
                    content = '\n'.join(content_lines)
                elif len(args) > 2:
                    # Content provided as argument
                    content = ' '.join(args[2:])
                else:
                    print("No content provided. Use '-' for multiline input.")
                    return
                
                # Write file
                result = asyncio.run(self.agent.write_code_file(file_path, content))
                
                if result["success"]:
                    print(f"Successfully wrote to {result['file_path']}")
                    print(f"Size: {result['file_size']} bytes")
                    print(f"Lines: {result['lines']}")
                else:
                    print(f"Error writing file: {result['error']}")
            
            else:
                print(f"Unknown command: {command}")
                print("Usage: code [read|write] <file_path> [content]")
                
        def do_exit(self, arg):
            """Exit the CLI: exit"""
            if hasattr(self, "observer"):
                self.observer.stop()
                self.observer.join()
            # Stop reflection worker
            asyncio.run(self.agent.stop_reflections())
            print("\nShutting down agent and cleaning up resources...")
            print("Goodbye!")
            return True
            
        def do_EOF(self, arg):
            """Exit on Ctrl+D"""
            print()  # Add newline
            return self.do_exit(arg)


# ------------------------------------------------------------------------------
# Main Entry Point: Choose RL, CLI, or FastAPI with enhanced command-line options
# ------------------------------------------------------------------------------

async def rl_bootstrap_training(file_path: str, num_episodes: int = 3):
    """Run reinforcement learning training on the agent's code"""
    code_manager = AdvancedCodeManager(file_path)
    env = RLBootstrapEnv(code_manager)
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # Random action for now
            obs, reward, done, info = env.step(action)
            logging.info(f"Episode {episode + 1}, Reward: {reward}, Info: {info}")
    return "RL training completed"

def check_restart_signal():
    """Check if a restart signal exists and handle it"""
    restart_file = "agent_restart_signal.txt"
    if os.path.exists(restart_file):
        logging.info("Restart signal detected. Restarting agent...")
        os.remove(restart_file)
        return True
    return False

def print_usage():
    """Print usage information"""
    print("Usage: python rl_cli.py [OPTIONS]")
    print("\nOptions:")
    print("  --help, -h           Show this help message and exit")
    print("  --api                Run as a FastAPI server")
    print("  --rl [EPISODES]      Run RL bootstrap training with optional episode count")
    print("  --autonomous [MAX]   Start in autonomous self-improvement mode with optional max changes")
    print("  --cli                Start in CLI mode (default)")
    print("  --dashboard          Start directly in dashboard mode")
    print("  --watch DIR          Watch directory for PDFs (default: ./pdfs)")
    print("  --port PORT          Set API server port (default: 8080)")
    print("  --host HOST          Set API server host (default: 0.0.0.0)")
    print("  --no-reflections     Disable background reflections")
    print("  --db PATH            Set database path (default: ./agent.db)")
    print("  --debug              Enable debug logging")
    print("\nExamples:")
    print("  python rl_cli.py --cli                    # Start in CLI mode")
    print("  python rl_cli.py --api --port 9000        # Run API server on port 9000")
    print("  python rl_cli.py --autonomous 10          # Start with autonomous mode, max 10 changes")
    print("  python rl_cli.py --rl 5                   # Run RL training with 5 episodes")
    print("  python rl_cli.py --watch ./documents      # Watch ./documents for PDFs")

if __name__ == "__main__":
    # Check for restart signal
    if check_restart_signal():
        logging.info("Restarting due to code changes...")
        # Force reload modules
        for module in list(sys.modules.keys()):
            if module.startswith('rl_cli'):
                del sys.modules[module]
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Advanced Agent with self-modification capabilities", add_help=False)
    
    # Add arguments
    parser.add_argument('--help', '-h', action='store_true', help='Show help message and exit')
    parser.add_argument('--api', action='store_true', help='Run as a FastAPI server')
    parser.add_argument('--rl', nargs='?', const=3, type=int, help='Run RL bootstrap training with optional episode count')
    parser.add_argument('--autonomous', nargs='?', const=5, type=int, help='Start in autonomous self-improvement mode')
    parser.add_argument('--cli', action='store_true', help='Start in CLI mode (default)')
    parser.add_argument('--dashboard', action='store_true', help='Start directly in dashboard mode')
    parser.add_argument('--watch', type=str, default='./pdfs', help='Watch directory for PDFs')
    parser.add_argument('--port', type=int, default=8080, help='Set API server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Set API server host')
    parser.add_argument('--no-reflections', action='store_true', help='Disable background reflections')
    parser.add_argument('--db', type=str, default='./agent.db', help='Set database path')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    # Parse arguments
    args, unknown = parser.parse_known_args()
    
    # Show help if requested
    if args.help:
        print_usage()
        sys.exit(0)
    
    # Set up logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Run in appropriate mode
    if args.rl is not None:
        asyncio.run(rl_bootstrap_training(__file__, num_episodes=args.rl))
    elif args.api:
        # Run as API server
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        # Default to CLI mode
        cli = AgentCLI()
        
        # Set up PDF watcher with custom directory
        cli.setup_pdf_watcher(args.watch)
        
        # Enable autonomous mode if requested
        if args.autonomous is not None:
            cli.do_enable_autonomous(str(args.autonomous))
        
        # Start or disable reflections
        if args.no_reflections:
            asyncio.run(cli.agent.stop_reflections())
        else:
            asyncio.run(cli.agent.start_reflections())
        
        # Start dashboard directly if requested
        if args.dashboard:
            cli.do_dashboard("")
        
        # Start CLI
        print(f"Starting Agent CLI with {'autonomous mode' if args.autonomous else 'standard mode'}")
        cli.cmdloop()
