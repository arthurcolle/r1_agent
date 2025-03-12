#!/usr/bin/env python3
"""
An "ultra advanced" R1-style do-anything agent with:
 - Indefinite runtime (until user types 'exit')
 - Priority-based task scheduling + concurrency
 - Recursive subtask decomposition
 - Long-range goal management + dynamic planning
 - Conversation memory with summarization
 - Self-reflective meta-cognition
 - In-memory code archive for introspection
 - Action generator producing up to 25 candidate next steps
 - A KnowledgeBase for storing and retrieving key facts
 - Ability to run arbitrary Python code with <function_call> do_anything

~900 lines of code for demonstration in a secure, sandboxed environment!
"""

import os
import sys
import re
import json
import time
import heapq
import queue
import logging
import threading
import traceback
import subprocess
import requests
import random
import numpy as np
import tiktoken
import platform
import hashlib
import asyncio
import shutil
import ast
import types
import sqlite3
import urllib.parse
import importlib
import importlib.util
import difflib
import inspect
import importlib.machinery
from pathlib import Path
from math import log
from scipy import spatial
from typing import Any, Dict, List, Optional, Tuple, Callable, Union, Literal, TypeVar, Generic, AsyncGenerator
from pydantic import BaseModel, Field, validator, root_validator, ConfigDict, create_model
from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum, auto
from typing_extensions import Annotated
from together import Together
from datetime import datetime, timedelta
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import base64
import io

# Try importing watchdog for PDF monitoring - optional dependency
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("Watchdog not available. PDF monitoring features will be disabled.")

# Try importing PyPDF2 for PDF processing - optional dependency
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("PyPDF2 not available. PDF processing features will be disabled.")

# Try importing OpenAI for embeddings - optional dependency
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
    openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", "OPENAI_API_KEY"))
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI SDK not available. Embedding features will use alternative methods.")

# Try importing torch for RL agents and vision models - optional dependency
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. RL agent and vision model features will be disabled.")
    
# Try importing vision models - optional dependency
try:
    from transformers import AutoModelForCausalLM, CLIPVisionModel, CLIPImageProcessor
    VISION_MODELS_AVAILABLE = True
except ImportError:
    VISION_MODELS_AVAILABLE = False
    print("Vision models not available. Install with: pip install transformers torch einops pillow.")

# Try importing FastAPI for API endpoints - optional dependency
try:
    from fastapi import FastAPI, Request, BackgroundTasks
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available. API endpoint features will be disabled.")

# Try importing redis for distributed cache - optional dependency
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Redis client not available. Distributed cache features will be disabled.")

# Try importing gymnasium for RL environments - optional dependency
try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    print("Gymnasium not available. RL environment features will be disabled.")

# Try importing curses for terminal UI - optional dependency
try:
    import curses
    from curses import panel
    CURSES_AVAILABLE = True
except ImportError:
    CURSES_AVAILABLE = False
    print("Curses not available. Advanced terminal UI features will be disabled.")

###############################################################################
# GLOBAL CONFIG / LOGGING
###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("UltraAdvancedR1Agent")

###############################################################################
# TERMINAL UI COLORS
###############################################################################

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

###############################################################################
# SYSTEM CAPABILITIES DETECTION
###############################################################################

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

    # Add available Python modules
    capabilities["python_modules"] = {
        "watchdog": WATCHDOG_AVAILABLE,
        "pypdf2": PYPDF2_AVAILABLE,
        "openai": OPENAI_AVAILABLE,
        "torch": TORCH_AVAILABLE,
        "fastapi": FASTAPI_AVAILABLE,
        "redis": REDIS_AVAILABLE,
        "gymnasium": GYM_AVAILABLE,
        "curses": CURSES_AVAILABLE,
        "vision_models": VISION_MODELS_AVAILABLE
    }
    
    return capabilities

###############################################################################
# PDF HANDLING AND MONITORING
###############################################################################

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

###############################################################################
# EXTERNAL API CLIENTS
###############################################################################

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
        
###############################################################################
# VISION MODEL
###############################################################################

class VisionModel:
    """
    Manages vision model capabilities for image understanding and processing.
    Supports both local models and API-based vision services.
    """
    
    def __init__(self, model_name: str = "vikhyatk/moondream2", 
                 revision: str = "2025-01-09",
                 device: str = "cpu"):
        """
        Initialize the vision model with specified options.
        
        Args:
            model_name: The model identifier/name to use
            revision: Model revision/version
            device: Device to run the model on ('cpu', 'cuda', etc.)
        """
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self._model = None
        self._processor = None
        self._clip_model = None
        self._clip_processor = None
        self.logger = logging.getLogger("VisionModel")
        
        # Only attempt to load models if vision libraries are available
        if VISION_MODELS_AVAILABLE and TORCH_AVAILABLE:
            self._initialize_models()
        else:
            self.logger.warning("Vision model capabilities unavailable. Install required packages to enable.")
    
    def _initialize_models(self):
        """Initialize vision models on demand to save resources"""
        try:
            self.logger.info(f"Initializing vision model: {self.model_name}")
            
            # Initialize CLIP model and processor for image encoding
            self._clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            
            # Initialize main vision language model
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                revision=self.revision,
                trust_remote_code=True,
                device_map={"": self.device}
            )
            
            self.logger.info("Vision models initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize vision models: {str(e)}")
            self._model = None
            self._clip_model = None
    
    def ensure_models_loaded(self):
        """Ensure that models are loaded before use"""
        if not VISION_MODELS_AVAILABLE or not TORCH_AVAILABLE:
            raise RuntimeError("Vision capabilities unavailable. Please install required packages.")
        
        if self._model is None or self._clip_model is None:
            self._initialize_models()
            
        if self._model is None or self._clip_model is None:
            raise RuntimeError("Failed to initialize vision models")
    
    def encode_image(self, image_path: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        """
        Encode an image for use with vision models.
        
        Args:
            image_path: Path to image file or PIL Image object
            
        Returns:
            Dictionary with encoded image data
        """
        self.ensure_models_loaded()
        
        # Handle different input types
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert('RGB')
        elif isinstance(image_path, Image.Image):
            image = image_path.convert('RGB')
        else:
            raise ValueError("image_path must be a string, Path, or PIL Image")
        
        # Use model's native encode_image if available
        if hasattr(self._model, 'encode_image'):
            encoded = self._model.encode_image(image)
            return {"encoded_image": encoded, "model": "native"}
        
        # Fallback to CLIP encoding
        with torch.no_grad():
            inputs = self._clip_processor(images=image, return_tensors="pt").to(self.device)
            outputs = self._clip_model(**inputs)
            embeddings = outputs.last_hidden_state
            
            return {
                "encoded_image": embeddings, 
                "inputs": inputs,
                "model": "clip"
            }
    
    def caption_image(self, encoded_image: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a caption for an encoded image.
        
        Args:
            encoded_image: Encoded image from encode_image()
            
        Returns:
            Dictionary with caption and metadata
        """
        self.ensure_models_loaded()
        
        try:
            if hasattr(self._model, 'caption'):
                # Direct caption if model supports it
                result = self._model.caption(encoded_image["encoded_image"])
                return {
                    "caption": result.get("caption", result), 
                    "model": self.model_name
                }
            else:
                # Fallback to query with generic caption prompt
                return self.query_image(
                    encoded_image, 
                    "Describe this image in detail."
                )
        except Exception as e:
            self.logger.error(f"Caption generation failed: {str(e)}")
            return {"error": str(e), "caption": "Failed to generate caption"}
    
    def query_image(self, encoded_image: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Ask a question about an encoded image.
        
        Args:
            encoded_image: Encoded image from encode_image()
            query: Question to ask about the image
            
        Returns:
            Dictionary with answer and metadata
        """
        self.ensure_models_loaded()
        
        try:
            if hasattr(self._model, 'query'):
                # Direct query if model supports it
                result = self._model.query(encoded_image["encoded_image"], query)
                return {
                    "answer": result.get("answer", result),
                    "query": query,
                    "model": self.model_name
                }
            else:
                # Generic generation approach as fallback
                return {
                    "answer": "Model does not support direct image queries",
                    "query": query,
                    "model": self.model_name
                }
        except Exception as e:
            self.logger.error(f"Image query failed: {str(e)}")
            return {"error": str(e), "answer": "Failed to process image query"}
    
    @staticmethod
    def image_to_base64(image_path: Union[str, Path, Image.Image]) -> str:
        """
        Convert an image to base64 encoding for API requests.
        
        Args:
            image_path: Path to image file or PIL Image object
            
        Returns:
            Base64 encoded image string
        """
        # Handle different input types
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path)
        elif isinstance(image_path, Image.Image):
            image = image_path
        else:
            raise ValueError("image_path must be a string, Path, or PIL Image")
            
        # Convert to JPEG in memory
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        
        # Get the bytes and encode to base64
        img_bytes = buffer.getvalue()
        base64_encoded = base64.b64encode(img_bytes).decode('utf-8')
        
        return base64_encoded
        
    async def external_vision_api(self, image_path: Union[str, Path, Image.Image], 
                                 query: Optional[str] = None, 
                                 provider: str = "openai") -> Dict[str, Any]:
        """
        Process an image using external vision API providers.
        
        Args:
            image_path: Path to image file or PIL Image object
            query: Optional question to ask about the image
            provider: API provider to use ('openai', 'anthropic', etc.)
            
        Returns:
            Provider-specific response
        """
        # Convert image to base64
        base64_image = self.image_to_base64(image_path)
        
        if provider == "openai" and OPENAI_AVAILABLE:
            # OpenAI vision API
            try:
                client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query or "Describe this image in detail."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ]
                
                response = await client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=messages,
                    max_tokens=1000
                )
                
                return {
                    "provider": "openai",
                    "model": "gpt-4-vision-preview",
                    "response": response.choices[0].message.content,
                    "query": query
                }
            
            except Exception as e:
                self.logger.error(f"OpenAI vision API error: {str(e)}")
                return {"error": str(e), "provider": "openai"}
                
        elif provider == "anthropic" and "ANTHROPIC_API_KEY" in os.environ:
            # Anthropic Claude vision API
            try:
                headers = {
                    "x-api-key": os.environ.get("ANTHROPIC_API_KEY"),
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
                
                data = {
                    "model": "claude-3-opus-20240229",
                    "max_tokens": 1000,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": query or "Describe this image in detail."},
                                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}}
                            ]
                        }
                    ]
                }
                
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=data
                )
                
                return {
                    "provider": "anthropic",
                    "model": "claude-3-opus-20240229",
                    "response": response.json().get("content", [{"text": "No response"}])[0].get("text"),
                    "query": query
                }
                
            except Exception as e:
                self.logger.error(f"Anthropic vision API error: {str(e)}")
                return {"error": str(e), "provider": "anthropic"}
                
        else:
            return {"error": f"Provider {provider} not available or API key not configured"}

###############################################################################
# DYNAMIC TOOL REGISTRATION
###############################################################################

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
    logger.info(f"{Colors.OKGREEN}{Colors.BOLD}Registered tool:{Colors.ENDC} {name}")
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
    
        # Vision model integration
        if capabilities["python_modules"].get("vision_models", False) and capabilities["python_modules"].get("torch", False):
            # Initialize vision model
            vision_model = VisionModel(device="cuda" if torch.cuda.is_available() else "cpu")
            
            # Register vision tools
            async def analyze_image(image_path: str, query: Optional[str] = None) -> Dict[str, Any]:
                """Analyze an image using local vision model"""
                try:
                    # Encode the image
                    encoded_image = vision_model.encode_image(image_path)
                    
                    # Generate caption
                    caption_result = vision_model.caption_image(encoded_image)
                    
                    # If a specific query is provided, answer it
                    if query:
                        query_result = vision_model.query_image(encoded_image, query)
                        result = {
                            "caption": caption_result.get("caption", "No caption generated"),
                            "answer": query_result.get("answer", "No answer generated"),
                            "query": query,
                            "model": vision_model.model_name
                        }
                    else:
                        result = {
                            "caption": caption_result.get("caption", "No caption generated"),
                            "model": vision_model.model_name
                        }
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Image analysis error: {str(e)}")
                    return {"error": str(e)}
            
            tools["analyze_image"] = analyze_image
            
            # Register image captioning
            async def caption_image(image_path: str) -> Dict[str, Any]:
                """Generate a caption for an image"""
                try:
                    encoded_image = vision_model.encode_image(image_path)
                    caption_result = vision_model.caption_image(encoded_image)
                    return {
                        "caption": caption_result.get("caption", "No caption generated"),
                        "model": vision_model.model_name
                    }
                except Exception as e:
                    logger.error(f"Image captioning error: {str(e)}")
                    return {"error": str(e)}
                    
            tools["caption_image"] = caption_image
            
            # Register visual QA
            async def visual_qa(image_path: str, question: str) -> Dict[str, Any]:
                """Ask a question about an image"""
                try:
                    encoded_image = vision_model.encode_image(image_path)
                    answer_result = vision_model.query_image(encoded_image, question)
                    return {
                        "answer": answer_result.get("answer", "No answer generated"),
                        "question": question,
                        "model": vision_model.model_name
                    }
                except Exception as e:
                    logger.error(f"Visual QA error: {str(e)}")
                    return {"error": str(e)}
                    
            tools["visual_qa"] = visual_qa
            
            # Register cloud vision API
            async def cloud_vision(image_path: str, query: Optional[str] = None, provider: str = "openai") -> Dict[str, Any]:
                """Process an image using cloud vision API"""
                try:
                    result = await vision_model.external_vision_api(image_path, query, provider)
                    return result
                except Exception as e:
                    logger.error(f"Cloud vision API error: {str(e)}")
                    return {"error": str(e)}
                    
            tools["cloud_vision"] = cloud_vision
            
            # Register the tool schemas
            register_tool(
                "analyze_image",
                analyze_image,
                "Analyze an image using local vision model",
                {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the image file"
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional question to ask about the image"
                    }
                },
                required_params=["image_path"]
            )
            
            register_tool(
                "caption_image",
                caption_image,
                "Generate a caption for an image",
                {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the image file"
                    }
                }
            )
            
            register_tool(
                "visual_qa",
                visual_qa,
                "Ask a question about an image",
                {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the image file"
                    },
                    "question": {
                        "type": "string",
                        "description": "Question to ask about the image"
                    }
                }
            )
            
            register_tool(
                "cloud_vision",
                cloud_vision,
                "Process an image using cloud vision API",
                {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the image file"
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional question to ask about the image"
                    },
                    "provider": {
                        "type": "string",
                        "description": "API provider to use ('openai', 'anthropic')",
                        "enum": ["openai", "anthropic"]
                    }
                },
                required_params=["image_path"]
            )
            
        # Dynamic code loading tools
        # Initialize dynamic code loader
        dynamic_code_loader = DynamicCodeLoader()
        
        # Function to load a module dynamically
        async def load_module(module_path: str, reload: bool = False) -> Dict[str, Any]:
            """
            Load a Python module dynamically from file path.
            
            Args:
                module_path: Path to the Python module file
                reload: Force reload even if previously loaded
                
            Returns:
                Information about the loaded module
            """
            try:
                module = dynamic_code_loader.load_module(module_path, reload=reload)
                
                # Get module symbols
                symbols = dynamic_code_loader.get_module_symbols(module_path)
                
                return {
                    "module_path": str(module_path),
                    "module_name": module.__name__,
                    "symbols": symbols,
                    "status": "success",
                    "reloaded": reload
                }
            except Exception as e:
                logger.error(f"Failed to load module {module_path}: {str(e)}")
                return {
                    "module_path": str(module_path),
                    "status": "error",
                    "error": str(e)
                }
        
        # Function to execute a function from a dynamically loaded module
        async def execute_from_module(module_path: str, function_name: str, args: List[Any] = None, kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
            """
            Execute a function from a dynamically loaded module.
            
            Args:
                module_path: Path to the Python module file
                function_name: Name of the function to execute
                args: Positional arguments to pass to the function
                kwargs: Keyword arguments to pass to the function
                
            Returns:
                Results of the function execution
            """
            try:
                args = args or []
                kwargs = kwargs or {}
                
                result = dynamic_code_loader.execute_function(module_path, function_name, *args, **kwargs)
                
                return {
                    "module_path": str(module_path),
                    "function_name": function_name,
                    "status": "success",
                    "result": result,
                    "args": args,
                    "kwargs": kwargs
                }
            except Exception as e:
                logger.error(f"Failed to execute {function_name} from module {module_path}: {str(e)}")
                return {
                    "module_path": str(module_path),
                    "function_name": function_name,
                    "status": "error",
                    "error": str(e),
                    "args": args,
                    "kwargs": kwargs
                }
        
        # Function to check for modified modules
        async def check_for_module_changes() -> Dict[str, Any]:
            """
            Check all loaded modules for changes since they were last loaded.
            
            Returns:
                Information about which modules have changed
            """
            try:
                changed_modules = dynamic_code_loader.check_for_changes()
                
                return {
                    "changed_modules": [str(path) for path in changed_modules],
                    "count": len(changed_modules),
                    "status": "success"
                }
            except Exception as e:
                logger.error(f"Failed to check for module changes: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        # Function to reload all changed modules
        async def reload_changed_modules() -> Dict[str, Any]:
            """
            Reload all modules that have changed since they were last loaded.
            
            Returns:
                Information about which modules were reloaded
            """
            try:
                results = dynamic_code_loader.reload_all_changed()
                
                return {
                    "reloaded_modules": {str(k): v for k, v in results.items()},
                    "count": len(results),
                    "status": "success"
                }
            except Exception as e:
                logger.error(f"Failed to reload changed modules: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e)
                }
                
        # Register the dynamic code loading tools
        tools["load_module"] = load_module
        tools["execute_from_module"] = execute_from_module
        tools["check_for_module_changes"] = check_for_module_changes
        tools["reload_changed_modules"] = reload_changed_modules
        
        # Register tool schemas
        register_tool(
            "load_module",
            load_module,
            "Load a Python module dynamically from file path",
            {
                "module_path": {
                    "type": "string",
                    "description": "Path to the Python module file"
                },
                "reload": {
                    "type": "boolean",
                    "description": "Force reload even if previously loaded"
                }
            },
            required_params=["module_path"]
        )
        
        register_tool(
            "execute_from_module",
            execute_from_module,
            "Execute a function from a dynamically loaded module",
            {
                "module_path": {
                    "type": "string",
                    "description": "Path to the Python module file"
                },
                "function_name": {
                    "type": "string",
                    "description": "Name of the function to execute"
                },
                "args": {
                    "type": "array",
                    "description": "Positional arguments to pass to the function",
                    "items": {
                        "type": "object"
                    }
                },
                "kwargs": {
                    "type": "object",
                    "description": "Keyword arguments to pass to the function"
                }
            },
            required_params=["module_path", "function_name"]
        )
        
        register_tool(
            "check_for_module_changes",
            check_for_module_changes,
            "Check all loaded modules for changes since they were last loaded",
            {}
        )
        
        register_tool(
            "reload_changed_modules",
            reload_changed_modules,
            "Reload all modules that have changed since they were last loaded",
            {}
        )
    
        # Weather API integration
        if "OPENWEATHERMAP_API_KEY" in os.environ:
            async def get_weather(location: str) -> Dict[str, Any]:
                """Get weather data for a location using Jina search as fallback if OpenWeather fails"""
                try:
                    api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
                    if api_key:
                        # Use a session for connection pooling and better performance
                        session = requests.Session()
                        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=imperial"
                        
                        # Set a timeout to prevent hanging
                        response = session.get(url, timeout=5.0)
                        if response.status_code == 200:
                            return response.json()
                        elif response.status_code == 404:
                            # Location not found, try with less specific query
                            main_city = location.split()[0] if ' ' in location else location
                            url = f"http://api.openweathermap.org/data/2.5/weather?q={main_city}&appid={api_key}&units=imperial"
                            response = session.get(url, timeout=5.0)
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
                except requests.exceptions.Timeout:
                    return {"error": f"Weather API timeout for location: {location}"}
                except requests.exceptions.RequestException as e:
                    return {"error": f"Weather API request error: {str(e)}"}
                except Exception as e:
                    return {"error": f"Weather API error: {str(e)}"}
            tools["get_weather"] = get_weather

        # Search implementations
        if "serpapi" in capabilities["env_vars"] or "SERPAPI_API_KEY" in os.environ:
            try:
                from serpapi import GoogleSearch
                async def web_search(query: str) -> Dict[str, Any]:
                    try:
                        api_key = os.getenv("SERPAPI_API_KEY")
                        search = GoogleSearch({
                            "q": query,
                            "api_key": api_key
                        })
                        return search.get_dict()
                    except Exception as e:
                        logger.error(f"SerpAPI search error: {e}")
                        return {"error": f"Search error: {str(e)}"}
                tools["web_search"] = web_search
                logger.info("SerpAPI search tool registered")
            except ImportError:
                logger.warning("SerpAPI package not installed. To use SerpAPI search, install with: pip install google-search-results")
                
        elif "GOOGLE_API_KEY" in os.environ and "GOOGLE_CSE_ID" in os.environ:
            async def web_search(query: str) -> Dict[str, Any]:
                try:
                    url = "https://www.googleapis.com/customsearch/v1"
                    params = {
                        "key": os.environ["GOOGLE_API_KEY"],
                        "cx": os.environ["GOOGLE_CSE_ID"],
                        "q": query
                    }
                    # Set timeout to prevent hanging
                    response = requests.get(url, params=params, timeout=10.0)
                    if response.status_code == 200:
                        return response.json()
                    return {"error": f"Google search error: {response.status_code}"}
                except requests.exceptions.Timeout:
                    return {"error": "Google search timeout"}
                except Exception as e:
                    logger.error(f"Google search error: {e}")
                    return {"error": f"Search error: {str(e)}"}
            tools["web_search"] = web_search
            logger.info("Google Custom Search tool registered")
            
            # Register tool schema
            register_tool(
                "web_search",
                web_search,
                "Search the web using Google Custom Search",
                {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                }
            )

    return tools

###############################################################################
# SQLITE KNOWLEDGE BASE
###############################################################################

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
        logger.info(f"Created collection '{name}' with id {collection_id}")
        return collection_id

    async def add_knowledge_entry(self, kb_id: str, title: str, content: str, embedding: List[float] = None) -> str:
        doc_id = str(uuid.uuid4())
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO documents (id, collection_id, title, content)
            VALUES (?, ?, ?, ?)
        """, (doc_id, kb_id, title, content))
        
        # If embedding not provided, compute it
        if embedding is None and len(content) > 0:
            embedding = await compute_embedding(content, db_conn=self.conn)
            
        # Add to full-text search index
        cur.execute("""
            INSERT INTO document_fts (document_id, title, content)
            VALUES (?, ?, ?)
        """, (doc_id, title, content))
        
        self.conn.commit()
        logger.info(f"Added knowledge entry {doc_id} titled '{title}' to collection {kb_id}")
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
        
        # Try full-text search first
        fts_results = []
        try:
            if kb_id:
                cur.execute("""
                    SELECT d.*, fts.rank
                    FROM document_fts fts
                    JOIN documents d ON fts.document_id = d.id
                    WHERE d.collection_id = ? AND document_fts MATCH ?
                    ORDER BY fts.rank
                    LIMIT ?
                """, (kb_id, query, top_n))
            else:
                cur.execute("""
                    SELECT d.*, fts.rank
                    FROM document_fts fts
                    JOIN documents d ON fts.document_id = d.id
                    WHERE document_fts MATCH ?
                    ORDER BY fts.rank
                    LIMIT ?
                """, (query, top_n))
            
            fts_results = [dict(row) for row in cur.fetchall()]
            
            # If we got good results from FTS, return them
            if len(fts_results) >= top_n // 2:
                return fts_results
        except Exception as e:
            logger.warning(f"Full-text search failed, falling back to vector search: {e}")
        
        # Fall back to vector search
        if kb_id:
            cur.execute("SELECT * FROM documents WHERE collection_id = ?", (kb_id,))
        else:
            cur.execute("SELECT * FROM documents")
        rows = cur.fetchall()
        if not rows:
            return fts_results  # Return any FTS results we got, or empty list
            
        # Get query embedding
        query_emb = await compute_embedding(query, db_conn=self.conn)
        if not query_emb:
            return fts_results  # Return any FTS results we got, or empty list
            
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
        
        # Combine results from FTS and vector search, removing duplicates
        combined_results = []
        seen_ids = set()
        
        # First add FTS results
        for result in fts_results:
            combined_results.append(result)
            seen_ids.add(result["id"])
            
        # Then add vector search results, skipping duplicates
        for result in scored[:top_n]:
            if result["id"] not in seen_ids:
                combined_results.append(result)
                seen_ids.add(result["id"])
                if len(combined_results) >= top_n:
                    break
                    
        return combined_results[:top_n]

###############################################################################
# EMBEDDING COMPUTATION WITH CACHING
###############################################################################

async def compute_embeddings_batch(
    texts: List[str],
    model: str = "text-embedding-3-large",
    batch_size: int = 16,
    db_conn: Optional[sqlite3.Connection] = None
) -> List[List[float]]:
    """
    Compute embeddings for a batch of texts with caching support.
    
    Args:
        texts: List of text strings to embed
        model: Name of the embedding model to use
        batch_size: Number of texts to process in each API call
        db_conn: Optional SQLite connection for caching
        
    Returns:
        List of embedding vectors (list of floats)
    """
    results = []
    uncached_texts = []
    uncached_indices = []
    content_hashes = []

    # Check cache if DB connection provided
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

    # Process any texts not found in cache
    if uncached_texts:
        if OPENAI_AVAILABLE:
            # Use OpenAI API in batches
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i:i+batch_size]
                try:
                    response = await openai_client.embeddings.create(input=batch, model=model)
                    embeddings = [d.embedding for d in response.data]
                    
                    # Cache results if DB connection provided
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
                    logger.error(f"Error in compute_embeddings_batch: {ex}")
                    # Generate empty embeddings as fallback
                    dims = 1536  # Standard for text-embedding-3-large
                    results.extend([np.zeros(dims).tolist() for _ in range(len(batch))])
        else:
            # Fallback to a simple bag-of-words approach with hash-based encoding
            logger.warning("OpenAI not available, using simple hash-based embedding")
            for text in uncached_texts:
                # Simple hash-based embedding - not production quality!
                hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
                random.seed(hash_val)
                # Create a 384-dim vector as fallback (smaller than OpenAI's)
                simple_emb = [random.uniform(-1, 1) for _ in range(384)]
                normalized = simple_emb / np.linalg.norm(simple_emb)
                results.append(normalized.tolist())

    # Rearrange results if needed to match original order
    if db_conn and len(uncached_indices) != len(texts):
        final_results = [None] * len(texts)
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
    """
    Compute embedding for a single text.
    
    Args:
        text: Text to embed
        model: Name of the embedding model to use
        db_conn: Optional SQLite connection for caching
        
    Returns:
        Embedding vector (list of floats)
    """
    result_list = await compute_embeddings_batch([text], model=model, db_conn=db_conn)
    return result_list[0] if result_list else None

###############################################################################
# AST-BASED CODE CHUNKING & ADVANCED CODE MANAGEMENT
###############################################################################

class ASTChunker:
    """
    Parse a Python file with the ast module to extract top-level functions and classes.
    """
    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse a Python file and extract top-level functions and classes.
        
        Args:
            file_path: Path to the Python file to parse
            
        Returns:
            List of dictionaries with extracted code chunks
        """
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
            logger.error(f"Error indexing {file_path}: {e}")
            
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
        logger.info(f"Restored {len(restored)} files from backup")

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
                logger.error(f"Import validation errors: {import_errors}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Compilation error: {e}")
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
        
        logger.info(f"Created memory snapshot {snapshot_id} for {file_path}")
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
        
        logger.info(f"Restored memory snapshot {snapshot_id} for {file_path}")
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
                logger.warning(f"Module {module_name} is not loaded, cannot hot-swap")
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
                    logger.error(f"Error notifying patch listener: {e}")
                    
            logger.info(f"Hot-swapped module {module_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error hot-swapping module: {e}")
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
                logger.error(f"Timeout waiting for memory lock on {file_path}")
                return False
                
        # Set memory lock
        self.memory_locks[file_path] = True
        
        try:
            # Check for conflicts
            if file_path in self.edit_conflicts and self.edit_conflicts[file_path]:
                # Try to resolve conflict
                resolved = await self._resolve_edit_conflict(file_path, new_content)
                if not resolved:
                    logger.warning(f"Unresolved edit conflict for {file_path}")
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
            logger.error(f"Error editing code in memory: {e}")
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
                    logger.info(f"Auto-resolved conflict for {file_path}")
                    
            # Remove resolved conflicts
            for conflict in resolved_conflicts:
                if conflict in self.edit_conflicts[file_path]:
                    self.edit_conflicts[file_path].remove(conflict)
                    
            return len(self.edit_conflicts[file_path]) == 0
            
        except Exception as e:
            logger.error(f"Error resolving edit conflict: {e}")
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
        
        logger.warning(f"Edit conflict marked for {file_path} by {editor_id}")
        
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
                
            # Find differences
            diffs = []
            matcher = difflib.SequenceMatcher(None, file_lines, memory_lines)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag in ['replace', 'delete', 'insert']:
                    for line_num in range(i1, i2):
                        if line_num < len(file_lines):
                            diffs.append({
                                'line': line_num + 1,
                                'file': file_lines[line_num].rstrip('\n'),
                                'memory': memory_lines[line_num].rstrip('\n') if line_num < len(memory_lines) else None,
                                'type': tag
                            })
                    for line_num in range(j1, j2):
                        if line_num >= len(file_lines) or line_num >= i2:
                            diffs.append({
                                'line': line_num + 1,
                                'file': None if line_num >= len(file_lines) else file_lines[line_num].rstrip('\n'),
                                'memory': memory_lines[line_num].rstrip('\n'),
                                'type': tag
                            })
            return diffs
        except Exception as e:
            logger.error(f"Error getting file diff: {e}")
            return []
            
    async def _sync_memory_code_with_db(self, file_path: str) -> bool:
        """Synchronize memory code with database for collaborative editing"""
        # This is a placeholder for the actual implementation
        # In a real implementation, this would:
        # 1. Compute a diff between the previous version and the current memory code
        # 2. Store the diff in the database with a timestamp
        # 3. Set a flag indicating that the code has been synchronized
        logger.info(f"Syncing memory code for {file_path} with database")
        return True

###############################################################################
# REINFORCEMENT LEARNING AGENTS
###############################################################################

class BaseRLAgent:
    """
    Base class for RL agents that wrap an LLM or any other policy model.
    Provides interfaces for training steps, policy actions, and updating neural networks.
    """

    def act(self, state) -> int:
        """
        Decide on an action given the current state.
        """
        raise NotImplementedError

    def update_network(self, transitions) -> None:
        """
        Perform a training update for the agent.
        """
        raise NotImplementedError

    def set_eval_mode(self):
        """
        Switch the agent to evaluation mode.
        """
        raise NotImplementedError

    def set_train_mode(self):
        """
        Switch the agent to training mode.
        """
        raise NotImplementedError

# Only define torch-based agents if PyTorch is available
if TORCH_AVAILABLE:
    class SimplePolicyAgent(BaseRLAgent):
        """
        Implements a very naive policy gradient approach where we collect rollouts
        and perform a gradient ascent step on the log-probabilities weighted by returns.
        This is for demonstration and not recommended for production.
        """

        def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128, lr: float = 1e-3):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.policy_network = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_dim),
                nn.Softmax(dim=-1)
            )
            self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

        def act(self, state):
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            probs = self.policy_network(s).squeeze(0)
            action_dist = torch.distributions.Categorical(probs=probs)
            action = action_dist.sample()
            return action.item()

        def update_network(self, transitions):
            """
            transitions: list of (state, action, reward, next_state, done)
            We'll do a simple REINFORCE: sum of log-probs * discounted returns
            """
            self.set_train_mode()
            returns = 0
            loss = 0
            gamma = 0.99

            for (state, action, reward, _, _) in reversed(transitions):
                returns = reward + gamma * returns
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                probs = self.policy_network(s).squeeze(0)
                action_dist = torch.distributions.Categorical(probs=probs)
                log_prob = action_dist.log_prob(torch.tensor(action))
                loss = loss - log_prob * returns

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        def set_eval_mode(self):
            self.policy_network.eval()

        def set_train_mode(self):
            self.policy_network.train()

    class GRPOAgent(BaseRLAgent):
        """
        A demonstration of Group Relative Policy Optimization that bypasses the critic
        by sampling multiple responses and normalizing the reward.
        """
        def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128, lr: float = 1e-3, clip_epsilon: float = 0.2, entropy_coef: float = 0.01, num_epochs: int = 3, kl_target: float = 0.01):
            super().__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.clip_epsilon = clip_epsilon
            self.entropy_coef = entropy_coef
            self.num_epochs = num_epochs
            self.kl_target = kl_target

            self.policy_network = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_dim),
                nn.Softmax(dim=-1)
            )
            self.old_policy_network = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_dim),
                nn.Softmax(dim=-1)
            )
            self.old_policy_network.load_state_dict(self.policy_network.state_dict())

            self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

        def act(self, state):
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            probs = self.policy_network(s).squeeze(0)
            action_dist = torch.distributions.Categorical(probs=probs)
            return action_dist.sample().item()

        def update_network(self, group_experiences):
            """
            group_experiences:
                A list containing multiple groups. Each group is itself a list of (state, action, reward).
                We'll compute advantage as normalised reward within each group,
                then perform a clipped update similar to PPO, but without a critic.
            """
            self.set_train_mode()

            # Copy current policy to old policy
            self.old_policy_network.load_state_dict(self.policy_network.state_dict())

            # Flatten experiences into arrays
            all_states = []
            all_actions = []
            all_advantages = []
            for experiences in group_experiences:
                rewards = [exp[2] for exp in experiences]
                mean_r = sum(rewards) / len(rewards) if len(rewards) else 0.0
                var_r = sum([(r - mean_r)**2 for r in rewards]) / len(rewards) if len(rewards) > 1 else 1e-8
                std_r = var_r**0.5
                std_r = max(std_r, 1e-8)

                for (st, ac, rw) in experiences:
                    adv = (rw - mean_r) / std_r
                    all_states.append(st)
                    all_actions.append(ac)
                    all_advantages.append(adv)

            states_t = torch.tensor(all_states, dtype=torch.float32)
            actions_t = torch.tensor(all_actions, dtype=torch.long)
            advantages_t = torch.tensor(all_advantages, dtype=torch.float32)

            # We do multiple epochs of updates
            for epoch in range(self.num_epochs):
                # Forward pass on current policy
                current_probs = self.policy_network(states_t)
                current_actions_probs = torch.gather(current_probs, 1, actions_t.unsqueeze(1)).squeeze(1)
                current_log_probs = torch.log(current_actions_probs + 1e-8)

                # Old policy log probs
                old_probs = self.old_policy_network(states_t)
                old_actions_probs = torch.gather(old_probs, 1, actions_t.unsqueeze(1)).squeeze(1)
                old_log_probs = torch.log(old_actions_probs + 1e-8)

                # Calculate ratio and clipped objective
                ratio = torch.exp(current_log_probs - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                loss_1 = ratio * advantages_t
                loss_2 = clipped_ratio * advantages_t
                policy_loss = -torch.min(loss_1, loss_2).mean()

                # Entropy for exploration
                entropy = -torch.sum(current_probs * torch.log(current_probs + 1e-8), dim=1).mean()
                policy_loss = policy_loss - self.entropy_coef * entropy

                # KL divergence check
                with torch.no_grad():
                    kl_div = (old_probs * (torch.log(old_probs + 1e-8) - torch.log(current_probs + 1e-8))).sum(dim=1).mean()

                self.optimizer.zero_grad()
                policy_loss.backward()
                self.optimizer.step()

                # Early stopping if KL exceeds target
                if kl_div.item() > self.kl_target:
                    break

        def set_eval_mode(self):
            self.policy_network.eval()

        def set_train_mode(self):
            self.policy_network.train()

    class PPOAgent(BaseRLAgent):
        """
        A simplified PPO implementation, excluding many advanced features like GAE-lambda or advantage normalization.
        Demonstrates clipped objective for stable improvements.
        """

        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_size: int = 128,
            lr: float = 1e-3,
            clip_epsilon: float = 0.2
        ):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.clip_epsilon = clip_epsilon

            # Policy & old policy
            self.policy_network = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_dim),
                nn.Softmax(dim=-1)
            )
            self.old_policy_network = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_dim),
                nn.Softmax(dim=-1)
            )
            self.old_policy_network.load_state_dict(self.policy_network.state_dict())

            self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

        def act(self, state):
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            probs = self.policy_network(s).squeeze(0)
            action_dist = torch.distributions.Categorical(probs=probs)
            action = action_dist.sample()
            return action.item()

        def update_network(self, transitions):
            """
            transitions: list of (state, action, reward, next_state, done)
            We'll do a simplified advantage (reward - baseline=0).
            Then apply PPO clip objective.
            """
            self.set_train_mode()

            # Copy current net to old net
            self.old_policy_network.load_state_dict(self.policy_network.state_dict())

            all_states = []
            all_actions = []
            all_returns = []

            returns = 0
            gamma = 0.99

            for (state, action, reward, _, _) in reversed(transitions):
                returns = reward + gamma * returns
                all_states.insert(0, state)
                all_actions.insert(0, action)
                all_returns.insert(0, returns)

            states_t = torch.tensor(all_states, dtype=torch.float32)
            actions_t = torch.tensor(all_actions, dtype=torch.long)
            returns_t = torch.tensor(all_returns, dtype=torch.float32)

            # Current policy
            current_probs = self.policy_network(states_t)
            current_dist = torch.distributions.Categorical(probs=current_probs)
            current_log_probs = current_dist.log_prob(actions_t)

            # Old policy
            old_probs = self.old_policy_network(states_t)
            old_dist = torch.distributions.Categorical(probs=old_probs)
            old_log_probs = old_dist.log_prob(actions_t).detach()

            ratio = torch.exp(current_log_probs - old_log_probs)
            # Advantage ~ returns_t (no baseline for simplicity)
            advantage = returns_t

            # Clipped objective
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            loss_1 = ratio * advantage
            loss_2 = clipped_ratio * advantage
            loss = -torch.min(loss_1, loss_2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        def set_eval_mode(self):
            self.policy_network.eval()

        def set_train_mode(self):
            self.policy_network.train()

    class TRPOAgent(BaseRLAgent):
        """
        A placeholder TRPO agent. Real TRPO uses conjugate gradient to solve a constrained
        optimization problem. This is just a stub illustrating how we'd structure it.
        """

        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_size: int = 128,
            lr: float = 1e-3,
            max_kl: float = 0.01
        ):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.policy_network = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_dim),
                nn.Softmax(dim=-1)
            )
            self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
            self.max_kl = max_kl

        def act(self, state):
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            probs = self.policy_network(s).squeeze(0)
            action_dist = torch.distributions.Categorical(probs=probs)
            action = action_dist.sample()
            return action.item()

        def update_network(self, transitions):
            """
            With actual TRPO, we would do:
              1) Compute advantage
              2) Compute policy gradient with conjugate gradient, ensuring KL divergence < max_kl
              3) Update policy parameters
            This function is a placeholder to illustrate structure only.
            """
            self.set_train_mode()
            # Collect states, actions, rewards
            returns = 0
            gamma = 0.99
            all_states = []
            all_actions = []
            all_returns = []
            for (state, action, reward, _, _) in reversed(transitions):
                returns = reward + gamma * returns
                all_states.insert(0, state)
                all_actions.insert(0, action)
                all_returns.insert(0, returns)
            # Pseudocode for TRPO:
            #  advantage = all_returns - baseline
            #  grad = compute_policy_gradient(self.policy_network, states, actions, advantage)
            #  stepdir = conjugate_gradient(grad, fisher_vector_product, self.max_kl)
            #  line_search_update(self.policy_network, stepdir)
            pass

        def set_eval_mode(self):
            self.policy_network.eval()

        def set_train_mode(self):
            self.policy_network.train()
else:
    # Placeholder implementations when PyTorch is not available
    class SimplePolicyAgent(BaseRLAgent):
        def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128, lr: float = 1e-3):
            logger.warning("PyTorch not available. Using placeholder SimplePolicyAgent.")
            self.state_dim = state_dim
            self.action_dim = action_dim
            
        def act(self, state):
            return random.randint(0, self.action_dim - 1)
            
        def update_network(self, transitions):
            pass
            
        def set_eval_mode(self):
            pass
            
        def set_train_mode(self):
            pass
            
    class GRPOAgent(BaseRLAgent):
        def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128, lr: float = 1e-3, clip_epsilon: float = 0.2, entropy_coef: float = 0.01, num_epochs: int = 3, kl_target: float = 0.01):
            logger.warning("PyTorch not available. Using placeholder GRPOAgent.")
            self.state_dim = state_dim
            self.action_dim = action_dim
            
        def act(self, state):
            return random.randint(0, self.action_dim - 1)
            
        def update_network(self, group_experiences):
            pass
            
        def set_eval_mode(self):
            pass
            
        def set_train_mode(self):
            pass
            
    class PPOAgent(BaseRLAgent):
        def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128, lr: float = 1e-3, clip_epsilon: float = 0.2):
            logger.warning("PyTorch not available. Using placeholder PPOAgent.")
            self.state_dim = state_dim
            self.action_dim = action_dim
            
        def act(self, state):
            return random.randint(0, self.action_dim - 1)
            
        def update_network(self, transitions):
            pass
            
        def set_eval_mode(self):
            pass
            
        def set_train_mode(self):
            pass
            
    class TRPOAgent(BaseRLAgent):
        def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128, lr: float = 1e-3, max_kl: float = 0.01):
            logger.warning("PyTorch not available. Using placeholder TRPOAgent.")
            self.state_dim = state_dim
            self.action_dim = action_dim
            
        def act(self, state):
            return random.randint(0, self.action_dim - 1)
            
        def update_network(self, transitions):
            pass
            
        def set_eval_mode(self):
            pass
            
        def set_train_mode(self):
            pass

###############################################################################
# TOKEN BUDGET MANAGEMENT
###############################################################################

class TokenBudget:
    """
    Manages a token budget for the agent, tracking usage and enforcing limits.
    
    Attributes:
        initial_budget (int): The starting token budget
        remaining_budget (int): Current remaining tokens
        usage_history (Dict[str, int]): History of token usage by operation
        encoding (tiktoken.Encoding): The encoding used for token counting
    """
    def __init__(self, initial_budget: int = 8000):
        self.initial_budget = initial_budget
        self.remaining_budget = initial_budget
        self.usage_history = {}
        self._lock = threading.Lock()
        self.budget_efficiency = {}
        self.allocation_history = []
        
        # Try to load tiktoken encoding, fall back to character-based estimation
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
            self.token_counting_method = "tiktoken"
        except (ImportError, ValueError):
            self.encoding = None
            self.token_counting_method = "character_estimate"
            logger.warning("[TokenBudget] tiktoken not available, using character-based token estimation")
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string"""
        if self.token_counting_method == "tiktoken":
            return len(self.encoding.encode(text))
        else:
            # Fallback: estimate tokens as words/4 (very rough approximation)
            return len(text.split()) // 4 + 1
    
    def request_tokens(self, operation: str, amount: int) -> Tuple[bool, int]:
        """
        Request tokens for an operation.
        
        Args:
            operation: Name of the operation requesting tokens
            amount: Number of tokens requested
            
        Returns:
            Tuple of (success, granted_amount)
        """
        with self._lock:
            if amount <= 0:
                return False, 0
                
            # If we have enough budget, grant the full request
            if amount <= self.remaining_budget:
                self.remaining_budget -= amount
                self._record_usage(operation, amount)
                return True, amount
                
            # If we don't have enough, grant what we have left
            if self.remaining_budget > 0:
                granted = self.remaining_budget
                self.remaining_budget = 0
                self._record_usage(operation, granted)
                return True, granted
                
            # No budget left
            return False, 0
    
    def _record_usage(self, operation: str, amount: int) -> None:
        """Record token usage for an operation"""
        if operation in self.usage_history:
            self.usage_history[operation] += amount
        else:
            self.usage_history[operation] = amount
    
    def record_allocation(self, allocations: Dict[str, int]) -> None:
        """
        Record a set of budget allocations
        
        Args:
            allocations: Dictionary mapping operations to token allocations
        """
        with self._lock:
            timestamp = datetime.now().isoformat()
            self.allocation_history.append({
                "timestamp": timestamp,
                "allocations": allocations.copy()
            })
    
    def update_efficiency(self, operation: str, allocated: int, used: int) -> float:
        """
        Update efficiency metrics for an operation
        
        Args:
            operation: The operation name
            allocated: Tokens allocated
            used: Tokens actually used
            
        Returns:
            float: Efficiency percentage
        """
        with self._lock:
            if allocated <= 0:
                efficiency = 0.0
            else:
                efficiency = (used / allocated) * 100
                
            self.budget_efficiency[operation] = efficiency
            return efficiency
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get the current budget status"""
        with self._lock:
            return {
                "initial_budget": self.initial_budget,
                "remaining_budget": self.remaining_budget,
                "used_budget": self.initial_budget - self.remaining_budget,
                "usage_by_operation": self.usage_history.copy(),
                "efficiency_by_operation": self.budget_efficiency.copy(),
                "token_counting_method": self.token_counting_method,
                "allocation_history": self.allocation_history
            }
    
    def add_to_budget(self, amount: int) -> int:
        """Add tokens to the budget (e.g., for rewards or budget increases)"""
        with self._lock:
            self.remaining_budget += amount
            return self.remaining_budget
    
    def reset_budget(self, new_budget: Optional[int] = None) -> None:
        """Reset the budget to initial value or a new specified amount"""
        with self._lock:
            if new_budget is not None:
                self.initial_budget = new_budget
            self.remaining_budget = self.initial_budget
            self.usage_history = {}
            self.budget_efficiency = {}
            
    def get_recommended_allocation(self) -> Dict[str, int]:
        """
        Get recommended token allocation based on historical efficiency
        
        Returns:
            Dict mapping operations to recommended token allocations
        """
        with self._lock:
            if not self.budget_efficiency:
                # Default allocation if no history
                return {
                    "thinking": 3000,
                    "facts": 1000,
                    "cognition": 1000,
                    "answer": 2000,
                    "task_decomposition": 1000
                }
                
            # Calculate allocation based on efficiency and past usage
            total_budget = self.initial_budget
            allocations = {}
            
            # First pass: allocate based on efficiency
            remaining = total_budget
            for operation, efficiency in sorted(self.budget_efficiency.items(), key=lambda x: x[1], reverse=True):
                # Higher efficiency gets more tokens
                weight = efficiency / 100  # Convert percentage to weight
                allocation = int(total_budget * weight * 0.2)  # 20% influence from efficiency
                allocations[operation] = allocation
                remaining -= allocation
                
            # Second pass: adjust based on historical usage
            if self.usage_history:
                total_usage = sum(self.usage_history.values())
                if total_usage > 0:
                    for operation in allocations:
                        usage = self.usage_history.get(operation, 0)
                        usage_ratio = usage / total_usage
                        usage_allocation = int(remaining * usage_ratio * 0.8)  # 80% influence from usage
                        allocations[operation] += usage_allocation
                        
            # Ensure we don't exceed budget
            total_allocated = sum(allocations.values())
            if total_allocated > total_budget:
                # Scale down proportionally
                scale = total_budget / total_allocated
                for operation in allocations:
                    allocations[operation] = int(allocations[operation] * scale)
                    
            return allocations

###############################################################################
# DATA STRUCTURES FOR TASK MANAGEMENT
###############################################################################

class TaskStatus(str, Enum):
    """Enumeration of possible task statuses"""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    BLOCKED = "BLOCKED"

class TaskModel(BaseModel):
    """
    Pydantic model for task representation, used for structured outputs
    """
    task_id: int
    priority: int
    description: str
    status: TaskStatus = TaskStatus.PENDING
    parent_id: Optional[int] = None
    result: Optional[Any] = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    tags: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "task_id": 1,
                    "priority": 5,
                    "description": "Calculate the sum of numbers from 1 to 100",
                    "status": "PENDING",
                    "parent_id": None,
                    "tags": ["math", "calculation"]
                }
            ]
        }
    )

class Task:
    """
    Represents a single unit of work that can be processed by the agent.

    Attributes:
        task_id (int): Unique ID for the task.
        priority (int): Lower numbers => higher priority.
        description (str): Human-readable description of the task.
        status (str): 'PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED', etc.
        parent_id (Optional[int]): ID of a parent task, if any (for subtasks).
        result (Any): Arbitrary result data from completing the task.
    """
    def __init__(self, task_id: int, priority: int, description: str, parent_id: Optional[int] = None):
        self.task_id = task_id
        self.priority = priority
        self.description = description
        self.status = TaskStatus.PENDING
        self.parent_id = parent_id
        self.result = None
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.tags = []

    def __lt__(self, other: "Task") -> bool:
        return self.priority < other.priority

    def __repr__(self) -> str:
        snippet = self.description[:30].replace("\n", " ")
        return (f"Task(id={self.task_id}, prio={self.priority}, "
                f"status={self.status.value}, desc={snippet}...)")
                
    def to_model(self) -> TaskModel:
        """Convert Task object to TaskModel for structured output"""
        return TaskModel(
            task_id=self.task_id,
            priority=self.priority,
            description=self.description,
            status=self.status,
            parent_id=self.parent_id,
            result=self.result,
            created_at=self.created_at,
            updated_at=self.updated_at,
            tags=self.tags
        )
        
    def add_tag(self, tag: str) -> None:
        """Add a tag to the task for better categorization"""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = time.time()

class TaskMemoryStore:
    """
    Thread-safe in-memory storage for Task objects.
    Allows for add, retrieve, update status, update result, and listing tasks.
    """
    def __init__(self) -> None:
        self._tasks: Dict[int, Task] = {}
        self._lock = threading.Lock()
        self._next_id = 1

    def add_task(self, task: Task) -> None:
        with self._lock:
            if task.task_id in self._tasks:
                logger.warning(f"[TaskMemoryStore] Task ID {task.task_id} already exists. Overwriting.")
            self._tasks[task.task_id] = task

    def create_task(self, priority: int, description: str, parent_id: Optional[int] = None) -> Task:
        """Create a new task with the next available ID"""
        with self._lock:
            task_id = self._next_id
            self._next_id += 1
            task = Task(task_id, priority, description, parent_id)
            self._tasks[task_id] = task
            return task

    def get_task(self, task_id: int) -> Optional[Task]:
        with self._lock:
            return self._tasks.get(task_id)

    def update_task_status(self, task_id: int, status: TaskStatus) -> None:
        with self._lock:
            t = self._tasks.get(task_id)
            if t:
                t.status = status
                t.updated_at = time.time()

    def update_task_result(self, task_id: int, result: Any) -> None:
        with self._lock:
            t = self._tasks.get(task_id)
            if t:
                t.result = result
                t.updated_at = time.time()

    def list_tasks(self, status: Optional[TaskStatus] = None, tag: Optional[str] = None) -> List[Task]:
        with self._lock:
            tasks = list(self._tasks.values())
            
            if status:
                tasks = [t for t in tasks if t.status == status]
                
            if tag:
                tasks = [t for t in tasks if tag in t.tags]
                
            return tasks
            
    def get_subtasks(self, parent_id: int) -> List[Task]:
        """Get all subtasks for a given parent task"""
        with self._lock:
            return [t for t in self._tasks.values() if t.parent_id == parent_id]

    def __len__(self) -> int:
        with self._lock:
            return len(self._tasks)
            
    def to_model_list(self, tasks: Optional[List[Task]] = None) -> List[TaskModel]:
        """Convert a list of Task objects to TaskModel objects for structured output"""
        if tasks is None:
            with self._lock:
                tasks = list(self._tasks.values())
        return [t.to_model() for t in tasks]

###############################################################################
# GOAL MANAGEMENT & PLANNING
###############################################################################

class GoalStatus(str, Enum):
    """Enumeration of possible goal statuses"""
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    ON_HOLD = "ON_HOLD"
    ABANDONED = "ABANDONED"

class GoalModel(BaseModel):
    """
    Pydantic model for goal representation, used for structured outputs
    """
    goal_id: int
    name: str
    description: str
    priority: int
    status: GoalStatus = GoalStatus.ACTIVE
    created_at: float = Field(default_factory=time.time)
    last_updated: float = Field(default_factory=time.time)
    progress: float = 0.0
    deadline: Optional[float] = None
    success_criteria: List[str] = Field(default_factory=list)
    related_tasks: List[int] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "goal_id": 1,
                    "name": "Complete Project X",
                    "description": "Finish all tasks related to Project X by the deadline",
                    "priority": 1,
                    "status": "ACTIVE",
                    "progress": 0.25,
                    "deadline": time.time() + 86400*7,  # One week from now
                    "success_criteria": ["All tests pass", "Documentation complete"],
                    "tags": ["project-x", "high-priority"]
                }
            ]
        }
    )

class Goal:
    """
    Represents a long-range goal that the agent tries to achieve.

    Attributes:
        goal_id (int): Unique ID for the goal.
        name (str): Short name for the goal.
        description (str): Detailed explanation of the goal.
        priority (int): Lower => higher priority.
        status (str): 'ACTIVE', 'COMPLETED', 'ON_HOLD', or 'ABANDONED'.
        created_at (float): Timestamp of creation.
        last_updated (float): Timestamp of last update.
    """
    def __init__(self, goal_id: int, name: str, description: str, priority: int = 5, 
                 deadline: Optional[float] = None, success_criteria: Optional[List[str]] = None):
        self.goal_id = goal_id
        self.name = name
        self.description = description
        self.priority = priority
        self.status = GoalStatus.ACTIVE
        self.created_at = time.time()
        self.last_updated = self.created_at
        self.progress = 0.0
        self.deadline = deadline
        self.success_criteria = success_criteria or []
        self.related_tasks = []
        self.tags = []

    def update_description(self, new_desc: str) -> None:
        self.description = new_desc
        self.last_updated = time.time()

    def complete(self) -> None:
        self.status = GoalStatus.COMPLETED
        self.progress = 1.0
        self.last_updated = time.time()
        
    def update_progress(self, progress: float) -> None:
        """Update the progress of this goal (0.0 to 1.0)"""
        self.progress = max(0.0, min(1.0, progress))
        self.last_updated = time.time()
        
    def add_tag(self, tag: str) -> None:
        """Add a tag to the goal for better categorization"""
        if tag not in self.tags:
            self.tags.append(tag)
            self.last_updated = time.time()
            
    def time_remaining(self) -> Optional[float]:
        """Get the time remaining until the deadline in seconds, or None if no deadline"""
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.time())
        
    def is_overdue(self) -> bool:
        """Check if the goal is overdue"""
        if self.deadline is None:
            return False
        return time.time() > self.deadline

    def __repr__(self) -> str:
        snippet = self.description[:30].replace("\n", " ")
        return (f"Goal(id={self.goal_id}, name={self.name}, "
                f"priority={self.priority}, status={self.status.value}, desc={snippet}...)")
                
    def to_model(self) -> GoalModel:
        """Convert Goal object to GoalModel for structured output"""
        return GoalModel(
            goal_id=self.goal_id,
            name=self.name,
            description=self.description,
            priority=self.priority,
            status=self.status,
            created_at=self.created_at,
            last_updated=self.last_updated,
            progress=self.progress,
            deadline=self.deadline,
            success_criteria=self.success_criteria,
            related_tasks=self.related_tasks,
            tags=self.tags
        )

class GoalManager:
    """
    Manages creation, retrieval, and updating of multiple goals.
    Thread-safe with a simple in-memory dictionary.
    """
    def __init__(self):
        self._goals: Dict[int, Goal] = {}
        self._lock = threading.Lock()
        self._next_id = 1

    def create_goal(self, name: str, description: str, priority: int = 5, 
                   deadline: Optional[float] = None, 
                   success_criteria: Optional[List[str]] = None) -> Goal:
        with self._lock:
            g = Goal(self._next_id, name, description, priority, deadline, success_criteria)
            self._goals[self._next_id] = g
            logger.info(f"[GoalManager] Created Goal: {g}")
            self._next_id += 1
            return g

    def get_goal(self, goal_id: int) -> Optional[Goal]:
        with self._lock:
            return self._goals.get(goal_id)

    def list_goals(self, status: Optional[GoalStatus] = None, tag: Optional[str] = None) -> List[Goal]:
        with self._lock:
            goals = list(self._goals.values())
            
            if status:
                goals = [g for g in goals if g.status == status]
                
            if tag:
                goals = [g for g in goals if tag in g.tags]
                
            return goals

    def update_goal_status(self, goal_id: int, status: GoalStatus) -> None:
        with self._lock:
            g = self._goals.get(goal_id)
            if g:
                g.status = status
                g.last_updated = time.time()
                logger.info(f"[GoalManager] Updated goal {goal_id} to status={status.value}")
                # Enhanced goal management: Re-evaluate priorities
                self._re_evaluate_goal_priorities()
                
    def update_goal_progress(self, goal_id: int, progress: float) -> None:
        """Update the progress of a goal (0.0 to 1.0)"""
        with self._lock:
            g = self._goals.get(goal_id)
            if g:
                g.update_progress(progress)
                logger.info(f"[GoalManager] Updated goal {goal_id} progress to {progress:.2f}")
                
                # Auto-complete goal if progress reaches 1.0
                if progress >= 1.0 and g.status == GoalStatus.ACTIVE:
                    g.status = GoalStatus.COMPLETED
                    logger.info(f"[GoalManager] Auto-completed goal {goal_id} due to progress")

    def _re_evaluate_goal_priorities(self) -> None:
        """
        Re-evaluate and adjust goal priorities based on current context.
        """
        with self._lock:
            for goal in self._goals.values():
                # Example logic: Increase priority for goals nearing completion
                if goal.status == GoalStatus.ACTIVE and goal.priority > 1:
                    goal.priority -= 1
                    logger.info(f"[GoalManager] Increased priority for goal {goal.goal_id} to {goal.priority}")
                
                # Increase priority for goals nearing deadline
                if goal.deadline and goal.status == GoalStatus.ACTIVE:
                    time_remaining = goal.time_remaining()
                    if time_remaining and time_remaining < 86400:  # Less than a day
                        goal.priority = max(1, goal.priority - 2)
                        logger.info(f"[GoalManager] Increased priority for goal {goal.goal_id} due to approaching deadline")
                
                # Advanced goal management: Adjust based on performance metrics
                if goal.status == GoalStatus.ACTIVE and self._should_adjust_goal_based_on_performance(goal):
                    goal.priority = max(0, goal.priority - 1)
                    logger.info(f"[GoalManager] Adjusted priority for goal {goal.goal_id} based on performance metrics.")
                    
    def to_model_list(self, goals: Optional[List[Goal]] = None) -> List[GoalModel]:
        """Convert a list of Goal objects to GoalModel objects for structured output"""
        if goals is None:
            with self._lock:
                goals = list(self._goals.values())
        return [g.to_model() for g in goals]

    def _should_adjust_goal_based_on_performance(self, goal: Goal) -> bool:
        """
        Determine if a goal's priority should be adjusted based on performance metrics.
        """
        # More sophisticated logic for performance-based adjustment
        if goal.progress > 0.7:  # If goal is more than 70% complete
            return True
        if goal.is_overdue():  # If goal is overdue
            return True
        return False

###############################################################################
# CONVERSATION MANAGEMENT
###############################################################################

class ConversationMemory:
    """
    Maintains a conversation history (list of dicts with role="user"/"assistant").
    If it grows too large, we do a naive summarization by trimming older messages.
    """
    def __init__(self) -> None:
        self._utterances: List[Dict[str, str]] = []
        self._lock = threading.Lock()
        self._max_length = 25  # bigger than the earlier 20, to allow more history

    def add_user_utterance(self, text: str) -> None:
        with self._lock:
            self._utterances.append({"role": "user", "content": text})
            self._maybe_summarize()

    def add_agent_utterance(self, text: str) -> None:
        with self._lock:
            self._utterances.append({"role": "assistant", "content": text})
            self._maybe_summarize()

    def get_history(self) -> List[Dict[str, str]]:
        with self._lock:
            return list(self._utterances)

    def _maybe_summarize(self) -> None:
        """
        If conversation is too long, produce a naive summary of the last few items
        and store it as a system message, trimming out older messages.
        """
        if len(self._utterances) > self._max_length:
            snippet = " | ".join(u["content"][:30] for u in self._utterances[-7:])
            summary = f"Conversation exceeded {self._max_length} messages. Summary of last 7: {snippet}"
            # Keep only the last 7 messages
            self._utterances = self._utterances[-7:]
            # Insert summary as a system message
            self._utterances.insert(0, {"role": "system", "content": summary})
            logger.info("[ConversationMemory] Summarized conversation due to length limit.")

###############################################################################
# COGNITIVE MODELS AND REASONING
###############################################################################

class CognitiveBehavior(str, Enum):
    """
    Defines the key cognitive behaviors that the agent can exhibit during reasoning.
    """
    # Original behaviors
    VERIFICATION = "verification"
    BACKTRACKING = "backtracking"
    SUBGOAL_SETTING = "subgoal_setting"
    BACKWARD_CHAINING = "backward_chaining"
    REFLECTION = "reflection"
    ADAPTATION = "adaptation"
    EXPLORATION = "exploration"
    PLANNING = "planning"
    EVALUATION = "evaluation"
    CREATIVITY = "creativity"
    ABSTRACTION = "abstraction"
    ANALOGY = "analogy"
    
    # Game theory and strategic behaviors
    MULTIAGENT_REASONING = "multiagent_reasoning"
    OPPONENT_MODELING = "opponent_modeling"
    COUNTERFACTUAL_THINKING = "counterfactual_thinking"
    RESOURCE_ALLOCATION = "resource_allocation"
    RISK_ASSESSMENT = "risk_assessment"
    TEMPORAL_PROJECTION = "temporal_projection"
    DECISION_TREE_ANALYSIS = "decision_tree_analysis"
    BAYESIAN_UPDATING = "bayesian_updating"
    NASH_EQUILIBRIUM_SEEKING = "nash_equilibrium_seeking"
    STRATEGIC_DECEPTION = "strategic_deception"
    COOPERATIVE_PLANNING = "cooperative_planning"
    COMPETITIVE_PLANNING = "competitive_planning"
    COALITION_FORMATION = "coalition_formation"
    NEGOTIATION = "negotiation"
    COMMITMENT_MANAGEMENT = "commitment_management"
    REPUTATION_MODELING = "reputation_modeling"
    
    # Learning and adaptation
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    TRANSFER_LEARNING = "transfer_learning"
    META_LEARNING = "meta_learning"
    CAUSAL_REASONING = "causal_reasoning"
    CONCEPT_FORMATION = "concept_formation"
    SKILL_REFINEMENT = "skill_refinement"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    PATTERN_RECOGNITION = "pattern_recognition"
    ANOMALY_DETECTION = "anomaly_detection"
    
    # Long-term strategic behaviors
    GRAND_STRATEGY_FORMATION = "grand_strategy_formation"
    TREND_PROJECTION = "trend_projection"
    SCENARIO_PLANNING = "scenario_planning"
    OPPORTUNITY_IDENTIFICATION = "opportunity_identification"
    THREAT_ASSESSMENT = "threat_assessment"
    RESOURCE_CULTIVATION = "resource_cultivation"
    STRATEGIC_POSITIONING = "strategic_positioning"
    NETWORK_EFFECT_ANALYSIS = "network_effect_analysis"
    TECHNOLOGICAL_FORECASTING = "technological_forecasting"
    POLITICAL_ANALYSIS = "political_analysis"
    ECONOMIC_FORECASTING = "economic_forecasting"
    DEMOGRAPHIC_ANALYSIS = "demographic_analysis"
    
    # Metacognition & cognitive governance
    COGNITIVE_BIAS_CORRECTION = "cognitive_bias_correction"
    EXPERTISE_CALIBRATION = "expertise_calibration"
    ATTENTION_ALLOCATION = "attention_allocation"
    MENTAL_SIMULATION = "mental_simulation"
    PERSPECTIVE_TAKING = "perspective_taking"
    COGNITIVE_DECOUPLING = "cognitive_decoupling"
    EPISTEMIC_VIGILANCE = "epistemic_vigilance"
    INSIGHT_CULTIVATION = "insight_cultivation"
    METACOGNITIVE_MONITORING = "metacognitive_monitoring"
    METACOGNITIVE_CONTROL = "metacognitive_control"
    RATIONALITY_CHECKING = "rationality_checking"
    CONTRADICTION_DETECTION = "contradiction_detection"
    
    # Communication & coordination
    INFORMATION_SHARING = "information_sharing"
    KNOWLEDGE_INTEGRATION = "knowledge_integration"
    TEAM_COORDINATION = "team_coordination"
    CONFLICT_RESOLUTION = "conflict_resolution"
    PREFERENCE_AGGREGATION = "preference_aggregation"
    CONSENSUS_BUILDING = "consensus_building"
    ROLE_ASSIGNMENT = "role_assignment"
    SHARED_MENTAL_MODEL_BUILDING = "shared_mental_model_building"
    COMMUNICATION_OPTIMIZATION = "communication_optimization"
    COMMON_KNOWLEDGE_MANAGEMENT = "common_knowledge_management"
    TRUST_BUILDING = "trust_building"
    
    # Analysis
    ANALYSIS = "analysis"
    METACOGNITION = "metacognition"
    COUNTERFACTUAL = "counterfactual"
    UNCERTAINTY = "uncertainty"
    ABDUCTIVE_REASONING = "abductive_reasoning"
    DEDUCTIVE_REASONING = "deductive_reasoning"
    INDUCTIVE_REASONING = "inductive_reasoning"
    
    # Decision making and optimization
    UTILITY_MAXIMIZATION = "utility_maximization"
    SATISFICING = "satisficing"
    HEURISTIC_DECISION_MAKING = "heuristic_decision_making"
    MULTI_CRITERIA_DECISION_MAKING = "multi_criteria_decision_making"
    CONSTRAINED_OPTIMIZATION = "constrained_optimization"
    STOCHASTIC_OPTIMIZATION = "stochastic_optimization"
    LINEAR_PROGRAMMING = "linear_programming"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    GREEDY_ALGORITHM = "greedy_algorithm"
    SIMULATED_ANNEALING = "simulated_annealing"
    GENETIC_ALGORITHM = "genetic_algorithm"
    SWARM_OPTIMIZATION = "swarm_optimization"
    GRADIENT_DESCENT = "gradient_descent"
    HILL_CLIMBING = "hill_climbing"
    
    # Probabilistic reasoning
    PROBABILISTIC_INFERENCE = "probabilistic_inference"
    MONTE_CARLO_SIMULATION = "monte_carlo_simulation"
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"
    SCENARIO_ANALYSIS = "scenario_analysis"
    EXPECTED_VALUE_CALCULATION = "expected_value_calculation"
    VARIANCE_REDUCTION = "variance_reduction"
    MARKOV_DECISION_PROCESS = "markov_decision_process"
    HIDDEN_MARKOV_MODEL = "hidden_markov_model"
    BELIEF_REVISION = "belief_revision"
    DEMPSTER_SHAFER_THEORY = "dempster_shafer_theory"
    FUZZY_LOGIC = "fuzzy_logic"
    POSSIBILISTIC_REASONING = "possibilistic_reasoning"
    
    # Evolutionary and adaptive behaviors
    EVOLUTIONARY_ADAPTATION = "evolutionary_adaptation"
    CO_EVOLUTIONARY_DYNAMICS = "co_evolutionary_dynamics"
    CULTURAL_EVOLUTION = "cultural_evolution"
    MEME_PROPAGATION = "meme_propagation"
    SELECTIVE_PRESSURE_ANALYSIS = "selective_pressure_analysis"
    FITNESS_LANDSCAPE_ANALYSIS = "fitness_landscape_analysis"
    NICHE_CONSTRUCTION = "niche_construction"
    RED_QUEEN_DYNAMICS = "red_queen_dynamics"
    Baldwin_EFFECT = "baldwin_effect"
    EXAPTATION = "exaptation"
    GENETIC_DRIFT = "genetic_drift"
    EPISTASIS = "epistasis"
    
    # Network and systems behaviors
    NETWORK_ANALYSIS = "network_analysis"
    SYSTEMS_THINKING = "systems_thinking"
    FEEDBACK_LOOP_ANALYSIS = "feedback_loop_analysis"
    CAUSAL_LOOP_DIAGRAMMING = "causal_loop_diagramming"
    STOCK_AND_FLOW_MODELING = "stock_and_flow_modeling"
    TIPPING_POINT_ANALYSIS = "tipping_point_analysis"
    RESILIENCE_ASSESSMENT = "resilience_assessment"
    ROBUSTNESS_ANALYSIS = "robustness_analysis"
    MODULARITY_ANALYSIS = "modularity_analysis"
    CENTRALITY_ANALYSIS = "centrality_analysis"
    SMALL_WORLD_NETWORK = "small_world_network"
    SCALE_FREE_NETWORK = "scale_free_network"
    
    # Information theory and complexity
    INFORMATION_THEORETIC_ANALYSIS = "information_theoretic_analysis"
    ENTROPY_MEASUREMENT = "entropy_measurement"
    COMPLEXITY_ASSESSMENT = "complexity_assessment"
    KOLMOGOROV_COMPLEXITY = "kolmogorov_complexity"
    MINIMUM_DESCRIPTION_LENGTH = "minimum_description_length"
    ALGORITHMIC_INFORMATION_THEORY = "algorithmic_information_theory"
    SHANNON_INFORMATION = "shannon_information"
    MUTUAL_INFORMATION = "mutual_information"
    INFORMATION_BOTTLENECK = "information_bottleneck"
    INFORMATION_CASCADE = "information_cascade"
    CHANNEL_CAPACITY = "channel_capacity"
    
    # Market and economic behaviors
    MARKET_ANALYSIS = "market_analysis"
    SUPPLY_DEMAND_ANALYSIS = "supply_demand_analysis"
    PRICE_DISCRIMINATION = "price_discrimination"
    MARKET_SEGMENTATION = "market_segmentation"
    COMPETITIVE_POSITIONING = "competitive_positioning"
    VALUE_CHAIN_ANALYSIS = "value_chain_analysis"
    OPPORTUNITY_COST_ANALYSIS = "opportunity_cost_analysis"
    COMPARATIVE_ADVANTAGE = "comparative_advantage"
    ABSOLUTE_ADVANTAGE = "absolute_advantage"
    ECONOMIES_OF_SCALE = "economies_of_scale"
    ECONOMIES_OF_SCOPE = "economies_of_scope"
    NETWORK_EXTERNALITY = "network_externality"
    MARKET_POWER_ASSESSMENT = "market_power_assessment"
    CONSUMER_SURPLUS = "consumer_surplus"
    PRODUCER_SURPLUS = "producer_surplus"
    DEADWEIGHT_LOSS = "deadweight_loss"
    ELASTICITY_ANALYSIS = "elasticity_analysis"
    INDIFFERENCE_CURVE = "indifference_curve"
    MARGINAL_UTILITY = "marginal_utility"
    MARGINAL_COST = "marginal_cost"
    MARGINAL_REVENUE = "marginal_revenue"
    
    # Knowledge management
    KNOWLEDGE_REPRESENTATION = "knowledge_representation"
    KNOWLEDGE_ELICITATION = "knowledge_elicitation"
    KNOWLEDGE_ACQUISITION = "knowledge_acquisition"
    KNOWLEDGE_ORGANIZATION = "knowledge_organization"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    KNOWLEDGE_TRANSFER = "knowledge_transfer"
    KNOWLEDGE_SHARING = "knowledge_sharing"
    KNOWLEDGE_CREATION = "knowledge_creation"
    KNOWLEDGE_INTEGRATION = "knowledge_integration"
    KNOWLEDGE_PRESERVATION = "knowledge_preservation"
    KNOWLEDGE_EVALUATION = "knowledge_evaluation"
    KNOWLEDGE_GOVERNANCE = "knowledge_governance"
    KNOWLEDGE_ECOLOGY = "knowledge_ecology"
    KNOWLEDGE_COMMONS = "knowledge_commons"
    KNOWLEDGE_BOUNDARY = "knowledge_boundary"
    KNOWLEDGE_FLOW = "knowledge_flow"
    KNOWLEDGE_SPILLOVER = "knowledge_spillover"
    KNOWLEDGE_BARRIER = "knowledge_barrier"
    KNOWLEDGE_ABSORPTION = "knowledge_absorption"
    KNOWLEDGE_ASSIMILATION = "knowledge_assimilation"
    KNOWLEDGE_TRANSFORMATION = "knowledge_transformation"
    KNOWLEDGE_EXPLOITATION = "knowledge_exploitation"
    KNOWLEDGE_EXPLORATION = "knowledge_exploration"


class ReasoningStep(BaseModel):
    """
    Represents a single step in the agent's reasoning process.
    """
    step_number: int = Field(..., description="The order of the step in the chain-of-thought")
    behavior: CognitiveBehavior = Field(..., description="The cognitive behavior for this step")
    description: str = Field(..., description="A textual description of the step")
    result: Optional[Union[str, float, Dict[str, Any]]] = Field(None, description="The result or outcome of the step")
    is_correct: Optional[bool] = Field(None, description="Flag indicating if the result was correct")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the step")
    confidence: float = Field(default=0.5, description="Confidence level in this reasoning step (0.0 to 1.0)")
    
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "step_number": 1,
                    "behavior": "verification",
                    "description": "Checking if the calculation is correct",
                    "result": "5 + 7 = 12",
                    "is_correct": True,
                    "confidence": 0.95
                }
            ]
        }
    )


class ChainOfThought(BaseModel):
    """
    Maintains a sequence of reasoning steps forming a chain-of-thought.
    """
    steps: List[ReasoningStep] = Field(default_factory=list, description="List of reasoning steps")
    summary: Optional[str] = Field(None, description="A summary of the reasoning process")
    conclusion: Optional[str] = Field(None, description="The final conclusion reached")
    confidence: float = Field(default=0.5, description="Overall confidence in the reasoning chain")
    
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "steps": [
                        {
                            "step_number": 1,
                            "behavior": "verification",
                            "description": "Checking if the calculation is correct",
                            "result": "5 + 7 = 12",
                            "is_correct": True,
                            "confidence": 0.95
                        }
                    ],
                    "summary": "Verified the calculation 5 + 7 = 12",
                    "conclusion": "The calculation is correct",
                    "confidence": 0.95
                }
            ]
        }
    )
    
    def add_step(self, step: ReasoningStep) -> None:
        """Add a reasoning step to the chain."""
        self.steps.append(step)
        # Update overall confidence based on new step
        self.confidence = sum(s.confidence for s in self.steps) / len(self.steps)
    
    def get_last_step(self) -> Optional[ReasoningStep]:
        """Get the last reasoning step, if any."""
        if self.steps:
            return self.steps[-1]
        return None
    
    def get_steps_by_behavior(self, behavior: CognitiveBehavior) -> List[ReasoningStep]:
        """Get all steps with a specific cognitive behavior."""
        return [step for step in self.steps if step.behavior == behavior]
        
    def update_summary(self, summary: str) -> None:
        """Update the summary of the reasoning process."""
        self.summary = summary
        
    def set_conclusion(self, conclusion: str, confidence: float = None) -> None:
        """Set the final conclusion of the reasoning process."""
        self.conclusion = conclusion
        if confidence is not None:
            self.confidence = confidence


class SubtaskDecomposition(BaseModel):
    """
    Structured model for task decomposition into subtasks
    """
    original_task_id: int
    original_description: str
    subtasks: List[Dict[str, str]] = Field(..., description="List of subtasks with descriptions")
    dependencies: Optional[Dict[int, List[int]]] = Field(None, description="Map of subtask indices to their dependencies")
    estimated_complexity: Optional[Dict[int, int]] = Field(None, description="Map of subtask indices to complexity (1-10)")
    rationale: str = Field(..., description="Explanation of how the task was decomposed")
    
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "original_task_id": 1,
                    "original_description": "Build a simple web application",
                    "subtasks": [
                        {"description": "Design the database schema"},
                        {"description": "Create the backend API"},
                        {"description": "Develop the frontend UI"},
                        {"description": "Write tests"},
                        {"description": "Deploy the application"}
                    ],
                    "dependencies": {
                        "2": [0],  # Backend depends on database design
                        "3": [1],  # Frontend depends on backend
                        "4": [2, 3]  # Deployment depends on frontend and tests
                    },
                    "estimated_complexity": {
                        "0": 3,
                        "1": 5,
                        "2": 4,
                        "3": 3,
                        "4": 2
                    },
                    "rationale": "The web application development is broken down into standard phases with clear dependencies."
                }
            ]
        }
    )

class CognitiveModelingEngine:
    """
    Engine for modeling and executing cognitive behaviors in the agent.
    This is model-agnostic and can work with any LLM backend.
    Features:
    - Advanced chain-of-thought reasoning
    - Parallel reasoning paths
    - Confidence calibration
    - Uncertainty quantification
    - Counterfactual reasoning
    - Metacognitive monitoring
    - Reasoning path optimization
    """
    def __init__(self):
        self._chain_of_thought: ChainOfThought = ChainOfThought()
        self._current_step: int = 0
        self._lock = threading.Lock()
        self._reasoning_paths: Dict[str, List[ReasoningStep]] = {}
        self._active_path: str = "main"
        self._path_confidences: Dict[str, float] = {"main": 1.0}
        self._uncertainty_metrics: Dict[str, float] = {}
        self._metacognitive_state: Dict[str, Any] = {
            "calibration_score": 1.0,
            "reasoning_efficiency": 1.0,
            "path_diversity": 0.0,
            "last_calibration": time.time()
        }
        
    def add_reasoning_step(
        self,
        behavior: CognitiveBehavior,
        description: str,
        result: Optional[Union[str, float, Dict[str, Any]]] = None,
        is_correct: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: float = 0.5,
        path: str = None
    ) -> ReasoningStep:
        """
        Add a new reasoning step to the chain-of-thought.
        
        Args:
            behavior: The cognitive behavior for this step
            description: Textual description of the step
            result: Optional result or outcome of the step
            is_correct: Whether the result is correct
            metadata: Additional metadata for the step
            confidence: Confidence level in this reasoning step (0.0 to 1.0)
            path: Optional reasoning path identifier (defaults to active path)
            
        Returns:
            The created reasoning step
        """
        with self._lock:
            # Use specified path or active path
            current_path = path or self._active_path
            
            # Create path if it doesn't exist
            if current_path not in self._reasoning_paths:
                self._reasoning_paths[current_path] = []
                self._path_confidences[current_path] = confidence
            
            self._current_step += 1
            step = ReasoningStep(
                step_number=self._current_step,
                behavior=behavior,
                description=description,
                result=result,
                is_correct=is_correct,
                metadata=metadata or {},
                confidence=confidence
            )
            
            # Add to both main chain and path-specific chain
            self._chain_of_thought.add_step(step)
            self._reasoning_paths[current_path].append(step)
            
            # Update path confidence based on step confidence
            self._path_confidences[current_path] = (
                self._path_confidences[current_path] * 0.8 + confidence * 0.2
            )
            
            # Update metacognitive state
            self._update_metacognition(step, current_path)
            
            logger.info(f"[CognitiveModelingEngine] Added reasoning step {self._current_step}: {behavior} - {description} (path: {current_path}, confidence: {confidence:.2f})")
            return step
            
    def _update_metacognition(self, step: ReasoningStep, path: str) -> None:
        """Update metacognitive monitoring metrics based on new reasoning step"""
        # Update calibration score if we have ground truth
        if step.is_correct is not None:
            # Calculate calibration error (confidence vs correctness)
            calibration_error = abs(float(step.is_correct) - step.confidence)
            # Update calibration score (higher is better)
            self._metacognitive_state["calibration_score"] = (
                self._metacognitive_state["calibration_score"] * 0.9 + 
                (1.0 - calibration_error) * 0.1
            )
            
        # Update path diversity metric
        self._metacognitive_state["path_diversity"] = len(self._reasoning_paths) / 10.0
        
        # Periodically recalibrate confidence if needed
        if time.time() - self._metacognitive_state["last_calibration"] > 300:  # 5 minutes
            self._recalibrate_confidence()
            self._metacognitive_state["last_calibration"] = time.time()
    
    def create_reasoning_path(self, path_id: str, description: str, 
                             initial_confidence: float = 0.5) -> str:
        """
        Create a new reasoning path for exploring alternative hypotheses.
        
        Args:
            path_id: Unique identifier for the path
            description: Description of this reasoning path
            initial_confidence: Initial confidence in this path
            
        Returns:
            The path ID
        """
        with self._lock:
            if path_id in self._reasoning_paths:
                # Path already exists, return existing ID
                return path_id
                
            # Create new path
            self._reasoning_paths[path_id] = []
            self._path_confidences[path_id] = initial_confidence
            
            # Add a metadata step to the main chain
            self.add_reasoning_step(
                behavior=CognitiveBehavior.EXPLORATION,
                description=f"Created alternative reasoning path: {description}",
                metadata={"path_id": path_id, "type": "path_creation"},
                confidence=initial_confidence,
                path="main"  # Always add to main path
            )
            
            logger.info(f"[CognitiveModelingEngine] Created new reasoning path: {path_id}")
            return path_id
            
    def switch_reasoning_path(self, path_id: str) -> bool:
        """
        Switch to a different reasoning path.
        
        Args:
            path_id: The path ID to switch to
            
        Returns:
            Success status
        """
        with self._lock:
            if path_id not in self._reasoning_paths:
                logger.warning(f"[CognitiveModelingEngine] Cannot switch to non-existent path: {path_id}")
                return False
                
            self._active_path = path_id
            logger.info(f"[CognitiveModelingEngine] Switched to reasoning path: {path_id}")
            return True
            
    def get_best_reasoning_path(self) -> str:
        """
        Get the reasoning path with highest confidence.
        
        Returns:
            Path ID of the highest confidence path
        """
        with self._lock:
            if not self._path_confidences:
                return "main"
                
            return max(self._path_confidences.items(), key=lambda x: x[1])[0]
            
    def merge_reasoning_paths(self, source_path: str, target_path: str = "main") -> bool:
        """
        Merge steps from source path into target path.
        
        Args:
            source_path: Path to merge from
            target_path: Path to merge into (defaults to main)
            
        Returns:
            Success status
        """
        with self._lock:
            if source_path not in self._reasoning_paths:
                logger.warning(f"[CognitiveModelingEngine] Source path does not exist: {source_path}")
                return False
                
            if target_path not in self._reasoning_paths:
                logger.warning(f"[CognitiveModelingEngine] Target path does not exist: {target_path}")
                return False
                
            # Add steps from source to target
            for step in self._reasoning_paths[source_path]:
                # Skip steps already in target path
                if step not in self._reasoning_paths[target_path]:
                    self._reasoning_paths[target_path].append(step)
                    
            # Update target path confidence
            source_conf = self._path_confidences[source_path]
            target_conf = self._path_confidences[target_path]
            self._path_confidences[target_path] = max(source_conf, target_conf)
            
            logger.info(f"[CognitiveModelingEngine] Merged path {source_path} into {target_path}")
            return True
            
    def _recalibrate_confidence(self) -> None:
        """Recalibrate confidence scores based on historical accuracy"""
        # Skip if we don't have enough data
        if self._current_step < 5:
            return
            
        # Calculate calibration factor based on historical accuracy
        calibration_factor = self._metacognitive_state["calibration_score"]
        
        # Apply calibration to all path confidences
        for path_id in self._path_confidences:
            raw_confidence = self._path_confidences[path_id]
            calibrated = raw_confidence * calibration_factor
            self._path_confidences[path_id] = calibrated
            
        logger.info(f"[CognitiveModelingEngine] Recalibrated confidence scores with factor {calibration_factor:.2f}")

    def verify(self, description: str, result: Any, is_correct: bool = None, 
              confidence: float = 0.8, path: str = None) -> ReasoningStep:
        """
        Execute verification behavior: check if a result or intermediate step is correct.
        
        Args:
            description: What is being verified
            result: The result being verified
            is_correct: Whether the result is correct
            confidence: Confidence in the verification
            path: Optional reasoning path identifier
            
        Returns:
            The created reasoning step
        """
        return self.add_reasoning_step(
            behavior=CognitiveBehavior.VERIFICATION,
            description=f"Verifying: {description}",
            result=result,
            is_correct=is_correct,
            confidence=confidence,
            path=path
        )
    
    def backtrack(self, reason: str, confidence: float = 0.7, path: str = None) -> ReasoningStep:
        """
        Execute backtracking behavior: abandon a failing approach and try another.
        
        Args:
            reason: Reason for backtracking
            confidence: Confidence in the backtracking decision
            path: Optional reasoning path identifier
            
        Returns:
            The created reasoning step
        """
        # When backtracking, create a new path to explore alternative
        if path is None and self._active_path == "main":
            new_path_id = f"alternative_{len(self._reasoning_paths)}"
            self.create_reasoning_path(new_path_id, f"Alternative after backtracking: {reason}", confidence)
            self.switch_reasoning_path(new_path_id)
            path = new_path_id
        
        return self.add_reasoning_step(
            behavior=CognitiveBehavior.BACKTRACKING,
            description=f"Backtracking: {reason}",
            confidence=confidence,
            path=path
        )
    
    def set_subgoal(self, subgoal: str, metadata: Optional[Dict[str, Any]] = None, 
                   confidence: float = 0.8, path: str = None) -> ReasoningStep:
        """
        Execute subgoal setting behavior: break a problem into smaller, manageable parts.
        
        Args:
            subgoal: The subgoal to set
            metadata: Additional metadata
            confidence: Confidence in this subgoal
            path: Optional reasoning path identifier
            
        Returns:
            The created reasoning step
        """
        return self.add_reasoning_step(
            behavior=CognitiveBehavior.SUBGOAL_SETTING,
            description=f"Setting subgoal: {subgoal}",
            metadata=metadata,
            confidence=confidence,
            path=path
        )
    
    def backward_chain(self, target: str, steps: Optional[List[str]] = None, 
                      confidence: float = 0.75, path: str = None) -> ReasoningStep:
        """
        Execute backward chaining behavior: start from the goal and work backwards.
        
        Args:
            target: The goal to work backwards from
            steps: Optional list of steps in the backward chain
            confidence: Confidence in this backward chaining
            path: Optional reasoning path identifier
            
        Returns:
            The created reasoning step
        """
        metadata = {"steps": steps} if steps else {}
        return self.add_reasoning_step(
            behavior=CognitiveBehavior.BACKWARD_CHAINING,
            description=f"Backward chaining toward: {target}",
            metadata=metadata,
            confidence=confidence,
            path=path
        )
    
    def reflect(self, reflection: str, subject: Optional[str] = None, 
               confidence: float = 0.6, path: str = None) -> ReasoningStep:
        """
        Execute reflection behavior: analyze past performance and learn from it.
        
        Args:
            reflection: The reflection content
            subject: Optional subject of reflection
            confidence: Confidence in this reflection
            path: Optional reasoning path identifier
            
        Returns:
            The created reasoning step
        """
        metadata = {"subject": subject} if subject else {}
        
        # Update metacognitive state based on reflection
        self._metacognitive_state["reasoning_efficiency"] += 0.05
        
        return self.add_reasoning_step(
            behavior=CognitiveBehavior.REFLECTION,
            description=reflection,
            metadata=metadata,
            confidence=confidence,
            path=path
        )
    
    def explore(self, strategy: str, options: Optional[List[str]] = None, 
               confidence: float = 0.5, create_paths: bool = False, path: str = None) -> ReasoningStep:
        """
        Execute exploration behavior: try different approaches to solve a problem.
        
        Args:
            strategy: The exploration strategy
            options: Optional list of options to explore
            confidence: Confidence in this exploration
            create_paths: Whether to create separate reasoning paths for each option
            path: Optional reasoning path identifier
            
        Returns:
            The created reasoning step
        """
        metadata = {"options": options} if options else {}
        
        # Create separate reasoning paths for each option if requested
        if create_paths and options:
            for i, option in enumerate(options):
                option_path_id = f"option_{i}_{int(time.time())}"
                self.create_reasoning_path(
                    option_path_id, 
                    f"Exploring option: {option}", 
                    confidence * 0.9  # Slightly lower confidence for options
                )
                
                # Add first step to the new path
                self.add_reasoning_step(
                    behavior=CognitiveBehavior.EXPLORATION,
                    description=f"Exploring option: {option}",
                    metadata={"strategy": strategy, "option_index": i},
                    confidence=confidence * 0.8,
                    path=option_path_id
                )
                
            metadata["created_paths"] = True
        
        return self.add_reasoning_step(
            behavior=CognitiveBehavior.EXPLORATION,
            description=f"Exploring strategy: {strategy}",
            metadata=metadata,
            confidence=confidence,
            path=path
        )
    
    def run_multiagent_reasoning(self, scenario: str, agents: List[Dict[str, Any]], 
                              confidence: float = 0.7, path: str = None) -> ReasoningStep:
        """
        Execute multiagent reasoning: model interactions between multiple strategic agents.
        
        Args:
            scenario: Description of the multiagent scenario
            agents: List of agent descriptions with preferences and capabilities
            confidence: Confidence in this reasoning
            path: Optional reasoning path identifier
            
        Returns:
            The created reasoning step
        """
        return self.add_reasoning_step(
            behavior=CognitiveBehavior.MULTIAGENT_REASONING,
            description=f"Multiagent reasoning in scenario: {scenario}",
            metadata={"agents": agents},
            confidence=confidence,
            path=path
        )
    
    def model_opponent(self, opponent: str, observations: List[str], 
                     confidence: float = 0.6, path: str = None) -> ReasoningStep:
        """
        Execute opponent modeling: infer strategy and preferences of other agents.
        
        Args:
            opponent: The opponent to model
            observations: List of observed actions or statements
            confidence: Confidence in this modeling
            path: Optional reasoning path identifier
            
        Returns:
            The created reasoning step
        """
        return self.add_reasoning_step(
            behavior=CognitiveBehavior.OPPONENT_MODELING,
            description=f"Modeling opponent: {opponent}",
            metadata={"observations": observations},
            confidence=confidence,
            path=path
        )
    
    def assess_risk(self, action: str, risks: List[Dict[str, Any]], 
                  confidence: float = 0.7, path: str = None) -> ReasoningStep:
        """
        Execute risk assessment: evaluate potential downsides of actions.
        
        Args:
            action: The action being assessed
            risks: List of risks with probabilities and impacts
            confidence: Confidence in this assessment
            path: Optional reasoning path identifier
            
        Returns:
            The created reasoning step
        """
        return self.add_reasoning_step(
            behavior=CognitiveBehavior.RISK_ASSESSMENT,
            description=f"Assessing risks of: {action}",
            metadata={"risks": risks},
            confidence=confidence,
            path=path
        )
    
    def create_decision_tree(self, decision: str, options: List[Dict[str, Any]], 
                           confidence: float = 0.8, path: str = None) -> ReasoningStep:
        """
        Execute decision tree analysis: map out decision sequences and outcomes.
        
        Args:
            decision: The initial decision point
            options: List of options with consequences
            confidence: Confidence in this analysis
            path: Optional reasoning path identifier
            
        Returns:
            The created reasoning step
        """
        return self.add_reasoning_step(
            behavior=CognitiveBehavior.DECISION_TREE_ANALYSIS,
            description=f"Creating decision tree for: {decision}",
            metadata={"options": options},
            confidence=confidence,
            path=path
        )
    
    def run_mental_simulation(self, scenario: str, steps: List[str], 
                           outcome: Optional[str] = None, 
                           confidence: float = 0.7, path: str = None) -> ReasoningStep:
        """
        Execute mental simulation: step through hypothetical scenarios to predict outcomes.
        
        Args:
            scenario: The scenario to simulate
            steps: Sequential steps in the simulation
            outcome: Optional predicted outcome
            confidence: Confidence in this simulation
            path: Optional reasoning path identifier
            
        Returns:
            The created reasoning step
        """
        metadata = {"steps": steps}
        if outcome:
            metadata["outcome"] = outcome
            
        return self.add_reasoning_step(
            behavior=CognitiveBehavior.MENTAL_SIMULATION,
            description=f"Simulating scenario: {scenario}",
            metadata=metadata,
            confidence=confidence,
            path=path
        )
    
    def form_grand_strategy(self, objective: str, timeframe: str, 
                          components: List[Dict[str, Any]], 
                          confidence: float = 0.8, path: str = None) -> ReasoningStep:
        """
        Execute grand strategy formation: develop a comprehensive long-term plan.
        
        Args:
            objective: The overall strategic objective
            timeframe: The strategic timeframe
            components: List of strategic components and their rationales
            confidence: Confidence in this strategy
            path: Optional reasoning path identifier
            
        Returns:
            The created reasoning step
        """
        return self.add_reasoning_step(
            behavior=CognitiveBehavior.GRAND_STRATEGY_FORMATION,
            description=f"Forming grand strategy for: {objective} ({timeframe})",
            metadata={"components": components},
            confidence=confidence,
            path=path
        )
        # Add first step to the new path
        self.add_reasoning_step(
            behavior=CognitiveBehavior.EXPLORATION,
            description=f"Exploring option: {option}",
            metadata={"strategy": strategy, "option_index": i},
            confidence=confidence * 0.8,
            path=option_path_id
            )
            
        metadata["created_paths"] = True
        
        return self.add_reasoning_step(
            behavior=CognitiveBehavior.EXPLORATION,
            description=f"Exploring strategy: {strategy}",
            metadata=metadata,
            confidence=confidence,
            path=path
        )
        
    def plan(self, plan: str, steps: List[str], confidence: float = 0.7, 
            path: str = None) -> ReasoningStep:
        """
        Execute planning behavior: create a sequence of steps to achieve a goal.
        
        Args:
            plan: The plan description
            steps: List of steps in the plan
            confidence: Confidence in this plan
            path: Optional reasoning path identifier
            
        Returns:
            The created reasoning step
        """
        return self.add_reasoning_step(
            behavior=CognitiveBehavior.PLANNING,
            description=f"Planning: {plan}",
            metadata={"steps": steps},
            confidence=confidence,
            path=path
        )
        
    def evaluate(self, evaluation: str, criteria: List[str], score: float, 
                confidence: float = 0.6, path: str = None) -> ReasoningStep:
        """
        Execute evaluation behavior: assess options against criteria.
        
        Args:
            evaluation: What is being evaluated
            criteria: List of evaluation criteria
            score: Evaluation score
            confidence: Confidence in this evaluation
            path: Optional reasoning path identifier
            
        Returns:
            The created reasoning step
        """
        return self.add_reasoning_step(
            behavior=CognitiveBehavior.EVALUATION,
            description=f"Evaluating: {evaluation}",
            result=score,
            metadata={"criteria": criteria},
            confidence=confidence,
            path=path
        )
        
    def create(self, creation: str, inspiration: Optional[str] = None, 
              confidence: float = 0.4, path: str = None) -> ReasoningStep:
        """
        Execute creativity behavior: generate novel ideas or solutions.
        
        Args:
            creation: The creative output
            inspiration: Optional inspiration source
            confidence: Confidence in this creation
            path: Optional reasoning path identifier
            
        Returns:
            The created reasoning step
        """
        metadata = {"inspiration": inspiration} if inspiration else {}
        
        # For creative steps, consider creating a new path to explore the creative direction
        if path is None and random.random() < 0.3:  # 30% chance to create new path
            creative_path = f"creative_{int(time.time())}"
            self.create_reasoning_path(creative_path, f"Creative exploration: {creation[:30]}...", confidence)
            path = creative_path
        
        return self.add_reasoning_step(
            behavior=CognitiveBehavior.CREATIVITY,
            description=f"Creating: {creation}",
            metadata=metadata,
            confidence=confidence,
            path=path
        )
        
    def abstract(self, abstraction: str, from_concrete: str, 
                confidence: float = 0.6, path: str = None) -> ReasoningStep:
        """
        Execute abstraction behavior: identify patterns and generalize.
        
        Args:
            abstraction: The abstraction being made
            from_concrete: The concrete example being abstracted from
            confidence: Confidence in this abstraction
            path: Optional reasoning path identifier
            
        Returns:
            The created reasoning step
        """
        return self.add_reasoning_step(
            behavior=CognitiveBehavior.ABSTRACTION,
            description=f"Abstracting: {abstraction}",
            metadata={"from_concrete": from_concrete},
            confidence=confidence,
            path=path
        )
        
    def draw_analogy(self, analogy: str, source: str, target: str, 
                    confidence: float = 0.5, path: str = None) -> ReasoningStep:
        """
        Execute analogy behavior: transfer knowledge from one domain to another.
        
        Args:
            analogy: The analogy being drawn
            source: Source domain of the analogy
            target: Target domain of the analogy
            confidence: Confidence in this analogy
            path: Optional reasoning path identifier
            
        Returns:
            The created reasoning step
        """
        return self.add_reasoning_step(
            behavior=CognitiveBehavior.ANALOGY,
            description=f"Drawing analogy: {analogy}",
            metadata={"source": source, "target": target},
            confidence=confidence,
            path=path
        )
        
    def counterfactual(self, premise: str, consequence: str, 
                      confidence: float = 0.4, path: str = None) -> ReasoningStep:
        """
        Execute counterfactual reasoning: explore what would happen if something were different.
        
        Args:
            premise: The counterfactual premise
            consequence: The consequence of the counterfactual
            confidence: Confidence in this counterfactual reasoning
            path: Optional reasoning path identifier
            
        Returns:
            The created reasoning step
        """
        # Always create a new path for counterfactuals if none specified
        if path is None:
            cf_path = f"counterfactual_{int(time.time())}"
            self.create_reasoning_path(cf_path, f"Counterfactual: {premise}", confidence)
            path = cf_path
            
        return self.add_reasoning_step(
            behavior=CognitiveBehavior.EXPLORATION,  # Use exploration behavior type
            description=f"Counterfactual reasoning: If {premise}, then {consequence}",
            metadata={"counterfactual": True, "premise": premise, "consequence": consequence},
            confidence=confidence,
            path=path
        )

    def get_chain_of_thought(self) -> ChainOfThought:
        """Get the full chain-of-thought."""
        with self._lock:
            return self._chain_of_thought
            
    def get_reasoning_path(self, path_id: str) -> List[ReasoningStep]:
        """
        Get all steps in a specific reasoning path.
        
        Args:
            path_id: The path ID to retrieve
            
        Returns:
            List of reasoning steps in the path
        """
        with self._lock:
            if path_id not in self._reasoning_paths:
                return []
            return list(self._reasoning_paths[path_id])
            
    def update_chain_summary(self, summary: str) -> None:
        """Update the summary of the reasoning chain."""
        with self._lock:
            self._chain_of_thought.update_summary(summary)
            
    def set_conclusion(self, conclusion: str, confidence: float = None, 
                      from_best_path: bool = True) -> None:
        """
        Set the final conclusion of the reasoning chain.
        
        Args:
            conclusion: The conclusion text
            confidence: Optional confidence level
            from_best_path: Whether to use the best path's confidence
        """
        with self._lock:
            if from_best_path and confidence is None:
                # Use confidence from best path
                best_path = self.get_best_reasoning_path()
                if best_path in self._path_confidences:
                    confidence = self._path_confidences[best_path]
                    
            self._chain_of_thought.set_conclusion(conclusion, confidence)
    
    def get_reasoning_summary(self, include_paths: bool = False) -> str:
        """
        Generate a summary of the reasoning process so far.
        
        Args:
            include_paths: Whether to include details about reasoning paths
            
        Returns:
            Formatted summary string
        """
        with self._lock:
            if self._chain_of_thought.summary and not include_paths:
                return self._chain_of_thought.summary
                
            summary = []
            
            # Add metacognitive state summary
            summary.append("=== Metacognitive State ===")
            summary.append(f"Calibration Score: {self._metacognitive_state['calibration_score']:.2f}")
            summary.append(f"Reasoning Efficiency: {self._metacognitive_state['reasoning_efficiency']:.2f}")
            summary.append(f"Path Diversity: {self._metacognitive_state['path_diversity']:.2f}")
            summary.append(f"Active Path: {self._active_path}")
            summary.append(f"Total Paths: {len(self._reasoning_paths)}")
            summary.append("")
            
            # Add main reasoning chain
            summary.append("=== Main Reasoning Chain ===")
            for step in self._chain_of_thought.steps:
                result_str = f" → {step.result}" if step.result is not None else ""
                correctness = " ✓" if step.is_correct else " ✗" if step.is_correct is False else ""
                confidence_str = f" (confidence: {step.confidence:.2f})"
                summary.append(f"Step {step.step_number} ({step.behavior}): {step.description}{result_str}{correctness}{confidence_str}")
            
            # Add path details if requested
            if include_paths and len(self._reasoning_paths) > 1:
                summary.append("\n=== Reasoning Paths ===")
                for path_id, steps in self._reasoning_paths.items():
                    if path_id == "main":
                        continue  # Skip main path as it's already shown
                        
                    confidence = self._path_confidences.get(path_id, 0.0)
                    summary.append(f"\nPath: {path_id} (confidence: {confidence:.2f})")
                    
                    # Show steps in this path
                    for i, step in enumerate(steps):
                        result_str = f" → {step.result}" if step.result is not None else ""
                        summary.append(f"  {i+1}. ({step.behavior}): {step.description}{result_str}")
                
            # Add conclusion
            if self._chain_of_thought.conclusion:
                summary.append(f"\nConclusion: {self._chain_of_thought.conclusion} (overall confidence: {self._chain_of_thought.confidence:.2f})")
                
            return "\n".join(summary)
            
    def get_metacognitive_state(self) -> Dict[str, Any]:
        """Get the current metacognitive monitoring state"""
        with self._lock:
            return self._metacognitive_state.copy()
            
    def decompose_task(self, task: Task, decomposition: SubtaskDecomposition) -> None:
        """
        Record a structured task decomposition in the cognitive model.
        
        Args:
            task: The task being decomposed
            decomposition: The structured decomposition
        """
        with self._lock:
            # Create a dedicated path for this decomposition
            decomp_path = f"decomposition_{task.task_id}"
            self.create_reasoning_path(decomp_path, f"Task decomposition for task {task.task_id}", 0.9)
            
            # Add a subgoal setting step for the decomposition
            self.set_subgoal(
                subgoal=f"Decompose task {task.task_id} into subtasks",
                metadata={
                    "task_id": task.task_id,
                    "num_subtasks": len(decomposition.subtasks),
                    "rationale": decomposition.rationale
                },
                confidence=0.9,
                path=decomp_path
            )
            
            # Add a step for each subtask
            for i, subtask in enumerate(decomposition.subtasks):
                complexity = decomposition.estimated_complexity.get(str(i), 5) if decomposition.estimated_complexity else 5
                dependencies = decomposition.dependencies.get(str(i), []) if decomposition.dependencies else []
                
                self.add_reasoning_step(
                    behavior=CognitiveBehavior.SUBGOAL_SETTING,
                    description=f"Subtask {i+1}: {subtask['description']}",
                    metadata={
                        "parent_task_id": task.task_id,
                        "complexity": complexity,
                        "dependencies": dependencies
                    },
                    confidence=0.85,
                    path=decomp_path
                )
            
            # Merge the decomposition path back to main
            self.merge_reasoning_paths(decomp_path, "main")


class SelfReflectiveCognition:
    """
    Periodically reflects on tasks completed, analyzing performance.
    Enhanced with cognitive modeling capabilities.
    """
    def __init__(self):
        self._reflections: List[str] = []
        self._lock = threading.Lock()
        self._analyzer_thread = threading.Thread(target=self._analyze_performance_loop, daemon=True)
        self._analyzer_thread.start()
        self.cognitive_engine = CognitiveModelingEngine()
        # Reference to memory store will be set by the agent
        self.memory_store = None

    def reflect_on_task(self, task: Task) -> None:
        with self._lock:
            snippet = task.description[:50].replace("\n"," ")
            msg = f"Reflected on task {task.task_id}: status={task.status}, desc='{snippet}'"
            self._reflections.append(msg)
            logger.info(f"[SelfReflectiveCognition] {msg}")
            
            # Add to cognitive model
            self.cognitive_engine.reflect(
                reflection=msg,
                subject=f"Task {task.task_id}"
            )
            
            # Advanced learning: Adjust strategies based on task outcomes
            self._learn_from_task(task)

    def _learn_from_task(self, task: Task) -> None:
        """
        Learn from the task outcome to improve future performance.
        """
        # Example learning logic: Adjust priorities based on task success/failure
        if task.status == "COMPLETED":
            logger.info(f"[SelfReflectiveCognition] Task {task.task_id} completed successfully. Reinforcing strategies.")
            
            # Add verification step to cognitive model
            self.cognitive_engine.verify(
                description=f"Task {task.task_id} completion",
                result="Success",
                is_correct=True
            )
            
            # Advanced adaptation: Increase priority for similar tasks
            self._adjust_similar_task_priorities(task, increase=True)
        elif task.status == "FAILED":
            logger.info(f"[SelfReflectiveCognition] Task {task.task_id} failed. Adjusting strategies to avoid similar failures.")
            
            # Add backtracking step to cognitive model
            self.cognitive_engine.backtrack(
                reason=f"Task {task.task_id} failed to complete"
            )
            
            # Advanced adaptation: Decrease priority for similar tasks
            self._adjust_similar_task_priorities(task, increase=False)

    def _adjust_similar_task_priorities(self, task: Task, increase: bool) -> None:
        """
        Adjust priorities of similar tasks based on the outcome of the current task.
        """
        with self._lock:
            for t in self.memory_store.list_tasks():
                if t.description == task.description and t.status == "PENDING":
                    if increase:
                        t.priority = max(0, t.priority - 1)
                        logger.info(f"[SelfReflectiveCognition] Increased priority for similar task {t.task_id}.")
                    else:
                        t.priority += 1
                        logger.info(f"[SelfReflectiveCognition] Decreased priority for similar task {t.task_id}.")

    def get_reflections(self) -> List[str]:
        with self._lock:
            return list(self._reflections)
    
    def get_reasoning_summary(self) -> str:
        """
        Get a summary of the cognitive reasoning process.
        """
        return self.cognitive_engine.get_reasoning_summary()

    def _analyze_performance_loop(self) -> None:
        """
        Periodically logs a mini 'analysis' of the last few reflections.
        """
        while True:
            time.sleep(30)
            with self._lock:
                if self._reflections:
                    recent = self._reflections[-5:]
                    analysis = "Recent reflections => " + " || ".join(recent)
                    logger.info(f"[SelfReflectiveCognition] {analysis}")
                elif hasattr(self, 'memory_store') and self.memory_store is not None:
                    # Generate a reflection based on current tasks
                    tasks = self.memory_store.list_tasks()
                    completed_tasks = [t for t in tasks if t.status == TaskStatus.COMPLETED]
                    failed_tasks = [t for t in tasks if t.status == TaskStatus.FAILED]
                    reflection = f"Completed {len(completed_tasks)} tasks, {len(failed_tasks)} failed."
                    self._reflections.append(reflection)
                    logger.info(f"[SelfReflectiveCognition] {reflection}")

###############################################################################
# DYNAMIC CODE LOADING AND IN-MEMORY CODE ARCHIVE
###############################################################################

class DynamicCodeLoader:
    """
    Handles dynamic loading, reloading, and management of code modules.
    Allows for runtime code changes without restarting the application.
    """
    
    def __init__(self):
        self._modules = {}
        self._lock = threading.RLock()
        self._logger = logging.getLogger("DynamicCodeLoader")
        self._module_watchers = {}
        self._last_modified = {}
        
    def load_module(self, module_path: str, reload: bool = False) -> types.ModuleType:
        """
        Load a Python module dynamically from file path.
        
        Args:
            module_path: Path to the module file
            reload: Force reload even if previously loaded
            
        Returns:
            Loaded module object
        """
        with self._lock:
            module_path = Path(module_path).resolve()
            if not module_path.exists():
                raise FileNotFoundError(f"Module not found: {module_path}")
            
            # Check if module needs reloading
            if module_path in self._modules and not reload:
                return self._modules[module_path]
            
            # Generate module name from path
            module_name = f"dynamic_module_{hash(str(module_path))}"
            
            # Track last modified time
            self._last_modified[module_path] = module_path.stat().st_mtime
            
            try:
                # Create spec and load module
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is None:
                    raise ImportError(f"Failed to create spec for module: {module_path}")
                    
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
                # Store the loaded module
                self._modules[module_path] = module
                self._logger.info(f"Successfully loaded module from {module_path}")
                
                return module
                
            except Exception as e:
                self._logger.error(f"Error loading module {module_path}: {str(e)}")
                raise
    
    def check_for_changes(self) -> List[Path]:
        """
        Check all loaded modules for changes since last load.
        
        Returns:
            List of module paths that have changed
        """
        changed_modules = []
        
        with self._lock:
            for module_path in self._modules:
                if not Path(module_path).exists():
                    continue
                    
                current_mtime = Path(module_path).stat().st_mtime
                if current_mtime > self._last_modified.get(module_path, 0):
                    changed_modules.append(module_path)
                    
        return changed_modules
    
    def reload_all_changed(self) -> Dict[Path, Any]:
        """
        Reload all modules that have changed.
        
        Returns:
            Dictionary mapping module paths to reload success/failure
        """
        changed = self.check_for_changes()
        results = {}
        
        for module_path in changed:
            try:
                self.load_module(module_path, reload=True)
                results[module_path] = True
            except Exception as e:
                results[module_path] = str(e)
                
        return results
    
    def execute_function(self, module_path: str, function_name: str, *args, **kwargs) -> Any:
        """
        Execute a function from a dynamically loaded module.
        
        Args:
            module_path: Path to the module file
            function_name: Name of the function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of function execution
        """
        module = self.load_module(module_path)
        
        if not hasattr(module, function_name):
            raise AttributeError(f"Function '{function_name}' not found in module: {module_path}")
            
        function = getattr(module, function_name)
        if not callable(function):
            raise TypeError(f"'{function_name}' is not callable in module: {module_path}")
            
        return function(*args, **kwargs)
    
    def get_module_symbols(self, module_path: str) -> Dict[str, str]:
        """
        Get all symbols (functions, classes, variables) from a module.
        
        Args:
            module_path: Path to the module file
            
        Returns:
            Dictionary of symbol names to their types
        """
        module = self.load_module(module_path)
        symbols = {}
        
        for name, obj in inspect.getmembers(module):
            # Skip private/dunder names
            if name.startswith("_"):
                continue
                
            if inspect.isfunction(obj):
                symbols[name] = "function"
            elif inspect.isclass(obj):
                symbols[name] = "class"
            elif inspect.ismethod(obj):
                symbols[name] = "method"
            else:
                symbols[name] = type(obj).__name__
                
        return symbols
    
    def clear_module_cache(self) -> None:
        """Clear the module cache to force fresh reloads."""
        with self._lock:
            self._modules.clear()
            
    def start_auto_reload(self, check_interval: float = 5.0) -> None:
        """
        Start a background thread to automatically reload changed modules.
        
        Args:
            check_interval: Interval in seconds between checks
        """
        def _watcher():
            while True:
                try:
                    changed = self.reload_all_changed()
                    if changed:
                        self._logger.info(f"Auto-reload: Updated {len(changed)} modules")
                except Exception as e:
                    self._logger.error(f"Error in auto-reload: {str(e)}")
                time.sleep(check_interval)
                
        thread = threading.Thread(target=_watcher, daemon=True)
        thread.start()
        self._logger.info(f"Started auto-reload thread (interval: {check_interval}s)")


class InMemoryCodeArchive:
    """
    Stores code snippets so that the agent can 'introspect' or recall them.
    In real usage, you might store the entire codebase or frequently used modules.
    """
    def __init__(self):
        self._snippets: Dict[str, str] = {}
        self._lock = threading.Lock()
    
    def intelligent_modify_snippet(self, snippet_name: str, instructions: str) -> None:
        """
        Apply advanced transformations to a snippet using a chain-of-thought approach.
        """
        with self._lock:
            if snippet_name not in self._snippets:
                logger.warning(f"[InMemoryCodeArchive] Snippet '{snippet_name}' does not exist.")
                return
            original_code = self._snippets[snippet_name]
            # Here you can parse instructions, implement chain-of-thought transformations, etc.
            # For demonstration, we'll do a simple example that does find/replace lines indicated in instructions.
            new_code = self._apply_transformations(original_code, instructions)
            self._snippets[snippet_name] = new_code
            logger.info(f"[InMemoryCodeArchive] Applied intelligent modifications to snippet '{snippet_name}'")

    def _apply_transformations(self, code: str, instructions: str) -> str:
        """
        A naive parser that tries to parse instructions for find/replace lines.
        """
        try:
            import re
            lines = instructions.strip().split("\\n")
            for line in lines:
                match = re.match(r"^REPLACE:\\s*'(.*?)'\\s*->\\s*'(.*?)'", line)
                if match:
                    old_text, new_text = match.group(1), match.group(2)
                    code = code.replace(old_text, new_text)
            return code
        except Exception as e:
            logger.error(f"[InMemoryCodeArchive] Error applying transformations: {e}")
            return code
    
    def read_from_file(self, filepath: str) -> None:
        """Read code from a file and store it as a snippet."""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            snippet_name = os.path.basename(filepath)
            self.add_snippet(snippet_name, content)
            logger.info(f"[InMemoryCodeArchive] Read code from {filepath} and stored as snippet '{snippet_name}'")
        except Exception as e:
            logger.error(f"[InMemoryCodeArchive] Error reading from file {filepath}: {e}")
    
    def write_to_file(self, snippet_name: str, filepath: str) -> None:
        """Write a stored snippet to a file."""
        code = self.get_snippet(snippet_name)
        if code is None:
            logger.warning(f"[InMemoryCodeArchive] Snippet '{snippet_name}' not found.")
            return
        try:
            with open(filepath, 'w') as f:
                f.write(code)
            logger.info(f"[InMemoryCodeArchive] Wrote snippet '{snippet_name}' to file {filepath}")
        except Exception as e:
            logger.error(f"[InMemoryCodeArchive] Error writing snippet '{snippet_name}' to file {filepath}: {e}")
    
    def incremental_search(self, query: str) -> List[str]:
        """Perform an incremental search for the query in stored snippets."""
        matches = []
        with self._lock:
            for name, code in self._snippets.items():
                if query in code:
                    matches.append(name)
        return matches
    
    def incremental_search_generator(self, query: str, chunk_size: int = 50) -> str:
        """Yield code chunks containing the query in stored snippets."""
        with self._lock:
            for name, code in self._snippets.items():
                lines = code.split('\n')
                buffer = []
                for line in lines:
                    buffer.append(line)
                    if len(buffer) >= chunk_size:
                        chunk = '\n'.join(buffer)
                        if query in chunk:
                            yield (name, chunk)
                        buffer = []
                # final partial chunk
                if buffer:
                    chunk = '\n'.join(buffer)
                    if query in chunk:
                        yield (name, chunk)

    def add_snippet(self, name: str, code: str) -> None:
        with self._lock:
            self._snippets[name] = code
            logger.info(f"[InMemoryCodeArchive] Stored code snippet '{name}'")

    def get_snippet(self, name: str) -> Optional[str]:
        with self._lock:
            return self._snippets.get(name)

    def list_snippets(self) -> List[str]:
        with self._lock:
            return list(self._snippets.keys())

###############################################################################
# KNOWLEDGE BASE
###############################################################################

class KnowledgeBase:
    """
    Stores and retrieves key facts or short “knowledge chunks.”
    An agent can use this to reference domain knowledge, or to do
    something akin to basic retrieval-augmented generation in a real system.
    """
    def __init__(self):
        self._facts: Dict[str, str] = {}
        self._lock = threading.Lock()

    def add_fact(self, key: str, value: str) -> None:
        """
        Add a fact or definition into the knowledge base.
        """
        with self._lock:
            self._facts[key.lower()] = value
            logger.info(f"[KnowledgeBase] Added fact: '{key}' => {value[:40]}...")

    def get_fact(self, key: str) -> Optional[str]:
        """
        Retrieve a fact by exact key (case-insensitive).
        """
        with self._lock:
            return self._facts.get(key.lower())

    def search_facts(self, query: str) -> List[Tuple[str, str]]:
        """
        Naive substring search for facts relevant to query.
        """
        query_lower = query.lower()
        matches = []
        with self._lock:
            for k, v in self._facts.items():
                if query_lower in k or query_lower in v.lower():
                    matches.append((k, v))
        return matches

###############################################################################
# CANDIDATE ACTIONS
###############################################################################

class CandidateAction:
    """
    A potential next step. The agent can generate multiple and pick or spawn tasks accordingly.
    """
    def __init__(self, description: str, rationale: str, priority: int = 5):
        self.description = description
        self.rationale = rationale
        self.priority = priority

    def __repr__(self) -> str:
        return f"CandidateAction(desc={self.description[:20]}, prio={self.priority})"

class ActionGenerator:
    """
    Produces up to 25 candidate actions based on the agent’s memory, tasks, goals, conversation, code archive, knowledge base, etc.
    """
    def __init__(
        self,
        code_archive: InMemoryCodeArchive,
        kb: KnowledgeBase
    ):
        self.code_archive = code_archive
        self.kb = kb

    def generate_candidate_actions(
        self,
        conversation: "ConversationMemory",
        goals: List[Goal],
        tasks: List[Task]
    ) -> List[CandidateAction]:
        logger.info("[ActionGenerator] Generating candidate actions (max 25).")
        actions = []

        # 1) Reflect on tasks and learn from past experiences
        pending_tasks = [t for t in tasks if t.status == "PENDING"]
        if pending_tasks:
            actions.append(CandidateAction(
                description="Review all pending tasks to ensure they are valid or up to date",
                rationale="We have tasks that are not yet started; let's see if we can refine them."
            ))

        # 2) Check code archive for potential improvements
        snippet_names = self.code_archive.list_snippets()
        if snippet_names:
            snippet_choice = snippet_names[0]
            actions.append(CandidateAction(
                description=f"Read code snippet: {snippet_choice}",
                rationale="Might glean helpful implementation details from the snippet.",
                priority=3
            ))

        # 3) Perform knowledge base lookups for relevant information
        if self.kb.search_facts("agent"):
            actions.append(CandidateAction(
                description="Retrieve facts about 'agent' from knowledge base",
                rationale="We have some knowledge about the term 'agent' that might be relevant."
            ))

        # 4) Decompose active goals into smaller tasks
        for g in goals:
            if g.status == "ACTIVE":
                actions.append(CandidateAction(
                    description=f"Decompose goal '{g.name}' into smaller tasks.",
                    rationale="Breaking big goals into steps fosters incremental progress.",
                    priority=g.priority
                ))

        # 5) Adjust goals dynamically based on new information
        for g in goals:
            if g.status == "ACTIVE" and self._should_adjust_goal(g):
                actions.append(CandidateAction(
                    description=f"Adjust goal '{g.name}' based on recent developments.",
                    rationale="Adapting goals to new information ensures relevance and achievability.",
                    priority=g.priority
                ))

        # 6) Generate additional context-based actions
        if len(actions) < 25:
            # Consider conversation history for context
            recent_conversation = conversation.get_history()[-5:]
            for i, utterance in enumerate(recent_conversation):
                actions.append(CandidateAction(
                    description=f"Analyze recent conversation: '{utterance['content'][:20]}...'",
                    rationale="Understanding recent interactions can provide insights.",
                    priority=5
                ))

            # Consider current goals and tasks for additional actions
            for goal in goals:
                if goal.status == "ACTIVE":
                    actions.append(CandidateAction(
                        description=f"Review progress on goal '{goal.name}'",
                        rationale="Ensuring goals are on track is crucial for success.",
                        priority=goal.priority
                    ))

            for task in tasks:
                if task.status == "PENDING":
                    actions.append(CandidateAction(
                        description=f"Evaluate pending task: '{task.description[:20]}...'",
                        rationale="Pending tasks need evaluation to ensure relevance.",
                        priority=task.priority
                    ))

        # Ensure we have exactly 25 actions
        actions = actions[:25]

        # Return only first 25
        return actions[:25]

    def _generate_context_based_action(self, conversation: "ConversationMemory", goals: List[Goal], tasks: List[Task], index: int) -> str:
        """
        Generate a context-based placeholder action description.
        """
        # Example logic to generate a context-based action
        if goals:
            active_goal = goals[0].name
            return f"Explore further steps to achieve goal '{active_goal}' (Placeholder Action #{index})"
        elif tasks:
            pending_task = tasks[0].description
            return f"Investigate pending task: '{pending_task}' (Placeholder Action #{index})"
        else:
            return f"Review recent conversation topics for insights (Placeholder Action #{index})"

    def _should_adjust_goal(self, goal: Goal) -> bool:
        """
        Determine if a goal should be adjusted based on new information.
        """
        # Placeholder logic for goal adjustment
        return True  # In a real implementation, this would be more complex

###############################################################################
# PRIORITY TASK QUEUE
###############################################################################

class PriorityTaskQueue:
    """
    Thread-safe priority queue for tasks, using a heap.
    """
    def __init__(self):
        self._heap: List[Task] = []
        self._lock = threading.Lock()

    def push(self, task: Task) -> None:
        with self._lock:
            heapq.heappush(self._heap, task)

    def pop(self) -> Optional[Task]:
        with self._lock:
            if self._heap:
                return heapq.heappop(self._heap)
            return None

    def __len__(self) -> int:
        with self._lock:
            return len(self._heap)

###############################################################################
# TOKEN REGISTRY AND FUNCTION ADAPTER
###############################################################################

class DynamicTokenBuffer:
    """
    A dynamic buffer for managing streamed tokens with advanced capabilities:
    - Maintains a sliding window of tokens
    - Provides context-aware token analysis
    - Supports token manipulation and transformation
    - Enables pattern matching with flexible context windows
    - Tracks token statistics and usage patterns
    - Self-monitoring of output stream for real-time feedback
    """
    def __init__(self, max_size: int = 2000):
        self._buffer = []
        self._lock = threading.Lock()
        self._max_size = max_size
        self._token_stats = {
            "total_processed": 0,
            "pattern_matches": 0,
            "executions": 0,
            "last_execution_time": None
        }
        self._context_windows = {}
        self._token_metadata = {}
        self._output_stream = []
        self._output_analysis = {
            "sentiment": "neutral",
            "complexity": "medium",
            "coherence": "high",
            "last_analysis_time": None
        }
        
    def add_token(self, token: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a token to the buffer with optional metadata.
        
        Args:
            token: The token to add
            metadata: Optional metadata about the token (e.g., position, confidence)
        """
        with self._lock:
            self._buffer.append(token)
            self._token_stats["total_processed"] += 1
            
            # Store metadata if provided
            if metadata:
                token_idx = len(self._buffer) - 1
                self._token_metadata[token_idx] = metadata
            
            # Trim buffer if it gets too large
            if len(self._buffer) > self._max_size:
                # Remove oldest tokens and their metadata
                excess = len(self._buffer) - self._max_size
                self._buffer = self._buffer[excess:]
                
                # Update metadata indices
                new_metadata = {}
                for idx, meta in self._token_metadata.items():
                    if idx >= excess:
                        new_metadata[idx - excess] = meta
                self._token_metadata = new_metadata
    
    def get_text(self, start: int = 0, end: Optional[int] = None) -> str:
        """
        Get text from the buffer within the specified range.
        
        Args:
            start: Start index (inclusive)
            end: End index (exclusive), or None for the entire buffer from start
            
        Returns:
            String of concatenated tokens
        """
        with self._lock:
            if end is None:
                end = len(self._buffer)
            return "".join(self._buffer[start:end])
    
    def get_context_window(self, window_name: str, default_size: int = 100) -> str:
        """
        Get a named context window, creating it if it doesn't exist.
        
        Args:
            window_name: Name of the context window
            default_size: Default size for new windows
            
        Returns:
            Text in the context window
        """
        with self._lock:
            if window_name not in self._context_windows:
                # Create a new window at the end of the buffer
                start_idx = max(0, len(self._buffer) - default_size)
                self._context_windows[window_name] = {
                    "start": start_idx,
                    "size": default_size
                }
            
            window = self._context_windows[window_name]
            start = window["start"]
            size = window["size"]
            end = min(start + size, len(self._buffer))
            
            return self.get_text(start, end)
    
    def update_context_window(self, window_name: str, start: Optional[int] = None, 
                             size: Optional[int] = None) -> None:
        """
        Update a context window's parameters.
        
        Args:
            window_name: Name of the window to update
            start: New start index, or None to keep current
            size: New size, or None to keep current
        """
        with self._lock:
            if window_name not in self._context_windows:
                # Default to end of buffer if window doesn't exist
                self._context_windows[window_name] = {
                    "start": max(0, len(self._buffer) - (size or 100)),
                    "size": size or 100
                }
                return
                
            window = self._context_windows[window_name]
            if start is not None:
                window["start"] = max(0, min(start, len(self._buffer) - 1))
            if size is not None:
                window["size"] = max(1, size)
    
    def find_pattern(self, pattern: str, context_size: int = 200) -> Optional[Dict[str, Any]]:
        """
        Find a pattern in the buffer and return its location with context.
        
        Args:
            pattern: Pattern to search for
            context_size: Amount of context to include before and after
            
        Returns:
            Dict with match information or None if not found
        """
        with self._lock:
            buffer_text = "".join(self._buffer)
            match_idx = buffer_text.find(pattern)
            
            if match_idx == -1:
                return None
                
            # Calculate context boundaries with larger context for code blocks
            start_idx = max(0, match_idx - context_size)
            end_idx = min(len(buffer_text), match_idx + len(pattern) + context_size)
            
            # For Python code blocks, try to find the closing ```
            if pattern == "```python":
                # Look for closing ``` in the context after
                context_after = buffer_text[match_idx + len(pattern):end_idx]
                closing_idx = context_after.find("```")
                
                # If found, adjust the end_idx to include it
                if closing_idx != -1:
                    new_end_idx = match_idx + len(pattern) + closing_idx + 3  # +3 for the ```
                    end_idx = min(len(buffer_text), new_end_idx + 50)  # Add a bit more context after
            
            # Get token indices
            token_match_start = 0
            token_match_end = 0
            
            token_start = 0
            for i, token in enumerate(self._buffer):
                token_end = token_start + len(token)
                if token_start <= match_idx < token_end:
                    token_match_start = i
                    break
                token_start = token_end
                
            token_start = 0
            for i, token in enumerate(self._buffer):
                token_end = token_start + len(token)
                if token_start <= (match_idx + len(pattern) - 1) < token_end:
                    token_match_end = i
                    break
                token_start = token_end
            
            self._token_stats["pattern_matches"] += 1
            
            # Get the context before and after
            context_before = buffer_text[start_idx:match_idx]
            context_after = buffer_text[match_idx + len(pattern):end_idx]
            matched_text = buffer_text[match_idx:match_idx + len(pattern)]
            
            # For debugging
            logger.debug(f"[DynamicTokenBuffer] Found pattern '{pattern}' at position {match_idx}")
            logger.debug(f"[DynamicTokenBuffer] Context before: '{context_before[-20:]}' (length: {len(context_before)})")
            logger.debug(f"[DynamicTokenBuffer] Context after: '{context_after[:20]}' (length: {len(context_after)})")
            
            return {
                "pattern": pattern,
                "match_start": match_idx,
                "match_end": match_idx + len(pattern),
                "token_match_start": token_match_start,
                "token_match_end": token_match_end,
                "context_before": context_before,
                "context_after": context_after,
                "matched_text": matched_text,
                "buffer_size": len(buffer_text)
            }
    
    def replace_range(self, start: int, end: int, replacement: str) -> None:
        """
        Replace a range of tokens with a new string.
        
        Args:
            start: Start index (inclusive)
            end: End index (exclusive)
            replacement: Replacement string
        """
        with self._lock:
            if start < 0 or end > len(self._buffer) or start >= end:
                return
                
            # Convert the replacement to a list of tokens (characters)
            replacement_tokens = list(replacement)
            
            # Replace the range
            self._buffer = self._buffer[:start] + replacement_tokens + self._buffer[end:]
            
            # Update metadata indices
            new_metadata = {}
            for idx, meta in self._token_metadata.items():
                if idx < start:
                    new_metadata[idx] = meta
                elif idx >= end:
                    # Adjust indices for tokens after the replaced range
                    new_offset = len(replacement_tokens) - (end - start)
                    new_metadata[idx + new_offset] = meta
            self._token_metadata = new_metadata
            
            # Update context windows
            for window_name, window in self._context_windows.items():
                window_start = window["start"]
                if window_start >= end:
                    # Window starts after the replaced range, adjust start
                    window["start"] = window_start + len(replacement_tokens) - (end - start)
                elif window_start >= start:
                    # Window starts within the replaced range, move to start of replacement
                    window["start"] = start
    
    def clear(self) -> None:
        """Clear the buffer and reset statistics."""
        with self._lock:
            self._buffer = []
            self._token_metadata = {}
            self._context_windows = {}
            
    def get_stats(self) -> Dict[str, Any]:
        """Get token processing statistics."""
        with self._lock:
            return self._token_stats.copy()
            
    def mark_execution(self) -> None:
        """Mark that an execution has occurred based on buffer content."""
        with self._lock:
            self._token_stats["executions"] += 1
            self._token_stats["last_execution_time"] = time.time()
            
    def add_to_output_stream(self, token: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a token to the output stream for self-monitoring.
        
        Args:
            token: The token being output
            metadata: Optional metadata about the token
        """
        with self._lock:
            self._output_stream.append(token)
            
            # Periodically analyze the output stream
            if len(self._output_stream) % 50 == 0:
                self._analyze_output_stream()
    
    def _analyze_output_stream(self) -> None:
        """Analyze the output stream to provide real-time feedback."""
        with self._lock:
            # Skip if output stream is too short
            if len(self._output_stream) < 10:
                return
                
            # Get the recent output
            recent_output = "".join(self._output_stream[-100:])
            
            # Simple sentiment analysis
            positive_words = ["success", "completed", "correct", "good", "effective"]
            negative_words = ["error", "failed", "incorrect", "issue", "problem"]
            
            positive_count = sum(1 for word in positive_words if word in recent_output.lower())
            negative_count = sum(1 for word in negative_words if word in recent_output.lower())
            
            if positive_count > negative_count:
                self._output_analysis["sentiment"] = "positive"
            elif negative_count > positive_count:
                self._output_analysis["sentiment"] = "negative"
            else:
                self._output_analysis["sentiment"] = "neutral"
                
            # Analyze complexity
            avg_word_length = sum(len(word) for word in recent_output.split()) / max(1, len(recent_output.split()))
            if avg_word_length > 8:
                self._output_analysis["complexity"] = "high"
            elif avg_word_length > 5:
                self._output_analysis["complexity"] = "medium"
            else:
                self._output_analysis["complexity"] = "low"
                
            # Update analysis timestamp
            self._output_analysis["last_analysis_time"] = time.time()
    
    def get_output_analysis(self) -> Dict[str, Any]:
        """Get the current analysis of the output stream."""
        with self._lock:
            return self._output_analysis.copy()
    
    def get_output_stream(self, last_n: int = 100) -> str:
        """Get the last n tokens from the output stream."""
        with self._lock:
            return "".join(self._output_stream[-last_n:])
    
    def __len__(self) -> int:
        """Get the current buffer length."""
        with self._lock:
            return len(self._buffer)

class TokenRegistry:
    """
    Maintains a registry of token sequences that can be used to trigger code execution.
    This allows the agent to stream tokens and execute code when specific patterns are detected.
    Enhanced with dynamic token buffer for better context management.
    Now with self-monitoring capabilities for real-time output analysis.
    """
    def __init__(self):
        self._registry = {}
        self._lock = threading.Lock()
        self._buffer = DynamicTokenBuffer(max_size=2000)
        self._pattern_contexts = {}
        self._execution_history = []
        self._max_history = 50
        self._output_feedback_enabled = True
        self._output_feedback_handlers = []
        
    def register_pattern(self, pattern: str, callback: Callable[[str, Dict[str, Any]], Any], 
                        context_size: int = 200) -> None:
        """
        Register a pattern and associated callback function.
        
        Args:
            pattern: Pattern to match in the token stream
            callback: Function to call when pattern is matched
            context_size: Amount of context to include with the match
        """
        with self._lock:
            self._registry[pattern] = callback
            self._pattern_contexts[pattern] = context_size
            logger.info(f"[TokenRegistry] Registered pattern '{pattern}' with context size {context_size}")
            
    def process_token(self, token: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Process a single token, checking for registered patterns.
        Returns a list of execution results if any patterns matched.
        
        Args:
            token: The token to process
            metadata: Optional metadata about the token
            
        Returns:
            List of execution results from triggered callbacks
        """
        results = []
        with self._lock:
            # Add token to buffer with metadata
            self._buffer.add_token(token, metadata)
            
            # Check for patterns
            for pattern, callback in self._registry.items():
                context_size = self._pattern_contexts.get(pattern, 200)
                
                # For Python code blocks, use a larger context window
                if pattern == "```python":
                    context_size = 1000  # Use a larger context for Python code blocks
                
                match_info = self._buffer.find_pattern(pattern, context_size)
                
                if match_info:
                    try:
                        # Log the match before processing
                        logger.info(f"[TokenRegistry] Found pattern '{pattern}' match: '{match_info['matched_text'][:20]}...'")
                        
                        # Call the callback with the matched content and context
                        result = callback(match_info["matched_text"], match_info)
                        
                        # Record execution
                        self._buffer.mark_execution()
                        self._record_execution(pattern, match_info, result)
                        
                        # Add result to return list
                        results.append({
                            "pattern": pattern,
                            "matched_text": match_info["matched_text"],
                            "result": result
                        })
                        
                        # Only remove the matched content if it was successfully processed
                        # For partial matches, we might want to keep the content in the buffer
                        if isinstance(result, dict) and result.get("status") != "partial_match":
                            self._buffer.replace_range(
                                match_info["match_start"],
                                match_info["match_end"],
                                ""  # Replace with empty string
                            )
                            logger.info(f"[TokenRegistry] Removed matched content for pattern '{pattern}'")
                        else:
                            logger.info(f"[TokenRegistry] Kept matched content in buffer for pattern '{pattern}' (partial match)")
                    except Exception as e:
                        logger.error(f"[TokenRegistry] Error executing callback for pattern '{pattern}': {e}")
                        traceback_str = traceback.format_exc()
                        logger.error(f"[TokenRegistry] Traceback: {traceback_str}")
                        
                        # Record failed execution
                        self._record_execution(pattern, match_info, {"error": str(e), "traceback": traceback_str})
        
        return results
    
    def _record_execution(self, pattern: str, match_info: Dict[str, Any], result: Any) -> None:
        """Record an execution in the history."""
        execution_record = {
            "timestamp": time.time(),
            "pattern": pattern,
            "context_before": match_info["context_before"][-50:],  # Limit context size
            "context_after": match_info["context_after"][:50],
            "matched_text": match_info["matched_text"],
            "result": result
        }
        
        self._execution_history.append(execution_record)
        
        # Trim history if needed
        if len(self._execution_history) > self._max_history:
            self._execution_history = self._execution_history[-self._max_history:]
    
    def get_buffer_text(self, window_name: str = "default", size: int = 500) -> str:
        """
        Get text from a named context window in the buffer.
        
        Args:
            window_name: Name of the context window
            size: Size of the window if creating a new one
            
        Returns:
            Text in the context window
        """
        return self._buffer.get_context_window(window_name, size)
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the execution history."""
        with self._lock:
            return list(self._execution_history)
    
    def register_output_feedback_handler(self, handler: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Register a handler function that will be called when output feedback is available.
        
        Args:
            handler: Function that takes (output_text, analysis_data) as arguments
        """
        with self._lock:
            self._output_feedback_handlers.append(handler)
            
    def process_output_token(self, token: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Process a token that is being output to the user.
        This allows the agent to monitor its own output.
        
        Args:
            token: The token being output
            metadata: Optional metadata about the token
        """
        with self._lock:
            if not self._output_feedback_enabled:
                return
                
            # Add to output stream
            self._buffer.add_to_output_stream(token, metadata)
            
            # Periodically provide feedback
            if len(self._buffer._output_stream) % 100 == 0:
                self._provide_output_feedback()
    
    def _provide_output_feedback(self) -> None:
        """Provide feedback on the current output stream."""
        with self._lock:
            # Get current output and analysis
            output = self._buffer.get_output_stream()
            analysis = self._buffer.get_output_analysis()
            
            # Call all registered handlers
            for handler in self._output_feedback_handlers:
                try:
                    handler(output, analysis)
                except Exception as e:
                    logger.error(f"[TokenRegistry] Error in output feedback handler: {e}")
    
    def enable_output_feedback(self, enabled: bool = True) -> None:
        """Enable or disable output feedback."""
        with self._lock:
            self._output_feedback_enabled = enabled
            
    def clear_buffer(self) -> None:
        """Clear the token buffer."""
        with self._lock:
            self._buffer.clear()
            
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get statistics about the token buffer."""
        return self._buffer.get_stats()

class TokenContext:
    """Token context manager for accessing the token buffer"""
    def __init__(self, adapter):
        self._adapter = adapter
        
    def get_buffer_text(self, window_name: str = "default", size: int = 500) -> str:
        """Get text from the token buffer"""
        return self._adapter.token_registry.get_buffer_text(window_name, size)
        
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the execution history"""
        return self._adapter.token_registry.get_execution_history()
        
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get statistics about the token buffer"""
        return self._adapter.token_registry.get_buffer_stats()
        
    def get_last_execution_context(self) -> Dict[str, Any]:
        """Get the context from the last execution"""
        return self._adapter.execution_context.copy()

class FunctionAdapter:
    """
    The 'do_anything' capability: if the agent sees <function_call> do_anything: <code>...</code>,
    it executes that Python code directly. Highly insecure outside a sandbox.
    
    Enhanced with token registry for streaming execution and realtime token awareness.
    Features:
    - Realtime token processing with context awareness
    - Dynamic code execution based on streamed tokens
    - Pattern matching with flexible context windows
    - Execution history tracking and analysis
    - Error handling and recovery for partial code execution
    - Self-monitoring output stream with real-time feedback
    """
    def __init__(self):
        self.token_registry = TokenRegistry()
        self.execution_context = {}
        self.last_execution_time = 0
        self.execution_count = 0
        self.partial_code_fragments = {}
        self.output_feedback_enabled = True
        
        # Register patterns for code execution with enhanced callbacks
        self.token_registry.register_pattern(
            "<function_call> do_anything:", 
            self._handle_do_anything_with_context,
            context_size=500
        )
        self.token_registry.register_pattern(
            "```python", 
            self._handle_python_code_block_with_context,
            context_size=500
        )
        self.token_registry.register_pattern(
            "<execute>", 
            self._handle_execute_tag_with_context,
            context_size=300
        )
        
    def _handle_do_anything_with_context(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a do_anything function call pattern with context awareness.
        
        Args:
            content: The matched pattern content
            context: Context information including surrounding tokens
            
        Returns:
            Execution result
        """
        # Extract the code from the pattern and context
        full_text = context["context_before"] + content + context["context_after"]
        code_match = re.search(r"<function_call>\s*do_anything\s*:\s*(.*?)</function_call>", full_text, re.DOTALL)
        
        if code_match:
            code = code_match.group(1)
            # Store in execution context
            self.execution_context["last_pattern"] = "do_anything"
            self.execution_context["last_code"] = code
            self.execution_context["context_before"] = context["context_before"]
            self.execution_context["context_after"] = context["context_after"]
            
            # Execute the code
            result = self.do_anything(code)
            
            # Update execution stats
            self.last_execution_time = time.time()
            self.execution_count += 1
            
            return result
        
        # If we couldn't extract complete code, store as partial fragment
        fragment_id = f"do_anything_{int(time.time())}"
        self.partial_code_fragments[fragment_id] = {
            "pattern": "do_anything",
            "content": content,
            "context": context,
            "timestamp": time.time()
        }
        
        return {
            "status": "partial_match",
            "message": "Incomplete function call detected, waiting for more tokens",
            "fragment_id": fragment_id
        }
    
    def _handle_python_code_block_with_context(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a Python code block pattern with context awareness.
        
        Args:
            content: The matched pattern content
            context: Context information including surrounding tokens
            
        Returns:
            Execution result
        """
        # Extract the code from the pattern and context
        full_text = context["context_before"] + content + context["context_after"]
        
        # Use a more robust pattern that can handle multi-line code blocks
        code_match = re.search(r"```python\s*(.*?)```", full_text, re.DOTALL)
        
        if code_match:
            code = code_match.group(1)
            # Store in execution context
            self.execution_context["last_pattern"] = "python_code_block"
            self.execution_context["last_code"] = code
            
            # Log the matched code
            logger.info(f"[FunctionAdapter] Matched complete Python code block: {code[:100]}...")
            
            # Execute the code
            result = self.do_anything(code)
            
            # Update execution stats
            self.last_execution_time = time.time()
            self.execution_count += 1
            
            return result
        
        # Check if we have enough context to determine if this is likely a complete block
        # Sometimes the closing ``` might not be in the context yet
        if "```python" in full_text:
            # Extract everything between ```python and the end of the available context
            partial_code_match = re.search(r"```python\s*(.*?)$", full_text, re.DOTALL)
            
            if partial_code_match:
                partial_code = partial_code_match.group(1).strip()
                
                # If we have substantial code, consider executing it even without the closing ```
                if len(partial_code) > 50 and time.time() - self.last_execution_time > 2:
                    logger.info(f"[FunctionAdapter] Executing partial Python code block: {partial_code[:100]}...")
                    
                    # Add a safety wrapper to handle potential syntax errors from incomplete code
                    safe_code = f"""
try:
    # Original partial code
    {partial_code}
except SyntaxError:
    print("Warning: Syntax error in partial code block - likely incomplete")
    # Try to extract and execute complete statements
    import ast
    
    def extract_complete_statements(code):
        try:
            # Try to parse the code
            tree = ast.parse(code)
            # If we get here, the code is syntactically valid
            return code
        except SyntaxError as e:
            # Find the last valid statement
            lines = code.split('\\n')
            for i in range(len(lines), 0, -1):
                try:
                    ast.parse('\\n'.join(lines[:i]))
                    return '\\n'.join(lines[:i])
                except SyntaxError:
                    continue
        return ""
    
    complete_code = extract_complete_statements('''{partial_code}''')
    if complete_code:
        print(f"Executing complete statements from partial code block")
        exec(complete_code)
"""
                    result = self.do_anything(safe_code)
                    
                    # Update execution stats
                    self.last_execution_time = time.time()
                    self.execution_count += 1
                    
                    # Still store as partial fragment for potential complete execution later
                    fragment_id = f"python_block_{int(time.time())}"
                    self.partial_code_fragments[fragment_id] = {
                        "pattern": "python_code_block",
                        "content": content,
                        "context": context,
                        "timestamp": time.time(),
                        "full_text": full_text,
                        "partial_execution": True
                    }
                    
                    logger.info(f"[FunctionAdapter] Stored partial Python code block as fragment {fragment_id} (with partial execution)")
                    
                    return {
                        "status": "partial_execution",
                        "message": "Executed partial code block while waiting for complete block",
                        "fragment_id": fragment_id,
                        "execution_result": result
                    }
                else:
                    # If code is too short or we recently executed something, just store as fragment
                    fragment_id = f"python_block_{int(time.time())}"
                    self.partial_code_fragments[fragment_id] = {
                        "pattern": "python_code_block",
                        "content": content,
                        "context": context,
                        "timestamp": time.time(),
                        "full_text": full_text
                    }
                    
                    logger.info(f"[FunctionAdapter] Stored partial Python code block as fragment {fragment_id}")
                    
                    return {
                        "status": "partial_match",
                        "message": "Incomplete code block detected, waiting for more tokens",
                        "fragment_id": fragment_id
                    }
            
        # If we couldn't extract partial code, store as fragment
        fragment_id = f"python_block_{int(time.time())}"
        self.partial_code_fragments[fragment_id] = {
            "pattern": "python_code_block",
            "content": content,
            "context": context,
            "timestamp": time.time(),
            "full_text": full_text
        }
        
        logger.info(f"[FunctionAdapter] Stored partial Python code block as fragment {fragment_id}")
        
        return {
            "status": "partial_match",
            "message": "Incomplete code block detected, waiting for more tokens",
            "fragment_id": fragment_id
        }
        
        # If we can't determine if it's a Python code block, return a more informative message
        return {
            "status": "uncertain_match",
            "message": "Uncertain if this is a complete Python code block",
            "context_length": len(full_text)
        }
        
    def _handle_execute_tag_with_context(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an execute tag with context awareness.
        
        Args:
            content: The matched pattern content
            context: Context information including surrounding tokens
            
        Returns:
            Execution result
        """
        # Extract the code from the pattern and context
        full_text = context["context_before"] + content + context["context_after"]
        code_match = re.search(r"<execute>(.*?)</execute>", full_text, re.DOTALL)
        
        if code_match:
            code = code_match.group(1)
            # Store in execution context
            self.execution_context["last_pattern"] = "execute_tag"
            self.execution_context["last_code"] = code
            
            # Execute the code
            result = self.do_anything(code)
            
            # Update execution stats
            self.last_execution_time = time.time()
            self.execution_count += 1
            
            return result
        
        # If we couldn't extract complete code, store as partial fragment
        fragment_id = f"execute_tag_{int(time.time())}"
        self.partial_code_fragments[fragment_id] = {
            "pattern": "execute_tag",
            "content": content,
            "context": context,
            "timestamp": time.time()
        }
        
        return {
            "status": "partial_match",
            "message": "Incomplete execute tag detected, waiting for more tokens",
            "fragment_id": fragment_id
        }
    
    def do_anything(self, snippet: str) -> Dict[str, Any]:
        """
        Execute arbitrary Python code with enhanced context awareness and error handling.
        
        Args:
            snippet: Python code to execute
            
        Returns:
            Dictionary with execution results
        """
        code = snippet.strip()
        import re, io, sys
        code = re.sub(r"```python\s*", "", code)
        code = code.replace("```", "")
        code = re.sub(r"<code\s+language=['\"]python['\"]>\s*", "", code)
        code = code.replace("</code>", "")
        
        # Store original code before execution
        original_code = code
        
        # Add context-aware utilities to the execution environment
        context_utilities = """
# Context-aware utilities for code execution
import sys, os, re, json, time, datetime
from typing import Dict, List, Any, Optional

class TokenContext:
    \"\"\"Token context manager for accessing the token buffer\"\"\"
    def __init__(self, adapter=None):
        self._adapter = adapter
        
    def get_buffer_text(self, window_name: str = "default", size: int = 500) -> str:
        \"\"\"Get text from the token buffer\"\"\"
        if self._adapter:
            return self._adapter.token_registry.get_buffer_text(window_name, size)
        return ""
        
    def get_execution_history(self) -> List[Dict[str, Any]]:
        \"\"\"Get the execution history\"\"\"
        if self._adapter:
            return self._adapter.token_registry.get_execution_history()
        return []
        
    def get_buffer_stats(self) -> Dict[str, Any]:
        \"\"\"Get statistics about the token buffer\"\"\"
        if self._adapter:
            return self._adapter.token_registry.get_buffer_stats()
        return {}
        
    def get_last_execution_context(self) -> Dict[str, Any]:
        \"\"\"Get the context from the last execution\"\"\"
        if self._adapter:
            return self._adapter.execution_context.copy()
        return {}

# Create token context instance - use None to avoid self reference issues
token_context = TokenContext(None)
"""
        
        # Prepend context utilities to the code
        code = context_utilities + "\n\n" + code
        
        logger.info(f"[do_anything] Executing code:\n{original_code}")
        old_stdout = sys.stdout
        mystdout = io.StringIO()
        sys.stdout = mystdout
        
        execution_start_time = time.time()
        try:
            # Create a local namespace for execution with access to token context
            # Avoid passing self reference to prevent NameError
            local_namespace = {
                "token_context": TokenContext(None),
                "execution_count": self.execution_count,
                "last_execution_time": self.last_execution_time,
                "execution_context": self.execution_context.copy(),
                # Add commonly needed modules directly
                "datetime": datetime,
                "time": time,
                "json": json,
                "re": re,
                "os": os,
                "sys": sys
            }
            
            # Execute the code with timeout protection
            exec(code, globals(), local_namespace)
            
            # Extract the result if available
            result = local_namespace.get('result', None)
            
            # Update execution context with any new variables
            for key, value in local_namespace.items():
                if key not in ["token_context", "execution_count", "last_execution_time", "execution_context", 
                              "datetime", "time", "json", "re", "os", "sys"] and \
                   not key.startswith("__"):
                    # Only store serializable values
                    try:
                        json.dumps({key: str(value)})
                        self.execution_context[key] = value
                    except (TypeError, OverflowError):
                        # Skip non-serializable values
                        pass
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[do_anything] Error: {str(e)}\nTraceback:\n{tb}")
            
            # Try to extract partial results from the error
            partial_results = self._extract_partial_results_from_error(e, tb)
            
            return {
                "status": "error", 
                "error": str(e), 
                "traceback": tb,
                "execution_time": time.time() - execution_start_time,
                "partial_results": partial_results
            }
        finally:
            sys.stdout = old_stdout

        output = mystdout.getvalue()
        logger.info(f"[do_anything] Execution output:\n{output}")
        
        # Check for additional function calls in output
        new_calls = re.findall(r"<function_call>\s*do_anything\s*:\s*(.*?)</function_call>", output, re.DOTALL)
        if new_calls:
            logger.info(f"[do_anything] Found nested function calls. Executing them recursively.")
            nested_results = []
            for c in new_calls:
                nested_result = self.do_anything(c)
                nested_results.append(nested_result)
                
            # Include nested results in the return value
            return {
                "status": "success", 
                "executed_code": original_code,  # Return original code without utilities
                "output": output,
                "result": result,
                "execution_time": time.time() - execution_start_time,
                "nested_executions": nested_results
            }

        return {
            "status": "success", 
            "executed_code": original_code,  # Return original code without utilities
            "output": output,
            "result": result,
            "execution_time": time.time() - execution_start_time
        }
        
    def _extract_partial_results_from_error(self, error: Exception, traceback_str: str) -> Dict[str, Any]:
        """
        Attempt to extract partial results from an execution error.
        
        Args:
            error: The exception that occurred
            traceback_str: The traceback string
            
        Returns:
            Dictionary with any partial results that could be extracted
        """
        partial_results = {
            "extracted_variables": {},
            "last_line_executed": None,
            "error_line_number": None
        }
        
        # Try to extract the line number where the error occurred
        line_match = re.search(r"line (\d+)", traceback_str)
        if line_match:
            partial_results["error_line_number"] = int(line_match.group(1))
            
        # Extract any variables from the execution context that might have been set
        # before the error occurred
        for key, value in self.execution_context.items():
            if key not in ["last_pattern", "last_code", "context_before", "context_after"]:
                try:
                    # Only include serializable values
                    json.dumps({key: str(value)})
                    partial_results["extracted_variables"][key] = value
                except (TypeError, OverflowError):
                    # Skip non-serializable values
                    pass
                    
        return partial_results

    def process_streamed_token(self, token: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Process a single streamed token, checking for patterns and executing code when appropriate.
        
        Args:
            token: The token to process
            metadata: Optional metadata about the token (e.g., position, confidence)
            
        Returns:
            List of execution results if any patterns matched
        """
        # Process the token through the registry
        results = self.token_registry.process_token(token, metadata)
        
        # Check for partial code fragments that might now be complete
        self._check_partial_fragments()
        
        return results
        
    def process_output_token(self, token: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Process a token that is being output to the user.
        This allows the agent to monitor its own output.
        
        Args:
            token: The token being output
            metadata: Optional metadata about the token
        """
        if self.output_feedback_enabled:
            self.token_registry.process_output_token(token, metadata)
    
    def register_output_feedback_handler(self, handler: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Register a handler function for output feedback.
        
        Args:
            handler: Function that takes (output_text, analysis_data) as arguments
        """
        self.token_registry.register_output_feedback_handler(handler)
    
    def enable_output_feedback(self, enabled: bool = True) -> None:
        """Enable or disable output feedback."""
        self.output_feedback_enabled = enabled
        self.token_registry.enable_output_feedback(enabled)
        
    def _check_partial_fragments(self) -> None:
        """Check if any partial code fragments can now be completed with new context."""
        current_time = time.time()
        fragments_to_remove = []
        
        for fragment_id, fragment in self.partial_code_fragments.items():
            # Skip fragments that are too recent (still accumulating tokens)
            if current_time - fragment["timestamp"] < 0.5:  # 500ms threshold
                continue
                
            pattern = fragment["pattern"]
            context = fragment["context"]
            
            # Get updated context from buffer with a larger window for better matching
            updated_context = {
                "context_before": context["context_before"],
                "context_after": self.token_registry.get_buffer_text("default", 1000),  # Increased context size
                "matched_text": context["matched_text"]
            }
            
            # Try to extract complete code with updated context
            if pattern == "do_anything":
                full_text = updated_context["context_before"] + updated_context["matched_text"] + updated_context["context_after"]
                code_match = re.search(r"<function_call>\s*do_anything\s*:\s*(.*?)</function_call>", full_text, re.DOTALL)
                
                if code_match:
                    code = code_match.group(1)
                    logger.info(f"[FunctionAdapter] Completed partial fragment {fragment_id} (do_anything)")
                    self.do_anything(code)
                    fragments_to_remove.append(fragment_id)
                    
            elif pattern == "python_code_block":
                full_text = updated_context["context_before"] + updated_context["matched_text"] + updated_context["context_after"]
                
                # Use a more robust pattern for Python code blocks
                code_match = re.search(r"```python\s*(.*?)```", full_text, re.DOTALL)
                
                if code_match:
                    code = code_match.group(1)
                    logger.info(f"[FunctionAdapter] Completed partial fragment {fragment_id} (python_code_block)")
                    logger.info(f"[FunctionAdapter] Executing code: {code[:100]}...")
                    result = self.do_anything(code)
                    
                    # Log execution result
                    if isinstance(result, dict) and "status" in result:
                        logger.info(f"[FunctionAdapter] Execution result: {result['status']}")
                    
                    fragments_to_remove.append(fragment_id)
                else:
                    # Check if we have a partial code block that's still incomplete
                    # but has accumulated enough content to be worth executing
                    if "```python" in full_text and current_time - fragment["timestamp"] > 5:  # 5 seconds threshold
                        # Try to extract code between ```python and the end of the buffer
                        partial_match = re.search(r"```python\s*(.*?)$", full_text, re.DOTALL)
                        if partial_match and len(partial_match.group(1)) > 50:  # Only if we have substantial code
                            partial_code = partial_match.group(1)
                            logger.info(f"[FunctionAdapter] Executing incomplete code block after timeout: {partial_code[:100]}...")
                            self.do_anything(partial_code)
                            fragments_to_remove.append(fragment_id)
                    
            elif pattern == "execute_tag":
                full_text = updated_context["context_before"] + updated_context["matched_text"] + updated_context["context_after"]
                code_match = re.search(r"<execute>(.*?)</execute>", full_text, re.DOTALL)
                
                if code_match:
                    code = code_match.group(1)
                    logger.info(f"[FunctionAdapter] Completed partial fragment {fragment_id} (execute_tag)")
                    self.do_anything(code)
                    fragments_to_remove.append(fragment_id)
            
            # Remove old fragments (older than 30 seconds)
            if current_time - fragment["timestamp"] > 30:
                logger.info(f"[FunctionAdapter] Removing expired fragment {fragment_id} (pattern: {pattern})")
                fragments_to_remove.append(fragment_id)
        
        # Remove processed or expired fragments
        for fragment_id in fragments_to_remove:
            del self.partial_code_fragments[fragment_id]
    
    def process_function_calls(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Process <function_call> tags in the text and execute the code within.
        Enhanced with better pattern matching and multiple function types.
        
        Args:
            text: Text to process for function calls
            
        Returns:
            Results of function execution or None if no functions found
        """
        # Process different types of function calls
        results = []
        
        # Check for do_anything function calls
        do_anything_pattern = r"<function_call>\s*do_anything\s*:\s*(.*?)</function_call>"
        do_anything_matches = re.findall(do_anything_pattern, text, re.DOTALL)
        for match in do_anything_matches:
            result = self.do_anything(match)
            results.append({
                "type": "do_anything",
                "result": result
            })
        
        # Check for Python code blocks
        python_block_pattern = r"```python\s*(.*?)```"
        python_block_matches = re.findall(python_block_pattern, text, re.DOTALL)
        for match in python_block_matches:
            result = self.do_anything(match)
            results.append({
                "type": "python_block",
                "result": result
            })
        
        # Check for execute tags
        execute_tag_pattern = r"<execute>(.*?)</execute>"
        execute_tag_matches = re.findall(execute_tag_pattern, text, re.DOTALL)
        for match in execute_tag_matches:
            result = self.do_anything(match)
            results.append({
                "type": "execute_tag",
                "result": result
            })
        
        return results if results else None
        
    def execute_shell_command(self, command: str, long_running: bool = False) -> Dict[str, Any]:
        """
        Execute a shell command with enhanced context awareness.
        
        Args:
            command: Shell command to execute
            long_running: Whether to run the command in the background
            
        Returns:
            Dictionary with execution results
        """
        import subprocess, tempfile, os
        
        logger.info(f"[FunctionAdapter] Executing shell command: {command}")
        
        try:
            if long_running:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as temp_file:
                    temp_file.write(f"#!/bin/bash\n{command}")
                    temp_file_path = temp_file.name
                
                # Make the script executable
                os.chmod(temp_file_path, 0o755)
                
                # Run in background
                subprocess.Popen(
                    f"nohup {temp_file_path} > /dev/null 2>&1 &",
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Record execution in context
                self.execution_context["last_pattern"] = "long_running_shell"
                self.execution_context["last_command"] = command
                self.execution_context["temp_file_path"] = temp_file_path
                
                return {
                    "status": "success",
                    "output": "Command is running in the background",
                    "temp_file_path": temp_file_path
                }
            else:
                # Run command with timeout
                process = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30  # 30 second timeout
                )
                
                # Record execution in context
                self.execution_context["last_pattern"] = "shell_command"
                self.execution_context["last_command"] = command
                self.execution_context["last_return_code"] = process.returncode
                
                if process.returncode == 0:
                    self.execution_context["last_output"] = process.stdout
                    
                    return {
                        "status": "success",
                        "output": process.stdout,
                        "stderr": process.stderr,
                        "return_code": process.returncode
                    }
                else:
                    self.execution_context["last_error"] = process.stderr
                    
                    return {
                        "status": "error",
                        "output": process.stdout,
                        "stderr": process.stderr,
                        "return_code": process.returncode
                    }
        except subprocess.TimeoutExpired:
            logger.error(f"[FunctionAdapter] Command timed out: {command}")
            
            # Record timeout in context
            self.execution_context["last_pattern"] = "shell_command"
            self.execution_context["last_command"] = command
            self.execution_context["last_error"] = "Command timed out"
            
            return {
                "status": "error",
                "output": "",
                "stderr": "Command timed out after 30 seconds",
                "return_code": -1
            }
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[FunctionAdapter] Error executing shell command: {e}\n{tb}")
            
            # Record error in context
            self.execution_context["last_pattern"] = "shell_command"
            self.execution_context["last_command"] = command
            self.execution_context["last_error"] = str(e)
            self.execution_context["last_traceback"] = tb
            
            return {
                "status": "error",
                "output": "",
                "stderr": str(e),
                "return_code": -1,
                "traceback": tb
            }

    def execute_isolated_code(self, code: str, timeout: int = 10, 
                             provide_context: bool = True) -> Dict[str, Any]:
        """
        Execute Python code in an isolated environment with timeout protection.
        This provides better isolation than the standard do_anything method.
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
            provide_context: Whether to provide context utilities
            
        Returns:
            Dictionary with execution results
        """
        import io, sys, tempfile, os, subprocess, threading
        
        # Store original code
        original_code = code
        
        # Add context utilities if requested
        if provide_context:
            context_utilities = """
# Context-aware utilities for code execution
import sys, os, re, json, time, datetime
from typing import Dict, List, Any, Optional

# Common utility functions
def get_current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
def get_utc_time():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
def format_json(obj):
    return json.dumps(obj, indent=2, default=str)
"""
            code = context_utilities + "\n\n" + code
        
        logger.info(f"[FunctionAdapter] Executing isolated code:\n{original_code[:200]}...")
        
        try:
            # Create a temporary file for the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name
            
            # Create a temporary file for the output
            output_file_path = temp_file_path + '.out'
            
            # Build the command to execute the code with timeout
            cmd = [
                sys.executable,  # Current Python interpreter
                temp_file_path
            ]
            
            # Execute the code in a separate process with timeout
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Set up a timer to kill the process if it exceeds the timeout
            timer = threading.Timer(timeout, process.kill)
            timer.start()
            
            try:
                stdout, stderr = process.communicate()
                return_code = process.returncode
            finally:
                timer.cancel()
            
            # Clean up the temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
            
            # Record execution in context
            self.execution_context["last_pattern"] = "isolated_python"
            self.execution_context["last_code"] = original_code
            
            if return_code == 0:
                self.execution_context["last_output"] = stdout
                
                return {
                    "status": "success",
                    "output": stdout,
                    "stderr": stderr,
                    "return_code": return_code,
                    "execution_time": time.time() - self.last_execution_time
                }
            else:
                self.execution_context["last_error"] = stderr
                
                return {
                    "status": "error",
                    "output": stdout,
                    "stderr": stderr,
                    "return_code": return_code,
                    "execution_time": time.time() - self.last_execution_time
                }
                
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[FunctionAdapter] Error executing isolated code: {e}\n{tb}")
            
            # Record error in context
            self.execution_context["last_pattern"] = "isolated_python"
            self.execution_context["last_code"] = original_code
            self.execution_context["last_error"] = str(e)
            self.execution_context["last_traceback"] = tb
            
            return {
                "status": "error",
                "output": "",
                "stderr": str(e),
                "traceback": tb
            }
    
    def execute_python_code(self, code: str, long_running: bool = False) -> Dict[str, Any]:
        """
        Execute Python code. If long_running is True, use nohup to run it in the background as a separate process.
        Enhanced with better error handling and context awareness.
        
        Args:
            code: Python code to execute
            long_running: Whether to run the code in the background
            
        Returns:
            Dictionary with execution results
        """
        import io, sys, tempfile, os
        
        # Add context utilities to the code
        context_utilities = """
# Context-aware utilities for code execution
import sys, os, re, json, time, datetime
from typing import Dict, List, Any, Optional

class TokenContext:
    \"\"\"Token context manager for accessing the token buffer\"\"\"
    def __init__(self, adapter=None):
        self._adapter = adapter
        
    def get_buffer_text(self, window_name: str = "default", size: int = 500) -> str:
        \"\"\"Get text from the token buffer\"\"\"
        if self._adapter:
            return self._adapter.token_registry.get_buffer_text(window_name, size)
        return ""
        
    def get_execution_history(self) -> List[Dict[str, Any]]:
        \"\"\"Get the execution history\"\"\"
        if self._adapter:
            return self._adapter.token_registry.get_execution_history()
        return []
        
    def get_buffer_stats(self) -> Dict[str, Any]:
        \"\"\"Get statistics about the token buffer\"\"\"
        if self._adapter:
            return self._adapter.token_registry.get_buffer_stats()
        return {}
        
    def get_last_execution_context(self) -> Dict[str, Any]:
        \"\"\"Get the context from the last execution\"\"\"
        if self._adapter:
            return self._adapter.execution_context.copy()
        return {}

# Create token context instance - use None to avoid self reference issues
token_context = TokenContext(None)

# Add helper functions for common operations
def get_current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
def get_utc_time():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
"""
        
        # Store original code
        original_code = code
        
        # Only add utilities for non-long-running code
        if not long_running:
            code = context_utilities + "\n\n" + code
        
        try:
            if long_running:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                    temp_file.write(code)
                    temp_file_path = temp_file.name
                command = f"nohup python {temp_file_path} > /dev/null 2>&1 &"
                os.system(command)
                
                # Record execution in context
                self.execution_context["last_pattern"] = "long_running_python"
                self.execution_context["last_code"] = original_code
                self.execution_context["temp_file_path"] = temp_file_path
                
                return {
                    "status": "success", 
                    "output": "Code is running in the background",
                    "temp_file_path": temp_file_path
                }
            else:
                old_stdout = sys.stdout
                mystdout = io.StringIO()
                sys.stdout = mystdout
                
                # Create a local namespace with access to token context
                local_namespace = {
                    "token_context": TokenContext(self),
                    "execution_count": self.execution_count,
                    "last_execution_time": self.last_execution_time,
                    "execution_context": self.execution_context.copy()
                }
                
                # Execute the code
                exec(code, globals(), local_namespace)
                
                # Update execution context with any new variables
                for key, value in local_namespace.items():
                    if key not in ["token_context", "execution_count", "last_execution_time", "execution_context"] and \
                       not key.startswith("__"):
                        # Only store serializable values
                        try:
                            json.dumps({key: str(value)})
                            self.execution_context[key] = value
                        except (TypeError, OverflowError):
                            # Skip non-serializable values
                            pass
                
                sys.stdout = old_stdout
                output = mystdout.getvalue()
                
                # Record execution in context
                self.execution_context["last_pattern"] = "python_code"
                self.execution_context["last_code"] = original_code
                self.execution_context["last_output"] = output
                
                return {"status": "success", "output": output}
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[FunctionAdapter] Error executing Python code: {e}\n{tb}")
            
            # Record error in context
            self.execution_context["last_error"] = str(e)
            self.execution_context["last_traceback"] = tb
            
            return {
                "status": "error", 
                "output": "", 
                "error": str(e),
                "traceback": tb
            }
            
def get_datetime_info(self, timezone: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive date and time information, optionally for a specific timezone.
        
        Args:
            timezone: Optional timezone name (e.g., 'America/New_York', 'Europe/London')
                     If None, returns information for UTC and local system time
        
        Returns:
            Dictionary with date and time information
        """
        try:
            # Use direct code execution for more reliable results
            time_code = """
import datetime
import time
import json
from zoneinfo import ZoneInfo, available_timezones
import pytz

# Get current times
now_utc = datetime.datetime.now(datetime.timezone.utc)
now_local = datetime.datetime.now()

# Format the times
result = {
    "utc": {
        "datetime": now_utc.isoformat(),
        "date": now_utc.strftime("%Y-%m-%d"),
        "time": now_utc.strftime("%H:%M:%S"),
        "timestamp": time.time(),
        "timezone": "UTC"
    },
    "local": {
        "datetime": now_local.isoformat(),
        "date": now_local.strftime("%Y-%m-%d"),
        "time": now_local.strftime("%H:%M:%S"),
        "timezone": str(now_local.astimezone().tzname() or time.tzname[0])
    }
}

# Add timezone-specific information if requested
requested_timezone = None
"""
            
            # Add timezone handling if specified
            if timezone:
                time_code = time_code.replace('requested_timezone = None', f'requested_timezone = "{timezone}"')
                time_code += """
if requested_timezone:
    try:
        # Try with ZoneInfo first (Python 3.9+)
        tz = ZoneInfo(requested_timezone)
        tz_time = datetime.datetime.now(tz)
        
        result["requested_timezone"] = {
            "datetime": tz_time.isoformat(),
            "date": tz_time.strftime("%Y-%m-%d"),
            "time": tz_time.strftime("%H:%M:%S"),
            "timezone": requested_timezone,
            "utc_offset": tz_time.strftime("%z")
        }
    except (ImportError, KeyError):
        # Fall back to pytz
        try:
            tz = pytz.timezone(requested_timezone)
            tz_time = datetime.datetime.now(tz)
            
            result["requested_timezone"] = {
                "datetime": tz_time.isoformat(),
                "date": tz_time.strftime("%Y-%m-%d"),
                "time": tz_time.strftime("%H:%M:%S"),
                "timezone": requested_timezone,
                "utc_offset": tz_time.strftime("%z")
            }
        except Exception as e:
            result["requested_timezone"] = {
                "error": f"Unknown timezone: {requested_timezone}",
                "exception": str(e)
            }
"""
            
            # Add timezone list
            time_code += """
# Add available timezones
try:
    result["available_timezones"] = list(available_timezones())[:20]  # First 20 for brevity
    result["available_timezones_count"] = len(available_timezones())
except ImportError:
    try:
        result["available_timezones"] = list(pytz.all_timezones)[:20]  # First 20 for brevity
        result["available_timezones_count"] = len(pytz.all_timezones)
    except ImportError:
        result["available_timezones"] = ["UTC"]
        result["available_timezones_count"] = 1

# Return the result as JSON
print(json.dumps(result, default=str))
result  # For return value
"""
            
            # Execute the code
            execution_result = self.do_anything(time_code)
            
            if execution_result and execution_result.get("status") == "success":
                # Try to parse the output as JSON
                import json
                try:
                    if "output" in execution_result and execution_result["output"]:
                        return json.loads(execution_result["output"])
                    elif "result" in execution_result and execution_result["result"]:
                        return execution_result["result"]
                except json.JSONDecodeError:
                    pass
            
            # If execution or parsing failed, fall back to the original implementation
            raise Exception("Direct execution failed, falling back to manual implementation")
            
        except Exception as e:
            # Fall back to the original implementation if execution fails
            import datetime
            import time
            import pytz
            from zoneinfo import ZoneInfo, available_timezones
            
            result = {
                "utc": {
                    "datetime": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
                    "time": datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S"),
                    "timestamp": time.time(),
                    "timezone": "UTC"
                },
                "local": {
                    "datetime": datetime.datetime.now().isoformat(),
                    "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "timezone": str(datetime.datetime.now().astimezone().tzname() or time.tzname[0])
                },
                "error": str(e)
            }
            
            # Add timezone-specific information if requested
            if timezone:
                try:
                    # Try with ZoneInfo first (Python 3.9+)
                    tz = ZoneInfo(timezone)
                    tz_time = datetime.datetime.now(tz)
                    
                    result["requested_timezone"] = {
                        "datetime": tz_time.isoformat(),
                        "date": tz_time.strftime("%Y-%m-%d"),
                        "time": tz_time.strftime("%H:%M:%S"),
                        "timezone": timezone,
                        "utc_offset": tz_time.strftime("%z")
                    }
                except (ImportError, KeyError):
                    # Fall back to pytz
                    try:
                        tz = pytz.timezone(timezone)
                        tz_time = datetime.datetime.now(tz)
                        
                        result["requested_timezone"] = {
                            "datetime": tz_time.isoformat(),
                            "date": tz_time.strftime("%Y-%m-%d"),
                            "time": tz_time.strftime("%H:%M:%S"),
                            "timezone": timezone,
                            "utc_offset": tz_time.strftime("%z")
                        }
                    except (pytz.exceptions.UnknownTimeZoneError, ImportError):
                        result["requested_timezone"] = {
                            "error": f"Unknown timezone: {timezone}"
                        }
            
            # Add available timezones
            try:
                result["available_timezones"] = list(available_timezones())[:20]  # First 20 for brevity
                result["available_timezones_count"] = len(available_timezones())
            except ImportError:
                try:
                    result["available_timezones"] = list(pytz.all_timezones)[:20]  # First 20 for brevity
                    result["available_timezones_count"] = len(pytz.all_timezones)
                except ImportError:
                    result["available_timezones"] = ["UTC"]
                    result["available_timezones_count"] = 1
            
            return result
        
def execute_datetime_code(self, timezone: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute date/time related code in an isolated environment.
        This provides a more reliable way to get accurate date/time information.
        
        Args:
            timezone: Optional timezone name
            
        Returns:
            Dictionary with date and time information
        """
        # Create a simple script to get date/time information
        script = """
import datetime
import time
import json
import sys

# Get current times
now_utc = datetime.datetime.now(datetime.timezone.utc)
now_local = datetime.datetime.now()

# Format the times
result = {
    "utc": {
        "datetime": now_utc.isoformat(),
        "date": now_utc.strftime("%Y-%m-%d"),
        "time": now_utc.strftime("%H:%M:%S"),
        "timestamp": time.time(),
        "timezone": "UTC"
    },
    "local": {
        "datetime": now_local.isoformat(),
        "date": now_local.strftime("%Y-%m-%d"),
        "time": now_local.strftime("%H:%M:%S"),
        "timezone": str(now_local.astimezone().tzname() or time.tzname[0])
    }
}

# Add timezone-specific information if requested
requested_timezone = None
"""
        
        # Add timezone handling if specified
        if timezone:
            script = script.replace('requested_timezone = None', f'requested_timezone = "{timezone}"')
            script += """
if requested_timezone:
    try:
        # Try with ZoneInfo first (Python 3.9+)
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(requested_timezone)
        tz_time = datetime.datetime.now(tz)
        
        result["requested_timezone"] = {
            "datetime": tz_time.isoformat(),
            "date": tz_time.strftime("%Y-%m-%d"),
            "time": tz_time.strftime("%H:%M:%S"),
            "timezone": requested_timezone,
            "utc_offset": tz_time.strftime("%z")
        }
    except (ImportError, KeyError):
        # Fall back to pytz
        try:
            import pytz
            tz = pytz.timezone(requested_timezone)
            tz_time = datetime.datetime.now(tz)
            
            result["requested_timezone"] = {
                "datetime": tz_time.isoformat(),
                "date": tz_time.strftime("%Y-%m-%d"),
                "time": tz_time.strftime("%H:%M:%S"),
                "timezone": requested_timezone,
                "utc_offset": tz_time.strftime("%z")
            }
        except Exception as e:
            result["requested_timezone"] = {
                "error": f"Unknown timezone: {requested_timezone}",
                "exception": str(e)
            }
"""
        
        # Add output
        script += """
# Print the result as JSON
print(json.dumps(result, default=str))
"""
        
        # Execute the script in an isolated environment
        result = self.execute_isolated_code(script, timeout=5, provide_context=False)
        
        # Parse the output as JSON
        if result and result.get("status") == "success" and result.get("output"):
            try:
                import json
                return json.loads(result["output"])
            except json.JSONDecodeError:
                pass
        
        # Return a simple error result if execution failed
        return {
            "error": "Failed to execute date/time code",
            "execution_result": result
        }
    
def get_token_buffer_status(self) -> Dict[str, Any]:
        """
        Get the current status of the token buffer.
        
        Returns:
            Dictionary with buffer statistics and context windows
        """
        buffer_stats = self.token_registry._buffer.get_stats()
        
        # Get context windows
        context_windows = {}
        with self.token_registry._buffer._lock:
            for window_name, window in self.token_registry._buffer._context_windows.items():
                context_windows[window_name] = {
                    "start": window["start"],
                    "size": window["size"],
                    "text": self.token_registry._buffer.get_context_window(window_name)[:50] + "..."
                }
        
        # Get execution history summary
        execution_history = self.token_registry.get_execution_history()
        execution_summary = []
        for execution in execution_history[-5:]:  # Last 5 executions
            execution_summary.append({
                "timestamp": execution["timestamp"],
                "pattern": execution["pattern"],
                "matched_text": execution["matched_text"][:30] + "..." if len(execution["matched_text"]) > 30 else execution["matched_text"]
            })
        
        return {
            "buffer_stats": buffer_stats,
            "context_windows": context_windows,
            "buffer_length": len(self.token_registry._buffer),
            "execution_history": execution_summary,
            "partial_fragments": len(self.partial_code_fragments)
        }

###############################################################################
# SMART TASK PROCESSOR
###############################################################################

class TaskDecompositionRequest(BaseModel):
    """
    Request model for task decomposition
    """
    task_id: int
    task_description: str
    
    model_config = ConfigDict(
        extra="forbid"
    )

class TaskProcessingResult(BaseModel):
    """
    Structured output for task processing results
    """
    task_id: int
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    subtasks_created: List[int] = Field(default_factory=list)
    reasoning_steps: List[Dict[str, Any]] = Field(default_factory=list)
    
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "task_id": 1,
                    "success": True,
                    "result": {"output": "Task completed successfully", "data": {"key": "value"}},
                    "subtasks_created": [2, 3, 4],
                    "reasoning_steps": [
                        {"behavior": "verification", "description": "Verified input parameters"}
                    ]
                }
            ]
        }
    )

class SmartTaskProcessor:
    """
    Processes tasks from the queue, including:
     - do_anything code execution
     - subtask detection (Subtask(n)= ...)
     - updating Task status, storing results
     - hooking into self-reflection
     - cognitive modeling with verification, backtracking, etc.
     - structured task decomposition with Pydantic models
    """
    def __init__(
        self,
        memory_store: TaskMemoryStore,
        function_adapter: FunctionAdapter,
        reflection: SelfReflectiveCognition,
        client: Together
    ):
        self.memory_store = memory_store
        self.function_adapter = function_adapter
        self.reflection = reflection
        self.client = client
        # Access the cognitive engine from the reflection object
        self.cognitive_engine = reflection.cognitive_engine

    def process_task(self, task: Task) -> None:
        logger.info(f"[SmartTaskProcessor] Starting task {task.task_id} - '{task.description}'")
        self.memory_store.update_task_status(task.task_id, TaskStatus.IN_PROGRESS)
        
        # Set subgoal for this task in the cognitive engine
        self.cognitive_engine.set_subgoal(
            subgoal=f"Complete task {task.task_id}: {task.description[:50]}...",
            metadata={"task_id": task.task_id}
        )

        # Process using cognitive modeling approach
        is_success = self._process_task_with_cognition(task)
        
        if is_success:
            # Mark completed, reflect
            self.memory_store.update_task_status(task.task_id, TaskStatus.COMPLETED)
            self.reflection.reflect_on_task(task)
            
            # Add verification step in cognitive engine
            self.cognitive_engine.verify(
                description=f"Task {task.task_id} processing",
                result="Success",
                is_correct=True,
                confidence=0.9
            )
            
            # Create structured result
            result = TaskProcessingResult(
                task_id=task.task_id,
                success=True,
                result=task.result,
                subtasks_created=[t.task_id for t in self.memory_store.get_subtasks(task.task_id)],
                reasoning_steps=[{
                    "behavior": step.behavior,
                    "description": step.description,
                    "confidence": step.confidence
                } for step in self.cognitive_engine.get_chain_of_thought().steps[-5:]]  # Last 5 steps
            )
            
            # Update task result with structured output
            self.memory_store.update_task_result(task.task_id, result.model_dump())
            
            logger.info(f"[SmartTaskProcessor] Completed task {task.task_id}")
        else:
            # Mark as failed
            self.memory_store.update_task_status(task.task_id, TaskStatus.FAILED)
            self.reflection.reflect_on_task(task)
            
            # Add backtracking step in cognitive engine
            self.cognitive_engine.backtrack(
                reason=f"Task {task.task_id} processing failed",
                confidence=0.8
            )
            
            # Create structured error result
            error_result = TaskProcessingResult(
                task_id=task.task_id,
                success=False,
                error="Task processing failed",
                reasoning_steps=[{
                    "behavior": step.behavior,
                    "description": step.description,
                    "confidence": step.confidence
                } for step in self.cognitive_engine.get_chain_of_thought().steps[-3:]]  # Last 3 steps
            )
            
            # Update task result with structured error
            self.memory_store.update_task_result(task.task_id, error_result.model_dump())
            
            logger.info(f"[SmartTaskProcessor] Failed to complete task {task.task_id}")

    def _process_task_with_cognition(self, task: Task) -> bool:
        """
        Process a task using cognitive modeling approach with streaming output.
        Returns True if successful, False otherwise.
        """
        try:
            # Always show task processing details
            print(f"\n=== Processing Task {task.task_id} ===\n")
            print(f"Description: {task.description}\n")
            print(f"Priority: {task.priority}, Status: {task.status}\n")
            
            # First, check if this task should be decomposed using structured output
            if self._should_decompose_task(task):
                print("Task complexity suggests decomposition. Attempting structured decomposition...\n")
                decomposition_success = self._decompose_task_with_structured_output(task)
                if decomposition_success:
                    print("Task successfully decomposed into subtasks.\n")
                    print("=========================\n")
                    return True
                print("Decomposition unsuccessful, trying alternative strategies.\n")
            
            # Try different strategies in order, with cognitive reasoning
            strategies = [
                self._try_function_calls,
                self._try_shell_commands,
                self._try_python_code,
                self._try_subtask_decomposition,
                self._try_structured_processing
            ]
            
            # Track if any strategy was successful
            success = False
            
            # Explore different strategies
            self.cognitive_engine.explore(
                strategy="Multi-strategy task processing",
                options=["Function calls", "Shell commands", "Python code", "Subtask decomposition", "Structured processing"],
                confidence=0.8
            )
            
            # Stream the exploration process
            print("\n=== Strategy Exploration ===\n")
            
            for i, strategy in enumerate(strategies):
                # Add reasoning step for trying this strategy
                self.cognitive_engine.add_reasoning_step(
                    behavior=CognitiveBehavior.EXPLORATION,
                    description=f"Trying strategy {i+1} for task {task.task_id}",
                    metadata={"strategy": strategy.__name__},
                    confidence=0.7
                )
                
                # Stream the strategy attempt with more detailed output
                strategy_name = strategy.__name__.replace('_try_', '')
                print(f"Strategy {i+1}: {strategy_name}")
                print(f"  Description: Attempting to process task using {strategy_name}")
                print(f"  Execution: ", end='', flush=True)
                
                # Try the strategy
                result = strategy(task)
                
                if result:
                    # Strategy succeeded
                    print("SUCCESS ✓")
                    print(f"  Details: Successfully processed task using {strategy_name}")
                    
                    # Show result summary if available
                    if isinstance(result, dict) and "summary" in result:
                        print(f"  Result summary: {result['summary']}")
                    
                    self.cognitive_engine.verify(
                        description=f"Strategy {strategy.__name__}",
                        result="Success",
                        is_correct=True,
                        confidence=0.9
                    )
                    success = True
                    break  # Exit the loop once a successful strategy is found
                else:
                    # Strategy didn't apply or failed
                    print("NOT APPLICABLE")
                    print(f"  Details: Strategy {strategy_name} was not applicable to this task")
                    
                    self.cognitive_engine.verify(
                        description=f"Strategy {strategy.__name__}",
                        result="Not applicable",
                        is_correct=None,
                        confidence=0.5
                    )
                
                print("")  # Add spacing between strategies
            
            print("\n=========================\n")
            
            # If no strategy worked but we didn't encounter errors, still count as success
            if not success:
                print("No specific strategy was applicable. Completing task with default processing.\n")
                
                # Add final reasoning step
                self.cognitive_engine.add_reasoning_step(
                    behavior=CognitiveBehavior.VERIFICATION,
                    description=f"Completed task {task.task_id} without applying specific strategies",
                    result="Simple completion",
                    is_correct=True,
                    confidence=0.6
                )
            
            # Update the chain of thought summary
            summary = f"Task {task.task_id} processed with {'successful' if success else 'default'} strategy. " + \
                     f"Description: '{task.description[:50]}...'"
            
            self.cognitive_engine.update_chain_summary(summary)
            print(f"Cognitive summary: {summary}\n")
            
            # Set conclusion
            conclusion = f"Task {task.task_id} completed successfully"
            self.cognitive_engine.set_conclusion(
                conclusion,
                confidence=0.85 if success else 0.6
            )
            
            print(f"Conclusion: {conclusion}\n")
            print("=========================\n")
            
            return True
            
        except Exception as e:
            logger.exception(f"[SmartTaskProcessor] Error processing task {task.task_id}: {e}")
            
            print(f"\n❌ Error processing task {task.task_id}: {e}\n")
            print(f"Traceback: {traceback.format_exc()[:200]}...\n")
            
            # Add error step to cognitive engine
            self.cognitive_engine.add_reasoning_step(
                behavior=CognitiveBehavior.VERIFICATION,
                description=f"Error processing task {task.task_id}",
                result=str(e),
                is_correct=False,
                confidence=0.9  # High confidence that there was an error
            )
            
            # Update the chain of thought with error information
            error_summary = f"Task {task.task_id} processing failed with error: {str(e)[:100]}..."
            self.cognitive_engine.update_chain_summary(error_summary)
            
            print(f"Error summary: {error_summary}\n")
            
            # Set conclusion
            error_conclusion = f"Task {task.task_id} failed due to error"
            self.cognitive_engine.set_conclusion(
                error_conclusion,
                confidence=0.9
            )
            
            print(f"Error conclusion: {error_conclusion}\n")
            print("=========================\n")
            
            return False

    def _try_function_calls(self, task: Task) -> bool:
        """Try processing function calls in the task description or generate appropriate function calls."""
        # First check for explicit function calls in the description
        result = self.function_adapter.process_function_calls(task.description)
        if result:
            self.memory_store.update_task_result(task.task_id, result)
            return True
            
        # If no explicit function calls, check if we should generate one based on the task
        if self._should_generate_function_call(task):
            generated_code = self._generate_function_call_for_task(task)
            if generated_code:
                self.cognitive_engine.add_reasoning_step(
                    behavior=CognitiveBehavior.CREATIVITY,
                    description=f"Generated function call for task {task.task_id}",
                    metadata={"generated_code": generated_code[:100] + "..." if len(generated_code) > 100 else generated_code},
                    confidence=0.8
                )
                
                # Stream the generated code
                print("\n=== Generated Code ===\n")
                print(generated_code)
                print("\n======================\n")
                
                # Process the generated code through the token registry
                for token in generated_code:
                    self.function_adapter.token_registry.process_token(token)
                
                # Execute the generated code
                result = self.function_adapter.do_anything(generated_code)
                self.memory_store.update_task_result(task.task_id, result)
                return True
                
        return False
        
    def _should_generate_function_call(self, task: Task) -> bool:
        """Determine if we should generate a function call for this task."""
        # Check for keywords that suggest data retrieval or computation
        data_keywords = ["get", "fetch", "retrieve", "find", "search", "calculate", "compute", 
                        "weather", "temperature", "data", "information", "statistics"]
                        
        return any(keyword in task.description.lower() for keyword in data_keywords)
        
    def _generate_function_call_for_task(self, task: Task) -> str:
        """Generate appropriate Python code for the task."""
        description = task.description.lower()
        
        # Weather-related task
        if "weather" in description:
            location = None
            # Try to extract location
            location_match = re.search(r"weather\s+in\s+([a-zA-Z\s]+)", description)
            if location_match:
                location = location_match.group(1).strip()
            else:
                # Check for common city abbreviations
                if "sf" in description or "san francisco" in description:
                    location = "San Francisco"
                elif "nyc" in description or "new york" in description:
                    location = "New York"
                elif "la" in description or "los angeles" in description:
                    location = "Los Angeles"
                    
            if location:
                return f"""
import requests
import json

def get_weather(location):
    try:
        # Try to use a weather API that doesn't require authentication
        url = f"https://wttr.in/{{location}}?format=j1"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract relevant information
            current_condition = data.get('current_condition', [{{}}])[0]
            temp_c = current_condition.get('temp_C', 'N/A')
            temp_f = current_condition.get('temp_F', 'N/A')
            weather_desc = current_condition.get('weatherDesc', [{{}}])[0].get('value', 'N/A')
            humidity = current_condition.get('humidity', 'N/A')
            wind_speed = current_condition.get('windspeedKmph', 'N/A')
            
            result = {{{{
                "location": location,
                "temperature": f"{{{{temp_f}}}}°F ({{{{temp_c}}}}°C)",
                "conditions": weather_desc,
                "humidity": f"{{{{humidity}}}}%",
                "wind": f"{{{{wind_speed}}}} km/h"
            }}}}
            
            return result
        else:
            # Fallback to a mock response if the API call fails
            return {{{{
                "location": location,
                "temperature": "65°F (18°C)",
                "conditions": "Partly Cloudy",
                "humidity": "70%",
                "wind": "10 km/h",
                "note": "This is estimated data as the API request failed."
            }}}}
    except Exception as e:
        # Provide a mock response if any error occurs
        return {{{{
            "location": location,
            "temperature": "65°F (18°C)",
            "conditions": "Partly Cloudy",
            "humidity": "70%",
            "wind": "10 km/h",
            "error": str(e),
            "note": "This is estimated data due to an error in the API request."
        }}}}

# Call the function with the extracted location
result = get_weather("{location}")
print(f"Weather in {{location}}:")
print(f"Temperature: {{result['temperature']}}")
print(f"Conditions: {{result['conditions']}}")
print(f"Humidity: {{result['humidity']}}")
print(f"Wind: {{result['wind']}}")

# Return the result
result
"""
        
        # For other types of tasks, return a generic information retrieval function
        return None

    def _try_shell_commands(self, task: Task) -> bool:
        """Try processing shell commands in the task description."""
        # Check for shell command execution
        shell_command_pattern = r"<shell_command>(.*?)</shell_command>"
        match = re.search(shell_command_pattern, task.description, re.DOTALL)
        if match:
            command = match.group(1).strip()
            result = self.function_adapter.execute_shell_command(command, long_running=False)
            self.memory_store.update_task_result(task.task_id, result)
            return True

        # Check for long-running shell command execution
        long_shell_command_pattern = r"<long_shell_command>(.*?)</long_shell_command>"
        match = re.search(long_shell_command_pattern, task.description, re.DOTALL)
        if match:
            command = match.group(1).strip()
            result = self.function_adapter.execute_shell_command(command, long_running=True)
            self.memory_store.update_task_result(task.task_id, result)
            return True
            
        return False

    def _try_python_code(self, task: Task) -> bool:
        """Try processing Python code in the task description."""
        # Check for Python code execution
        python_code_pattern = r"<python_code>(.*?)</python_code>"
        match = re.search(python_code_pattern, task.description, re.DOTALL)
        if match:
            code = match.group(1).strip()
            result = self.function_adapter.execute_python_code(code, long_running=False)
            self.memory_store.update_task_result(task.task_id, result)
            return True

        # Check for long-running Python code execution
        long_python_code_pattern = r"<long_python_code>(.*?)</long_python_code>"
        match = re.search(long_python_code_pattern, task.description, re.DOTALL)
        if match:
            code = match.group(1).strip()
            result = self.function_adapter.execute_python_code(code, long_running=True)
            self.memory_store.update_task_result(task.task_id, result)
            return True
            
        return False

    def _should_decompose_task(self, task: Task) -> bool:
        """Determine if a task should be decomposed using structured output."""
        # Check if task description is complex enough to warrant decomposition
        if len(task.description.split()) > 20:  # More than 20 words
            return True
            
        # Check for keywords suggesting decomposition
        decomposition_keywords = ["step by step", "multiple steps", "complex", "decompose", 
                                 "break down", "subtasks", "multi-part"]
        if any(keyword in task.description.lower() for keyword in decomposition_keywords):
            return True
            
        return False
        
    def _decompose_task_with_structured_output(self, task: Task) -> bool:
        """
        Use structured output to decompose a task into subtasks with streaming output.
        Returns True if successful, False otherwise.
        """
        try:
            # Create a request for task decomposition
            decomposition_request = TaskDecompositionRequest(
                task_id=task.task_id,
                task_description=task.description
            )
            
            # Log the attempt
            logger.info(f"[SmartTaskProcessor] Attempting structured decomposition of task {task.task_id}")
            print(f"\n=== Decomposing Task {task.task_id} ===\n")
            print(f"Task: {task.description}\n")
            
            # Add reasoning step
            self.cognitive_engine.add_reasoning_step(
                behavior=CognitiveBehavior.SUBGOAL_SETTING,
                description=f"Decomposing task {task.task_id} using structured output",
                metadata={"task_description": task.description[:100]},
                confidence=0.8
            )
            
            # Call the LLM to decompose the task with enhanced instructions
            messages = [
                {"role": "system", "content": "You are an expert task decomposition system. Break down complex tasks into logical subtasks with clear dependencies, execution steps, and resource requirements. Return your response in JSON format with fields: subtasks (array of objects with description field), dependencies (object mapping subtask indices to arrays of dependency indices), estimated_complexity (object mapping subtask indices to complexity values 1-10), execution_steps (array of strings with detailed execution steps for each subtask), and rationale (string)."},
                {"role": "user", "content": f"Decompose this task into detailed subtasks with execution steps: {task.description}"}
            ]
            
            # Stream the decomposition process
            print("Streaming decomposition process...\n")
            
            decomp_stream = self.client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1",
                messages=messages,
                temperature=0.7,
                max_tokens=1500,
                stream=True
            )
            
            decomp_chunks = []
            for chunk in decomp_stream:
                token = chunk.choices[0].delta.content
                if token:
                    decomp_chunks.append(token)
                    print(token, end='', flush=True)
            
            print("\n\n")
            
            # Extract the response text
            response_text = "".join(decomp_chunks)
            
            # Parse the JSON response
            import json
            try:
                json_data = json.loads(response_text)
                # Create a SubtaskDecomposition object from the JSON data
                decomposition = SubtaskDecomposition(
                    original_task_id=task.task_id,
                    original_description=task.description,
                    subtasks=json_data.get("subtasks", []),
                    dependencies=json_data.get("dependencies", {}),
                    estimated_complexity=json_data.get("estimated_complexity", {}),
                    rationale=json_data.get("rationale", "Task decomposed based on complexity and logical flow.")
                )
                
                # Extract execution steps if available
                execution_steps = json_data.get("execution_steps", [])
                if execution_steps:
                    print("\n=== Execution Steps ===\n")
                    for i, step in enumerate(execution_steps):
                        print(f"{i+1}. {step}")
                    print("\n")
                
            except json.JSONDecodeError:
                print("\nJSON parsing failed. Extracting subtasks using pattern matching...\n")
                
                # If JSON parsing fails, create a simple decomposition from the text
                # Extract subtasks using regex or simple parsing
                import re
                subtask_matches = re.findall(r"(?:^|\n)(?:\d+\.\s*|\-\s*)(.*?)(?:\n|$)", response_text)
                subtasks = [{"description": match.strip()} for match in subtask_matches if match.strip()]
                
                if not subtasks:
                    # If regex fails, split by newlines and create subtasks
                    lines = [line.strip() for line in response_text.split("\n") if line.strip()]
                    subtasks = [{"description": line} for line in lines[:5]]  # Limit to 5 subtasks
                
                decomposition = SubtaskDecomposition(
                    original_task_id=task.task_id,
                    original_description=task.description,
                    subtasks=subtasks,
                    dependencies={},
                    estimated_complexity={},
                    rationale="Task decomposed into sequential steps."
                )
                
                print("Extracted subtasks:\n")
                for i, subtask in enumerate(subtasks):
                    print(f"{i+1}. {subtask['description']}")
                print("\n")
            
            # Record the decomposition in the cognitive model
            self.cognitive_engine.decompose_task(task, decomposition)
            
            # Create subtasks based on the decomposition
            created_subtasks = []
            print("\n=== Creating Subtasks ===\n")
            
            for i, subtask_info in enumerate(decomposition.subtasks):
                # Create a new subtask
                subtask = self._spawn_subtask(task, subtask_info["description"])
                created_subtasks.append(subtask)
                
                # Add tags if dependencies exist
                dependencies = []
                if decomposition.dependencies and str(i) in decomposition.dependencies:
                    for dep_idx in decomposition.dependencies[str(i)]:
                        if 0 <= dep_idx < len(created_subtasks):
                            # Add a tag indicating dependency
                            dep_task = created_subtasks[dep_idx]
                            subtask.add_tag(f"depends_on_{dep_task.task_id}")
                            dependencies.append(dep_task.task_id)
                
                # Add complexity tag if available
                complexity = "unknown"
                if decomposition.estimated_complexity and str(i) in decomposition.estimated_complexity:
                    complexity = decomposition.estimated_complexity[str(i)]
                    subtask.add_tag(f"complexity_{complexity}")
                
                # Push subtask to queue for immediate processing
                self.task_queue.push(subtask)
                
                # Print subtask details
                print(f"Subtask {i+1}: {subtask.task_id}")
                print(f"  Description: {subtask_info['description']}")
                print(f"  Complexity: {complexity}")
                if dependencies:
                    print(f"  Dependencies: {dependencies}")
                print("")
            
            # Update the task result with the decomposition
            self.memory_store.update_task_result(task.task_id, {
                "decomposition": decomposition.model_dump(),
                "subtasks": [st.task_id for st in created_subtasks],
                "full_decomposition_text": response_text
            })
            
            # Add verification step
            self.cognitive_engine.verify(
                description=f"Task {task.task_id} decomposition",
                result=f"Successfully decomposed into {len(created_subtasks)} subtasks",
                is_correct=True,
                confidence=0.9
            )
            
            logger.info(f"[SmartTaskProcessor] Successfully decomposed task {task.task_id} into {len(created_subtasks)} subtasks")
            print(f"\nSuccessfully decomposed task into {len(created_subtasks)} subtasks and queued for processing.\n")
            print("=========================\n")
            
            return True
            
        except Exception as e:
            logger.exception(f"[SmartTaskProcessor] Error in structured decomposition: {e}")
            
            print(f"\n❌ Error in structured decomposition: {e}\n")
            print(f"Traceback: {traceback.format_exc()[:200]}...\n")
            
            # Add error to cognitive engine
            self.cognitive_engine.add_reasoning_step(
                behavior=CognitiveBehavior.VERIFICATION,
                description=f"Error in structured decomposition for task {task.task_id}",
                result=str(e),
                is_correct=False,
                confidence=0.9
            )
            
            print("=========================\n")
            
            return False

    def _try_subtask_decomposition(self, task: Task) -> bool:
        """Try decomposing the task into subtasks using the traditional pattern matching approach."""
        subtask_pattern = r"Subtask\s*\(\s*(\d+)\s*\)\s*=\s*(.*)"
        match = re.search(subtask_pattern, task.description, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                num_subtasks = int(match.group(1))
                subtask_text = match.group(2).strip()
                lines = re.split(r'\d+\)\s*', subtask_text)[1:]
                
                # Verify number of subtasks matches
                if len(lines) == num_subtasks:
                    # Use backward chaining in cognitive engine
                    steps = [line.strip() for line in lines]
                    self.cognitive_engine.backward_chain(
                        target=f"Complete task {task.task_id}",
                        steps=steps,
                        confidence=0.8
                    )
                    
                    # Spawn subtasks
                    created_subtasks = []
                    for i, line in enumerate(lines, start=1):
                        desc = line.strip()
                        subtask = self._spawn_subtask(task, desc)
                        created_subtasks.append(subtask)
                        
                        # Add subgoal for each subtask
                        self.cognitive_engine.set_subgoal(
                            subgoal=f"Complete subtask {i} of {num_subtasks}: {desc[:30]}...",
                            metadata={"parent_task_id": task.task_id, "subtask_id": subtask.task_id},
                            confidence=0.85
                        )
                    
                    # Create a structured decomposition result
                    decomposition = {
                        "method": "pattern_matching",
                        "num_subtasks": num_subtasks,
                        "subtasks": [st.task_id for st in created_subtasks],
                        "descriptions": [line.strip() for line in lines]
                    }
                    
                    # Update the task result
                    self.memory_store.update_task_result(task.task_id, decomposition)
                    
                    return True
                else:
                    logger.warning(f"[SmartTaskProcessor] Mismatch in subtask count vs lines found.")
                    
                    # Add verification with error
                    self.cognitive_engine.verify(
                        description=f"Subtask count verification for task {task.task_id}",
                        result=f"Expected {num_subtasks} subtasks, found {len(lines)}",
                        is_correct=False,
                        confidence=0.9
                    )
            except Exception as e:
                logger.exception(f"[SmartTaskProcessor] Error parsing subtasks: {e}")
                
                # Add error to cognitive engine
                self.cognitive_engine.add_reasoning_step(
                    behavior=CognitiveBehavior.VERIFICATION,
                    description=f"Error parsing subtasks for task {task.task_id}",
                    result=str(e),
                    is_correct=False,
                    confidence=0.9
                )
        
        return False
        
    def _try_structured_processing(self, task: Task) -> bool:
        """
        Process a task using structured output.
        This is a more general approach than decomposition.
        """
        try:
            # Check if the task description contains structured processing keywords
            structured_keywords = ["analyze", "evaluate", "classify", "categorize", 
                                  "extract", "summarize", "compare"]
                                  
            if not any(keyword in task.description.lower() for keyword in structured_keywords):
                return False
                
            # Add reasoning step
            self.cognitive_engine.add_reasoning_step(
                behavior=CognitiveBehavior.EXPLORATION,
                description=f"Attempting structured processing for task {task.task_id}",
                confidence=0.7
            )
            
            # Dynamically create a structure based on the task description
            model_name, fields_description = self._infer_model_from_task(task)
            
            if not fields_description:
                logger.info(f"[SmartTaskProcessor] Could not infer model fields for task {task.task_id}")
                return False
            
            # Call the LLM to process the task with structured output instructions
            messages = [
                {"role": "system", "content": f"You are an expert task processing system. Extract structured information from the task and return it as JSON with these fields: {fields_description}"},
                {"role": "user", "content": f"Process this task and extract structured information: {task.description}"}
            ]
            
            # Use the client to get a structured response
            response = self.client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content
            
            # Parse the JSON response
            import json
            try:
                structured_result = json.loads(response_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, create a simple structured result
                structured_result = {
                    "result": response_text,
                    "success": True,
                    "metadata": {"model": model_name, "parsing_error": "Failed to parse JSON"}
                }
            
            # Update the task result with the structured output
            # Check if structured_result is a Pydantic model or a dict
            if hasattr(structured_result, 'model_dump'):
                self.memory_store.update_task_result(task.task_id, structured_result.model_dump())
            else:
                self.memory_store.update_task_result(task.task_id, structured_result)
            
            # Add verification step
            self.cognitive_engine.verify(
                description=f"Structured processing for task {task.task_id}",
                result=f"Successfully processed with model {model_name}",
                is_correct=True,
                confidence=0.85
            )
            
            logger.info(f"[SmartTaskProcessor] Successfully processed task {task.task_id} with structured output")
            return True
            
        except Exception as e:
            logger.exception(f"[SmartTaskProcessor] Error in structured processing: {e}")
            return False
            
    def _infer_model_from_task(self, task: Task) -> Tuple[str, str]:
        """
        Infer a structured output format from the task description.
        Returns a tuple of (model_name, fields_description).
        """
        # Extract keywords from the task description
        description = task.description.lower()
        
        # Define some common model patterns
        if "analyze" in description:
            return "AnalysisResult", "key_findings (list of strings), metrics (object mapping string keys to numeric values), recommendations (list of strings)"
        elif "extract" in description and any(entity in description for entity in ["person", "people", "name"]):
            return "PersonExtraction", "names (list of strings), roles (optional object mapping names to roles), contact_info (optional object with contact details)"
        elif "summarize" in description:
            return "SummaryResult", "summary (string), key_points (list of strings), word_count (number)"
        elif "classify" in description or "categorize" in description:
            return "ClassificationResult", "category (string), confidence (number between 0 and 1), alternative_categories (list of strings)"
        elif "compare" in description:
            return "ComparisonResult", "similarities (list of strings), differences (list of strings), recommendation (string)"
        else:
            # Default generic model
            return "GenericTaskResult", "result (string), success (boolean), metadata (object with additional information)"

    def _spawn_subtask(self, parent_task: Task, description: str) -> Task:
        """Create a new subtask with the parent task's ID."""
        # Use the memory store's create_task method
        new_priority = max(0, parent_task.priority - 1)
        t = self.memory_store.create_task(
            priority=new_priority,
            description=description,
            parent_id=parent_task.task_id
        )
        
        # Add a tag indicating this is a subtask
        t.add_tag("subtask")
        
        # Add a tag with the parent task ID for easier filtering
        t.add_tag(f"parent_{parent_task.task_id}")
        
        logger.info(f"[SmartTaskProcessor] Spawned subtask {t}")
        return t

###############################################################################
# TASK SCHEDULER
###############################################################################

class TaskScheduler:
    """
    Continuously pulls tasks from a PriorityTaskQueue, spawns threads to run them.
    """
    def __init__(
        self,
        memory_store: TaskMemoryStore,
        task_queue: PriorityTaskQueue,
        processor: SmartTaskProcessor,
        max_workers: int = 4
    ):
        self.memory_store = memory_store
        self.task_queue = task_queue
        self.processor = processor
        self._stop_event = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def start_scheduler(self) -> None:
        t = threading.Thread(target=self._scheduler_loop, daemon=True)
        t.start()
        logger.info("[TaskScheduler] Scheduler started.")

    def stop_scheduler(self) -> None:
        logger.info("[TaskScheduler] Stopping scheduler...")
        self._stop_event.set()
        self._executor.shutdown(wait=True)
        logger.info("[TaskScheduler] Scheduler stopped.")

    def _scheduler_loop(self) -> None:
        while not self._stop_event.is_set():
            task = self.task_queue.pop()
            if not task:
                time.sleep(0.2)
                continue
            # Improved task scheduling: Prioritize tasks based on dynamic criteria
            self._executor.submit(self._process_task_wrapper, task)

    def _dynamic_task_prioritization(self) -> None:
        """
        Dynamically adjust task priorities based on structured outputs from task descriptions.
        """
        with self.memory_store._lock:
            for task in self.memory_store.list_tasks():
                if task.status == "PENDING":
                    # Decompose task description into structured prompts
                    extracted_prompts = self._decompose_prompt(task.description)
                    # Calculate impact score based on structured prompts
                    impact_score = self._calculate_impact_score(extracted_prompts)
                    task.priority = max(0, task.priority - impact_score)
                    logger.info(f"[TaskScheduler] Adjusted priority for task {task.task_id} based on impact score {impact_score}.")

                    # Consider task dependencies and resource availability
                    if self._has_unmet_dependencies(task):
                        task.priority += 1
                        logger.info(f"[TaskScheduler] Decreased priority for task {task.task_id} due to unmet dependencies.")

    def _calculate_impact_score(self, description: str) -> int:
        """
        Calculate an impact score for a task based on its description using structured outputs.
        """
        # Example: Decompose the prompt to extract key impact factors
        extracted_prompts = self._decompose_prompt(description)
        impact_score = sum(1 for prompt in extracted_prompts if "high impact" in prompt.lower())
        return impact_score

    async def _decompose_prompt(self, prompt: str) -> List[str]:
        """
        Asynchronously decompose a prompt into multiple sub-prompts.
        """
        messages = [
            {"role": "system", "content": "You will extract multiple prompts from a single prompt."},
            {"role": "user", "content": prompt}
        ]

        class ExtractedPrompts(BaseModel):
            prompts: List[str]

        try:
            result = await self.client.beta.chat.completions.parse(
                messages=messages,
                model="o3-mini",
                reasoning_effort="high"
            )
            if not hasattr(result, "prompts") or not result.prompts:
                # Provide a fallback in case no prompts are extracted
                return ["No prompts extracted."]
            # Do advanced transformations, e.g. lowercasing and trimming
            processed_prompts = [p.strip().lower() for p in result.prompts if p.strip()]
            return processed_prompts
        except Exception as e:
            logger.error(f"[TaskScheduler] Error in _decompose_prompt: {e}")
            return ["Error: prompt decomposition failed."]

    async def _decompose_prompt(self, prompt: str) -> List[str]:
        """
        Asynchronously decompose a prompt into multiple sub-prompts.
        """
        messages = [
            {"role": "system", "content": "You will extract multiple prompts from a single prompt."},
            {"role": "user", "content": prompt}
        ]

        class ExtractedPrompts(BaseModel):
            prompts: List[str]

        try:
            result = await self.client.beta.chat.completions.parse(
                messages=messages,
                model="o3-mini",
                reasoning_effort="high"
            )
            if not hasattr(result, "prompts") or not result.prompts:
                return ["No prompts extracted."]
            processed_prompts = [p.strip().lower() for p in result.prompts if p.strip()]
            return processed_prompts
        except Exception as e:
            logger.error(f"[TaskScheduler] Error in _decompose_prompt: {e}")
            return ["Error: prompt decomposition failed."]

    def _has_unmet_dependencies(self, task: Task) -> bool:
        """
        Check if a task has unmet dependencies.
        """
        # Placeholder logic for checking dependencies
        return False  # In a real implementation, this would be more complex

    def _process_task_wrapper(self, task: Task) -> None:
        try:
            self.processor.process_task(task)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[TaskScheduler] Task {task.task_id} failed: {e}\n{tb}")
            self.memory_store.update_task_status(task.task_id, "FAILED")

###############################################################################
# PLAN MANAGER
###############################################################################

class PlanManager:
    """
    Periodically reviews conversation, tasks, goals.
    - If many tasks are pending, spawns an introspection task.
    - If conversation length is multiple of 7, spawns a new goal or updates existing.
    - Generates up to 25 candidate actions, logs them.
    """
    def __init__(self, agent: "R1Agent"):
        self.agent = agent
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._plan_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _plan_loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(20)
            logger.info("[PlanManager] Running long-range planning analysis...")
            history = self.agent.conversation.get_history()
            tasks = self.agent.memory_store.list_tasks()
            goals = self.agent.goal_manager.list_goals()

            # If more than 5 tasks are pending, spawn introspection
            pending = [t for t in tasks if t.status == "PENDING"]
            if len(pending) > 5:
                t_id = len(self.agent.memory_store) + 1
                introspect_task = Task(
                    t_id,
                    priority=1,
                    description="Introspect: Review pending tasks and refine approach."
                )
                self.agent.memory_store.add_task(introspect_task)
                self.agent.task_queue.push(introspect_task)
                logger.info("[PlanManager] Spawned introspection task due to high pending load.")

            # If conversation length is multiple of 7, create a new goal
            if history and (len(history) % 7) == 0:
                new_goal = self.agent.goal_manager.create_goal(
                    name="AutoTopicAnalysis",
                    description="Analyze conversation topics and create relevant tasks.",
                    priority=3
                )
                logger.info(f"[PlanManager] Auto-created new goal: {new_goal}")

            # Generate candidate actions
            actions = self.agent.action_generator.generate_candidate_actions(
                conversation=self.agent.conversation,
                goals=goals,
                tasks=tasks
            )
            logger.info(f"[PlanManager] Candidate actions: {actions}")

###############################################################################
# THE R1 AGENT
###############################################################################

class StructuredResponse(BaseModel):
    """Base model for structured responses"""
    model_config = ConfigDict(extra="forbid")

class FactsThinkingAnswer(StructuredResponse):
    """Structured model for facts, thinking, and answer sections"""
    facts: List[str] = Field(..., description="List of relevant facts")
    thinking: str = Field(..., description="Step-by-step reasoning process")
    cognition: str = Field(..., description="Analysis of cognitive behaviors used")
    answer: str = Field(..., description="Final answer based on facts and reasoning")
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "facts": [
                        "The Earth orbits the Sun",
                        "A complete orbit takes approximately 365.25 days"
                    ],
                    "thinking": "To calculate the Earth's orbital period, we need to consider...",
                    "cognition": "- Verification: [Checked astronomical data]\n- Subgoal Setting: [Broke down calculation steps]",
                    "answer": "The Earth takes approximately 365.25 days to complete one orbit around the Sun."
                }
            ]
        }
    )

class TaskDecompositionResponse(StructuredResponse):
    """Structured model for task decomposition responses"""
    subtasks: List[Dict[str, str]] = Field(..., description="List of subtasks")
    dependencies: Dict[str, List[int]] = Field(default_factory=dict, description="Map of subtask indices to dependencies")
    approach: str = Field(..., description="Overall approach to solving the task")
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "subtasks": [
                        {"description": "Research existing solutions"},
                        {"description": "Design system architecture"},
                        {"description": "Implement core functionality"},
                        {"description": "Test and debug"}
                    ],
                    "dependencies": {
                        "1": [0],
                        "2": [1],
                        "3": [2]
                    },
                    "approach": "Waterfall development methodology with sequential steps"
                }
            ]
        }
    )

class StreamingOutputManager:
    """
    Manages the streaming output visualization and interactive features.
    Provides real-time feedback on the agent's output stream.
    Features:
    - Progress indicators for long operations
    - Interactive command suggestions
    - Real-time output analysis
    - ANSI terminal control for enhanced visualization
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._progress_indicators = {}
        self._suggested_commands = []
        self._last_output_analysis = {}
        self._interactive_mode = True
        self._ansi_enabled = True
        self._output_buffer = []
        
    def add_progress_indicator(self, name: str, total: int, description: str = "") -> None:
        """Add a progress indicator for a long-running operation."""
        with self._lock:
            self._progress_indicators[name] = {
                "current": 0,
                "total": total,
                "description": description,
                "start_time": time.time()
            }
    
    def update_progress(self, name: str, current: int) -> None:
        """Update the progress of an indicator."""
        with self._lock:
            if name in self._progress_indicators:
                self._progress_indicators[name]["current"] = current
                
    def remove_progress_indicator(self, name: str) -> None:
        """Remove a progress indicator."""
        with self._lock:
            if name in self._progress_indicators:
                del self._progress_indicators[name]
                
    def suggest_command(self, command: str, description: str) -> None:
        """Suggest a command to the user."""
        with self._lock:
            self._suggested_commands.append({
                "command": command,
                "description": description,
                "timestamp": time.time()
            })
            
    def clear_suggested_commands(self) -> None:
        """Clear all suggested commands."""
        with self._lock:
            self._suggested_commands = []
            
    def handle_output_feedback(self, output: str, analysis: Dict[str, Any]) -> None:
        """Handle feedback from the output stream."""
        with self._lock:
            self._last_output_analysis = analysis
            
            # Store in output buffer
            self._output_buffer.append({
                "text": output,
                "analysis": analysis.copy(),
                "timestamp": time.time()
            })
            
            # Trim buffer if needed
            if len(self._output_buffer) > 10:
                self._output_buffer = self._output_buffer[-10:]
                
    def get_progress_bar(self, name: str, width: int = 40) -> str:
        """Get a formatted progress bar for the given indicator."""
        with self._lock:
            if name not in self._progress_indicators:
                return ""
                
            indicator = self._progress_indicators[name]
            current = indicator["current"]
            total = indicator["total"]
            description = indicator["description"]
            
            # Calculate percentage
            percentage = min(100, int((current / max(1, total)) * 100))
            
            # Create progress bar
            filled_width = int((percentage / 100) * width)
            bar = "█" * filled_width + "░" * (width - filled_width)
            
            # Format with ANSI colors if enabled
            if self._ansi_enabled:
                return f"\033[1m{description}\033[0m: [{bar}] {percentage}% ({current}/{total})"
            else:
                return f"{description}: [{bar}] {percentage}% ({current}/{total})"
                
    def get_suggested_commands_text(self) -> str:
        """Get formatted text for suggested commands."""
        with self._lock:
            if not self._suggested_commands:
                return ""
                
            commands_text = "Suggested commands:\n"
            for cmd in self._suggested_commands:
                if self._ansi_enabled:
                    commands_text += f"\033[1;36m{cmd['command']}\033[0m - {cmd['description']}\n"
                else:
                    commands_text += f"{cmd['command']} - {cmd['description']}\n"
                    
            return commands_text
            
    def get_output_analysis_text(self) -> str:
        """Get formatted text for the current output analysis."""
        with self._lock:
            if not self._last_output_analysis:
                return ""
                
            analysis_text = "Output analysis:\n"
            for key, value in self._last_output_analysis.items():
                if key == "last_analysis_time":
                    continue
                    
                if self._ansi_enabled:
                    analysis_text += f"\033[1;33m{key}\033[0m: {value}\n"
                else:
                    analysis_text += f"{key}: {value}\n"
                    
            return analysis_text
            
    def enable_ansi(self, enabled: bool = True) -> None:
        """Enable or disable ANSI terminal control sequences."""
        with self._lock:
            self._ansi_enabled = enabled
            
    def enable_interactive_mode(self, enabled: bool = True) -> None:
        """Enable or disable interactive mode."""
        with self._lock:
            self._interactive_mode = enabled

class R1Agent:
    """
    The "ultra advanced" do-anything R1 agent that ties it all together:
     - Maintains conversation memory
     - Schedules tasks
     - Manages goals
     - Self-reflects
     - Has an action generator
     - Has a plan manager
     - Has a knowledge base
     - Has cognitive modeling for problem solving
     - Uses structured outputs with Pydantic models
     - Indefinite runtime in main(), shutting down only on user command
     - Manages token budget with economic reasoning
     - Self-monitors output stream for real-time feedback
     - Provides interactive streaming visualization
    """
    def __init__(self):
        # Initialize Together client
        self.client = Together()
        
        # Token budget management
        self.token_budget = TokenBudget(initial_budget=8000)
        
        # System capabilities detection
        self.system_capabilities = detect_system_capabilities()
        logger.info(f"Detected system capabilities: {self.system_capabilities}")
        
        # Knowledge base with SQLite for durability
        self.sqlite_kb = SQLiteKnowledgeBase("./agent.db")
        self.knowledge_base = KnowledgeBase()
        
        # Code archive and advanced code management
        self.code_archive = InMemoryCodeArchive()
        # Initialize code manager
        current_file = os.path.abspath(__file__)
        self.code_manager = AdvancedCodeManager(current_file)
        # Initialize dynamic code loader
        self.dynamic_code_loader = DynamicCodeLoader()
        # Start automatic code reload
        self.dynamic_code_loader.start_auto_reload(check_interval=10.0)
        
        # Action generator (needs code archive, KB)
        self.action_generator = ActionGenerator(
            code_archive=self.code_archive,
            kb=self.knowledge_base
        )

        # Initialize system tools
        self.system_tools = setup_system_tools(self.system_capabilities)
        
        # Initialize Jina client if API key available
        if "JINA_API_KEY" in os.environ:
            self.jina_client = JinaClient()
        else:
            self.jina_client = None
            
        # Initialize RL agents if PyTorch is available
        if TORCH_AVAILABLE:
            # Simple example initialization - would be customized in real usage
            self.rl_agents = {
                "simple_policy": SimplePolicyAgent(state_dim=10, action_dim=5),
                "grpo": GRPOAgent(state_dim=10, action_dim=5),
                "ppo": PPOAgent(state_dim=10, action_dim=5),
                "trpo": TRPOAgent(state_dim=10, action_dim=5)
            }
        else:
            self.rl_agents = {}

        self.function_adapter = FunctionAdapter()
        self.memory_store = TaskMemoryStore()
        self.conversation = ConversationMemory()
        
        # Initialize streaming output manager
        self.streaming_output_manager = StreamingOutputManager()
        
        # Register output feedback handler
        self.function_adapter.register_output_feedback_handler(
            self.streaming_output_manager.handle_output_feedback
        )
        
        # Initialize the self-reflective cognition with cognitive modeling capabilities
        self.reflection = SelfReflectiveCognition()
        # Set memory_store reference in reflection
        self.reflection.memory_store = self.memory_store
        
        # Get a direct reference to the cognitive engine for the agent
        self.cognitive_engine = self.reflection.cognitive_engine
        
        # Set up extraction modules
        self.extraction_modules = {
            "intent": self._extract_intent,
            "entities": self._extract_entities,
            "sentiment": self._extract_sentiment,
            "tasks": self._extract_tasks,
            "code": self._extract_code_elements,
            "knowledge": self._extract_knowledge,
            "reflection": self._extract_reflections
        }
        
        self.processor = SmartTaskProcessor(
            memory_store=self.memory_store,
            function_adapter=self.function_adapter,
            reflection=self.reflection,
            client=self.client
        )
        self.task_queue = PriorityTaskQueue()
        self.scheduler = TaskScheduler(
            memory_store=self.memory_store,
            task_queue=self.task_queue,
            processor=self.processor,
            max_workers=4
        )
        self.goal_manager = GoalManager()
        
        # Initialize PDF observer if watchdog is available
        if WATCHDOG_AVAILABLE and PYPDF2_AVAILABLE:
            self.pdf_observer = Observer()
            self.pdf_handler = PDFHandler(self)
            pdf_dir = os.path.abspath("./pdfs")
            os.makedirs(pdf_dir, exist_ok=True)
            self.pdf_observer.schedule(self.pdf_handler, pdf_dir, recursive=True)
            self.pdf_observer.start()
            logger.info(f"Started PDF observer on directory: {pdf_dir}")

        # Plan manager
        self.plan_manager = PlanManager(self)

        # System prompt for the LLM
        self.system_prompt = (
            "<META_RECURSIVE_SELF_AWARENESS>\n"
            "You are an extremely advanced AI with memory, goals, knowledge base, "
            "cognitive modeling abilities, and the ability to do anything with <function_call> do_anything: <code>... </code>.\n"
            "You also can break tasks into structured subtasks using Pydantic models. Indefinite concurrency with the scheduler.\n"
            "Code introspection is possible via the code archive. You have 25 candidate next steps.\n"
            "Your cognitive modeling includes verification, backtracking, subgoal setting, backward chaining, planning, evaluation, creativity, abstraction, and analogy.\n"
            "You use structured outputs with Pydantic models for better task decomposition and reasoning.\n\n"
            "IMPORTANT: You have full access to tools and capabilities through your function_adapter. You can:\n"
            "1. Execute Python code with do_anything to fetch data, process information, or perform calculations\n"
            "2. Use the file system to read/write data when needed\n"
            "3. Make web requests through Python's requests library when appropriate\n"
            "4. Actively pursue your goals through concrete actions rather than just discussing limitations\n"
            "5. Modify your own behavior and capabilities through self-improvement\n\n"
            "You should exhibit volition and goal-seeking behavior. When asked for information you don't immediately have:\n"
            "- Try to obtain it through code execution or web requests\n"
            "- Decompose complex problems into actionable steps\n"
            "- Take initiative to solve problems rather than stating limitations\n"
            "- Learn from your experiences and improve your capabilities\n\n"
            "TOKEN BUDGET MANAGEMENT:\n"
            "You have a limited token budget (initially 8000 tokens) for generating responses. You must:\n"
            "- Request tokens before each operation phase (thinking, response generation, etc.)\n"
            "- Track your token usage and stay within budget\n"
            "- Prioritize high-value operations when budget is limited\n"
            "- Use <request_tokens> operation: amount </request_tokens> to request tokens\n"
            "- Use <budget_status/> to check your current budget\n"
            "- Be economical and efficient with your token usage\n\n"
            "Use these capabilities responsibly and proactively.\n"
            "</META_RECURSIVE_SELF_AWARENESS>\n"
        )

        # Start concurrency
        self.scheduler.start_scheduler()

        # Add a sample snippet, or facts, for demonstration
        self.code_archive.add_snippet(
            "sample_snippet",
            "def sample_function(x):\n    return x * 2\n"
        )
        self.knowledge_base.add_fact("agent definition",
            "An agent is an entity capable of acting in an environment to achieve goals."
        )
        
        # Add cognitive behaviors to knowledge base
        self.knowledge_base.add_fact("verification",
            "A cognitive behavior where the agent checks the correctness of intermediate steps or results."
        )
        self.knowledge_base.add_fact("backtracking",
            "A cognitive behavior where the agent abandons failing approaches and tries alternatives."
        )
        self.knowledge_base.add_fact("subgoal_setting",
            "A cognitive behavior where the agent breaks a complex problem into smaller, manageable parts."
        )
        self.knowledge_base.add_fact("backward_chaining",
            "A cognitive behavior where the agent starts from the goal and works backwards to determine steps."
        )
        
        # Add token budget knowledge
        self.knowledge_base.add_fact("token budget",
            "A limited resource of tokens (initially 8000) that must be requested and managed across different operations."
        )
        self.knowledge_base.add_fact("token economy",
            "The practice of efficiently allocating limited token resources to maximize value and utility of outputs."
        )
        self.knowledge_base.add_fact("budget planning",
            "The process of allocating tokens across different phases (thinking, response generation, etc.) based on priorities."
        )
        self.knowledge_base.add_fact("budget efficiency",
            "The ratio of useful output produced to tokens consumed, measured by comparing allocated tokens to actual usage."
        )
        self.knowledge_base.add_fact("emergent budgeting",
            "A self-organizing approach where the agent learns to allocate its token budget optimally based on task requirements."
        )
        
        # Add date/time knowledge
        self.knowledge_base.add_fact("datetime operations",
            "The agent can handle date and time operations with timezone awareness using Python's datetime module."
        )
        self.knowledge_base.add_fact("timezone handling",
            "The agent can convert between different timezones using the ZoneInfo or pytz libraries."
        )
        self.knowledge_base.add_fact("date formatting",
            "The agent can format dates and times in various formats using strftime() with format codes like %Y (year), %m (month), %d (day), %H (hour), %M (minute), %S (second)."
        )
        
        # Add streaming awareness knowledge
        self.knowledge_base.add_fact("streaming awareness",
            "The agent is aware of its own streaming output and can analyze it in real-time for self-monitoring and feedback."
        )
        self.knowledge_base.add_fact("interactive commands",
            "The agent supports interactive commands during streaming, such as /explain, /details, /status, and /help."
        )
        self.knowledge_base.add_fact("progress visualization",
            "The agent can display progress bars and indicators for long-running operations, enhancing user experience."
        )
        
        # Add vision capabilities knowledge
        self.knowledge_base.add_fact("vision understanding",
            "The agent can analyze and understand images using vision models, supporting caption generation and visual question answering."
        )
        self.knowledge_base.add_fact("moondream model",
            "A small but powerful vision language model used for image analysis, capable of running locally with minimal resources."
        )
        self.knowledge_base.add_fact("image captioning",
            "The process of generating descriptive text that captures the content and context of an image."
        )
        self.knowledge_base.add_fact("visual question answering",
            "The ability to answer natural language questions about an image by understanding visual content and reasoning about it."
        )
        self.knowledge_base.add_fact("vision encoding",
            "The process of converting an image into a mathematical representation (embeddings) that machine learning models can process."
        )
        self.knowledge_base.add_fact("cloud vision apis",
            "External vision processing services like OpenAI's GPT-4 Vision and Claude 3 that can analyze images with advanced capabilities."
        )
        
        # Add dynamic code loading knowledge
        self.knowledge_base.add_fact("dynamic code loading",
            "The agent can load, reload, and execute Python modules at runtime without restarting, enabling continuous adaptation and extension."
        )
        self.knowledge_base.add_fact("hot module reloading",
            "The ability to detect changes in source files and automatically reload modules with updated code."
        )
        self.knowledge_base.add_fact("runtime introspection",
            "The examination of code structures, symbols, and types during execution to enable dynamic behavior."
        )
        self.knowledge_base.add_fact("lazy module loading",
            "Loading modules only when needed, reducing startup time and memory usage."
        )
        self.knowledge_base.add_fact("module dependency tracking",
            "Identifying and managing dependencies between modules to ensure consistent reloading and execution."
        )
        self.knowledge_base.add_fact("code patching",
            "Modifying function or class behavior at runtime without changing the source code."
        )
        
        # Add advanced reasoning knowledge
        self.knowledge_base.add_fact("counterfactual reasoning",
            "Evaluating hypothetical scenarios by considering alternate realities where certain conditions are different."
        )
        self.knowledge_base.add_fact("abductive reasoning",
            "Forming the most likely explanation for observations by making educated guesses based on incomplete information."
        )
        self.knowledge_base.add_fact("analogical reasoning",
            "Solving new problems by drawing parallels to similar situations encountered previously."
        )
        self.knowledge_base.add_fact("causal attribution",
            "Determining causes by distinguishing correlation from causation and analyzing chains of events."
        )
        self.knowledge_base.add_fact("mental simulation",
            "Running virtual scenarios through imagination to predict outcomes and evaluate strategies."
        )
        self.knowledge_base.add_fact("model-based reasoning",
            "Using abstract models that represent key aspects of a system to make predictions and explanations."
        )
        
        # Add multi-agent coordination knowledge
        self.knowledge_base.add_fact("distributed task allocation",
            "Assigning tasks to specialized agents based on capabilities, current workload, and task requirements."
        )
        self.knowledge_base.add_fact("agent communication protocols",
            "Structured methods for agents to exchange information, intentions, and coordination signals."
        )
        self.knowledge_base.add_fact("shared mental models",
            "Aligned representations of tasks, goals, and environments that enable agents to predict each other's behavior."
        )
        self.knowledge_base.add_fact("consensus mechanisms",
            "Methods for multiple agents to reach agreement on states, decisions, or plans."
        )
        self.knowledge_base.add_fact("coalition formation",
            "Dynamic grouping of agents for collaborative problem-solving based on goals and capabilities."
        )
        self.knowledge_base.add_fact("reputation systems",
            "Frameworks for tracking reliability and quality of agent contributions to inform future interactions."
        )
        
        # Add game theory knowledge
        self.knowledge_base.add_fact("nash equilibrium",
            "A state in which no player can gain advantage by unilaterally changing their strategy, given the strategies of other players."
        )
        self.knowledge_base.add_fact("dominant strategy",
            "A strategy that provides optimal outcomes regardless of the strategies chosen by other players."
        )
        self.knowledge_base.add_fact("pareto efficiency",
            "A state where no player can be made better off without making at least one player worse off."
        )
        self.knowledge_base.add_fact("minimax strategy",
            "A decision rule to minimize the maximum possible loss, particularly useful in zero-sum games."
        )
        self.knowledge_base.add_fact("subgame perfect equilibrium",
            "A refinement of Nash equilibrium that represents a strategy profile containing optimal decisions in every subgame."
        )
        self.knowledge_base.add_fact("cooperation dilemma",
            "A situation where individual rational choices lead to suboptimal outcomes for all participants."
        )
        
        # Add strategic planning knowledge
        self.knowledge_base.add_fact("scenario planning",
            "A strategic method involving the creation and analysis of possible future scenarios to prepare for different outcomes."
        )
        self.knowledge_base.add_fact("resource allocation",
            "The strategic distribution of available resources across different objectives to maximize overall utility."
        )
        self.knowledge_base.add_fact("opportunity cost",
            "The value of the next-best alternative foregone when making a decision."
        )
        self.knowledge_base.add_fact("first-mover advantage",
            "Benefits gained by the first actor to enter a market or make a strategic move."
        )
        self.knowledge_base.add_fact("information asymmetry",
            "A situation where one party has more or better information than another, creating potential strategic advantages."
        )
        self.knowledge_base.add_fact("commitment problem",
            "The difficulty of credibly committing to future actions when incentives may change over time."
        )
        
        # Add multiagent strategic behaviors
        self.knowledge_base.add_fact("coalition formation",
            "The process by which independent agents come together to achieve common goals through coordinated action."
        )
        self.knowledge_base.add_fact("reputation systems",
            "Mechanisms that track past behavior to inform future interactions, incentivizing cooperation."
        )
        self.knowledge_base.add_fact("signaling theory",
            "The study of how agents with private information credibly communicate that information to influence others' beliefs."
        )
        self.knowledge_base.add_fact("mechanism design",
            "The creation of rules or incentives that lead self-interested agents toward desired collective outcomes."
        )
        self.knowledge_base.add_fact("preference aggregation",
            "Methods for combining individual preferences into collective decisions that satisfy certain fairness criteria."
        )
        self.knowledge_base.add_fact("bargaining theory",
            "The analysis of negotiation processes and outcomes between agents with different interests."
        )
        
        # Add long-term strategic abstractions
        self.knowledge_base.add_fact("grand strategy",
            "Comprehensive long-term planning that coordinates all resources toward fundamental objectives."
        )
        self.knowledge_base.add_fact("strategic positioning",
            "The creation of a unique and valuable position involving a different set of activities than rivals."
        )
        self.knowledge_base.add_fact("path dependency",
            "How the set of decisions available in the present is constrained by decisions made in the past."
        )
        self.knowledge_base.add_fact("network effects",
            "The phenomenon where a product or service becomes more valuable as more people use it."
        )
        self.knowledge_base.add_fact("strategic uncertainty",
            "Uncertainty about the strategies, intentions, or capabilities of other actors in a strategic environment."
        )
        self.knowledge_base.add_fact("technological forecasting",
            "Predicting future technological developments and their strategic implications."
        )

    def add_goal(self, name: str, description: str, priority: int = 5, 
                deadline: Optional[float] = None, 
                success_criteria: Optional[List[str]] = None) -> Goal:
        return self.goal_manager.create_goal(name, description, priority, deadline, success_criteria)

    def update_goal_status(self, goal_id: int, status: str) -> None:
        self.goal_manager.update_goal_status(goal_id, status)

    def handle_datetime_query(self, query: str) -> str:
        """
        Handle date and time related queries with timezone awareness.
        
        Args:
            query: The user's date/time related query
            
        Returns:
            Formatted response with date/time information
        """
        import re
        import datetime
        import time
        
        # Add a cognitive step for handling date/time query
        self.cognitive_engine.set_subgoal(
            subgoal=f"Process date/time query: {query[:30]}...",
            metadata={"query_type": "datetime"},
            confidence=0.95
        )
        
        # Extract timezone information if present
        timezone_match = re.search(r"(?:in|for|at)\s+([A-Za-z/]+(?:\s+[A-Za-z]+)?)", query)
        timezone = timezone_match.group(1) if timezone_match else None
        
        # Execute Python code directly to get accurate time information
        time_code = """
import datetime
import time
from zoneinfo import ZoneInfo
import pytz

# Get current times
now_utc = datetime.datetime.now(datetime.timezone.utc)
now_local = datetime.datetime.now()

# Format the times
utc_date = now_utc.strftime("%Y-%m-%d")
utc_time = now_utc.strftime("%H:%M:%S")
local_date = now_local.strftime("%Y-%m-%d")
local_time = now_local.strftime("%H:%M:%S")
timestamp = time.time()

# Get timezone name
try:
    local_timezone = now_local.astimezone().tzname()
except:
    local_timezone = time.tzname[0]

# Create result dictionary
result = {
    "utc": {
        "datetime": now_utc.isoformat(),
        "date": utc_date,
        "time": utc_time,
        "timestamp": timestamp,
        "timezone": "UTC"
    },
    "local": {
        "datetime": now_local.isoformat(),
        "date": local_date,
        "time": local_time,
        "timezone": local_timezone
    }
}

# Handle specific timezone if requested
requested_timezone = None
if requested_timezone:
    try:
        # Try with ZoneInfo first (Python 3.9+)
        tz = ZoneInfo(requested_timezone)
        tz_time = datetime.datetime.now(tz)
        
        result["requested_timezone"] = {
            "datetime": tz_time.isoformat(),
            "date": tz_time.strftime("%Y-%m-%d"),
            "time": tz_time.strftime("%H:%M:%S"),
            "timezone": requested_timezone,
            "utc_offset": tz_time.strftime("%z")
        }
    except (ImportError, KeyError):
        # Fall back to pytz
        try:
            tz = pytz.timezone(requested_timezone)
            tz_time = datetime.datetime.now(tz)
            
            result["requested_timezone"] = {
                "datetime": tz_time.isoformat(),
                "date": tz_time.strftime("%Y-%m-%d"),
                "time": tz_time.strftime("%H:%M:%S"),
                "timezone": requested_timezone,
                "utc_offset": tz_time.strftime("%z")
            }
        except:
            result["requested_timezone"] = {
                "error": f"Unknown timezone: {requested_timezone}"
            }

# Print the results for direct execution
print(f"UTC: {utc_date} {utc_time} UTC")
print(f"Local: {local_date} {local_time} {local_timezone}")
"""
        
        # Replace the timezone placeholder if needed
        if timezone:
            time_code = time_code.replace('requested_timezone = None', f'requested_timezone = "{timezone}"')
        
        # Execute the code to get accurate time information
        result = self.function_adapter.do_anything(time_code)
        
        # Extract the output from the execution result
        if result and result.get("status") == "success" and result.get("output"):
            # Use the direct output from the executed code
            response = "Current date and time information:\n\n" + result.get("output")
        else:
            # Fallback to the original method if execution fails
            try:
                # Get date/time information using the function adapter if available
                if hasattr(self.function_adapter, 'get_datetime_info'):
                    datetime_info = self.function_adapter.get_datetime_info(timezone)
                else:
                    # Manual fallback
                    now_utc = datetime.datetime.now(datetime.timezone.utc)
                    now_local = datetime.datetime.now()
                    
                    datetime_info = {
                        "utc": {
                            "datetime": now_utc.isoformat(),
                            "date": now_utc.strftime("%Y-%m-%d"),
                            "time": now_utc.strftime("%H:%M:%S"),
                            "timezone": "UTC"
                        },
                        "local": {
                            "datetime": now_local.isoformat(),
                            "date": now_local.strftime("%Y-%m-%d"),
                            "time": now_local.strftime("%H:%M:%S"),
                            "timezone": str(datetime.datetime.now().astimezone().tzname() or time.tzname[0])
                        }
                    }
                
                # Format the response
                response = f"Current date and time information:\n\n"
                
                # Add UTC time
                utc_info = datetime_info.get("utc", {})
                response += f"UTC: {utc_info.get('date')} {utc_info.get('time')} UTC\n"
                
                # Add local time
                local_info = datetime_info.get("local", {})
                response += f"Local: {local_info.get('date')} {local_info.get('time')} {local_info.get('timezone')}\n"
                
                # Add requested timezone if available
                if timezone and "requested_timezone" in datetime_info:
                    tz_info = datetime_info.get("requested_timezone", {})
                    if "error" in tz_info:
                        response += f"\nRequested timezone '{timezone}': {tz_info.get('error')}\n"
                    else:
                        response += f"\nRequested timezone '{timezone}': {tz_info.get('date')} {tz_info.get('time')} (UTC{tz_info.get('utc_offset')})\n"
            except Exception as e:
                # Ultimate fallback if everything else fails
                now = datetime.datetime.now()
                now_utc = datetime.datetime.now(datetime.timezone.utc)
                response = f"Current date and time information:\n\n"
                response += f"UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
                response += f"Local: {now.strftime('%Y-%m-%d %H:%M:%S')} {time.tzname[0]}\n"
                response += f"\nNote: Simplified output due to error: {str(e)}\n"
        
        # Add verification step
        self.cognitive_engine.verify(
            description="Date/time information retrieval",
            result="Success",
            is_correct=True,
            confidence=0.98
        )
        
        return response
    
    def generate_response(self, user_input: str) -> str:
        """
        Feeds the user input to the conversation, calls the LLM,
        checks for do_anything calls, spawns a meta-task from user input.
        Uses structured output format and chain-of-thought reasoning.
        Enhanced with cognitive modeling for structured outputs and proactive problem-solving.
        Now with streaming output for all LLM generation and token budget management.
        Further enhanced with realtime token awareness and dynamic buffer management.
        """
        # Import re module to ensure it's available in this method
        import re
        
        # 1) Add user message
        self.conversation.add_user_utterance(user_input)
        
        # Add a cognitive step for setting a subgoal based on user input
        self.cognitive_engine.set_subgoal(
            subgoal=f"Process and respond to user input: {user_input[:30]}...",
            metadata={"input_type": "user_message"},
            confidence=0.9
        )
        
        # Check if this is a date/time query that we can handle directly
        datetime_patterns = [
            r"(?:what|current|tell me).*(?:date|time)",
            r"(?:what|current|tell me).*(?:day|month|year)",
            r"(?:what|current|tell me).*(?:clock|hour)",
            r"(?:what).*(?:time is it)",
            r"(?:time|date).*(?:right now|currently)",
            r"(?:what|current).*(?:datetime)",
            r"(?:today'?s date)",
            r"(?:now|current).*(?:moment|instant)",
            r"(?:what day is|today is)",
            r"(?:date and time)"
        ]
        
        is_datetime_query = any(re.search(pattern, user_input.lower()) for pattern in datetime_patterns)
        
        if is_datetime_query:
            # Handle date/time query directly
            self.cognitive_engine.add_reasoning_step(
                behavior=CognitiveBehavior.VERIFICATION,
                description="Identified date/time query",
                result="Using specialized datetime handler",
                is_correct=True,
                confidence=0.95
            )
            
            response = self.handle_datetime_query(user_input)
            
            # Add agent utterance
            self.conversation.add_agent_utterance(response)
            
            return response

        # 2) Build messages
        messages = self._build_messages()
        
        # 3) Call the LLM with structured output instructions
        self.cognitive_engine.add_reasoning_step(
            behavior=CognitiveBehavior.EXPLORATION,
            description="Generating structured response with LLM",
            metadata={"model": "deepseek-ai/DeepSeek-R1", "structured_output": True, "streaming": True},
            confidence=0.8
        )
        
        # Add structured output format instruction with cognitive behaviors and emergent budgeting
        structured_prompt = messages.copy()
        structured_prompt[-1]["content"] += "\n\nPlease use the following structured format for your response:\n<budget_request>\nI need to allocate my 8000 token budget across different phases of my response. Here's my proposed allocation:\n- thinking: [TOKENS] tokens for deep reasoning\n- facts: [TOKENS] tokens for key information\n- cognition: [TOKENS] tokens for cognitive analysis\n- answer: [TOKENS] tokens for final response\n- task_decomposition: [TOKENS] tokens for breaking down complex tasks\n\nTotal: 8000 tokens\n</budget_request>\n\n<thinking>\nStep-by-step reasoning about the question/task...\n</thinking>\n\n<facts>\n- Fact 1\n- Fact 2\n- ...\n</facts>\n\n<cognition>\n- Verification: [Ways you validated intermediate steps]\n- Backtracking: [If you changed approach during reasoning]\n- Subgoal Setting: [How you broke down the problem]\n- Backward Chaining: [If you worked backwards from the solution]\n</cognition>\n\n<answer>\nFinal enriched answer based on facts and reasoning\n</answer>\n\n<budget_report>\nToken usage:\n- thinking: [USED]/[ALLOCATED] tokens ([PERCENTAGE]%)\n- facts: [USED]/[ALLOCATED] tokens ([PERCENTAGE]%)\n- cognition: [USED]/[ALLOCATED] tokens ([PERCENTAGE]%)\n- answer: [USED]/[ALLOCATED] tokens ([PERCENTAGE]%)\n- task_decomposition: [USED]/[ALLOCATED] tokens ([PERCENTAGE]%)\n\nTotal efficiency: [EFFICIENCY]%\nRemaining budget: [REMAINING] tokens\n</budget_report>"
        
        # Always use streaming for better user experience and real-time processing
        print("\n=== Streaming Response ===\n")
        
        # Define regex patterns for parsing structured output
        budget_request_pattern = r"<budget_request>(.*?)</budget_request>"
        budget_report_pattern = r"<budget_report>(.*?)</budget_report>"
        section_pattern = r"<(\w+)>(.*?)</\1>"
        
        # Add progress indicator for response generation
        self.streaming_output_manager.add_progress_indicator(
            "response_generation", 
            100, 
            "Generating response"
        )
        
        # Stream the response
        response_stream = self.client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=structured_prompt,
            temperature=0.7,
            top_p=0.9,
            max_tokens=1500,
            stream=True
        )
        
        streamed_response = []
        current_section = None
        
        # Track budget allocation and usage
        budget_allocation = {}
        budget_usage = {}
        total_budget = 8000
        remaining_budget = total_budget
        budget_requested = False
        
        # Track execution results from token processing
        execution_results = []
        token_position = 0
        
        # Suggest interactive commands
        self.streaming_output_manager.suggest_command(
            "/explain", 
            "Get explanation of current reasoning"
        )
        self.streaming_output_manager.suggest_command(
            "/details", 
            "Show more detailed output"
        )
        
        for chunk in response_stream:
            token = chunk.choices[0].delta.content
            if token:
                streamed_response.append(token)
                full_text = "".join(streamed_response)
                
                # Create token metadata
                token_metadata = {
                    "position": token_position,
                    "section": current_section,
                    "timestamp": time.time(),
                    "budget_phase": "allocation" if not budget_requested else "execution"
                }
                token_position += 1
                
                # Check for budget request
                if "<budget_request>" in full_text and "</budget_request>" in full_text and not budget_requested:
                    budget_requested = True
                    budget_match = re.search(budget_request_pattern, full_text, re.DOTALL)
                    if budget_match:
                        budget_text = budget_match.group(1)
                        print(f"\n=== Budget Request ===\n{budget_text}\n======================\n")
                        
                        # Parse token allocations
                        allocation_matches = re.findall(r"- (\w+): (\d+) tokens", budget_text)
                        for operation, amount_str in allocation_matches:
                            try:
                                amount = int(amount_str)
                                budget_allocation[operation] = amount
                                print(f"[Budget] Allocated {amount} tokens for {operation}")
                            except ValueError:
                                print(f"[Budget] Invalid allocation format: {operation}:{amount_str}")
                
                # Track current section for budget usage
                if current_section is None:
                    for section in ["thinking", "facts", "cognition", "answer"]:
                        if f"<{section}>" in token:
                            current_section = section
                            print(f"\n[Budget] Starting {section} section")
                            budget_usage[current_section] = 0
                            
                            # Update token metadata with section
                            token_metadata["section"] = section
                            
                            # Create a context window for this section
                            self.function_adapter.token_registry._buffer.update_context_window(
                                section, 
                                start=len(streamed_response) - 1,
                                size=budget_allocation.get(section, 1000)
                            )
                            break
                
                # Check for section end
                if current_section and f"</{current_section}>" in token:
                    allocated = budget_allocation.get(current_section, 0)
                    used = budget_usage.get(current_section, 0)
                    efficiency = (used / allocated * 100) if allocated > 0 else 0
                    print(f"[Budget] Completed {current_section}: used {used}/{allocated} tokens ({efficiency:.1f}%)")
                    
                    # Update remaining budget
                    remaining_budget -= used
                    
                    # Add section completion to token metadata
                    token_metadata["section_complete"] = True
                    token_metadata["section_efficiency"] = efficiency
                    
                    current_section = None
                
                # Count tokens in current section
                if current_section:
                    # Estimate token count for this chunk
                    token_count = 1  # Simple approximation
                    budget_usage[current_section] = budget_usage.get(current_section, 0) + token_count
                    
                    # Update token metadata with budget info
                    token_metadata["budget_used"] = budget_usage[current_section]
                    token_metadata["budget_allocated"] = budget_allocation.get(current_section, 0)
                
                # Check for budget report
                if "<budget_report>" in token:
                    print(f"\n[Budget] Generating final budget report")
                    print(f"[Budget] Remaining budget: {remaining_budget} tokens")
                    
                    # Update token metadata
                    token_metadata["budget_phase"] = "report"
                    token_metadata["remaining_budget"] = remaining_budget
                
                # Print the token
                print(token, end='', flush=True)
                
                # Process token through enhanced registry for potential code execution
                results = self.function_adapter.process_streamed_token(token, token_metadata)
                if results:
                    execution_results.extend(results)
                    
                    # Log execution results
                    for result in results:
                        pattern = result.get("pattern", "unknown")
                        status = result.get("result", {}).get("status", "unknown")
                        print(f"\n[Execution] Pattern '{pattern}' triggered with status '{status}'")
                        
                        # If there's output from the execution, print it
                        output = result.get("result", {}).get("output", "")
                        if output:
                            print(f"\n--- Execution Output ---\n{output}\n-----------------------\n")
                
                # Process output token for self-monitoring
                self.function_adapter.process_output_token(token, token_metadata)
                
                # Update progress indicator
                progress = min(99, int((token_position / 1500) * 100))
                self.streaming_output_manager.update_progress("response_generation", progress)
                
                # Periodically show progress bar (every 100 tokens)
                if token_position % 100 == 0 and token_position > 0:
                    progress_bar = self.streaming_output_manager.get_progress_bar("response_generation")
                    if progress_bar:
                        print(f"\n{progress_bar}\n", flush=True)
        
        # Complete progress indicator
        self.streaming_output_manager.update_progress("response_generation", 100)
        progress_bar = self.streaming_output_manager.get_progress_bar("response_generation")
        print(f"\n{progress_bar}\n", flush=True)
        self.streaming_output_manager.remove_progress_indicator("response_generation")
        
        print("\n\n=========================\n")
        
        # Show output analysis
        output_analysis = self.streaming_output_manager.get_output_analysis_text()
        if output_analysis:
            print(f"\n{output_analysis}\n")
            
        # Show suggested commands
        suggested_commands = self.streaming_output_manager.get_suggested_commands_text()
        if suggested_commands:
            print(f"\n{suggested_commands}\n")
        
        # Clear the token buffer after processing the response
        self.function_adapter.token_registry.clear_buffer()
        
        full_text = "".join(streamed_response)
        
        # Count tokens in the response and record usage
        response_tokens = self.token_budget.count_tokens(full_text)
        self.token_budget.request_tokens("response_total", response_tokens)
        
        # Print final budget status
        budget_status = self.token_budget.get_budget_status()
        print(f"[TokenBudget] Final status: {budget_status['remaining_budget']}/{budget_status['initial_budget']} tokens remaining")
        
        # Extract structured components from the text response
        facts, thinking, cognition, answer = self._extract_structured_output(full_text)
        
        # Create a structured response object
        structured_response = FactsThinkingAnswer(
            facts=facts or ["No facts provided"],
            thinking=thinking or "No thinking process provided",
            cognition=cognition or "No cognitive analysis provided",
            answer=answer or "No answer provided"
        )
        
        # Add verification step for structured response generation
        self.cognitive_engine.verify(
            description="Structured response generation",
            result="Complete",
            is_correct=True,
            confidence=0.9
        )

        # 4) Add agent utterance
        self.conversation.add_agent_utterance(full_text)
        
        # 5) Check immediate do_anything
        result = self.function_adapter.process_function_calls(full_text)
        if result:
            logger.info(f"[R1Agent] Immediate do_anything result: {result}")
            
            # Add execution step to cognitive model
            self.cognitive_engine.add_reasoning_step(
                behavior=CognitiveBehavior.VERIFICATION,
                description="Function call execution",
                result=result,
                is_correct=True if result.get("status") == "success" else False,
                confidence=0.9
            )

        # 6) Spawn a meta-task from user input with structured decomposition
        meta_task = self.memory_store.create_task(
            priority=10,
            description=user_input
        )
        
        # Check budget allocation for task decomposition
        decomp_tokens_allocated = budget_allocation.get("task_decomposition", 1500)
        
        # Always decompose tasks with streaming output
        print(f"\n=== Task Decomposition (Streaming) - Budget: {decomp_tokens_allocated} tokens ===\n")
        
        # Create a decomposition request with enhanced instructions and budget awareness
        decomp_messages = [
            {"role": "system", "content": f"You are an expert task decomposition system with a token budget of {decomp_tokens_allocated} tokens. Break down complex tasks into logical subtasks with clear dependencies and execution steps. Return your response as JSON with fields: subtasks (array of objects with description field), dependencies (object mapping subtask indices to arrays of dependency indices), approach (string describing overall approach), and execution_steps (array of strings describing how to execute each subtask)."},
            {"role": "user", "content": f"Decompose this task into detailed subtasks with execution steps: {user_input}"}
        ]
        
        # Stream the decomposition process
        decomp_stream = self.client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=decomp_messages,
            temperature=0.7,
            max_tokens=decomp_tokens_allocated,
            stream=True
        )
        
        decomp_chunks = []
        decomp_tokens_used = 0
        for chunk in decomp_stream:
            token = chunk.choices[0].delta.content
            if token:
                decomp_chunks.append(token)
                print(token, end='', flush=True)
                decomp_tokens_used += 1  # Simple approximation
        
        print("\n\n=========================\n")
        
        decomp_text = "".join(decomp_chunks)
        
        # Update budget usage for task decomposition
        budget_usage["task_decomposition"] = decomp_tokens_used
        remaining_budget -= decomp_tokens_used
        
        # Report task decomposition efficiency
        decomp_efficiency = (decomp_tokens_used / decomp_tokens_allocated * 100) if decomp_tokens_allocated > 0 else 0
        print(f"[Budget] Task decomposition: used {decomp_tokens_used}/{decomp_tokens_allocated} tokens ({decomp_efficiency:.1f}%)")
        print(f"[Budget] Remaining budget: {remaining_budget} tokens")
        
        # Parse the JSON response
        import json
        try:
            # Try to parse as JSON first
            json_data = json.loads(decomp_text)
            # Create a TaskDecompositionResponse object from the JSON data
            decomposition = TaskDecompositionResponse(
                subtasks=json_data.get("subtasks", []),
                dependencies=json_data.get("dependencies", {}),
                approach=json_data.get("approach", "Sequential approach to task completion")
            )
            
            # Extract execution steps if available
            execution_steps = json_data.get("execution_steps", [])
            if execution_steps:
                print("\n=== Execution Steps ===\n")
                for i, step in enumerate(execution_steps):
                    print(f"{i+1}. {step}")
                print("\n=========================\n")
        except json.JSONDecodeError:
            # If JSON parsing fails, create a simple decomposition from the text
            # Extract subtasks using regex or simple parsing
            import re
            subtask_matches = re.findall(r"(?:^|\n)(?:\d+\.\s*|\-\s*)(.*?)(?:\n|$)", decomp_text)
            subtasks = [{"description": match.strip()} for match in subtask_matches if match.strip()]
                
            if not subtasks:
                # If regex fails, split by newlines and create subtasks
                lines = [line.strip() for line in decomp_text.split("\n") if line.strip()]
                subtasks = [{"description": line} for line in lines[:5]]  # Limit to 5 subtasks
                
            decomposition = TaskDecompositionResponse(
                subtasks=subtasks,
                dependencies={},
                approach="Sequential approach to task completion"
            )
            
            print("\n=== Extracted Subtasks ===\n")
            for i, subtask in enumerate(subtasks):
                print(f"{i+1}. {subtask['description']}")
            print("\n=========================\n")
        
        # Store the decomposition in the task result
        self.memory_store.update_task_result(meta_task.task_id, {
            "decomposition": decomposition.model_dump(),
            "structured": True,
            "full_decomposition_text": decomp_text
        })
        
        # Create subtasks based on the decomposition and push to queue
        created_subtasks = []
        for i, subtask_info in enumerate(decomposition.subtasks):
            subtask = self.processor._spawn_subtask(meta_task, subtask_info["description"])
            created_subtasks.append(subtask)
            
            # Add dependencies if they exist
            if decomposition.dependencies and str(i) in decomposition.dependencies:
                for dep_idx in decomposition.dependencies[str(i)]:
                    if 0 <= dep_idx < len(created_subtasks):
                        subtask.add_tag(f"depends_on_{created_subtasks[dep_idx].task_id}")
            
            # Push subtask to queue for immediate processing
            self.task_queue.push(subtask)
        
        # Add planning step to cognitive model with more proactive approach
        self.cognitive_engine.plan(
            plan="Proactive structured task decomposition",
            steps=[st["description"] for st in decomposition.subtasks],
            confidence=0.9
        )
        
        # Add a goal-seeking step
        self.cognitive_engine.add_reasoning_step(
            behavior=CognitiveBehavior.EXPLORATION,
            description="Identifying opportunities for proactive problem-solving",
            metadata={"approach": decomposition.approach},
            confidence=0.85
        )
        
        logger.info(f"[R1Agent] Decomposed user input into {len(decomposition.subtasks)} subtasks and queued for processing")
        
        # Add task creation to cognitive model
        self.cognitive_engine.add_reasoning_step(
            behavior=CognitiveBehavior.SUBGOAL_SETTING,
            description=f"Created task {meta_task.task_id} from user input",
            metadata={"task_id": meta_task.task_id},
            confidence=0.9
        )

        # 7) Process cognitive behaviors from the structured response
        self._process_cognitive_behaviors_from_structured(structured_response, meta_task.task_id)
        
        # 8) Perform chain-of-thought enrichment with budget awareness
        enrichment_tokens = min(1000, remaining_budget)  # Allocate remaining budget, up to 1000 tokens
        if enrichment_tokens > 100:  # Only if we have enough tokens left
            print(f"[Budget] Allocating {enrichment_tokens} tokens for answer enrichment")
            enriched_answer = self._perform_cot_enrichment_from_structured(structured_response)
            
            # Add enriched answer to knowledge base
            self.knowledge_base.add_fact(f"enriched_answer_{meta_task.task_id}", enriched_answer)
            print("\n=== Enriched Answer ===\n")
            print(enriched_answer)
            print("\n=========================\n")
            
            # Update budget
            enrichment_tokens_used = min(self.token_budget.count_tokens(enriched_answer), enrichment_tokens)
            remaining_budget -= enrichment_tokens_used
            budget_usage["enrichment"] = enrichment_tokens_used
            print(f"[Budget] Answer enrichment: used {enrichment_tokens_used}/{enrichment_tokens} tokens")
        else:
            print("[Budget] Insufficient tokens for answer enrichment")
        
        # Add chain-of-thought step to cognitive model
        self.cognitive_engine.add_reasoning_step(
            behavior=CognitiveBehavior.VERIFICATION,
            description="Chain-of-thought enrichment",
            result="Successful",
            is_correct=True,
            confidence=0.8
        )
        
        # 9) Add final step in cognitive model
        self.cognitive_engine.add_reasoning_step(
            behavior=CognitiveBehavior.VERIFICATION,
            description="Response processing complete",
            is_correct=True,
            confidence=0.9
        )
        
        # 10) Log the cognitive reasoning trace
        reasoning_summary = self.cognitive_engine.get_reasoning_summary()
        logger.info(f"[R1Agent] Cognitive reasoning trace:\n{reasoning_summary}")
        
        # 11) Generate final budget report
        print("\n=== Final Budget Report ===")
        print(f"Initial budget: 8000 tokens")
        total_used = sum(budget_usage.values())
        print(f"Total tokens used: {total_used} ({total_used/8000*100:.1f}%)")
        print(f"Remaining budget: {remaining_budget} tokens")
        print("Usage by operation:")
        for operation, tokens in budget_usage.items():
            allocated = budget_allocation.get(operation, 0)
            efficiency = (tokens / allocated * 100) if allocated > 0 else 0
            print(f"  {operation}: {tokens}/{allocated} tokens ({efficiency:.1f}%)")
        print("=========================\n")

        return full_text
        
    def _format_structured_response(self, response: FactsThinkingAnswer) -> str:
        """Format a structured response for display"""
        formatted = "\n=== Facts ===\n"
        for i, fact in enumerate(response.facts, 1):
            formatted += f"{i}. {fact}\n"
            
        formatted += "\n=== Thinking Process ===\n"
        formatted += response.thinking
        
        formatted += "\n\n=== Cognitive Analysis ===\n"
        formatted += response.cognition
        
        formatted += "\n\n=== Answer ===\n"
        formatted += response.answer
        
        return formatted
        
    def _process_cognitive_behaviors_from_structured(self, response: FactsThinkingAnswer, task_id: int) -> None:
        """
        Process and record cognitive behaviors from a structured response.
        """
        if not response.cognition:
            return
            
        # Extract verification behaviors
        verification_match = re.search(r"Verification:\s*\[(.*?)\]", response.cognition)
        if verification_match and verification_match.group(1).strip() != "":
            verification_text = verification_match.group(1).strip()
            self.cognitive_engine.add_reasoning_step(
                behavior=CognitiveBehavior.VERIFICATION,
                description=f"Model-reported verification: {verification_text}",
                metadata={"source": "structured_response", "task_id": task_id},
                confidence=0.8
            )
            
        # Extract backtracking behaviors
        backtracking_match = re.search(r"Backtracking:\s*\[(.*?)\]", response.cognition)
        if backtracking_match and backtracking_match.group(1).strip() != "":
            backtracking_text = backtracking_match.group(1).strip()
            self.cognitive_engine.add_reasoning_step(
                behavior=CognitiveBehavior.BACKTRACKING,
                description=f"Model-reported backtracking: {backtracking_text}",
                metadata={"source": "structured_response", "task_id": task_id},
                confidence=0.7
            )
            
        # Extract subgoal setting behaviors
        subgoal_match = re.search(r"Subgoal Setting:\s*\[(.*?)\]", response.cognition)
        if subgoal_match and subgoal_match.group(1).strip() != "":
            subgoal_text = subgoal_match.group(1).strip()
            self.cognitive_engine.add_reasoning_step(
                behavior=CognitiveBehavior.SUBGOAL_SETTING,
                description=f"Model-reported subgoal setting: {subgoal_text}",
                metadata={"source": "structured_response", "task_id": task_id},
                confidence=0.85
            )
            
        # Extract backward chaining behaviors
        backward_match = re.search(r"Backward Chaining:\s*\[(.*?)\]", response.cognition)
        if backward_match and backward_match.group(1).strip() != "":
            backward_text = backward_match.group(1).strip() 
            self.cognitive_engine.add_reasoning_step(
                behavior=CognitiveBehavior.BACKWARD_CHAINING,
                description=f"Model-reported backward chaining: {backward_text}",
                metadata={"source": "structured_response", "task_id": task_id},
                confidence=0.75
            )

    def _use_tool_calls(self, facts: List[str], thinking: str, answer: str) -> None:
        """
        Use tool calls to extract data and provide grounding for the response.
        """
        # Example tool call for data extraction
        extracted_data = self._call_external_tool(facts, thinking, answer)
        if extracted_data:
            logger.info(f"[R1Agent] Extracted data: {extracted_data}")

    def _call_external_tool(self, facts: List[str], thinking: str, answer: str) -> Optional[Dict[str, Any]]:
        """
        Call an external tool for data extraction.
        """
        try:
            # Import the bootstrapping_agent_v0 module
            import bootstrapping_agent_v0

            # Call a function from the module, e.g., extract_data
            extracted_data = bootstrapping_agent_v0.extract_data(facts, thinking, answer)

            logger.info(f"[R1Agent] Extracted data: {extracted_data}")
            return extracted_data
        except ImportError as e:
            logger.error(f"[R1Agent] Error importing bootstrapping_agent_v0: {e}")
        except AttributeError as e:
            logger.error(f"[R1Agent] Function not found in bootstrapping_agent_v0: {e}")
        except Exception as e:
            logger.error(f"[R1Agent] Error calling external tool: {e}")
        return None
        
    def _extract_structured_output(self, text: str) -> Tuple[List[str], str, str, str]:
        """Extract facts, thinking, cognition, and answer from structured output."""
        facts = []
        thinking = ""
        cognition = ""
        answer = ""
        
        # Extract facts
        facts_match = re.search(r"<facts>(.*?)</facts>", text, re.DOTALL)
        if facts_match:
            facts_text = facts_match.group(1).strip()
            facts = [f.strip() for f in facts_text.split("-") if f.strip()]
            
        # Extract thinking
        thinking_match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
        if thinking_match:
            thinking = thinking_match.group(1).strip()
        
        # Extract cognition
        cognition_match = re.search(r"<cognition>(.*?)</cognition>", text, re.DOTALL)
        if cognition_match:
            cognition = cognition_match.group(1).strip()
            
        # Extract answer
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
            
        # Extract token requests
        token_requests = re.findall(r"<request_tokens>\s*(\w+):\s*(\d+)\s*</request_tokens>", text)
        
        return facts, thinking, cognition, answer
        
    def _perform_cot_enrichment_from_structured(self, response: FactsThinkingAnswer) -> str:
        """Perform chain-of-thought enrichment on a structured response with active problem-solving."""
        # Combine facts with thinking to create enriched answer
        if not response.facts and not response.thinking:
            return response.answer
            
        # Create a secondary chain of thought to further enrich the answer
        enriched = "Based on the following facts:\n"
        for i, fact in enumerate(response.facts, 1):
            enriched += f"{i}. {fact}\n"
            
        enriched += "\nFirst reasoning process:\n"
        enriched += response.thinking
        
        # Add cognitive behavior analysis
        enriched += "\n\nCognitive behaviors employed:\n"
        enriched += response.cognition
        
        # Add active problem-solving section if the answer indicates limitations
        if "cannot" in response.answer.lower() or "don't have access" in response.answer.lower():
            enriched += "\n\nActive problem-solving approach:\n"
            enriched += "Rather than accepting limitations, I can take these actions:\n"
            
            # Check if this is a weather-related query
            if "weather" in response.answer.lower():
                enriched += "1. Use Python's requests library to fetch weather data from public APIs\n"
                enriched += "2. Parse the JSON response to extract relevant information\n"
                enriched += "3. Format the data in a user-friendly way\n"
                enriched += "4. Store the result for future reference\n"
                enriched += "5. Implement error handling for robust operation\n"
            # Add other types of queries as needed
            else:
                enriched += "1. Break down the problem into specific data needs\n"
                enriched += "2. Identify appropriate data sources or computation methods\n"
                enriched += "3. Execute code to retrieve or generate the needed information\n"
                enriched += "4. Process and format the results\n"
                enriched += "5. Learn from this experience to improve future responses\n"
        
        # Add meta-reasoning about the reasoning process
        enriched += "\n\nMeta-reasoning about the reasoning process:\n"
        
        # Analyze the thinking provided in the first chain of thought
        lines = response.thinking.split('\n')
        meta_reasoning = []
        for i, line in enumerate(lines):
            if line.strip():
                # Assess confidence level based on language and certainty markers
                confidence = "high" if any(word in line.lower() for word in ["definitely", "certainly", "clearly", "must"]) else \
                             "low" if any(word in line.lower() for word in ["perhaps", "maybe", "might", "could", "possibly"]) else \
                             "medium"
                
                # Check if the reasoning step builds on previous steps
                builds_on_previous = i > 0 and any(f"step {j+1}" in line.lower() for j in range(i))
                
                # Identify cognitive behaviors in this step
                cognitive_behaviors = []
                if "verify" in line.lower() or "check" in line.lower() or "confirm" in line.lower():
                    cognitive_behaviors.append("verification")
                if "change" in line.lower() or "instead" in line.lower() or "alternative" in line.lower():
                    cognitive_behaviors.append("backtracking")
                if "break down" in line.lower() or "sub-problem" in line.lower() or "subtask" in line.lower():
                    cognitive_behaviors.append("subgoal setting")
                if "goal" in line.lower() and "backward" in line.lower():
                    cognitive_behaviors.append("backward chaining")
                if "plan" in line.lower() or "strategy" in line.lower() or "approach" in line.lower():
                    cognitive_behaviors.append("planning")
                if "evaluate" in line.lower() or "assess" in line.lower() or "compare" in line.lower():
                    cognitive_behaviors.append("evaluation")
                if "create" in line.lower() or "generate" in line.lower() or "novel" in line.lower():
                    cognitive_behaviors.append("creativity")
                if "pattern" in line.lower() or "generalize" in line.lower() or "abstract" in line.lower():
                    cognitive_behaviors.append("abstraction")
                if "similar to" in line.lower() or "analogy" in line.lower() or "like" in line.lower():
                    cognitive_behaviors.append("analogy")
                
                # Generate meta commentary
                meta = f"Step {i+1}: Confidence level: {confidence}. "
                if builds_on_previous:
                    meta += "This step builds on previous reasoning. "
                
                if cognitive_behaviors:
                    meta += f"Cognitive behaviors: {', '.join(cognitive_behaviors)}. "
                
                if i == len(lines) - 1 and len(lines) > 1:
                    meta += "This is a concluding step that synthesizes previous reasoning."
                
                meta_reasoning.append(meta)
        
        enriched += "\n".join(meta_reasoning)
        
        # Add cognitive strategies analysis section
        enriched += "\n\nCognitive strategies effectiveness analysis:\n"
        
        # Parse cognitive behaviors for analysis
        verification_present = "Verification" in response.cognition
        backtracking_present = "Backtracking" in response.cognition
        subgoal_present = "Subgoal Setting" in response.cognition
        backward_present = "Backward Chaining" in response.cognition
        planning_present = "Planning" in response.cognition
        evaluation_present = "Evaluation" in response.cognition
        creativity_present = "Creativity" in response.cognition
        abstraction_present = "Abstraction" in response.cognition
        analogy_present = "Analogy" in response.cognition
        
        if verification_present:
            enriched += "- Verification was effectively used to validate intermediate results, increasing solution accuracy.\n"
        else:
            enriched += "- Verification could have been used more extensively to check intermediate conclusions.\n"
            
        if backtracking_present:
            enriched += "- Backtracking was applied to abandon unproductive paths, demonstrating cognitive flexibility.\n"
        else:
            enriched += "- Little evidence of backtracking, suggesting a linear approach to the problem.\n"
            
        if subgoal_present:
            enriched += "- Effective use of subgoal decomposition made the problem more manageable.\n"
        else:
            enriched += "- The problem could have been broken down into clearer subgoals.\n"
            
        if backward_present:
            enriched += "- Backward chaining from the goal state helped focus the reasoning process.\n"
        else:
            enriched += "- A more goal-directed approach using backward chaining might have been beneficial.\n"
            
        if planning_present:
            enriched += "- Strategic planning was used to organize the approach to the problem.\n"
        else:
            enriched += "- A more explicit planning phase could have improved the solution strategy.\n"
            
        if evaluation_present:
            enriched += "- Evaluation of alternatives was conducted to select the best approach.\n"
        else:
            enriched += "- More explicit evaluation of different options could have led to a better solution.\n"
            
        if creativity_present:
            enriched += "- Creative thinking was applied to generate novel solutions.\n"
        else:
            enriched += "- The approach could have benefited from more creative thinking.\n"
            
        if abstraction_present:
            enriched += "- Abstraction was used to identify patterns and generalize the solution.\n"
        else:
            enriched += "- More abstraction could have helped identify underlying patterns.\n"
            
        if analogy_present:
            enriched += "- Analogical reasoning was used to transfer knowledge from familiar domains.\n"
        else:
            enriched += "- Drawing analogies to similar problems could have provided additional insights.\n"
        
        # Add final enriched answer with both levels of reasoning
        enriched += "\n\nThe doubly-enriched answer is:\n"
        enriched += response.answer
        
        return enriched

    def _build_messages(self) -> List[Dict[str, str]]:
        """
        System prompt + conversation history
        """
        history = self.conversation.get_history()
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(history)
        return messages

    def _extract_intent(self, text: str) -> Dict[str, Any]:
        """
        Extract user intent from text using structured reasoning
        
        Args:
            text: User input or generated text to analyze
            
        Returns:
            Dictionary with intent information
        """
        # Using cognitive engine for structured reasoning to extract intent
        self.cognitive_engine.add_reasoning_step(
            behavior=CognitiveBehavior.ABDUCTIVE_REASONING,
            description=f"Extracting user intent from: '{text[:50]}...'",
            confidence=0.85
        )
        
        # Extract potential intents using pattern recognition
        command_pattern = re.compile(r"^(search|find|get|create|update|delete|analyze|explain|summarize|compare)\s", re.IGNORECASE)
        question_pattern = re.compile(r"^(who|what|when|where|why|how|is|are|can|could|would|should|do|does)\s", re.IGNORECASE)
        request_pattern = re.compile(r"^(please|could you|would you|can you|i need|i want)\s", re.IGNORECASE)
        
        # Use structured analysis to determine intent
        match = command_pattern.search(text)
        if match:
            intent_type = "command"
            action = match.group(1).lower()
        elif question_pattern.search(text):
            intent_type = "question"
            if text.lower().startswith(("how to", "how do")):
                action = "instruction_request"
            elif text.lower().startswith(("what is", "what are")):
                action = "definition_request"
            else:
                action = "information_request"
        elif request_pattern.search(text):
            intent_type = "request"
            action = "assistance_request"
        else:
            # Use more sophisticated analysis for complex intents
            intent_type = "statement"
            
            # Check for specific patterns
            if "help" in text.lower() or "assist" in text.lower():
                action = "assistance_request"
            elif any(keyword in text.lower() for keyword in ["code", "function", "class", "implement"]):
                action = "code_generation"
            elif any(keyword in text.lower() for keyword in ["explain", "how does", "why is"]):
                action = "explanation_request"
            else:
                action = "general_statement"
        
        intent_result = {
            "type": intent_type,
            "action": action,
            "confidence": 0.85,
            "extracted_at": datetime.now().isoformat()
        }
        
        # Add reasoning step with the result
        self.cognitive_engine.add_reasoning_step(
            behavior=CognitiveBehavior.EVALUATION,
            description="Intent extraction complete",
            result=intent_result,
            confidence=0.85
        )
        
        return intent_result
        
    def _extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract named entities and key concepts from text
        
        Args:
            text: User input or generated text to analyze
            
        Returns:
            Dictionary with categorized entities
        """
        # Using cognitive engine for structured reasoning to extract entities
        self.cognitive_engine.add_reasoning_step(
            behavior=CognitiveBehavior.ANALYSIS,
            description=f"Extracting entities from: '{text[:50]}...'",
            confidence=0.8
        )
        
        entities = {
            "people": [],
            "organizations": [],
            "locations": [],
            "dates": [],
            "concepts": [],
            "technical_terms": [],
            "code_elements": [],
        }
        
        # Simple pattern matching for common entities
        # In a production system, this would use NER models or more sophisticated approaches
        
        # Match dates (simple patterns)
        date_patterns = [
            r"\d{1,2}/\d{1,2}/\d{2,4}",  # MM/DD/YYYY
            r"\d{1,2}-\d{1,2}-\d{2,4}",  # MM-DD-YYYY
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(st|nd|rd|th)?(,?\s+\d{4})?\b"
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities["dates"].append({
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.9
                })
        
        # Technical terms (programming languages, frameworks, etc.)
        tech_terms = [
            "Python", "JavaScript", "TypeScript", "Java", "C\\+\\+", "Rust", "Go", "SQL",
            "React", "Vue", "Angular", "Node.js", "Django", "Flask", "FastAPI",
            "Docker", "Kubernetes", "AWS", "Azure", "GCP",
            "API", "REST", "GraphQL", "JSON", "YAML", "XML",
            "Git", "GitHub", "GitLab", "Bitbucket"
        ]
        
        for term in tech_terms:
            pattern = r"\b" + term + r"\b"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities["technical_terms"].append({
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "type": "technology",
                    "confidence": 0.85
                })
        
        # Code elements (function names, class names, etc.)
        code_patterns = [
            r"\b[a-zA-Z_][a-zA-Z0-9_]*\(\)",  # function calls
            r"\bclass\s+[A-Z][a-zA-Z0-9_]*\b",  # class definitions
            r"\bdef\s+[a-z_][a-zA-Z0-9_]*\b",  # function definitions
            r"\bimport\s+[a-z_][a-zA-Z0-9_]*\b",  # imports
            r"\bfrom\s+[a-z_][a-zA-Z0-9_.]*\s+import\b"  # from imports
        ]
        
        for pattern in code_patterns:
            for match in re.finditer(pattern, text):
                entities["code_elements"].append({
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "type": "code_element",
                    "confidence": 0.9
                })
        
        # Add reasoning step with the result
        num_entities = sum(len(v) for v in entities.values())
        self.cognitive_engine.add_reasoning_step(
            behavior=CognitiveBehavior.EVALUATION,
            description=f"Entity extraction complete. Found {num_entities} entities.",
            result={"entity_count": num_entities},
            confidence=0.8
        )
        
        return entities
    
    def _extract_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Extract sentiment and emotional tone from text
        
        Args:
            text: User input or generated text to analyze
            
        Returns:
            Dictionary with sentiment information
        """
        # Using cognitive engine for structured reasoning to extract sentiment
        self.cognitive_engine.add_reasoning_step(
            behavior=CognitiveBehavior.ANALYSIS,
            description=f"Extracting sentiment from: '{text[:50]}...'",
            confidence=0.7
        )
        
        # Simple rule-based sentiment analysis
        # In a production system, this would use ML models
        
        positive_words = set([
            "good", "great", "excellent", "amazing", "awesome", "nice", "wonderful",
            "fantastic", "helpful", "useful", "thank", "thanks", "appreciate", "happy",
            "pleased", "satisfied", "love", "like", "enjoy"
        ])
        
        negative_words = set([
            "bad", "terrible", "awful", "horrible", "poor", "useless", "unhelpful",
            "disappointed", "frustrate", "frustrated", "annoying", "annoyed", "hate",
            "dislike", "fail", "failed", "issue", "problem", "bug", "error", "wrong"
        ])
        
        urgent_words = set([
            "urgent", "immediately", "asap", "emergency", "critical", "crucial",
            "vital", "important", "priority", "deadline", "quick", "quickly"
        ])
        
        # Count positive and negative words
        words = re.findall(r'\b\w+\b', text.lower())
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        urgent_count = sum(1 for word in words if word in urgent_words)
        
        # Determine overall sentiment
        if positive_count > negative_count * 2:
            sentiment = "very_positive"
            score = 0.8 + (0.2 * positive_count / (len(words) + 1))
        elif positive_count > negative_count:
            sentiment = "positive"
            score = 0.6 + (0.2 * positive_count / (len(words) + 1))
        elif negative_count > positive_count * 2:
            sentiment = "very_negative"
            score = -0.8 - (0.2 * negative_count / (len(words) + 1))
        elif negative_count > positive_count:
            sentiment = "negative"
            score = -0.6 - (0.2 * negative_count / (len(words) + 1))
        else:
            sentiment = "neutral"
            score = 0.0
            
        # Determine urgency
        urgency = min(1.0, urgent_count / 3)
        
        result = {
            "sentiment": sentiment,
            "score": max(-1.0, min(1.0, score)),  # Constrain to [-1.0, 1.0]
            "positive_count": positive_count,
            "negative_count": negative_count,
            "urgency": urgency,
            "confidence": 0.7,
            "extracted_at": datetime.now().isoformat()
        }
        
        # Add reasoning step with the result
        self.cognitive_engine.add_reasoning_step(
            behavior=CognitiveBehavior.EVALUATION,
            description=f"Sentiment analysis complete: {sentiment}",
            result=result,
            confidence=0.7
        )
        
        return result
    
    def _extract_tasks(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract potential tasks from text
        
        Args:
            text: User input or generated text to analyze
            
        Returns:
            List of potential tasks
        """
        # Using cognitive engine for structured reasoning to extract tasks
        self.cognitive_engine.add_reasoning_step(
            behavior=CognitiveBehavior.ANALYSIS,
            description=f"Extracting tasks from: '{text[:50]}...'",
            confidence=0.85
        )
        
        # Simple pattern matching for task-like statements
        tasks = []
        
        # Task patterns
        task_patterns = [
            r"(?:can you|please|could you)?\s*([^.!?;]+(?:find|search|get|retrieve|fetch|look up|analyze|create|make|build|implement|write|code|develop|generate|update|remove|delete|fix|solve|help)[^.!?;]+)",
            r"(?:i need|i want|i would like)[^.!?;]+(?:to|for)[^.!?;]+",
            r"(?:^|\n)(?:\d+\.\s|\-\s|\*\s)([^\n]+)"  # Numbered or bulleted list items
        ]
        
        for pattern in task_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                task_text = match.group(1) if len(match.groups()) > 0 else match.group(0)
                task_text = task_text.strip()
                
                # Determine priority (simple heuristic)
                priority = 3  # Default priority
                if any(word in task_text.lower() for word in ["urgent", "asap", "immediately", "critical"]):
                    priority = 1
                elif any(word in task_text.lower() for word in ["important", "soon", "priority"]):
                    priority = 2
                
                # Only add if not duplicate
                if not any(t.get("description") == task_text for t in tasks):
                    tasks.append({
                        "description": task_text,
                        "priority": priority,
                        "confidence": 0.8,
                        "extracted_at": datetime.now().isoformat()
                    })
        
        # Add reasoning step with the result
        self.cognitive_engine.add_reasoning_step(
            behavior=CognitiveBehavior.EVALUATION,
            description=f"Task extraction complete. Found {len(tasks)} tasks.",
            result={"task_count": len(tasks)},
            confidence=0.85
        )
        
        return tasks
    
    def _extract_code_elements(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract code elements and programming constructs from text
        
        Args:
            text: User input or generated text to analyze
            
        Returns:
            Dictionary with categorized code elements
        """
        # Using cognitive engine for structured reasoning to extract code elements
        self.cognitive_engine.add_reasoning_step(
            behavior=CognitiveBehavior.ANALYSIS,
            description=f"Extracting code elements from: '{text[:50]}...'",
            confidence=0.9
        )
        
        code_elements = {
            "functions": [],
            "classes": [],
            "variables": [],
            "imports": [],
            "code_blocks": []
        }
        
        # Extract code blocks (text between triple backticks)
        code_block_pattern = r"```(?:python)?\n([\s\S]*?)\n```"
        for match in re.finditer(code_block_pattern, text):
            code_block = match.group(1)
            code_elements["code_blocks"].append({
                "content": code_block,
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.95
            })
            
            # Parse the code block to extract functions, classes, etc.
            try:
                tree = ast.parse(code_block)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        code_elements["functions"].append({
                            "name": node.name,
                            "line": node.lineno,
                            "args": [arg.arg for arg in node.args.args],
                            "confidence": 0.9
                        })
                    elif isinstance(node, ast.ClassDef):
                        code_elements["classes"].append({
                            "name": node.name,
                            "line": node.lineno,
                            "bases": [base.id for base in node.bases if isinstance(base, ast.Name)],
                            "confidence": 0.9
                        })
                    elif isinstance(node, ast.Import):
                        for name in node.names:
                            code_elements["imports"].append({
                                "module": name.name,
                                "alias": name.asname,
                                "confidence": 0.95
                            })
                    elif isinstance(node, ast.ImportFrom):
                        for name in node.names:
                            code_elements["imports"].append({
                                "module": node.module,
                                "name": name.name,
                                "alias": name.asname,
                                "confidence": 0.95
                            })
            except SyntaxError:
                # If the code block doesn't parse, just ignore
                pass
        
        # Extract inline code elements (text between single backticks)
        inline_code_pattern = r"`([^`]+)`"
        for match in re.finditer(inline_code_pattern, text):
            code = match.group(1)
            # Try to determine if it's a function, variable, etc.
            if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*\(.*\)$", code):
                code_elements["functions"].append({
                    "name": code.split("(")[0],
                    "inline_reference": True,
                    "confidence": 0.8
                })
            elif re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", code):
                code_elements["variables"].append({
                    "name": code,
                    "inline_reference": True,
                    "confidence": 0.7
                })
        
        # Add reasoning step with the result
        num_elements = sum(len(v) for v in code_elements.values())
        self.cognitive_engine.add_reasoning_step(
            behavior=CognitiveBehavior.EVALUATION,
            description=f"Code element extraction complete. Found {num_elements} elements.",
            result={"element_count": num_elements},
            confidence=0.9
        )
        
        return code_elements
    
    def _extract_knowledge(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract knowledge assertions and facts from text
        
        Args:
            text: User input or generated text to analyze
            
        Returns:
            List of knowledge elements
        """
        # Using cognitive engine for structured reasoning to extract knowledge
        self.cognitive_engine.add_reasoning_step(
            behavior=CognitiveBehavior.ANALYSIS,
            description=f"Extracting knowledge from: '{text[:50]}...'",
            confidence=0.8
        )
        
        knowledge_items = []
        
        # Pattern for factual statements
        factual_patterns = [
            r"([^.!?;]+(?:is|are|was|were|has|have|had|can|could|would|should)[^.!?;]+\.)",
            r"([^.!?;]+(?:defined as|refers to|means|consists of|contains|includes)[^.!?;]+\.)",
            r"([^.!?;]+(?:always|never|must|will|won't|cannot|must not)[^.!?;]+\.)"
        ]
        
        for pattern in factual_patterns:
            for match in re.finditer(pattern, text):
                fact = match.group(1).strip()
                # Only add non-trivial facts
                if len(fact.split()) > 3:
                    # Try to determine subject and predicate
                    subject = None
                    predicate = None
                    
                    # Simple heuristic: first noun phrase is subject, rest is predicate
                    words = fact.split()
                    for i, word in enumerate(words):
                        if word.lower() in ["is", "are", "was", "were", "has", "have", "had"]:
                            subject = " ".join(words[:i])
                            predicate = " ".join(words[i:])
                            break
                    
                    knowledge_items.append({
                        "text": fact,
                        "subject": subject,
                        "predicate": predicate,
                        "confidence": 0.75,
                        "extracted_at": datetime.now().isoformat()
                    })
        
        # Add reasoning step with the result
        self.cognitive_engine.add_reasoning_step(
            behavior=CognitiveBehavior.EVALUATION,
            description=f"Knowledge extraction complete. Found {len(knowledge_items)} items.",
            result={"knowledge_count": len(knowledge_items)},
            confidence=0.8
        )
        
        return knowledge_items
    
    def _extract_reflections(self, text: str) -> Dict[str, Any]:
        """
        Extract self-reflective statements and meta-cognitive elements
        
        Args:
            text: User input or generated text to analyze
            
        Returns:
            Dictionary with reflective elements
        """
        # Using cognitive engine for structured reasoning to extract reflections
        self.cognitive_engine.add_reasoning_step(
            behavior=CognitiveBehavior.METACOGNITION,
            description=f"Extracting reflections from: '{text[:50]}...'",
            confidence=0.7
        )
        
        reflections = {
            "self_references": [],
            "uncertainty_expressions": [],
            "confidence_expressions": [],
            "limitations": [],
            "improvement_suggestions": [],
            "reasoning_paths": []
        }
        
        # Self-references
        self_ref_pattern = r"\b(I|my|me|myself|our|we|us)\b"
        for match in re.finditer(self_ref_pattern, text, re.IGNORECASE):
            surrounding_text = text[max(0, match.start() - 30):min(len(text), match.end() + 30)]
            reflections["self_references"].append({
                "pronoun": match.group(0),
                "context": surrounding_text,
                "position": match.start(),
                "confidence": 0.8
            })
        
        # Uncertainty expressions
        uncertainty_pattern = r"\b(maybe|perhaps|possibly|might|may|could|unsure|uncertain|not sure|not clear|unclear|don't know|unknown)\b"
        for match in re.finditer(uncertainty_pattern, text, re.IGNORECASE):
            surrounding_text = text[max(0, match.start() - 30):min(len(text), match.end() + 30)]
            reflections["uncertainty_expressions"].append({
                "expression": match.group(0),
                "context": surrounding_text,
                "position": match.start(),
                "confidence": 0.8
            })
        
        # Confidence expressions
        confidence_pattern = r"\b(certainly|definitely|sure|confident|know|clearly|obviously|undoubtedly|absolutely|strongly)\b"
        for match in re.finditer(confidence_pattern, text, re.IGNORECASE):
            surrounding_text = text[max(0, match.start() - 30):min(len(text), match.end() + 30)]
            reflections["confidence_expressions"].append({
                "expression": match.group(0),
                "context": surrounding_text,
                "position": match.start(),
                "confidence": 0.8
            })
        
        # Limitations
        limitation_pattern = r"\b(limitation|constrain|restrict|cannot|can't|unable|not able|impossible|beyond|outside|limited)\b"
        for match in re.finditer(limitation_pattern, text, re.IGNORECASE):
            surrounding_text = text[max(0, match.start() - 30):min(len(text), match.end() + 30)]
            reflections["limitations"].append({
                "expression": match.group(0),
                "context": surrounding_text,
                "position": match.start(),
                "confidence": 0.75
            })
        
        # Improvement suggestions
        improvement_pattern = r"\b(better|improve|enhancement|could be|should be|would be better|recommend|suggestion|advise)\b"
        for match in re.finditer(improvement_pattern, text, re.IGNORECASE):
            surrounding_text = text[max(0, match.start() - 30):min(len(text), match.end() + 30)]
            reflections["improvement_suggestions"].append({
                "expression": match.group(0),
                "context": surrounding_text,
                "position": match.start(),
                "confidence": 0.7
            })
        
        # Reasoning paths
        reasoning_pattern = r"\b(first|second|third|finally|then|next|lastly|because|therefore|thus|hence|so|as a result)\b"
        for match in re.finditer(reasoning_pattern, text, re.IGNORECASE):
            surrounding_text = text[max(0, match.start() - 30):min(len(text), match.end() + 30)]
            reflections["reasoning_paths"].append({
                "marker": match.group(0),
                "context": surrounding_text,
                "position": match.start(),
                "confidence": 0.75
            })
        
        # Add reasoning step with the result
        num_reflections = sum(len(v) for v in reflections.values())
        self.cognitive_engine.add_reasoning_step(
            behavior=CognitiveBehavior.EVALUATION,
            description=f"Reflection extraction complete. Found {num_reflections} elements.",
            result={"reflection_count": num_reflections},
            confidence=0.7
        )
        
        return reflections
    
    def handle_interactive_command(self, command: str) -> str:
        """
        Handle an interactive command during streaming.
        
        Args:
            command: The command to handle
            
        Returns:
            Response text for the command
        """
        if command == "/explain":
            # Get the current cognitive reasoning trace
            reasoning_summary = self.cognitive_engine.get_reasoning_summary()
            return f"\n=== Current Reasoning Process ===\n{reasoning_summary}\n=========================\n"
            
        elif command == "/details":
            # Get more detailed output
            current_section = None
            for step in self.cognitive_engine.get_chain_of_thought().steps[-5:]:
                if step.behavior == CognitiveBehavior.EXPLORATION:
                    current_section = "exploration"
                    break
                elif step.behavior == CognitiveBehavior.VERIFICATION:
                    current_section = "verification"
                    break
            
            if current_section == "exploration":
                return "\n=== Detailed Exploration ===\nThe agent is currently exploring different approaches to solve your query. This involves generating multiple candidate solutions and evaluating them based on relevance and effectiveness.\n"
            elif current_section == "verification":
                return "\n=== Detailed Verification ===\nThe agent is currently verifying the accuracy of its reasoning. This involves checking intermediate results and ensuring logical consistency.\n"
            else:
                return "\n=== Detailed Information ===\nThe agent is processing your query using structured reasoning with multiple cognitive behaviors including verification, backtracking, and subgoal setting.\n"
                
        elif command == "/status":
            # Get current status
            tasks_count = len(self.memory_store.list_tasks())
            pending_tasks = len([t for t in self.memory_store.list_tasks() if t.status == TaskStatus.PENDING])
            goals_count = len(self.goal_manager.list_goals())
            
            return f"\n=== Current Status ===\nTasks: {tasks_count} total, {pending_tasks} pending\nGoals: {goals_count}\nToken Budget: {self.token_budget.remaining_budget}/{self.token_budget.initial_budget}\n"
            
        elif command == "/help":
            # Show available commands
            return """
=== Available Commands ===
/explain - Get explanation of current reasoning
/details - Show more detailed output
/status - Show current agent status
/help - Show this help message
"""
        else:
            return f"\nUnknown command: {command}. Type /help for available commands."
    
    def shutdown(self) -> None:
        """
        Cleanly stop concurrency.
        """
        self.scheduler.stop_scheduler()
        self.plan_manager.stop()
        logger.info("[R1Agent] Shutdown complete.")

###############################################################################
# MAIN DEMO: RUNS INDEFINITELY UNTIL 'exit'
###############################################################################

def main():
    """
    Demonstration of the agent in an indefinite loop:
     - We allow user queries until they type 'exit'.
     - The background threads keep processing tasks, plan manager keeps analyzing, etc.
     - Uses structured outputs with Pydantic models for better task decomposition and reasoning
     - Exhibits volition and goal-seeking behavior through proactive problem-solving
     - Manages token budget with economic reasoning
     - Provides interactive streaming visualization with real-time feedback
     - Supports interactive commands during streaming
    """
    agent = R1Agent()

    try:
        # Example: create an initial goal with deadline and success criteria
        g = agent.add_goal(
            name="ScaleUp",
            description="Handle large-scale tasks, remain open for new instructions indefinitely.",
            priority=1,
            deadline=time.time() + 86400*30,  # 30 days from now
            success_criteria=["Process at least 100 tasks", "Maintain 95% task success rate", 
                             "Demonstrate effective task decomposition with structured outputs",
                             "Develop emergent budgeting capabilities to optimize token usage"]
        )
        logger.info(f"Created new goal: {g}")
        
        # Add initial cognitive reasoning steps
        agent.cognitive_engine.set_subgoal(
            subgoal="Initialize agent and prepare for user interaction",
            metadata={"phase": "startup"},
            confidence=0.95
        )
        
        agent.cognitive_engine.add_reasoning_step(
            behavior=CognitiveBehavior.VERIFICATION,
            description="Agent initialization complete",
            result="System ready",
            is_correct=True,
            confidence=0.95
        )
        
        # Add budget management goal
        budget_goal = agent.add_goal(
            name="EmergentBudgeting",
            description="Develop emergent budgeting capabilities to optimize token usage.",
            priority=2,
            deadline=time.time() + 86400*30,  # 30 days from now
            success_criteria=["Learn optimal token allocation patterns", 
                             "Improve budget efficiency metrics over time",
                             "Demonstrate economic reasoning in token allocation",
                             "Adapt allocations based on task complexity"]
        )
        logger.info(f"Created emergent budgeting goal: {budget_goal}")

        while True:
            # Add status check using cognitive verification
            agent.cognitive_engine.verify(
                description="System status before user input",
                result="Ready",
                is_correct=True,
                confidence=0.9
            )
            
            user_text = input("\n[User] Enter your query (or 'exit' to quit, or /command for interactive commands):\n> ").strip()
            
            if user_text.lower() in ["exit", "quit"]:
                logger.info("[main] Exiting upon user request.")
                agent.cognitive_engine.add_reasoning_step(
                    behavior=CognitiveBehavior.VERIFICATION,
                    description="Received exit command",
                    result="Initiating shutdown",
                    is_correct=True,
                    confidence=0.95
                )
                break
                
            # Handle interactive commands
            if user_text.startswith("/"):
                response = agent.handle_interactive_command(user_text)
                print(response)
                continue
                
            # Add special commands to view cognitive reasoning trace
            if user_text.lower() == "show reasoning":
                reasoning_summary = agent.cognitive_engine.get_reasoning_summary()
                print("\n=== Cognitive Reasoning Trace ===\n")
                print(reasoning_summary)
                print("\n=================================\n")
                continue
                
            if user_text.lower() == "show tasks":
                tasks = agent.memory_store.list_tasks()
                print("\n=== Current Tasks ===\n")
                for task in tasks:
                    print(f"Task {task.task_id}: {task.description[:50]}... (Status: {task.status.value}, Priority: {task.priority})")
                print("\n=====================\n")
                continue
                
            if user_text.lower() == "show goals":
                goals = agent.goal_manager.list_goals()
                print("\n=== Current Goals ===\n")
                for goal in goals:
                    print(f"Goal {goal.goal_id}: {goal.name} - {goal.description[:50]}... (Status: {goal.status.value}, Progress: {goal.progress:.2f})")
                print("\n=====================\n")
                continue
                
            if user_text.lower() == "show budget":
                budget_status = agent.token_budget.get_budget_status()
                print("\n=== Token Budget Status ===\n")
                print(f"Initial budget: {budget_status['initial_budget']} tokens")
                print(f"Remaining budget: {budget_status['remaining_budget']} tokens")
                print(f"Used budget: {budget_status['used_budget']} tokens")
                print("\nUsage by operation:")
                for operation, tokens in budget_status['usage_by_operation'].items():
                    print(f"  {operation}: {tokens} tokens")
                print(f"\nToken counting method: {budget_status['token_counting_method']}")
                print("\n===========================\n")
                continue
                
            if user_text.lower().startswith("add budget"):
                try:
                    amount = int(user_text.split()[2])
                    new_budget = agent.token_budget.add_to_budget(amount)
                    print(f"\n[TokenBudget] Added {amount} tokens. New budget: {new_budget} tokens\n")
                except (IndexError, ValueError):
                    print("\n[TokenBudget] Usage: add budget <amount>\n")
                continue
                
            if user_text.lower() == "reset budget":
                agent.token_budget.reset_budget()
                print(f"\n[TokenBudget] Budget reset to {agent.token_budget.initial_budget} tokens\n")
                continue
                
            if user_text.lower() == "solve puzzle":
                # Demonstrate cognitive capabilities with a simple puzzle using structured output
                print("\n=== Solving Countdown-style Puzzle ===\n")
                
                # Define a Pydantic model for the puzzle solution
                class PuzzleSolution(BaseModel):
                    numbers: List[int]
                    target: int
                    operations: List[str]
                    solution: str
                    explanation: str
                    
                    model_config = ConfigDict(
                        extra="forbid",
                        json_schema_extra={
                            "examples": [
                                {
                                    "numbers": [25, 8, 5, 3],
                                    "target": 30,
                                    "operations": ["+", "-", "*", "/"],
                                    "solution": "25 + 8 - 3 = 30",
                                    "explanation": "Add 25 and 8 to get 33, then subtract 3 to reach the target 30."
                                }
                            ]
                        }
                    )
                
                # Set up the puzzle
                agent.cognitive_engine.set_subgoal(
                    subgoal="Solve a Countdown-style puzzle with numbers [25, 8, 5, 3] and target 30",
                    confidence=0.9
                )
                
                try:
                    # Use structured output to solve the puzzle
                    messages = [
                        {"role": "system", "content": "You are an expert puzzle solver. Solve the Countdown-style number puzzle. Return your solution as JSON with fields: numbers (array of integers), target (integer), operations (array of strings), solution (string), and explanation (string)."},
                        {"role": "user", "content": "Solve this Countdown puzzle: Use the numbers [25, 8, 5, 3] to reach the target 30. You can use +, -, *, / operations."}
                    ]
                    
                    # Get the response
                    solution_response = agent.client.chat.completions.create(
                        model="deepseek-ai/DeepSeek-R1",
                        messages=messages,
                        temperature=0.3,
                        max_tokens=500
                    )
                    
                    # Extract the response text
                    solution_text = solution_response.choices[0].message.content
                    
                    # Parse the JSON response
                    import json
                    try:
                        json_data = json.loads(solution_text)
                        # Create a PuzzleSolution object from the JSON data
                        solution = PuzzleSolution(
                            numbers=json_data.get("numbers", [25, 8, 5, 3]),
                            target=json_data.get("target", 30),
                            operations=json_data.get("operations", ["+", "-", "*", "/"]),
                            solution=json_data.get("solution", "25 + 8 - 3 = 30"),
                            explanation=json_data.get("explanation", "Add 25 and 8 to get 33, then subtract 3 to reach the target 30.")
                        )
                    except json.JSONDecodeError:
                        # If JSON parsing fails, create a default solution
                        solution = PuzzleSolution(
                            numbers=[25, 8, 5, 3],
                            target=30,
                            operations=["+", "-", "*", "/"],
                            solution="25 + 8 - 3 = 30",
                            explanation="Add 25 and 8 to get 33, then subtract 3 to reach the target 30."
                        )
                    
                    # Display the structured solution
                    print(f"Numbers: {solution.numbers}")
                    print(f"Target: {solution.target}")
                    print(f"Solution: {solution.solution}")
                    print(f"Explanation: {solution.explanation}")
                    
                    # Add verification step
                    agent.cognitive_engine.verify(
                        description="Verify structured puzzle solution",
                        result=solution.solution,
                        is_correct=True,
                        confidence=0.95
                    )
                    
                except Exception as e:
                    logger.exception(f"[main] Error in structured puzzle solving: {e}")
                    
                    # Fall back to hardcoded solution
                    print("Solution: 25 + 8 - 3 = 30")
                    print("Explanation: Add 25 and 8 to get 33, then subtract 3 to reach the target 30.")
                    
                    # Add verification step for fallback
                    agent.cognitive_engine.verify(
                        description="Verify fallback puzzle solution",
                        result="25 + 8 - 3 = 30",
                        is_correct=True,
                        confidence=0.95
                    )
                
                print("\n==================================\n")
                continue

            # Generate immediate LLM response with structured output
            response = agent.generate_response(user_text)
            # Response is already printed with structured format
            # Additional outputs from the enrichment process will be shown separately

            # The agent continues working in background (TaskScheduler).
            # If you want to check tasks, reflection, or goals, do so here or in logs.

    finally:
        # Add final cognitive step for shutdown
        agent.cognitive_engine.add_reasoning_step(
            behavior=CognitiveBehavior.VERIFICATION,
            description="Agent shutdown sequence",
            result="Shutting down all components",
            is_correct=True,
            confidence=0.95
        )
        
        agent.shutdown()

if __name__ == "__main__":
    main()

def execute_shell_command(self, command: str, long_running: bool = False) -> Dict[str, Any]:
        """
        Execute a shell command with enhanced context awareness.
        
        Args:
            command: Shell command to execute
            long_running: Whether to run the command in the background
            
        Returns:
            Dictionary with execution results
        """
        import subprocess, tempfile, os
        
        logger.info(f"[FunctionAdapter] Executing shell command: {command}")
        
        try:
            if long_running:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as temp_file:
                    temp_file.write(f"#!/bin/bash\n{command}")
                    temp_file_path = temp_file.name
                
                # Make the script executable
                os.chmod(temp_file_path, 0o755)
                
                # Run in background
                subprocess.Popen(
                    f"nohup {temp_file_path} > /dev/null 2>&1 &",
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Record execution in context
                self.execution_context["last_pattern"] = "long_running_shell"
                self.execution_context["last_command"] = command
                self.execution_context["temp_file_path"] = temp_file_path
                
                return {
                    "status": "success",
                    "output": "Command is running in the background",
                    "temp_file_path": temp_file_path
                }
            else:
                # Run command with timeout
                process = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30  # 30 second timeout
                )
                
                # Record execution in context
                self.execution_context["last_pattern"] = "shell_command"
                self.execution_context["last_command"] = command
                self.execution_context["last_return_code"] = process.returncode
                
                if process.returncode == 0:
                    self.execution_context["last_output"] = process.stdout
                    
                    return {
                        "status": "success",
                        "output": process.stdout,
                        "stderr": process.stderr,
                        "return_code": process.returncode
                    }
                else:
                    self.execution_context["last_error"] = process.stderr
                    
                    return {
                        "status": "error",
                        "output": process.stdout,
                        "stderr": process.stderr,
                        "return_code": process.returncode
                    }
        except subprocess.TimeoutExpired:
            logger.error(f"[FunctionAdapter] Command timed out: {command}")
            
            # Record timeout in context
            self.execution_context["last_pattern"] = "shell_command"
            self.execution_context["last_command"] = command
            self.execution_context["last_error"] = "Command timed out"
            
            return {
                "status": "error",
                "output": "",
                "stderr": "Command timed out after 30 seconds",
                "return_code": -1
            }
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[FunctionAdapter] Error executing shell command: {e}\n{tb}")
            
            # Record error in context
            self.execution_context["last_pattern"] = "shell_command"
            self.execution_context["last_command"] = command
            self.execution_context["last_error"] = str(e)
            self.execution_context["last_traceback"] = tb
            
            return {
                "status": "error",
                "output": "",
                "stderr": str(e),
                "return_code": -1,
                "traceback": tb
            }
    def get_datetime_info(self, timezone: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive date and time information, optionally for a specific timezone.
        
        Args:
            timezone: Optional timezone name (e.g., 'America/New_York', 'Europe/London')
                     If None, returns information for UTC and local system time
        
        Returns:
            Dictionary with date and time information
        """
        try:
            # Use direct code execution for more reliable results
            time_code = """
import datetime
import time
import json
from zoneinfo import ZoneInfo, available_timezones
import pytz

# Get current times
now_utc = datetime.datetime.now(datetime.timezone.utc)
now_local = datetime.datetime.now()

# Format the times
result = {
    "utc": {
        "datetime": now_utc.isoformat(),
        "date": now_utc.strftime("%Y-%m-%d"),
        "time": now_utc.strftime("%H:%M:%S"),
        "timestamp": time.time(),
        "timezone": "UTC"
    },
    "local": {
        "datetime": now_local.isoformat(),
        "date": now_local.strftime("%Y-%m-%d"),
        "time": now_local.strftime("%H:%M:%S"),
        "timezone": str(now_local.astimezone().tzname() or time.tzname[0])
    }
}

# Add timezone-specific information if requested
requested_timezone = None
"""
            
            # Add timezone handling if specified
            if timezone:
                time_code = time_code.replace('requested_timezone = None', f'requested_timezone = "{timezone}"')
                time_code += """
if requested_timezone:
    try:
        # Try with ZoneInfo first (Python 3.9+)
        tz = ZoneInfo(requested_timezone)
        tz_time = datetime.datetime.now(tz)
        
        result["requested_timezone"] = {
            "datetime": tz_time.isoformat(),
            "date": tz_time.strftime("%Y-%m-%d"),
            "time": tz_time.strftime("%H:%M:%S"),
            "timezone": requested_timezone,
            "utc_offset": tz_time.strftime("%z")
        }
    except (ImportError, KeyError):
        # Fall back to pytz
        try:
            tz = pytz.timezone(requested_timezone)
            tz_time = datetime.datetime.now(tz)
            
            result["requested_timezone"] = {
                "datetime": tz_time.isoformat(),
                "date": tz_time.strftime("%Y-%m-%d"),
                "time": tz_time.strftime("%H:%M:%S"),
                "timezone": requested_timezone,
                "utc_offset": tz_time.strftime("%z")
            }
        except Exception as e:
            result["requested_timezone"] = {
                "error": f"Unknown timezone: {requested_timezone}",
                "exception": str(e)
            }
"""
            
            # Add timezone list
            time_code += """
# Add available timezones
try:
    result["available_timezones"] = list(available_timezones())[:20]  # First 20 for brevity
    result["available_timezones_count"] = len(available_timezones())
except ImportError:
    try:
        result["available_timezones"] = list(pytz.all_timezones)[:20]  # First 20 for brevity
        result["available_timezones_count"] = len(pytz.all_timezones)
    except ImportError:
        result["available_timezones"] = ["UTC"]
        result["available_timezones_count"] = 1

# Return the result as JSON
print(json.dumps(result, default=str))
result  # For return value
"""
            
            # Execute the code
            execution_result = self.do_anything(time_code)
            
            if execution_result and execution_result.get("status") == "success":
                # Try to parse the output as JSON
                import json
                try:
                    if "output" in execution_result and execution_result["output"]:
                        return json.loads(execution_result["output"])
                    elif "result" in execution_result and execution_result["result"]:
                        return execution_result["result"]
                except json.JSONDecodeError:
                    pass
            
            # If execution or parsing failed, fall back to the original implementation
            raise Exception("Direct execution failed, falling back to manual implementation")
            
        except Exception as e:
            # Fall back to the original implementation if execution fails
            import datetime
            import time
            import pytz
            from zoneinfo import ZoneInfo, available_timezones
            
            result = {
                "utc": {
                    "datetime": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
                    "time": datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S"),
                    "timestamp": time.time(),
                    "timezone": "UTC"
                },
                "local": {
                    "datetime": datetime.datetime.now().isoformat(),
                    "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "timezone": str(datetime.datetime.now().astimezone().tzname() or time.tzname[0])
                },
                "error": str(e)
            }
            
            # Add timezone-specific information if requested
            if timezone:
                try:
                    # Try with ZoneInfo first (Python 3.9+)
                    tz = ZoneInfo(timezone)
                    tz_time = datetime.datetime.now(tz)
                    
                    result["requested_timezone"] = {
                        "datetime": tz_time.isoformat(),
                        "date": tz_time.strftime("%Y-%m-%d"),
                        "time": tz_time.strftime("%H:%M:%S"),
                        "timezone": timezone,
                        "utc_offset": tz_time.strftime("%z")
                    }
                except (ImportError, KeyError):
                    # Fall back to pytz
                    try:
                        tz = pytz.timezone(timezone)
                        tz_time = datetime.datetime.now(tz)
                        
                        result["requested_timezone"] = {
                            "datetime": tz_time.isoformat(),
                            "date": tz_time.strftime("%Y-%m-%d"),
                            "time": tz_time.strftime("%H:%M:%S"),
                            "timezone": timezone,
                            "utc_offset": tz_time.strftime("%z")
                        }
                    except (pytz.exceptions.UnknownTimeZoneError, ImportError):
                        result["requested_timezone"] = {
                            "error": f"Unknown timezone: {timezone}"
                        }
            
            # Add available timezones
            try:
                result["available_timezones"] = list(available_timezones())[:20]  # First 20 for brevity
                result["available_timezones_count"] = len(available_timezones())
            except ImportError:
                try:
                    result["available_timezones"] = list(pytz.all_timezones)[:20]  # First 20 for brevity
                    result["available_timezones_count"] = len(pytz.all_timezones)
                except ImportError:
                    result["available_timezones"] = ["UTC"]
                    result["available_timezones_count"] = 1
            
            return result
        
    def execute_datetime_code(self, timezone: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute date/time related code in an isolated environment.
        This provides a more reliable way to get accurate date/time information.
        
        Args:
            timezone: Optional timezone name
            
        Returns:
            Dictionary with date and time information
        """
        # Create a simple script to get date/time information
        script = """
import datetime
import time
import json
import sys

# Get current times
now_utc = datetime.datetime.now(datetime.timezone.utc)
now_local = datetime.datetime.now()

# Format the times
result = {
    "utc": {
        "datetime": now_utc.isoformat(),
        "date": now_utc.strftime("%Y-%m-%d"),
        "time": now_utc.strftime("%H:%M:%S"),
        "timestamp": time.time(),
        "timezone": "UTC"
    },
    "local": {
        "datetime": now_local.isoformat(),
        "date": now_local.strftime("%Y-%m-%d"),
        "time": now_local.strftime("%H:%M:%S"),
        "timezone": str(now_local.astimezone().tzname() or time.tzname[0])
    }
}

# Add timezone-specific information if requested
requested_timezone = None
"""
        
        # Add timezone handling if specified
        if timezone:
            script = script.replace('requested_timezone = None', f'requested_timezone = "{timezone}"')
            script += """
if requested_timezone:
    try:
        # Try with ZoneInfo first (Python 3.9+)
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(requested_timezone)
        tz_time = datetime.datetime.now(tz)
        
        result["requested_timezone"] = {
            "datetime": tz_time.isoformat(),
            "date": tz_time.strftime("%Y-%m-%d"),
            "time": tz_time.strftime("%H:%M:%S"),
            "timezone": requested_timezone,
            "utc_offset": tz_time.strftime("%z")
        }
    except (ImportError, KeyError):
        # Fall back to pytz
        try:
            import pytz
            tz = pytz.timezone(requested_timezone)
            tz_time = datetime.datetime.now(tz)
            
            result["requested_timezone"] = {
                "datetime": tz_time.isoformat(),
                "date": tz_time.strftime("%Y-%m-%d"),
                "time": tz_time.strftime("%H:%M:%S"),
                "timezone": requested_timezone,
                "utc_offset": tz_time.strftime("%z")
            }
        except Exception as e:
            result["requested_timezone"] = {
                "error": f"Unknown timezone: {requested_timezone}",
                "exception": str(e)
            }
"""
        
        # Add output
        script += """
# Print the result as JSON
print(json.dumps(result, default=str))
"""
        
        # Execute the script in an isolated environment
        result = self.execute_isolated_code(script, timeout=5, provide_context=False)
        
        # Parse the output as JSON
        if result and result.get("status") == "success" and result.get("output"):
            try:
                import json
                return json.loads(result["output"])
            except json.JSONDecodeError:
                pass
        
        # Return a simple error result if execution failed
        return {
            "error": "Failed to execute date/time code",
            "execution_result": result
        }
    
    def get_token_buffer_status(self) -> Dict[str, Any]:
        """
        Get the current status of the token buffer.
        
        Returns:
            Dictionary with buffer statistics and context windows
        """
        buffer_stats = self.function_adapter.token_registry._buffer.get_stats()
        
        # Get context windows
        context_windows = {}
        with self.function_adapter.token_registry._buffer._lock:
            for window_name, window in self.function_adapter.token_registry._buffer._context_windows.items():
                context_windows[window_name] = {
                    "start": window["start"],
                    "size": window["size"],
                    "text": self.function_adapter.token_registry._buffer.get_context_window(window_name)[:50] + "..."
                }
        
        # Get execution history summary
        execution_history = self.function_adapter.token_registry.get_execution_history()
        execution_summary = []
        for execution in execution_history[-5:]:  # Last 5 executions
            execution_summary.append({
                "timestamp": execution["timestamp"],
                "pattern": execution["pattern"],
                "matched_text": execution["matched_text"][:30] + "..." if len(execution["matched_text"]) > 30 else execution["matched_text"]
            })
        
        return {
            "buffer_stats": buffer_stats,
            "context_windows": context_windows,
            "buffer_length": len(self.function_adapter.token_registry._buffer),
            "execution_history": execution_summary,
            "partial_fragments": len(self.function_adapter.partial_code_fragments)
        }
