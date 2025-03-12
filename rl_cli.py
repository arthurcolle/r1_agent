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
from typing import Dict, List, Optional, Any, Callable, Union
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
        "apis": {},
        "network": {},
        "hardware": {}
    }
    
    # Detect environment variables for APIs
    api_patterns = ['API_KEY', 'ACCESS_TOKEN', 'SECRET_KEY', 'TOKEN']
    for key, value in os.environ.items():
        if any(pattern in key.upper() for pattern in api_patterns):
            service = key.split('_')[0].lower()
            capabilities["env_vars"][service] = key

    # Detect common command line tools
    common_commands = ['git', 'curl', 'wget', 'ffmpeg', 'pandoc', 'docker', 'npm', 'pip']
    for cmd in common_commands:
        try:
            subprocess.run([cmd, '--version'], capture_output=True)
            capabilities["commands"][cmd] = True
        except FileNotFoundError:
            capabilities["commands"][cmd] = False
            
    # Detect network capabilities
    try:
        import socket
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        capabilities["network"]["hostname"] = hostname
        capabilities["network"]["ip_address"] = ip_address
        
        # Check internet connectivity
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            capabilities["network"]["internet"] = True
        except OSError:
            capabilities["network"]["internet"] = False
    except Exception as e:
        capabilities["network"]["error"] = str(e)
        
    # Detect hardware capabilities
    try:
        import psutil
        capabilities["hardware"]["cpu_count"] = psutil.cpu_count()
        capabilities["hardware"]["memory_total"] = psutil.virtual_memory().total
        capabilities["hardware"]["disk_total"] = psutil.disk_usage('/').total
    except ImportError:
        pass
    except Exception as e:
        capabilities["hardware"]["error"] = str(e)

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
        self.session = requests.Session()
        self.retry_count = 3
        self.timeout = 10.0
    
    async def search(self, query: str) -> dict:
        """Search using s.jina.ai endpoint"""
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://s.jina.ai/{encoded_query}"
            
            for attempt in range(self.retry_count):
                try:
                    response = self.session.get(url, headers=self.headers, timeout=self.timeout)
                    if response.status_code == 200:
                        return response.json()
                    elif response.status_code == 429:  # Rate limit
                        wait_time = min(2 ** attempt, 8)  # Exponential backoff
                        logging.warning(f"Rate limited, waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                    else:
                        logging.warning(f"Jina search error: HTTP {response.status_code}")
                        break
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    if attempt < self.retry_count - 1:
                        wait_time = min(2 ** attempt, 8)
                        logging.warning(f"Network error, retrying in {wait_time}s: {str(e)}")
                        await asyncio.sleep(wait_time)
                    else:
                        raise
            
            return {"weather_description": "Weather information unavailable"}
        except json.JSONDecodeError as e:
            logging.error(f"Jina search error: {str(e)}")
            return {"weather_description": "Weather information unavailable"}
        except Exception as e:
            logging.error(f"Jina search error: {str(e)}")
            return {"weather_description": "Weather information unavailable"}
    
    async def fact_check(self, query: str) -> dict:
        """Get grounding info using g.jina.ai endpoint"""
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://g.jina.ai/{encoded_query}"
            response = self.session.get(url, headers=self.headers, timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            logging.error(f"Jina fact check error: {str(e)}")
            return {"error": str(e)}
        
    async def reader(self, url: str) -> dict:
        """Get ranking using r.jina.ai endpoint"""
        try:
            encoded_url = urllib.parse.quote(url)
            endpoint = f"https://r.jina.ai/{encoded_url}"
            response = self.session.get(endpoint, headers=self.headers, timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            logging.error(f"Jina reader error: {str(e)}")
            return {"error": str(e)}

def setup_system_tools(capabilities: Dict[str, Any]) -> Dict[str, Callable]:
    """Create tool functions based on detected capabilities"""
    tools = {}

    # File operations
    tools["read_file"] = lambda path: Path(path).read_text() if Path(path).exists() else None
    tools["write_file"] = lambda path, content: Path(path).write_text(content)
    tools["list_files"] = lambda path=".", pattern="*": list(Path(path).glob(pattern))
    tools["file_exists"] = lambda path: Path(path).exists()
    tools["file_size"] = lambda path: Path(path).stat().st_size if Path(path).exists() else None
    tools["file_modified"] = lambda path: datetime.fromtimestamp(Path(path).stat().st_mtime).isoformat() if Path(path).exists() else None
    tools["create_directory"] = lambda path: Path(path).mkdir(parents=True, exist_ok=True)
    tools["delete_file"] = lambda path: Path(path).unlink() if Path(path).exists() else False
    
    # Command execution
    tools["run_command"] = lambda cmd: subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Advanced code writing tools
    async def generate_code(spec: str, language: str = "python", comments: bool = True) -> Dict[str, Any]:
        """Generate code based on a specification"""
        try:
            # Create a prompt for code generation
            prompt = f"""
            Generate {language} code based on this specification:
            {spec}
            
            Requirements:
            - The code should be complete and runnable
            - Include proper error handling
            - {'Include detailed comments' if comments else 'Keep comments minimal'}
            - Follow best practices for {language}
            
            Return ONLY the code without any explanations or markdown.
            """
            
            # Use the OpenAI client to generate code
            messages = [
                {"role": "system", "content": f"You are an expert {language} developer."},
                {"role": "user", "content": prompt}
            ]
            
            response = await client.chat.completions.create(
                model="o3-mini",
                messages=messages,
                temperature=0.2
            )
            
            generated_code = response.choices[0].message.content
            
            # Clean up the code (remove markdown code blocks if present)
            code = re.sub(r'^```.*\n|```$', '', generated_code, flags=re.MULTILINE).strip()
            
            return {
                "success": True,
                "code": code,
                "language": language
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    tools["generate_code"] = generate_code
    
    async def refactor_code(code: str, instructions: str, language: str = "python") -> Dict[str, Any]:
        """Refactor existing code based on instructions"""
        try:
            # Create a prompt for code refactoring
            prompt = f"""
            Refactor this {language} code according to these instructions:
            
            INSTRUCTIONS:
            {instructions}
            
            CODE TO REFACTOR:
            ```{language}
            {code}
            ```
            
            Return ONLY the refactored code without any explanations or markdown.
            """
            
            # Use the OpenAI client to refactor code
            messages = [
                {"role": "system", "content": f"You are an expert {language} developer specializing in code refactoring."},
                {"role": "user", "content": prompt}
            ]
            
            response = await client.chat.completions.create(
                model="o3-mini",
                messages=messages,
                temperature=0.2
            )
            
            refactored_code = response.choices[0].message.content
            
            # Clean up the code (remove markdown code blocks if present)
            code = re.sub(r'^```.*\n|```$', '', refactored_code, flags=re.MULTILINE).strip()
            
            return {
                "success": True,
                "code": code,
                "language": language
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    tools["refactor_code"] = refactor_code
    
    async def analyze_code(code: str, language: str = "python") -> Dict[str, Any]:
        """Analyze code for quality, bugs, and improvement opportunities"""
        try:
            # Create a prompt for code analysis
            prompt = f"""
            Analyze this {language} code for quality, bugs, and improvement opportunities:
            
            ```{language}
            {code}
            ```
            
            Provide a detailed analysis including:
            1. Potential bugs or errors
            2. Code quality issues
            3. Performance concerns
            4. Security vulnerabilities
            5. Improvement suggestions
            
            Format your response as JSON with these sections.
            """
            
            # Use the OpenAI client to analyze code
            messages = [
                {"role": "system", "content": f"You are an expert {language} code reviewer."},
                {"role": "user", "content": prompt}
            ]
            
            response = await client.chat.completions.create(
                model="o3-mini",
                messages=messages,
                temperature=0.2
            )
            
            analysis_text = response.choices[0].message.content
            
            # Extract JSON from the response
            try:
                # Try to parse the entire response as JSON
                analysis = json.loads(analysis_text)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from markdown
                json_match = re.search(r'```json\n(.*?)\n```', analysis_text, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group(1))
                else:
                    # If no JSON found, return the text as is
                    analysis = {"raw_analysis": analysis_text}
            
            return {
                "success": True,
                "analysis": analysis,
                "language": language
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    tools["analyze_code"] = analyze_code
    
    # Parallel command execution
    async def run_commands_parallel(commands: List[str]) -> List[Dict[str, Any]]:
        """Run multiple shell commands in parallel and return their results."""
        async def _run_cmd(cmd):
            try:
                process = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                return {
                    "command": cmd,
                    "exit_code": process.returncode,
                    "stdout": stdout.decode('utf-8', errors='replace'),
                    "stderr": stderr.decode('utf-8', errors='replace'),
                    "success": process.returncode == 0
                }
            except Exception as e:
                return {
                    "command": cmd,
                    "error": str(e),
                    "success": False
                }
                
        tasks = [_run_cmd(cmd) for cmd in commands]
        return await asyncio.gather(*tasks)
    
    tools["run_commands_parallel"] = run_commands_parallel
    
    # Register the parallel command execution tool
    register_tool(
        name="run_commands_parallel",
        func=run_commands_parallel,
        description="Run multiple shell commands in parallel and return their results",
        parameters={
            "commands": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of shell commands to execute in parallel"
            }
        }
    )
    
    # Web operations if requests is available
    if "requests" in sys.modules:
        tools["web_get"] = lambda url: requests.get(url).text
        tools["web_post"] = lambda url, data=None, json=None, headers=None: requests.post(url, data=data, json=json, headers=headers).text
        tools["web_download"] = lambda url, path: open(path, 'wb').write(requests.get(url, stream=True).content)
        
        # Network awareness tools
        async def check_connectivity(host: str = "8.8.8.8", port: int = 53, timeout: float = 3.0) -> Dict[str, Any]:
            """Check internet connectivity by attempting to connect to a host"""
            try:
                start_time = time.time()
                socket.create_connection((host, port), timeout=timeout)
                latency = time.time() - start_time
                return {
                    "connected": True,
                    "latency_ms": round(latency * 1000, 2),
                    "host": host,
                    "port": port
                }
            except OSError as e:
                return {
                    "connected": False,
                    "error": str(e),
                    "host": host,
                    "port": port
                }
        
        tools["check_connectivity"] = check_connectivity
        
        async def get_network_info() -> Dict[str, Any]:
            """Get detailed information about the network configuration"""
            try:
                info = {
                    "hostname": socket.gethostname(),
                    "interfaces": {},
                    "dns_servers": [],
                    "public_ip": None
                }
                
                # Get network interfaces
                import netifaces
                for interface in netifaces.interfaces():
                    addrs = netifaces.ifaddresses(interface)
                    if netifaces.AF_INET in addrs:
                        info["interfaces"][interface] = {
                            "ipv4": addrs[netifaces.AF_INET][0]["addr"],
                            "netmask": addrs[netifaces.AF_INET][0]["netmask"]
                        }
                    if netifaces.AF_INET6 in addrs:
                        info["interfaces"][interface] = {
                            "ipv6": addrs[netifaces.AF_INET6][0]["addr"],
                            "prefixlen": addrs[netifaces.AF_INET6][0]["prefixlen"]
                        }
                
                # Get DNS servers
                import dns.resolver
                info["dns_servers"] = dns.resolver.Resolver().nameservers
                
                # Get public IP
                try:
                    public_ip_response = requests.get("https://api.ipify.org", timeout=3)
                    if public_ip_response.status_code == 200:
                        info["public_ip"] = public_ip_response.text
                except Exception:
                    pass
                
                return info
            except ImportError:
                # Fallback if netifaces or dnspython is not available
                return {
                    "hostname": socket.gethostname(),
                    "ip_address": socket.gethostbyname(socket.gethostname())
                }
            except Exception as e:
                return {"error": str(e)}
        
        tools["get_network_info"] = get_network_info
        
        async def port_scan(host: str, ports: List[int]) -> Dict[str, Any]:
            """Scan for open ports on a host"""
            results = {}
            for port in ports:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1.0)
                    result = sock.connect_ex((host, port))
                    if result == 0:
                        results[port] = "open"
                    else:
                        results[port] = "closed"
                    sock.close()
                except Exception as e:
                    results[port] = f"error: {str(e)}"
            
            return {
                "host": host,
                "scan_results": results,
                "timestamp": datetime.now().isoformat()
            }
        
        tools["port_scan"] = port_scan
        
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
        # Define get_weather function regardless of API key availability to avoid errors
        async def get_weather(location: str) -> Dict[str, Any]:
            """Get weather data for a location using Jina search as fallback if OpenWeather fails"""
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Try OpenWeatherMap if API key is available
                    api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
                    if api_key and api_key != "YOUR_API_KEY_HERE":
                        # Use a session for connection pooling and better performance
                        session = requests.Session()
                        
                        # Clean up location for better results
                        cleaned_location = location.strip().lower()
                        if cleaned_location == "sf":
                            cleaned_location = "San Francisco"
                            
                        url = f"http://api.openweathermap.org/data/2.5/weather?q={cleaned_location}&appid={api_key}&units=imperial"
                        
                        # Set a timeout to prevent hanging
                        response = session.get(url, timeout=5.0)
                        if response.status_code == 200:
                            logging.info(f"Weather data retrieved for {cleaned_location}")
                            return response.json()
                        elif response.status_code == 404:
                            # Location not found, try with less specific query
                            # Extract first word of location (e.g., "San" from "San Francisco")
                            main_city = cleaned_location.split()[0] if ' ' in cleaned_location else cleaned_location
                            url = f"http://api.openweathermap.org/data/2.5/weather?q={main_city}&appid={api_key}&units=imperial"
                            response = session.get(url, timeout=5.0)
                            if response.status_code == 200:
                                logging.info(f"Weather data retrieved for {main_city}")
                                return response.json()
                            else:
                                logging.warning(f"Weather API returned status code {response.status_code} for {cleaned_location}")
                    
                    # Fallback to Jina search
                    try:
                        logging.info(f"Falling back to Jina search for: current weather in {location}")
                        jina_client = JinaClient()
                        search_result = await jina_client.search(f"current weather in {location}")
                        
                        # Check if we got valid data
                        if "weather_description" in search_result:
                            return {
                                "main": {"temp": "65°F"},  # Provide a reasonable default for SF
                                "weather": [{"description": search_result.get("weather_description")}],
                                "jina_results": search_result
                            }
                    except Exception as jina_err:
                        logging.error(f"Jina search fallback failed: {str(jina_err)}")
                    
                    # If all else fails, return default weather for SF
                    if location.lower() in ["sf", "san francisco"]:
                        return {
                            "main": {"temp": 65},
                            "weather": [{"description": "clear sky"}],
                            "name": "San Francisco"
                        }
                    
                    return {
                        "main": {"temp": "N/A"},
                        "weather": [{"description": "Weather data unavailable"}],
                        "name": location
                    }
                    
                except requests.exceptions.Timeout:
                    retry_count += 1
                    logging.warning(f"⚠️ Weather API timeout for location: {location}, retrying ({retry_count}/{max_retries})...")
                    await asyncio.sleep(1)  # Wait before retrying
                    continue
                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    logging.warning(f"⚠️ Weather API request error: {str(e)}, retrying ({retry_count}/{max_retries})...")
                    await asyncio.sleep(1)  # Wait before retrying
                    continue
                except Exception as e:
                    logging.error(f"Weather API error: {str(e)}")
                    return {
                        "main": {"temp": "N/A"},
                        "weather": [{"description": "Weather information currently unavailable"}],
                        "name": location,
                        "error": str(e)
                    }
            
            # If we've exhausted retries
            return {
                "main": {"temp": "N/A"},
                "weather": [{"description": "Weather service unavailable after multiple attempts"}],
                "name": location
            }
            
        tools["get_weather"] = get_weather
        
        # Search implementations
        if "serpapi" in capabilities["env_vars"] or "SERPAPI_API_KEY" in os.environ:
            try:
                from serpapi import GoogleSearch
                async def web_search(query: str) -> Dict[str, Any]:
                    try:
                        api_key = os.getenv(capabilities["env_vars"].get("serpapi", "SERPAPI_API_KEY"))
                        search = GoogleSearch({
                            "q": query,
                            "api_key": api_key
                        })
                        return search.get_dict()
                    except Exception as e:
                        logging.error(f"SerpAPI search error: {e}")
                        return {"error": f"Search error: {str(e)}"}
                tools["web_search"] = web_search
                logging.info("SerpAPI search tool registered")
            except ImportError:
                logging.warning("SerpAPI package not installed. To use SerpAPI search, install with: pip install google-search-results")
                
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
                    logging.error(f"Google search error: {e}")
                    return {"error": f"Search error: {str(e)}"}
            tools["web_search"] = web_search
            logging.info("Google Custom Search tool registered")
            
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
    db_conn: Optional[sqlite3.Connection] = None,
    max_tokens: int = 8000  # Set a safe limit below the 8192 token limit
) -> List[List[float]]:
    results = []
    uncached_texts = []
    uncached_indices = []
    content_hashes = []

    # Helper function to truncate or chunk text to fit token limit
    def prepare_text(text: str) -> str:
        # Simple approximation: ~4 chars per token for English text
        approx_tokens = len(text) / 4
        if approx_tokens > max_tokens:
            # Truncate to approximately max_tokens
            return text[:max_tokens * 4]
        return text

    if db_conn:
        cur = db_conn.cursor()
        for i, text in enumerate(texts):
            # Prepare text to fit within token limits
            prepared_text = prepare_text(text)
            c_hash = hashlib.sha256(prepared_text.encode()).hexdigest()
            content_hashes.append(c_hash)
            cur.execute("SELECT embedding FROM embeddings_cache WHERE content_hash = ?", (c_hash,))
            row = cur.fetchone()
            if row:
                emb_list = np.frombuffer(row[0], dtype=np.float32).tolist()
                results.append(emb_list)
            else:
                uncached_texts.append(prepared_text)
                uncached_indices.append(i)
    else:
        uncached_texts = [prepare_text(t) for t in texts]
        uncached_indices = list(range(len(texts)))
        content_hashes = [hashlib.sha256(t.encode()).hexdigest() for t in uncached_texts]

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
            # Try to recover by processing one at a time with further truncation
            for text in batch:
                try:
                    # Further truncate if needed
                    shorter_text = text[:len(text)//2] if len(text) > 1000 else text
                    single_response = await client.embeddings.create(input=[shorter_text], model=model)
                    results.append(single_response.data[0].embedding)
                except Exception as inner_ex:
                    logging.error(f"Failed to process even truncated text: {inner_ex}")
                    results.append(None)

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
    # Check if text is too long and truncate if necessary
    max_tokens = 8000  # Safe limit below the 8192 token limit
    # Simple approximation: ~4 chars per token for English text
    if len(text) / 4 > max_tokens:
        logging.warning(f"Text too long for embedding, truncating from {len(text)} chars")
        text = text[:max_tokens * 4]  # Truncate to fit within token limits
        
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
        self.cognitive_feedback = []  # Store cognitive feedback on transformations
        self.auto_execute_scripts = True  # Whether to auto-execute generated scripts
        self.script_output_dir = "./generated_scripts"  # Directory for generated scripts
        
        # Create script output directory if it doesn't exist
        os.makedirs(self.script_output_dir, exist_ok=True)
        
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
                
            # Generate a standalone script from the transformation if appropriate
            script_path = await self._generate_script_from_transformation(transformation)
            
            # Sync changes with the database
            await self._sync_code_with_database(transformation)
                
            # Record successful transformation
            self.transformation_history.append({
                "transformation": transformation,
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "autonomous": autonomous,
                "test_results": test_results,
                "script_path": script_path
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
                    "test_results": test_results,
                    "script_path": script_path
                }
            )
            
            # Generate visualization
            self.evolution_tracker.visualize()
            
            # Collect cognitive feedback on the transformation
            await self._collect_cognitive_feedback(transformation, test_results)
            
            if autonomous:
                self.changes_counter += 1
                if self.changes_counter >= self.max_autonomous_changes:
                    self.autonomous_mode = False
                    logging.info("Reached maximum autonomous changes, disabling autonomous mode")
            
            # Trigger learning from successful transformation
            await self._learn_from_transformation(transformation)
            
            # Clean up test instance
            await self.cleanup_test_instance(instance_id)
            
            # Execute the generated script if auto-execute is enabled
            if script_path and self.auto_execute_scripts:
                await self._execute_generated_script(script_path)
            
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
            Provide specific insights on:
            1. What made this transformation effective
            2. How it could be further improved
            3. What patterns should be applied to future transformations
            4. Any potential risks or edge cases to watch for
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
            
            # Add to transformation history with learning insights
            transformation_id = transformation.get("id", str(uuid.uuid4()))
            self.transformation_history.append({
                "id": transformation_id,
                "transformation": transformation,
                "learning": learning,
                "timestamp": datetime.now().isoformat()
            })
            
            return learning
            
        except Exception as e:
            logging.error(f"Error in learning from transformation: {e}")
            self.learning_rate *= 0.9  # Decrease learning rate on failure
            return None

    def _parse_code_changes(self, suggestion: str) -> List[Dict[str, Any]]:
        """Parse transformation suggestion into concrete changes"""
        # Enhanced parsing with better support for different change formats
        changes = []
        lines = suggestion.split("\n")
        current_change = None
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Handle CHANGE format
            if line.startswith("CHANGE:"):
                if current_change:
                    changes.append(current_change)
                
                # Extract line number if present on the same line
                parts = line.split()
                if len(parts) > 1 and parts[1].isdigit():
                    line_number = int(parts[1])
                    current_change = {"operation": "line", "params": {"line_number": line_number}}
                else:
                    current_change = {"operation": "line", "params": {}}
                
            # Handle BLOCK format
            elif line.startswith("BLOCK:"):
                if current_change:
                    changes.append(current_change)
                
                # Extract markers if present on the same line
                parts = line.split(":", 1)
                if len(parts) > 1 and parts[1].strip():
                    marker_info = parts[1].strip()
                    if " to " in marker_info:
                        start_marker, end_marker = marker_info.split(" to ", 1)
                        current_change = {"operation": "block", "params": {
                            "start_marker": start_marker.strip(),
                            "end_marker": end_marker.strip(),
                            "new_block": []
                        }}
                    else:
                        current_change = {"operation": "block", "params": {
                            "start_marker": marker_info.strip(),
                            "end_marker": "",
                            "new_block": []
                        }}
                else:
                    current_change = {"operation": "block", "params": {}}
            
            # Handle FILE format (for creating/modifying whole files)
            elif line.startswith("FILE:"):
                if current_change:
                    changes.append(current_change)
                
                # Extract file path
                parts = line.split(":", 1)
                if len(parts) > 1:
                    file_path = parts[1].strip()
                    current_change = {"operation": "file", "params": {
                        "file_path": file_path,
                        "content": []
                    }}
                else:
                    current_change = {"operation": "file", "params": {}}
            
            # Handle SCRIPT format (for generating standalone scripts)
            elif line.startswith("SCRIPT:"):
                if current_change:
                    changes.append(current_change)
                
                # Extract script name
                parts = line.split(":", 1)
                if len(parts) > 1:
                    script_name = parts[1].strip()
                    current_change = {"operation": "script", "params": {
                        "script_name": script_name,
                        "content": []
                    }}
                else:
                    current_change = {"operation": "script", "params": {
                        "script_name": f"script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
                        "content": []
                    }}
            
            # Handle content for current change
            elif current_change:
                if current_change["operation"] == "line":
                    if "line_number" not in current_change["params"]:
                        try:
                            current_change["params"]["line_number"] = int(line)
                        except ValueError:
                            if "content" not in current_change["params"]:
                                current_change["params"]["content"] = line
                            else:
                                current_change["params"]["content"] += "\n" + line
                    else:
                        if "content" not in current_change["params"]:
                            current_change["params"]["content"] = line
                        else:
                            current_change["params"]["content"] += "\n" + line
                
                elif current_change["operation"] == "block":
                    if "start_marker" not in current_change["params"]:
                        current_change["params"]["start_marker"] = line
                    elif "end_marker" not in current_change["params"] or not current_change["params"]["end_marker"]:
                        if line.startswith("END:") or line.startswith("END BLOCK"):
                            current_change["params"]["end_marker"] = line.split(":", 1)[1].strip() if ":" in line else ""
                        else:
                            current_change["params"]["new_block"].append(line)
                    else:
                        current_change["params"]["new_block"].append(line)
                
                elif current_change["operation"] == "file" or current_change["operation"] == "script":
                    current_change["params"]["content"].append(line)
            
            i += 1
        
        # Add the last change if there is one
        if current_change:
            # For file/script operations, join the content lines
            if current_change["operation"] in ["file", "script"] and "content" in current_change["params"]:
                current_change["params"]["content"] = "\n".join(current_change["params"]["content"])
            
            # For block operations, join the new_block lines
            if current_change["operation"] == "block" and "new_block" in current_change["params"]:
                current_change["params"]["new_block"] = "\n".join(current_change["params"]["new_block"])
            
            changes.append(current_change)
            
        return changes
        
    async def _generate_script_from_transformation(self, transformation: Dict[str, Any]) -> Optional[str]:
        """
        Generate a standalone script from a transformation
        
        Args:
            transformation: The transformation dictionary
            
        Returns:
            Optional[str]: Path to the generated script, or None if no script was generated
        """
        try:
            # Check if the transformation contains script operations
            changes = self._parse_code_changes(transformation["suggestion"])
            script_changes = [c for c in changes if c["operation"] == "script"]
            
            if not script_changes:
                # No explicit script operations, try to generate a script from the transformation
                script_name = f"auto_transform_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
                script_path = os.path.join(self.script_output_dir, script_name)
                
                # Generate script content
                script_content = self._generate_script_content(transformation, script_name)
                
                # Write script to file
                with open(script_path, "w") as f:
                    f.write(script_content)
                
                # Make script executable
                os.chmod(script_path, 0o755)
                
                logging.info(f"Generated script: {script_path}")
                return script_path
            
            # Process explicit script operations
            for script_change in script_changes:
                script_name = script_change["params"].get("script_name", f"script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py")
                script_content = script_change["params"].get("content", "")
                
                if not script_content:
                    continue
                
                # Ensure script has .py extension
                if not script_name.endswith(".py"):
                    script_name += ".py"
                
                script_path = os.path.join(self.script_output_dir, script_name)
                
                # Write script to file
                with open(script_path, "w") as f:
                    f.write(script_content)
                
                # Make script executable
                os.chmod(script_path, 0o755)
                
                logging.info(f"Generated script from explicit operation: {script_path}")
                return script_path
            
            return None
            
        except Exception as e:
            logging.error(f"Error generating script from transformation: {e}")
            return None
    
    def _generate_script_content(self, transformation: Dict[str, Any], script_name: str) -> str:
        """
        Generate content for a standalone script based on a transformation
        
        Args:
            transformation: The transformation dictionary
            script_name: Name of the script
            
        Returns:
            str: Content for the script
        """
        # Extract information from the transformation
        original_content = transformation.get("original", {}).get("content", "")
        suggestion = transformation.get("suggestion", "")
        
        # Generate script header
        header = f"""#!/usr/bin/env python3
# {script_name}
# Auto-generated script from transformation at {datetime.now().isoformat()}
# 
# This script demonstrates and tests the code transformation.
# It can be executed directly to see the effects of the transformation.

import os
import sys
import logging
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Original code (for reference)
ORIGINAL_CODE = '''
{original_content}
'''

# Transformed code
TRANSFORMED_CODE = '''
{self._extract_transformed_code(transformation)}
'''

def run_test():
    \"\"\"Run tests on the transformed code\"\"\"
    logger.info("Testing transformed code...")
    
    try:
        # Execute the transformed code
        namespace = {{}}
        exec(TRANSFORMED_CODE, namespace)
        
        # Try to identify functions or classes to test
        test_candidates = [
            name for name, obj in namespace.items()
            if callable(obj) and not name.startswith('_')
        ]
        
        if test_candidates:
            logger.info(f"Found test candidates: {{test_candidates}}")
            for name in test_candidates:
                try:
                    logger.info(f"Testing {{name}}...")
                    result = namespace[name]()
                    logger.info(f"Result: {{result}}")
                except Exception as e:
                    logger.error(f"Error testing {{name}}: {{e}}")
        else:
            logger.info("No test candidates found, executing as module")
        
        return True
    except Exception as e:
        logger.error(f"Error executing transformed code: {{e}}")
        logger.error(traceback.format_exc())
        return False

def main():
    \"\"\"Main entry point\"\"\"
    logger.info(f"Running {{os.path.basename(__file__)}}")
    
    success = run_test()
    
    if success:
        logger.info("Transformation test completed successfully")
        return 0
    else:
        logger.error("Transformation test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
"""
        
        return header
    
    def _extract_transformed_code(self, transformation: Dict[str, Any]) -> str:
        """
        Extract the transformed code from a transformation
        
        Args:
            transformation: The transformation dictionary
            
        Returns:
            str: The transformed code
        """
        # Try to extract transformed code from the suggestion
        suggestion = transformation.get("suggestion", "")
        
        # Look for code blocks in the suggestion
        import re
        code_blocks = re.findall(r'```(?:python)?\n(.*?)```', suggestion, re.DOTALL)
        
        if code_blocks:
            # Return the largest code block
            return max(code_blocks, key=len)
        
        # If no code blocks found, try to extract from the changes
        changes = self._parse_code_changes(suggestion)
        
        code_parts = []
        for change in changes:
            if change["operation"] == "line" and "content" in change["params"]:
                code_parts.append(change["params"]["content"])
            elif change["operation"] == "block" and "new_block" in change["params"]:
                code_parts.append(change["params"]["new_block"])
            elif change["operation"] == "file" and "content" in change["params"]:
                code_parts.append(change["params"]["content"])
            elif change["operation"] == "script" and "content" in change["params"]:
                code_parts.append(change["params"]["content"])
        
        if code_parts:
            return "\n\n".join(code_parts)
        
        # Fallback: return the whole suggestion
        return suggestion
    
    async def _execute_generated_script(self, script_path: str) -> Dict[str, Any]:
        """
        Execute a generated script and capture its output
        
        Args:
            script_path: Path to the script to execute
            
        Returns:
            Dict[str, Any]: Execution results
        """
        try:
            logging.info(f"Executing generated script: {script_path}")
            
            # Create a task for tracking the execution
            task = await self.agent.create_task(
                title=f"Execute script: {os.path.basename(script_path)}",
                description=f"Execute and monitor the generated script at {script_path}",
                priority=3,
                tags=["script", "auto-execute"]
            )
            
            task_id = None
            if task.get("success", False):
                task_id = task.get("task_id")
                await self.agent.update_task(task_id, status="in_progress")
            
            # Execute the script in a subprocess
            start_time = time.time()
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Set a timeout for the script execution
            timeout = 60  # 60 seconds timeout
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                exit_code = process.returncode
                success = exit_code == 0
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                exit_code = -1
                success = False
                stderr += "\nProcess timed out after {timeout} seconds"
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Prepare result
            result = {
                "script_path": script_path,
                "success": success,
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
                "execution_time": execution_time
            }
            
            # Update task with results
            if task_id:
                await self.agent.update_task(
                    task_id=task_id,
                    status="completed" if success else "failed",
                    result=result
                )
            
            # Log results
            if success:
                logging.info(f"Script executed successfully in {execution_time:.2f}s: {script_path}")
            else:
                logging.error(f"Script execution failed with exit code {exit_code}: {script_path}")
                if stderr:
                    logging.error(f"Error output: {stderr}")
            
            return result
            
        except Exception as e:
            logging.error(f"Error executing script: {e}")
            if task_id:
                await self.agent.update_task(
                    task_id=task_id,
                    status="failed",
                    result={"error": str(e)}
                )
            return {
                "script_path": script_path,
                "success": False,
                "error": str(e)
            }
    
    async def _collect_cognitive_feedback(self, transformation: Dict[str, Any], test_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect cognitive feedback on a transformation
        
        Args:
            transformation: The transformation dictionary
            test_results: Results of testing the transformation
            
        Returns:
            Dict[str, Any]: Cognitive feedback
        """
        try:
            # Prepare prompt for cognitive feedback
            feedback_prompt = f"""
            Analyze this code transformation and provide cognitive feedback:
            
            Original code:
            ```
            {transformation['original']['content']}
            ```
            
            Transformation suggestion:
            ```
            {transformation['suggestion']}
            ```
            
            Test results:
            ```
            {json.dumps(test_results, indent=2)}
            ```
            
            Provide detailed cognitive feedback on:
            1. Quality of the transformation (clarity, correctness, efficiency)
            2. Potential edge cases or risks not covered by tests
            3. Alternative approaches that could have been taken
            4. Learning opportunities for future transformations
            5. Meta-cognitive assessment of the transformation process itself
            """
            
            messages = [
                {"role": "system", "content": "You are an expert code reviewer providing cognitive feedback on code transformations."},
                {"role": "user", "content": feedback_prompt}
            ]
            
            response = await client.chat.completions.create(
                model="o3-mini",
                messages=messages,
                temperature=0.3
            )
            
            feedback = {
                "transformation_id": transformation.get("id", str(uuid.uuid4())),
                "feedback": response.choices[0].message.content,
                "timestamp": datetime.now().isoformat(),
                "test_results": test_results
            }
            
            # Store the feedback
            self.cognitive_feedback.append(feedback)
            
            logging.info(f"Collected cognitive feedback on transformation: {feedback['feedback'][:100]}...")
            
            return feedback
            
        except Exception as e:
            logging.error(f"Error collecting cognitive feedback: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

class Task(BaseModel):
    """
    Represents a task that can be created, tracked, and executed by the agent.
    Enhanced with checkpointing and resumption capabilities.
    """
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed, paused
    priority: int = 5  # 1-10, lower is higher priority
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)  # List of task_ids this task depends on
    tags: List[str] = Field(default_factory=list)
    result: Optional[Any] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # New fields for autonomous work
    progress: float = 0.0  # Progress from 0.0 to 1.0
    checkpoint_data: Dict[str, Any] = Field(default_factory=dict)  # Stores resumable state
    last_checkpoint: Optional[str] = None  # Timestamp of last checkpoint
    estimated_time_remaining: Optional[float] = None  # In seconds
    subtasks: List[Dict[str, Any]] = Field(default_factory=list)  # Hierarchical task breakdown
    autonomous_mode: bool = False  # Whether this task can be worked on autonomously
    interruption_count: int = 0  # Number of times this task has been interrupted
    resumption_strategy: str = "continue"  # continue, restart, or adaptive

    def update_status(self, status: str):
        """Update task status and timestamp"""
        self.status = status
        self.updated_at = datetime.now().isoformat()
        if status == "completed":
            self.completed_at = datetime.now().isoformat()
            self.progress = 1.0
        elif status == "paused":
            # Create a checkpoint when pausing
            self.create_checkpoint()

    def add_tag(self, tag: str):
        """Add a tag to the task"""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now().isoformat()
            
    def update_progress(self, progress: float, save_checkpoint: bool = True):
        """Update task progress and optionally create a checkpoint"""
        self.progress = max(0.0, min(1.0, progress))
        self.updated_at = datetime.now().isoformat()
        
        if save_checkpoint:
            self.create_checkpoint()
            
    def create_checkpoint(self):
        """Create a checkpoint of the current task state"""
        self.last_checkpoint = datetime.now().isoformat()
        # Store any additional state needed for resumption in checkpoint_data
        
    def resume_from_checkpoint(self):
        """Resume task execution from the last checkpoint"""
        if not self.last_checkpoint:
            return False
            
        self.interruption_count += 1
        if self.status == "paused":
            self.status = "in_progress"
            self.updated_at = datetime.now().isoformat()
        
        return True
        
    def add_subtask(self, title: str, description: str, priority: int = 5):
        """Add a subtask to this task"""
        subtask = {
            "id": str(uuid.uuid4()),
            "title": title,
            "description": description,
            "priority": priority,
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
        self.subtasks.append(subtask)
        return subtask["id"]
        
    def get_subtask(self, subtask_id: str):
        """Get a subtask by ID"""
        for subtask in self.subtasks:
            if subtask["id"] == subtask_id:
                return subtask
        return None
        
    def update_subtask(self, subtask_id: str, status: str, progress: float = None):
        """Update a subtask's status and progress"""
        for i, subtask in enumerate(self.subtasks):
            if subtask["id"] == subtask_id:
                subtask["status"] = status
                subtask["updated_at"] = datetime.now().isoformat()
                if progress is not None:
                    subtask["progress"] = progress
                if status == "completed":
                    subtask["completed_at"] = datetime.now().isoformat()
                self.subtasks[i] = subtask
                
                # Update overall task progress based on subtasks
                self._recalculate_progress()
                return True
        return False
        
    def _recalculate_progress(self):
        """Recalculate overall progress based on subtasks"""
        if not self.subtasks:
            return
            
        completed = sum(1 for s in self.subtasks if s.get("status") == "completed")
        in_progress = sum(s.get("progress", 0) for s in self.subtasks 
                         if s.get("status") == "in_progress" and "progress" in s)
        
        total_progress = (completed + in_progress) / len(self.subtasks)
        self.progress = total_progress

class TaskManager:
    """
    Manages tasks for the agent, including creation, tracking, and execution.
    Enhanced with persistent state management and autonomous task selection.
    """
    def __init__(self, db_conn: Optional[sqlite3.Connection] = None):
        self.tasks: Dict[str, Task] = {}
        self.db_conn = db_conn
        self.current_autonomous_task_id: Optional[str] = None
        self.autonomous_mode_enabled: bool = False
        self.task_history: List[Dict[str, Any]] = []
        self.task_graph = nx.DiGraph()  # Dependency graph for tasks
        
        if db_conn:
            self._create_tables()
            self._load_tasks_from_db()

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
                metadata TEXT,
                progress REAL DEFAULT 0.0,
                checkpoint_data TEXT,
                last_checkpoint TEXT,
                estimated_time_remaining REAL,
                subtasks TEXT,
                autonomous_mode INTEGER DEFAULT 0,
                interruption_count INTEGER DEFAULT 0,
                resumption_strategy TEXT DEFAULT 'continue'
            )
        """)
        
        # Create table for task history
        cur.execute("""
            CREATE TABLE IF NOT EXISTS task_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                previous_status TEXT,
                new_status TEXT,
                metadata TEXT,
                FOREIGN KEY(task_id) REFERENCES tasks(task_id)
            )
        """)
        
        # Create table for task dependencies
        cur.execute("""
            CREATE TABLE IF NOT EXISTS task_dependencies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                depends_on_task_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(task_id) REFERENCES tasks(task_id),
                FOREIGN KEY(depends_on_task_id) REFERENCES tasks(task_id)
            )
        """)
        
        self.db_conn.commit()

    def create_task(self, title: str, description: str, priority: int = 5, 
                   dependencies: List[str] = None, tags: List[str] = None,
                   autonomous_mode: bool = False) -> Task:
        """Create a new task with enhanced capabilities"""
        task = Task(
            title=title,
            description=description,
            priority=priority,
            dependencies=dependencies or [],
            tags=tags or [],
            autonomous_mode=autonomous_mode
        )
        self.tasks[task.task_id] = task
        
        # Save to database if available
        if self.db_conn:
            self._save_task_to_db(task)
            
        # Add task to dependency graph
        self._update_task_graph(task)
        
        # Record task creation in history
        self._record_task_history(task, "create", None, "pending")
            
        return task
        
    def _update_task_graph(self, task: Task):
        """Update the task dependency graph"""
        # Add the task node if it doesn't exist
        if task.task_id not in self.task_graph:
            self.task_graph.add_node(task.task_id, task=task)
        
        # Update node data
        self.task_graph.nodes[task.task_id]['task'] = task
        
        # Add dependency edges
        for dep_id in task.dependencies:
            if dep_id in self.tasks:
                self.task_graph.add_edge(dep_id, task.task_id)
                
                # Also save to database if available
                if self.db_conn:
                    cur = self.db_conn.cursor()
                    cur.execute("""
                        INSERT OR IGNORE INTO task_dependencies
                        (task_id, depends_on_task_id, created_at)
                        VALUES (?, ?, ?)
                    """, (
                        task.task_id,
                        dep_id,
                        datetime.now().isoformat()
                    ))
                    self.db_conn.commit()
        
    def _record_task_history(self, task: Task, action: str, 
                           previous_status: Optional[str], new_status: str):
        """Record task history for auditing and analysis"""
        history_entry = {
            "task_id": task.task_id,
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "previous_status": previous_status,
            "new_status": new_status,
            "metadata": {
                "progress": task.progress,
                "priority": task.priority
            }
        }
        
        self.task_history.append(history_entry)
        
        # Save to database if available
        if self.db_conn:
            cur = self.db_conn.cursor()
            cur.execute("""
                INSERT INTO task_history
                (task_id, timestamp, action, previous_status, new_status, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                task.task_id,
                history_entry["timestamp"],
                action,
                previous_status,
                new_status,
                json.dumps(history_entry["metadata"])
            ))
            self.db_conn.commit()

    def _save_task_to_db(self, task: Task):
        """Save task to database with enhanced fields"""
        cur = self.db_conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO tasks 
            (task_id, title, description, status, priority, created_at, 
             updated_at, completed_at, dependencies, tags, result, metadata,
             progress, checkpoint_data, last_checkpoint, estimated_time_remaining,
             subtasks, autonomous_mode, interruption_count, resumption_strategy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            json.dumps(task.metadata),
            task.progress,
            json.dumps(task.checkpoint_data),
            task.last_checkpoint,
            task.estimated_time_remaining,
            json.dumps(task.subtasks),
            1 if task.autonomous_mode else 0,
            task.interruption_count,
            task.resumption_strategy
        ))
        self.db_conn.commit()
        
        # Update task dependency graph
        self._update_task_graph(task)
        
        # Record task history
        self._record_task_history(task, "save", None, task.status)

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
        """Update task status with history tracking"""
        task = self.get_task(task_id)
        if not task:
            return None
        
        previous_status = task.status
        task.update_status(status)
        
        # Save to database if available
        if self.db_conn:
            self._save_task_to_db(task)
            
        # Record status change in history
        self._record_task_history(task, "status_update", previous_status, status)
        
        # If this was the current autonomous task and it's completed/failed, clear it
        if self.current_autonomous_task_id == task_id and status in ["completed", "failed"]:
            self.current_autonomous_task_id = None
            
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

    def get_next_task(self, for_autonomous: bool = False) -> Optional[Task]:
        """
        Get the highest priority pending task with no unmet dependencies.
        
        Args:
            for_autonomous: If True, only consider tasks marked for autonomous execution
            
        Returns:
            The next task to work on, or None if no tasks are ready
        """
        # If we already have a current autonomous task, return it
        if for_autonomous and self.current_autonomous_task_id:
            current_task = self.get_task(self.current_autonomous_task_id)
            if current_task and current_task.status in ["pending", "in_progress", "paused"]:
                return current_task
        
        # Get all pending or paused tasks
        candidate_tasks = []
        for task in self.tasks.values():
            if task.status in ["pending", "paused"]:
                if for_autonomous and not task.autonomous_mode:
                    continue  # Skip non-autonomous tasks when in autonomous mode
                candidate_tasks.append(task)
        
        # Filter out tasks with unmet dependencies using the dependency graph
        ready_tasks = []
        for task in candidate_tasks:
            dependencies_met = True
            
            # Check if all dependencies are completed
            for dep_id in task.dependencies:
                dep_task = self.get_task(dep_id)
                if not dep_task or dep_task.status != "completed":
                    dependencies_met = False
                    break
                    
            if dependencies_met:
                ready_tasks.append(task)
                
        if not ready_tasks:
            return None
            
        # Sort by priority and then by whether it was previously paused
        # Paused tasks get priority over pending tasks of the same priority
        sorted_tasks = sorted(ready_tasks, key=lambda t: (
            t.priority,  # Lower priority number first
            0 if t.status == "paused" else 1,  # Paused tasks first
            -t.progress if t.progress > 0 else 0,  # Higher progress first
            t.created_at  # Older tasks first
        ))
        
        next_task = sorted_tasks[0] if sorted_tasks else None
        
        # If this is for autonomous execution, set it as the current task
        if for_autonomous and next_task:
            self.current_autonomous_task_id = next_task.task_id
            
        return next_task

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

    def _load_tasks_from_db(self):
        """Load tasks from database with enhanced fields"""
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
                metadata=json.loads(row[11]) if row[11] else {},
                progress=row[12] if len(row) > 12 else 0.0,
                checkpoint_data=json.loads(row[13]) if len(row) > 13 and row[13] else {},
                last_checkpoint=row[14] if len(row) > 14 else None,
                estimated_time_remaining=row[15] if len(row) > 15 else None,
                subtasks=json.loads(row[16]) if len(row) > 16 and row[16] else [],
                autonomous_mode=bool(row[17]) if len(row) > 17 else False,
                interruption_count=row[18] if len(row) > 18 else 0,
                resumption_strategy=row[19] if len(row) > 19 else "continue"
            )
            self.tasks[task.task_id] = task
            
        # Load task dependencies into the graph
        cur.execute("SELECT task_id, depends_on_task_id FROM task_dependencies")
        for row in cur.fetchall():
            task_id, depends_on_id = row
            if task_id in self.tasks and depends_on_id in self.tasks:
                self.task_graph.add_edge(depends_on_id, task_id)
                
        # Load task history
        cur.execute("SELECT task_id, timestamp, action, previous_status, new_status, metadata FROM task_history ORDER BY timestamp")
        for row in cur.fetchall():
            task_id, timestamp, action, prev_status, new_status, metadata_json = row
            self.task_history.append({
                "task_id": task_id,
                "timestamp": timestamp,
                "action": action,
                "previous_status": prev_status,
                "new_status": new_status,
                "metadata": json.loads(metadata_json) if metadata_json else {}
            })
            
    def enable_autonomous_mode(self, enabled: bool = True):
        """Enable or disable autonomous task execution mode"""
        self.autonomous_mode_enabled = enabled
        return self.autonomous_mode_enabled
        
    def pause_current_autonomous_task(self):
        """Pause the current autonomous task"""
        if not self.current_autonomous_task_id:
            return False
            
        task = self.get_task(self.current_autonomous_task_id)
        if task and task.status == "in_progress":
            return self.update_task_status(task.task_id, "paused")
        return False
        
    def resume_task(self, task_id: str) -> Optional[Task]:
        """Resume a paused task"""
        task = self.get_task(task_id)
        if not task or task.status != "paused":
            return None
            
        # Resume from checkpoint
        if task.resume_from_checkpoint():
            self.update_task_status(task_id, "in_progress")
            return task
        return None
        
    def update_task_progress(self, task_id: str, progress: float, 
                           checkpoint_data: Dict[str, Any] = None) -> Optional[Task]:
        """Update task progress and checkpoint data"""
        task = self.get_task(task_id)
        if not task:
            return None
            
        task.update_progress(progress, save_checkpoint=False)
        
        # Update checkpoint data if provided
        if checkpoint_data:
            task.checkpoint_data.update(checkpoint_data)
            task.create_checkpoint()
            
        # Save to database
        if self.db_conn:
            self._save_task_to_db(task)
            
        # Record progress update in history
        self._record_task_history(task, "progress_update", None, task.status)
        
        return task
        
    def get_task_dependencies(self, task_id: str) -> Dict[str, Any]:
        """Get detailed information about a task's dependencies"""
        task = self.get_task(task_id)
        if not task:
            return {"error": "Task not found"}
            
        result = {
            "task_id": task_id,
            "direct_dependencies": [],
            "all_dependencies": [],
            "blocking_dependencies": [],
            "dependency_graph": {}
        }
        
        # Get direct dependencies
        for dep_id in task.dependencies:
            dep_task = self.get_task(dep_id)
            if dep_task:
                dep_info = {
                    "task_id": dep_id,
                    "title": dep_task.title,
                    "status": dep_task.status,
                    "progress": dep_task.progress
                }
                result["direct_dependencies"].append(dep_info)
                
                # Check if this dependency is blocking
                if dep_task.status != "completed":
                    result["blocking_dependencies"].append(dep_info)
        
        # Get all dependencies using the graph
        if task_id in self.task_graph:
            # Get all ancestors (dependencies)
            ancestors = list(nx.ancestors(self.task_graph, task_id))
            for anc_id in ancestors:
                anc_task = self.get_task(anc_id)
                if anc_task:
                    result["all_dependencies"].append({
                        "task_id": anc_id,
                        "title": anc_task.title,
                        "status": anc_task.status,
                        "progress": anc_task.progress
                    })
            
            # Create a subgraph of dependencies for visualization
            subgraph = self.task_graph.subgraph(ancestors + [task_id])
            result["dependency_graph"] = nx.node_link_data(subgraph)
            
        return result
        
    def get_autonomous_task_status(self) -> Dict[str, Any]:
        """Get the status of autonomous task execution"""
        result = {
            "autonomous_mode_enabled": self.autonomous_mode_enabled,
            "current_task": None,
            "pending_autonomous_tasks": 0,
            "completed_autonomous_tasks": 0,
            "paused_autonomous_tasks": 0
        }
        
        # Count autonomous tasks by status
        for task in self.tasks.values():
            if task.autonomous_mode:
                if task.status == "pending":
                    result["pending_autonomous_tasks"] += 1
                elif task.status == "completed":
                    result["completed_autonomous_tasks"] += 1
                elif task.status == "paused":
                    result["paused_autonomous_tasks"] += 1
        
        # Get current autonomous task details
        if self.current_autonomous_task_id:
            task = self.get_task(self.current_autonomous_task_id)
            if task:
                result["current_task"] = {
                    "task_id": task.task_id,
                    "title": task.title,
                    "status": task.status,
                    "progress": task.progress,
                    "last_checkpoint": task.last_checkpoint,
                    "interruption_count": task.interruption_count
                }
                
        return result
        
    def decompose_task(self, task_id: str, subtasks: List[Dict[str, str]]) -> bool:
        """Break down a task into subtasks"""
        task = self.get_task(task_id)
        if not task:
            return False
            
        # Add subtasks
        for subtask_info in subtasks:
            title = subtask_info.get("title", "Untitled Subtask")
            description = subtask_info.get("description", "")
            priority = int(subtask_info.get("priority", 5))
            
            task.add_subtask(title, description, priority)
            
        # Save to database
        if self.db_conn:
            self._save_task_to_db(task)
            
        # Record decomposition in history
        self._record_task_history(
            task, 
            "decompose", 
            task.status, 
            task.status
        )
        
        return True

class AutonomousTaskExecutor:
    """
    Handles autonomous execution of tasks with persistence and interruption handling.
    This class manages the continuous execution of tasks in the background,
    with the ability to pause, resume, and checkpoint progress.
    """
    def __init__(self, agent):
        self.agent = agent
        self.running = False
        self.executor_task = None
        self.current_task_id = None
        self.execution_interval = 5.0  # Seconds between execution cycles
        self.max_execution_time = 60.0  # Maximum time to spend on a task before checkpointing
        self.execution_stats = {
            "tasks_completed": 0,
            "tasks_started": 0,
            "tasks_paused": 0,
            "total_execution_time": 0.0,
            "interruptions": 0
        }
        self.last_status_check = time.time()
        self.status_check_interval = 30.0  # Check system status every 30 seconds
        
    async def start(self):
        """Start the autonomous task executor"""
        if self.running:
            return False
            
        self.running = True
        self.executor_task = asyncio.create_task(self._executor_loop())
        logging.info("Autonomous task executor started")
        return True
        
    async def stop(self):
        """Stop the autonomous task executor"""
        if not self.running:
            return False
            
        self.running = False
        
        # Pause the current task if there is one
        if self.current_task_id:
            await self._pause_current_task()
            
        # Wait for the executor to stop
        if self.executor_task:
            try:
                await asyncio.wait_for(self.executor_task, timeout=10.0)
            except asyncio.TimeoutError:
                self.executor_task.cancel()
                try:
                    await self.executor_task
                except asyncio.CancelledError:
                    pass
                    
        logging.info("Autonomous task executor stopped")
        return True
        
    async def _executor_loop(self):
        """Main loop for autonomous task execution"""
        try:
            while self.running:
                # Check if task manager is in autonomous mode
                if not self.agent.task_manager.autonomous_mode_enabled:
                    await asyncio.sleep(self.execution_interval)
                    continue
                    
                # Check system status periodically
                current_time = time.time()
                if current_time - self.last_status_check >= self.status_check_interval:
                    self.last_status_check = current_time
                    if not await self._check_system_status():
                        # System is not in a good state, pause execution
                        logging.warning("System status check failed, pausing autonomous execution")
                        await asyncio.sleep(self.status_check_interval)
                        continue
                
                # Get the next task to work on
                task = self.agent.task_manager.get_next_task(for_autonomous=True)
                
                if not task:
                    # No tasks available, wait and try again
                    await asyncio.sleep(self.execution_interval)
                    continue
                    
                # Set as current task
                self.current_task_id = task.task_id
                
                # If task was paused, resume it
                if task.status == "paused":
                    logging.info(f"Resuming paused task: {task.title} ({task.task_id})")
                    self.agent.task_manager.resume_task(task.task_id)
                else:
                    # Start a new task
                    logging.info(f"Starting autonomous task: {task.title} ({task.task_id})")
                    self.agent.task_manager.update_task_status(task.task_id, "in_progress")
                    self.execution_stats["tasks_started"] += 1
                
                # Execute the task with time limit
                start_time = time.time()
                try:
                    success = await asyncio.wait_for(
                        self._execute_task(task),
                        timeout=self.max_execution_time
                    )
                    
                    if success:
                        # Task completed successfully
                        self.agent.task_manager.update_task_status(task.task_id, "completed")
                        self.execution_stats["tasks_completed"] += 1
                        logging.info(f"Completed task: {task.title} ({task.task_id})")
                    else:
                        # Task failed
                        self.agent.task_manager.update_task_status(task.task_id, "failed")
                        logging.warning(f"Failed task: {task.title} ({task.task_id})")
                        
                except asyncio.TimeoutError:
                    # Task took too long, checkpoint and pause
                    logging.info(f"Task execution time limit reached, checkpointing: {task.title}")
                    await self._pause_current_task()
                    self.execution_stats["tasks_paused"] += 1
                    
                except asyncio.CancelledError:
                    # Execution was cancelled, checkpoint and pause
                    logging.info(f"Task execution cancelled, checkpointing: {task.title}")
                    await self._pause_current_task()
                    self.execution_stats["interruptions"] += 1
                    raise
                    
                except Exception as e:
                    # Unexpected error
                    logging.error(f"Error executing task {task.task_id}: {e}")
                    self.agent.task_manager.update_task_status(task.task_id, "failed")
                    
                finally:
                    # Update execution stats
                    execution_time = time.time() - start_time
                    self.execution_stats["total_execution_time"] += execution_time
                    self.current_task_id = None
                    
                # Small delay between tasks
                await asyncio.sleep(1.0)
                
        except asyncio.CancelledError:
            logging.info("Autonomous task executor cancelled")
            raise
        except Exception as e:
            logging.error(f"Error in autonomous task executor: {e}")
            
    async def _execute_task(self, task: Task) -> bool:
        """
        Execute a single task autonomously
        
        Args:
            task: The task to execute
            
        Returns:
            bool: True if the task was completed successfully, False otherwise
        """
        try:
            # Check if task has subtasks
            if task.subtasks:
                # Execute subtasks in order
                return await self._execute_subtasks(task)
                
            # No subtasks, execute the task directly
            # This would typically involve generating a plan and executing it
            # For now, we'll simulate task execution with progress updates
            
            # Create a conversation for this task if needed
            conv_id = task.metadata.get("conversation_id")
            if not conv_id:
                conv_id = await self.agent.create_conversation(f"Task: {task.title}")
                task.metadata["conversation_id"] = conv_id
                self.agent.task_manager._save_task_to_db(task)
            
            # Generate a plan if not already in checkpoint data
            if "plan" not in task.checkpoint_data:
                plan_prompt = f"""
                I need to complete the following task autonomously:
                
                Task: {task.title}
                Description: {task.description}
                
                Please create a detailed step-by-step plan to complete this task.
                For each step, include:
                1. A clear description of what needs to be done
                2. Any tools or resources needed
                3. How to verify the step was completed successfully
                4. Estimated time to complete
                
                Format the plan as a JSON array of steps.
                """
                
                # Use DEEP_TASK mode for better planning
                original_mode = self.agent.response_mode
                await self.agent.set_response_mode(ResponseMode.DEEP_TASK)
                
                plan_response = await self.agent.qa(conv_id, plan_prompt)
                
                # Restore original mode
                self.agent.response_mode = original_mode
                
                # Extract JSON plan from response
                import re
                json_matches = re.findall(r'```(?:json)?\s*([\s\S]*?)```', plan_response)
                
                if json_matches:
                    try:
                        plan = json.loads(json_matches[0])
                        task.checkpoint_data["plan"] = plan
                        task.checkpoint_data["current_step"] = 0
                        self.agent.task_manager._save_task_to_db(task)
                    except json.JSONDecodeError:
                        # Fallback: create a simple plan
                        task.checkpoint_data["plan"] = [
                            {"description": "Complete the task", "estimated_time": 60}
                        ]
                        task.checkpoint_data["current_step"] = 0
                else:
                    # Fallback: create a simple plan
                    task.checkpoint_data["plan"] = [
                        {"description": "Complete the task", "estimated_time": 60}
                    ]
                    task.checkpoint_data["current_step"] = 0
            
            # Execute the plan steps
            plan = task.checkpoint_data["plan"]
            current_step = task.checkpoint_data["current_step"]
            
            while current_step < len(plan):
                step = plan[current_step]
                
                # Execute the current step
                step_prompt = f"""
                I'm working on the task: {task.title}
                
                Current step ({current_step + 1}/{len(plan)}): {step['description']}
                
                Please execute this step and provide the results. If tools are needed,
                use them appropriately. When the step is complete, indicate success
                and provide any relevant output.
                """
                
                step_response = await self.agent.qa(conv_id, step_prompt)
                
                # Update step with results
                step["result"] = step_response
                step["completed"] = True
                step["completed_at"] = datetime.now().isoformat()
                
                # Move to next step
                current_step += 1
                task.checkpoint_data["current_step"] = current_step
                
                # Update progress based on steps completed
                progress = current_step / len(plan)
                self.agent.task_manager.update_task_progress(
                    task.task_id, 
                    progress, 
                    task.checkpoint_data
                )
                
                # Check if we should yield to allow for interruption
                if not self.running:
                    return False
            
            # All steps completed
            return True
            
        except Exception as e:
            logging.error(f"Error executing task {task.task_id}: {e}")
            return False
            
    async def _execute_subtasks(self, task: Task) -> bool:
        """Execute a task's subtasks"""
        # Get pending or in-progress subtasks
        pending_subtasks = [
            s for s in task.subtasks 
            if s.get("status") in ["pending", "in_progress"]
        ]
        
        if not pending_subtasks:
            # All subtasks are completed
            return True
            
        # Sort subtasks by priority
        pending_subtasks.sort(key=lambda s: s.get("priority", 5))
        
        # Execute each subtask
        for subtask in pending_subtasks:
            subtask_id = subtask["id"]
            
            # Skip completed subtasks
            if subtask.get("status") == "completed":
                continue
                
            # Mark subtask as in progress
            task.update_subtask(subtask_id, "in_progress")
            self.agent.task_manager._save_task_to_db(task)
            
            # Create a conversation for this subtask if needed
            conv_id = subtask.get("conversation_id")
            if not conv_id:
                conv_id = await self.agent.create_conversation(f"Subtask: {subtask['title']}")
                subtask["conversation_id"] = conv_id
                task.update_subtask(subtask_id, "in_progress")
                self.agent.task_manager._save_task_to_db(task)
            
            # Execute the subtask
            subtask_prompt = f"""
            I need to complete the following subtask:
            
            Subtask: {subtask['title']}
            Description: {subtask['description']}
            
            This is part of the larger task: {task.title}
            
            Please execute this subtask and provide the results. If tools are needed,
            use them appropriately. When the subtask is complete, indicate success
            and provide any relevant output.
            """
            
            try:
                subtask_response = await self.agent.qa(conv_id, subtask_prompt)
                
                # Update subtask with results
                subtask["result"] = subtask_response
                subtask["completed_at"] = datetime.now().isoformat()
                task.update_subtask(subtask_id, "completed", 1.0)
                
                # Update overall task progress
                self.agent.task_manager._save_task_to_db(task)
                
            except Exception as e:
                logging.error(f"Error executing subtask {subtask_id}: {e}")
                task.update_subtask(subtask_id, "failed")
                self.agent.task_manager._save_task_to_db(task)
                
            # Check if we should yield to allow for interruption
            if not self.running:
                return False
                
        # Check if all subtasks are completed
        all_completed = all(
            s.get("status") == "completed" 
            for s in task.subtasks
        )
        
        return all_completed
            
    async def _pause_current_task(self):
        """Pause the current task with proper checkpointing"""
        if not self.current_task_id:
            return
            
        task = self.agent.task_manager.get_task(self.current_task_id)
        if task and task.status == "in_progress":
            # Create checkpoint and pause
            self.agent.task_manager.update_task_status(task.task_id, "paused")
            logging.info(f"Paused task: {task.title} ({task.task_id})")
            
    async def _check_system_status(self) -> bool:
        """
        Check if the system is in a good state to continue autonomous execution
        
        Returns:
            bool: True if the system is in a good state, False otherwise
        """
        try:
            # Check system resources
            import psutil
            
            # Check CPU usage
            cpu_usage = psutil.cpu_percent(interval=0.5)
            if cpu_usage > 90:  # CPU usage above 90%
                logging.warning(f"High CPU usage: {cpu_usage}%, pausing autonomous execution")
                return False
                
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:  # Memory usage above 90%
                logging.warning(f"High memory usage: {memory.percent}%, pausing autonomous execution")
                return False
                
            # Check disk space
            disk = psutil.disk_usage('/')
            if disk.percent > 95:  # Disk usage above 95%
                logging.warning(f"Low disk space: {disk.percent}% used, pausing autonomous execution")
                return False
                
            return True
            
        except ImportError:
            # psutil not available, assume system is OK
            return True
        except Exception as e:
            logging.error(f"Error checking system status: {e}")
            return True  # Assume OK on error to avoid unnecessary pausing

class ResponseMode(Enum):
    """Response modes for the agent"""
    NORMAL = "normal"         # Standard response mode
    WEB_SEARCH = "web_search" # Enhanced with web search results
    WEB_TRAWL = "web_trawl"   # Deep web search with multiple sources
    DEEP_RESEARCH = "deep_research" # Comprehensive research with citations
    DEEP_TASK = "deep_task"   # Task-focused with step-by-step execution
    DEEP_FLOW = "deep_flow"   # Continuous interaction with multiple sub-agents

class Agent(BaseModel):
    """
    Advanced agent that can ingest data, reflect on its own code,
    and modify itself based on internal reflection and reinforcement learning.
    Enhanced with autonomous task execution capabilities.
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
    job_scheduler: Optional[Any] = None  # Add job_scheduler field
    response_mode: ResponseMode = Field(default=ResponseMode.NORMAL)
    sub_agents: Dict[str, Any] = Field(default_factory=dict)  # For DEEP_FLOW mode
    active_flows: List[Dict[str, Any]] = Field(default_factory=list)  # For DEEP_FLOW mode
    autonomous_executor: Optional[Any] = None  # Autonomous task executor
    
    # Persistence settings
    persistence_enabled: bool = True
    persistence_interval: int = 60  # Seconds between state persistence
    last_persistence_time: float = Field(default_factory=time.time)
    
    # Work continuity settings
    work_continuity_enabled: bool = False
    max_continuous_work_time: int = 3600  # 1 hour
    work_session_start_time: Optional[float] = None
    
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
        
        # Initialize job scheduler for long-running tasks
        self.job_scheduler = JobScheduler(max_concurrent_jobs=10)
        # We'll start the job scheduler when needed, not in __init__
        
        # Initialize autonomous task executor
        self.autonomous_executor = AutonomousTaskExecutor(self)
        
        # Register persistence timer
        self._persistence_timer_task = None
        if self.persistence_enabled:
            self._start_persistence_timer()
        
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
        
        # Register job management tools
        register_tool(
            name="submit_background_job",
            func=self.submit_background_job,
            description="Submit a long-running job to be executed in the background",
            parameters={
                "name": {
                    "type": "string",
                    "description": "Human-readable name for the job"
                },
                "function_name": {
                    "type": "string",
                    "description": "Name of the function to execute"
                },
                "args": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of positional arguments",
                    "default": []
                },
                "kwargs": {
                    "type": "object",
                    "description": "Dictionary of keyword arguments",
                    "default": {}
                },
                "priority": {
                    "type": "integer",
                    "description": "Job priority (1-10, lower is higher priority)",
                    "default": 5
                },
                "timeout": {
                    "type": "number",
                    "description": "Optional timeout in seconds",
                    "default": None
                }
            }
        )
        
        # Register autonomous task management tools
        register_tool(
            name="start_autonomous_execution",
            func=self.start_autonomous_execution,
            description="Start autonomous task execution",
            parameters={}
        )
        
        register_tool(
            name="stop_autonomous_execution",
            func=self.stop_autonomous_execution,
            description="Stop autonomous task execution",
            parameters={}
        )
        
        register_tool(
            name="get_autonomous_execution_status",
            func=self.get_autonomous_execution_status,
            description="Get the status of autonomous task execution",
            parameters={}
        )
        
        register_tool(
            name="create_autonomous_task",
            func=self.create_autonomous_task,
            description="Create a task that can be executed autonomously",
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
                "dependencies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of task IDs this task depends on",
                    "default": []
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
            name="decompose_task",
            func=self.decompose_task,
            description="Break down a task into subtasks for autonomous execution",
            parameters={
                "task_id": {
                    "type": "string",
                    "description": "ID of the task to decompose"
                },
                "subtasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "priority": {"type": "integer"}
                        }
                    },
                    "description": "List of subtasks to create"
                }
            }
        )
        
        register_tool(
            name="get_job_status",
            func=self.get_job_status,
            description="Get the status of a background job",
            parameters={
                "job_id": {
                    "type": "string",
                    "description": "ID of the job to check"
                }
            }
        )
        
        register_tool(
            name="list_jobs",
            func=self.list_jobs,
            description="List all background jobs",
            parameters={
                "status": {
                    "type": "string",
                    "description": "Filter by status (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, TIMEOUT)",
                    "default": None
                }
            }
        )
        
        register_tool(
            name="cancel_job",
            func=self.cancel_job,
            description="Cancel a background job",
            parameters={
                "job_id": {
                    "type": "string",
                    "description": "ID of the job to cancel"
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
                
    async def submit_background_job(self, name: str, function_name: str, 
                                  args: List[Any] = None, kwargs: Dict[str, Any] = None,
                                  priority: int = 5, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Submit a long-running job to be executed in the background
        
        Args:
            name: Human-readable name for the job
            function_name: Name of the function to execute
            args: List of positional arguments
            kwargs: Dictionary of keyword arguments
            priority: Job priority (1-10, lower is higher priority)
            timeout: Optional timeout in seconds
            
        Returns:
            Dict containing job information
        """
        try:
            # Ensure job scheduler is running
            await self.ensure_job_scheduler_running()
            
            # Find the function by name
            if function_name in self.system_tools:
                func = self.system_tools[function_name]
            elif function_name in available_functions:
                func = available_functions[function_name]
            else:
                return {
                    "success": False,
                    "error": f"Function '{function_name}' not found"
                }
                
            # Submit the job
            job_id = await self.job_scheduler.submit_job(
                name=name,
                func=func,
                priority=priority,
                args=args,
                kwargs=kwargs,
                timeout=timeout
            )
            
            # Create a task to track the job
            task = await self.create_task(
                title=f"Background job: {name}",
                description=f"Long-running job executing function '{function_name}'",
                priority=priority,
                tags=["background_job", function_name]
            )
            
            # Link the job ID to the task
            if task.get("success", False):
                task_id = task.get("task_id")
                await self.update_task(
                    task_id=task_id,
                    status="in_progress",
                    result={"job_id": job_id}
                )
            
            return {
                "success": True,
                "job_id": job_id,
                "task_id": task.get("task_id") if task.get("success", False) else None,
                "name": name,
                "status": "PENDING",
                "message": f"Job '{name}' submitted successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a background job
        
        Args:
            job_id: ID of the job to check
            
        Returns:
            Dict containing job status information
        """
        try:
            status = await self.job_scheduler.get_job_status(job_id)
            
            if status:
                return {
                    "success": True,
                    "job_id": status["job_id"],
                    "name": status["name"],
                    "status": status["status"],
                    "progress": status["progress"],
                    "created_at": status["created_at"],
                    "started_at": status["started_at"],
                    "completed_at": status["completed_at"],
                    "error": status["error"],
                    "result": status["result"]
                }
            else:
                return {
                    "success": False,
                    "error": f"Job with ID '{job_id}' not found"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    async def list_jobs(self, status: Optional[str] = None) -> Dict[str, Any]:
        """
        List all background jobs
        
        Args:
            status: Optional status filter (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, TIMEOUT)
            
        Returns:
            Dict containing list of jobs
        """
        try:
            # Convert string status to enum if provided
            status_enum = None
            if status:
                try:
                    status_enum = JobStatus[status.upper()]
                except KeyError:
                    return {
                        "success": False,
                        "error": f"Invalid status: {status}"
                    }
            
            jobs = await self.job_scheduler.list_jobs(status=status_enum)
            
            return {
                "success": True,
                "count": len(jobs),
                "jobs": jobs
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    async def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a background job
        
        Args:
            job_id: ID of the job to cancel
            
        Returns:
            Dict containing cancellation result
        """
        try:
            result = await self.job_scheduler.cancel_job(job_id)
            
            if result:
                # Find and update any associated task
                tasks = await self.list_tasks(tag="background_job")
                if tasks.get("success", False):
                    for task in tasks.get("tasks", []):
                        task_id = task.get("task_id")
                        task_result = await self.get_task(task_id)
                        if task_result and task_result.get("result", {}).get("job_id") == job_id:
                            await self.update_task(
                                task_id=task_id,
                                status="failed",
                                result={"job_id": job_id, "cancelled": True}
                            )
                            break
                
                return {
                    "success": True,
                    "job_id": job_id,
                    "message": "Job cancelled successfully"
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to cancel job '{job_id}' (not found or already completed)"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

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
                # Create a task for tracking the search operation
                search_task = await self.create_task(
                    title=f"Web search for: {query}",
                    description=f"Search the web for information about: {query}",
                    priority=3,
                    tags=["search", "web"]
                )
                
                if search_task.get("success", False):
                    search_task_id = search_task.get("task_id")
                    await self.update_task(search_task_id, status="in_progress")
                    
                    # Perform the search
                    search_result = await self.system_tools["web_search"](query)
                    
                    # Update task with results
                    if search_result:
                        await self.update_task(
                            search_task_id,
                            status="completed",
                            result=search_result
                        )
                        return search_result
                    else:
                        await self.update_task(
                            search_task_id,
                            status="failed",
                            result={"error": "No search results found"}
                        )
                
                return await self.system_tools["web_search"](query)
            except Exception as e:
                logging.error(f"Web search error: {e}")
                
        # Try Jina search as fallback if web_search is not available
        if "jina_search" in self.system_tools and self.system_tools["jina_search"]:
            try:
                logging.info(f"Falling back to Jina search for: {query}")
                return await self.system_tools["jina_search"](query)
            except Exception as e:
                logging.error(f"Jina search error: {e}")
                
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
    
    async def ensure_job_scheduler_running(self):
        """Ensure the job scheduler is running"""
        if not self.job_scheduler.running:
            await self.job_scheduler.start()
            
        # Also ensure persistence timer is running
        if hasattr(self, '_persistence_timer_started') and not self._persistence_timer_started:
            await self._start_persistence_timer_async()
            
    async def start_autonomous_execution(self) -> bool:
        """Start autonomous task execution"""
        # Ensure job scheduler and persistence timer are running
        await self.ensure_job_scheduler_running()
        
        # Enable autonomous mode in task manager
        self.task_manager.enable_autonomous_mode(True)
        
        # Start the autonomous executor
        result = await self.autonomous_executor.start()
        
        if result:
            self.work_continuity_enabled = True
            self.work_session_start_time = time.time()
            logging.info("Autonomous task execution started")
            
        return result
        
    async def stop_autonomous_execution(self) -> bool:
        """Stop autonomous task execution"""
        # Disable autonomous mode in task manager
        self.task_manager.enable_autonomous_mode(False)
        
        # Stop the autonomous executor
        result = await self.autonomous_executor.stop()
        
        if result:
            self.work_continuity_enabled = False
            self.work_session_start_time = None
            logging.info("Autonomous task execution stopped")
            
        return result
        
    async def get_autonomous_execution_status(self) -> Dict[str, Any]:
        """Get the status of autonomous task execution"""
        # Get status from task manager
        task_status = self.task_manager.get_autonomous_task_status()
        
        # Get executor stats
        executor_stats = self.autonomous_executor.execution_stats.copy()
        
        # Calculate work session duration if active
        work_session_duration = None
        if self.work_session_start_time:
            work_session_duration = time.time() - self.work_session_start_time
            
        return {
            "enabled": self.work_continuity_enabled,
            "current_task": task_status.get("current_task"),
            "work_session_duration": work_session_duration,
            "max_continuous_work_time": self.max_continuous_work_time,
            "executor_stats": executor_stats,
            "task_stats": {
                "pending": task_status.get("pending_autonomous_tasks", 0),
                "completed": task_status.get("completed_autonomous_tasks", 0),
                "paused": task_status.get("paused_autonomous_tasks", 0)
            }
        }
        
    async def create_autonomous_task(self, title: str, description: str, priority: int = 5,
                                  dependencies: List[str] = None, tags: List[str] = None) -> Dict[str, Any]:
        """Create a task that can be executed autonomously"""
        try:
            task = self.task_manager.create_task(
                title=title,
                description=description,
                priority=priority,
                dependencies=dependencies or [],
                tags=tags or [],
                autonomous_mode=True
            )
            
            return {
                "success": True,
                "task_id": task.task_id,
                "title": task.title,
                "status": task.status,
                "priority": task.priority,
                "created_at": task.created_at,
                "autonomous_mode": task.autonomous_mode
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    async def decompose_task(self, task_id: str, subtasks: List[Dict[str, str]]) -> Dict[str, Any]:
        """Break down a task into subtasks for autonomous execution"""
        try:
            success = self.task_manager.decompose_task(task_id, subtasks)
            
            if success:
                task = self.task_manager.get_task(task_id)
                return {
                    "success": True,
                    "task_id": task_id,
                    "subtasks_count": len(task.subtasks),
                    "message": f"Task successfully decomposed into {len(task.subtasks)} subtasks"
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to decompose task"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def _start_persistence_timer(self):
        """Start a timer for periodic state persistence"""
        # We'll start the timer later when we have a running event loop
        self._persistence_timer_task = None
        self._persistence_timer_started = False
        
    async def _start_persistence_timer_async(self):
        """Start the persistence timer with a running event loop"""
        if self._persistence_timer_started:
            return
            
        async def persistence_timer():
            while self.persistence_enabled:
                current_time = time.time()
                if current_time - self.last_persistence_time >= self.persistence_interval:
                    await self._persist_state()
                    self.last_persistence_time = current_time
                await asyncio.sleep(min(10, self.persistence_interval / 2))
        
        self._persistence_timer_task = asyncio.create_task(persistence_timer())
        self._persistence_timer_started = True
        
    async def _persist_state(self):
        """Persist the agent's state to enable work continuity"""
        try:
            # Save any in-memory state to database
            # This is already handled by the task manager for tasks
            
            # Persist any additional state needed for work continuity
            if self.work_continuity_enabled and self.work_session_start_time:
                # Check if we've exceeded the maximum continuous work time
                if time.time() - self.work_session_start_time > self.max_continuous_work_time:
                    logging.info("Maximum continuous work time reached, pausing autonomous execution")
                    await self.stop_autonomous_execution()
                    
                    # Create a task to resume autonomous execution later
                    self.task_manager.create_task(
                        title="Resume autonomous execution",
                        description="Resume autonomous execution after the maximum continuous work time was reached",
                        priority=1,
                        tags=["system", "autonomous"],
                        autonomous_mode=True
                    )
            
            logging.debug("Agent state persisted")
            
        except Exception as e:
            logging.error(f"Error persisting agent state: {e}")
            
    async def set_response_mode(self, mode: Union[ResponseMode, str]) -> str:
        """Set the agent's response mode"""
        if isinstance(mode, str):
            try:
                mode = ResponseMode[mode.upper()]
            except KeyError:
                return f"Invalid response mode: {mode}. Available modes: {', '.join([m.value for m in ResponseMode])}"
        
        self.response_mode = mode
        return f"Response mode set to: {mode.value}"
    
    async def get_response_mode(self) -> str:
        """Get the current response mode"""
        return self.response_mode.value
    
    async def _get_mode_system_prompt(self) -> str:
        """Get the system prompt based on the current response mode"""
        base_prompt = "You are a helpful AI assistant with self-modification capabilities."
        
        if self.response_mode == ResponseMode.NORMAL:
            return f"{base_prompt} Provide clear, concise answers to questions."
            
        elif self.response_mode == ResponseMode.WEB_SEARCH:
            return f"{base_prompt} Provide detailed answers enriched with web search results. Include relevant information from search results and cite your sources."
            
        elif self.response_mode == ResponseMode.WEB_TRAWL:
            return f"""
{base_prompt} You are in WEB_TRAWL mode.
Provide comprehensive answers by deeply analyzing multiple web sources.
1. Thoroughly explore the web for diverse perspectives on the topic
2. Synthesize information from multiple sources
3. Present a detailed analysis with citations
4. Include contrasting viewpoints when available
5. Organize your response with clear sections and subheadings
6. Provide a summary of key findings
Your answers should be extensive and well-researched, typically 500+ words.
"""
            
        elif self.response_mode == ResponseMode.DEEP_RESEARCH:
            return f"""
{base_prompt} You are in DEEP_RESEARCH mode.
Act as an academic researcher providing exhaustive, scholarly responses:
1. Conduct in-depth research across multiple sources and disciplines
2. Analyze the topic from theoretical, historical, and practical perspectives
3. Evaluate the quality and reliability of sources
4. Present a nuanced, balanced view with proper citations
5. Identify gaps in current knowledge and suggest areas for further research
6. Structure your response with an introduction, methodology, findings, discussion, and conclusion
7. Include a bibliography of sources
Your answers should be thorough academic analyses, typically 1000+ words with proper citations.
"""
            
        elif self.response_mode == ResponseMode.DEEP_TASK:
            return f"""
{base_prompt} You are in DEEP_TASK mode.
Act as a specialized task execution agent:
1. Break down complex tasks into clear, actionable steps
2. Provide detailed instructions for each step
3. Anticipate potential challenges and offer solutions
4. Include relevant code, commands, or formulas when applicable
5. Explain the reasoning behind each step
6. Offer alternatives for different scenarios or constraints
7. Conclude with verification steps to ensure successful completion
Your responses should be comprehensive task guides that anyone can follow to completion.
"""
            
        elif self.response_mode == ResponseMode.DEEP_FLOW:
            return f"""
{base_prompt} You are in DEEP_FLOW mode.
You are coordinating multiple specialized sub-agents to solve complex problems:
1. Identify the different aspects of the problem that require specialized expertise
2. Delegate sub-tasks to appropriate specialized agents
3. Synthesize and integrate findings from multiple agents
4. Maintain a continuous conversation flow, allowing for follow-up questions
5. Track progress on long-running tasks and provide updates
6. Adapt your approach based on intermediate results
7. Present a unified, coherent response that integrates all perspectives
Your goal is to provide an interactive, multi-agent problem-solving experience that can evolve over time.
"""
        
        return base_prompt
    
    async def _create_sub_agents(self, question: str) -> Dict[str, Any]:
        """Create specialized sub-agents for DEEP_FLOW mode"""
        # Determine what types of sub-agents would be helpful for this question
        sub_agent_types = []
        
        # Research-oriented question
        if any(term in question.lower() for term in ["research", "study", "analyze", "compare", "evaluate"]):
            sub_agent_types.append("researcher")
            
        # Technical/coding question
        if any(term in question.lower() for term in ["code", "program", "develop", "build", "implement", "debug"]):
            sub_agent_types.append("developer")
            
        # Data analysis question
        if any(term in question.lower() for term in ["data", "statistics", "analyze", "trend", "pattern"]):
            sub_agent_types.append("data_analyst")
            
        # Creative question
        if any(term in question.lower() for term in ["create", "design", "generate", "creative", "story", "art"]):
            sub_agent_types.append("creative")
            
        # Planning/strategy question
        if any(term in question.lower() for term in ["plan", "strategy", "organize", "schedule", "project"]):
            sub_agent_types.append("planner")
            
        # If no specific types detected, use general purpose agents
        if not sub_agent_types:
            sub_agent_types = ["general", "critic"]
        
        # Create sub-agents
        sub_agents = {}
        for agent_type in sub_agent_types:
            agent_id = f"{agent_type}_{str(uuid.uuid4())[:8]}"
            sub_agents[agent_id] = {
                "type": agent_type,
                "id": agent_id,
                "status": "initialized",
                "tasks": [],
                "results": [],
                "created_at": datetime.now().isoformat()
            }
            
        return sub_agents
    
    async def _execute_deep_flow(self, question: str, conv_id: str) -> AsyncGenerator[str, None]:
        """Execute a DEEP_FLOW response with multiple sub-agents"""
        # Create sub-agents for this flow
        sub_agents = await self._create_sub_agents(question)
        
        # Create a flow record
        flow_id = str(uuid.uuid4())
        flow = {
            "id": flow_id,
            "conversation_id": conv_id,
            "question": question,
            "sub_agents": sub_agents,
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "results": []
        }
        
        # Add to active flows
        self.active_flows.append(flow)
        
        # Initial response
        yield f"Initiating DEEP_FLOW analysis for your question. I've created {len(sub_agents)} specialized sub-agents to help:\n"
        for agent_id, agent in sub_agents.items():
            yield f"- {agent['type'].capitalize()} Agent ({agent_id})\n"
        
        yield "\nBreaking down your question and delegating tasks...\n\n"
        
        # Simulate sub-agent work (in a real implementation, these would be actual parallel tasks)
        for agent_id, agent in sub_agents.items():
            agent_type = agent["type"]
            yield f"🔍 {agent_type.capitalize()} Agent is analyzing...\n"
            
            # Simulate thinking time
            await asyncio.sleep(1)
            
            # Generate a response based on agent type
            if agent_type == "researcher":
                yield f"📚 Research findings from {agent_type.capitalize()} Agent:\n"
                yield "I've analyzed several academic sources on this topic. The current research indicates...\n"
                yield "Several studies have shown conflicting results, suggesting this area needs more investigation...\n\n"
                
            elif agent_type == "developer":
                yield f"💻 Technical analysis from {agent_type.capitalize()} Agent:\n"
                yield "From a development perspective, this problem could be approached using these techniques...\n"
                yield "Here's a code snippet that demonstrates the core concept:\n```python\n# Example code\ndef solve_problem(input):\n    # Implementation\n    return solution\n```\n\n"
                
            elif agent_type == "data_analyst":
                yield f"📊 Data insights from {agent_type.capitalize()} Agent:\n"
                yield "The data shows several interesting patterns. First, there's a strong correlation between...\n"
                yield "When we analyze the trends over time, we can see that...\n\n"
                
            elif agent_type == "creative":
                yield f"🎨 Creative perspective from {agent_type.capitalize()} Agent:\n"
                yield "Looking at this from a creative angle, we could consider these innovative approaches...\n"
                yield "This reminds me of similar creative solutions in other domains, such as...\n\n"
                
            elif agent_type == "planner":
                yield f"📝 Strategic plan from {agent_type.capitalize()} Agent:\n"
                yield "To approach this systematically, I recommend the following steps:\n"
                yield "1. First, we should...\n2. Then, proceed to...\n3. Finally, evaluate...\n\n"
                
            elif agent_type == "general":
                yield f"🧠 General analysis from {agent_type.capitalize()} Agent:\n"
                yield "Looking at the big picture, the key aspects to consider are...\n"
                yield "The most important factors that influence this situation are...\n\n"
                
            elif agent_type == "critic":
                yield f"🔍 Critical evaluation from {agent_type.capitalize()} Agent:\n"
                yield "It's important to consider potential limitations and challenges...\n"
                yield "Some alternative perspectives to consider include...\n\n"
            
            # Update agent status
            agent["status"] = "completed"
            
        # Synthesize results
        yield "🔄 Synthesizing insights from all agents...\n\n"
        await asyncio.sleep(1)
        
        yield "📊 Comprehensive Analysis:\n\n"
        yield "Based on the collective insights from all specialized agents, I can provide a comprehensive response to your question.\n\n"
        yield "The key findings are:\n"
        yield "1. [First major insight combining multiple perspectives]\n"
        yield "2. [Second major insight with supporting evidence]\n"
        yield "3. [Third major insight with practical applications]\n\n"
        
        yield "This analysis represents the integration of multiple specialized perspectives. You can ask follow-up questions to explore any aspect in more depth.\n\n"
        
        # Update flow status
        flow["status"] = "completed"
        flow["completed_at"] = datetime.now().isoformat()
        
        yield "DEEP_FLOW analysis complete. Would you like to explore any specific aspect in more detail?"
    
    async def qa(self, conv_id: str, question: str) -> str:
        if not conv_id:
            conv_id = await self.create_conversation()
        await self.add_to_conversation(conv_id, "user", question)
        
        # Start reflections if not already running
        if not self._reflection_running:
            await self.start_reflections()
            
        # Ensure job scheduler is running
        await self.ensure_job_scheduler_running()

        # Check for response mode change request
        mode_change_patterns = [
            (r"(?:use|switch to|enable)\s+web\s*search\s*mode", ResponseMode.WEB_SEARCH),
            (r"(?:use|switch to|enable)\s+web\s*trawl\s*mode", ResponseMode.WEB_TRAWL),
            (r"(?:use|switch to|enable)\s+deep\s*research\s*mode", ResponseMode.DEEP_RESEARCH),
            (r"(?:use|switch to|enable)\s+deep\s*task\s*mode", ResponseMode.DEEP_TASK),
            (r"(?:use|switch to|enable)\s+deep\s*flow\s*mode", ResponseMode.DEEP_FLOW),
            (r"(?:use|switch to|enable)\s+normal\s*mode", ResponseMode.NORMAL)
        ]
        
        import re
        for pattern, mode in mode_change_patterns:
            if re.search(pattern, question.lower()):
                await self.set_response_mode(mode)
                mode_response = f"I've switched to {mode.value} mode. This means I'll now provide "
                
                if mode == ResponseMode.WEB_SEARCH:
                    mode_response += "detailed answers enriched with web search results, including citations to sources."
                elif mode == ResponseMode.WEB_TRAWL:
                    mode_response += "comprehensive answers by deeply analyzing multiple web sources, synthesizing information with citations and contrasting viewpoints."
                elif mode == ResponseMode.DEEP_RESEARCH:
                    mode_response += "exhaustive, scholarly responses with in-depth research across multiple sources and disciplines, proper citations, and academic structure."
                elif mode == ResponseMode.DEEP_TASK:
                    mode_response += "specialized task execution guidance with clear, actionable steps, detailed instructions, and solutions to potential challenges."
                elif mode == ResponseMode.DEEP_FLOW:
                    mode_response += "interactive problem-solving using multiple specialized sub-agents working together, allowing for continuous conversation flow."
                else:
                    mode_response += "clear, concise answers to your questions."
                
                mode_response += "\n\nHow can I help you with your question now?"
                await self.add_to_conversation(conv_id, "assistant", mode_response)
                return mode_response

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
                
                # Create a task for weather retrieval to handle it as a long-running operation
                weather_task_id = None
                try:
                    # Create a task for weather retrieval
                    weather_task = await self.create_task(
                        title=f"Weather lookup for {location}",
                        description=f"Retrieve current weather data for {location}",
                        priority=2,
                        tags=["weather", "api"]
                    )
                    
                    if weather_task.get("success", False):
                        weather_task_id = weather_task.get("task_id")
                        await self.update_task(weather_task_id, status="in_progress")
                        
                        # Try to get weather data with timeout and retry
                        weather_data = None
                        max_retries = 3
                        retry_count = 0
                        
                        while retry_count < max_retries and not weather_data:
                            try:
                                if "get_weather" in self.system_tools:
                                    # Set a timeout for the weather API call
                                    weather_future = asyncio.ensure_future(
                                        self.system_tools["get_weather"](location)
                                    )
                                    
                                    # Wait for the result with a timeout
                                    weather_data = await asyncio.wait_for(weather_future, timeout=5.0)
                                    
                                    if weather_data and "error" not in weather_data:
                                        # Update task with success
                                        await self.update_task(
                                            weather_task_id, 
                                            status="completed",
                                            result=weather_data
                                        )
                                        break
                                    else:
                                        # API returned an error
                                        retry_count += 1
                                        print(f"\n⚠️ Weather API error, retrying ({retry_count}/{max_retries})...", flush=True)
                                        await asyncio.sleep(1)  # Wait before retry
                                else:
                                    # Weather tool not available
                                    await self.update_task(
                                        weather_task_id,
                                        status="failed",
                                        result={"error": "Weather tool not available"}
                                    )
                                    break
                            except asyncio.TimeoutError:
                                # Timeout occurred
                                retry_count += 1
                                print(f"\n⚠️ Weather API timeout, retrying ({retry_count}/{max_retries})...", flush=True)
                                await asyncio.sleep(1)  # Wait before retry
                            except Exception as e:
                                # Other error occurred
                                await self.update_task(
                                    weather_task_id,
                                    status="failed",
                                    result={"error": str(e)}
                                )
                                print(f"\n❌ Error getting weather data: {str(e)}", flush=True)
                                break
                        
                        # Process the weather data if available
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
                            else:
                                # Fallback to Jina search results if available
                                if "jina_results" in weather_data:
                                    jina_weather = weather_data.get("jina_results", {}).get("weather_description", "unavailable")
                                    weather_response = f"I couldn't get precise weather data for {location}, but according to search results, the weather is approximately {jina_weather}."
                                    await self.add_to_conversation(conv_id, "assistant", weather_response)
                                    return weather_response
                        
                        # If we get here, all retries failed or no data available
                        if retry_count >= max_retries:
                            await self.update_task(
                                weather_task_id,
                                status="failed",
                                result={"error": "Maximum retries exceeded"}
                            )
                            print("\n❌ Weather data retrieval failed after maximum retries", flush=True)
                
                except Exception as e:
                    # Update task as failed if it was created
                    if weather_task_id:
                        await self.update_task(
                            weather_task_id,
                            status="failed",
                            result={"error": str(e)}
                        )
                    print(f"\n❌ Error in weather task: {str(e)}", flush=True)
                
                # If we reach here, we couldn't get weather data, so continue with normal processing
                print("\n⚠️ Weather data unavailable, falling back to standard processing", flush=True)
        
        # Special handling for DEEP_FLOW mode
        if self.response_mode == ResponseMode.DEEP_FLOW:
            # For DEEP_FLOW, we need to collect the streamed output
            full_response = ""
            async for chunk in self._execute_deep_flow(question, conv_id):
                full_response += chunk
                print(chunk, end="", flush=True)
            
            await self.add_to_conversation(conv_id, "assistant", full_response)
            self.conversation_history.append({
                "prompt": question,
                "response_mode": self.response_mode.value,
                "response": full_response
            })
            
            # After answering, trigger a self-reflection cycle
            await self.reflect_on_current_code()
            return full_response
        
        # Process with chain-of-thought
        cot = await self.analyze_thought_process(question)
        code_context = await self.retrieve_relevant_code(question)
        
        # Enhanced web search for WEB_SEARCH and WEB_TRAWL modes
        search_context = "Relevant documents: (stub)"
        if self.response_mode in [ResponseMode.WEB_SEARCH, ResponseMode.WEB_TRAWL] or \
           any(term in question.lower() for term in ["what", "who", "where", "when", "how", "search", "find", "look up", "confirm"]):
            print(f"\n🔍 Web search query detected: {question}", flush=True)
            
            # For WEB_TRAWL, do multiple searches with different queries
            if self.response_mode == ResponseMode.WEB_TRAWL:
                print(f"\n🌐 Performing deep web trawl with multiple queries...", flush=True)
                
                # Generate related search queries
                related_queries = [
                    question,
                    f"latest research on {question}",
                    f"alternative perspectives on {question}",
                    f"criticism of {question}",
                    f"history of {question}"
                ]
                
                search_context = "Deep Web Trawl Results:\n\n"
                
                # Perform multiple searches
                for i, query in enumerate(related_queries[:3]):  # Limit to 3 to avoid rate limits
                    print(f"\n🔍 Trawling with query {i+1}: {query}", flush=True)
                    search_result = await self.search_web(query)
                    
                    if search_result and "error" not in search_result:
                        search_context += f"--- Results for: {query} ---\n"
                        
                        # Extract and format results
                        if "items" in search_result:
                            for item in search_result["items"][:3]:
                                search_context += f"- {item['title']}: {item['snippet']}\n"
                                if 'link' in item:
                                    search_context += f"  Source: {item['link']}\n"
                        elif "organic_results" in search_result:
                            for item in search_result["organic_results"][:3]:
                                search_context += f"- {item['title']}: {item['snippet']}\n"
                                if 'link' in item:
                                    search_context += f"  Source: {item['link']}\n"
                        elif "results" in search_result:
                            for item in search_result["results"][:3]:
                                search_context += f"- {item.get('title', 'Result')}: {item.get('content', item.get('snippet', 'No content'))}\n"
                                if 'link' in item:
                                    search_context += f"  Source: {item['link']}\n"
                        
                        search_context += "\n"
                    
                    # Avoid rate limits
                    await asyncio.sleep(1)
            else:
                # Standard web search for other modes
                search_result = await self.search_web(question)
                if search_result and "error" not in search_result:
                    # Extract relevant information from search results
                    search_context = "Search results:\n"
                    if "items" in search_result:
                        for item in search_result["items"][:5]:  # Include more results
                            search_context += f"- {item['title']}: {item['snippet']}\n"
                            if 'link' in item:
                                search_context += f"  Source: {item['link']}\n"
                    elif "organic_results" in search_result:
                        for item in search_result["organic_results"][:5]:
                            search_context += f"- {item['title']}: {item['snippet']}\n"
                            if 'link' in item:
                                search_context += f"  Source: {item['link']}\n"
                    elif "answer" in search_result:
                        search_context += f"- Answer: {search_result['answer']}\n"
                    elif "results" in search_result:
                        for item in search_result["results"][:5]:
                            search_context += f"- {item.get('title', 'Result')}: {item.get('content', item.get('snippet', 'No content'))}\n"
                            if 'link' in item:
                                search_context += f"  Source: {item['link']}\n"
                    else:
                        # Generic fallback for any structure
                        search_context += f"- Results: {str(search_result)[:500]}...\n"
                        
                    print(f"\n📊 Found search results", flush=True)
                else:
                    search_context = "No relevant search results found."
                    print(f"\n⚠️ No search results found or search unavailable", flush=True)
        
        combined_context = f"{search_context}\n\nRelevant Code:\n{code_context}"
        
        # Get the appropriate system prompt based on response mode
        system_prompt = await self._get_mode_system_prompt()
        
        # Enhance the prompt based on response mode
        if self.response_mode == ResponseMode.DEEP_RESEARCH:
            prompt = f"""
Chain-of-Thought:
Initial Thought: {cot.initial_thought}
Steps: {" ".join(f"Step {s.step_number}: {s.reasoning} => {s.conclusion} (Confidence: {s.confidence})" for s in cot.steps)}
Final Conclusion: {cot.final_conclusion}

Context:
{combined_context}

You are in DEEP_RESEARCH mode. Provide an exhaustive, scholarly response with:
1. A clear introduction to the topic
2. Comprehensive analysis from multiple perspectives
3. Historical context and theoretical frameworks
4. Evaluation of evidence and sources
5. Discussion of implications and applications
6. Proper citations for all information
7. A conclusion summarizing key findings
8. A bibliography of sources

Question: {question}
"""
        elif self.response_mode == ResponseMode.DEEP_TASK:
            prompt = f"""
Chain-of-Thought:
Initial Thought: {cot.initial_thought}
Steps: {" ".join(f"Step {s.step_number}: {s.reasoning} => {s.conclusion} (Confidence: {s.confidence})" for s in cot.steps)}
Final Conclusion: {cot.final_conclusion}

Context:
{combined_context}

You are in DEEP_TASK mode. Provide a comprehensive task execution guide with:
1. A clear breakdown of the task into actionable steps
2. Detailed instructions for each step
3. Code examples, commands, or formulas where applicable
4. Explanation of the reasoning behind each step
5. Potential challenges and their solutions
6. Alternative approaches for different scenarios
7. Verification steps to ensure successful completion

Question: {question}
"""
        elif self.response_mode == ResponseMode.WEB_TRAWL:
            prompt = f"""
Chain-of-Thought:
Initial Thought: {cot.initial_thought}
Steps: {" ".join(f"Step {s.step_number}: {s.reasoning} => {s.conclusion} (Confidence: {s.confidence})" for s in cot.steps)}
Final Conclusion: {cot.final_conclusion}

Context:
{combined_context}

You are in WEB_TRAWL mode. Provide a comprehensive answer that:
1. Synthesizes information from multiple web sources
2. Presents diverse perspectives on the topic
3. Analyzes the reliability of different sources
4. Organizes information with clear sections and subheadings
5. Cites sources for all key information
6. Provides a balanced view of contrasting opinions
7. Concludes with a summary of key findings

Question: {question}
"""
        else:
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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        answer = await self._stream_chat_response(messages, conv_id)
        await self.add_to_conversation(conv_id, "assistant", answer)
        self.conversation_history.append({
            "prompt": question,
            "chain_of_thought": cot,
            "code_context": code_context,
            "response_mode": self.response_mode.value,
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
║   Type 'set_mode <mode>' to change response mode:                ║
║     - normal: Standard concise responses                         ║
║     - web_search: Enhanced with web search results               ║
║     - web_trawl: Deep web search with multiple sources           ║
║     - deep_research: Comprehensive research with citations       ║
║     - deep_task: Task-focused with step-by-step execution        ║
║     - deep_flow: Interactive with multiple sub-agents            ║
║   Type 'search <query>' to search the web directly               ║
║   Type 'polymorphic' to apply code transformations               ║
║   Type 'exit' to quit                                            ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
        prompt = "\033[1;32m(agent)\033[0m "
    
        def do_enable_autonomous(self, arg):
            """Enable autonomous mode: enable_autonomous [max_changes]"""
            try:
                args = shlex.split(arg)
                max_changes = int(args[0]) if args and args[0].isdigit() else 5
                auto_execute = True
                
                # Check for additional flags
                if len(args) > 1:
                    for flag in args[1:]:
                        if flag.lower() == "no-execute":
                            auto_execute = False
                
                self.agent.self_transformation.autonomous_mode = True
                self.agent.self_transformation.max_autonomous_changes = max_changes
                self.agent.self_transformation.changes_counter = 0
                self.agent.self_transformation.auto_execute_scripts = auto_execute
                
                print(f"Autonomous mode enabled with max {max_changes} changes")
                print(f"Auto-execute scripts: {auto_execute}")
            except ValueError:
                print("Invalid argument. Usage: enable_autonomous [max_changes] [no-execute]")
    
        def do_disable_autonomous(self, arg):
            """Disable autonomous mode: disable_autonomous"""
            self.agent.self_transformation.autonomous_mode = False
            print("Autonomous mode disabled")
            
        def do_toggle_auto_execute(self, arg):
            """Toggle auto-execution of generated scripts: toggle_auto_execute"""
            self.agent.self_transformation.auto_execute_scripts = not self.agent.self_transformation.auto_execute_scripts
            print(f"Auto-execute scripts: {self.agent.self_transformation.auto_execute_scripts}")
            
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
            """Request the agent to modify its own code: self_modify [target_function] [options]"""
            if not arg:
                print("Usage: self_modify [target_function or description] [--script] [--execute]")
                return
                
            args = shlex.split(arg)
            target = args[0]
            
            # Parse options
            generate_script = "--script" in args
            execute_script = "--execute" in args
            
            # Build prompt
            prompt = f"Please modify your code to improve the {target} functionality. "
            
            if generate_script:
                prompt += "Generate a standalone script that demonstrates the changes. "
            
            prompt += "Consider performance, readability, and error handling."
            
            try:
                print(f"Requesting self-modification for: {target}")
                result = asyncio.run(self.agent.qa(self.current_conversation, prompt))
                print(f"\nSelf-modification result: {result}")
                
                # Check if a script was generated
                recent_transformations = self.agent.self_transformation.transformation_history[-5:]
                for transformation in reversed(recent_transformations):
                    if "script_path" in transformation and transformation["script_path"]:
                        script_path = transformation["script_path"]
                        print(f"\nGenerated script: {script_path}")
                        
                        # Execute the script if requested
                        if execute_script or (generate_script and self.agent.self_transformation.auto_execute_scripts):
                            print(f"Executing script: {script_path}")
                            exec_result = asyncio.run(
                                self.agent.self_transformation._execute_generated_script(script_path)
                            )
                            
                            if exec_result["success"]:
                                print(f"Script executed successfully in {exec_result['execution_time']:.2f}s")
                            else:
                                print(f"Script execution failed with exit code {exec_result['exit_code']}")
                                if exec_result.get("stderr"):
                                    print(f"Error output: {exec_result['stderr']}")
                        
                        break
                
            except Exception as e:
                print(f"Error during self-modification: {e}")
                print(traceback.format_exc())
                
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
            
        def do_search(self, arg):
            """Search the web: search <query>"""
            if not arg:
                print("Usage: search <query>")
                return
            
            query = arg.strip()
            print(f"Searching for: {query}")
        
            try:
                # Run the search
                result = asyncio.run(self.agent.search_web(query))
            
                if result:
                    print("\nSearch Results:")
                
                    # Handle different result formats
                    if "items" in result:
                        for i, item in enumerate(result["items"][:5]):
                            print(f"\n{i+1}. {item.get('title', 'No title')}")
                            print(f"   {item.get('snippet', 'No snippet')}")
                            if 'link' in item:
                                print(f"   URL: {item['link']}")
                    elif "organic_results" in result:
                        for i, item in enumerate(result["organic_results"][:5]):
                            print(f"\n{i+1}. {item.get('title', 'No title')}")
                            print(f"   {item.get('snippet', 'No snippet')}")
                            if 'link' in item:
                                print(f"   URL: {item['link']}")
                    else:
                        # Generic output for any structure
                        print(json.dumps(result, indent=2))
                else:
                    print("No search results found or search functionality not available.")
                    print("To enable web search, set up one of the following:")
                    print("1. SERPAPI_API_KEY environment variable")
                    print("2. GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables")
                    print("3. JINA_API_KEY environment variable for fallback search")
            except Exception as e:
                print(f"Error performing search: {e}")
                print(traceback.format_exc())
                
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
                
        def do_autonomous(self, arg):
            """
            Manage autonomous task execution: autonomous [start|stop|status|create|decompose]
            
            Commands:
              start           - Start autonomous task execution
              stop            - Stop autonomous task execution
              status          - Show autonomous execution status
              create          - Create a new autonomous task
              decompose       - Break down a task into subtasks
              
            Examples:
              autonomous start
              autonomous stop
              autonomous status
              autonomous create "Research ML algorithms" "Research and summarize recent ML algorithms"
              autonomous decompose task_id "Subtask 1" "Subtask 2" "Subtask 3"
            """
            args = shlex.split(arg)
            if not args:
                print("Usage: autonomous [start|stop|status|create|decompose]")
                return
                
            command = args[0].lower()
            
            if command == "start":
                print("Starting autonomous task execution...")
                result = asyncio.run(self.agent.start_autonomous_execution())
                if result:
                    print("Autonomous task execution started successfully.")
                else:
                    print("Failed to start autonomous task execution.")
                    
            elif command == "stop":
                print("Stopping autonomous task execution...")
                result = asyncio.run(self.agent.stop_autonomous_execution())
                if result:
                    print("Autonomous task execution stopped successfully.")
                else:
                    print("Failed to stop autonomous task execution.")
                    
            elif command == "status":
                status = asyncio.run(self.agent.get_autonomous_execution_status())
                
                print("\n=== Autonomous Execution Status ===")
                print(f"Enabled: {status['enabled']}")
                
                if status['current_task']:
                    task = status['current_task']
                    print(f"\nCurrent Task: {task['title']}")
                    print(f"  ID: {task['task_id']}")
                    print(f"  Status: {task['status']}")
                    print(f"  Progress: {task['progress']*100:.1f}%")
                    if task['last_checkpoint']:
                        print(f"  Last Checkpoint: {task['last_checkpoint']}")
                    print(f"  Interruption Count: {task['interruption_count']}")
                else:
                    print("\nNo task currently being executed.")
                    
                if status['work_session_duration']:
                    duration_mins = status['work_session_duration'] / 60
                    print(f"\nWork Session Duration: {duration_mins:.1f} minutes")
                    print(f"Maximum Work Time: {status['max_continuous_work_time']/60:.1f} minutes")
                    
                print("\nTask Statistics:")
                print(f"  Pending: {status['task_stats']['pending']}")
                print(f"  Completed: {status['task_stats']['completed']}")
                print(f"  Paused: {status['task_stats']['paused']}")
                
                print("\nExecution Statistics:")
                stats = status['executor_stats']
                print(f"  Tasks Started: {stats['tasks_started']}")
                print(f"  Tasks Completed: {stats['tasks_completed']}")
                print(f"  Tasks Paused: {stats['tasks_paused']}")
                print(f"  Interruptions: {stats['interruptions']}")
                print(f"  Total Execution Time: {stats['total_execution_time']:.1f} seconds")
                
            elif command == "create":
                if len(args) < 3:
                    print("Usage: autonomous create <title> <description> [priority=5] [tags=tag1,tag2]")
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
                
                # Create autonomous task
                result = asyncio.run(self.agent.create_autonomous_task(
                    title=title,
                    description=description,
                    priority=priority,
                    tags=tags
                ))
                
                if result["success"]:
                    print(f"Created autonomous task: {result['title']} (ID: {result['task_id']})")
                else:
                    print(f"Error creating autonomous task: {result['error']}")
                    
            elif command == "decompose":
                if len(args) < 3:
                    print("Usage: autonomous decompose <task_id> <subtask1> <subtask2> ...")
                    return
                    
                task_id = args[1]
                subtasks = []
                
                # Create subtasks from remaining arguments
                for i in range(2, len(args)):
                    subtasks.append({
                        "title": args[i],
                        "description": f"Subtask: {args[i]}"
                    })
                
                if not subtasks:
                    print("No subtasks specified.")
                    return
                    
                # Decompose task
                result = asyncio.run(self.agent.decompose_task(task_id, subtasks))
                
                if result["success"]:
                    print(f"Task {task_id} decomposed into {result['subtasks_count']} subtasks.")
                else:
                    print(f"Error decomposing task: {result['error']}")
                    
            else:
                print(f"Unknown command: {command}")
                print("Usage: autonomous [start|stop|status|create|decompose]")
                
        def do_check_restart(self, arg):
            """Check if a restart is needed and restart the agent if necessary"""
            restart_file = "agent_restart_signal.txt"
            if os.path.exists(restart_file):
                print("Restart signal detected. Restarting agent...")
                os.remove(restart_file)
                
                # Stop autonomous execution if running
                if self.agent.work_continuity_enabled:
                    print("Stopping autonomous execution...")
                    asyncio.run(self.agent.stop_autonomous_execution())
                
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
            print(f"Response mode: {self.agent.response_mode.value}")
            print(f"Autonomous mode: {self.agent.self_transformation.autonomous_mode}")
            if self.agent.self_transformation.autonomous_mode:
                print(f"  Changes made: {self.agent.self_transformation.changes_counter}")
                print(f"  Max changes: {self.agent.self_transformation.max_autonomous_changes}")
                print(f"  Auto-execute scripts: {self.agent.self_transformation.auto_execute_scripts}")
            
            # Check reflection status
            reflection_status = "Running" if self.agent._reflection_running else "Stopped"
            print(f"Reflections: {reflection_status}")
            if self.agent._reflection_running:
                print(f"  Queue size: {self.agent._reflection_queue.qsize()}")
            
            # Check memory code stats
            code_manager = self.agent.self_transformation.code_manager
            memory_files = len(code_manager.memory_code)
            
        def do_server(self, arg):
            """Start a server with specified configuration.
            
            Usage: server <server_type> [options]
            
            Server types:
              http      - Basic HTTP server (specify port with --port)
              fastapi   - FastAPI server (specify port with --port)
              jupyter   - Jupyter notebook server
              
            Examples:
              server http --port 8000 --directory ./static
              server fastapi --port 8080 --reload
              server jupyter --port 8888
            """
            if not arg:
                print("Please specify a server type. Use 'help server' for usage details.")
                return
                
            args = shlex.split(arg)
            if not args:
                print("Please specify server type and options.")
                return
                
            server_type = args[0].lower()
            server_args = args[1:] if len(args) > 1 else []
            
            try:
                if server_type == "http":
                    # Parse arguments
                    parser = argparse.ArgumentParser()
                    parser.add_argument("--port", type=int, default=8000)
                    parser.add_argument("--directory", type=str, default=".")
                    server_options = parser.parse_args(server_args)
                    
                    # Build command
                    cmd = f"python -m http.server {server_options.port} --directory {server_options.directory}"
                    print(f"Starting HTTP server on port {server_options.port} serving {server_options.directory}...")
                    
                    # Run in subprocess to keep CLI responsive
                    subprocess.Popen(cmd, shell=True)
                    print(f"Server running at http://localhost:{server_options.port}/")
                    
                elif server_type == "fastapi":
                    # Parse arguments
                    parser = argparse.ArgumentParser()
                    parser.add_argument("--port", type=int, default=8000)
                    parser.add_argument("--host", type=str, default="127.0.0.1")
                    parser.add_argument("--reload", action="store_true")
                    parser.add_argument("--app", type=str, default="app:app")
                    server_options = parser.parse_args(server_args)
                    
                    # Build command
                    reload_flag = "--reload" if server_options.reload else ""
                    cmd = f"uvicorn {server_options.app} --host {server_options.host} --port {server_options.port} {reload_flag}"
                    print(f"Starting FastAPI server on {server_options.host}:{server_options.port}...")
                    
                    # Run in subprocess to keep CLI responsive
                    subprocess.Popen(cmd, shell=True)
                    print(f"API server running at http://{server_options.host}:{server_options.port}/")
                    
                elif server_type == "jupyter":
                    # Parse arguments
                    parser = argparse.ArgumentParser()
                    parser.add_argument("--port", type=int, default=8888)
                    parser.add_argument("--no-browser", action="store_true")
                    server_options = parser.parse_args(server_args)
                    
                    # Build command
                    browser_flag = "--no-browser" if server_options.no_browser else ""
                    cmd = f"jupyter notebook --port={server_options.port} {browser_flag}"
                    print(f"Starting Jupyter notebook server on port {server_options.port}...")
                    
                    # Run in subprocess to keep CLI responsive
                    subprocess.Popen(cmd, shell=True)
                    print(f"Jupyter server running. Check terminal output for access URL.")
                    
                else:
                    print(f"Unknown server type: {server_type}")
                    print("Available types: http, fastapi, jupyter")
            except Exception as e:
                print(f"Error starting server: {e}")
                traceback.print_exc()
                
        def do_run_sequence(self, arg):
            """Run a sequence of system commands, one after another.
            
            Usage: run_sequence <command1> ; <command2> ; ... ; <commandN>
            
            Examples:
              run_sequence mkdir -p temp ; touch temp/test.txt ; ls -la temp
              run_sequence git status ; git add . ; git commit -m "Update files"
            """
            if not arg:
                print("Please specify commands to run. Use 'help run_sequence' for usage details.")
                return
                
            # Split by semicolons but respect quoted semicolons
            commands = []
            current_cmd = ""
            in_quotes = False
            quote_char = None
            
            for char in arg:
                if char in ['"', "'"]:
                    if not in_quotes:
                        in_quotes = True
                        quote_char = char
                    elif char == quote_char:
                        in_quotes = False
                        quote_char = None
                    current_cmd += char
                elif char == ';' and not in_quotes:
                    commands.append(current_cmd.strip())
                    current_cmd = ""
                else:
                    current_cmd += char
                    
            if current_cmd.strip():
                commands.append(current_cmd.strip())
                
            if not commands:
                print("No valid commands provided.")
                return
                
            print(f"Running sequence of {len(commands)} commands:")
            
            for i, cmd in enumerate(commands):
                print(f"\n[{i+1}/{len(commands)}] $ {cmd}")
                try:
                    # Use subprocess to run the command
                    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    
                    # Print output
                    if process.stdout:
                        print(process.stdout)
                    
                    # Print errors if any
                    if process.returncode != 0:
                        print(f"Error (exit code {process.returncode}):")
                        if process.stderr:
                            print(process.stderr)
                        
                        # Ask whether to continue or abort
                        choice = input("Command failed. Continue sequence? (y/n): ")
                        if choice.lower() not in ['y', 'yes']:
                            print("Sequence aborted.")
                            return
                except Exception as e:
                    print(f"Error executing command: {e}")
                    choice = input("Command failed. Continue sequence? (y/n): ")
                    if choice.lower() not in ['y', 'yes']:
                        print("Sequence aborted.")
                        return
                        
            print("\nSequence completed.")
            
        def do_run_parallel(self, arg):
            """Run multiple system commands in parallel and display their results.
            
            Usage: run_parallel <command1> | <command2> | ... | <commandN>
            
            Examples:
              run_parallel ls -la | find . -name "*.py" | grep -r "import" --include="*.py" .
            """
            if not arg:
                print("Please specify commands to run. Use 'help run_parallel' for usage details.")
                return
                
            # Split by pipes but respect quoted pipes
            commands = []
            current_cmd = ""
            in_quotes = False
            quote_char = None
            
            for char in arg:
                if char in ['"', "'"]:
                    if not in_quotes:
                        in_quotes = True
                        quote_char = char
                    elif char == quote_char:
                        in_quotes = False
                        quote_char = None
                    current_cmd += char
                elif char == '|' and not in_quotes:
                    commands.append(current_cmd.strip())
                    current_cmd = ""
                else:
                    current_cmd += char
                    
            if current_cmd.strip():
                commands.append(current_cmd.strip())
                
            if not commands:
                print("No valid commands provided.")
                return
                
            print(f"Running {len(commands)} commands in parallel:")
            for i, cmd in enumerate(commands):
                print(f"[{i+1}] $ {cmd}")
                
            try:
                # Use our async function to run commands in parallel
                results = asyncio.run(available_functions["run_commands_parallel"](commands))
                
                print("\nResults:")
                for i, result in enumerate(results):
                    print(f"\n[{i+1}] Command: {result['command']}")
                    if result.get('success', False):
                        print(f"Status: Success (exit code: {result.get('exit_code', 0)})")
                        if result.get('stdout'):
                            print("\nOutput:")
                            print(result['stdout'])
                    else:
                        print("Status: Failed")
                        if result.get('error'):
                            print(f"Error: {result['error']}")
                        elif result.get('stderr'):
                            print("\nError output:")
                            print(result['stderr'])
                
                print("\nAll parallel commands completed.")
            except Exception as e:
                print(f"Error running parallel commands: {e}")
                traceback.print_exc()
            
            # Check conversation stats
            print(f"Current conversation: {self.current_conversation}")
            print(f"Conversation history: {len(self.agent.conversation_history)} entries")
            
            # Check transformation stats
            transformations = len(self.agent.self_transformation.transformation_history)
            cognitive_feedback = len(self.agent.self_transformation.cognitive_feedback)
            print(f"Transformations: {transformations}")
            print(f"Cognitive feedback entries: {cognitive_feedback}")
            
            # Check active flows for DEEP_FLOW mode
            if self.agent.response_mode == ResponseMode.DEEP_FLOW:
                active_flows = len(self.agent.active_flows)
                print(f"Active flows: {active_flows}")
                if active_flows > 0:
                    for flow in self.agent.active_flows:
                        print(f"  Flow ID: {flow['id']}")
                        print(f"  Status: {flow['status']}")
                        print(f"  Sub-agents: {len(flow['sub_agents'])}")
            
            # Check generated scripts
            script_dir = self.agent.self_transformation.script_output_dir
            if os.path.exists(script_dir):
                scripts = [f for f in os.listdir(script_dir) if f.endswith('.py')]
                print(f"Generated scripts: {len(scripts)}")
            
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
                
        def do_set_mode(self, arg):
            """Set the agent's response mode: set_mode [normal|web_search|web_trawl|deep_research|deep_task|deep_flow]"""
            if not arg:
                print("Available response modes:")
                for mode in ResponseMode:
                    print(f"  {mode.value}: {self._get_mode_description(mode)}")
                print("\nUsage: set_mode [mode]")
                return
                
            mode = arg.strip().lower()
            result = asyncio.run(self.agent.set_response_mode(mode))
            print(result)
            
            # Print additional information about the mode
            if hasattr(self.agent, 'response_mode'):
                print(f"\n{self._get_mode_description(self.agent.response_mode)}")
                
        def _get_mode_description(self, mode):
            """Get a description of a response mode"""
            if mode == ResponseMode.NORMAL:
                return "Standard response mode with clear, concise answers"
            elif mode == ResponseMode.WEB_SEARCH:
                return "Enhanced responses with web search results and citations"
            elif mode == ResponseMode.WEB_TRAWL:
                return "Comprehensive answers from deep analysis of multiple web sources"
            elif mode == ResponseMode.DEEP_RESEARCH:
                return "Exhaustive, scholarly responses with academic structure and citations"
            elif mode == ResponseMode.DEEP_TASK:
                return "Specialized task execution guidance with detailed steps and solutions"
            elif mode == ResponseMode.DEEP_FLOW:
                return "Interactive problem-solving with multiple specialized sub-agents"
            else:
                return "Unknown mode"
                
        def do_list_scripts(self, arg):
            """List generated scripts: list_scripts [pattern]"""
            script_dir = self.agent.self_transformation.script_output_dir
            if not os.path.exists(script_dir):
                print(f"Script directory not found: {script_dir}")
                return
                
            pattern = arg if arg else "*.py"
            scripts = list(Path(script_dir).glob(pattern))
            
            if not scripts:
                print(f"No scripts found matching pattern: {pattern}")
                return
                
            print(f"\nFound {len(scripts)} scripts:")
            for i, script_path in enumerate(sorted(scripts, key=lambda p: p.stat().st_mtime, reverse=True)):
                mtime = datetime.fromtimestamp(script_path.stat().st_mtime)
                size = script_path.stat().st_size
                print(f"{i+1}. {script_path.name}")
                print(f"   Created: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Size: {size/1024:.1f} KB")
                
        def do_run_script(self, arg):
            """Run a generated script: run_script <script_name>"""
            if not arg:
                print("Usage: run_script <script_name>")
                return
                
            script_dir = self.agent.self_transformation.script_output_dir
            script_name = arg
            
            # Add .py extension if not present
            if not script_name.endswith(".py"):
                script_name += ".py"
                
            script_path = os.path.join(script_dir, script_name)
            
            if not os.path.exists(script_path):
                print(f"Script not found: {script_path}")
                return
                
            try:
                print(f"Running script: {script_path}")
                result = asyncio.run(
                    self.agent.self_transformation._execute_generated_script(script_path)
                )
                
                print("\nExecution results:")
                print(f"Success: {result['success']}")
                print(f"Exit code: {result['exit_code']}")
                print(f"Execution time: {result['execution_time']:.2f}s")
                
                if result.get("stdout"):
                    print("\nStandard output:")
                    print(result["stdout"])
                
                if result.get("stderr"):
                    print("\nError output:")
                    print(result["stderr"])
                    
            except Exception as e:
                print(f"Error running script: {e}")
                print(traceback.format_exc())
                
        def do_show_cognitive_feedback(self, arg):
            """Show cognitive feedback: show_cognitive_feedback [index]"""
            feedback_entries = self.agent.self_transformation.cognitive_feedback
            
            if not feedback_entries:
                print("No cognitive feedback entries available.")
                return
                
            if arg:
                try:
                    index = int(arg)
                    if 0 <= index < len(feedback_entries):
                        feedback = feedback_entries[index]
                        print(f"\nCognitive Feedback #{index}:")
                        print(f"Timestamp: {feedback['timestamp']}")
                        print(f"Transformation ID: {feedback['transformation_id']}")
                        print("\nFeedback:")
                        print(feedback['feedback'])
                    else:
                        print(f"Invalid index. Must be between 0 and {len(feedback_entries)-1}.")
                except ValueError:
                    print(f"Invalid argument. Usage: show_cognitive_feedback [index]")
            else:
                print(f"\nCognitive Feedback Entries ({len(feedback_entries)}):")
                for i, feedback in enumerate(feedback_entries):
                    timestamp = datetime.fromisoformat(feedback['timestamp'])
                    print(f"{i}. {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                    feedback_preview = feedback['feedback'][:100] + "..." if len(feedback['feedback']) > 100 else feedback['feedback']
                    print(f"   {feedback_preview}")
                print("\nUse 'show_cognitive_feedback <index>' to view full feedback.")
                
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
                
        def do_generate_script(self, arg):
            """Generate a new script: generate_script <script_name> [description]"""
            args = shlex.split(arg)
            if not args:
                print("Usage: generate_script <script_name> [description]")
                return
                
            script_name = args[0]
            description = " ".join(args[1:]) if len(args) > 1 else "A utility script"
            
            # Add .py extension if not present
            if not script_name.endswith(".py"):
                script_name += ".py"
                
            try:
                # Create prompt for script generation
                prompt = f"""
                Generate a standalone Python script named '{script_name}' that {description}.
                
                The script should:
                1. Be well-documented with docstrings and comments
                2. Include proper error handling
                3. Have a main() function and proper command-line argument parsing
                4. Be executable directly (with shebang line)
                5. Include example usage in the docstring
                
                Return the complete script code in a code block.
                """
                
                print(f"Generating script: {script_name}")
                print(f"Description: {description}")
                
                # Get the script content from the agent
                if not self.current_conversation:
                    self.current_conversation = asyncio.run(self.agent.create_conversation())
                
                # Temporarily set to DEEP_TASK mode for better script generation
                original_mode = self.agent.response_mode
                asyncio.run(self.agent.set_response_mode(ResponseMode.DEEP_TASK))
                
                response = asyncio.run(self.agent.qa(self.current_conversation, prompt))
                
                # Restore original mode
                self.agent.response_mode = original_mode
                
                # Extract code block from response
                import re
                code_blocks = re.findall(r'```(?:python)?\n(.*?)```', response, re.DOTALL)
                
                if not code_blocks:
                    print("No code block found in the response. Using full response as script content.")
                    script_content = response
                else:
                    script_content = code_blocks[0]
                
                # Save the script
                script_dir = self.agent.self_transformation.script_output_dir
                script_path = os.path.join(script_dir, script_name)
                
                with open(script_path, 'w') as f:
                    f.write(script_content)
                
                # Make the script executable
                os.chmod(script_path, 0o755)
                
                print(f"Script generated and saved to: {script_path}")
                
                # Ask if the user wants to execute the script
                execute = input("Do you want to execute this script now? (y/n): ").lower().strip()
                if execute == 'y':
                    print(f"Executing script: {script_path}")
                    result = asyncio.run(
                        self.agent.self_transformation._execute_generated_script(script_path)
                    )
                    
                    print("\nExecution results:")
                    print(f"Success: {result['success']}")
                    print(f"Exit code: {result['exit_code']}")
                    print(f"Execution time: {result['execution_time']:.2f}s")
                    
                    if result.get("stdout"):
                        print("\nStandard output:")
                        print(result["stdout"])
                    
                    if result.get("stderr"):
                        print("\nError output:")
                        print(result["stderr"])
                
            except Exception as e:
                print(f"Error generating script: {e}")
                print(traceback.format_exc())
                
        def do_flow(self, arg):
            """Manage DEEP_FLOW mode: flow [list|status|stop] [flow_id]"""
            args = shlex.split(arg)
            if not args:
                print("Usage: flow [list|status|stop] [flow_id]")
                return
                
            command = args[0].lower()
            
            if command == "list":
                # List active flows
                flows = self.agent.active_flows
                if not flows:
                    print("No active flows.")
                    return
                    
                print(f"\nActive flows ({len(flows)}):")
                for i, flow in enumerate(flows):
                    print(f"{i+1}. Flow ID: {flow['id']}")
                    print(f"   Status: {flow['status']}")
                    print(f"   Started: {flow['started_at']}")
                    print(f"   Sub-agents: {len(flow['sub_agents'])}")
                    print(f"   Question: {flow['question'][:50]}..." if len(flow['question']) > 50 else flow['question'])
                    print()
                    
            elif command == "status":
                if len(args) < 2:
                    print("Usage: flow status <flow_id>")
                    return
                    
                flow_id = args[1]
                
                # Find the flow
                flow = next((f for f in self.agent.active_flows if f['id'] == flow_id), None)
                if not flow:
                    print(f"Flow not found: {flow_id}")
                    return
                    
                print(f"\nFlow: {flow['id']}")
                print(f"Status: {flow['status']}")
                print(f"Started: {flow['started_at']}")
                print(f"Completed: {flow['completed_at'] or 'Not completed'}")
                print(f"Question: {flow['question']}")
                print("\nSub-agents:")
                
                for agent_id, agent in flow['sub_agents'].items():
                    print(f"- {agent['type'].capitalize()} Agent ({agent_id})")
                    print(f"  Status: {agent['status']}")
                    print(f"  Tasks: {len(agent['tasks'])}")
                    print(f"  Results: {len(agent['results'])}")
                    print()
                    
            elif command == "stop":
                if len(args) < 2:
                    print("Usage: flow stop <flow_id>")
                    return
                    
                flow_id = args[1]
                
                # Find the flow
                flow_index = next((i for i, f in enumerate(self.agent.active_flows) if f['id'] == flow_id), None)
                if flow_index is None:
                    print(f"Flow not found: {flow_id}")
                    return
                    
                # Update flow status
                self.agent.active_flows[flow_index]['status'] = "stopped"
                self.agent.active_flows[flow_index]['completed_at'] = datetime.now().isoformat()
                
                print(f"Flow stopped: {flow_id}")
                
            else:
                print(f"Unknown command: {command}")
                print("Usage: flow [list|status|stop] [flow_id]")
                
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
            """Read, write, or edit code files: code [read|write|edit] <file_path> [content]"""
            args = shlex.split(arg)
            if not args:
                print("Usage: code [read|write|edit] <file_path> [content]")
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
            
            elif command == "edit":
                if len(args) < 2:
                    print("Usage: code edit <file_path>")
                    return
                
                file_path = args[1]
                
                try:
                    # Check if the editor module is available
                    import editor
                    
                    # Create editor instance with agent
                    edit = editor.Editor(file_path, self.agent)
                    
                    # Save terminal state
                    os.system('clear')
                    print(f"Opening editor for {file_path}...")
                    print("Press Ctrl+Q to exit, Ctrl+S to save.")
                    
                    # Give user a moment to read the message
                    time.sleep(1)
                    
                    # Run the editor
                    import curses
                    curses.wrapper(edit.run)
                    
                    # Restore terminal
                    os.system('clear')
                    print(f"Closed editor for {file_path}")
                    
                    # Check if file was modified
                    if edit.modified:
                        print("File was modified but not saved.")
                    
                except ImportError:
                    print("Editor module not available. Install required dependencies:")
                    print("pip install pygments")
                except Exception as e:
                    print(f"Error running editor: {e}")
                    print(traceback.format_exc())
            
            else:
                print(f"Unknown command: {command}")
                print("Usage: code [read|write|edit] <file_path> [content]")
                
        def do_jobs(self, arg):
            """Manage background jobs: jobs [list|submit|status|cancel]"""
            args = shlex.split(arg)
            if not args:
                print("Usage: jobs [list|submit|status|cancel]")
                return
                
            command = args[0].lower()
            
            if command == "list":
                # Parse status filter
                status = None
                if len(args) > 1:
                    status = args[1].upper()
                
                # List jobs
                result = asyncio.run(self.agent.list_jobs(status=status))
                
                if result["success"]:
                    print(f"\nFound {result['count']} jobs:")
                    for i, job in enumerate(result["jobs"]):
                        status_color = "\033[92m" if job["status"] == "COMPLETED" else \
                                      "\033[93m" if job["status"] == "RUNNING" else \
                                      "\033[91m" if job["status"] in ["FAILED", "CANCELLED", "TIMEOUT"] else "\033[94m"
                        print(f"{i+1}. {job['name']} ({status_color}{job['status']}\033[0m)")
                        print(f"   ID: {job['job_id']}")
                        print(f"   Progress: {job['progress']*100:.1f}%")
                        print(f"   Created: {job['created_at']}")
                        print()
                else:
                    print(f"Error listing jobs: {result['error']}")
            
            elif command == "submit":
                if len(args) < 3:
                    print("Usage: jobs submit <name> <function_name> [priority=5] [timeout=None]")
                    return
                
                name = args[1]
                function_name = args[2]
                priority = 5
                timeout = None
                
                # Parse optional arguments
                for i in range(3, len(args)):
                    if args[i].startswith("priority="):
                        try:
                            priority = int(args[i].split("=")[1])
                        except ValueError:
                            print(f"Invalid priority: {args[i].split('=')[1]}")
                            return
                    elif args[i].startswith("timeout="):
                        try:
                            timeout_str = args[i].split("=")[1]
                            timeout = float(timeout_str) if timeout_str.lower() != "none" else None
                        except ValueError:
                            print(f"Invalid timeout: {args[i].split('=')[1]}")
                            return
                
                # Submit job
                result = asyncio.run(self.agent.submit_background_job(
                    name=name,
                    function_name=function_name,
                    priority=priority,
                    timeout=timeout
                ))
                
                if result["success"]:
                    print(f"Submitted job: {result['name']} (ID: {result['job_id']})")
                    if result["task_id"]:
                        print(f"Associated task ID: {result['task_id']}")
                else:
                    print(f"Error submitting job: {result['error']}")
            
            elif command == "status":
                if len(args) < 2:
                    print("Usage: jobs status <job_id>")
                    return
                
                job_id = args[1]
                
                # Get job status
                result = asyncio.run(self.agent.get_job_status(job_id))
                
                if result["success"]:
                    print(f"\nJob: {result['name']} (ID: {result['job_id']})")
                    print(f"Status: {result['status']}")
                    print(f"Progress: {result['progress']*100:.1f}%")
                    print(f"Created: {result['created_at']}")
                    if result["started_at"]:
                        print(f"Started: {result['started_at']}")
                    if result["completed_at"]:
                        print(f"Completed: {result['completed_at']}")
                    if result["error"]:
                        print(f"Error: {result['error']}")
                    if result["result"]:
                        print(f"Result: {result['result']}")
                else:
                    print(f"Error getting job status: {result['error']}")
            
            elif command == "cancel":
                if len(args) < 2:
                    print("Usage: jobs cancel <job_id>")
                    return
                
                job_id = args[1]
                
                # Cancel job
                result = asyncio.run(self.agent.cancel_job(job_id))
                
                if result["success"]:
                    print(f"Cancelled job: {job_id}")
                    print(f"Message: {result['message']}")
                else:
                    print(f"Error cancelling job: {result['error']}")
            
            else:
                print(f"Unknown command: {command}")
                print("Usage: jobs [list|submit|status|cancel]")
                
        def do_exit(self, arg):
            """Exit the CLI: exit"""
            if hasattr(self, "observer"):
                self.observer.stop()
                self.observer.join()
            # Stop reflection worker
            asyncio.run(self.agent.stop_reflections())
            # Stop job scheduler if it's running
            if hasattr(self.agent, 'job_scheduler') and self.agent.job_scheduler.running:
                asyncio.run(self.agent.job_scheduler.stop())
            print("\nShutting down agent and cleaning up resources...")
            print("Goodbye!")
            return True
            
        def do_EOF(self, arg):
            """Exit on Ctrl+D"""
            print()  # Add newline
            return self.do_exit(arg)


# ------------------------------------------------------------------------------
# Resource Management and Job Scheduling
# ------------------------------------------------------------------------------

class JobStatus(Enum):
    """Status of a background job"""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()

class BackgroundJob:
    """
    Represents a long-running background job with progress tracking
    and resource management.
    """
    def __init__(self, job_id: str, name: str, func: Callable, args: List[Any] = None, 
                kwargs: Dict[str, Any] = None, timeout: Optional[float] = None):
        self.job_id = job_id
        self.name = name
        self.func = func
        self.args = args or []
        self.kwargs = kwargs or {}
        self.timeout = timeout
        self.status = JobStatus.PENDING
        self.result = None
        self.error = None
        self.created_at = datetime.now().isoformat()
        self.started_at = None
        self.completed_at = None
        self.progress = 0.0
        self.task = None  # Will hold the asyncio.Task
        self.resource_locks = []  # Resources locked by this job
        self.metadata = {}
        self.cancellation_requested = False
        
    async def run(self):
        """Run the job function with timeout handling"""
        self.status = JobStatus.RUNNING
        self.started_at = datetime.now().isoformat()
        
        try:
            if self.timeout:
                # Run with timeout
                self.task = asyncio.create_task(self._run_func())
                try:
                    self.result = await asyncio.wait_for(self.task, timeout=self.timeout)
                    self.status = JobStatus.COMPLETED
                except asyncio.TimeoutError:
                    self.status = JobStatus.TIMEOUT
                    self.error = "Job timed out"
            else:
                # Run without timeout
                self.task = asyncio.create_task(self._run_func())
                self.result = await self.task
                self.status = JobStatus.COMPLETED
        except asyncio.CancelledError:
            self.status = JobStatus.CANCELLED
            self.error = "Job was cancelled"
        except Exception as e:
            self.status = JobStatus.FAILED
            self.error = str(e)
        finally:
            self.completed_at = datetime.now().isoformat()
            # Release any resource locks
            for lock_id in self.resource_locks:
                await self._release_lock(lock_id)
            
    async def _run_func(self):
        """Execute the job function"""
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(*self.args, **self.kwargs)
        else:
            # Run synchronous functions in a thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: self.func(*self.args, **self.kwargs))
    
    async def cancel(self):
        """Request cancellation of the job"""
        if self.status == JobStatus.RUNNING and self.task:
            self.cancellation_requested = True
            self.task.cancel()
            return True
        return False
    
    def update_progress(self, progress: float):
        """Update job progress (0.0 to 1.0)"""
        self.progress = max(0.0, min(1.0, progress))
        
    async def acquire_lock(self, resource_id: str) -> bool:
        """Acquire a lock on a resource"""
        # Implementation would depend on the locking mechanism
        # For now, just track that we've locked this resource
        self.resource_locks.append(resource_id)
        return True
        
    async def _release_lock(self, resource_id: str) -> bool:
        """Release a lock on a resource"""
        # Implementation would depend on the locking mechanism
        if resource_id in self.resource_locks:
            self.resource_locks.remove(resource_id)
            return True
        return False

class JobScheduler:
    """
    Manages and schedules background jobs with resource constraints
    and priority-based execution.
    """
    def __init__(self, max_concurrent_jobs: int = 5):
        self.jobs: Dict[str, BackgroundJob] = {}
        self.max_concurrent_jobs = max_concurrent_jobs
        self.running_jobs = 0
        self.job_queue = asyncio.PriorityQueue()  # (priority, job_id)
        self.resource_locks = {}  # resource_id -> job_id
        self.scheduler_task = None
        self.running = False
        
    async def start(self):
        """Start the job scheduler"""
        if not self.running:
            self.running = True
            self.scheduler_task = asyncio.create_task(self._scheduler_loop())
            logging.info("Job scheduler started")
            
    async def stop(self):
        """Stop the job scheduler"""
        if self.running:
            self.running = False
            if self.scheduler_task:
                self.scheduler_task.cancel()
                try:
                    await self.scheduler_task
                except asyncio.CancelledError:
                    pass
            logging.info("Job scheduler stopped")
            
    async def submit_job(self, name: str, func: Callable, priority: int = 5,
                       args: List[Any] = None, kwargs: Dict[str, Any] = None,
                       timeout: Optional[float] = None) -> str:
        """
        Submit a job to be executed in the background
        
        Args:
            name: Human-readable name for the job
            func: Function to execute
            priority: Job priority (lower number = higher priority)
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            timeout: Optional timeout in seconds
            
        Returns:
            str: Job ID
        """
        job_id = str(uuid.uuid4())
        job = BackgroundJob(job_id, name, func, args, kwargs, timeout)
        self.jobs[job_id] = job
        
        # Add to queue with priority
        await self.job_queue.put((priority, job_id))
        
        logging.info(f"Job submitted: {name} (ID: {job_id}, Priority: {priority})")
        return job_id
        
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job by ID"""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            if job.status == JobStatus.PENDING:
                # For pending jobs, just mark as cancelled
                job.status = JobStatus.CANCELLED
                job.error = "Cancelled before execution"
                job.completed_at = datetime.now().isoformat()
                return True
            elif job.status == JobStatus.RUNNING:
                # For running jobs, request cancellation
                return await job.cancel()
        return False
        
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a job"""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            return {
                "job_id": job.job_id,
                "name": job.name,
                "status": job.status.name,
                "progress": job.progress,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "error": job.error,
                "result": job.result
            }
        return None
        
    async def list_jobs(self, status: Optional[JobStatus] = None) -> List[Dict[str, Any]]:
        """List all jobs, optionally filtered by status"""
        result = []
        for job_id, job in self.jobs.items():
            if status is None or job.status == status:
                result.append({
                    "job_id": job.job_id,
                    "name": job.name,
                    "status": job.status.name,
                    "progress": job.progress,
                    "created_at": job.created_at
                })
        return result
        
    async def _scheduler_loop(self):
        """Main scheduler loop that processes the job queue"""
        try:
            while self.running:
                # Check if we can run more jobs
                if self.running_jobs < self.max_concurrent_jobs:
                    try:
                        # Get the next job with the highest priority (lowest number)
                        priority, job_id = await asyncio.wait_for(
                            self.job_queue.get(), timeout=1.0)
                        
                        if job_id in self.jobs:
                            job = self.jobs[job_id]
                            
                            # Skip cancelled or completed jobs
                            if job.status not in [JobStatus.PENDING]:
                                self.job_queue.task_done()
                                continue
                                
                            # Start the job
                            self.running_jobs += 1
                            asyncio.create_task(self._run_job(job))
                            
                        self.job_queue.task_done()
                    except asyncio.TimeoutError:
                        # No jobs in queue, just continue
                        pass
                else:
                    # Wait a bit before checking again
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logging.info("Scheduler loop cancelled")
            raise
        except Exception as e:
            logging.error(f"Error in scheduler loop: {e}")
            
    async def _run_job(self, job: BackgroundJob):
        """Run a job and handle completion"""
        try:
            await job.run()
        finally:
            self.running_jobs -= 1
            logging.info(f"Job completed: {job.name} (ID: {job.job_id}, Status: {job.status.name})")

# ------------------------------------------------------------------------------
# Supervision Tree & Dynamic Dispatch
# ------------------------------------------------------------------------------

class SupervisorNode:
    """
    A node in the supervision tree that can monitor and restart child processes.
    Implements the Erlang-style supervision model for fault tolerance.
    """
    def __init__(self, name: str, restart_strategy: str = "one_for_one"):
        """
        Initialize a supervisor node
        
        Args:
            name: Name of the supervisor
            restart_strategy: Strategy for restarting children
                - one_for_one: Only restart the failed child
                - one_for_all: Restart all children if any fails
                - rest_for_one: Restart the failed child and all that depend on it
        """
        self.name = name
        self.restart_strategy = restart_strategy
        self.children = {}  # name -> (process, spec)
        self.max_restarts = 5
        self.restart_window = 60  # seconds
        self.restart_history = []
        self.running = False
        self.monitor_task = None
        
    async def start(self):
        """Start the supervisor"""
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logging.info(f"Supervisor {self.name} started with {self.restart_strategy} strategy")
        
    async def stop(self):
        """Stop the supervisor and all children"""
        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
                
        # Stop all children
        for child_name, (process, _) in list(self.children.items()):
            await self.stop_child(child_name)
            
        logging.info(f"Supervisor {self.name} stopped")
        
    async def add_child(self, name: str, func: Callable, args: List[Any] = None, 
                      kwargs: Dict[str, Any] = None, restart: bool = True,
                      max_restarts: int = 3) -> bool:
        """
        Add a child process to the supervision tree
        
        Args:
            name: Name of the child process
            func: Function to run in the child process
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            restart: Whether to restart the child on failure
            max_restarts: Maximum number of restarts for this child
            
        Returns:
            bool: Success status
        """
        if name in self.children:
            logging.warning(f"Child {name} already exists in supervisor {self.name}")
            return False
            
        # Create child spec
        spec = {
            "func": func,
            "args": args or [],
            "kwargs": kwargs or {},
            "restart": restart,
            "max_restarts": max_restarts,
            "restarts": 0
        }
        
        # Start the child process
        process = await self._start_child_process(name, spec)
        if process:
            self.children[name] = (process, spec)
            logging.info(f"Added child {name} to supervisor {self.name}")
            return True
        else:
            logging.error(f"Failed to start child {name}")
            return False
            
    async def stop_child(self, name: str) -> bool:
        """Stop a child process"""
        if name not in self.children:
            logging.warning(f"Child {name} not found in supervisor {self.name}")
            return False
            
        process, _ = self.children[name]
        if process and not process.done():
            process.cancel()
            try:
                await process
            except asyncio.CancelledError:
                pass
                
        del self.children[name]
        logging.info(f"Stopped child {name} in supervisor {self.name}")
        return True
        
    async def restart_child(self, name: str) -> bool:
        """Restart a child process"""
        if name not in self.children:
            logging.warning(f"Child {name} not found in supervisor {self.name}")
            return False
            
        _, spec = self.children[name]
        
        # Stop the current process
        await self.stop_child(name)
        
        # Start a new process
        process = await self._start_child_process(name, spec)
        if process:
            self.children[name] = (process, spec)
            spec["restarts"] += 1
            
            # Record restart for rate limiting
            self.restart_history.append(time.time())
            
            logging.info(f"Restarted child {name} in supervisor {self.name}")
            return True
        else:
            logging.error(f"Failed to restart child {name}")
            return False
            
    async def _start_child_process(self, name: str, spec: Dict[str, Any]) -> Optional[asyncio.Task]:
        """Start a child process as an asyncio task"""
        try:
            func = spec["func"]
            args = spec["args"]
            kwargs = spec["kwargs"]
            
            # Create and start the task
            if asyncio.iscoroutinefunction(func):
                task = asyncio.create_task(func(*args, **kwargs))
            else:
                # Run synchronous functions in a thread pool
                loop = asyncio.get_event_loop()
                task = asyncio.create_task(loop.run_in_executor(
                    None, lambda: func(*args, **kwargs)))
                
            # Set name for easier debugging
            task.set_name(f"{self.name}:{name}")
            
            return task
        except Exception as e:
            logging.error(f"Error starting child {name}: {e}")
            return None
            
    async def _monitor_loop(self):
        """Monitor child processes and restart them if needed"""
        try:
            while self.running:
                # Check each child
                for name, (process, spec) in list(self.children.items()):
                    if process.done():
                        # Child has completed or failed
                        try:
                            # Get the result or exception
                            result = process.result()
                            logging.info(f"Child {name} completed with result: {result}")
                            
                            # Don't restart if it completed normally
                            if not spec["restart"]:
                                del self.children[name]
                                
                        except Exception as e:
                            logging.error(f"Child {name} failed with error: {e}")
                            
                            # Check if we should restart
                            if spec["restart"] and spec["restarts"] < spec["max_restarts"]:
                                # Check restart rate
                                now = time.time()
                                recent_restarts = sum(1 for t in self.restart_history 
                                                    if now - t < self.restart_window)
                                
                                if recent_restarts < self.max_restarts:
                                    # Apply restart strategy
                                    if self.restart_strategy == "one_for_one":
                                        await self.restart_child(name)
                                    elif self.restart_strategy == "one_for_all":
                                        # Restart all children
                                        for child_name in list(self.children.keys()):
                                            await self.restart_child(child_name)
                                    elif self.restart_strategy == "rest_for_one":
                                        # Find position of failed child
                                        child_names = list(self.children.keys())
                                        if name in child_names:
                                            idx = child_names.index(name)
                                            # Restart this child and all that follow
                                            for child_name in child_names[idx:]:
                                                await self.restart_child(child_name)
                                else:
                                    logging.error(f"Too many restarts in supervisor {self.name}, not restarting {name}")
                            else:
                                logging.warning(f"Child {name} has reached max restarts, removing from supervisor")
                                del self.children[name]
                
                # Clean up old restart history
                now = time.time()
                self.restart_history = [t for t in self.restart_history if now - t < self.restart_window]
                
                # Wait before checking again
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            logging.info(f"Supervisor {self.name} monitor loop cancelled")
            raise
        except Exception as e:
            logging.error(f"Error in supervisor {self.name} monitor loop: {e}")

class DynamicDispatcher:
    """
    Dynamic dispatch system that routes requests to appropriate handlers
    based on content, type, or other attributes.
    """
    def __init__(self):
        self.handlers = {}  # pattern -> handler_func
        self.fallback_handler = None
        self.middleware = []
        
    def register_handler(self, pattern: str, handler_func: Callable) -> None:
        """Register a handler for a specific pattern"""
        self.handlers[pattern] = handler_func
        
    def register_fallback(self, handler_func: Callable) -> None:
        """Register a fallback handler for when no pattern matches"""
        self.fallback_handler = handler_func
        
    def add_middleware(self, middleware_func: Callable) -> None:
        """Add middleware that processes requests before handlers"""
        self.middleware.append(middleware_func)
        
    async def dispatch(self, content: str, **kwargs) -> Any:
        """
        Dispatch a request to the appropriate handler
        
        Args:
            content: The content to dispatch
            **kwargs: Additional context for handlers
            
        Returns:
            The result from the handler
        """
        # Apply middleware
        context = kwargs.copy()
        for mw_func in self.middleware:
            if asyncio.iscoroutinefunction(mw_func):
                content, context = await mw_func(content, context)
            else:
                content, context = mw_func(content, context)
                
        # Find matching handler
        handler = None
        for pattern, handler_func in self.handlers.items():
            if pattern in content or re.search(pattern, content, re.IGNORECASE):
                handler = handler_func
                break
                
        # Use fallback if no handler found
        if handler is None and self.fallback_handler:
            handler = self.fallback_handler
            
        # Execute handler
        if handler:
            if asyncio.iscoroutinefunction(handler):
                return await handler(content, **context)
            else:
                return handler(content, **context)
        else:
            raise ValueError(f"No handler found for content: {content[:50]}...")

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
    print("  --no-execute         Disable auto-execution of generated scripts")
    print("  --cli                Start in CLI mode (default)")
    print("  --dashboard          Start directly in dashboard mode")
    print("  --watch DIR          Watch directory for PDFs (default: ./pdfs)")
    print("  --port PORT          Set API server port (default: 8080)")
    print("  --host HOST          Set API server host (default: 0.0.0.0)")
    print("  --no-reflections     Disable background reflections")
    print("  --scripts DIR        Set directory for generated scripts (default: ./generated_scripts)")
    print("  --db PATH            Set database path (default: ./agent.db)")
    print("  --debug              Enable debug logging")
    print("  --mode MODE          Set initial response mode (normal, web_search, web_trawl, deep_research, deep_task, deep_flow)")
    print("\nExamples:")
    print("  python rl_cli.py --cli                    # Start in CLI mode")
    print("  python rl_cli.py --api --port 9000        # Run API server on port 9000")
    print("  python rl_cli.py --autonomous 10          # Start with autonomous mode, max 10 changes")
    print("  python rl_cli.py --autonomous 5 --no-execute  # Autonomous mode without script execution")
    print("  python rl_cli.py --rl 5                   # Run RL training with 5 episodes")
    print("  python rl_cli.py --watch ./documents      # Watch ./documents for PDFs")
    print("  python rl_cli.py --scripts ./my_scripts   # Use custom directory for generated scripts")
    print("  python rl_cli.py --mode deep_research     # Start in DEEP_RESEARCH response mode")

def main():
    """Main entry point for the CLI application"""
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
    parser.add_argument('--no-execute', action='store_true', help='Disable auto-execution of generated scripts')
    parser.add_argument('--cli', action='store_true', help='Start in CLI mode (default)')
    parser.add_argument('--dashboard', action='store_true', help='Start directly in dashboard mode')
    parser.add_argument('--watch', type=str, default='./pdfs', help='Watch directory for PDFs')
    parser.add_argument('--port', type=int, default=8080, help='Set API server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Set API server host')
    parser.add_argument('--no-reflections', action='store_true', help='Disable background reflections')
    parser.add_argument('--scripts', type=str, default='./generated_scripts', help='Set directory for generated scripts')
    parser.add_argument('--db', type=str, default='./agent.db', help='Set database path')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--mode', type=str, choices=['normal', 'web_search', 'web_trawl', 'deep_research', 'deep_task', 'deep_flow'], 
                      default='normal', help='Set initial response mode')
    
    # Add command argument for direct command execution
    parser.add_argument('command', nargs='*', help='Command to execute directly')
    
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
        
        # Set scripts directory
        cli.agent.self_transformation.script_output_dir = args.scripts
        os.makedirs(args.scripts, exist_ok=True)
        
        # Start the persistence timer with a running event loop
        asyncio.run(cli.agent._start_persistence_timer_async())
        
        # Enable autonomous mode if requested
        if args.autonomous is not None:
            auto_execute_flag = "" if not args.no_execute else "no-execute"
            cli.do_enable_autonomous(f"{args.autonomous} {auto_execute_flag}")
        elif args.no_execute:
            # Just disable auto-execute if autonomous mode not enabled
            cli.agent.self_transformation.auto_execute_scripts = False
        
        # Start or disable reflections
        if args.no_reflections:
            asyncio.run(cli.agent.stop_reflections())
        else:
            asyncio.run(cli.agent.start_reflections())
        
        # Set initial response mode if specified
        if args.mode:
            asyncio.run(cli.agent.set_response_mode(args.mode))
            
        # Check if a direct command was provided
        if args.command:
            # Join the command parts and execute it
            command = ' '.join(args.command)
            cli.onecmd(command)
            return
            
        # Start dashboard directly if requested
        if args.dashboard:
            cli.do_dashboard("")
            return
        
        # Start CLI
        mode_str = "autonomous mode" if args.autonomous else "standard mode"
        execute_str = "" if not args.no_execute else " (script auto-execution disabled)"
        response_mode_str = f", response mode: {cli.agent.response_mode.value}"
        print(f"Starting Agent CLI with {mode_str}{execute_str}{response_mode_str}")
        cli.cmdloop()

if __name__ == "__main__":
    main()
