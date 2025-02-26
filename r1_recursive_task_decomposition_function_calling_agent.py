#!/usr/bin/env python3
"""
A "do-anything" R1-style agent with:
 - Priority-based task queue
 - Recursive decomposition into subtasks
 - Concurrent execution
 - Memory of completed tasks
 - Example usage of <function_call> do_anything
 - Detailed docstrings and logging for clarity

DISCLAIMER:
This code is for demonstration and testing in a secure environment.
It is unsafe by design: if the agent decides to run destructive code,
it will do so. The developer is responsible for running it securely.
"""

import os
import sys
import re
import json
import time
import queue
import heapq
import logging
import threading
import traceback
import subprocess
import requests
import importlib
import importlib.util
import ast
import hashlib
import shutil
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from typing import Any, Dict, List, Optional, Tuple, Callable, Union, AsyncGenerator
from datetime import datetime, timedelta
from together import Together
from collections import defaultdict, deque

###############################################################################
# GLOBAL CONFIG / LOGGING SETUP
###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("DoAnythingAgent")

###############################################################################
# DATA STRUCTURES FOR TASK MANAGEMENT
###############################################################################

class Task:
    """
    A class representing a single unit of work.

    Attributes:
        task_id (int): Unique identifier for the task.
        priority (int): Lower numbers = higher priority in standard usage.
        description (str): Human-readable text describing the task.
        status (str): Current status: 'PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED', 'TIMEOUT', etc.
        parent_id (Optional[int]): The ID of a parent task, if any. This enables
            recursive decomposition, where subtasks can track their parent's ID.
        result (Any): Output or result data from the task, if any.
        created_at (datetime): When the task was created.
        started_at (Optional[datetime]): When the task started execution.
        completed_at (Optional[datetime]): When the task completed execution.
        progress (float): Progress indicator from 0.0 to 1.0.
        timeout_seconds (Optional[int]): Maximum time allowed for task execution.
    """
    def __init__(self, task_id: int, priority: int, description: str, 
                 parent_id: Optional[int] = None, timeout_seconds: Optional[int] = None):
        self.task_id = task_id
        self.priority = priority
        self.description = description
        self.status = "PENDING"
        self.parent_id = parent_id
        self.result = None
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.progress = 0.0
        self.timeout_seconds = timeout_seconds
        self.error_message = None

    def start(self) -> None:
        """Mark the task as started."""
        self.status = "IN_PROGRESS"
        self.started_at = datetime.now()

    def complete(self, result: Any = None) -> None:
        """Mark the task as completed with an optional result."""
        self.status = "COMPLETED"
        self.completed_at = datetime.now()
        self.progress = 1.0
        if result is not None:
            self.result = result

    def fail(self, error_message: str) -> None:
        """Mark the task as failed with an error message."""
        self.status = "FAILED"
        self.completed_at = datetime.now()
        self.error_message = error_message

    def timeout(self) -> None:
        """Mark the task as timed out."""
        self.status = "TIMEOUT"
        self.completed_at = datetime.now()
        self.error_message = "Task execution exceeded timeout limit"

    def update_progress(self, progress: float) -> None:
        """Update the progress indicator (0.0 to 1.0)."""
        self.progress = max(0.0, min(1.0, progress))

    def is_timed_out(self) -> bool:
        """Check if the task has exceeded its timeout."""
        if not self.timeout_seconds or not self.started_at:
            return False
        elapsed = datetime.now() - self.started_at
        return elapsed.total_seconds() > self.timeout_seconds

    def get_runtime(self) -> Optional[float]:
        """Get the runtime in seconds, or None if not started."""
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()

    def __lt__(self, other: "Task") -> bool:
        """
        Allows tasks to be compared based on priority for use in a priority queue.
        A lower priority value means the task is more urgent.
        """
        return self.priority < other.priority

    def __repr__(self) -> str:
        status_info = f"{self.status}"
        if self.progress > 0 and self.progress < 1 and self.status == "IN_PROGRESS":
            status_info += f" ({self.progress:.0%})"
        return f"Task(id={self.task_id}, priority={self.priority}, status={status_info}, desc={self.description[:30]})"


class TaskMemoryStore:
    """
    Maintains storage of tasks in memory with advanced features:
    - Persistent storage options
    - Task relationship tracking
    - Historical performance metrics
    - Pattern recognition for similar tasks
    - Semantic search capabilities

    In real production code, you could store these in a DB or persistent store.
    """
    def __init__(self, persistent_storage: bool = False) -> None:
        self._tasks: Dict[int, Task] = {}
        self._lock = threading.Lock()
        self._next_id = 1
        self._task_embeddings: Dict[int, List[float]] = {}
        self._task_relationships: Dict[int, List[int]] = defaultdict(list)
        self._task_performance_history: Dict[str, List[float]] = defaultdict(list)
        self._persistent_storage = persistent_storage
        self._task_patterns = defaultdict(int)

    def add_task(self, task: Task) -> None:
        """
        Add a new Task object to memory with advanced tracking.
        """
        with self._lock:
            if task.task_id in self._tasks:
                logger.warning(f"Task ID {task.task_id} already exists!")
            self._tasks[task.task_id] = task
            
            # Track task patterns for optimization
            task_type = self._extract_task_type(task.description)
            self._task_patterns[task_type] += 1
            
            # If persistent storage is enabled, save to disk
            if self._persistent_storage:
                self._save_task_to_disk(task)

    def create_task(self, priority: int, description: str, 
                   parent_id: Optional[int] = None, 
                   timeout_seconds: Optional[int] = None) -> Task:
        """
        Create and store a new task, returning the Task object.
        """
        with self._lock:
            task_id = self._next_id
            self._next_id += 1
            task = Task(
                task_id=task_id,
                priority=priority,
                description=description,
                parent_id=parent_id,
                timeout_seconds=timeout_seconds
            )
            self._tasks[task_id] = task
            return task

    def get_task(self, task_id: int) -> Optional[Task]:
        """
        Fetch a Task by its ID.
        """
        with self._lock:
            return self._tasks.get(task_id)

    def get_subtasks(self, parent_id: int) -> List[Task]:
        """
        Get all subtasks for a given parent task.
        """
        with self._lock:
            return [task for task in self._tasks.values() if task.parent_id == parent_id]

    def update_task_status(self, task_id: int, status: str) -> None:
        """
        Change the status of a task in memory.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = status
                if status == "IN_PROGRESS" and not task.started_at:
                    task.started_at = datetime.now()
                elif status in ["COMPLETED", "FAILED", "TIMEOUT"] and not task.completed_at:
                    task.completed_at = datetime.now()

    def update_task_result(self, task_id: int, result: Any) -> None:
        """
        Store the result in the task's record.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.result = result

    def update_task_progress(self, task_id: int, progress: float) -> None:
        """
        Update the progress of a task (0.0 to 1.0).
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.update_progress(progress)

    def get_pending_tasks(self) -> List[Task]:
        """
        Get all tasks with PENDING status.
        """
        with self._lock:
            return [task for task in self._tasks.values() if task.status == "PENDING"]

    def get_in_progress_tasks(self) -> List[Task]:
        """
        Get all tasks with IN_PROGRESS status.
        """
        with self._lock:
            return [task for task in self._tasks.values() if task.status == "IN_PROGRESS"]

    def list_tasks(self) -> List[Task]:
        """
        Return a list of all tasks in memory (for debugging or inspection).
        """
        with self._lock:
            return list(self._tasks.values())

    def get_task_summary(self) -> Dict[str, int]:
        """
        Get a summary count of tasks by status.
        """
        with self._lock:
            summary = {}
            for task in self._tasks.values():
                if task.status not in summary:
                    summary[task.status] = 0
                summary[task.status] += 1
            return summary

    def __len__(self) -> int:
        with self._lock:
            return len(self._tasks)
            
    def _extract_task_type(self, description: str) -> str:
        """Extract a general task type from the description for pattern recognition."""
        # Simple keyword-based categorization
        keywords = {
            "calculate": "computation",
            "compute": "computation",
            "generate": "generation",
            "create": "creation",
            "fetch": "data_retrieval",
            "download": "data_retrieval",
            "analyze": "analysis",
            "summarize": "summarization",
            "modify": "modification",
            "transform": "transformation"
        }
        
        description_lower = description.lower()
        for keyword, category in keywords.items():
            if keyword in description_lower:
                return category
        return "general"
    
    def _save_task_to_disk(self, task: Task) -> None:
        """Save task to disk for persistence."""
        try:
            os.makedirs("task_storage", exist_ok=True)
            with open(f"task_storage/task_{task.task_id}.json", "w") as f:
                # Convert task to serializable format
                task_data = {
                    "task_id": task.task_id,
                    "priority": task.priority,
                    "description": task.description,
                    "status": task.status,
                    "parent_id": task.parent_id,
                    "created_at": task.created_at.isoformat(),
                    "progress": task.progress
                }
                if task.started_at:
                    task_data["started_at"] = task.started_at.isoformat()
                if task.completed_at:
                    task_data["completed_at"] = task.completed_at.isoformat()
                if task.result:
                    # Handle non-serializable objects
                    try:
                        json.dumps(task.result)
                        task_data["result"] = task.result
                    except (TypeError, OverflowError):
                        task_data["result"] = str(task.result)
                
                json.dump(task_data, f)
        except Exception as e:
            logger.error(f"Error saving task to disk: {e}")
    
    def get_similar_tasks(self, description: str, threshold: float = 0.7) -> List[Task]:
        """Find semantically similar tasks based on description embeddings."""
        try:
            # This would use the embedding function from R1Agent
            # For now, implement a simple keyword matching
            keywords = description.lower().split()
            matches = []
            
            for task_id, task in self._tasks.items():
                task_keywords = task.description.lower().split()
                # Calculate Jaccard similarity
                intersection = len(set(keywords) & set(task_keywords))
                union = len(set(keywords) | set(task_keywords))
                similarity = intersection / union if union > 0 else 0
                
                if similarity >= threshold:
                    matches.append(task)
            
            return matches
        except Exception as e:
            logger.error(f"Error finding similar tasks: {e}")
            return []
    
    def record_task_performance(self, task: Task) -> None:
        """Record performance metrics for a completed task."""
        if task.status == "COMPLETED" and task.started_at and task.completed_at:
            task_type = self._extract_task_type(task.description)
            duration = (task.completed_at - task.started_at).total_seconds()
            self._task_performance_history[task_type].append(duration)
            
            # Keep only the last 100 entries to avoid unbounded growth
            if len(self._task_performance_history[task_type]) > 100:
                self._task_performance_history[task_type] = self._task_performance_history[task_type][-100:]
    
    def get_performance_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for different task types."""
        stats = {}
        for task_type, durations in self._task_performance_history.items():
            if durations:
                stats[task_type] = {
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "count": len(durations)
                }
        return stats


class PriorityTaskQueue:
    """
    PriorityQueue-based structure for tasks, but using heapq under the hood.
    The 'priority' attribute in Task dictates order. Lower number => higher priority.
    """
    def __init__(self) -> None:
        self._heap: List[Task] = []
        self._lock = threading.Lock()

    def push(self, task: Task) -> None:
        """
        Insert a Task into the priority queue.
        """
        with self._lock:
            heapq.heappush(self._heap, task)

    def pop(self) -> Optional[Task]:
        """
        Pop the highest-priority (lowest .priority) Task from the queue.
        Return None if empty.
        """
        with self._lock:
            if not self._heap:
                return None
            return heapq.heappop(self._heap)

    def __len__(self) -> int:
        with self._lock:
            return len(self._heap)

###############################################################################
# FUNCTION ADAPTER FOR "DO ANYTHING"
###############################################################################

class FunctionAdapter:
    """
    Middleware that executes or calls external functions.
    - For demonstration, we have "do_anything" function which executes arbitrary Python code
    - We also have specialized functions for common tasks like HTTP requests
    """

    def do_anything(self, snippet: str) -> Dict[str, Any]:
        """
        Execute arbitrary Python code from snippet (DANGEROUS).
        """
        code = snippet.strip()

        # Remove <code> tags if present
        code = code.replace("<code>", "").replace("</code>", "")

        logger.info(f"[do_anything] Executing code:\n{code}")

        # Create a local namespace for execution
        local_vars = {}
        
        try:
            exec(code, globals(), local_vars)
            return {
                "status": "success", 
                "executed_code": code,
                "local_vars": {k: str(v) for k, v in local_vars.items() if not k.startswith("_")}
            }
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[do_anything] Error: {str(e)}\nTraceback:\n{tb}")
            return {"status": "error", "error": str(e), "traceback": tb}

    def fetch_url(self, url: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Fetch content from a URL.
        """
        logger.info(f"[fetch_url] Fetching: {url}")
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return {
                "status": "success",
                "url": url,
                "status_code": response.status_code,
                "content_type": response.headers.get("Content-Type", ""),
                "content_length": len(response.content),
                "content": response.text[:10000],  # Limit content size
                "headers": dict(response.headers)
            }
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[fetch_url] Error fetching {url}: {str(e)}")
            return {"status": "error", "url": url, "error": str(e), "traceback": tb}

    def summarize_html(self, html_content: str, max_length: int = 500) -> Dict[str, Any]:
        """
        Extract and summarize content from HTML.
        """
        logger.info(f"[summarize_html] Summarizing HTML content of length {len(html_content)}")
        try:
            # Simple extraction of text from HTML
            # In a real implementation, you might use BeautifulSoup or similar
            text = re.sub(r'<[^>]+>', ' ', html_content)
            text = re.sub(r'\s+', ' ', text).strip()
            
            summary = text[:max_length] + ("..." if len(text) > max_length else "")
            
            return {
                "status": "success",
                "original_length": len(html_content),
                "summary_length": len(summary),
                "summary": summary
            }
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[summarize_html] Error: {str(e)}")
            return {"status": "error", "error": str(e), "traceback": tb}

    def process_function_calls(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Detect function calls in the text and execute them.
        Supported formats:
        - <function_call> do_anything: <code> </function_call>
        - <function_call> fetch_url: "https://example.com" </function_call>
        - <function_call> summarize_html: "<html>...</html>" </function_call>
        """
        # Check for do_anything
        do_pattern = r"<function_call>\s*do_anything\s*:\s*(.*?)</function_call>"
        do_match = re.search(do_pattern, text, re.DOTALL)
        if do_match:
            snippet = do_match.group(1)
            logger.info(f"[FunctionAdapter] Detected do_anything snippet")
            return self.do_anything(snippet)
        
        # Check for fetch_url
        fetch_pattern = r"<function_call>\s*fetch_url\s*:\s*[\"']?(.*?)[\"']?\s*</function_call>"
        fetch_match = re.search(fetch_pattern, text, re.DOTALL)
        if fetch_match:
            url = fetch_match.group(1).strip()
            logger.info(f"[FunctionAdapter] Detected fetch_url: {url}")
            return self.fetch_url(url)
        
        # Check for summarize_html
        summarize_pattern = r"<function_call>\s*summarize_html\s*:\s*(.*?)\s*</function_call>"
        summarize_match = re.search(summarize_pattern, text, re.DOTALL)
        if summarize_match:
            html = summarize_match.group(1)
            logger.info(f"[FunctionAdapter] Detected summarize_html")
            return self.summarize_html(html)
        
        return None

###############################################################################
# AGENT & RECURSIVE DECOMPOSITION LOGIC
###############################################################################

class SmartTaskProcessor:
    """
    The 'brains' that tries to handle tasks in a 'smart' way:
    - If a Task's description suggests it should be decomposed, we can create subtasks.
    - If a Task's description triggers a function call, we pass that to the FunctionAdapter.
    - We store results in memory, update the queue, etc.
    - Handles long-running tasks with progress updates
    - Dynamically constructs environments and representations for OOD inputs
    """

    def __init__(self, memory_store: TaskMemoryStore, function_adapter: FunctionAdapter, task_queue: "PriorityTaskQueue"):
        self.memory_store = memory_store
        self.function_adapter = function_adapter
        self.task_queue = task_queue
        self.task_handlers = {
            # File and system operations
            "Create a temporary directory": self._handle_create_temp_dir,
            "Create a text file": self._handle_create_file,
            "Read the file contents": self._handle_read_file,
            "Delete the file": self._handle_delete_file,
            
            # Data processing
            "Calculate prime numbers": self._handle_calculate_primes,
            "Generate random numbers": self._handle_generate_random_numbers,
            "Calculate their statistical": self._handle_calculate_statistics,
            
            # Computational tasks
            "Calculate factorial": self._handle_calculate_factorial,
            "Generate Fibonacci": self._handle_generate_fibonacci,
            "Find all prime Fibonacci": self._handle_find_prime_fibonacci,
            
            # Simulation tasks
            "Simulate creating": self._handle_simulate_image,
            "Apply a blur filter": self._handle_apply_filter,
            "Calculate average pixel": self._handle_calculate_pixel_avg,
            "Simulate a long computation": self._handle_long_computation,
            
            # Error handling tasks
            "Attempt division by zero": self._handle_division_by_zero,
            "Simulate a task that should timeout": self._handle_timeout_task,
            
            # Complex dependency tasks
            "Complex dependency chain": self._handle_dependency_chain,
            "Level": self._handle_dependency_level,
            "Parent task with multiple": self._handle_parallel_parent,
            "Parallel subtask": self._handle_parallel_subtask,
            
            # Resource-intensive tasks
            "Calculate SHA-256": self._handle_calculate_hashes,
            
            # Conditional tasks
            "Check if pandas": self._handle_check_pandas,
            "Create a sample DataFrame": self._handle_create_dataframe,
            "Generate a random number": self._handle_random_number,
            "If even, calculate": self._handle_even_calculation,
            "If odd, calculate": self._handle_odd_calculation,
            
            # Retry tasks
            "Simulate a flaky operation": self._handle_flaky_operation,
            
            # Priority tasks
            "Initially low priority": self._handle_low_priority,
            "Update priority of task": self._handle_update_priority,
            
            # Data processing tasks
            "Generate 1MB of random data": self._handle_generate_large_data,
            "Compress the data": self._handle_compress_data,
            "Calculate compression ratio": self._handle_compression_ratio,
            
            # Progress reporting
            "Custom progress reporting": self._handle_custom_progress,
            
            # Dynamic tasks
            "Dynamic task that spawns": self._handle_dynamic_spawner,
            
            # Complex result tasks
            "Generate a complex nested": self._handle_complex_result,
            "Generate dataset A": self._handle_dataset_a,
            "Generate dataset B": self._handle_dataset_b,
            "Merge results from tasks": self._handle_merge_results,
            
            # Status update tasks
            "Long-running task with periodic": self._handle_periodic_updates,
            
            # Dynamic environment construction for OOD inputs
            "Construct dynamic environment": self._handle_dynamic_environment,
            "Handle OOD input": self._handle_dynamic_environment,
            "Process out-of-distribution": self._handle_dynamic_environment,
            
            # Self-modification tasks
            "Retrieve code": self._handle_retrieve_code,
            "Modify code": self._handle_modify_code,
            "Analyze code": self._handle_analyze_code,
            
            # Advanced computational tasks
            "Calculate the first 1000 digits": self._handle_calculate_pi,
            "Generate all permutations": self._handle_permutations,
            "Solve the Tower of Hanoi": self._handle_tower_of_hanoi,
            "Generate a 1000x1000 matrix": self._handle_matrix_operations,
            "Find eigenvalues": self._handle_matrix_eigenvalues,
            "Calculate matrix determinant": self._handle_matrix_determinant,
            "Perform matrix inversion": self._handle_matrix_inversion,
            "Generate a random directed graph": self._handle_generate_graph,
            "Find all strongly connected": self._handle_strongly_connected,
            "Calculate shortest paths": self._handle_shortest_paths,
            "Solve the Traveling Salesman": self._handle_traveling_salesman,
            "Generate an RSA key pair": self._handle_rsa_generation,
            "Encrypt a sample message": self._handle_rsa_encrypt,
            "Decrypt the message": self._handle_rsa_decrypt,
            "Generate a synthetic corpus": self._handle_generate_corpus,
            "Build a frequency distribution": self._handle_word_frequency,
            "Implement TF-IDF scoring": self._handle_tfidf,
            "Find the most significant terms": self._handle_significant_terms,
            "Calculate the first 100 terms": self._handle_recaman_sequence,
        }

    def _has_subtasks(self, task: Task) -> bool:
        """
        Determine if the given task has any subtasks.
        """
        return len(self.memory_store.get_subtasks(task.task_id)) > 0

    def process_task(self, task: Task) -> None:
        """
        Main logic for how we handle tasks:
         - 'Recursive decomposition' if needed
         - 'Function calls' if the text includes them
         - Update status + result
         - Handle timeouts

        If the Task spawns subtasks, we create them and place them in the queue.
        """
        logger.info(f"[SmartTaskProcessor] Starting task {task.task_id} - '{task.description}'")
        task.start()  # Mark as started and record timestamp

        # Check for timeout before we even start
        if task.is_timed_out():
            task.timeout()
            logger.warning(f"[SmartTaskProcessor] Task {task.task_id} timed out before processing")
            return

        try:
            # Process the task based on its description
            self._process_task_content(task)
            
            # Check if all subtasks are completed before marking this task as complete
            if self._has_subtasks(task):
                subtasks = self.memory_store.get_subtasks(task.task_id)
                if all(st.status in ["COMPLETED", "FAILED", "TIMEOUT"] for st in subtasks):
                    # All subtasks are done, so we can complete this task
                    task.complete()
                    logger.info(f"[SmartTaskProcessor] All subtasks of {task.task_id} completed, marking task as COMPLETED")
                else:
                    # Some subtasks are still pending/in progress, so we'll requeue this task for later
                    # But with lower priority to avoid starving other tasks
                    task.priority += 5
                    self.task_queue.push(task)
                    logger.info(f"[SmartTaskProcessor] Task {task.task_id} has pending subtasks, requeuing with lower priority")
                    return  # Don't mark as completed yet
            else:
                # No subtasks, so we can complete this task
                task.complete()
                logger.info(f"[SmartTaskProcessor] Completed task {task.task_id}")
                
        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"Error processing task: {str(e)}"
            logger.error(f"[SmartTaskProcessor] {error_msg}\n{tb}")
            task.fail(error_msg)
        """
        Main logic for how we handle tasks:
         - 'Recursive decomposition' if needed
         - 'Function calls' if the text includes them
         - Update status + result
         - Handle timeouts

        If the Task spawns subtasks, we create them and place them in the queue.
        """
        logger.info(f"[SmartTaskProcessor] Starting task {task.task_id} - '{task.description}'")
        task.start()  # Mark as started and record timestamp

        # Check for timeout before we even start
        if task.is_timed_out():
            task.timeout()
            logger.warning(f"[SmartTaskProcessor] Task {task.task_id} timed out before processing")
            return

        try:
            # Process the task based on its description
            self._process_task_content(task)
            
            # Check if all subtasks are completed before marking this task as complete
            if self._has_subtasks(task):
                subtasks = self.memory_store.get_subtasks(task.task_id)
                if all(st.status in ["COMPLETED", "FAILED", "TIMEOUT"] for st in subtasks):
                    # All subtasks are done, so we can complete this task
                    task.complete()
                    logger.info(f"[SmartTaskProcessor] All subtasks of {task.task_id} completed, marking task as COMPLETED")
                else:
                    # Some subtasks are still pending/in progress, so we'll requeue this task for later
                    # But with lower priority to avoid starving other tasks
                    task.priority += 5
                    self.task_queue.push(task)
                    logger.info(f"[SmartTaskProcessor] Task {task.task_id} has pending subtasks, requeuing with lower priority")
                    return  # Don't mark as completed yet
            else:
                # No subtasks, so we can complete this task
                task.complete()
                logger.info(f"[SmartTaskProcessor] Completed task {task.task_id}")
                
        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"Error processing task: {str(e)}"
            logger.error(f"[SmartTaskProcessor] {error_msg}\n{tb}")
            task.fail(error_msg)

    def _process_task_content(self, task: Task) -> None:
        """
        Process the actual content of the task based on its description.
        """
        # 1) Handle special task types based on description
        if task.description.startswith("'Fetch http") or task.description.startswith("'Download http"):
            self._handle_fetch_task(task)
            return
            
        if task.description.startswith("'Summarize") and "HTML" in task.description:
            self._handle_summarize_html_task(task)
            return
            
        # 2) Check for task handlers based on description keywords
        for keyword, handler in self.task_handlers.items():
            if keyword in task.description:
                handler(task)
                return
            
        # 3) Default calculation handler for other calculation tasks
        if task.description.startswith("'Calculate") or task.description.startswith("'Process"):
            self._handle_calculation_task(task)
            return

        # 4) Check for <function_call> usage in the description
        result = self.function_adapter.process_function_calls(task.description)
        if result:
            # If function was executed, store the result
            self.memory_store.update_task_result(task.task_id, result)
            logger.info(f"[SmartTaskProcessor] Task {task.task_id} executed function with result status: {result.get('status')}")
            task.update_progress(1.0)  # Mark as 100% complete
            return

        # 5) Potentially do "recursive decomposition"
        #    For demonstration: if we see a phrase like "Subtask(n)=...", we parse out n subtask lines
        #    E.g. "Please do X. Subtask(2)=1) do partial step. 2) do partial step."
        subtask_pattern = r"Subtask\s*\(\s*(\d+)\s*\)\s*=\s*(.*)"
        match = re.search(subtask_pattern, task.description, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                num_subtasks = int(match.group(1))
                subtask_text = match.group(2).strip()
                # Attempt to parse lines like "1) do partial step" from subtask_text
                lines = re.split(r'\d+\)\s*', subtask_text)[1:]  # skip the first empty
                if len(lines) == num_subtasks:
                    for i, line in enumerate(lines, start=1):
                        subtask_desc = line.strip()
                        new_task = self._create_subtask(task, subtask_desc)
                        # Add the subtask to the queue
                        self.task_queue.push(new_task)
                        logger.info(f"[SmartTaskProcessor] Created and queued subtask {new_task.task_id} from parent {task.task_id}")
                    task.update_progress(0.5)  # Mark as 50% complete since we've created subtasks
                else:
                    logger.warning(f"Number of lines found ({len(lines)}) does not match subtask count {num_subtasks}")
            except Exception as e:
                logger.exception(f"[SmartTaskProcessor] Error parsing subtask instructions: {str(e)}")

    def _handle_fetch_task(self, task: Task) -> None:
        """
        Handle a task that fetches content from a URL.
        """
        # Extract URL from task description
        url_match = re.search(r"'(?:Fetch|Download) (https?://[^\s']+)'", task.description)
        if not url_match:
            task.fail("Could not extract URL from task description")
            return
            
        url = url_match.group(1)
        logger.info(f"[SmartTaskProcessor] Fetching URL: {url} for task {task.task_id}")
        
        # Update progress to show we're working
        task.update_progress(0.3)
        
        # Use the function adapter to fetch the URL
        result = self.function_adapter.fetch_url(url)
        self.memory_store.update_task_result(task.task_id, result)
        
        if result["status"] == "success":
            task.update_progress(1.0)
            logger.info(f"[SmartTaskProcessor] Successfully fetched URL for task {task.task_id}")
        else:
            task.fail(f"Failed to fetch URL: {result.get('error', 'Unknown error')}")

    def _handle_summarize_html_task(self, task: Task) -> None:
        """
        Handle a task that summarizes HTML content.
        """
        # First, check if we have a parent task with HTML content
        if not task.parent_id:
            task.fail("Summarize HTML task requires a parent task with HTML content")
            return
            
        parent_task = self.memory_store.get_task(task.parent_id)
        if not parent_task or not parent_task.result:
            task.fail("Parent task has no result with HTML content")
            return
            
        # Get the HTML content from the parent task's result
        parent_result = parent_task.result
        html_content = parent_result.get("content", "")
        
        if not html_content:
            task.fail("No HTML content found in parent task result")
            return
            
        # Update progress
        task.update_progress(0.5)
        
        # Summarize the HTML
        result = self.function_adapter.summarize_html(html_content)
        self.memory_store.update_task_result(task.task_id, result)
        
        if result["status"] == "success":
            task.update_progress(1.0)
            logger.info(f"[SmartTaskProcessor] Successfully summarized HTML for task {task.task_id}")
        else:
            task.fail(f"Failed to summarize HTML: {result.get('error', 'Unknown error')}")

    # ===== File and System Operation Handlers =====
    
    def _handle_create_temp_dir(self, task: Task) -> None:
        """Handle creating a temporary directory."""
        import tempfile
        try:
            temp_dir = tempfile.mkdtemp()
            task.update_progress(1.0)
            result = {
                "status": "success",
                "temp_directory": temp_dir,
                "timestamp": datetime.now().isoformat()
            }
            self.memory_store.update_task_result(task.task_id, result)
            logger.info(f"[SmartTaskProcessor] Created temp directory: {temp_dir}")
        except Exception as e:
            task.fail(f"Failed to create temp directory: {str(e)}")
    
    def _handle_create_file(self, task: Task) -> None:
        """Handle creating a text file with current date."""
        import tempfile
        try:
            # Create a temporary file with the current date
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
            temp_file_path = temp_file.name
            current_date = datetime.now().isoformat()
            
            with open(temp_file_path, 'w') as f:
                f.write(f"File created at: {current_date}\n")
                f.write(f"This is a test file for task {task.task_id}\n")
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "file_path": temp_file_path,
                "content": f"File created at: {current_date}\nThis is a test file for task {task.task_id}\n",
                "timestamp": current_date
            }
            self.memory_store.update_task_result(task.task_id, result)
            logger.info(f"[SmartTaskProcessor] Created file at: {temp_file_path}")
            
            # If this is a subtask, store the file path in the parent task for later subtasks
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task:
                    if not parent_task.result:
                        parent_task.result = {}
                    parent_task.result["temp_file_path"] = temp_file_path
        except Exception as e:
            task.fail(f"Failed to create file: {str(e)}")
    
    def _handle_read_file(self, task: Task) -> None:
        """Handle reading a file created by a previous task."""
        try:
            # Get the file path from the parent task
            file_path = None
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task and parent_task.result:
                    file_path = parent_task.result.get("temp_file_path")
            
            if not file_path:
                # Try to find a sibling task that created a file
                if task.parent_id:
                    siblings = self.memory_store.get_subtasks(task.parent_id)
                    for sibling in siblings:
                        if sibling.task_id != task.task_id and sibling.result:
                            file_path = sibling.result.get("file_path")
                            if file_path:
                                break
            
            if not file_path:
                task.fail("Could not find a file path to read")
                return
                
            task.update_progress(0.5)
            
            # Read the file
            with open(file_path, 'r') as f:
                content = f.read()
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "file_path": file_path,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            self.memory_store.update_task_result(task.task_id, result)
            logger.info(f"[SmartTaskProcessor] Read file: {file_path}")
            
            # Store in parent task for other subtasks
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task:
                    if not parent_task.result:
                        parent_task.result = {}
                    parent_task.result["file_content"] = content
        except Exception as e:
            task.fail(f"Failed to read file: {str(e)}")
    
    def _handle_delete_file(self, task: Task) -> None:
        """Handle deleting a file created by a previous task."""
        import os
        try:
            # Get the file path from the parent task or sibling tasks
            file_path = None
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task and parent_task.result:
                    file_path = parent_task.result.get("temp_file_path")
            
            if not file_path:
                # Try to find a sibling task that created or read a file
                if task.parent_id:
                    siblings = self.memory_store.get_subtasks(task.parent_id)
                    for sibling in siblings:
                        if sibling.task_id != task.task_id and sibling.result:
                            file_path = sibling.result.get("file_path")
                            if file_path:
                                break
            
            if not file_path:
                task.fail("Could not find a file path to delete")
                return
                
            task.update_progress(0.5)
            
            # Delete the file
            os.remove(file_path)
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "deleted_file": file_path,
                "timestamp": datetime.now().isoformat()
            }
            self.memory_store.update_task_result(task.task_id, result)
            logger.info(f"[SmartTaskProcessor] Deleted file: {file_path}")
        except Exception as e:
            task.fail(f"Failed to delete file: {str(e)}")
    
    # ===== Data Processing Handlers =====
    
    def _handle_calculate_primes(self, task: Task) -> None:
        """Calculate prime numbers in a range."""
        try:
            # Extract range from description or use default
            range_match = re.search(r'between (\d+) and (\d+)', task.description)
            start = int(range_match.group(1)) if range_match else 100
            end = int(range_match.group(2)) if range_match else 200
            
            task.update_progress(0.2)
            
            # Simple prime number calculation
            def is_prime(n):
                if n <= 1:
                    return False
                if n <= 3:
                    return True
                if n % 2 == 0 or n % 3 == 0:
                    return False
                i = 5
                while i * i <= n:
                    if n % i == 0 or n % (i + 2) == 0:
                        return False
                    i += 6
                return True
            
            task.update_progress(0.4)
            
            primes = []
            for num in range(start, end + 1):
                if is_prime(num):
                    primes.append(num)
                # Update progress proportionally
                progress = 0.4 + 0.5 * (num - start) / (end - start)
                task.update_progress(progress)
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "range": [start, end],
                "prime_count": len(primes),
                "primes": primes,
                "timestamp": datetime.now().isoformat()
            }
            self.memory_store.update_task_result(task.task_id, result)
            logger.info(f"[SmartTaskProcessor] Found {len(primes)} primes between {start} and {end}")
        except Exception as e:
            task.fail(f"Failed to calculate primes: {str(e)}")
    
    def _handle_generate_random_numbers(self, task: Task) -> None:
        """Generate random numbers."""
        import random
        try:
            # Extract count from description or use default
            count_match = re.search(r'Generate (\d+) random', task.description)
            count = int(count_match.group(1)) if count_match else 1000
            
            # Generate random numbers with progress updates
            numbers = []
            for i in range(count):
                numbers.append(random.random())
                if i % (count // 10) == 0:  # Update progress every 10%
                    progress = i / count
                    task.update_progress(progress)
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "count": count,
                "numbers": numbers[:10] + ["..."] + numbers[-10:] if count > 20 else numbers,  # Truncate for readability
                "full_data": numbers,  # Store full data for subsequent tasks
                "timestamp": datetime.now().isoformat()
            }
            self.memory_store.update_task_result(task.task_id, result)
            
            # Store in parent task for other subtasks
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task:
                    if not parent_task.result:
                        parent_task.result = {}
                    parent_task.result["random_numbers"] = numbers
                    
            logger.info(f"[SmartTaskProcessor] Generated {count} random numbers")
        except Exception as e:
            task.fail(f"Failed to generate random numbers: {str(e)}")
    
    def _handle_calculate_statistics(self, task: Task) -> None:
        """Calculate statistical properties of numbers."""
        try:
            # Get numbers from parent task or sibling task
            numbers = None
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task and parent_task.result:
                    numbers = parent_task.result.get("random_numbers")
            
            if not numbers:
                # Try to find a sibling task that generated numbers
                if task.parent_id:
                    siblings = self.memory_store.get_subtasks(task.parent_id)
                    for sibling in siblings:
                        if sibling.task_id != task.task_id and sibling.result:
                            numbers = sibling.result.get("full_data")
                            if numbers:
                                break
            
            if not numbers:
                task.fail("Could not find numbers to calculate statistics")
                return
                
            task.update_progress(0.3)
            
            # Calculate basic statistics
            count = len(numbers)
            mean = sum(numbers) / count
            task.update_progress(0.5)
            
            # Calculate variance and standard deviation
            variance = sum((x - mean) ** 2 for x in numbers) / count
            std_dev = variance ** 0.5
            task.update_progress(0.7)
            
            # Calculate min, max, median
            sorted_numbers = sorted(numbers)
            minimum = sorted_numbers[0]
            maximum = sorted_numbers[-1]
            median = sorted_numbers[count // 2] if count % 2 == 1 else (sorted_numbers[count // 2 - 1] + sorted_numbers[count // 2]) / 2
            task.update_progress(0.9)
            
            # Calculate quartiles
            q1 = sorted_numbers[count // 4]
            q3 = sorted_numbers[3 * count // 4]
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "count": count,
                "mean": mean,
                "median": median,
                "std_dev": std_dev,
                "min": minimum,
                "max": maximum,
                "q1": q1,
                "q3": q3,
                "timestamp": datetime.now().isoformat()
            }
            self.memory_store.update_task_result(task.task_id, result)
            logger.info(f"[SmartTaskProcessor] Calculated statistics for {count} numbers")
        except Exception as e:
            task.fail(f"Failed to calculate statistics: {str(e)}")
    
    # ===== Computational Task Handlers =====
    
    def _handle_calculate_factorial(self, task: Task) -> None:
        """Calculate factorial of a number."""
        import math
        try:
            # Extract number from description
            num_match = re.search(r'factorial of (\d+)', task.description)
            num = int(num_match.group(1)) if num_match else 100
            
            task.update_progress(0.5)
            
            # Calculate factorial
            result_value = math.factorial(num)
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "number": num,
                "factorial": str(result_value),  # Convert to string as it might be very large
                "factorial_length": len(str(result_value)),
                "timestamp": datetime.now().isoformat()
            }
            self.memory_store.update_task_result(task.task_id, result)
            logger.info(f"[SmartTaskProcessor] Calculated factorial of {num} (length: {len(str(result_value))} digits)")
        except Exception as e:
            task.fail(f"Failed to calculate factorial: {str(e)}")
    
    def _handle_generate_fibonacci(self, task: Task) -> None:
        """Generate Fibonacci sequence up to a limit."""
        try:
            # Extract limit from description
            limit_match = re.search(r'up to (\d+)', task.description)
            limit = int(limit_match.group(1)) if limit_match else 1000
            
            # Generate Fibonacci sequence
            fibonacci = [0, 1]
            while fibonacci[-1] + fibonacci[-2] <= limit:
                fibonacci.append(fibonacci[-1] + fibonacci[-2])
                # Update progress based on how close we are to the limit
                progress = min(0.9, fibonacci[-1] / limit)
                task.update_progress(progress)
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "limit": limit,
                "sequence_length": len(fibonacci),
                "fibonacci_sequence": fibonacci,
                "timestamp": datetime.now().isoformat()
            }
            self.memory_store.update_task_result(task.task_id, result)
            
            # Store in parent task for other subtasks
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task:
                    if not parent_task.result:
                        parent_task.result = {}
                    parent_task.result["fibonacci_sequence"] = fibonacci
                    
            logger.info(f"[SmartTaskProcessor] Generated Fibonacci sequence with {len(fibonacci)} numbers up to {limit}")
        except Exception as e:
            task.fail(f"Failed to generate Fibonacci sequence: {str(e)}")
    
    def _handle_find_prime_fibonacci(self, task: Task) -> None:
        """Find prime Fibonacci numbers."""
        try:
            # Get Fibonacci sequence from parent task or sibling task
            fibonacci = None
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task and parent_task.result:
                    fibonacci = parent_task.result.get("fibonacci_sequence")
            
            if not fibonacci:
                # Try to find a sibling task that generated Fibonacci numbers
                if task.parent_id:
                    siblings = self.memory_store.get_subtasks(task.parent_id)
                    for sibling in siblings:
                        if sibling.task_id != task.task_id and sibling.result:
                            fibonacci = sibling.result.get("fibonacci_sequence")
                            if fibonacci:
                                break
            
            if not fibonacci:
                task.fail("Could not find Fibonacci sequence to check for primes")
                return
                
            task.update_progress(0.2)
            
            # Function to check if a number is prime
            def is_prime(n):
                if n <= 1:
                    return False
                if n <= 3:
                    return True
                if n % 2 == 0 or n % 3 == 0:
                    return False
                i = 5
                while i * i <= n:
                    if n % i == 0 or n % (i + 2) == 0:
                        return False
                    i += 6
                return True
            
            # Find prime Fibonacci numbers
            prime_fibonacci = []
            for i, num in enumerate(fibonacci):
                if is_prime(num):
                    prime_fibonacci.append(num)
                # Update progress
                progress = 0.2 + 0.8 * (i / len(fibonacci))
                task.update_progress(progress)
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "fibonacci_count": len(fibonacci),
                "prime_count": len(prime_fibonacci),
                "prime_fibonacci": prime_fibonacci,
                "timestamp": datetime.now().isoformat()
            }
            self.memory_store.update_task_result(task.task_id, result)
            logger.info(f"[SmartTaskProcessor] Found {len(prime_fibonacci)} prime Fibonacci numbers")
        except Exception as e:
            task.fail(f"Failed to find prime Fibonacci numbers: {str(e)}")
    
    # ===== Simulation Task Handlers =====
    
    def _handle_simulate_image(self, task: Task) -> None:
        """Simulate creating an image."""
        try:
            # Extract dimensions from description
            dim_match = re.search(r'(\d+)x(\d+)', task.description)
            width = int(dim_match.group(1)) if dim_match else 1000
            height = int(dim_match.group(2)) if dim_match else 1000
            
            # Simulate creating an image (just create a 2D array with random values)
            import random
            
            # Create image in chunks to show progress
            image = []
            chunk_size = height // 10
            for i in range(0, height, chunk_size):
                chunk = [[random.randint(0, 255) for _ in range(width)] for _ in range(min(chunk_size, height - i))]
                image.extend(chunk)
                progress = (i + chunk_size) / height
                task.update_progress(progress)
            
            # Calculate average value for verification
            avg_value = sum(sum(row) for row in image) / (width * height)
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "width": width,
                "height": height,
                "avg_value": avg_value,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store the "image" in the parent task for other subtasks
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task:
                    if not parent_task.result:
                        parent_task.result = {}
                    parent_task.result["simulated_image"] = image
                    parent_task.result["image_dimensions"] = (width, height)
            
            self.memory_store.update_task_result(task.task_id, result)
            logger.info(f"[SmartTaskProcessor] Simulated creating a {width}x{height} image with avg value {avg_value:.2f}")
        except Exception as e:
            task.fail(f"Failed to simulate image creation: {str(e)}")
    
    def _handle_apply_filter(self, task: Task) -> None:
        """Apply a blur filter to a simulated image."""
        try:
            # Get image from parent task
            image = None
            dimensions = None
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task and parent_task.result:
                    image = parent_task.result.get("simulated_image")
                    dimensions = parent_task.result.get("image_dimensions")
            
            if not image or not dimensions:
                # Try to find a sibling task that created an image
                if task.parent_id:
                    siblings = self.memory_store.get_subtasks(task.parent_id)
                    for sibling in siblings:
                        if sibling.task_id != task.task_id and sibling.result:
                            if "simulated_image" in sibling.result:
                                image = sibling.result["simulated_image"]
                                dimensions = sibling.result.get("image_dimensions")
                                break
            
            if not image or not dimensions:
                task.fail("Could not find image to apply filter")
                return
                
            width, height = dimensions
            task.update_progress(0.2)
            
            # Simulate applying a blur filter (simple box blur)
            # We'll just average each pixel with its neighbors
            filtered_image = [[0 for _ in range(width)] for _ in range(height)]
            
            # Process the image in chunks to show progress
            chunk_size = height // 10
            for y_start in range(0, height, chunk_size):
                for y in range(y_start, min(y_start + chunk_size, height)):
                    for x in range(width):
                        # Simple box blur: average of 3x3 neighborhood
                        neighbors = []
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < width and 0 <= ny < height:
                                    neighbors.append(image[ny][nx])
                        filtered_image[y][x] = sum(neighbors) // len(neighbors)
                
                progress = 0.2 + 0.7 * ((y_start + chunk_size) / height)
                task.update_progress(progress)
            
            # Calculate average value after filtering
            avg_value = sum(sum(row) for row in filtered_image) / (width * height)
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "filter_type": "blur",
                "width": width,
                "height": height,
                "avg_value_after_filter": avg_value,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store the filtered image in the parent task for other subtasks
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task:
                    if not parent_task.result:
                        parent_task.result = {}
                    parent_task.result["filtered_image"] = filtered_image
            
            self.memory_store.update_task_result(task.task_id, result)
            logger.info(f"[SmartTaskProcessor] Applied blur filter to {width}x{height} image, new avg value: {avg_value:.2f}")
        except Exception as e:
            task.fail(f"Failed to apply filter: {str(e)}")
    
    def _handle_calculate_pixel_avg(self, task: Task) -> None:
        """Calculate average pixel value of an image."""
        try:
            # Get image from parent task (either original or filtered)
            image = None
            dimensions = None
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task and parent_task.result:
                    # Prefer filtered image if available
                    image = parent_task.result.get("filtered_image")
                    if not image:
                        image = parent_task.result.get("simulated_image")
                    dimensions = parent_task.result.get("image_dimensions")
            
            if not image:
                # Try to find a sibling task that has an image
                if task.parent_id:
                    siblings = self.memory_store.get_subtasks(task.parent_id)
                    for sibling in siblings:
                        if sibling.task_id != task.task_id and sibling.result:
                            if "filtered_image" in sibling.result:
                                image = sibling.result["filtered_image"]
                                break
                            elif "simulated_image" in sibling.result:
                                image = sibling.result["simulated_image"]
                                break
            
            if not image:
                task.fail("Could not find image to calculate average pixel value")
                return
                
            task.update_progress(0.5)
            
            # Calculate average pixel value
            total_sum = sum(sum(row) for row in image)
            pixel_count = len(image) * len(image[0])
            avg_value = total_sum / pixel_count
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "pixel_count": pixel_count,
                "average_value": avg_value,
                "timestamp": datetime.now().isoformat()
            }
            self.memory_store.update_task_result(task.task_id, result)
            logger.info(f"[SmartTaskProcessor] Calculated average pixel value: {avg_value:.2f} from {pixel_count} pixels")
        except Exception as e:
            task.fail(f"Failed to calculate average pixel value: {str(e)}")
    
    def _handle_long_computation(self, task: Task) -> None:
        """Simulate a long computation with progress updates."""
        try:
            # Simulate a long computation with 20 steps
            steps = 20
            for i in range(steps):
                # Check for timeout
                if task.is_timed_out():
                    task.timeout()
                    return
                
                # Simulate work
                time.sleep(0.5)
                
                # Update progress
                progress = (i + 1) / steps
                task.update_progress(progress)
                logger.info(f"[SmartTaskProcessor] Long computation progress: {progress:.0%}")
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "steps_completed": steps,
                "computation_time": steps * 0.5,
                "timestamp": datetime.now().isoformat()
            }
            self.memory_store.update_task_result(task.task_id, result)
            logger.info(f"[SmartTaskProcessor] Completed long computation with {steps} steps")
        except Exception as e:
            task.fail(f"Failed during long computation: {str(e)}")
    
    # ===== Error Handling Task Handlers =====
    
    def _handle_division_by_zero(self, task: Task) -> None:
        """Deliberately cause a division by zero error to test error handling."""
        try:
            task.update_progress(0.5)
            logger.info(f"[SmartTaskProcessor] Attempting division by zero...")
            
            # This will cause an error
            result = 1 / 0
            
            # We should never reach this point
            task.update_progress(1.0)
            self.memory_store.update_task_result(task.task_id, {"status": "success", "result": result})
        except Exception as e:
            error_msg = f"Division by zero error: {str(e)}"
            logger.info(f"[SmartTaskProcessor] Successfully caught error: {error_msg}")
            task.fail(error_msg)
    
    def _handle_timeout_task(self, task: Task) -> None:
        """Simulate a task that should timeout."""
        try:
            logger.info(f"[SmartTaskProcessor] Starting task that should timeout (timeout: {task.timeout_seconds}s)")
            
            # Sleep for longer than the timeout
            for i in range(10):
                if task.is_timed_out():
                    logger.info(f"[SmartTaskProcessor] Task detected its own timeout")
                    task.timeout()
                    return
                
                time.sleep(0.5)
                task.update_progress((i + 1) / 10)
            
            # If we get here, the timeout didn't work
            logger.warning(f"[SmartTaskProcessor] Task that should have timed out completed successfully")
            task.complete({"status": "unexpected_success", "message": "This task should have timed out"})
        except Exception as e:
            task.fail(f"Error in timeout task: {str(e)}")
    
    # ===== Dependency Chain Task Handlers =====
    
    def _handle_dependency_chain(self, task: Task) -> None:
        """Handle the parent task of a dependency chain."""
        try:
            task.update_progress(0.2)
            logger.info(f"[SmartTaskProcessor] Starting dependency chain parent task")
            
            # This task will be completed when all its subtasks are done
            # We'll just update its progress based on subtasks
            subtasks = self.memory_store.get_subtasks(task.task_id)
            if subtasks:
                # We already have subtasks, so we're waiting for them
                completed = sum(1 for st in subtasks if st.status == "COMPLETED")
                progress = 0.2 + 0.8 * (completed / len(subtasks))
                task.update_progress(progress)
                logger.info(f"[SmartTaskProcessor] Dependency chain has {completed}/{len(subtasks)} subtasks completed")
            else:
                # No subtasks yet, we'll complete when they're added and finished
                task.update_progress(0.5)
            
            # The task will be marked as completed by the main process_task method
            # when all subtasks are done
        except Exception as e:
            task.fail(f"Error in dependency chain parent: {str(e)}")
    
    def _handle_dependency_level(self, task: Task) -> None:
        """Handle a level in a dependency chain."""
        try:
            # Extract level from description
            level_match = re.search(r'Level (\d+)', task.description)
            level = int(level_match.group(1)) if level_match else 1
            
            logger.info(f"[SmartTaskProcessor] Processing dependency chain level {level}")
            
            # Simulate work for this level
            for i in range(5):
                time.sleep(0.2)
                progress = (i + 1) / 5
                task.update_progress(progress)
            
            result = {
                "status": "success",
                "level": level,
                "message": f"Completed level {level} in dependency chain",
                "timestamp": datetime.now().isoformat()
            }
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Completed dependency chain level {level}")
        except Exception as e:
            task.fail(f"Error in dependency level {level}: {str(e)}")
    
    def _handle_parallel_parent(self, task: Task) -> None:
        """Handle a parent task with multiple parallel children."""
        try:
            task.update_progress(0.1)
            logger.info(f"[SmartTaskProcessor] Starting parallel parent task")
            
            # This task will be completed when all its subtasks are done
            # We'll just update its progress based on subtasks
            subtasks = self.memory_store.get_subtasks(task.task_id)
            if subtasks:
                # We already have subtasks, so we're waiting for them
                completed = sum(1 for st in subtasks if st.status == "COMPLETED")
                progress = 0.1 + 0.9 * (completed / len(subtasks))
                task.update_progress(progress)
                logger.info(f"[SmartTaskProcessor] Parallel parent has {completed}/{len(subtasks)} subtasks completed")
            else:
                # No subtasks yet, we'll complete when they're added and finished
                task.update_progress(0.5)
            
            # The task will be marked as completed by the main process_task method
            # when all subtasks are done
        except Exception as e:
            task.fail(f"Error in parallel parent: {str(e)}")
    
    def _handle_parallel_subtask(self, task: Task) -> None:
        """Handle a parallel subtask with shared parent."""
        try:
            # Extract subtask number from description
            num_match = re.search(r'subtask (\d+)', task.description)
            num = int(num_match.group(1)) if num_match else 0
            
            logger.info(f"[SmartTaskProcessor] Processing parallel subtask {num}")
            
            # Simulate work with different durations based on subtask number
            steps = 5 + num  # More steps for higher numbered tasks
            for i in range(steps):
                time.sleep(0.1)
                progress = (i + 1) / steps
                task.update_progress(progress)
            
            result = {
                "status": "success",
                "subtask_number": num,
                "steps_completed": steps,
                "timestamp": datetime.now().isoformat()
            }
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Completed parallel subtask {num}")
        except Exception as e:
            task.fail(f"Error in parallel subtask {num}: {str(e)}")
    
    # ===== Resource-Intensive Task Handlers =====
    
    def _handle_calculate_hashes(self, task: Task) -> None:
        """Calculate SHA-256 hashes of random strings."""
        import hashlib
        import random
        import string
        
        try:
            # Extract count from description
            count_match = re.search(r'of (\d+) random', task.description)
            count = int(count_match.group(1)) if count_match else 10000
            
            logger.info(f"[SmartTaskProcessor] Calculating SHA-256 hashes for {count} strings")
            
            # Generate random strings and calculate hashes
            hashes = []
            for i in range(count):
                # Generate a random string
                random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=50))
                
                # Calculate SHA-256 hash
                hash_obj = hashlib.sha256(random_string.encode())
                hash_hex = hash_obj.hexdigest()
                hashes.append(hash_hex)
                
                # Update progress every 5%
                if i % (count // 20) == 0:
                    progress = i / count
                    task.update_progress(progress)
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "count": count,
                "sample_hashes": hashes[:5],  # Just show a few samples
                "timestamp": datetime.now().isoformat()
            }
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Completed calculating {count} SHA-256 hashes")
        except Exception as e:
            task.fail(f"Error calculating hashes: {str(e)}")
    
    # ===== Conditional Task Handlers =====
    
    def _handle_check_pandas(self, task: Task) -> None:
        """Check if pandas is installed."""
        try:
            task.update_progress(0.5)
            
            # Try to import pandas
            pandas_available = False
            pandas_version = None
            try:
                import pandas
                pandas_available = True
                pandas_version = pandas.__version__
            except ImportError:
                pass
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "pandas_available": pandas_available,
                "pandas_version": pandas_version,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store result in parent task for other subtasks
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task:
                    if not parent_task.result:
                        parent_task.result = {}
                    parent_task.result["pandas_available"] = pandas_available
            
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Pandas availability check: {pandas_available}")
        except Exception as e:
            task.fail(f"Error checking pandas: {str(e)}")
    
    def _handle_create_dataframe(self, task: Task) -> None:
        """Create a sample DataFrame if pandas is available."""
        try:
            # Check if pandas is available from parent or sibling task
            pandas_available = False
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task and parent_task.result:
                    pandas_available = parent_task.result.get("pandas_available", False)
            
            if not pandas_available:
                # Try to find a sibling task that checked pandas
                if task.parent_id:
                    siblings = self.memory_store.get_subtasks(task.parent_id)
                    for sibling in siblings:
                        if sibling.task_id != task.task_id and sibling.result:
                            pandas_available = sibling.result.get("pandas_available", False)
                            if pandas_available is not None:
                                break
            
            task.update_progress(0.3)
            
            if not pandas_available:
                result = {
                    "status": "skipped",
                    "reason": "pandas is not available",
                    "timestamp": datetime.now().isoformat()
                }
                task.complete(result)
                logger.info(f"[SmartTaskProcessor] Skipped creating DataFrame because pandas is not available")
                return
            
            # Create a sample DataFrame
            import pandas as pd
            import numpy as np
            
            # Create a DataFrame with random data
            df = pd.DataFrame(np.random.randn(10, 4), columns=list('ABCD'))
            
            task.update_progress(0.7)
            
            # Calculate some statistics
            stats = df.describe().to_dict()
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "dataframe_shape": df.shape,
                "dataframe_columns": list(df.columns),
                "dataframe_stats": stats,
                "timestamp": datetime.now().isoformat()
            }
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Created pandas DataFrame with shape {df.shape}")
        except Exception as e:
            task.fail(f"Error creating DataFrame: {str(e)}")
    
    def _handle_random_number(self, task: Task) -> None:
        """Generate a random number and store it for conditional tasks."""
        import random
        try:
            task.update_progress(0.5)
            
            # Generate a random number
            number = random.randint(1, 100)
            is_even = number % 2 == 0
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "random_number": number,
                "is_even": is_even,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in parent task for conditional subtasks
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task:
                    if not parent_task.result:
                        parent_task.result = {}
                    parent_task.result["random_number"] = number
                    parent_task.result["is_even"] = is_even
            
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Generated random number: {number} (even: {is_even})")
        except Exception as e:
            task.fail(f"Error generating random number: {str(e)}")
    
    def _handle_even_calculation(self, task: Task) -> None:
        """Calculate square root if the number is even."""
        import math
        try:
            # Get number and even/odd status from parent or sibling task
            number = None
            is_even = None
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task and parent_task.result:
                    number = parent_task.result.get("random_number")
                    is_even = parent_task.result.get("is_even")
            
            if number is None or is_even is None:
                # Try to find a sibling task with the number
                if task.parent_id:
                    siblings = self.memory_store.get_subtasks(task.parent_id)
                    for sibling in siblings:
                        if sibling.task_id != task.task_id and sibling.result:
                            number = sibling.result.get("random_number")
                            is_even = sibling.result.get("is_even")
                            if number is not None and is_even is not None:
                                break
            
            if number is None:
                task.fail("Could not find random number to process")
                return
                
            task.update_progress(0.5)
            
            if not is_even:
                result = {
                    "status": "skipped",
                    "reason": f"Number {number} is odd, not calculating square root",
                    "timestamp": datetime.now().isoformat()
                }
                task.complete(result)
                logger.info(f"[SmartTaskProcessor] Skipped square root calculation for odd number {number}")
                return
            
            # Calculate square root for even number
            sqrt_value = math.sqrt(number)
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "number": number,
                "square_root": sqrt_value,
                "timestamp": datetime.now().isoformat()
            }
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Calculated square root of {number}: {sqrt_value}")
        except Exception as e:
            task.fail(f"Error calculating square root: {str(e)}")
    
    def _handle_odd_calculation(self, task: Task) -> None:
        """Calculate cube if the number is odd."""
        try:
            # Get number and even/odd status from parent or sibling task
            number = None
            is_even = None
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task and parent_task.result:
                    number = parent_task.result.get("random_number")
                    is_even = parent_task.result.get("is_even")
            
            if number is None or is_even is None:
                # Try to find a sibling task with the number
                if task.parent_id:
                    siblings = self.memory_store.get_subtasks(task.parent_id)
                    for sibling in siblings:
                        if sibling.task_id != task.task_id and sibling.result:
                            number = sibling.result.get("random_number")
                            is_even = sibling.result.get("is_even")
                            if number is not None and is_even is not None:
                                break
            
            if number is None:
                task.fail("Could not find random number to process")
                return
                
            task.update_progress(0.5)
            
            if is_even:
                result = {
                    "status": "skipped",
                    "reason": f"Number {number} is even, not calculating cube",
                    "timestamp": datetime.now().isoformat()
                }
                task.complete(result)
                logger.info(f"[SmartTaskProcessor] Skipped cube calculation for even number {number}")
                return
            
            # Calculate cube for odd number
            cube_value = number ** 3
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "number": number,
                "cube": cube_value,
                "timestamp": datetime.now().isoformat()
            }
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Calculated cube of {number}: {cube_value}")
        except Exception as e:
            task.fail(f"Error calculating cube: {str(e)}")
    
    # ===== Retry Task Handlers =====
    
    def _handle_flaky_operation(self, task: Task) -> None:
        """Simulate a flaky operation that needs retries."""
        import random
        try:
            max_attempts = 5
            attempt = 0
            success = False
            
            while attempt < max_attempts and not success:
                attempt += 1
                task.update_progress(0.2 * attempt / max_attempts)
                
                logger.info(f"[SmartTaskProcessor] Flaky operation attempt {attempt}/{max_attempts}")
                
                # 30% chance of success on each attempt
                if random.random() < 0.3:
                    success = True
                else:
                    # Simulate a delay before retry
                    time.sleep(0.5)
            
            if success:
                task.update_progress(1.0)
                result = {
                    "status": "success",
                    "attempts": attempt,
                    "message": f"Operation succeeded after {attempt} attempts",
                    "timestamp": datetime.now().isoformat()
                }
                task.complete(result)
                logger.info(f"[SmartTaskProcessor] Flaky operation succeeded after {attempt} attempts")
            else:
                task.fail(f"Operation failed after {max_attempts} attempts")
        except Exception as e:
            task.fail(f"Error in flaky operation: {str(e)}")
    
    # ===== Priority Task Handlers =====
    
    def _handle_low_priority(self, task: Task) -> None:
        """Handle a task that starts with low priority."""
        try:
            logger.info(f"[SmartTaskProcessor] Starting low priority task (priority: {task.priority})")
            
            # This task will wait until its priority is updated
            # We'll just check if the priority has been changed
            if task.priority < 50:  # If priority has been increased
                task.update_progress(0.5)
                
                # Now that we have higher priority, do the work
                time.sleep(1)
                
                task.update_progress(1.0)
                result = {
                    "status": "success",
                    "original_priority": 100,
                    "final_priority": task.priority,
                    "message": "Task completed after priority was increased",
                    "timestamp": datetime.now().isoformat()
                }
                task.complete(result)
                logger.info(f"[SmartTaskProcessor] Completed task after priority increase to {task.priority}")
            else:
                # Still low priority, requeue with a small progress update
                current_progress = task.progress or 0
                task.update_progress(min(0.2, current_progress + 0.05))
                self.task_queue.push(task)
                logger.info(f"[SmartTaskProcessor] Requeued low priority task (still at priority {task.priority})")
        except Exception as e:
            task.fail(f"Error in low priority task: {str(e)}")
    
    def _handle_update_priority(self, task: Task) -> None:
        """Update the priority of another task."""
        try:
            # Extract task ID from description
            task_id_match = re.search(r'task (\d+)', task.description)
            if not task_id_match:
                task.fail("Could not extract task ID from description")
                return
                
            target_task_id = int(task_id_match.group(1))
            task.update_progress(0.5)
            
            # Find the target task
            target_task = self.memory_store.get_task(target_task_id)
            if not target_task:
                task.fail(f"Could not find task with ID {target_task_id}")
                return
            
            # Update the priority
            old_priority = target_task.priority
            target_task.priority = 1  # Set to very high priority
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "target_task_id": target_task_id,
                "old_priority": old_priority,
                "new_priority": target_task.priority,
                "timestamp": datetime.now().isoformat()
            }
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Updated task {target_task_id} priority from {old_priority} to {target_task.priority}")
        except Exception as e:
            task.fail(f"Error updating priority: {str(e)}")
    
    # ===== Data Processing Task Handlers =====
    
    def _handle_generate_large_data(self, task: Task) -> None:
        """Generate 1MB of random data."""
        import random
        try:
            # Calculate how many integers we need for ~1MB
            # Each integer is 4 bytes, so we need ~262,144 integers
            count = 262144
            
            logger.info(f"[SmartTaskProcessor] Generating {count} random integers (~1MB)")
            
            # Generate data in chunks to show progress
            data = []
            chunk_size = count // 10
            for i in range(0, count, chunk_size):
                chunk = [random.randint(0, 1000000) for _ in range(min(chunk_size, count - i))]
                data.extend(chunk)
                progress = (i + chunk_size) / count
                task.update_progress(progress)
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "data_size": len(data),
                "approximate_bytes": len(data) * 4,
                "sample": data[:10],
                "timestamp": datetime.now().isoformat()
            }
            
            # Store data in parent task for other subtasks
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task:
                    if not parent_task.result:
                        parent_task.result = {}
                    parent_task.result["large_data"] = data
            
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Generated {len(data)} integers (~{len(data)*4/1024:.1f} KB)")
        except Exception as e:
            task.fail(f"Error generating large data: {str(e)}")
    
    def _handle_compress_data(self, task: Task) -> None:
        """Compress the large data."""
        import zlib
        try:
            # Get data from parent task
            data = None
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task and parent_task.result:
                    data = parent_task.result.get("large_data")
            
            if not data:
                # Try to find a sibling task with the data
                if task.parent_id:
                    siblings = self.memory_store.get_subtasks(task.parent_id)
                    for sibling in siblings:
                        if sibling.task_id != task.task_id and sibling.result:
                            if "large_data" in sibling.result:
                                data = sibling.result["large_data"]
                                break
            
            if not data:
                task.fail("Could not find data to compress")
                return
                
            task.update_progress(0.3)
            
            # Convert data to bytes for compression
            data_bytes = str(data).encode()
            original_size = len(data_bytes)
            
            task.update_progress(0.6)
            
            # Compress the data
            compressed_data = zlib.compress(data_bytes)
            compressed_size = len(compressed_data)
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store compressed data in parent task for other subtasks
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task:
                    if not parent_task.result:
                        parent_task.result = {}
                    parent_task.result["compressed_data"] = {
                        "original_size": original_size,
                        "compressed_size": compressed_size
                    }
            
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Compressed data from {original_size/1024:.1f} KB to {compressed_size/1024:.1f} KB")
        except Exception as e:
            task.fail(f"Error compressing data: {str(e)}")
    
    def _handle_compression_ratio(self, task: Task) -> None:
        """Calculate compression ratio."""
        try:
            # Get compression data from parent task
            compression_data = None
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task and parent_task.result:
                    compression_data = parent_task.result.get("compressed_data")
            
            if not compression_data:
                # Try to find a sibling task with the compression data
                if task.parent_id:
                    siblings = self.memory_store.get_subtasks(task.parent_id)
                    for sibling in siblings:
                        if sibling.task_id != task.task_id and sibling.result:
                            if "original_size" in sibling.result and "compressed_size" in sibling.result:
                                compression_data = {
                                    "original_size": sibling.result["original_size"],
                                    "compressed_size": sibling.result["compressed_size"]
                                }
                                break
            
            if not compression_data:
                task.fail("Could not find compression data")
                return
                
            task.update_progress(0.5)
            
            # Calculate compression ratio and savings
            original_size = compression_data["original_size"]
            compressed_size = compression_data["compressed_size"]
            ratio = original_size / compressed_size if compressed_size > 0 else 0
            savings_percent = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "original_size_kb": original_size / 1024,
                "compressed_size_kb": compressed_size / 1024,
                "compression_ratio": ratio,
                "space_savings_percent": savings_percent,
                "timestamp": datetime.now().isoformat()
            }
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Compression ratio: {ratio:.2f}x, savings: {savings_percent:.1f}%")
        except Exception as e:
            task.fail(f"Error calculating compression ratio: {str(e)}")
    
    # ===== Progress Reporting Task Handlers =====
    
    def _handle_custom_progress(self, task: Task) -> None:
        """Task with custom detailed progress reporting."""
        try:
            # Extract step count from description
            steps_match = re.search(r'with (\d+) steps', task.description)
            steps = int(steps_match.group(1)) if steps_match else 10
            
            logger.info(f"[SmartTaskProcessor] Starting custom progress task with {steps} steps")
            
            # Perform steps with detailed progress reporting
            for i in range(steps):
                step_name = f"Step {i+1}/{steps}: {['Initializing', 'Loading', 'Processing', 'Analyzing', 'Validating', 'Transforming', 'Optimizing', 'Finalizing', 'Exporting', 'Cleaning up'][i % 10]}"
                
                # Log detailed progress
                progress = (i + 0.5) / steps
                logger.info(f"[SmartTaskProcessor] {step_name} - {progress:.0%} complete")
                task.update_progress(progress)
                
                # Simulate work
                time.sleep(0.3)
                
                # Log step completion
                progress = (i + 1) / steps
                logger.info(f"[SmartTaskProcessor] Completed {step_name} - {progress:.0%} complete")
                task.update_progress(progress)
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "steps_completed": steps,
                "detailed_progress": True,
                "timestamp": datetime.now().isoformat()
            }
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Completed custom progress task with {steps} steps")
        except Exception as e:
            task.fail(f"Error in custom progress task: {str(e)}")
    
    # ===== Dynamic Task Handlers =====
    
    def _handle_dynamic_spawner(self, task: Task) -> None:
        """Dynamically spawn a random number of subtasks."""
        import random
        try:
            task.update_progress(0.2)
            
            # Decide how many subtasks to create (2-5)
            subtask_count = random.randint(2, 5)
            logger.info(f"[SmartTaskProcessor] Dynamically spawning {subtask_count} subtasks")
            
            # Create the subtasks
            for i in range(subtask_count):
                subtask = self._create_subtask(
                    task, 
                    f"'Dynamic child task {i+1}/{subtask_count} with random work'"
                )
                self.task_queue.push(subtask)
                logger.info(f"[SmartTaskProcessor] Created dynamic subtask {subtask.task_id}")
            
            task.update_progress(0.5)
            
            # The task will be marked as completed by the main process_task method
            # when all subtasks are done
        except Exception as e:
            task.fail(f"Error spawning dynamic subtasks: {str(e)}")
    
    # ===== Complex Result Task Handlers =====
    
    def _handle_complex_result(self, task: Task) -> None:
        """Generate a complex nested result structure."""
        try:
            task.update_progress(0.3)
            
            # Create a complex nested result structure
            result = {
                "status": "success",
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "version": "1.0",
                    "task_id": task.task_id
                },
                "statistics": {
                    "numeric_values": [1, 2, 3, 4, 5],
                    "mean": 3.0,
                    "median": 3.0,
                    "std_dev": 1.4142135623730951
                },
                "categories": {
                    "category_a": {
                        "count": 10,
                        "items": ["a1", "a2", "a3"]
                    },
                    "category_b": {
                        "count": 5,
                        "items": ["b1", "b2"]
                    }
                },
                "nested_arrays": [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]
                ],
                "timestamp": datetime.now().isoformat()
            }
            
            task.update_progress(1.0)
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Generated complex nested result structure")
        except Exception as e:
            task.fail(f"Error generating complex result: {str(e)}")
    
    def _handle_dataset_a(self, task: Task) -> None:
        """Generate dataset A."""
        import random
        try:
            task.update_progress(0.5)
            
            # Generate a simple dataset
            dataset = {
                "name": "Dataset A",
                "values": [random.randint(1, 100) for _ in range(20)],
                "created_at": datetime.now().isoformat()
            }
            
            task.update_progress(1.0)
            task.complete(dataset)
            logger.info(f"[SmartTaskProcessor] Generated Dataset A with 20 values")
        except Exception as e:
            task.fail(f"Error generating Dataset A: {str(e)}")
    
    def _handle_dataset_b(self, task: Task) -> None:
        """Generate dataset B."""
        import random
        try:
            task.update_progress(0.5)
            
            # Generate a simple dataset
            dataset = {
                "name": "Dataset B",
                "values": [random.random() * 100 for _ in range(15)],
                "created_at": datetime.now().isoformat()
            }
            
            task.update_progress(1.0)
            task.complete(dataset)
            logger.info(f"[SmartTaskProcessor] Generated Dataset B with 15 values")
        except Exception as e:
            task.fail(f"Error generating Dataset B: {str(e)}")
    
    def _handle_merge_results(self, task: Task) -> None:
        """Merge results from two other tasks."""
        try:
            # Extract task IDs from description
            id_match = re.search(r'tasks (\d+) and (\d+)', task.description)
            if not id_match:
                task.fail("Could not extract task IDs from description")
                return
                
            task_a_id = int(id_match.group(1))
            task_b_id = int(id_match.group(2))
            
            task.update_progress(0.3)
            
            # Get the results from both tasks
            task_a = self.memory_store.get_task(task_a_id)
            task_b = self.memory_store.get_task(task_b_id)
            
            if not task_a or not task_b:
                task.fail(f"Could not find one or both tasks: {task_a_id}, {task_b_id}")
                return
                
            if not task_a.result or not task_b.result:
                task.fail(f"One or both tasks have no results yet")
                return
            
            task.update_progress(0.6)
            
            # Merge the results
            merged_result = {
                "status": "success",
                "source_tasks": [task_a_id, task_b_id],
                "dataset_a": task_a.result,
                "dataset_b": task_b.result,
                "merged_data": {
                    "names": [task_a.result.get("name", "Unknown"), task_b.result.get("name", "Unknown")],
                    "total_values": len(task_a.result.get("values", [])) + len(task_b.result.get("values", [])),
                    "combined_values": task_a.result.get("values", []) + task_b.result.get("values", [])
                },
                "timestamp": datetime.now().isoformat()
            }
            
            task.update_progress(1.0)
            task.complete(merged_result)
            logger.info(f"[SmartTaskProcessor] Merged results from tasks {task_a_id} and {task_b_id}")
        except Exception as e:
            task.fail(f"Error merging results: {str(e)}")
    
    # ===== Status Update Task Handlers =====
    
    def _handle_periodic_updates(self, task: Task) -> None:
        """Long-running task with periodic status updates."""
        try:
            # Extract update interval from description
            interval_match = re.search(r'every (\d+) second', task.description)
            interval = int(interval_match.group(1)) if interval_match else 1
            
            # Total duration: 10 seconds
            duration = 10
            start_time = time.time()
            
            logger.info(f"[SmartTaskProcessor] Starting long-running task with updates every {interval}s")
            
            # Run until duration is reached
            while time.time() - start_time < duration:
                # Check for timeout
                if task.is_timed_out():
                    task.timeout()
                    return
                
                # Calculate progress
                elapsed = time.time() - start_time
                progress = min(1.0, elapsed / duration)
                task.update_progress(progress)
                
                # Log detailed status update
                logger.info(f"[SmartTaskProcessor] Status update: {progress:.0%} complete, elapsed: {elapsed:.1f}s")
                
                # Sleep for the interval
                time.sleep(interval)
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "duration": duration,
                "update_interval": interval,
                "updates_sent": duration // interval,
                "timestamp": datetime.now().isoformat()
            }
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Completed long-running task with periodic updates")
        except Exception as e:
            task.fail(f"Error in periodic update task: {str(e)}")
    
    # ===== Advanced Computational Task Handlers =====
    
    def _handle_calculate_pi(self, task: Task) -> None:
        """Calculate digits of pi using the Chudnovsky algorithm."""
        try:
            import mpmath
            
            task.update_progress(0.1)
            logger.info(f"[SmartTaskProcessor] Starting calculation of 1000 digits of pi")
            
            # Set precision to calculate 1000 digits
            mpmath.mp.dps = 1000
            
            task.update_progress(0.3)
            
            # Calculate pi
            pi = mpmath.mp.pi
            pi_str = str(pi)
            
            task.update_progress(0.9)
            
            result = {
                "status": "success",
                "digits_calculated": len(pi_str) - 2,  # Subtract 2 for "3."
                "pi_first_50_digits": pi_str[:52],  # Include "3."
                "pi_last_50_digits": pi_str[-50:],
                "timestamp": datetime.now().isoformat()
            }
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Calculated {len(pi_str) - 2} digits of pi")
        except Exception as e:
            task.fail(f"Error calculating pi: {str(e)}")
    
    def _handle_permutations(self, task: Task) -> None:
        """Generate permutations of a string."""
        try:
            import itertools
            
            # Extract string from description or use default
            string_match = re.search(r'of the string ["\']([^"\']+)["\']', task.description)
            input_string = string_match.group(1) if string_match else "ALGORITHM"
            
            task.update_progress(0.2)
            logger.info(f"[SmartTaskProcessor] Generating permutations of '{input_string}'")
            
            # Generate all permutations
            perms = list(itertools.permutations(input_string))
            
            task.update_progress(0.6)
            
            # Convert tuples to strings
            perm_strings = [''.join(p) for p in perms]
            
            # Count unique permutations (handling repeated letters)
            unique_perms = set(perm_strings)
            
            task.update_progress(0.8)
            
            # Find lexicographically smallest
            smallest = min(perm_strings)
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "input_string": input_string,
                "total_permutations": len(perms),
                "unique_permutations": len(unique_perms),
                "lexicographically_smallest": smallest,
                "sample_permutations": perm_strings[:5] + ["..."] + perm_strings[-5:] if len(perm_strings) > 10 else perm_strings,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in parent task for other subtasks
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task:
                    if not parent_task.result:
                        parent_task.result = {}
                    parent_task.result["permutations"] = perm_strings
                    parent_task.result["unique_permutations"] = len(unique_perms)
            
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Generated {len(perms)} permutations of '{input_string}', {len(unique_perms)} unique")
        except Exception as e:
            task.fail(f"Error generating permutations: {str(e)}")
    
    def _handle_tower_of_hanoi(self, task: Task) -> None:
        """Solve the Tower of Hanoi problem."""
        try:
            # Extract number of disks from description
            disks_match = re.search(r'for (\d+) disks', task.description)
            num_disks = int(disks_match.group(1)) if disks_match else 7
            
            task.update_progress(0.1)
            logger.info(f"[SmartTaskProcessor] Solving Tower of Hanoi for {num_disks} disks")
            
            # Function to solve Tower of Hanoi
            moves = []
            
            def hanoi(n, source, target, auxiliary):
                if n > 0:
                    # Move n-1 disks from source to auxiliary
                    hanoi(n-1, source, auxiliary, target)
                    # Move disk n from source to target
                    moves.append(f"Move disk {n} from {source} to {target}")
                    # Move n-1 disks from auxiliary to target
                    hanoi(n-1, auxiliary, target, source)
            
            # Solve the problem
            hanoi(num_disks, 'A', 'C', 'B')
            
            # Update progress as we go
            total_moves = 2**num_disks - 1
            task.update_progress(0.9)
            
            result = {
                "status": "success",
                "num_disks": num_disks,
                "total_moves": total_moves,
                "first_10_moves": moves[:10],
                "last_10_moves": moves[-10:] if len(moves) > 10 else [],
                "timestamp": datetime.now().isoformat()
            }
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Solved Tower of Hanoi for {num_disks} disks in {total_moves} moves")
        except Exception as e:
            task.fail(f"Error solving Tower of Hanoi: {str(e)}")
    
    def _handle_matrix_operations(self, task: Task) -> None:
        """Generate a random matrix for further operations."""
        try:
            import numpy as np
            
            # Extract dimensions from description
            dim_match = re.search(r'(\d+)x(\d+)', task.description)
            rows = int(dim_match.group(1)) if dim_match else 1000
            cols = int(dim_match.group(2)) if dim_match else 1000
            
            task.update_progress(0.2)
            logger.info(f"[SmartTaskProcessor] Generating {rows}x{cols} random matrix")
            
            # Generate a random matrix with integers
            # Use a smaller matrix for actual computation to avoid memory issues
            actual_size = min(rows, 100)  # Limit to 100x100 for computation
            matrix = np.random.randint(-10, 10, size=(actual_size, actual_size))
            
            task.update_progress(0.8)
            
            result = {
                "status": "success",
                "requested_dimensions": f"{rows}x{cols}",
                "actual_dimensions": f"{actual_size}x{actual_size}",
                "sample_values": matrix[:3, :3].tolist(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store matrix in parent task for other subtasks
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task:
                    if not parent_task.result:
                        parent_task.result = {}
                    parent_task.result["matrix"] = matrix
            
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Generated {actual_size}x{actual_size} random matrix")
        except Exception as e:
            task.fail(f"Error generating matrix: {str(e)}")
    
    def _handle_matrix_eigenvalues(self, task: Task) -> None:
        """Find eigenvalues and eigenvectors of a matrix."""
        try:
            import numpy as np
            
            # Get matrix from parent task
            matrix = None
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task and parent_task.result:
                    matrix = parent_task.result.get("matrix")
            
            if matrix is None:
                # Try to find a sibling task with the matrix
                if task.parent_id:
                    siblings = self.memory_store.get_subtasks(task.parent_id)
                    for sibling in siblings:
                        if sibling.task_id != task.task_id and sibling.result:
                            if "matrix" in sibling.result:
                                matrix = sibling.result["matrix"]
                                break
            
            if matrix is None:
                task.fail("Could not find matrix to calculate eigenvalues")
                return
                
            task.update_progress(0.3)
            logger.info(f"[SmartTaskProcessor] Calculating eigenvalues for {matrix.shape[0]}x{matrix.shape[1]} matrix")
            
            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            
            task.update_progress(0.9)
            
            result = {
                "status": "success",
                "matrix_shape": matrix.shape,
                "num_eigenvalues": len(eigenvalues),
                "eigenvalues_sample": eigenvalues[:5].tolist() if len(eigenvalues) > 5 else eigenvalues.tolist(),
                "eigenvectors_sample": eigenvectors[:3, :3].tolist(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in parent task for other subtasks
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task:
                    if not parent_task.result:
                        parent_task.result = {}
                    parent_task.result["eigenvalues"] = eigenvalues
                    parent_task.result["eigenvectors"] = eigenvectors
            
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Calculated {len(eigenvalues)} eigenvalues")
        except Exception as e:
            task.fail(f"Error calculating eigenvalues: {str(e)}")
    
    def _handle_matrix_determinant(self, task: Task) -> None:
        """Calculate the determinant of a matrix."""
        try:
            import numpy as np
            
            # Get matrix from parent task
            matrix = None
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task and parent_task.result:
                    matrix = parent_task.result.get("matrix")
            
            if matrix is None:
                # Try to find a sibling task with the matrix
                if task.parent_id:
                    siblings = self.memory_store.get_subtasks(task.parent_id)
                    for sibling in siblings:
                        if sibling.task_id != task.task_id and sibling.result:
                            if "matrix" in sibling.result:
                                matrix = sibling.result["matrix"]
                                break
            
            if matrix is None:
                task.fail("Could not find matrix to calculate determinant")
                return
                
            task.update_progress(0.5)
            logger.info(f"[SmartTaskProcessor] Calculating determinant for {matrix.shape[0]}x{matrix.shape[1]} matrix")
            
            # Calculate determinant
            determinant = np.linalg.det(matrix)
            
            task.update_progress(1.0)
            result = {
                "status": "success",
                "matrix_shape": matrix.shape,
                "determinant": determinant,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in parent task for other subtasks
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task:
                    if not parent_task.result:
                        parent_task.result = {}
                    parent_task.result["determinant"] = determinant
            
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Calculated determinant: {determinant}")
        except Exception as e:
            task.fail(f"Error calculating determinant: {str(e)}")
    
    def _handle_matrix_inversion(self, task: Task) -> None:
        """Perform matrix inversion."""
        try:
            import numpy as np
            
            # Get matrix from parent task
            matrix = None
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task and parent_task.result:
                    matrix = parent_task.result.get("matrix")
            
            if matrix is None:
                # Try to find a sibling task with the matrix
                if task.parent_id:
                    siblings = self.memory_store.get_subtasks(task.parent_id)
                    for sibling in siblings:
                        if sibling.task_id != task.task_id and sibling.result:
                            if "matrix" in sibling.result:
                                matrix = sibling.result["matrix"]
                                break
            
            if matrix is None:
                task.fail("Could not find matrix to invert")
                return
                
            task.update_progress(0.3)
            logger.info(f"[SmartTaskProcessor] Inverting {matrix.shape[0]}x{matrix.shape[1]} matrix")
            
            # Calculate inverse
            inverse = np.linalg.inv(matrix)
            
            # Verify inversion by multiplying original and inverse
            verification = np.matmul(matrix, inverse)
            is_identity = np.allclose(verification, np.eye(matrix.shape[0]), rtol=1e-5, atol=1e-8)
            
            task.update_progress(0.9)
            
            result = {
                "status": "success",
                "matrix_shape": matrix.shape,
                "inverse_sample": inverse[:3, :3].tolist(),
                "verification_successful": is_identity,
                "timestamp": datetime.now().isoformat()
            }
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Matrix inversion completed, verification: {is_identity}")
        except Exception as e:
            task.fail(f"Error inverting matrix: {str(e)}")
    
    def _handle_generate_graph(self, task: Task) -> None:
        """Generate a random directed graph."""
        try:
            import networkx as nx
            import random
            
            # Extract node count from description
            nodes_match = re.search(r'with (\d+) nodes', task.description)
            num_nodes = int(nodes_match.group(1)) if nodes_match else 100
            
            task.update_progress(0.2)
            logger.info(f"[SmartTaskProcessor] Generating random directed graph with {num_nodes} nodes")
            
            # Create a directed graph
            G = nx.DiGraph()
            
            # Add nodes
            G.add_nodes_from(range(num_nodes))
            
            # Add random edges (about 5 per node on average)
            num_edges = num_nodes * 5
            for _ in range(num_edges):
                source = random.randint(0, num_nodes-1)
                target = random.randint(0, num_nodes-1)
                weight = random.uniform(1, 10)
                G.add_edge(source, target, weight=weight)
            
            task.update_progress(0.8)
            
            result = {
                "status": "success",
                "num_nodes": G.number_of_nodes(),
                "num_edges": G.number_of_edges(),
                "average_degree": G.number_of_edges() / G.number_of_nodes(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store graph in parent task for other subtasks
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task:
                    if not parent_task.result:
                        parent_task.result = {}
                    parent_task.result["graph"] = G
            
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Generated directed graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        except Exception as e:
            task.fail(f"Error generating graph: {str(e)}")
    
    def _handle_strongly_connected(self, task: Task) -> None:
        """Find strongly connected components in a directed graph."""
        try:
            import networkx as nx
            
            # Get graph from parent task
            G = None
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task and parent_task.result:
                    G = parent_task.result.get("graph")
            
            if G is None:
                # Try to find a sibling task with the graph
                if task.parent_id:
                    siblings = self.memory_store.get_subtasks(task.parent_id)
                    for sibling in siblings:
                        if sibling.task_id != task.task_id and sibling.result:
                            if "graph" in sibling.result:
                                G = sibling.result["graph"]
                                break
            
            if G is None:
                task.fail("Could not find graph to analyze")
                return
                
            task.update_progress(0.3)
            logger.info(f"[SmartTaskProcessor] Finding strongly connected components in graph with {G.number_of_nodes()} nodes")
            
            # Find strongly connected components
            strongly_connected = list(nx.strongly_connected_components(G))
            
            task.update_progress(0.9)
            
            result = {
                "status": "success",
                "num_components": len(strongly_connected),
                "largest_component_size": max(len(c) for c in strongly_connected) if strongly_connected else 0,
                "component_sizes": [len(c) for c in strongly_connected],
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in parent task for other subtasks
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task:
                    if not parent_task.result:
                        parent_task.result = {}
                    parent_task.result["strongly_connected_components"] = strongly_connected
            
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Found {len(strongly_connected)} strongly connected components")
        except Exception as e:
            task.fail(f"Error finding strongly connected components: {str(e)}")
    
    def _handle_shortest_paths(self, task: Task) -> None:
        """Calculate shortest paths using Dijkstra's algorithm."""
        try:
            import networkx as nx
            import random
            
            # Get graph from parent task
            G = None
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task and parent_task.result:
                    G = parent_task.result.get("graph")
            
            if G is None:
                # Try to find a sibling task with the graph
                if task.parent_id:
                    siblings = self.memory_store.get_subtasks(task.parent_id)
                    for sibling in siblings:
                        if sibling.task_id != task.task_id and sibling.result:
                            if "graph" in sibling.result:
                                G = sibling.result["graph"]
                                break
            
            if G is None:
                task.fail("Could not find graph to calculate shortest paths")
                return
                
            task.update_progress(0.2)
            logger.info(f"[SmartTaskProcessor] Calculating shortest paths in graph with {G.number_of_nodes()} nodes")
            
            # Select a random source node
            source = random.choice(list(G.nodes()))
            
            # Calculate shortest paths from source to all other nodes
            shortest_paths = nx.single_source_dijkstra_path_length(G, source, weight='weight')
            
            task.update_progress(0.8)
            
            # Calculate some statistics
            reachable_nodes = len(shortest_paths)
            avg_path_length = sum(shortest_paths.values()) / reachable_nodes if reachable_nodes > 0 else 0
            max_path_length = max(shortest_paths.values()) if shortest_paths else 0
            
            result = {
                "status": "success",
                "source_node": source,
                "reachable_nodes": reachable_nodes,
                "average_path_length": avg_path_length,
                "maximum_path_length": max_path_length,
                "sample_paths": dict(list(shortest_paths.items())[:5]),
                "timestamp": datetime.now().isoformat()
            }
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Calculated shortest paths from node {source} to {reachable_nodes} nodes")
        except Exception as e:
            task.fail(f"Error calculating shortest paths: {str(e)}")
    
    def _handle_traveling_salesman(self, task: Task) -> None:
        """Solve the Traveling Salesman Problem using a genetic algorithm."""
        try:
            import numpy as np
            import random
            
            # Extract city count from description
            cities_match = re.search(r'for (\d+) random cities', task.description)
            num_cities = int(cities_match.group(1)) if cities_match else 12
            
            task.update_progress(0.1)
            logger.info(f"[SmartTaskProcessor] Solving TSP for {num_cities} cities using genetic algorithm")
            
            # Generate random cities (x,y coordinates)
            cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_cities)]
            
            # Calculate distance matrix
            distance_matrix = np.zeros((num_cities, num_cities))
            for i in range(num_cities):
                for j in range(num_cities):
                    if i != j:
                        x1, y1 = cities[i]
                        x2, y2 = cities[j]
                        distance_matrix[i][j] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            task.update_progress(0.2)
            
            # Simple genetic algorithm for TSP
            population_size = 50
            generations = 100
            mutation_rate = 0.1
            
            # Create initial population
            population = []
            for _ in range(population_size):
                route = list(range(num_cities))
                random.shuffle(route)
                population.append(route)
            
            # Fitness function (total distance of route)
            def calculate_distance(route):
                total = 0
                for i in range(num_cities):
                    total += distance_matrix[route[i]][route[(i+1) % num_cities]]
                return total
            
            # Run genetic algorithm
            best_distance = float('inf')
            best_route = None
            
            for generation in range(generations):
                # Calculate fitness for each route
                fitness_scores = [1/calculate_distance(route) for route in population]
                
                # Select parents using roulette wheel selection
                total_fitness = sum(fitness_scores)
                selection_probs = [f/total_fitness for f in fitness_scores]
                
                # Create new population
                new_population = []
                
                for _ in range(population_size):
                    # Select two parents
                    parent1 = random.choices(population, weights=selection_probs)[0]
                    parent2 = random.choices(population, weights=selection_probs)[0]
                    
                    # Crossover (ordered crossover)
                    start, end = sorted(random.sample(range(num_cities), 2))
                    child = [-1] * num_cities
                    
                    # Copy a segment from parent1
                    for i in range(start, end+1):
                        child[i] = parent1[i]
                    
                    # Fill remaining positions with cities from parent2
                    j = 0
                    for i in range(num_cities):
                        if child[i] == -1:
                            while parent2[j] in child:
                                j += 1
                            child[i] = parent2[j]
                            j += 1
                    
                    # Mutation (swap mutation)
                    if random.random() < mutation_rate:
                        idx1, idx2 = random.sample(range(num_cities), 2)
                        child[idx1], child[idx2] = child[idx2], child[idx1]
                    
                    new_population.append(child)
                
                # Update population
                population = new_population
                
                # Track best solution
                for route in population:
                    distance = calculate_distance(route)
                    if distance < best_distance:
                        best_distance = distance
                        best_route = route.copy()
                
                # Update progress
                task.update_progress(0.2 + 0.7 * (generation / generations))
            
            task.update_progress(0.9)
            
            # Calculate city coordinates for the best route
            best_route_coords = [cities[i] for i in best_route]
            
            result = {
                "status": "success",
                "num_cities": num_cities,
                "best_distance": best_distance,
                "best_route": best_route,
                "generations": generations,
                "population_size": population_size,
                "timestamp": datetime.now().isoformat()
            }
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Solved TSP for {num_cities} cities, best distance: {best_distance:.2f}")
        except Exception as e:
            task.fail(f"Error solving Traveling Salesman Problem: {str(e)}")
    
    def _handle_rsa_generation(self, task: Task) -> None:
        """Generate an RSA key pair."""
        try:
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.hazmat.primitives import serialization
            
            # Extract key size from description
            bits_match = re.search(r'with (\d+) bits', task.description)
            key_size = int(bits_match.group(1)) if bits_match else 2048
            
            task.update_progress(0.2)
            logger.info(f"[SmartTaskProcessor] Generating {key_size}-bit RSA key pair")
            
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size
            )
            
            # Get public key
            public_key = private_key.public_key()
            
            task.update_progress(0.8)
            
            # Serialize keys to PEM format
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            result = {
                "status": "success",
                "key_size": key_size,
                "public_key": public_pem.decode('utf-8'),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store keys in parent task for other subtasks
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task:
                    if not parent_task.result:
                        parent_task.result = {}
                    parent_task.result["private_key"] = private_key
                    parent_task.result["public_key"] = public_key
                    parent_task.result["private_pem"] = private_pem
                    parent_task.result["public_pem"] = public_pem
            
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Generated {key_size}-bit RSA key pair")
        except Exception as e:
            task.fail(f"Error generating RSA key pair: {str(e)}")
    
    def _handle_rsa_encrypt(self, task: Task) -> None:
        """Encrypt a message using RSA."""
        try:
            from cryptography.hazmat.primitives.asymmetric import padding
            from cryptography.hazmat.primitives import hashes
            import base64
            
            # Get public key from parent task
            public_key = None
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task and parent_task.result:
                    public_key = parent_task.result.get("public_key")
            
            if public_key is None:
                # Try to find a sibling task with the key
                if task.parent_id:
                    siblings = self.memory_store.get_subtasks(task.parent_id)
                    for sibling in siblings:
                        if sibling.task_id != task.task_id and sibling.result:
                            if "public_key" in sibling.result:
                                public_key = sibling.result["public_key"]
                                break
            
            if public_key is None:
                task.fail("Could not find public key for encryption")
                return
                
            task.update_progress(0.3)
            
            # Sample message to encrypt
            message = "This is a secret message for testing RSA encryption and decryption."
            
            task.update_progress(0.5)
            logger.info(f"[SmartTaskProcessor] Encrypting message with RSA")
            
            # Encrypt the message
            ciphertext = public_key.encrypt(
                message.encode('utf-8'),
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Encode ciphertext as base64 for storage/display
            encoded_ciphertext = base64.b64encode(ciphertext).decode('utf-8')
            
            task.update_progress(0.9)
            
            result = {
                "status": "success",
                "original_message": message,
                "ciphertext_length": len(ciphertext),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store encrypted message in parent task for decryption
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task:
                    if not parent_task.result:
                        parent_task.result = {}
                    parent_task.result["original_message"] = message
                    parent_task.result["ciphertext"] = ciphertext
            
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Encrypted message of length {len(message)} to ciphertext of length {len(ciphertext)}")
        except Exception as e:
            task.fail(f"Error encrypting message: {str(e)}")
    
    def _handle_rsa_decrypt(self, task: Task) -> None:
        """Decrypt a message using RSA."""
        try:
            from cryptography.hazmat.primitives.asymmetric import padding
            from cryptography.hazmat.primitives import hashes
            
            # Get private key and ciphertext from parent task
            private_key = None
            ciphertext = None
            original_message = None
            
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task and parent_task.result:
                    private_key = parent_task.result.get("private_key")
                    ciphertext = parent_task.result.get("ciphertext")
                    original_message = parent_task.result.get("original_message")
            
            if private_key is None or ciphertext is None:
                # Try to find sibling tasks with the required data
                if task.parent_id:
                    siblings = self.memory_store.get_subtasks(task.parent_id)
                    for sibling in siblings:
                        if sibling.task_id != task.task_id and sibling.result:
                            if "private_key" in sibling.result:
                                private_key = sibling.result["private_key"]
                            if "ciphertext" in sibling.result:
                                ciphertext = sibling.result["ciphertext"]
                            if "original_message" in sibling.result:
                                original_message = sibling.result["original_message"]
            
            if private_key is None or ciphertext is None:
                task.fail("Could not find private key or ciphertext for decryption")
                return
                
            task.update_progress(0.4)
            logger.info(f"[SmartTaskProcessor] Decrypting message with RSA")
            
            # Decrypt the message
            plaintext = private_key.decrypt(
                ciphertext,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            ).decode('utf-8')
            
            task.update_progress(0.8)
            
            # Verify decryption was successful
            is_correct = plaintext == original_message if original_message else "Unknown (original message not available)"
            
            result = {
                "status": "success",
                "decrypted_message": plaintext,
                "verification_successful": is_correct,
                "timestamp": datetime.now().isoformat()
            }
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Decrypted message successfully, verification: {is_correct}")
        except Exception as e:
            task.fail(f"Error decrypting message: {str(e)}")
    
    def _handle_generate_corpus(self, task: Task) -> None:
        """Generate a synthetic corpus of sentences."""
        try:
            import random
            import string
            
            # Extract corpus size from description
            size_match = re.search(r'of ([\d,]+) sentences', task.description)
            num_sentences = int(size_match.group(1).replace(',', '')) if size_match else 10000
            
            # Limit corpus size for memory considerations
            actual_size = min(num_sentences, 10000)
            
            task.update_progress(0.1)
            logger.info(f"[SmartTaskProcessor] Generating synthetic corpus of {actual_size} sentences")
            
            # Define vocabulary for synthetic text
            nouns = ["time", "person", "year", "way", "day", "thing", "man", "world", "life", "hand", "part", "child", 
                    "eye", "woman", "place", "work", "week", "case", "point", "government", "company", "number", "group"]
            verbs = ["be", "have", "do", "say", "get", "make", "go", "know", "take", "see", "come", "think", "look", 
                    "want", "give", "use", "find", "tell", "ask", "work", "seem", "feel", "try", "leave", "call"]
            adjectives = ["good", "new", "first", "last", "long", "great", "little", "own", "other", "old", "right", 
                        "big", "high", "different", "small", "large", "next", "early", "young", "important", "few"]
            adverbs = ["up", "so", "out", "just", "now", "how", "then", "more", "also", "here", "well", "only", "very", 
                    "even", "back", "there", "down", "still", "in", "as", "too", "when", "never", "really"]
            prepositions = ["to", "of", "in", "for", "on", "with", "at", "by", "from", "up", "about", "into", "over", 
                            "after", "beneath", "under", "above", "through", "across", "beyond"]
            
            # Generate sentences
            corpus = []
            for i in range(actual_size):
                # Create a simple sentence structure
                sentence = []
                
                # Add a determiner
                sentence.append(random.choice(["The", "A", "This", "That", "Some", "Many", "Few"]))
                
                # Add an adjective (sometimes)
                if random.random() < 0.7:
                    sentence.append(random.choice(adjectives))
                
                # Add a noun
                sentence.append(random.choice(nouns))
                
                # Add a verb
                sentence.append(random.choice(verbs))
                
                # Add an adverb (sometimes)
                if random.random() < 0.3:
                    sentence.append(random.choice(adverbs))
                
                # Add a preposition (sometimes)
                if random.random() < 0.4:
                    sentence.append(random.choice(prepositions))
                    
                    # Add a determiner
                    sentence.append(random.choice(["the", "a", "this", "that", "some", "many", "few"]))
                    
                    # Add an adjective (sometimes)
                    if random.random() < 0.5:
                        sentence.append(random.choice(adjectives))
                    
                    # Add a noun
                    sentence.append(random.choice(nouns))
                
                # Join words and add punctuation
                sentence_text = " ".join(sentence) + random.choice([".", ".", ".", "!", "?"])
                corpus.append(sentence_text)
                
                # Update progress periodically
                if i % (actual_size // 10) == 0:
                    progress = 0.1 + 0.8 * (i / actual_size)
                    task.update_progress(progress)
            
            task.update_progress(0.9)
            
            result = {
                "status": "success",
                "requested_size": num_sentences,
                "actual_size": actual_size,
                "sample_sentences": corpus[:5],
                "vocabulary_size": len(set(nouns + verbs + adjectives + adverbs + prepositions)),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store corpus in parent task for other subtasks
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task:
                    if not parent_task.result:
                        parent_task.result = {}
                    parent_task.result["corpus"] = corpus
            
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Generated synthetic corpus of {actual_size} sentences")
        except Exception as e:
            task.fail(f"Error generating corpus: {str(e)}")
    
    def _handle_word_frequency(self, task: Task) -> None:
        """Build a frequency distribution of words in a corpus."""
        try:
            import re
            from collections import Counter
            
            # Get corpus from parent task
            corpus = None
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task and parent_task.result:
                    corpus = parent_task.result.get("corpus")
            
            if corpus is None:
                # Try to find a sibling task with the corpus
                if task.parent_id:
                    siblings = self.memory_store.get_subtasks(task.parent_id)
                    for sibling in siblings:
                        if sibling.task_id != task.task_id and sibling.result:
                            if "corpus" in sibling.result:
                                corpus = sibling.result["corpus"]
                                break
            
            if corpus is None:
                task.fail("Could not find corpus to analyze")
                return
                
            task.update_progress(0.3)
            logger.info(f"[SmartTaskProcessor] Building word frequency distribution for corpus of {len(corpus)} sentences")
            
            # Tokenize and count words
            words = []
            for sentence in corpus:
                # Convert to lowercase and split on non-alphanumeric characters
                tokens = re.findall(r'\b[a-z]+\b', sentence.lower())
                words.extend(tokens)
            
            # Count word frequencies
            word_counts = Counter(words)
            
            task.update_progress(0.8)
            
            # Get most common words
            most_common = word_counts.most_common(20)
            
            result = {
                "status": "success",
                "total_words": len(words),
                "unique_words": len(word_counts),
                "most_common": most_common,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store word counts in parent task for other subtasks
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task:
                    if not parent_task.result:
                        parent_task.result = {}
                    parent_task.result["word_counts"] = word_counts
                    parent_task.result["total_words"] = len(words)
            
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Built frequency distribution with {len(word_counts)} unique words")
        except Exception as e:
            task.fail(f"Error building word frequency distribution: {str(e)}")
    
    def _handle_tfidf(self, task: Task) -> None:
        """Implement TF-IDF scoring for a corpus."""
        try:
            import math
            from collections import Counter, defaultdict
            
            # Get corpus and word counts from parent task
            corpus = None
            word_counts = None
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task and parent_task.result:
                    corpus = parent_task.result.get("corpus")
                    word_counts = parent_task.result.get("word_counts")
            
            if corpus is None:
                # Try to find sibling tasks with the required data
                if task.parent_id:
                    siblings = self.memory_store.get_subtasks(task.parent_id)
                    for sibling in siblings:
                        if sibling.task_id != task.task_id and sibling.result:
                            if "corpus" in sibling.result:
                                corpus = sibling.result["corpus"]
                            if "word_counts" in sibling.result:
                                word_counts = sibling.result["word_counts"]
            
            if corpus is None:
                task.fail("Could not find corpus for TF-IDF calculation")
                return
                
            task.update_progress(0.2)
            logger.info(f"[SmartTaskProcessor] Calculating TF-IDF scores for corpus of {len(corpus)} sentences")
            
            # If we don't have word counts, calculate them
            if word_counts is None:
                import re
                words = []
                for sentence in corpus:
                    tokens = re.findall(r'\b[a-z]+\b', sentence.lower())
                    words.extend(tokens)
                word_counts = Counter(words)
            
            # Calculate document frequency (in how many sentences each word appears)
            doc_frequency = defaultdict(int)
            for sentence in corpus:
                # Get unique words in this sentence
                unique_words = set(re.findall(r'\b[a-z]+\b', sentence.lower()))
                for word in unique_words:
                    doc_frequency[word] += 1
            
            task.update_progress(0.5)
            
            # Calculate TF-IDF for each word
            tfidf_scores = {}
            num_docs = len(corpus)
            
            for word, count in word_counts.items():
                # Term frequency (normalized by total words)
                tf = count / sum(word_counts.values())
                
                # Inverse document frequency
                idf = math.log(num_docs / (1 + doc_frequency[word]))
                
                # TF-IDF score
                tfidf_scores[word] = tf * idf
            
            # Sort words by TF-IDF score
            sorted_tfidf = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
            
            task.update_progress(0.9)
            
            result = {
                "status": "success",
                "total_words_analyzed": len(word_counts),
                "highest_tfidf_words": sorted_tfidf[:20],
                "timestamp": datetime.now().isoformat()
            }
            
            # Store TF-IDF scores in parent task for other subtasks
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task:
                    if not parent_task.result:
                        parent_task.result = {}
                    parent_task.result["tfidf_scores"] = tfidf_scores
                    parent_task.result["sorted_tfidf"] = sorted_tfidf
            
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Calculated TF-IDF scores for {len(tfidf_scores)} words")
        except Exception as e:
            task.fail(f"Error calculating TF-IDF scores: {str(e)}")
    
    def _handle_significant_terms(self, task: Task) -> None:
        """Find the most significant terms based on TF-IDF scores."""
        try:
            # Get TF-IDF scores from parent task
            sorted_tfidf = None
            if task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task and parent_task.result:
                    sorted_tfidf = parent_task.result.get("sorted_tfidf")
            
            if sorted_tfidf is None:
                # Try to find a sibling task with the TF-IDF scores
                if task.parent_id:
                    siblings = self.memory_store.get_subtasks(task.parent_id)
                    for sibling in siblings:
                        if sibling.task_id != task.task_id and sibling.result:
                            if "sorted_tfidf" in sibling.result:
                                sorted_tfidf = sibling.result["sorted_tfidf"]
                                break
            
            if sorted_tfidf is None:
                task.fail("Could not find TF-IDF scores to analyze")
                return
                
            task.update_progress(0.5)
            logger.info(f"[SmartTaskProcessor] Analyzing significant terms from TF-IDF scores")
            
            # Get top and bottom terms
            top_terms = sorted_tfidf[:50]
            bottom_terms = sorted_tfidf[-50:]
            
            # Calculate statistics
            all_scores = [score for _, score in sorted_tfidf]
            avg_score = sum(all_scores) / len(all_scores)
            median_score = sorted(all_scores)[len(all_scores) // 2]
            
            task.update_progress(0.9)
            
            result = {
                "status": "success",
                "most_significant_terms": top_terms[:20],
                "least_significant_terms": bottom_terms[:20],
                "average_tfidf_score": avg_score,
                "median_tfidf_score": median_score,
                "timestamp": datetime.now().isoformat()
            }
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Identified most significant terms from {len(sorted_tfidf)} words")
        except Exception as e:
            task.fail(f"Error finding significant terms: {str(e)}")
    
    def _handle_recaman_sequence(self, task: Task) -> None:
        """Calculate terms of the Recamán sequence and find patterns."""
        try:
            # Extract term count from description
            terms_match = re.search(r'first (\d+) terms', task.description)
            num_terms = int(terms_match.group(1)) if terms_match else 100
            
            task.update_progress(0.2)
            logger.info(f"[SmartTaskProcessor] Calculating first {num_terms} terms of Recamán sequence")
            
            # Calculate Recamán sequence
            sequence = [0]
            seen = set([0])
            
            for i in range(1, num_terms):
                # Next term is a(n-1) - n if positive and not already in sequence
                # Otherwise, it's a(n-1) + n
                prev = sequence[i-1]
                next_term = prev - i
                
                if next_term < 0 or next_term in seen:
                    next_term = prev + i
                
                sequence.append(next_term)
                seen.add(next_term)
                
                # Update progress periodically
                if i % (num_terms // 10) == 0:
                    progress = 0.2 + 0.6 * (i / num_terms)
                    task.update_progress(progress)
            
            task.update_progress(0.8)
            
            # Analyze patterns
            # 1. Find duplicates
            value_counts = {}
            for term in sequence:
                if term in value_counts:
                    value_counts[term] += 1
                else:
                    value_counts[term] = 1
            
            duplicates = {k: v for k, v in value_counts.items() if v > 1}
            
            # 2. Find jumps (differences between consecutive terms)
            jumps = [sequence[i] - sequence[i-1] for i in range(1, len(sequence))]
            avg_jump = sum(abs(j) for j in jumps) / len(jumps)
            max_jump = max(abs(j) for j in jumps)
            
            # 3. Check for arithmetic progressions
            ap_lengths = []
            i = 0
            while i < len(jumps) - 1:
                current_diff = jumps[i]
                length = 1
                j = i + 1
                while j < len(jumps) and jumps[j] == current_diff:
                    length += 1
                    j += 1
                if length > 1:
                    ap_lengths.append(length)
                i = j
            
            task.update_progress(1.0)
            
            result = {
                "status": "success",
                "sequence_length": len(sequence),
                "first_20_terms": sequence[:20],
                "last_20_terms": sequence[-20:],
                "max_term": max(sequence),
                "duplicate_count": len(duplicates),
                "average_jump": avg_jump,
                "maximum_jump": max_jump,
                "longest_arithmetic_progression": max(ap_lengths) if ap_lengths else 0,
                "timestamp": datetime.now().isoformat()
            }
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Calculated {len(sequence)} terms of Recamán sequence, max term: {max(sequence)}")
        except Exception as e:
            task.fail(f"Error calculating Recamán sequence: {str(e)}")
    
    # ===== Dynamic Environment Construction for OOD Inputs =====
    
    def _handle_dynamic_environment(self, task: Task) -> None:
        """Dynamically construct an environment for handling out-of-distribution inputs."""
        try:
            # Extract the OOD input from the task description
            ood_input_match = re.search(r'OOD input[:\s]+[\'"]([^\'"]+)[\'"]', task.description, re.IGNORECASE)
            ood_input = ood_input_match.group(1) if ood_input_match else "Unknown input"
            
            task.update_progress(0.1)
            logger.info(f"[SmartTaskProcessor] Constructing dynamic environment for OOD input: {ood_input}")
            
            # Phase 1: Input Analysis
            task.update_progress(0.2)
            input_analysis = self._analyze_ood_input(ood_input)
            
            # Phase 2: Environment Construction
            task.update_progress(0.4)
            environment = self._construct_environment(input_analysis)
            
            # Phase 3: Representation Building
            task.update_progress(0.6)
            representation = self._build_representation(ood_input, environment)
            
            # Phase 4: Validation and Testing
            task.update_progress(0.8)
            validation_results = self._validate_environment(environment, representation, ood_input)
            
            # Compile results
            result = {
                "status": "success",
                "ood_input": ood_input,
                "input_analysis": input_analysis,
                "environment": {
                    "type": environment["type"],
                    "dimensions": environment["dimensions"],
                    "properties": environment["properties"]
                },
                "representation": {
                    "format": representation["format"],
                    "structure": representation["structure"],
                    "sample": representation["sample"]
                },
                "validation": validation_results,
                "timestamp": datetime.now().isoformat()
            }
            
            task.update_progress(1.0)
            task.complete(result)
            logger.info(f"[SmartTaskProcessor] Successfully constructed dynamic environment for OOD input")
        except Exception as e:
            task.fail(f"Error constructing dynamic environment: {str(e)}")
    
    def _analyze_ood_input(self, input_text: str) -> dict:
        """Analyze OOD input to determine its characteristics."""
        import re
        import math
        from collections import Counter
        
        # Determine if input is primarily numeric, text, or mixed
        numeric_pattern = r'^[0-9\.\-\+eE]+$'
        alpha_pattern = r'^[a-zA-Z\s\.\,\!\?\-\:\'\"]+$'
        
        # Count character types
        char_counts = Counter(input_text)
        digit_count = sum(char_counts[c] for c in char_counts if c.isdigit())
        alpha_count = sum(char_counts[c] for c in char_counts if c.isalpha())
        space_count = sum(char_counts[c] for c in char_counts if c.isspace())
        symbol_count = len(input_text) - digit_count - alpha_count - space_count
        
        # Determine primary type
        if digit_count > 0.7 * len(input_text):
            primary_type = "numeric"
        elif alpha_count > 0.7 * len(input_text):
            primary_type = "text"
        else:
            primary_type = "mixed"
        
        # Check for specific patterns
        patterns = {
            "email": bool(re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', input_text)),
            "url": bool(re.search(r'https?://[^\s]+', input_text)),
            "date": bool(re.search(r'\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}', input_text)),
            "json": input_text.strip().startswith('{') and input_text.strip().endswith('}'),
            "list": input_text.strip().startswith('[') and input_text.strip().endswith(']'),
            "code": bool(re.search(r'(def|class|function|import|from|var|const|let)\s', input_text))
        }
        
        # Determine complexity
        words = input_text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        complexity = {
            "length": len(input_text),
            "word_count": len(words),
            "avg_word_length": avg_word_length,
            "unique_chars": len(char_counts),
            "entropy": sum(-count/len(input_text) * math.log2(count/len(input_text)) 
                          for count in char_counts.values())
        }
        
        return {
            "primary_type": primary_type,
            "composition": {
                "digit_percentage": digit_count / len(input_text) if len(input_text) > 0 else 0,
                "alpha_percentage": alpha_count / len(input_text) if len(input_text) > 0 else 0,
                "space_percentage": space_count / len(input_text) if len(input_text) > 0 else 0,
                "symbol_percentage": symbol_count / len(input_text) if len(input_text) > 0 else 0
            },
            "detected_patterns": patterns,
            "complexity": complexity
        }
    
    def _construct_environment(self, input_analysis: dict) -> dict:
        """Construct an appropriate environment based on input analysis."""
        env_type = "unknown"
        dimensions = []
        properties = {}
        
        # Determine environment type based on input analysis
        if input_analysis["primary_type"] == "numeric":
            env_type = "numerical"
            dimensions = ["value", "precision", "range"]
            properties = {
                "continuous": True,
                "bounded": False,
                "operations": ["add", "subtract", "multiply", "divide"]
            }
        elif input_analysis["detected_patterns"]["code"]:
            env_type = "code_execution"
            dimensions = ["syntax", "semantics", "runtime"]
            properties = {
                "language": "python",  # Default assumption
                "sandbox": True,
                "operations": ["parse", "execute", "evaluate"]
            }
        elif input_analysis["detected_patterns"]["json"] or input_analysis["detected_patterns"]["list"]:
            env_type = "structured_data"
            dimensions = ["schema", "validation", "transformation"]
            properties = {
                "format": "json" if input_analysis["detected_patterns"]["json"] else "list",
                "operations": ["parse", "query", "transform"]
            }
        elif input_analysis["detected_patterns"]["url"]:
            env_type = "web_content"
            dimensions = ["fetch", "parse", "extract"]
            properties = {
                "content_type": "unknown",
                "operations": ["fetch", "parse", "extract"]
            }
        else:
            # Default to text processing environment
            env_type = "text_processing"
            dimensions = ["tokenization", "analysis", "generation"]
            properties = {
                "language": "english",  # Default assumption
                "operations": ["tokenize", "analyze", "generate"]
            }
        
        # Add complexity-based properties
        if input_analysis["complexity"]["length"] > 1000:
            properties["chunking"] = True
            properties["chunk_size"] = 500
        
        if input_analysis["complexity"]["entropy"] > 4.0:
            properties["high_entropy"] = True
            properties["compression"] = True
        
        return {
            "type": env_type,
            "dimensions": dimensions,
            "properties": properties,
            "created_at": datetime.now().isoformat()
        }
    
    def _build_representation(self, input_text: str, environment: dict) -> dict:
        """Build an appropriate representation for the input based on the environment."""
        representation_format = "unknown"
        structure = {}
        sample = {}
        
        env_type = environment["type"]
        
        if env_type == "numerical":
            # For numerical inputs, create a numerical representation
            try:
                # Try to parse as float first
                value = float(input_text)
                representation_format = "float"
                structure = {
                    "value": value,
                    "is_integer": value.is_integer(),
                    "sign": "positive" if value >= 0 else "negative"
                }
                sample = {
                    "original": input_text,
                    "normalized": value
                }
            except ValueError:
                # If not a simple number, try to extract numbers
                import re
                numbers = re.findall(r'-?\d+\.?\d*', input_text)
                representation_format = "numeric_sequence"
                structure = {
                    "count": len(numbers),
                    "values": [float(n) for n in numbers]
                }
                sample = {
                    "original": input_text,
                    "extracted": numbers[:5]
                }
                
        elif env_type == "code_execution":
            # For code, create a code representation
            import ast
            representation_format = "code_ast"
            try:
                parsed_ast = ast.parse(input_text)
                structure = {
                    "type": "module",
                    "body_length": len(parsed_ast.body),
                    "contains_functions": any(isinstance(node, ast.FunctionDef) for node in ast.walk(parsed_ast)),
                    "contains_classes": any(isinstance(node, ast.ClassDef) for node in ast.walk(parsed_ast)),
                    "imports": [node.names[0].name for node in ast.walk(parsed_ast) 
                               if isinstance(node, ast.Import) and node.names]
                }
                sample = {
                    "original": input_text[:100] + ("..." if len(input_text) > 100 else ""),
                    "ast_summary": str([(type(node).__name__, getattr(node, 'name', None)) 
                                      for node in parsed_ast.body])
                }
            except SyntaxError:
                # If not valid Python, treat as text
                representation_format = "text"
                structure = {
                    "lines": input_text.count('\n') + 1,
                    "tokens": len(input_text.split())
                }
                sample = {
                    "original": input_text[:100] + ("..." if len(input_text) > 100 else "")
                }
                
        elif env_type == "structured_data":
            # For structured data, create a structured representation
            import json
            representation_format = "json"
            try:
                # Try to parse as JSON
                parsed_json = json.loads(input_text)
                structure = {
                    "type": type(parsed_json).__name__,
                    "depth": self._get_json_depth(parsed_json),
                    "keys": list(parsed_json.keys()) if isinstance(parsed_json, dict) else None,
                    "length": len(parsed_json) if hasattr(parsed_json, '__len__') else 1
                }
                sample = {
                    "original": input_text[:100] + ("..." if len(input_text) > 100 else ""),
                    "parsed": str(parsed_json)[:100] + ("..." if str(parsed_json) > 100 else "")
                }
            except (json.JSONDecodeError, TypeError):
                # If not valid JSON, treat as text
                representation_format = "text"
                structure = {
                    "lines": input_text.count('\n') + 1,
                    "tokens": len(input_text.split())
                }
                sample = {
                    "original": input_text[:100] + ("..." if len(input_text) > 100 else "")
                }
                
        elif env_type == "web_content":
            # For web content, create a URL representation
            import re
            representation_format = "url"
            url_match = re.search(r'(https?://[^\s]+)', input_text)
            if url_match:
                url = url_match.group(1)
                structure = {
                    "url": url,
                    "domain": url.split('/')[2] if '://' in url else None,
                    "protocol": url.split('://')[0] if '://' in url else None
                }
                sample = {
                    "original": input_text,
                    "extracted_url": url
                }
            else:
                # Fallback to text
                representation_format = "text"
                structure = {
                    "lines": input_text.count('\n') + 1,
                    "tokens": len(input_text.split())
                }
                sample = {
                    "original": input_text[:100] + ("..." if len(input_text) > 100 else "")
                }
                
        else:
            # Default text processing
            representation_format = "text"
            words = input_text.split()
            structure = {
                "lines": input_text.count('\n') + 1,
                "tokens": len(words),
                "avg_token_length": sum(len(w) for w in words) / len(words) if words else 0
            }
            sample = {
                "original": input_text[:100] + ("..." if len(input_text) > 100 else ""),
                "tokens": words[:10] + (["..."] if len(words) > 10 else [])
            }
        
        return {
            "format": representation_format,
            "structure": structure,
            "sample": sample,
            "created_at": datetime.now().isoformat()
        }
    
    def _validate_environment(self, environment: dict, representation: dict, input_text: str) -> dict:
        """Validate the constructed environment and representation against the input."""
        validation_results = {
            "is_valid": True,
            "coverage": 1.0,
            "confidence": 0.8,
            "issues": []
        }
        
        # Check if environment type matches representation format
        expected_formats = {
            "numerical": ["float", "numeric_sequence"],
            "code_execution": ["code_ast", "text"],
            "structured_data": ["json", "text"],
            "web_content": ["url", "text"],
            "text_processing": ["text"]
        }
        
        if environment["type"] in expected_formats:
            if representation["format"] not in expected_formats[environment["type"]]:
                validation_results["is_valid"] = False
                validation_results["confidence"] = 0.5
                validation_results["issues"].append(
                    f"Representation format '{representation['format']}' doesn't match environment type '{environment['type']}'"
                )
        
        # Check for empty or minimal representation
        if not representation["structure"]:
            validation_results["is_valid"] = False
            validation_results["confidence"] = 0.3
            validation_results["issues"].append("Empty representation structure")
        
        # Check input coverage
        if len(input_text) > 0:
            # Calculate how much of the input is represented in the sample
            sample_text = str(representation["sample"].get("original", ""))
            if not sample_text:
                validation_results["coverage"] = 0.0
            elif len(sample_text) < len(input_text) and "..." in sample_text:
                # Estimate coverage based on sample length
                validation_results["coverage"] = len(sample_text.replace("...", "")) / len(input_text)
            
            if validation_results["coverage"] < 0.5:
                validation_results["issues"].append(f"Low input coverage: {validation_results['coverage']:.2f}")
        
        return validation_results
    
    def _handle_retrieve_code(self, task: Task) -> None:
        """Handle retrieving code based on a query."""
        try:
            # Extract query from description
            query_match = re.search(r'[\'"]([^\'"]+)[\'"]', task.description)
            if not query_match:
                task.fail("Could not extract query from task description")
                return
                
            query = query_match.group(1)
            task.update_progress(0.3)
            
            # Initialize the agent if needed
            agent = None
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if isinstance(attr, R1Agent):
                    agent = attr
                    break
            
            if not agent:
                # Create a new agent instance
                agent = R1Agent()
            
            # Retrieve code
            results = agent.retrieve_code(query)
            
            task.update_progress(0.9)
            
            if results:
                # Format results
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "type": result["type"],
                        "name": result["name"],
                        "start_line": result["start_line"],
                        "end_line": result["end_line"],
                        "content_preview": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"]
                    })
                
                task.complete({
                    "status": "success",
                    "query": query,
                    "results_count": len(results),
                    "results": formatted_results,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                task.complete({
                    "status": "success",
                    "query": query,
                    "results_count": 0,
                    "message": "No matching code found",
                    "timestamp": datetime.now().isoformat()
                })
                
            logger.info(f"[SmartTaskProcessor] Retrieved {len(results)} code chunks matching '{query}'")
            
        except Exception as e:
            task.fail(f"Error retrieving code: {str(e)}")
    
    def _handle_modify_code(self, task: Task) -> None:
        """Handle modifying code."""
        try:
            # Extract parameters from description
            element_type_match = re.search(r'(function|method|class) [\'"]([^\'"]+)[\'"]', task.description, re.IGNORECASE)
            if not element_type_match:
                task.fail("Could not extract element type and name from task description")
                return
                
            element_type = element_type_match.group(1).lower()
            name = element_type_match.group(2)
            
            # Get new code from task result or parent task
            new_code = None
            if task.result and isinstance(task.result, dict) and "new_code" in task.result:
                new_code = task.result["new_code"]
            elif task.parent_id:
                parent_task = self.memory_store.get_task(task.parent_id)
                if parent_task and parent_task.result and isinstance(parent_task.result, dict):
                    new_code = parent_task.result.get("new_code")
            
            if not new_code:
                task.fail("No new code provided for modification")
                return
                
            task.update_progress(0.3)
            
            # Initialize the agent if needed
            agent = None
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if isinstance(attr, R1Agent):
                    agent = attr
                    break
            
            if not agent:
                # Create a new agent instance
                agent = R1Agent()
            
            # Modify code
            result = agent.modify_code(element_type, name, new_code)
            
            task.update_progress(0.9)
            
            if result:
                task.complete({
                    "status": "success",
                    "element_type": element_type,
                    "name": name,
                    "message": f"Successfully modified {element_type} {name}",
                    "timestamp": datetime.now().isoformat()
                })
                logger.info(f"[SmartTaskProcessor] Successfully modified {element_type} {name}")
            else:
                task.fail(f"Failed to modify {element_type} {name}")
                
        except Exception as e:
            task.fail(f"Error modifying code: {str(e)}")
    
    def _handle_analyze_code(self, task: Task) -> None:
        """Handle analyzing code for potential improvements."""
        try:
            # Extract query from description
            query_match = re.search(r'[\'"]([^\'"]+)[\'"]', task.description)
            if not query_match:
                task.fail("Could not extract query from task description")
                return
                
            query = query_match.group(1)
            task.update_progress(0.2)
            
            # Initialize the agent if needed
            agent = None
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if isinstance(attr, R1Agent):
                    agent = attr
                    break
            
            if not agent:
                # Create a new agent instance
                agent = R1Agent()
            
            # Retrieve code
            results = agent.retrieve_code(query)
            
            if not results:
                task.complete({
                    "status": "success",
                    "query": query,
                    "results_count": 0,
                    "message": "No matching code found to analyze",
                    "timestamp": datetime.now().isoformat()
                })
                return
                
            task.update_progress(0.5)
            
            # Analyze the code using the LLM
            code_to_analyze = results[0]["content"]  # Analyze the first matching result
            
            messages = [
                {"role": "system", "content": "You are an expert code reviewer. Analyze the provided code and suggest improvements."},
                {"role": "user", "content": f"Analyze this code and suggest improvements:\n\n{code_to_analyze}"}
            ]
            
            response = agent.client.chat.completions.create(
                model="o3-mini",
                messages=messages,
                temperature=0.2
            )
            
            analysis = response.choices[0].message.content
            
            task.update_progress(0.9)
            
            task.complete({
                "status": "success",
                "query": query,
                "analyzed_element": {
                    "type": results[0]["type"],
                    "name": results[0]["name"]
                },
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"[SmartTaskProcessor] Analyzed {results[0]['type']} {results[0]['name']}")
            
        except Exception as e:
            task.fail(f"Error analyzing code: {str(e)}")
    
    # ===== NEWS AND RESEARCH TASK HANDLERS =====
    
    def _handle_search_news(self, task: Task) -> None:
        """Handle searching for news articles on a specific topic."""
        try:
            # Extract search query from task description
            search_match = re.search(r'news about ([^\'\"]+)', task.description, re.IGNORECASE)
            if not search_match:
                search_match = re.search(r'search for ([^\'\"]+)', task.description, re.IGNORECASE)
            
            if not search_match:
                task.fail("Could not extract search query from task description")
                return
                
            query = search_match.group(1).strip()
            task.update_progress(0.2)
            
            logger.info(f"[SmartTaskProcessor] Searching for news about: {query}")
            
            # Simulate news API call
            # In a real implementation, this would use a news API like NewsAPI, GDELT, or similar
            news_results = self._simulate_news_search(query)
            
            task.update_progress(0.7)
            
            # Process and summarize results
            summary = self._summarize_news_results(news_results)
            
            task.update_progress(0.9)
            
            task.complete({
                "status": "success",
                "query": query,
                "article_count": len(news_results),
                "articles": news_results[:5],  # Return top 5 articles
                "summary": summary,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"[SmartTaskProcessor] Found {len(news_results)} news articles about '{query}'")
            
        except Exception as e:
            task.fail(f"Error searching news: {str(e)}")
    
    def _simulate_news_search(self, query: str) -> List[Dict[str, Any]]:
        """Simulate searching for news articles (would use a real API in production)."""
        # This is a simulation - in a real implementation, you would use a news API
        import random
        from datetime import datetime, timedelta
        
        # Seed with query to get consistent results for the same query
        random.seed(hash(query) % 10000)
        
        # Generate realistic-looking news results
        results = []
        topics = {
            "climate change": ["UN Climate Report", "Extreme Weather", "Carbon Emissions", "Renewable Energy", "Climate Policy"],
            "tech": ["AI Developments", "Big Tech Regulation", "Startup Funding", "Cybersecurity", "Product Launches"],
            "finance": ["Market Analysis", "Interest Rates", "Banking Sector", "Investment Trends", "Economic Indicators"],
            "health": ["Medical Research", "Healthcare Policy", "Disease Outbreaks", "Wellness Trends", "Pharmaceutical Developments"],
            "politics": ["Election News", "Policy Debates", "International Relations", "Government Decisions", "Political Movements"]
        }
        
        # Find the most relevant topic category
        relevant_topics = []
        for topic, subtopics in topics.items():
            if topic in query.lower():
                relevant_topics = subtopics
                break
        
        if not relevant_topics:
            # Use a default category if no match
            relevant_topics = random.choice(list(topics.values()))
        
        # Generate between 5-15 articles
        num_articles = random.randint(5, 15)
        
        sources = ["The New York Times", "Reuters", "Associated Press", "Bloomberg", "The Guardian", 
                  "The Washington Post", "BBC News", "CNN", "CNBC", "The Wall Street Journal"]
        
        for i in range(num_articles):
            # Generate a random date within the last week
            days_ago = random.randint(0, 7)
            hours_ago = random.randint(0, 23)
            published_date = datetime.now() - timedelta(days=days_ago, hours=hours_ago)
            
            # Create article with realistic metadata
            article = {
                "title": f"{random.choice(relevant_topics)}: {query.title()} {random.choice(['Impact', 'Analysis', 'Report', 'Update', 'Development'])}",
                "source": random.choice(sources),
                "published_date": published_date.isoformat(),
                "url": f"https://example.com/news/{query.lower().replace(' ', '-')}-{i}",
                "snippet": f"Latest developments regarding {query} show significant {random.choice(['progress', 'challenges', 'changes', 'trends', 'concerns'])}. Experts suggest this could {random.choice(['impact', 'influence', 'affect', 'transform'])} the {random.choice(['industry', 'sector', 'field', 'market', 'community'])}."
            }
            
            results.append(article)
        
        # Sort by recency
        results.sort(key=lambda x: x["published_date"], reverse=True)
        
        return results
    
    def _summarize_news_results(self, articles: List[Dict[str, Any]]) -> str:
        """Create a summary of news articles."""
        if not articles:
            return "No relevant news articles found."
            
        # Count sources for diversity analysis
        sources = {}
        for article in articles:
            source = article.get("source", "Unknown")
            sources[source] = sources.get(source, 0) + 1
        
        # Get publication date range
        dates = [datetime.fromisoformat(article["published_date"]) for article in articles if "published_date" in article]
        date_range = ""
        if dates:
            oldest = min(dates)
            newest = max(dates)
            if oldest.date() == newest.date():
                date_range = f"all from {oldest.strftime('%B %d, %Y')}"
            else:
                date_range = f"from {oldest.strftime('%B %d, %Y')} to {newest.strftime('%B %d, %Y')}"
        
        # Create summary
        summary = f"Found {len(articles)} relevant news articles {date_range}. "
        summary += f"Sources include {', '.join(list(sources.keys())[:3])}"
        if len(sources) > 3:
            summary += f" and {len(sources) - 3} others"
        summary += ". "
        
        # Add top headlines
        if articles:
            summary += "Top headlines:\n"
            for i, article in enumerate(articles[:3]):
                summary += f"- {article['title']} ({article['source']})\n"
        
        return summary
    
    def _handle_research_compilation(self, task: Task) -> None:
        """Handle compiling research on a specific topic."""
        try:
            # Extract research topic from task description
            topic_match = re.search(r'research (?:on|about) ([^\'\"]+)', task.description, re.IGNORECASE)
            if not topic_match:
                topic_match = re.search(r'compile (?:a )?(?:research )?(?:report )?(?:on|about) ([^\'\"]+)', task.description, re.IGNORECASE)
            
            if not topic_match:
                task.fail("Could not extract research topic from task description")
                return
                
            topic = topic_match.group(1).strip()
            task.update_progress(0.1)
            
            logger.info(f"[SmartTaskProcessor] Compiling research on: {topic}")
            
            # Simulate academic database search
            academic_papers = self._simulate_academic_search(topic)
            task.update_progress(0.3)
            
            # Simulate news search for recent developments
            news_articles = self._simulate_news_search(topic)
            task.update_progress(0.5)
            
            # Simulate expert opinions
            expert_opinions = self._simulate_expert_opinions(topic)
            task.update_progress(0.7)
            
            # Compile research report
            report = self._compile_research_report(topic, academic_papers, news_articles, expert_opinions)
            task.update_progress(0.9)
            
            task.complete({
                "status": "success",
                "topic": topic,
                "academic_sources": len(academic_papers),
                "news_sources": len(news_articles),
                "expert_opinions": len(expert_opinions),
                "report": report,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"[SmartTaskProcessor] Compiled research report on '{topic}' with {len(academic_papers)} academic sources")
            
        except Exception as e:
            task.fail(f"Error compiling research: {str(e)}")
    
    def _simulate_academic_search(self, topic: str) -> List[Dict[str, Any]]:
        """Simulate searching academic databases for papers on a topic."""
        import random
        from datetime import datetime, timedelta
        
        # Seed with topic to get consistent results
        random.seed(hash(topic) % 10000)
        
        # Generate between 5-20 academic papers
        num_papers = random.randint(5, 20)
        results = []
        
        journals = ["Nature", "Science", "PNAS", "The Lancet", "JAMA", "IEEE Transactions", 
                   "Journal of Finance", "Psychological Review", "Cell", "Annual Review"]
                   
        authors = ["Smith et al.", "Johnson et al.", "Williams et al.", "Brown et al.", "Jones et al.",
                  "Miller et al.", "Davis et al.", "Garcia et al.", "Rodriguez et al.", "Wilson et al."]
        
        for i in range(num_papers):
            # Generate a random date within the last 3 years
            years_ago = random.randint(0, 3)
            months_ago = random.randint(0, 11)
            published_date = datetime.now() - timedelta(days=years_ago*365 + months_ago*30)
            
            # Create paper with realistic metadata
            paper = {
                "title": f"{'A' if random.random() > 0.5 else 'The'} {random.choice(['Novel', 'Comprehensive', 'Systematic', 'Comparative', 'Experimental'])} {random.choice(['Approach to', 'Analysis of', 'Study of', 'Investigation of', 'Framework for'])} {topic.title()}",
                "authors": random.choice(authors),
                "journal": random.choice(journals),
                "published_date": published_date.strftime("%B %Y"),
                "doi": f"10.{random.randint(1000, 9999)}/{random.randint(10000, 99999)}",
                "abstract": f"This paper presents a {random.choice(['novel', 'comprehensive', 'systematic', 'comparative', 'experimental'])} {random.choice(['approach to', 'analysis of', 'study of', 'investigation of', 'framework for'])} {topic}. Our research demonstrates significant {random.choice(['findings', 'results', 'outcomes', 'implications', 'applications'])} for the field.",
                "citations": random.randint(0, 500)
            }
            
            results.append(paper)
        
        # Sort by citation count (most cited first)
        results.sort(key=lambda x: x["citations"], reverse=True)
        
        return results
    
    def _simulate_expert_opinions(self, topic: str) -> List[Dict[str, Any]]:
        """Simulate gathering expert opinions on a topic."""
        import random
        
        # Seed with topic to get consistent results
        random.seed(hash(topic) % 20000)
        
        # Generate between 3-8 expert opinions
        num_experts = random.randint(3, 8)
        results = []
        
        institutions = ["Harvard University", "Stanford University", "MIT", "Oxford University", 
                       "Cambridge University", "UC Berkeley", "Princeton University", "Yale University",
                       "Columbia University", "University of Chicago"]
                       
        expert_titles = ["Professor", "Research Director", "Chief Scientist", "Senior Fellow", 
                        "Department Chair", "Lead Researcher", "Principal Investigator", "Distinguished Fellow"]
        
        first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", 
                      "William", "Elizabeth", "David", "Susan", "Richard", "Jessica", "Joseph", "Sarah"]
                      
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", 
                     "Rodriguez", "Wilson", "Martinez", "Anderson", "Taylor", "Thomas", "Hernandez", "Moore"]
        
        perspectives = ["optimistic", "cautious", "critical", "balanced", "innovative", "traditional", "progressive", "conservative"]
        
        for i in range(num_experts):
            # Generate expert name and affiliation
            name = f"Dr. {random.choice(first_names)} {random.choice(last_names)}"
            title = random.choice(expert_titles)
            institution = random.choice(institutions)
            perspective = random.choice(perspectives)
            
            # Create expert opinion
            expert = {
                "name": name,
                "title": title,
                "affiliation": institution,
                "perspective": perspective,
                "quote": f"From my research on {topic}, I've found that {random.choice(['recent developments show promise', 'there are significant challenges ahead', 'the field is evolving rapidly', 'traditional approaches are being reconsidered', 'new methodologies are yielding interesting results'])}. {random.choice(['We need more research in this area', 'The implications are profound', 'This could transform our understanding', 'The practical applications are just beginning to emerge', 'Both opportunities and challenges lie ahead'])}."
            }
            
            results.append(expert)
        
        return results
    
    def _compile_research_report(self, topic: str, academic_papers: List[Dict[str, Any]], 
                                news_articles: List[Dict[str, Any]], expert_opinions: List[Dict[str, Any]]) -> str:
        """Compile a comprehensive research report from various sources."""
        # Create report structure
        report = f"# Research Report: {topic.title()}\n\n"
        report += f"*Compiled on {datetime.now().strftime('%B %d, %Y')}*\n\n"
        
        # Executive summary
        report += "## Executive Summary\n\n"
        report += f"This report presents a comprehensive overview of current research, news, and expert opinions on {topic}. "
        report += f"It synthesizes information from {len(academic_papers)} academic sources, {len(news_articles)} news articles, "
        report += f"and insights from {len(expert_opinions)} experts in the field.\n\n"
        
        # Key findings
        report += "## Key Findings\n\n"
        
        if academic_papers:
            report += "### Academic Research\n\n"
            for i, paper in enumerate(academic_papers[:3]):
                report += f"- **{paper['title']}** ({paper['journal']}, {paper['published_date']}): {paper['abstract'][:150]}...\n"
            report += "\n"
        
        if news_articles:
            report += "### Recent Developments\n\n"
            for i, article in enumerate(news_articles[:3]):
                report += f"- **{article['title']}** ({article['source']}, {datetime.fromisoformat(article['published_date']).strftime('%B %d, %Y')}): {article['snippet'][:150]}...\n"
            report += "\n"
        
        if expert_opinions:
            report += "### Expert Perspectives\n\n"
            for i, expert in enumerate(expert_opinions[:3]):
                report += f"- **{expert['name']}** ({expert['title']} at {expert['affiliation']}): \"{expert['quote']}\"\n"
            report += "\n"
        
        # Methodology
        report += "## Methodology\n\n"
        report += "This research compilation employed a systematic approach to gather and analyze information:\n\n"
        report += "1. **Academic Database Search**: Identified and reviewed peer-reviewed literature\n"
        report += "2. **News Analysis**: Gathered recent news articles to capture current developments\n"
        report += "3. **Expert Consultation**: Collected insights from leading authorities in the field\n"
        report += "4. **Synthesis**: Integrated findings to identify patterns, trends, and implications\n\n"
        
        # Detailed findings
        report += "## Detailed Findings\n\n"
        
        # Academic research section
        if academic_papers:
            report += "### Academic Research\n\n"
            report += f"The academic literature on {topic} reveals several important themes:\n\n"
            
            # Group papers by recency
            recent_papers = [p for p in academic_papers if "2023" in p["published_date"] or "2024" in p["published_date"]]
            older_papers = [p for p in academic_papers if p not in recent_papers]
            
            if recent_papers:
                report += "#### Recent Studies\n\n"
                for paper in recent_papers[:5]:
                    report += f"- **{paper['title']}** by {paper['authors']} ({paper['journal']}, {paper['published_date']})\n"
                    report += f"  {paper['abstract'][:200]}...\n"
                    report += f"  DOI: {paper['doi']} | Citations: {paper['citations']}\n\n"
            
            if older_papers:
                report += "#### Foundational Research\n\n"
                for paper in sorted(older_papers, key=lambda x: x["citations"], reverse=True)[:3]:
                    report += f"- **{paper['title']}** by {paper['authors']} ({paper['journal']}, {paper['published_date']})\n"
                    report += f"  {paper['abstract'][:200]}...\n"
                    report += f"  DOI: {paper['doi']} | Citations: {paper['citations']}\n\n"
        
        # News analysis section
        if news_articles:
            report += "### Current Developments\n\n"
            report += f"Recent news coverage of {topic} highlights the following developments:\n\n"
            
            for article in news_articles[:5]:
                report += f"- **{article['title']}** ({article['source']}, {datetime.fromisoformat(article['published_date']).strftime('%B %d, %Y')})\n"
                report += f"  {article['snippet']}\n"
                report += f"  Source: {article['url']}\n\n"
        
        # Expert opinions section
        if expert_opinions:
            report += "### Expert Insights\n\n"
            report += f"Leading experts in {topic} offer diverse perspectives:\n\n"
            
            for expert in expert_opinions:
                report += f"- **{expert['name']}**, {expert['title']} at {expert['affiliation']} (Perspective: {expert['perspective'].title()})\n"
                report += f"  \"{expert['quote']}\"\n\n"
        
        # Conclusions and implications
        report += "## Conclusions and Implications\n\n"
        report += f"Based on the compiled research, several conclusions can be drawn about {topic}:\n\n"
        report += "1. The field is characterized by [observation about current state]\n"
        report += "2. Key challenges include [challenges identified in research]\n"
        report += "3. Opportunities exist in [areas of potential identified by experts]\n"
        report += "4. Future developments are likely to [predictions based on trends]\n\n"
        
        report += "## References\n\n"
        report += f"This report draws on {len(academic_papers)} academic papers, {len(news_articles)} news articles, and insights from {len(expert_opinions)} experts. Full reference list available upon request.\n"
        
        return report
    
    def _handle_sports_statistics(self, task: Task) -> None:
        """Handle sports statistics analysis tasks."""
        try:
            # Determine which sport to analyze
            sport = None
            if "NBA" in task.description:
                sport = "NBA"
            elif "NFL" in task.description:
                sport = "NFL"
            elif "MLB" in task.description:
                sport = "MLB"
            elif "Premier League" in task.description or "soccer" in task.description.lower():
                sport = "Soccer"
            elif "NHL" in task.description or "hockey" in task.description.lower():
                sport = "Hockey"
            else:
                # Default to a general sports analysis
                sport = "General"
            
            task.update_progress(0.2)
            logger.info(f"[SmartTaskProcessor] Analyzing {sport} statistics")
            
            # Simulate retrieving sports data
            sports_data = self._simulate_sports_data(sport)
            task.update_progress(0.5)
            
            # Analyze the data
            analysis = self._analyze_sports_data(sport, sports_data)
            task.update_progress(0.8)
            
            # Create visualizations (simulated)
            visualizations = self._simulate_sports_visualizations(sport, sports_data)
            task.update_progress(0.9)
            
            task.complete({
                "status": "success",
                "sport": sport,
                "data_points": len(sports_data),
                "analysis": analysis,
                "visualizations": visualizations,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"[SmartTaskProcessor] Completed {sport} statistics analysis")
            
        except Exception as e:
            task.fail(f"Error analyzing sports statistics: {str(e)}")
    
    def _simulate_sports_data(self, sport: str) -> List[Dict[str, Any]]:
        """Simulate retrieving sports statistics data."""
        import random
        
        # Seed with sport to get consistent results
        random.seed(hash(sport) % 30000)
        
        data = []
        
        if sport == "NBA":
            # Generate NBA player statistics
            players = ["LeBron James", "Stephen Curry", "Kevin Durant", "Giannis Antetokounmpo", 
                      "Nikola Jokić", "Joel Embiid", "Luka Dončić", "Jayson Tatum", 
                      "Ja Morant", "Devin Booker", "Trae Young", "Anthony Davis"]
            
            teams = ["Lakers", "Warriors", "Nets", "Bucks", "Nuggets", "76ers", "Mavericks", 
                    "Celtics", "Grizzlies", "Suns", "Hawks", "Lakers"]
            
            for i, player in enumerate(players):
                # Generate realistic NBA stats
                games_played = random.randint(65, 82)
                minutes_per_game = random.uniform(28.0, 38.0)
                
                player_data = {
                    "name": player,
                    "team": teams[i],
                    "position": random.choice(["PG", "SG", "SF", "PF", "C"]),
                    "games_played": games_played,
                    "minutes_per_game": round(minutes_per_game, 1),
                    "points_per_game": round(random.uniform(15.0, 32.0), 1),
                    "rebounds_per_game": round(random.uniform(4.0, 12.0), 1),
                    "assists_per_game": round(random.uniform(3.0, 11.0), 1),
                    "steals_per_game": round(random.uniform(0.5, 2.5), 1),
                    "blocks_per_game": round(random.uniform(0.2, 2.0), 1),
                    "field_goal_percentage": round(random.uniform(0.42, 0.58), 3),
                    "three_point_percentage": round(random.uniform(0.32, 0.45), 3),
                    "free_throw_percentage": round(random.uniform(0.70, 0.92), 3),
                    "player_efficiency_rating": round(random.uniform(15.0, 30.0), 1),
                    "true_shooting_percentage": round(random.uniform(0.52, 0.68), 3),
                    "usage_rate": round(random.uniform(20.0, 35.0), 1),
                    "win_shares": round(random.uniform(3.0, 15.0), 1)
                }
                
                data.append(player_data)
                
        elif sport == "NFL":
            # Generate NFL quarterback statistics
            quarterbacks = ["Patrick Mahomes", "Josh Allen", "Joe Burrow", "Lamar Jackson", 
                           "Justin Herbert", "Aaron Rodgers", "Tom Brady", "Matthew Stafford", 
                           "Dak Prescott", "Russell Wilson", "Kyler Murray", "Derek Carr"]
            
            teams = ["Chiefs", "Bills", "Bengals", "Ravens", "Chargers", "Packers", "Buccaneers", 
                    "Rams", "Cowboys", "Broncos", "Cardinals", "Raiders"]
            
            for i, qb in enumerate(quarterbacks):
                # Generate realistic NFL QB stats
                games_played = random.randint(15, 17)
                
                qb_data = {
                    "name": qb,
                    "team": teams[i],
                    "position": "QB",
                    "games_played": games_played,
                    "completions": random.randint(300, 450),
                    "attempts": random.randint(450, 650),
                    "completion_percentage": round(random.uniform(60.0, 72.0), 1),
                    "passing_yards": random.randint(3500, 5000),
                    "yards_per_attempt": round(random.uniform(6.5, 9.0), 1),
                    "passing_touchdowns": random.randint(20, 45),
                    "interceptions": random.randint(5, 15),
                    "passer_rating": round(random.uniform(85.0, 115.0), 1),
                    "rushing_attempts": random.randint(30, 120),
                    "rushing_yards": random.randint(100, 800),
                    "rushing_touchdowns": random.randint(0, 8),
                    "sacks": random.randint(15, 45),
                    "fumbles": random.randint(2, 10),
                    "qbr": round(random.uniform(50.0, 80.0), 1)
                }
                
                data.append(qb_data)
                
        elif sport == "Soccer":
            # Generate soccer player statistics
            players = ["Lionel Messi", "Cristiano Ronaldo", "Kylian Mbappé", "Erling Haaland", 
                      "Kevin De Bruyne", "Mohamed Salah", "Robert Lewandowski", "Neymar Jr", 
                      "Virgil van Dijk", "Karim Benzema", "Sadio Mané", "Harry Kane"]
            
            teams = ["Inter Miami", "Al-Nassr", "PSG", "Manchester City", "Manchester City", "Liverpool", 
                    "Barcelona", "Al-Hilal", "Liverpool", "Real Madrid", "Bayern Munich", "Tottenham"]
            
            for i, player in enumerate(players):
                # Generate realistic soccer stats
                matches_played = random.randint(30, 38)
                
                player_data = {
                    "name": player,
                    "team": teams[i],
                    "position": random.choice(["Forward", "Midfielder", "Defender", "Goalkeeper"]),
                    "matches_played": matches_played,
                    "goals": random.randint(5, 35),
                    "assists": random.randint(2, 20),
                    "minutes_played": matches_played * random.randint(70, 90),
                    "shots": random.randint(40, 150),
                    "shots_on_target": random.randint(20, 80),
                    "pass_completion": round(random.uniform(75.0, 92.0), 1),
                    "key_passes": random.randint(20, 100),
                    "dribbles_completed": random.randint(20, 120),
                    "tackles": random.randint(10, 100),
                    "interceptions": random.randint(5, 80),
                    "fouls_committed": random.randint(15, 60),
                    "fouls_drawn": random.randint(15, 80),
                    "yellow_cards": random.randint(2, 10),
                    "red_cards": random.randint(0, 2),
                    "rating": round(random.uniform(6.5, 8.5), 1)
                }
                
                data.append(player_data)
                
        elif sport == "MLB":
            # Generate MLB player statistics
            players = ["Shohei Ohtani", "Aaron Judge", "Mike Trout", "Mookie Betts", 
                      "Freddie Freeman", "Vladimir Guerrero Jr.", "Juan Soto", "Bryce Harper", 
                      "Fernando Tatis Jr.", "José Ramírez", "Rafael Devers", "Yordan Álvarez"]
            
            teams = ["Dodgers", "Yankees", "Angels", "Dodgers", "Dodgers", "Blue Jays", 
                    "Padres", "Phillies", "Padres", "Guardians", "Red Sox", "Astros"]
            
            for i, player in enumerate(players):
                # Generate realistic MLB stats
                games_played = random.randint(140, 162)
                at_bats = games_played * random.randint(3, 5)
                hits = int(at_bats * random.uniform(0.240, 0.330))
                
                player_data = {
                    "name": player,
                    "team": teams[i],
                    "position": random.choice(["1B", "2B", "3B", "SS", "LF", "CF", "RF", "C", "DH"]),
                    "games_played": games_played,
                    "at_bats": at_bats,
                    "runs": random.randint(70, 120),
                    "hits": hits,
                    "doubles": random.randint(20, 45),
                    "triples": random.randint(0, 10),
                    "home_runs": random.randint(15, 50),
                    "runs_batted_in": random.randint(60, 130),
                    "stolen_bases": random.randint(5, 40),
                    "caught_stealing": random.randint(1, 15),
                    "walks": random.randint(40, 100),
                    "strikeouts": random.randint(80, 200),
                    "batting_average": round(hits / at_bats, 3),
                    "on_base_percentage": round((hits + random.randint(40, 100)) / (at_bats + random.randint(150, 250)), 3),
                    "slugging_percentage": round((hits + random.randint(100, 300)) / at_bats, 3),
                    "ops": round(random.uniform(0.700, 1.050), 3),
                    "war": round(random.uniform(2.0, 8.0), 1)
                }
                
                data.append(player_data)
                
        else:
            # Generate generic sports data
            athletes = ["Athlete 1", "Athlete 2", "Athlete 3", "Athlete 4", "Athlete 5",
                       "Athlete 6", "Athlete 7", "Athlete 8", "Athlete 9", "Athlete 10"]
            
            for athlete in athletes:
                # Generate generic stats
                athlete_data = {
                    "name": athlete,
                    "team": f"Team {random.randint(1, 10)}",
                    "games_played": random.randint(10, 50),
                    "points_scored": random.randint(100, 1000),
                    "performance_rating": round(random.uniform(5.0, 10.0), 1),
                    "wins": random.randint(5, 30),
                    "losses": random.randint(5, 30)
                }
                
                data.append(athlete_data)
        
        return data
    
    def _analyze_sports_data(self, sport: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sports statistics data."""
        analysis = {
            "sport": sport,
            "total_players": len(data),
            "summary": f"Analysis of {len(data)} {sport} players/teams",
            "top_performers": [],
            "statistical_insights": [],
            "trends": []
        }
        
        if not data:
            return analysis
        
        # Sort data by key metrics based on sport
        if sport == "NBA":
            # Sort by points per game
            sorted_data = sorted(data, key=lambda x: x.get("points_per_game", 0), reverse=True)
            
            # Top scorers
            analysis["top_performers"] = [
                {
                    "category": "Top Scorers",
                    "players": [
                        {"name": player["name"], "team": player["team"], "value": player["points_per_game"], "unit": "PPG"}
                        for player in sorted_data[:5]
                    ]
                }
            ]
            
            # Sort by other metrics
            rebounds_leaders = sorted(data, key=lambda x: x.get("rebounds_per_game", 0), reverse=True)[:3]
            assists_leaders = sorted(data, key=lambda x: x.get("assists_per_game", 0), reverse=True)[:3]
            efficiency_leaders = sorted(data, key=lambda x: x.get("player_efficiency_rating", 0), reverse=True)[:3]
            
            analysis["top_performers"].extend([
                {
                    "category": "Top Rebounders",
                    "players": [
                        {"name": player["name"], "team": player["team"], "value": player["rebounds_per_game"], "unit": "RPG"}
                        for player in rebounds_leaders
                    ]
                },
                {
                    "category": "Top Playmakers",
                    "players": [
                        {"name": player["name"], "team": player["team"], "value": player["assists_per_game"], "unit": "APG"}
                        for player in assists_leaders
                    ]
                },
                {
                    "category": "Most Efficient",
                    "players": [
                        {"name": player["name"], "team": player["team"], "value": player["player_efficiency_rating"], "unit": "PER"}
                        for player in efficiency_leaders
                    ]
                }
            ])
            
            # Statistical insights
            avg_ppg = sum(player.get("points_per_game", 0) for player in data) / len(data)
            avg_rpg = sum(player.get("rebounds_per_game", 0) for player in data) / len(data)
            avg_apg = sum(player.get("assists_per_game", 0) for player in data) / len(data)
            
            analysis["statistical_insights"] = [
                {"metric": "Average Points", "value": round(avg_ppg, 1), "unit": "PPG"},
                {"metric": "Average Rebounds", "value": round(avg_rpg, 1), "unit": "RPG"},
                {"metric": "Average Assists", "value": round(avg_apg, 1), "unit": "APG"}
            ]
            
            # Trends and observations
            analysis["trends"] = [
                "Scoring efficiency continues to be a key factor in player evaluation",
                "Versatile players who contribute across multiple statistical categories are highly valued",
                "Three-point shooting remains a critical skill in the modern NBA"
            ]
            
        elif sport == "NFL":
            # Sort by passing yards
            sorted_data = sorted(data, key=lambda x: x.get("passing_yards", 0), reverse=True)
            
            # Top passers
            analysis["top_performers"] = [
                {
                    "category": "Top Passers",
                    "players": [
                        {"name": player["name"], "team": player["team"], "value": player["passing_yards"], "unit": "yards"}
                        for player in sorted_data[:5]
                    ]
                }
            ]
            
            # Sort by other metrics
            td_leaders = sorted(data, key=lambda x: x.get("passing_touchdowns", 0), reverse=True)[:3]
            rating_leaders = sorted(data, key=lambda x: x.get("passer_rating", 0), reverse=True)[:3]
            qbr_leaders = sorted(data, key=lambda x: x.get("qbr", 0), reverse=True)[:3]
            
            analysis["top_performers"].extend([
                {
                    "category": "Touchdown Leaders",
                    "players": [
                        {"name": player["name"], "team": player["team"], "value": player["passing_touchdowns"], "unit": "TDs"}
                        for player in td_leaders
                    ]
                },
                {
                    "category": "Passer Rating Leaders",
                    "players": [
                        {"name": player["name"], "team": player["team"], "value": player["passer_rating"], "unit": "rating"}
                        for player in rating_leaders
                    ]
                },
                {
                    "category": "QBR Leaders",
                    "players": [
                        {"name": player["name"], "team": player["team"], "value": player["qbr"], "unit": "QBR"}
                        for player in qbr_leaders
                    ]
                }
            ])
            
            # Statistical insights
            avg_yards = sum(player.get("passing_yards", 0) for player in data) / len(data)
            avg_tds = sum(player.get("passing_touchdowns", 0) for player in data) / len(data)
            avg_ints = sum(player.get("interceptions", 0) for player in data) / len(data)
            
            analysis["statistical_insights"] = [
                {"metric": "Average Passing Yards", "value": round(avg_yards, 1), "unit": "yards"},
                {"metric": "Average Touchdowns", "value": round(avg_tds, 1), "unit": "TDs"},
                {"metric": "Average Interceptions", "value": round(avg_ints, 1), "unit": "INTs"},
                {"metric": "TD to INT Ratio", "value": round(avg_tds / avg_ints, 2), "unit": "ratio"}
            ]
            
            # Trends and observations
            analysis["trends"] = [
                "Mobile quarterbacks continue to reshape offensive strategies",
                "Efficiency metrics like QBR are increasingly valued over raw yardage",
                "The league continues to trend toward pass-heavy offenses"
            ]
            
        elif sport == "Soccer":
            # Sort by goals
            sorted_data = sorted(data, key=lambda x: x.get("goals", 0), reverse=True)
            
            # Top goal scorers
            analysis["top_performers"] = [
                {
                    "category": "Top Goal Scorers",
                    "players": [
                        {"name": player["name"], "team": player["team"], "value": player["goals"], "unit": "goals"}
                        for player in sorted_data[:5]
                    ]
                }
            ]
            
            # Sort by other metrics
            assist_leaders = sorted(data, key=lambda x: x.get("assists", 0), reverse=True)[:3]
            rating_leaders = sorted(data, key=lambda x: x.get("rating", 0), reverse=True)[:3]
            
            analysis["top_performers"].extend([
                {
                    "category": "Top Assisters",
                    "players": [
                        {"name": player["name"], "team": player["team"], "value": player["assists"], "unit": "assists"}
                        for player in assist_leaders
                    ]
                },
                {
                    "category": "Highest Rated",
                    "players": [
                        {"name": player["name"], "team": player["team"], "value": player["rating"], "unit": "rating"}
                        for player in rating_leaders
                    ]
                }
            ])
            
            # Statistical insights
            avg_goals = sum(player.get("goals", 0) for player in data) / len(data)
            avg_assists = sum(player.get("assists", 0) for player in data) / len(data)
            avg_rating = sum(player.get("rating", 0) for player in data) / len(data)
            
            analysis["statistical_insights"] = [
                {"metric": "Average Goals", "value": round(avg_goals, 1), "unit": "goals"},
                {"metric": "Average Assists", "value": round(avg_assists, 1), "unit": "assists"},
                {"metric": "Average Rating", "value": round(avg_rating, 1), "unit": "rating"}
            ]
            
            # Trends and observations
            analysis["trends"] = [
                "Pressing and high-intensity play continues to dominate tactical approaches",
                "Versatile forwards who can both score and create are highly valued",
                "Data analytics is increasingly influencing player recruitment and tactical decisions"
            ]
            
        elif sport == "MLB":
            # Sort by batting average
            sorted_data = sorted(data, key=lambda x: x.get("batting_average", 0), reverse=True)
            
            # Top hitters
            analysis["top_performers"] = [
                {
                    "category": "Top Hitters",
                    "players": [
                        {"name": player["name"], "team": player["team"], "value": player["batting_average"], "unit": "AVG"}
                        for player in sorted_data[:5]
                    ]
                }
            ]
            
            # Sort by other metrics
            hr_leaders = sorted(data, key=lambda x: x.get("home_runs", 0), reverse=True)[:3]
            rbi_leaders = sorted(data, key=lambda x: x.get("runs_batted_in", 0), reverse=True)[:3]
            ops_leaders = sorted(data, key=lambda x: x.get("ops", 0), reverse=True)[:3]
            
            analysis["top_performers"].extend([
                {
                    "category": "Home Run Leaders",
                    "players": [
                        {"name": player["name"], "team": player["team"], "value": player["home_runs"], "unit": "HR"}
                        for player in hr_leaders
                    ]
                },
                {
                    "category": "RBI Leaders",
                    "players": [
                        {"name": player["name"], "team": player["team"], "value": player["runs_batted_in"], "unit": "RBI"}
                        for player in rbi_leaders
                    ]
                },
                {
                    "category": "OPS Leaders",
                    "players": [
                        {"name": player["name"], "team": player["team"], "value": player["ops"], "unit": "OPS"}
                        for player in ops_leaders
                    ]
                }
            ])
            
            # Statistical insights
            avg_ba = sum(player.get("batting_average", 0) for player in data) / len(data)
            avg_hr = sum(player.get("home_runs", 0) for player in data) / len(data)
            avg_rbi = sum(player.get("runs_batted_in", 0) for player in data) / len(data)
            
            analysis["statistical_insights"] = [
                {"metric": "Average Batting Average", "value": round(avg_ba, 3), "unit": "AVG"},
                {"metric": "Average Home Runs", "value": round(avg_hr, 1), "unit": "HR"},
                {"metric": "Average RBIs", "value": round(avg_rbi, 1), "unit": "RBI"}
            ]
            
            # Trends and observations
            analysis["trends"] = [
                "Power hitting continues to be emphasized in modern baseball",
                "Advanced metrics like OPS and WAR are increasingly used for player evaluation",
                "Versatile players who can contribute both offensively and defensively are highly valued"
            ]
            
        else:
            # Generic analysis for other sports
            sorted_data = sorted(data, key=lambda x: x.get("performance_rating", 0), reverse=True)
            
            analysis["top_performers"] = [
                {
                    "category": "Top Performers",
                    "players": [
                        {"name": player["name"], "team": player["team"], "value": player["performance_rating"], "unit": "rating"}
                        for player in sorted_data[:5]
                    ]
                }
            ]
            
            # Statistical insights
            avg_rating = sum(player.get("performance_rating", 0) for player in data) / len(data)
            
            analysis["statistical_insights"] = [
                {"metric": "Average Performance Rating", "value": round(avg_rating, 1), "unit": "rating"}
            ]
            
            # Trends and observations
            analysis["trends"] = [
                "Data analytics is transforming player evaluation across sports",
                "Multi-dimensional performance metrics provide more comprehensive player assessment",
                "Training methodologies continue to evolve based on performance data"
            ]
        
        return analysis
    
    def _simulate_sports_visualizations(self, sport: str, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simulate creating visualizations for sports data."""
        # In a real implementation, this would generate actual charts/graphs
        # Here we'll just return descriptions of what the visualizations would show
        
        visualizations = []
        
        if sport == "NBA":
            visualizations = [
                {
                    "title": "Points Per Game Leaders",
                    "type": "bar_chart",
                    "description": "Bar chart showing the top 10 players by points per game",
                    "data_fields": ["name", "points_per_game"]
                },
                {
                    "title": "Scoring Efficiency Comparison",
                    "type": "scatter_plot",
                    "description": "Scatter plot comparing true shooting percentage vs. usage rate",
                    "data_fields": ["name", "true_shooting_percentage", "usage_rate"]
                },
                {
                    "title": "Player Contribution Breakdown",
                    "type": "radar_chart",
                    "description": "Radar charts showing points, rebounds, assists, steals, and blocks for top players",
                    "data_fields": ["name", "points_per_game", "rebounds_per_game", "assists_per_game", "steals_per_game", "blocks_per_game"]
                }
            ]
        elif sport == "NFL":
            visualizations = [
                {
                    "title": "Quarterback Passing Yards",
                    "type": "bar_chart",
                    "description": "Bar chart showing passing yards for each quarterback",
                    "data_fields": ["name", "passing_yards"]
                },
                {
                    "title": "TD-to-INT Ratio",
                    "type": "scatter_plot",
                    "description": "Scatter plot showing touchdowns vs. interceptions for each quarterback",
                    "data_fields": ["name", "passing_touchdowns", "interceptions"]
                },
                {
                    "title": "Passer Rating vs. QBR",
                    "type": "scatter_plot",
                    "description": "Comparison of traditional passer rating vs. QBR metric",
                    "data_fields": ["name", "passer_rating", "qbr"]
                }
            ]
        elif sport == "Soccer":
            visualizations = [
                {
                    "title": "Goal Contributions",
                    "type": "stacked_bar_chart",
                    "description": "Stacked bar chart showing goals and assists for each player",
                    "data_fields": ["name", "goals", "assists"]
                },
                {
                    "title": "Shot Conversion Rate",
                    "type": "bar_chart",
                    "description": "Bar chart showing the percentage of shots that result in goals",
                    "data_fields": ["name", "goals", "shots"]
                },
                {
                    "title": "Player Performance Radar",
                    "type": "radar_chart",
                    "description": "Radar chart showing key performance metrics for top players",
                    "data_fields": ["name", "goals", "assists", "pass_completion", "dribbles_completed", "tackles"]
                }
            ]
        elif sport == "MLB":
            visualizations = [
                {
                    "title": "Batting Average Leaders",
                    "type": "bar_chart",
                    "description": "Bar chart showing batting averages for top hitters",
                    "data_fields": ["name", "batting_average"]
                },
                {
                    "title": "Power Metrics Comparison",
                    "type": "scatter_plot",
                    "description": "Scatter plot comparing home runs vs. slugging percentage",
                    "data_fields": ["name", "home_runs", "slugging_percentage"]
                },
                {
                    "title": "Offensive Production Breakdown",
                    "type": "pie_chart",
                    "description": "Breakdown of hits by type (singles, doubles, triples, home runs)",
                    "data_fields": ["name", "hits", "doubles", "triples", "home_runs"]
                }
            ]
        else:
            visualizations = [
                {
                    "title": "Performance Rating Comparison",
                    "type": "bar_chart",
                    "description": "Bar chart comparing performance ratings across athletes",
                    "data_fields": ["name", "performance_rating"]
                },
                {
                    "title": "Win-Loss Record",
                    "type": "stacked_bar_chart",
                    "description": "Stacked bar chart showing wins and losses for each athlete/team",
                    "data_fields": ["name", "wins", "losses"]
                }
            ]
        
        return visualizations
    
    def _handle_financial_analysis(self, task: Task) -> None:
        """Handle financial analysis and market research tasks."""
        try:
            # Determine what type of financial analysis to perform
            analysis_type = None
            if "stock" in task.description.lower() or "investment" in task.description.lower():
                analysis_type = "stock_analysis"
            elif "crypto" in task.description.lower() or "bitcoin" in task.description.lower():
                analysis_type = "crypto_analysis"
            elif "housing" in task.description.lower() or "real estate" in task.description.lower():
                analysis_type = "real_estate_analysis"
            elif "DCF" in task.description or "discounted cash flow" in task.description.lower():
                analysis_type = "dcf_model"
            else:
                # Default to general market analysis
                analysis_type = "market_analysis"
            
            task.update_progress(0.2)
            logger.info(f"[SmartTaskProcessor] Performing {analysis_type}")
            
            # Simulate retrieving financial data
            financial_data = self._simulate_financial_data(analysis_type)
            task.update_progress(0.5)
            
            # Analyze the data
            analysis = self._analyze_financial_data(analysis_type, financial_data)
            task.update_progress(0.8)
            
            # Generate recommendations
            recommendations = self._generate_financial_recommendations(analysis_type, analysis)
            task.update_progress(0.9)
            
            task.complete({
                "status": "success",
                "analysis_type": analysis_type,
                "data_points": len(financial_data) if isinstance(financial_data, list) else "N/A",
                "analysis": analysis,
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"[SmartTaskProcessor] Completed {analysis_type}")
            
        except Exception as e:
            task.fail(f"Error performing financial analysis: {str(e)}")
    
    def _simulate_financial_data(self, analysis_type: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Simulate retrieving financial data for analysis."""
        import random
        from datetime import datetime, timedelta
        
        # Seed with analysis type to get consistent results
        random.seed(hash(analysis_type) % 40000)
        
        if analysis_type == "stock_analysis":
            # Generate stock data for top tech companies
            companies = ["Apple", "Microsoft", "Google", "Amazon", "Meta"]
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
            
            stocks = []
            for i, company in enumerate(companies):
                # Generate realistic stock data
                current_price = random.uniform(100, 3000)
                pe_ratio = random.uniform(15, 40)
                market_cap = current_price * random.uniform(1000000000, 3000000000)
                
                # Generate price history (last 30 days)
                price_history = []
                for day in range(30, 0, -1):
                    date = (datetime.now() - timedelta(days=day)).strftime("%Y-%m-%d")
                    daily_price = current_price * (1 + random.uniform(-0.03, 0.03))
                    price_history.append({"date": date, "price": round(daily_price, 2)})
                
                # Last price is current price
                price_history.append({"date": datetime.now().strftime("%Y-%m-%d"), "price": round(current_price, 2)})
                
                stock_data = {
                    "company": company,
                    "ticker": tickers[i],
                    "current_price": round(current_price, 2),
                    "change_percent": round(random.uniform(-5, 5), 2),
                    "market_cap": round(market_cap, 2),
                    "pe_ratio": round(pe_ratio, 2),
                    "dividend_yield": round(random.uniform(0, 2.5), 2),
                    "52_week_high": round(current_price * (1 + random.uniform(0.05, 0.3)), 2),
                    "52_week_low": round(current_price * (1 - random.uniform(0.05, 0.3)), 2),
                    "analyst_rating": random.choice(["Buy", "Strong Buy", "Hold", "Sell", "Strong Buy"]),
                    "price_target": round(current_price * (1 + random.uniform(-0.2, 0.4)), 2),
                    "revenue_growth": round(random.uniform(5, 30), 2),
                    "profit_margin": round(random.uniform(10, 40), 2),
                    "price_history": price_history
                }
                
                stocks.append(stock_data)
                
            return stocks
            
        elif analysis_type == "crypto_analysis":
            # Generate cryptocurrency data
            cryptos = ["Bitcoin", "Ethereum", "Solana", "Cardano", "Binance Coin"]
            tickers = ["BTC", "ETH", "SOL", "ADA", "BNB"]
            
            crypto_data = []
            for i, crypto in enumerate(cryptos):
                # Generate realistic crypto data
                current_price = random.uniform(1, 50000)
                if crypto == "Bitcoin":
                    current_price = random.uniform(25000, 60000)
                elif crypto == "Ethereum":
                    current_price = random.uniform(1500, 4000)
                
                # Generate price history (last 30 days)
                price_history = []
                for day in range(30, 0, -1):
                    date = (datetime.now() - timedelta(days=day)).strftime("%Y-%m-%d")
                    daily_price = current_price * (1 + random.uniform(-0.08, 0.08))
                    price_history.append({"date": date, "price": round(daily_price, 2)})
                
                # Last price is current price
                price_history.append({"date": datetime.now().strftime("%Y-%m-%d"), "price": round(current_price, 2)})
                
                coin_data = {
                    "name": crypto,
                    "ticker": tickers[i],
                    "current_price": round(current_price, 2),
                    "change_24h": round(random.uniform(-15, 15), 2),
                    "change_7d": round(random.uniform(-30, 30), 2),
                    "market_cap": round(current_price * random.uniform(1000000, 1000000000), 2),
                    "volume_24h": round(current_price * random.uniform(1000000, 10000000), 2),
                    "circulating_supply": round(random.uniform(10000000, 1000000000), 0),
                    "all_time_high": round(current_price * (1 + random.uniform(0.1, 2.0)), 2),
                    "volatility_30d": round(random.uniform(30, 100), 2),
                    "correlation_to_btc": round(random.uniform(0.3, 0.9), 2),
                    "price_history": price_history
                }
                
                crypto_data.append(coin_data)
                
            return crypto_data
            
        elif analysis_type == "real_estate_analysis":
            # Generate housing market data for major cities
            cities = ["New York", "San Francisco", "Los Angeles", "Chicago", "Miami", 
                     "Seattle", "Austin", "Boston", "Denver", "Washington DC"]
            
            real_estate_data = []
            for city in cities:
                # Generate realistic housing data
                median_price = random.uniform(300000, 1500000)
                if city in ["San Francisco", "New York"]:
                    median_price = random.uniform(800000, 2000000)
                
                # Generate price history (last 12 months)
                price_history = []
                for month in range(12, 0, -1):
                    date = (datetime.now() - timedelta(days=30*month)).strftime("%Y-%m")
                    monthly_price = median_price * (1 + random.uniform(-0.02, 0.03))
                    price_history.append({"date": date, "price": round(monthly_price, 2)})
                
                # Last price is current price
                price_history.append({"date": datetime.now().strftime("%Y-%m"), "price": round(median_price, 2)})
                
                city_data = {
                    "city": city,
                    "median_home_price": round(median_price, 2),
                    "yoy_price_change": round(random.uniform(-5, 20), 2),
                    "median_rent": round(median_price * random.uniform(0.003, 0.006), 2),
                    "price_to_rent_ratio": round(random.uniform(15, 35), 2),
                    "days_on_market": round(random.uniform(10, 90), 0),
                    "inventory_months": round(random.uniform(1, 8), 1),
                    "affordability_index": round(random.uniform(50, 150), 1),
                    "population_growth": round(random.uniform(-2, 5), 1),
                    "job_growth": round(random.uniform(-1, 6), 1),
                    "price_history": price_history
                }
                
                real_estate_data.append(city_data)
                
            return real_estate_data
            
        elif analysis_type == "dcf_model":
            # Generate DCF model data for a specific company (e.g., Tesla)
            company = "Tesla"
            
            # Financial projections for next 5 years
            revenue_growth_rates = [round(random.uniform(20, 40), 2) for _ in range(5)]
            ebitda_margins = [round(random.uniform(15, 25), 2) for _ in range(5)]
            
            current_year = datetime.now().year
            projection_years = [current_year + i for i in range(1, 6)]
            
            # Current financials
            current_revenue = random.uniform(50000000000, 100000000000)
            
            # Projected financials
            financial_projections = []
            for i, year in enumerate(projection_years):
                growth_rate = revenue_growth_rates[i]
                margin = ebitda_margins[i]
                
                if i == 0:
                    revenue = current_revenue * (1 + growth_rate/100)
                else:
                    revenue = financial_projections[i-1]["revenue"] * (1 + growth_rate/100)
                
                ebitda = revenue * (margin/100)
                depreciation = revenue * random.uniform(0.05, 0.1)
                ebit = ebitda - depreciation
                taxes = ebit * 0.21  # Assuming 21% tax rate
                nopat = ebit - taxes
                capex = revenue * random.uniform(0.1, 0.2)
                change_in_nwc = revenue * random.uniform(0.01, 0.03)
                fcf = nopat + depreciation - capex - change_in_nwc
                
                projection = {
                    "year": year,
                    "revenue_growth": growth_rate,
                    "revenue": round(revenue, 2),
                    "ebitda_margin": margin,
                    "ebitda": round(ebitda, 2),
                    "depreciation": round(depreciation, 2),
                    "ebit": round(ebit, 2),
                    "taxes": round(taxes, 2),
                    "nopat": round(nopat, 2),
                    "capex": round(capex, 2),
                    "change_in_nwc": round(change_in_nwc, 2),
                    "fcf": round(fcf, 2)
                }
                
                financial_projections.append(projection)
            
            # DCF model parameters
            wacc = round(random.uniform(8, 12), 2)  # Weighted Average Cost of Capital
            terminal_growth_rate = round(random.uniform(2, 4), 2)
            
            # Terminal value calculation
            terminal_fcf = financial_projections[-1]["fcf"] * (1 + terminal_growth_rate/100)
            terminal_value = terminal_fcf / (wacc/100 - terminal_growth_rate/100)
            
            # Present value calculations
            present_value_fcf = []
            for i, projection in enumerate(financial_projections):
                pv_factor = 1 / ((1 + wacc/100) ** (i+1))
                pv_fcf = projection["fcf"] * pv_factor
                present_value_fcf.append(round(pv_fcf, 2))
            
            pv_terminal_value = terminal_value / ((1 + wacc/100) ** len(financial_projections))
            
            # Enterprise value and equity value
            enterprise_value = sum(present_value_fcf) + pv_terminal_value
            net_debt = random.uniform(-10000000000, 10000000000)  # Negative means net cash
            equity_value = enterprise_value - net_debt
            
            # Shares outstanding
            shares_outstanding = random.uniform(1000000000, 2000000000)
            
            # Fair value per share
            fair_value_per_share = equity_value / shares_outstanding
            
            # Current market price
            current_market_price = fair_value_per_share * (1 + random.uniform(-0.3, 0.3))
            
            dcf_data = {
                "company": company,
                "current_revenue": round(current_revenue, 2),
                "financial_projections": financial_projections,
                "wacc": wacc,
                "terminal_growth_rate": terminal_growth_rate,
                "terminal_value": round(terminal_value, 2),
                "present_value_fcf": present_value_fcf,
                "pv_terminal_value": round(pv_terminal_value, 2),
                "enterprise_value": round(enterprise_value, 2),
                "net_debt": round(net_debt, 2),
                "equity_value": round(equity_value, 2),
                "shares_outstanding": round(shares_outstanding, 2),
                "fair_value_per_share": round(fair_value_per_share, 2),
                "current_market_price": round(current_market_price, 2),
                "upside_downside": round(((fair_value_per_share / current_market_price) - 1) * 100, 2)
            }
            
            return dcf_data
            
        else:  # General market analysis
            # Generate general market data
            indices = ["S&P 500", "Dow Jones", "NASDAQ", "Russell 2000", "VIX"]
            
            market_data = []
            for index in indices:
                # Generate realistic index data
                current_value = random.uniform(1000, 40000)
                if index == "S&P 500":
                    current_value = random.uniform(4000, 5000)
                elif index == "Dow Jones":
                    current_value = random.uniform(30000, 40000)
                elif index == "NASDAQ":
                    current_value = random.uniform(12000, 16000)
                elif index == "VIX":
                    current_value = random.uniform(10, 30)
                
                # Generate value history (last 30 days)
                value_history = []
                for day in range(30, 0, -1):
                    date = (datetime.now() - timedelta(days=day)).strftime("%Y-%m-%d")
                    daily_value = current_value * (1 + random.uniform(-0.02, 0.02))
                    value_history.append({"date": date, "value": round(daily_value, 2)})
                
                # Last value is current value
                value_history.append({"date": datetime.now().strftime("%Y-%m-%d"), "value": round(current_value, 2)})
                
                index_data = {
                    "index": index,
                    "current_value": round(current_value, 2),
                    "change_percent": round(random.uniform(-2, 2), 2),
                    "ytd_return": round(random.uniform(-10, 20), 2),
                    "one_year_return": round(random.uniform(-15, 30), 2),
                    "value_history": value_history
                }
                
                market_data.append(index_data)
                
            # Add sector performance
            sectors = ["Technology", "Healthcare", "Financials", "Consumer Discretionary", 
                      "Communication Services", "Industrials", "Consumer Staples", 
                      "Energy", "Utilities", "Materials", "Real Estate"]
            
            sector_performance = []
            for sector in sectors:
                sector_data = {
                    "sector": sector,
                    "daily_change": round(random.uniform(-3, 3), 2),
                    "weekly_change": round(random.uniform(-5, 5), 2),
                    "monthly_change": round(random.uniform(-10, 10), 2),
                    "ytd_change": round(random.uniform(-20, 30), 2),
                    "pe_ratio": round(random.uniform(10, 30), 2)
                }
                
                sector_performance.append(sector_data)
            
            # Add economic indicators
            economic_indicators = {
                "gdp_growth": round(random.uniform(1, 4), 1),
                "inflation_rate": round(random.uniform(2, 8), 1),
                "unemployment_rate": round(random.uniform(3, 6), 1),
                "fed_funds_rate": round(random.uniform(3, 6), 2),
                "ten_year_treasury": round(random.uniform(3, 5), 2),
                "consumer_sentiment": round(random.uniform(60, 100), 1)
            }
            
            return {
                "market_indices": market_data,
                "sector_performance": sector_performance,
                "economic_indicators": economic_indicators
            }
    
    def _analyze_financial_data(self, analysis_type: str, data: Union[List[Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze financial data based on the analysis type."""
        analysis = {
            "type": analysis_type,
            "summary": "",
            "key_metrics": [],
            "trends": [],
            "risks": [],
            "opportunities": []
        }
        
        if analysis_type == "stock_analysis":
            # Analyze stock data
            stocks = data
            
            # Calculate average metrics
            avg_pe = sum(stock["pe_ratio"] for stock in stocks) / len(stocks)
            avg_dividend = sum(stock["dividend_yield"] for stock in stocks) / len(stocks)
            avg_growth = sum(stock["revenue_growth"] for stock in stocks) / len(stocks)
            
            # Find best and worst performers
            stocks_by_change = sorted(stocks, key=lambda x: x["change_percent"], reverse=True)
            best_performer = stocks_by_change[0]
            worst_performer = stocks_by_change[-1]
            
            # Summary
            analysis["summary"] = f"Analysis of {len(stocks)} major tech stocks shows an average P/E ratio of {avg_pe:.2f}, " \
                                 f"dividend yield of {avg_dividend:.2f}%, and revenue growth of {avg_growth:.2f}%. " \
                                 f"{best_performer['company']} ({best_performer['ticker']}) is the best performer with " \
                                 f"{best_performer['change_percent']}% change, while {worst_performer['company']} " \
                                 f"({worst_performer['ticker']}) is the worst performer with {worst_performer['change_percent']}% change."
            
            # Key metrics
            analysis["key_metrics"] = [
                {"name": "Average P/E Ratio", "value": round(avg_pe, 2)},
                {"name": "Average Dividend Yield", "value": f"{round(avg_dividend, 2)}%"},
                {"name": "Average Revenue Growth", "value": f"{round(avg_growth, 2)}%"},
                {"name": "Best Performer", "value": f"{best_performer['company']} ({best_performer['change_percent']}%)"},
                {"name": "Worst Performer", "value": f"{worst_performer['company']} ({worst_performer['change_percent']}%)"}
            ]
            
            # Trends
            analysis["trends"] = [
                "Tech stocks continue to show strong revenue growth despite economic headwinds",
                "Dividend yields remain relatively low compared to other sectors",
                "Valuations (P/E ratios) are above market average, reflecting growth expectations"
            ]
            
            # Risks
            analysis["risks"] = [
                "High valuations make tech stocks vulnerable to interest rate increases",
                "Regulatory scrutiny remains a concern for large tech companies",
                "Competition is intensifying in key markets like cloud computing and AI"
            ]
            
            # Opportunities
            analysis["opportunities"] = [
                "AI and machine learning continue to drive new growth opportunities",
                "Cloud computing expansion provides recurring revenue streams",
                "International markets offer growth potential for established players"
            ]
            
        elif analysis_type == "crypto_analysis":
            # Analyze cryptocurrency data
            cryptos = data
            
            # Calculate average metrics
            avg_24h_change = sum(crypto["change_24h"] for crypto in cryptos) / len(cryptos)
            avg_7d_change = sum(crypto["change_7d"] for crypto in cryptos) / len(cryptos)
            avg_volatility = sum(crypto["volatility_30d"] for crypto in cryptos) / len(cryptos)
            
            # Find best and worst performers
            cryptos_by_change = sorted(cryptos, key=lambda x: x["change_7d"], reverse=True)
            best_performer = cryptos_by_change[0]
            worst_performer = cryptos_by_change[-1]
            
            # Summary
            analysis["summary"] = f"Analysis of {len(cryptos)} major cryptocurrencies shows an average 24-hour change of {avg_24h_change:.2f}%, " \
                                 f"7-day change of {avg_7d_change:.2f}%, and 30-day volatility of {avg_volatility:.2f}%. " \
                                 f"{best_performer['name']} ({best_performer['ticker']}) is the best performer over 7 days with " \
                                 f"{best_performer['change_7d']}% change, while {worst_performer['name']} " \
                                 f"({worst_performer['ticker']}) is the worst performer with {worst_performer['change_7d']}% change."
            
            # Key metrics
            analysis["key_metrics"] = [
                {"name": "Average 24h Change", "value": f"{round(avg_24h_change, 2)}%"},
                {"name": "Average 7d Change", "value": f"{round(avg_7d_change, 2)}%"},
                {"name": "Average 30d Volatility", "value": f"{round(avg_volatility, 2)}%"},
                {"name": "Best 7d Performer", "value": f"{best_performer['name']} ({best_performer['change_7d']}%)"},
                {"name": "Worst 7d Performer", "value": f"{worst_performer['name']} ({worst_performer['change_7d']}%)"}
            ]
            
            # Trends
            analysis["trends"] = [
                "Cryptocurrency market continues to show high volatility compared to traditional assets",
                "Correlation between major cryptocurrencies remains high during market movements",
                "Institutional adoption is gradually increasing despite regulatory uncertainties"
            ]
            
            # Risks
            analysis["risks"] = [
                "Regulatory developments could significantly impact market sentiment",
                "High volatility makes cryptocurrencies unsuitable for risk-averse investors",
                "Market manipulation remains a concern in less liquid tokens"
            ]
            
            # Opportunities
            analysis["opportunities"] = [
                "Blockchain technology continues to find new use cases beyond cryptocurrencies",
                "DeFi (Decentralized Finance) applications are expanding the utility of crypto assets",
                "Institutional investment products are making crypto more accessible to traditional investors"
            ]
            
        elif analysis_type == "real_estate_analysis":
            # Analyze real estate data
            cities = data
            
            # Calculate average metrics
            avg_price = sum(city["median_home_price"] for city in cities) / len(cities)
            avg_price_change = sum(city["yoy_price_change"] for city in cities) / len(cities)
            avg_price_to_rent = sum(city["price_to_rent_ratio"] for city in cities) / len(cities)
            
            # Find best and worst markets for investment
            cities_by_price_to_rent = sorted(cities, key=lambda x: x["price_to_rent_ratio"])
            best_investment = cities_by_price_to_rent[0]  # Lower price-to-rent ratio is better for investors
            worst_investment = cities_by_price_to_rent[-1]
            
            # Find fastest growing markets
            cities_by_growth = sorted(cities, key=lambda x: x["yoy_price_change"], reverse=True)
            fastest_growing = cities_by_growth[0]
            slowest_growing = cities_by_growth[-1]
            
            # Summary
            analysis["summary"] = f"Analysis of {len(cities)} major US housing markets shows an average median home price of ${avg_price:,.2f}, " \
                                 f"year-over-year price change of {avg_price_change:.2f}%, and price-to-rent ratio of {avg_price_to_rent:.2f}. " \
                                 f"{fastest_growing['city']} is the fastest growing market with {fastest_growing['yoy_price_change']}% annual growth, " \
                                 f"while {best_investment['city']} offers the best investment potential with a price-to-rent ratio of {best_investment['price_to_rent_ratio']}."
            
            # Key metrics
            analysis["key_metrics"] = [
                {"name": "Average Median Home Price", "value": f"${round(avg_price, 2):,}"},
                {"name": "Average Annual Price Change", "value": f"{round(avg_price_change, 2)}%"},
                {"name": "Average Price-to-Rent Ratio", "value": round(avg_price_to_rent, 2)},
                {"name": "Fastest Growing Market", "value": f"{fastest_growing['city']} ({fastest_growing['yoy_price_change']}%)"},
                {"name": "Best Investment Market", "value": f"{best_investment['city']} (P/R: {best_investment['price_to_rent_ratio']})"}
            ]
            
            # Trends
            analysis["trends"] = [
                "Housing prices continue to rise in most markets, though at a slower pace than previous years",
                "Inventory levels remain constrained in many desirable markets",
                "Rising interest rates are affecting affordability and cooling some previously hot markets"
            ]
            
            # Risks
            analysis["risks"] = [
                "Affordability concerns as prices outpace wage growth in many markets",
                "Interest rate increases could further impact buyer demand",
                "Economic uncertainty may lead to increased market volatility"
            ]
            
            # Opportunities
            analysis["opportunities"] = [
                f"{best_investment['city']} offers attractive rental yields with a price-to-rent ratio of {best_investment['price_to_rent_ratio']}",
                "Secondary markets with strong job growth present value opportunities",
                "Markets with strong population growth tend to see sustained housing demand"
            ]
            
        elif analysis_type == "dcf_model":
            # Analyze DCF model data
            dcf = data
            
            # Calculate implied return
            implied_return = ((dcf["fair_value_per_share"] / dcf["current_market_price"]) - 1) * 100
            
            # Determine if stock is undervalued or overvalued
            valuation_status = "undervalued" if implied_return > 0 else "overvalued"
            
            # Summary
            analysis["summary"] = f"Discounted Cash Flow (DCF) analysis for {dcf['company']} indicates a fair value of ${dcf['fair_value_per_share']:.2f} " \
                                 f"per share, compared to the current market price of ${dcf['current_market_price']:.2f}. " \
                                 f"This suggests the stock is {valuation_status} by {abs(dcf['upside_downside']):.2f}%. " \
                                 f"The analysis uses a WACC of {dcf['wacc']}% and terminal growth rate of {dcf['terminal_growth_rate']}%."
            
            # Key metrics
            analysis["key_metrics"] = [
                {"name": "Fair Value Per Share", "value": f"${round(dcf['fair_value_per_share'], 2)}"},
                {"name": "Current Market Price", "value": f"${round(dcf['current_market_price'], 2)}"},
                {"name": "Upside/Downside", "value": f"{round(dcf['upside_downside'], 2)}%"},
                {"name": "WACC", "value": f"{dcf['wacc']}%"},
                {"name": "Terminal Growth Rate", "value": f"{dcf['terminal_growth_rate']}%"},
                {"name": "Enterprise Value", "value": f"${round(dcf['enterprise_value'] / 1000000000, 2)} billion"},
                {"name": "Equity Value", "value": f"${round(dcf['equity_value'] / 1000000000, 2)} billion"}
            ]
            
            # Trends
            analysis["trends"] = [
                f"Projected revenue growth averages {sum(proj['revenue_growth'] for proj in dcf['financial_projections']) / len(dcf['financial_projections']):.2f}% over the next 5 years",
                f"EBITDA margins are expected to average {sum(proj['ebitda_margin'] for proj in dcf['financial_projections']) / len(dcf['financial_projections']):.2f}%",
                f"Terminal value represents {(dcf['pv_terminal_value'] / dcf['enterprise_value']) * 100:.2f}% of the total enterprise value"
            ]
            
            # Risks
            analysis["risks"] = [
                "DCF models are highly sensitive to input assumptions, particularly WACC and terminal growth rate",
                "Projected growth rates may not materialize due to competitive pressures or market changes",
                "Macroeconomic factors like interest rates can significantly impact valuation"
            ]
            
            # Opportunities
            analysis["opportunities"] = [
                f"The stock appears to be {valuation_status} by {abs(dcf['upside_downside']):.2f}%, suggesting a potential investment opportunity" if implied_return > 10 else "The current valuation appears to be relatively fair",
                "Sensitivity analysis could reveal more conservative scenarios that still offer upside",
                "Comparing this DCF valuation with other valuation methods could provide additional confidence"
            ]
            
        else:  # General market analysis
            # Analyze general market data
            market_indices = data["market_indices"]
            sector_performance = data["sector_performance"]
            economic_indicators = data["economic_indicators"]
            
            # Calculate average market returns
            avg_daily_change = sum(index["change_percent"] for index in market_indices) / len(market_indices)
            avg_ytd_return = sum(index["ytd_return"] for index in market_indices) / len(market_indices)
            
            # Find best and worst performing sectors
            sectors_by_ytd = sorted(sector_performance, key=lambda x: x["ytd_change"], reverse=True)
            best_sector = sectors_by_ytd[0]
            worst_sector = sectors_by_ytd[-1]
            
            # Summary
            analysis["summary"] = f"Market analysis shows an average daily change of {avg_daily_change:.2f}% across major indices, " \
                                 f"with year-to-date returns averaging {avg_ytd_return:.2f}%. " \
                                 f"The {best_sector['sector']} sector is the best performer YTD with {best_sector['ytd_change']}% growth, " \
                                 f"while {worst_sector['sector']} is the worst performer at {worst_sector['ytd_change']}%. " \
                                 f"Current economic indicators show GDP growth at {economic_indicators['gdp_growth']}%, " \
                                 f"inflation at {economic_indicators['inflation_rate']}%, and unemployment at {economic_indicators['unemployment_rate']}%."
            
            # Key metrics
            analysis["key_metrics"] = [
                {"name": "Average Daily Change", "value": f"{round(avg_daily_change, 2)}%"},
                {"name": "Average YTD Return", "value": f"{round(avg_ytd_return, 2)}%"},
                {"name": "Best Performing Sector", "value": f"{best_sector['sector']} ({best_sector['ytd_change']}%)"},
                {"name": "Worst Performing Sector", "value": f"{worst_sector['sector']} ({worst_sector['ytd_change']}%)"},
                {"name": "Inflation Rate", "value": f"{economic_indicators['inflation_rate']}%"},
                {"name": "Fed Funds Rate", "value": f"{economic_indicators['fed_funds_rate']}%"},
                {"name": "10-Year Treasury Yield", "value": f"{economic_indicators['ten_year_treasury']}%"}
            ]
            
            # Trends
            analysis["trends"] = [
                f"The {best_sector['sector']} sector is outperforming the broader market, while {worst_sector['sector']} lags",
                f"Interest rates remain elevated with the 10-year Treasury at {economic_indicators['ten_year_treasury']}%",
                f"Inflation is running at {economic_indicators['inflation_rate']}%, affecting consumer sentiment and spending patterns"
            ]
            
            # Risks
            analysis["risks"] = [
                "Persistent inflation could lead to further monetary tightening",
                "Geopolitical tensions continue to create market uncertainty",
                "Consumer sentiment remains fragile, potentially impacting spending"
            ]
            
            # Opportunities
            analysis["opportunities"] = [
                f"The {best_sector['sector']} sector shows momentum and could continue to outperform",
                "Value stocks may present opportunities in a higher interest rate environment",
                "International diversification could help mitigate domestic market risks"
            ]
        
        return analysis
    
    def _generate_financial_recommendations(self, analysis_type: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate investment recommendations based on financial analysis."""
        recommendations = []
        
        if analysis_type == "stock_analysis":
            recommendations = [
                {
                    "type": "sector_allocation",
                    "title": "Technology Sector Allocation",
                    "recommendation": "Maintain a market-weight allocation to technology stocks",
                    "rationale": "While tech valuations remain elevated, strong revenue growth and innovation potential continue to support the sector's long-term outlook."
                },
                {
                    "type": "stock_picks",
                    "title": "Focus on Quality Tech Companies",
                    "recommendation": "Prioritize companies with strong balance sheets, consistent revenue growth, and competitive advantages",
                    "rationale": "In a potentially volatile market environment, companies with solid fundamentals are better positioned to weather economic uncertainties."
                },
                {
                    "type": "risk_management",
                    "title": "Implement Dollar-Cost Averaging",
                    "recommendation": "Consider a phased approach to building positions in high-conviction tech stocks",
                    "rationale": "Given current valuations and market uncertainties, spreading investments over time can help mitigate timing risk."
                }
            ]
        elif analysis_type == "crypto_analysis":
            recommendations = [
                {
                    "type": "portfolio_allocation",
                    "title": "Cryptocurrency Allocation",
                    "recommendation": "Limit cryptocurrency exposure to 1-5% of overall investment portfolio",
                    "rationale": "The high volatility of cryptocurrencies warrants a cautious allocation approach for most investors."
                },
                {
                    "type": "diversification",
                    "title": "Diversify Crypto Holdings",
                    "recommendation": "Maintain a core position in established cryptocurrencies while selectively adding exposure to promising alternatives",
                    "rationale": "Diversification within the crypto space can help balance risk while capturing growth opportunities in emerging blockchain technologies."
                },
                {
                    "type": "risk_management",
                    "title": "Implement Strict Risk Controls",
                    "recommendation": "Use stop-loss orders and regular portfolio rebalancing",
                    "rationale": "Given the extreme volatility in crypto markets, disciplined risk management is essential to protect capital."
                }
            ]
        elif analysis_type == "real_estate_analysis":
            recommendations = [
                {
                    "type": "market_selection",
                    "title": "Target Markets with Favorable Metrics",
                    "recommendation": "Focus on markets with strong job growth, affordable housing, and favorable price-to-rent ratios",
                    "rationale": "These fundamentals tend to support sustainable long-term appreciation and rental income potential."
                },
                {
                    "type": "investment_strategy",
                    "title": "Consider Secondary Markets",
                    "recommendation": "Look beyond primary markets to secondary cities with strong economic fundamentals",
                    "rationale": "Secondary markets often offer better value and yield potential compared to fully-priced primary markets."
                },
                {
                    "type": "financing",
                    "title": "Secure Long-Term Financing",
                    "recommendation": "Lock in fixed-rate financing where possible to mitigate interest rate risk",
                    "rationale": "In the current rising rate environment, fixed-rate financing provides predictability for investment returns."
                }
            ]
        elif analysis_type == "dcf_model":
            # Determine recommendation based on upside/downside
            upside_downside = analysis.get("key_metrics", [])[2].get("value", "0%").replace("%", "")
            try:
                upside_downside_value = float(upside_downside)
                if upside_downside_value > 15:
                    rating = "Strong Buy"
                    rationale = "Significant undervaluation based on DCF analysis suggests compelling long-term value."
                elif upside_downside_value > 5:
                    rating = "Buy"
                    rationale = "Moderate undervaluation offers an attractive entry point for long-term investors."
                elif upside_downside_value > -5:
                    rating = "Hold"
                    rationale = "Current valuation appears fair, with limited upside or downside potential."
                elif upside_downside_value > -15:
                    rating = "Reduce"
                    rationale = "Moderate overvaluation suggests gradually reducing position size."
                else:
                    rating = "Sell"
                    rationale = "Significant overvaluation indicates potential for material price correction."
            except ValueError:
                rating = "Hold"
                rationale = "Unable to determine precise valuation gap. Maintain current positions pending further analysis."
            
            recommendations = [
                {
                    "type": "investment_rating",
                    "title": f"{rating} Rating",
                    "recommendation": f"{rating} shares based on DCF valuation",
                    "rationale": rationale
                },
                {
                    "type": "position_sizing",
                    "title": "Position Sizing Recommendation",
                    "recommendation": "Consider a standard position size with room to add on potential pullbacks" if rating in ["Buy", "Strong Buy"] else "Maintain current position" if rating == "Hold" else "Consider reducing position size",
                    "rationale": "Position sizing should reflect both conviction level and risk management principles."
                },
                {
                    "type": "monitoring",
                    "title": "Key Metrics to Monitor",
                    "recommendation": "Closely track quarterly revenue growth and margin trends against DCF assumptions",
                    "rationale": "DCF valuations are highly sensitive to changes in growth rates and profitability metrics."
                }
            ]
        else:  # General market analysis
            recommendations = [
                {
                    "type": "asset_allocation",
                    "title": "Strategic Asset Allocation",
                    "recommendation": "Maintain a diversified portfolio with exposure to equities, fixed income, and alternative assets",
                    "rationale": "Diversification remains the most effective risk management strategy in uncertain market environments."
                },
                {
                    "type": "sector_rotation",
                    "title": "Tactical Sector Positioning",
                    "recommendation": "Overweight defensive sectors like healthcare and consumer staples; underweight interest-rate sensitive sectors",
                    "rationale": "The current economic environment favors sectors with pricing power and less sensitivity to interest rates."
                },
                {
                    "type": "risk_management",
                    "title": "Volatility Management",
                    "recommendation": "Implement hedging strategies for large equity positions and maintain adequate cash reserves",
                    "rationale": "Market volatility is likely to persist given economic uncertainties and geopolitical tensions."
                }
            ]
        
        return recommendations
    
    def _get_json_depth(self, obj, current_depth=0):
        """Helper function to calculate the depth of a JSON object."""
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._get_json_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._get_json_depth(v, current_depth + 1) for v in obj)
        else:
            return current_depth
    
    def _handle_calculation_task(self, task: Task) -> None:
        """
        Handle a task that performs calculations or data processing.
        """
        # This is a placeholder for a more complex calculation task
        # In a real implementation, this would do actual data processing
        
        # Simulate a long-running task with progress updates
        for progress in [0.2, 0.4, 0.6, 0.8, 1.0]:
            if task.is_timed_out():
                task.timeout()
                return
                
            task.update_progress(progress)
            time.sleep(0.5)  # Simulate work being done
            
        # Create a sample result
        result = {
            "status": "success",
            "calculation": "Sample calculation result",
            "timestamp": datetime.now().isoformat()
        }
        self.memory_store.update_task_result(task.task_id, result)
        logger.info(f"[SmartTaskProcessor] Completed calculation for task {task.task_id}")
    
    # ===== EDUCATIONAL CONTENT CREATION HANDLERS =====
    
    def _handle_create_learning_path(self, task: Task) -> None:
        """Handle creating a comprehensive learning path for a subject."""
        try:
            # Extract subject from task description
            subject_match = re.search(r'learning path for ([^\'\"]+)', task.description, re.IGNORECASE)
            if not subject_match:
                subject_match = re.search(r'curriculum for ([^\'\"]+)', task.description, re.IGNORECASE)
            
            if not subject_match:
                task.fail("Could not extract subject from task description")
                return
                
            subject = subject_match.group(1).strip()
            task.update_progress(0.2)
            
            logger.info(f"[SmartTaskProcessor] Creating learning path for: {subject}")
            
            # Generate learning path
            learning_path = self._generate_learning_path(subject)
            task.update_progress(0.8)
            
            # Create resources list
            resources = self._generate_learning_resources(subject)
            task.update_progress(0.9)
            
            task.complete({
                "status": "success",
                "subject": subject,
                "learning_path": learning_path,
                "resources": resources,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"[SmartTaskProcessor] Created learning path for '{subject}' with {len(learning_path)} modules")
            
        except Exception as e:
            task.fail(f"Error creating learning path: {str(e)}")
    
    def _generate_learning_path(self, subject: str) -> List[Dict[str, Any]]:
        """Generate a structured learning path for a subject."""
        import random
        
        # Seed with subject to get consistent results
        random.seed(hash(subject) % 50000)
        
        # Define learning path structure based on subject
        learning_path = []
        
        if "machine learning" in subject.lower():
            # Machine Learning learning path
            modules = [
                {
                    "title": "Foundations of Machine Learning",
                    "level": "Beginner",
                    "duration": "4 weeks",
                    "description": "Build a solid foundation in the core concepts and mathematics behind machine learning algorithms.",
                    "topics": [
                        "Introduction to Machine Learning",
                        "Linear Algebra for ML",
                        "Probability and Statistics",
                        "Calculus for ML",
                        "Python Programming Fundamentals",
                        "NumPy and Pandas for Data Manipulation"
                    ],
                    "projects": [
                        "Data exploration and visualization project",
                        "Implementing basic statistical models"
                    ],
                    "prerequisites": ["Basic programming knowledge", "High school mathematics"]
                },
                {
                    "title": "Supervised Learning Algorithms",
                    "level": "Beginner-Intermediate",
                    "duration": "6 weeks",
                    "description": "Master the fundamental supervised learning algorithms and their implementation.",
                    "topics": [
                        "Linear Regression",
                        "Logistic Regression",
                        "Decision Trees",
                        "Random Forests",
                        "Support Vector Machines",
                        "K-Nearest Neighbors",
                        "Naive Bayes",
                        "Evaluation Metrics"
                    ],
                    "projects": [
                        "House price prediction using regression",
                        "Customer churn prediction using classification"
                    ],
                    "prerequisites": ["Foundations of Machine Learning"]
                },
                {
                    "title": "Unsupervised Learning",
                    "level": "Intermediate",
                    "duration": "4 weeks",
                    "description": "Explore techniques for finding patterns in unlabeled data.",
                    "topics": [
                        "K-Means Clustering",
                        "Hierarchical Clustering",
                        "DBSCAN",
                        "Principal Component Analysis (PCA)",
                        "t-SNE",
                        "Anomaly Detection",
                        "Association Rule Learning"
                    ],
                    "projects": [
                        "Customer segmentation project",
                        "Dimensionality reduction and visualization"
                    ],
                    "prerequisites": ["Supervised Learning Algorithms"]
                },
                {
                    "title": "Deep Learning Fundamentals",
                    "level": "Intermediate",
                    "duration": "8 weeks",
                    "description": "Understand neural networks and implement deep learning models.",
                    "topics": [
                        "Neural Network Architecture",
                        "Activation Functions",
                        "Backpropagation",
                        "Optimization Algorithms",
                        "Regularization Techniques",
                        "Convolutional Neural Networks (CNNs)",
                        "Recurrent Neural Networks (RNNs)",
                        "TensorFlow and PyTorch"
                    ],
                    "projects": [
                        "Image classification with CNNs",
                        "Sequence prediction with RNNs"
                    ],
                    "prerequisites": ["Supervised Learning Algorithms"]
                },
                {
                    "title": "Advanced Deep Learning",
                    "level": "Advanced",
                    "duration": "6 weeks",
                    "description": "Explore cutting-edge deep learning architectures and techniques.",
                    "topics": [
                        "Transformers and Attention Mechanisms",
                        "Generative Adversarial Networks (GANs)",
                        "Variational Autoencoders (VAEs)",
                        "Transfer Learning",
                        "Self-Supervised Learning",
                        "Reinforcement Learning Basics",
                        "Model Deployment and Serving"
                    ],
                    "projects": [
                        "Text generation with transformers",
                        "Image generation with GANs"
                    ],
                    "prerequisites": ["Deep Learning Fundamentals"]
                },
                {
                    "title": "MLOps and Production ML",
                    "level": "Advanced",
                    "duration": "4 weeks",
                    "description": "Learn to deploy, monitor, and maintain machine learning models in production.",
                    "topics": [
                        "ML System Design",
                        "Data and Feature Engineering Pipelines",
                        "Model Versioning and Experiment Tracking",
                        "Containerization and Orchestration",
                        "Model Monitoring and Maintenance",
                        "A/B Testing",
                        "Ethics and Responsible AI"
                    ],
                    "projects": [
                        "End-to-end ML pipeline deployment",
                        "Building a model monitoring system"
                    ],
                    "prerequisites": ["Advanced Deep Learning"]
                }
            ]
            learning_path.extend(modules)
            
        elif "web development" in subject.lower():
            # Web Development learning path
            modules = [
                {
                    "title": "Web Development Fundamentals",
                    "level": "Beginner",
                    "duration": "4 weeks",
                    "description": "Learn the core technologies that power the web.",
                    "topics": [
                        "HTML5 Structure and Semantics",
                        "CSS3 Styling and Layout",
                        "CSS Flexbox and Grid",
                        "Responsive Design Principles",
                        "JavaScript Fundamentals",
                        "DOM Manipulation",
                        "Web Accessibility Basics"
                    ],
                    "projects": [
                        "Personal portfolio website",
                        "Responsive landing page"
                    ],
                    "prerequisites": ["Basic computer skills"]
                },
                {
                    "title": "Frontend Development",
                    "level": "Beginner-Intermediate",
                    "duration": "6 weeks",
                    "description": "Master modern frontend development techniques and frameworks.",
                    "topics": [
                        "Advanced JavaScript (ES6+)",
                        "Asynchronous JavaScript",
                        "Fetch API and AJAX",
                        "npm and Package Management",
                        "Introduction to React",
                        "React Components and Props",
                        "State Management",
                        "React Hooks"
                    ],
                    "projects": [
                        "Interactive web application with React",
                        "Data dashboard with API integration"
                    ],
                    "prerequisites": ["Web Development Fundamentals"]
                },
                {
                    "title": "Backend Development",
                    "level": "Intermediate",
                    "duration": "6 weeks",
                    "description": "Build server-side applications and APIs.",
                    "topics": [
                        "Node.js Fundamentals",
                        "Express.js Framework",
                        "RESTful API Design",
                        "Database Concepts (SQL vs NoSQL)",
                        "MongoDB and Mongoose",
                        "Authentication and Authorization",
                        "Error Handling and Logging",
                        "Testing Backend Applications"
                    ],
                    "projects": [
                        "RESTful API development",
                        "User authentication system"
                    ],
                    "prerequisites": ["Frontend Development"]
                },
                {
                    "title": "Full Stack Integration",
                    "level": "Intermediate",
                    "duration": "4 weeks",
                    "description": "Connect frontend and backend to create complete web applications.",
                    "topics": [
                        "Full Stack Application Architecture",
                        "API Integration with React",
                        "State Management with Context API or Redux",
                        "Form Validation and Submission",
                        "File Uploads",
                        "Real-time Features with WebSockets",
                        "Deployment Workflows"
                    ],
                    "projects": [
                        "Full stack social media application",
                        "E-commerce platform with payment integration"
                    ],
                    "prerequisites": ["Backend Development"]
                },
                {
                    "title": "Advanced Frontend Frameworks",
                    "level": "Advanced",
                    "duration": "4 weeks",
                    "description": "Explore advanced frontend frameworks and techniques.",
                    "topics": [
                        "Next.js for Server-Side Rendering",
                        "Static Site Generation",
                        "TypeScript Integration",
                        "Advanced State Management",
                        "Performance Optimization",
                        "Animation and Transitions",
                        "Micro-frontends Architecture"
                    ],
                    "projects": [
                        "Server-side rendered application with Next.js",
                        "Progressive Web App (PWA)"
                    ],
                    "prerequisites": ["Full Stack Integration"]
                },
                {
                    "title": "DevOps for Web Developers",
                    "level": "Advanced",
                    "duration": "4 weeks",
                    "description": "Learn to deploy, scale, and maintain web applications.",
                    "topics": [
                        "CI/CD Pipelines",
                        "Docker and Containerization",
                        "Cloud Deployment (AWS/Azure/GCP)",
                        "Serverless Architecture",
                        "Monitoring and Logging",
                        "Performance Testing",
                        "Security Best Practices"
                    ],
                    "projects": [
                        "Setting up CI/CD for a web application",
                        "Deploying microservices architecture"
                    ],
                    "prerequisites": ["Advanced Frontend Frameworks"]
                }
            ]
            learning_path.extend(modules)
            
        elif "financial literacy" in subject.lower() or "finance" in subject.lower():
            # Financial Literacy learning path
            modules = [
                {
                    "title": "Personal Finance Fundamentals",
                    "level": "Beginner",
                    "duration": "3 weeks",
                    "description": "Build a strong foundation in managing personal finances.",
                    "topics": [
                        "Budgeting Basics",
                        "Income and Expense Tracking",
                        "Emergency Funds",
                        "Banking and Account Types",
                        "Credit Scores and Reports",
                        "Debt Management",
                        "Setting Financial Goals"
                    ],
                    "projects": [
                        "Creating a personal budget",
                        "Developing a debt reduction plan"
                    ],
                    "prerequisites": ["None"]
                },
                {
                    "title": "Saving and Investing Basics",
                    "level": "Beginner-Intermediate",
                    "duration": "4 weeks",
                    "description": "Learn the fundamentals of saving and investing for the future.",
                    "topics": [
                        "Saving Strategies",
                        "Compound Interest",
                        "Investment Vehicles Overview",
                        "Risk and Return Concepts",
                        "Stock Market Basics",
                        "Bonds and Fixed Income",
                        "Mutual Funds and ETFs",
                        "Retirement Accounts (401k, IRA)"
                    ],
                    "projects": [
                        "Creating a savings plan",
                        "Building a starter investment portfolio"
                    ],
                    "prerequisites": ["Personal Finance Fundamentals"]
                },
                {
                    "title": "Tax Planning and Optimization",
                    "level": "Intermediate",
                    "duration": "3 weeks",
                    "description": "Understand tax concepts and strategies to minimize tax burden.",
                    "topics": [
                        "Income Tax Basics",
                        "Tax Deductions and Credits",
                        "Tax-Advantaged Accounts",
                        "Capital Gains and Losses",
                        "Tax Planning Throughout the Year",
                        "Self-Employment Taxes",
                        "State and Local Taxes"
                    ],
                    "projects": [
                        "Creating a tax planning checklist",
                        "Identifying applicable tax deductions"
                    ],
                    "prerequisites": ["Saving and Investing Basics"]
                },
                {
                    "title": "Retirement Planning",
                    "level": "Intermediate",
                    "duration": "4 weeks",
                    "description": "Develop strategies for a secure retirement.",
                    "topics": [
                        "Retirement Needs Analysis",
                        "Social Security Benefits",
                        "Pension Plans",
                        "401(k) and IRA Optimization",
                        "Withdrawal Strategies",
                        "Healthcare in Retirement",
                        "Estate Planning Basics"
                    ],
                    "projects": [
                        "Calculating retirement needs",
                        "Creating a retirement income plan"
                    ],
                    "prerequisites": ["Tax Planning and Optimization"]
                },
                {
                    "title": "Advanced Investing Strategies",
                    "level": "Advanced",
                    "duration": "5 weeks",
                    "description": "Explore sophisticated investment approaches and portfolio management.",
                    "topics": [
                        "Asset Allocation",
                        "Portfolio Diversification",
                        "Modern Portfolio Theory",
                        "Factor Investing",
                        "Alternative Investments",
                        "Real Estate Investing",
                        "Sustainable and ESG Investing",
                        "Portfolio Rebalancing"
                    ],
                    "projects": [
                        "Building a diversified investment portfolio",
                        "Creating an investment policy statement"
                    ],
                    "prerequisites": ["Retirement Planning"]
                },
                {
                    "title": "Financial Independence and Wealth Building",
                    "level": "Advanced",
                    "duration": "4 weeks",
                    "description": "Develop strategies for building wealth and achieving financial independence.",
                    "topics": [
                        "Financial Independence Concepts",
                        "Passive Income Streams",
                        "Business Ownership and Entrepreneurship",
                        "Advanced Tax Strategies",
                        "Generational Wealth Transfer",
                        "Charitable Giving",
                        "Financial Life Planning"
                    ],
                    "projects": [
                        "Creating a financial independence plan",
                        "Developing a passive income strategy"
                    ],
                    "prerequisites": ["Advanced Investing Strategies"]
                }
            ]
            learning_path.extend(modules)
            
        elif "language" in subject.lower() or "spanish" in subject.lower():
            # Language Learning path (Spanish example)
            modules = [
                {
                    "title": "Spanish Foundations (A1)",
                    "level": "Beginner",
                    "duration": "8 weeks",
                    "description": "Build a foundation in Spanish with basic vocabulary and grammar.",
                    "topics": [
                        "Spanish Alphabet and Pronunciation",
                        "Greetings and Introductions",
                        "Numbers and Basic Math",
                        "Days, Months, and Dates",
                        "Present Tense of Regular Verbs",
                        "Common Nouns and Articles",
                        "Basic Adjectives and Agreement",
                        "Simple Questions and Answers"
                    ],
                    "projects": [
                        "Self-introduction in Spanish",
                        "Basic conversation practice"
                    ],
                    "prerequisites": ["None"]
                },
                {
                    "title": "Elementary Spanish (A2)",
                    "level": "Beginner-Intermediate",
                    "duration": "10 weeks",
                    "description": "Expand vocabulary and grammar to handle everyday situations.",
                    "topics": [
                        "Present Tense of Irregular Verbs",
                        "Reflexive Verbs",
                        "Past Tense (Preterite)",
                        "Food and Restaurant Vocabulary",
                        "Shopping and Clothing",
                        "Travel and Transportation",
                        "Giving Directions",
                        "Making Plans and Invitations"
                    ],
                    "projects": [
                        "Role-play ordering in a restaurant",
                        "Writing a daily routine description"
                    ],
                    "prerequisites": ["Spanish Foundations (A1)"]
                },
                {
                    "title": "Intermediate Spanish (B1)",
                    "level": "Intermediate",
                    "duration": "12 weeks",
                    "description": "Develop more complex language skills and express opinions.",
                    "topics": [
                        "Past Tense (Imperfect)",
                        "Contrasting Preterite and Imperfect",
                        "Future Tense",
                        "Conditional Tense",
                        "Expressing Opinions and Emotions",
                        "Health and Medical Vocabulary",
                        "Work and Professional Life",
                        "Cultural Topics and Current Events"
                    ],
                    "projects": [
                        "Writing a short story in past tense",
                        "Presenting opinions on current events"
                    ],
                    "prerequisites": ["Elementary Spanish (A2)"]
                },
                {
                    "title": "Upper Intermediate Spanish (B2)",
                    "level": "Intermediate-Advanced",
                    "duration": "12 weeks",
                    "description": "Achieve fluency in most situations and express complex ideas.",
                    "topics": [
                        "Present and Past Subjunctive",
                        "Perfect Tenses",
                        "Passive Voice",
                        "Reported Speech",
                        "Idiomatic Expressions",
                        "Advanced Vocabulary Development",
                        "Debating Skills",
                        "Spanish Literature Introduction"
                    ],
                    "projects": [
                        "Book or film review in Spanish",
                        "Formal presentation on a complex topic"
                    ],
                    "prerequisites": ["Intermediate Spanish (B1)"]
                },
                {
                    "title": "Advanced Spanish (C1)",
                    "level": "Advanced",
                    "duration": "12 weeks",
                    "description": "Master complex language and cultural nuances for professional contexts.",
                    "topics": [
                        "Advanced Grammar Refinement",
                        "Regional Variations in Spanish",
                        "Professional and Academic Vocabulary",
                        "Literary Analysis",
                        "Advanced Writing Techniques",
                        "Cultural and Historical Contexts",
                        "Humor and Wordplay",
                        "Specialized Terminology (Business, Medical, Legal)"
                    ],
                    "projects": [
                        "Academic essay in Spanish",
                        "Podcast or video creation in Spanish"
                    ],
                    "prerequisites": ["Upper Intermediate Spanish (B2)"]
                },
                {
                    "title": "Spanish Mastery (C2)",
                    "level": "Advanced",
                    "duration": "12 weeks",
                    "description": "Achieve near-native fluency and cultural competence.",
                    "topics": [
                        "Nuanced Expression and Style",
                        "Advanced Dialectal Variations",
                        "Translation and Interpretation Skills",
                        "Spanish Cinema and Media Analysis",
                        "Contemporary Literature",
                        "Cultural Immersion Strategies",
                        "Advanced Debate and Persuasion",
                        "Teaching Spanish to Others"
                    ],
                    "projects": [
                        "Literary translation project",
                        "Research paper in Spanish"
                    ],
                    "prerequisites": ["Advanced Spanish (C1)"]
                }
            ]
            learning_path.extend(modules)
            
        elif "digital marketing" in subject.lower():
            # Digital Marketing learning path
            modules = [
                {
                    "title": "Digital Marketing Fundamentals",
                    "level": "Beginner",
                    "duration": "4 weeks",
                    "description": "Build a foundation in digital marketing principles and channels.",
                    "topics": [
                        "Introduction to Digital Marketing",
                        "Digital Marketing Strategy Framework",
                        "Customer Journey Mapping",
                        "Digital Marketing Channels Overview",
                        "Marketing Metrics and KPIs",
                        "Target Audience and Personas",
                        "Digital Marketing Tools Introduction"
                    ],
                    "projects": [
                        "Creating customer personas",
                        "Developing a basic digital marketing plan"
                    ],
                    "prerequisites": ["Basic internet skills"]
                },
                {
                    "title": "Content Marketing",
                    "level": "Beginner-Intermediate",
                    "duration": "4 weeks",
                    "description": "Learn to create and distribute valuable content to attract and engage audiences.",
                    "topics": [
                        "Content Marketing Strategy",
                        "Content Types and Formats",
                        "Content Creation Process",
                        "Content Calendar Planning",
                        "Storytelling Techniques",
                        "Content Distribution Channels",
                        "Content Performance Measurement",
                        "Content SEO Basics"
                    ],
                    "projects": [
                        "Creating a content marketing strategy",
                        "Producing a blog post and social media content"
                    ],
                    "prerequisites": ["Digital Marketing Fundamentals"]
                },
                {
                    "title": "Search Engine Optimization (SEO)",
                    "level": "Intermediate",
                    "duration": "6 weeks",
                    "description": "Master techniques to improve organic visibility in search engines.",
                    "topics": [
                        "SEO Fundamentals",
                        "Keyword Research and Analysis",
                        "On-Page SEO Techniques",
                        "Technical SEO",
                        "Off-Page SEO and Link Building",
                        "Local SEO",
                        "SEO Tools and Analytics",
                        "SEO Strategy Development"
                    ],
                    "projects": [
                        "Conducting a website SEO audit",
                        "Implementing on-page SEO improvements"
                    ],
                    "prerequisites": ["Content Marketing"]
                },
                {
                    "title": "Social Media Marketing",
                    "level": "Intermediate",
                    "duration": "5 weeks",
                    "description": "Develop strategies for effective social media marketing across platforms.",
                    "topics": [
                        "Social Media Strategy Development",
                        "Platform-Specific Marketing (Facebook, Instagram, Twitter, LinkedIn, TikTok)",
                        "Social Media Content Creation",
                        "Community Management",
                        "Social Media Advertising",
                        "Influencer Marketing",
                        "Social Media Analytics",
                        "Social Listening and Reputation Management"
                    ],
                    "projects": [
                        "Creating a social media marketing plan",
                        "Running a social media campaign"
                    ],
                    "prerequisites": ["Content Marketing"]
                },
                {
                    "title": "Email Marketing",
                    "level": "Intermediate",
                    "duration": "3 weeks",
                    "description": "Learn to create effective email marketing campaigns and automation.",
                    "topics": [
                        "Email Marketing Strategy",
                        "List Building and Management",
                        "Email Design and Copywriting",
                        "Email Automation and Workflows",
                        "A/B Testing for Emails",
                        "Deliverability and Compliance",
                        "Email Analytics and Optimization",
                        "Integration with Other Marketing Channels"
                    ],
                    "projects": [
                        "Setting up an email marketing campaign",
                        "Creating an automated email sequence"
                    ],
                    "prerequisites": ["Digital Marketing Fundamentals"]
                },
                {
                    "title": "Paid Advertising",
                    "level": "Intermediate-Advanced",
                    "duration": "6 weeks",
                    "description": "Master paid digital advertising across search, social, and display networks.",
                    "topics": [
                        "Paid Advertising Strategy",
                        "Google Ads (Search, Display, Video)",
                        "Facebook and Instagram Ads",
                        "LinkedIn Advertising",
                        "Retargeting Campaigns",
                        "Bidding Strategies and Budget Management",
                        "Ad Copywriting and Creative",
                        "Campaign Optimization and Scaling"
                    ],
                    "projects": [
                        "Setting up and managing a Google Ads campaign",
                        "Creating and optimizing social media ads"
                    ],
                    "prerequisites": ["Social Media Marketing"]
                },
                {
                    "title": "Analytics and Data-Driven Marketing",
                    "level": "Advanced",
                    "duration": "5 weeks",
                    "description": "Use data and analytics to measure, analyze, and optimize marketing performance.",
                    "topics": [
                        "Google Analytics Implementation and Setup",
                        "Key Metrics and Dimensions",
                        "Custom Reports and Dashboards",
                        "Conversion Tracking and Attribution Models",
                        "A/B Testing and Experimentation",
                        "Data Visualization",
                        "Marketing ROI Calculation",
                        "Data-Driven Decision Making"
                    ],
                    "projects": [
                        "Setting up Google Analytics and conversion tracking",
                        "Creating a marketing dashboard"
                    ],
                    "prerequisites": ["Paid Advertising"]
                },
                {
                    "title": "Digital Marketing Strategy and Integration",
                    "level": "Advanced",
                    "duration": "4 weeks",
                    "description": "Develop comprehensive digital marketing strategies and integrate multiple channels.",
                    "topics": [
                        "Integrated Marketing Communications",
                        "Omnichannel Marketing Strategy",
                        "Customer Experience Optimization",
                        "Marketing Automation",
                        "Growth Marketing Techniques",
                        "Marketing Technology Stack",
                        "Budget Allocation and Planning",
                        "Digital Transformation"
                    ],
                    "projects": [
                        "Developing a comprehensive digital marketing strategy",
                        "Creating an integrated marketing campaign"
                    ],
                    "prerequisites": ["Analytics and Data-Driven Marketing"]
                }
            ]
            learning_path.extend(modules)
            
        else:
            # Generic learning path for other subjects
            modules = [
                {
                    "title": f"{subject} Fundamentals",
                    "level": "Beginner",
                    "duration": "4 weeks",
                    "description": f"Build a solid foundation in {subject} concepts and principles.",
                    "topics": [
                        f"Introduction to {subject}",
                        "Core Concepts and Terminology",
                        "Historical Context and Development",
                        "Fundamental Principles",
                        "Basic Techniques and Methods",
                        "Tools and Resources"
                    ],
                    "projects": [
                        f"Introductory {subject} project",
                        "Concept application exercise"
                    ],
                    "prerequisites": ["None"]
                },
                {
                    "title": f"Intermediate {subject}",
                    "level": "Intermediate",
                    "duration": "6 weeks",
                    "description": f"Deepen your knowledge and skills in {subject}.",
                    "topics": [
                        "Advanced Concepts",
                        "Specialized Techniques",
                        "Problem-Solving Approaches",
                        "Case Studies and Applications",
                        "Current Trends and Developments",
                        "Practical Implementation"
                    ],
                    "projects": [
                        f"Comprehensive {subject} project",
                        "Real-world application"
                    ],
                    "prerequisites": [f"{subject} Fundamentals"]
                },
                {
                    "title": f"Advanced {subject}",
                    "level": "Advanced",
                    "duration": "8 weeks",
                    "description": f"Master complex aspects of {subject} and develop expertise.",
                    "topics": [
                        "Cutting-Edge Developments",
                        "Advanced Methodologies",
                        "Specialized Applications",
                        "Research and Innovation",
                        "Integration with Related Fields",
                        "Professional Best Practices"
                    ],
                    "projects": [
                        f"Advanced {subject} implementation",
                        "Specialized research project"
                    ],
                    "prerequisites": [f"Intermediate {subject}"]
                },
                {
                    "title": f"Professional {subject} Applications",
                    "level": "Advanced",
                    "duration": "6 weeks",
                    "description": f"Apply {subject} knowledge in professional contexts and real-world scenarios.",
                    "topics": [
                        "Industry Applications",
                        "Professional Standards and Ethics",
                        "Project Management",
                        "Collaboration and Communication",
                        "Problem-Solving in Complex Scenarios",
                        "Future Trends and Career Development"
                    ],
                    "projects": [
                        f"Professional portfolio in {subject}",
                        "Capstone project"
                    ],
                    "prerequisites": [f"Advanced {subject}"]
                }
            ]
            learning_path.extend(modules)
        
        return learning_path
    
    def _generate_learning_resources(self, subject: str) -> Dict[str, List[Dict[str, str]]]:
        """Generate a list of learning resources for a subject."""
        import random
        
        # Seed with subject to get consistent results
        random.seed(hash(subject) % 60000)
        
        # Define resource categories
        resource_categories = ["Books", "Online Courses", "Videos", "Websites", "Tools", "Communities"]
        
        # Generate resources based on subject
        resources = {}
        
        # Books
        books = []
        if "machine learning" in subject.lower():
            books = [
                {"title": "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow", "author": "Aurélien Géron", "level": "Beginner-Intermediate"},
                {"title": "Deep Learning", "author": "Ian Goodfellow, Yoshua Bengio, and Aaron Courville", "level": "Intermediate-Advanced"},
                {"title": "Pattern Recognition and Machine Learning", "author": "Christopher Bishop", "level": "Advanced"},
                {"title": "The Hundred-Page Machine Learning Book", "author": "Andriy Burkov", "level": "Beginner-Intermediate"},
                {"title": "Python Machine Learning", "author": "Sebastian Raschka", "level": "Beginner-Intermediate"}
            ]
        elif "web development" in subject.lower():
            books = [
                {"title": "Eloquent JavaScript", "author": "Marijn Haverbeke", "level": "Beginner-Intermediate"},
                {"title": "You Don't Know JS", "author": "Kyle Simpson", "level": "Intermediate"},
                {"title": "React Up and Running", "author": "Stoyan Stefanov", "level": "Intermediate"},
                {"title": "Node.js Design Patterns", "author": "Mario Casciaro", "level": "Intermediate-Advanced"},
                {"title": "CSS Secrets", "author": "Lea Verou", "level": "Intermediate"}
            ]
        elif "financial literacy" in subject.lower() or "finance" in subject.lower():
            books = [
                {"title": "The Psychology of Money", "author": "Morgan Housel", "level": "Beginner"},
                {"title": "I Will Teach You to Be Rich", "author": "Ramit Sethi", "level": "Beginner"},
                {"title": "The Simple Path to Wealth", "author": "J.L. Collins", "level": "Beginner-Intermediate"},
                {"title": "The Bogleheads' Guide to Investing", "author": "Taylor Larimore, Mel Lindauer, and Michael LeBoeuf", "level": "Intermediate"},
                {"title": "A Random Walk Down Wall Street", "author": "Burton G. Malkiel", "level": "Intermediate"}
            ]
        elif "language" in subject.lower() or "spanish" in subject.lower():
            books = [
                {"title": "Madrigal's Magic Key to Spanish", "author": "Margarita Madrigal", "level": "Beginner"},
                {"title": "Easy Spanish Step-by-Step", "author": "Barbara Bregstein", "level": "Beginner"},
                {"title": "Short Stories in Spanish for Beginners", "author": "Olly Richards", "level": "Beginner-Intermediate"},
                {"title": "Practice Makes Perfect: Complete Spanish Grammar", "author": "Gilda Nissenberg", "level": "Intermediate"},
                {"title": "Advanced Spanish Step-by-Step", "author": "Barbara Bregstein", "level": "Advanced"}
            ]
        elif "digital marketing" in subject.lower():
            books = [
                {"title": "Digital Marketing For Dummies", "author": "Ryan Deiss and Russ Henneberry", "level": "Beginner"},
                {"title": "This Is Marketing", "author": "Seth Godin", "level": "Beginner-Intermediate"},
                {"title": "Building a StoryBrand", "author": "Donald Miller", "level": "Intermediate"},
                {"title": "Hooked: How to Build Habit-Forming Products", "author": "Nir Eyal", "level": "Intermediate"},
                {"title": "Contagious: How to Build Word of Mouth in the Digital Age", "author": "Jonah Berger", "level": "Intermediate"}
            ]
        else:
            # Generic books
            books = [
                {"title": f"{subject} for Beginners", "author": "Various Authors", "level": "Beginner"},
                {"title": f"The Complete Guide to {subject}", "author": "Expert Author", "level": "Beginner-Intermediate"},
                {"title": f"Advanced {subject} Techniques", "author": "Professional Expert", "level": "Advanced"},
                {"title": f"{subject}: Theory and Practice", "author": "Academic Author", "level": "Intermediate"},
                {"title": f"Mastering {subject}", "author": "Industry Leader", "level": "Intermediate-Advanced"}
            ]
        
        resources["Books"] = books
        
        # Online Courses
        courses = []
        if "machine learning" in subject.lower():
            courses = [
                {"title": "Machine Learning", "platform": "Coursera (Stanford)", "instructor": "Andrew Ng", "level": "Beginner-Intermediate"},
                {"title": "Deep Learning Specialization", "platform": "Coursera (deeplearning.ai)", "instructor": "Andrew Ng", "level": "Intermediate"},
                {"title": "Machine Learning with Python", "platform": "edX", "instructor": "IBM", "level": "Beginner-Intermediate"},
                {"title": "Practical Deep Learning for Coders", "platform": "fast.ai", "instructor": "Jeremy Howard", "level": "Intermediate"},
                {"title": "Machine Learning Crash Course", "platform": "Google AI", "instructor": "Google", "level": "Beginner"}
            ]
        elif "web development" in subject.lower():
            courses = [
                {"title": "The Web Developer Bootcamp", "platform": "Udemy", "instructor": "Colt Steele", "level": "Beginner"},
                {"title": "JavaScript: Understanding the Weird Parts", "platform": "Udemy", "instructor": "Anthony Alicea", "level": "Intermediate"},
                {"title": "React - The Complete Guide", "platform": "Udemy", "instructor": "Maximilian Schwarzmüller", "level": "Beginner-Intermediate"},
                {"title": "Full Stack Open", "platform": "University of Helsinki", "instructor": "Various", "level": "Intermediate"},
                {"title": "CS50's Web Programming with Python and JavaScript", "platform": "edX (Harvard)", "instructor": "David Malan", "level": "Intermediate"}
            ]
        elif "financial literacy" in subject.lower() or "finance" in subject.lower():
            courses = [
                {"title": "Personal Finance", "platform": "edX (Purdue)", "instructor": "Various", "level": "Beginner"},
                {"title": "Financial Markets", "platform": "Coursera (Yale)", "instructor": "Robert Shiller", "level": "Intermediate"},
                {"title": "Investment Management", "platform": "Coursera (Geneva)", "instructor": "Various", "level": "Intermediate"},
                {"title": "Financial Planning for Young Adults", "platform": "Coursera (Illinois)", "instructor": "Various", "level": "Beginner"},
                {"title": "Behavioral Finance", "platform": "edX (Duke)", "instructor": "Various", "level": "Intermediate"}
            ]
        elif "language" in subject.lower() or "spanish" in subject.lower():
            courses = [
                {"title": "Spanish 1: Beginning Spanish", "platform": "edX (UPValenciaX)", "instructor": "Various", "level": "Beginner"},
                {"title": "Basic Spanish 1: Getting Started", "platform": "Coursera (UC Davis)", "instructor": "Various", "level": "Beginner"},
                {"title": "Spanish for Beginners", "platform": "Babbel", "instructor": "Various", "level": "Beginner"},
                {"title": "Intermediate Spanish", "platform": "Duolingo", "instructor": "Automated", "level": "Intermediate"},
                {"title": "Advanced Spanish Conversation", "platform": "italki", "instructor": "Various Tutors", "level": "Advanced"}
            ]
        elif "digital marketing" in subject.lower():
            courses = [
                {"title": "Fundamentals of Digital Marketing", "platform": "Google Digital Garage", "instructor": "Google", "level": "Beginner"},
                {"title": "Digital Marketing Specialization", "platform": "Coursera (Illinois)", "instructor": "Various", "level": "Beginner-Intermediate"},
                {"title": "The Complete Digital Marketing Course", "platform": "Udemy", "instructor": "Rob Percival & Daragh Walsh", "level": "Beginner-Intermediate"},
                {"title": "Social Media Marketing", "platform": "Coursera (Northwestern)", "instructor": "Randy Hlavac", "level": "Intermediate"},
                {"title": "SEO Training Course", "platform": "Semrush Academy", "instructor": "Various", "level": "Beginner-Intermediate"}
            ]
        else:
            # Generic courses
            courses = [
                {"title": f"Introduction to {subject}", "platform": "Coursera", "instructor": "University Professor", "level": "Beginner"},
                {"title": f"{subject} Fundamentals", "platform": "edX", "instructor": "Industry Expert", "level": "Beginner"},
                {"title": f"Complete {subject} Bootcamp", "platform": "Udemy", "instructor": "Professional Instructor", "level": "Beginner-Intermediate"},
                {"title": f"Advanced {subject} Techniques", "platform": "Pluralsight", "instructor": "Senior Specialist", "level": "Advanced"},
                {"title": f"{subject} Masterclass", "platform": "MasterClass", "instructor": "Industry Leader", "level": "Intermediate-Advanced"}
            ]
        
        resources["Online Courses"] = courses
        
        # Videos
        videos = []
        if "machine learning" in subject.lower():
            videos = [
                {"title": "StatQuest with Josh Starmer", "platform": "YouTube", "creator": "Josh Starmer", "focus": "Statistical concepts explained simply"},
                {"title": "3Blue1Brown: Neural Networks", "platform": "YouTube", "creator": "Grant Sanderson", "focus": "Visual explanations of neural networks"},
                {"title": "Two Minute Papers", "platform": "YouTube", "creator": "Károly Zsolnai-Fehér", "focus": "AI research paper summaries"},
                {"title": "Lex Fridman Podcast", "platform": "YouTube/Podcast", "creator": "Lex Fridman", "focus": "Interviews with AI researchers"},
                {"title": "Sentdex Python Machine Learning Tutorials", "platform": "YouTube", "creator": "Harrison Kinsley", "focus": "Practical ML implementations"}
            ]
        elif "web development" in subject.lower():
            videos = [
                {"title": "Traversy Media", "platform": "YouTube", "creator": "Brad Traversy", "focus": "Web development tutorials and projects"},
                {"title": "The Net Ninja", "platform": "YouTube", "creator": "Shaun Pelling", "focus": "Modern JavaScript and frameworks"},
                {"title": "Web Dev Simplified", "platform": "YouTube", "creator": "Kyle Cook", "focus": "Simplified web development concepts"},
                {"title": "Academind", "platform": "YouTube", "creator": "Maximilian Schwarzmüller", "focus": "React, Angular, Node.js tutorials"},
                {"title": "Fireship", "platform": "YouTube", "creator": "Jeff Delaney", "focus": "Quick, modern web development concepts"}
            ]
        elif "financial literacy" in subject.lower() or "finance" in subject.lower():
            videos = [
                {"title": "Two Cents", "platform": "YouTube", "creator": "PBS", "focus": "Basic personal finance concepts"},
                {"title": "Graham Stephan", "platform": "YouTube", "creator": "Graham Stephan", "focus": "Personal finance and real estate"},
                {"title": "The Plain Bagel", "platform": "YouTube", "creator": "Richard Coffin", "focus": "Investment concepts explained clearly"},
                {"title": "The Money Guy Show", "platform": "YouTube/Podcast", "creator": "Brian Preston & Bo Hanson", "focus": "Financial planning advice"},
                {"title": "Humphrey Yang", "platform": "YouTube/TikTok", "creator": "Humphrey Yang", "focus": "Bite-sized financial education"}
            ]
        elif "language" in subject.lower() or "spanish" in subject.lower():
            videos = [
                {"title": "Butterfly Spanish", "platform": "YouTube", "creator": "Ana", "focus": "Conversational Spanish lessons"},
                {"title": "Spanish with Vicente", "platform": "YouTube", "creator": "Vicente", "focus": "Grammar and vocabulary lessons"},
                {"title": "Why Not Spanish", "platform": "YouTube", "creator": "Cody and Maria", "focus": "Real-life Spanish conversations"},
                {"title": "Dreaming Spanish", "platform": "YouTube", "creator": "Pablo", "focus": "Comprehensible input method"},
                {"title": "Spanish Pod 101", "platform": "YouTube", "creator": "Various", "focus": "Structured Spanish lessons"}
            ]
        elif "digital marketing" in subject.lower():
            videos = [
                {"title": "Neil Patel", "platform": "YouTube", "creator": "Neil Patel", "focus": "SEO and digital marketing strategies"},
                {"title": "Ahrefs", "platform": "YouTube", "creator": "Ahrefs Team", "focus": "SEO tutorials and case studies"},
                {"title": "Income School", "platform": "YouTube", "creator": "Jim & Ricky", "focus": "Content marketing and blogging"},
                {"title": "Surfside PPC", "platform": "YouTube", "creator": "Jeff Romero", "focus": "Google Ads and PPC tutorials"},
                {"title": "Social Media Examiner", "platform": "YouTube", "creator": "Michael Stelzner", "focus": "Social media marketing tactics"}
            ]
        else:
            # Generic videos
            videos = [
                {"title": f"{subject} Fundamentals", "platform": "YouTube", "creator": "Educational Channel", "focus": "Basic concepts and principles"},
                {"title": f"{subject} Masterclass", "platform": "YouTube", "creator": "Expert Creator", "focus": "Advanced techniques and applications"},
                {"title": f"{subject} for Beginners", "platform": "YouTube", "creator": "Tutorial Channel", "focus": "Step-by-step learning for beginners"},
                {"title": f"{subject} Case Studies", "platform": "YouTube", "creator": "Industry Professional", "focus": "Real-world applications and examples"},
                {"title": f"{subject} Tips and Tricks", "platform": "YouTube", "creator": "Experienced Practitioner", "focus": "Practical advice and shortcuts"}
            ]
        
        resources["Videos"] = videos
        
        # Websites and Blogs
        websites = []
        if "machine learning" in subject.lower():
            websites = [
                {"title": "Towards Data Science", "url": "towardsdatascience.com", "focus": "Articles on data science and machine learning"},
                {"title": "Machine Learning Mastery", "url": "machinelearningmastery.com", "focus": "Practical tutorials and guides"},
                {"title": "Distill.pub", "url": "distill.pub", "focus": "Clear explanations of machine learning research"},
                {"title": "Papers With Code", "url": "paperswithcode.com", "focus": "ML research papers with implementation code"},
                {"title": "AI Summer", "url": "theaisummer.com", "focus": "Deep learning concepts explained"}
            ]
        elif "web development" in subject.lower():
            websites = [
                {"title": "MDN Web Docs", "url": "developer.mozilla.org", "focus": "Comprehensive web development documentation"},
                {"title": "CSS-Tricks", "url": "css-tricks.com", "focus": "CSS techniques and tutorials"},
                {"title": "Smashing Magazine", "url": "smashingmagazine.com", "focus": "Web design and development articles"},
                {"title": "Dev.to", "url": "dev.to", "focus": "Community of web developers sharing knowledge"},
                {"title": "A List Apart", "url": "alistapart.com", "focus": "Web standards and best practices"}
            ]
        elif "financial literacy" in subject.lower() or "finance" in subject.lower():
            websites = [
                {"title": "Investopedia", "url": "investopedia.com", "focus": "Financial terms and concepts"},
                {"title": "Nerdwallet", "url": "nerdwallet.com", "focus": "Personal finance advice and tools"},
                {"title": "Bogleheads", "url": "bogleheads.org", "focus": "Investment philosophy and forum"},
                {"title": "Mr. Money Mustache", "url": "mrmoneymustache.com", "focus": "Financial independence and frugality"},
                {"title": "The Balance", "url": "thebalance.com", "focus": "Personal finance education"}
            ]
        elif "language" in subject.lower() or "spanish" in subject.lower():
            websites = [
                {"title": "SpanishDict", "url": "spanishdict.com", "focus": "Dictionary, conjugation, and learning tools"},
                {"title": "StudySpanish", "url": "studyspanish.com", "focus": "Grammar lessons and exercises"},
                {"title": "Notes in Spanish", "url": "notesinspanish.com", "focus": "Spanish podcasts for different levels"},
                {"title": "Fluencia", "url": "fluencia.com", "focus": "Interactive Spanish courses"},
                {"title": "Lingolia Spanish", "url": "spanish.lingolia.com", "focus": "Grammar explanations and exercises"}
            ]
        elif "digital marketing" in subject.lower():
            websites = [
                {"title": "Search Engine Journal", "url": "searchenginejournal.com", "focus": "SEO and search marketing news"},
                {"title": "Moz Blog", "url": "moz.com/blog", "focus": "SEO techniques and industry insights"},
                {"title": "Content Marketing Institute", "url": "contentmarketinginstitute.com", "focus": "Content marketing strategies"},
                {"title": "Social Media Examiner", "url": "socialmediaexaminer.com", "focus": "Social media marketing tactics"},
                {"title": "HubSpot Blog", "url": "hubspot.com/blog", "focus": "Inbound marketing and sales"}
            ]
        else:
            # Generic websites
            websites = [
                {"title": f"{subject} Hub", "url": f"{subject.lower().replace(' ', '')}hub.com", "focus": "Comprehensive resources and guides"},
                {"title": f"{subject} Daily", "url": f"{subject.lower().replace(' ', '')}daily.com", "focus": "News and updates in the field"},
                {"title": f"{subject} Forum", "url": f"{subject.lower().replace(' ', '')}forum.com", "focus": "Community discussions and support"},
                {"title": f"{subject} Academy", "url": f"{subject.lower().replace(' ', '')}academy.com", "focus": "Structured learning resources"},
                {"title": f"{subject} Blog", "url": f"{subject.lower().replace(' ', '')}blog.com", "focus": "Expert articles and tutorials"}
            ]
        
        resources["Websites"] = websites
        
        # Tools
        tools = []
        if "machine learning" in subject.lower():
            tools = [
                {"title": "Python", "type": "Programming Language", "purpose": "Primary language for ML development"},
                {"title": "Jupyter Notebooks", "type": "Development Environment", "purpose": "Interactive code development and visualization"},
                {"title": "Scikit-learn", "type": "Library", "purpose": "Machine learning algorithms implementation"},
                {"title": "TensorFlow", "type": "Framework", "purpose": "Deep learning model development"},
                {"title": "PyTorch", "type": "Framework", "purpose": "Deep learning research and development"}
            ]
        elif "web development" in subject.lower():
            tools = [
                {"title": "Visual Studio Code", "type": "Code Editor", "purpose": "Writing and editing code"},
                {"title": "Chrome DevTools", "type": "Browser Tools", "purpose": "Debugging and performance analysis"},
                {"title": "Git", "type": "Version Control", "purpose": "Code versioning and collaboration"},
                {"title": "npm", "type": "Package Manager", "purpose": "Managing JavaScript dependencies"},
                {"title": "Figma", "type": "Design Tool", "purpose": "UI/UX design and prototyping"}
            ]
        elif "financial literacy" in subject.lower() or "finance" in subject.lower():
            tools = [
                {"title": "Mint", "type": "Budgeting App", "purpose": "Tracking expenses and budgeting"},
                {"title": "YNAB (You Need A Budget)", "type": "Budgeting App", "purpose": "Zero-based budgeting system"},
                {"title": "Personal Capital", "type": "Financial Dashboard", "purpose": "Investment tracking and net worth"},
                {"title": "Excel/Google Sheets", "type": "Spreadsheet", "purpose": "Custom financial calculations and tracking"},
                {"title": "Retirement Calculators", "type": "Online Tools", "purpose": "Retirement planning and projections"}
            ]
        elif "language" in subject.lower() or "spanish" in subject.lower():
            tools = [
                {"title": "Anki", "type": "Flashcard App", "purpose": "Spaced repetition vocabulary learning"},
                {"title": "Duolingo", "type": "Learning App", "purpose": "Gamified language learning"},
                {"title": "Tandem", "type": "Language Exchange App", "purpose": "Finding conversation partners"},
                {"title": "Reverso Context", "type": "Translation Tool", "purpose": "Seeing words in context"},
                {"title": "Language Immersion Extension", "type": "Browser Extension", "purpose": "Partial immersion while browsing"}
            ]
        elif "digital marketing" in subject.lower():
            tools = [
                {"title": "Google Analytics", "type": "Analytics Platform", "purpose": "Website traffic analysis"},
                {"title": "SEMrush", "type": "SEO Tool", "purpose": "Keyword research and competitor analysis"},
                {"title": "Ahrefs", "type": "SEO Tool", "purpose": "Backlink analysis and keyword research"},
                {"title": "Hootsuite", "type": "Social Media Management", "purpose": "Scheduling and managing social posts"},
                {"title": "Mailchimp", "type": "Email Marketing", "purpose": "Email campaign creation and automation"}
            ]
        else:
            # Generic tools
            tools = [
                {"title": f"{subject} Software 1", "type": "Primary Tool", "purpose": "Core functionality for the field"},
                {"title": f"{subject} Platform", "type": "Online Platform", "purpose": "Cloud-based work and collaboration"},
                {"title": f"{subject} Assistant", "type": "Productivity Tool", "purpose": "Automating routine tasks"},
                {"title": f"{subject} Analytics", "type": "Analysis Tool", "purpose": "Data analysis and reporting"},
                {"title": f"{subject} Designer", "type": "Creation Tool", "purpose": "Creating and editing field-specific content"}
            ]
        
        resources["Tools"] = tools
        
        # Communities
        communities = []
        if "machine learning" in subject.lower():
            communities = [
                {"title": "r/MachineLearning", "platform": "Reddit", "focus": "Discussion of ML research and applications"},
                {"title": "Kaggle", "platform": "Kaggle.com", "focus": "Competitions and datasets for practice"},
                {"title": "Machine Learning Discord", "platform": "Discord", "focus": "Real-time chat and problem-solving"},
                {"title": "Data Science Stack Exchange", "platform": "Stack Exchange", "focus": "Q&A for data science and ML"},
                {"title": "PyTorch Forums", "platform": "Forums", "focus": "PyTorch-specific discussions"}
            ]
        elif "web development" in subject.lower():
            communities = [
                {"title": "Stack Overflow", "platform": "Stack Exchange", "focus": "Programming Q&A"},
                {"title": "r/webdev", "platform": "Reddit", "focus": "Web development discussions"},
                {"title": "Dev.to", "platform": "Dev.to", "focus": "Developer blog posts and discussions"},
                {"title": "Frontend Mentor", "platform": "frontendmentor.io", "focus": "Frontend coding challenges"},
                {"title": "CodePen", "platform": "codepen.io", "focus": "Sharing and discovering code snippets"}
            ]
        elif "financial literacy" in subject.lower() or "finance" in subject.lower():
            communities = [
                {"title": "r/personalfinance", "platform": "Reddit", "focus": "Personal finance advice and discussions"},
                {"title": "Bogleheads Forum", "platform": "bogleheads.org", "focus": "Investment philosophy discussions"},
                {"title": "r/financialindependence", "platform": "Reddit", "focus": "FIRE (Financial Independence, Retire Early) community"},
                {"title": "Money Stack Exchange", "platform": "Stack Exchange", "focus": "Personal finance Q&A"},
                {"title": "White Coat Investor Forum", "platform": "Forum", "focus": "Financial advice for professionals"}
            ]
        elif "language" in subject.lower() or "spanish" in subject.lower():
            communities = [
                {"title": "r/Spanish", "platform": "Reddit", "focus": "Spanish learning discussions and questions"},
                {"title": "SpanishDict Forums", "platform": "Forums", "focus": "Q&A about Spanish language"},
                {"title": "Language Exchange Discord", "platform": "Discord", "focus": "Finding conversation partners"},
                {"title": "HelloTalk", "platform": "Mobile App", "focus": "Language exchange community"},
                {"title": "Spanish Stack Exchange", "platform": "Stack Exchange", "focus": "Spanish language Q&A"}
            ]
        elif "digital marketing" in subject.lower():
            communities = [
                {"title": "r/digital_marketing", "platform": "Reddit", "focus": "Digital marketing discussions"},
                {"title": "Moz Q&A", "platform": "Moz.com", "focus": "SEO questions and answers"},
                {"title": "Growth Hackers", "platform": "growthhackers.com", "focus": "Growth marketing community"},
                {"title": "Inbound.org", "platform": "Website", "focus": "Inbound marketing community"},
                {"title": "Digital Marketing Discord", "platform": "Discord", "focus": "Real-time marketing discussions"}
            ]
        else:
            # Generic communities
            communities = [
                {"title": f"r/{subject.replace(' ', '')}", "platform": "Reddit", "focus": f"{subject} discussions and news"},
                {"title": f"{subject} Forum", "platform": "Online Forum", "focus": "Community support and discussions"},
                {"title": f"{subject} Discord", "platform": "Discord", "focus": "Real-time chat and networking"},
                {"title": f"{subject} Stack Exchange", "platform": "Stack Exchange", "focus": f"{subject} Q&A"},
                {"title": f"{subject} Facebook Group", "platform": "Facebook", "focus": "Sharing resources and networking"}
            ]
        
        resources["Communities"] = communities
        
        return resources
    
    def _handle_travel_planning(self, task: Task) -> None:
        """Handle creating travel itineraries and guides."""
        try:
            # Extract destination from task description
            destination_match = re.search(r'(?:itinerary|travel|guide) for ([^\'\"]+)', task.description, re.IGNORECASE)
            if not destination_match:
                destination_match = re.search(r'to ([^\'\"]+)', task.description, re.IGNORECASE)
            
            if not destination_match:
                task.fail("Could not extract destination from task description")
                return
                
            destination = destination_match.group(1).strip()
            task.update_progress(0.2)
            
            logger.info(f"[SmartTaskProcessor] Creating travel plan for: {destination}")
            
            # Extract duration if specified
            duration = "2 weeks"  # Default
            duration_match = re.search(r'(\d+)[\s-](?:day|week)', task.description, re.IGNORECASE)
            if duration_match:
                duration_num = duration_match.group(1)
                if "week" in task.description.lower():
                    duration = f"{duration_num} week" + ("s" if int(duration_num) > 1 else "")
                else:
                    duration = f"{duration_num} day" + ("s" if int(duration_num) > 1 else "")
            
            # Generate travel itinerary
            itinerary = self._generate_travel_itinerary(destination, duration)
            task.update_progress(0.6)
            
            # Generate practical information
            practical_info = self._generate_travel_practical_info(destination)
            task.update_progress(0.8)
            
            # Generate cultural tips
            cultural_tips = self._generate_cultural_tips(destination)
            task.update_progress(0.9)
            
            task.complete({
                "status": "success",
                "destination": destination,
                "duration": duration,
                "itinerary": itinerary,
                "practical_info": practical_info,
                "cultural_tips": cultural_tips,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"[SmartTaskProcessor] Created {duration} travel itinerary for {destination}")
            
        except Exception as e:
            task.fail(f"Error creating travel plan: {str(e)}")
    
    def _generate_travel_itinerary(self, destination: str, duration: str) -> List[Dict[str, Any]]:
        """Generate a travel itinerary for a specific destination and duration."""
        import random
        
        # Seed with destination to get consistent results
        random.seed(hash(destination) % 70000)
        
        # Parse duration
        duration_parts = duration.split()
        duration_num = int(duration_parts[0])
        duration_unit = duration_parts[1].lower()
        
        # Convert to days
        days = duration_num
        if "week" in duration_unit:
            days = duration_num * 7
        
        # Define itinerary structure based on destination
        itinerary = []
        
        # Japan-specific itinerary
        if "japan" in destination.lower():
            # Major cities and regions
            regions = [
                {"name": "Tokyo", "days": 4, "highlights": ["Shibuya Crossing", "Meiji Shrine", "Tokyo Skytree", "Tsukiji Outer Market", "Akihabara", "Shinjuku Gyoen", "Harajuku"]},
                {"name": "Kyoto", "days": 3, "highlights": ["Fushimi Inari Shrine", "Kinkaku-ji (Golden Pavilion)", "Arashiyama Bamboo Grove", "Gion District", "Kiyomizu-dera Temple"]},
                {"name": "Osaka", "days": 2, "highlights": ["Osaka Castle", "Dotonbori", "Kuromon Market", "Universal Studios Japan", "Umeda Sky Building"]},
                {"name": "Hiroshima", "days": 1, "highlights": ["Peace Memorial Park", "Atomic Bomb Dome", "Hiroshima Castle", "Miyajima Island", "Itsukushima Shrine"]},
                {"name": "Hakone", "days": 1, "highlights": ["Mount Fuji views", "Lake Ashi", "Hakone Open-Air Museum", "Owakudani", "Hot springs"]},
                {"name": "Nara", "days": 1, "highlights": ["Nara Park", "Todai-ji Temple", "Deer feeding", "Kasuga Taisha Shrine", "Isuien Garden"]},
                {"name": "Kanazawa", "days": 1, "highlights": ["Kenrokuen Garden", "Kanazawa Castle", "Higashi Chaya District", "21st Century Museum", "Omicho Market"]},
                {"name": "Nikko", "days": 1, "highlights": ["Toshogu Shrine", "Shinkyo Bridge", "Lake Chuzenji", "Kegon Falls", "Rinnoji Temple"]},
                {"name": "Takayama", "days": 1, "highlights": ["Old Town", "Morning Markets", "Hida Folk Village", "Takayama Jinya", "Sanmachi Suji"]},
                {"name": "Kamakura", "days": 1, "highlights": ["Great Buddha", "Hase-dera Temple", "Hokokuji Bamboo Garden", "Tsurugaoka Hachimangu Shrine", "Enoshima Island"]}
            ]
            
            # Adjust for trip duration
            if days <= 7:
                # Short trip: Focus on Tokyo and Kyoto
                regions = [region for region in regions if region["name"] in ["Tokyo", "Kyoto"]]
                regions[0]["days"] = min(4, days - 2)  # Tokyo
                regions[1]["days"] = min(3, days - regions[0]["days"])  # Kyoto
            elif days <= 10:
                # Medium trip: Tokyo, Kyoto, Osaka, and maybe one more
                regions = [region for region in regions if region["name"] in ["Tokyo", "Kyoto", "Osaka", "Hakone"]]
                if sum(region["days"] for region in regions) > days:
                    # Adjust days
                    excess = sum(region["days"] for region in regions) - days
                    for i in range(excess):
                        idx = random.randint(0, len(regions) - 1)
                        if regions[idx]["days"] > 1:
                            regions[idx]["days"] -= 1
            else:
                # Longer trip: Can include more regions
                while sum(region["days"] for region in regions) > days:
                    # Remove one region at a time until we fit
                    if len(regions) <= 3:
                        # Don't remove core regions, just reduce days
                        excess = sum(region["days"] for region in regions) - days
                        for i in range(excess):
                            idx = random.randint(0, len(regions) - 1)
                            if regions[idx]["days"] > 1:
                                regions[idx]["days"] -= 1
                    else:
                        # Remove a random non-core region
                        non_core = [i for i, region in enumerate(regions) if region["name"] not in ["Tokyo", "Kyoto", "Osaka"]]
                        if non_core:
                            regions.pop(random.choice(non_core))
            
            # Create daily itinerary
            day_counter = 1
            for region in regions:
                for day in range(region["days"]):
                    # Select highlights for this day (2-4 per day)
                    num_highlights = min(len(region["highlights"]), random.randint(2, 4))
                    day_highlights = random.sample(region["highlights"], num_highlights)
                    
                    # Morning activity
                    morning = day_highlights[0] if day_highlights else "Explore the area"
                    
                    # Afternoon activity
                    afternoon = day_highlights[1] if len(day_highlights) > 1 else "Local sightseeing"
                    
                    # Evening activity
                    if region["name"] in ["Tokyo", "Osaka", "Kyoto"]:
                        evening_options = [
                            f"Dinner in {random.choice(['a local restaurant', 'an izakaya', 'a ramen shop'])}",
                            f"Explore {random.choice(['night markets', 'shopping districts', 'entertainment areas'])}",
                            "Experience local nightlife",
                            "Cultural performance or show"
                        ]
                    else:
                        evening_options = [
                            "Relax at accommodation",
                            "Dinner with local specialties",
                            "Hot springs (onsen) if available",
                            "Stroll through evening streets"
                        ]
                    evening = random.choice(evening_options)
                    
                    # Create day entry
                    day_entry = {
                        "day": day_counter,
                        "location": region["name"],
                        "title": f"Day {day_counter}: {region['name']}" + (" - Arrival" if day == 0 else ""),
                        "morning": morning,
                        "afternoon": afternoon,
                        "evening": evening,
                        "accommodation": f"{region['name']}",
                        "transportation": self._get_transportation_in_japan(day_counter, regions, day, region)
                    }
                    
                    itinerary.append(day_entry)
                    day_counter += 1
        
        # Italy-specific itinerary
        elif "italy" in destination.lower():
            # Major cities and regions
            regions = [
                {"name": "Rome", "days": 3, "highlights": ["Colosseum", "Vatican Museums", "Roman Forum", "Trevi Fountain", "Pantheon", "Spanish Steps", "Piazza Navona"]},
                {"name": "Florence", "days": 2, "highlights": ["Uffizi Gallery", "Duomo", "Ponte Vecchio", "Accademia Gallery (David)", "Pitti Palace", "Boboli Gardens"]},
                {"name": "Venice", "days": 2, "highlights": ["St. Mark's Square", "Doge's Palace", "Grand Canal", "Rialto Bridge", "Murano and Burano Islands", "Basilica di San Marco"]},
                {"name": "Amalfi Coast", "days": 2, "highlights": ["Positano", "Ravello", "Amalfi Town", "Boat Tour", "Path of the Gods Hike", "Emerald Grotto"]},
                {"name": "Cinque Terre", "days": 2, "highlights": ["Monterosso al Mare", "Vernazza", "Coastal Hiking", "Manarola", "Riomaggiore", "Beach time"]},
                {"name": "Milan", "days": 1, "highlights": ["Duomo di Milano", "Galleria Vittorio Emanuele II", "Sforza Castle", "The Last Supper", "Brera District"]},
                {"name": "Naples", "days": 1, "highlights": ["Archaeological Museum", "Spaccanapoli", "Pizza tasting", "Castel dell'Ovo", "Capodimonte Museum"]},
                {"name": "Tuscany countryside", "days": 2, "highlights": ["Siena", "San Gimignano", "Wine tasting", "Val d'Orcia", "Montepulciano", "Pienza"]},
                {"name": "Pompeii", "days": 1, "highlights": ["Archaeological Site", "Mount Vesuvius", "Villa of the Mysteries", "Forum", "Amphitheater"]},
                {"name": "Bologna", "days": 1, "highlights": ["Piazza Maggiore", "Two Towers", "Food tour", "Archiginnasio", "Santo Stefano"]}
            ]
            
            # Adjust for trip duration
            if days <= 7:
                # Short trip: Focus on Rome, Florence, Venice
                regions = [region for region in regions if region["name"] in ["Rome", "Florence", "Venice"]]
                if sum(region["days"] for region in regions) > days:
                    # Adjust days
                    excess = sum(region["days"] for region in regions) - days
                    for i in range(excess):
                        idx = random.randint(0, len(regions) - 1)
                        if regions[idx]["days"] > 1:
                            regions[idx]["days"] -= 1
            elif days <= 10:
                # Medium trip: Core cities plus one more region
                regions = [region for region in regions if region["name"] in ["Rome", "Florence", "Venice", "Amalfi Coast", "Cinque Terre"]]
                while sum(region["days"] for region in regions) > days:
                    # Remove one region at a time until we fit
                    if len(regions) <= 3:
                        # Don't remove core regions, just reduce days
                        excess = sum(region["days"] for region in regions) - days
                        for i in range(excess):
                            idx = random.randint(0, len(regions) - 1)
                            if regions[idx]["days"] > 1:
                                regions[idx]["days"] -= 1
                    else:
                        # Remove a random non-core region
                        non_core = [i for i, region in enumerate(regions) if region["name"] not in ["Rome", "Florence", "Venice"]]
                        if non_core:
                            regions.pop(random.choice(non_core))
            else:
                # Longer trip: Can include more regions
                while sum(region["days"] for region in regions) > days:
                    # Remove one region at a time until we fit
                    if len(regions) <= 3:
                        # Don't remove core regions, just reduce days
                        excess = sum(region["days"] for region in regions) - days
                        for i in range(excess):
                            idx = random.randint(0, len(regions) - 1)
                            if regions[idx]["days"] > 1:
                                regions[idx]["days"] -= 1
                    else:
                        # Remove a random non-core region
                        non_core = [i for i, region in enumerate(regions) if region["name"] not in ["Rome", "Florence", "Venice"]]
                        if non_core:
                            regions.pop(random.choice(non_core))
            
            # Create daily itinerary
            day_counter = 1
            for region in regions:
                for day in range(region["days"]):
                    # Select highlights for this day (2-4 per day)
                    num_highlights = min(len(region["highlights"]), random.randint(2, 4))
                    day_highlights = random.sample(region["highlights"], num_highlights)
                    
                    # Morning activity
                    morning = day_highlights[0] if day_highlights else "Explore the area"
                    
                    # Afternoon activity
                    afternoon = day_highlights[1] if len(day_highlights) > 1 else "Local sightseeing"
                    
                    # Evening activity
                    evening_options = [
                        f"Dinner at {random.choice(['a local trattoria', 'a family-run restaurant', 'a pizzeria', 'a wine bar'])}",
                        f"Evening passeggiata (stroll) in {random.choice(['the historic center', 'local piazzas', 'along the waterfront'])}",
                        "Gelato tasting",
                        "Aperitivo experience"
                    ]
                    evening = random.choice(evening_options)
                    
                    # Create day entry
                    day_entry = {
                        "day": day_counter,
                        "location": region["name"],
                        "title": f"Day {day_counter}: {region['name']}" + (" - Arrival" if day == 0 else ""),
                        "morning": morning,
                        "afternoon": afternoon,
                        "evening": evening,
                        "accommodation": f"{region['name']}",
                        "transportation": self._get_transportation_in_italy(day_counter, regions, day, region)
                    }
                    
                    itinerary.append(day_entry)
                    day_counter += 1
        
        # Generic itinerary for other destinations
        else:
            # Create a generic itinerary
            for day in range(1, days + 1):
                # Create day entry
                day_entry = {
                    "day": day,
                    "location": destination,
                    "title": f"Day {day}: {destination}" + (" - Arrival" if day == 1 else " - Departure" if day == days else ""),
                    "morning": f"Explore {destination} - Morning activities",
                    "afternoon": f"Explore {destination} - Afternoon activities",
                    "evening": f"Explore {destination} - Evening activities",
                    "accommodation": f"{destination}",
                    "transportation": "Local transportation"
                }
                
                itinerary.append(day_entry)
        
        return itinerary
    
    def _get_transportation_in_japan(self, day_counter, regions, day_in_region, current_region):
        """Helper to determine transportation for Japan itinerary."""
        import random
        
        # First day - arrival
        if day_counter == 1:
            return "Airport transfer to accommodation"
        
        # First day in a new region (except first day of trip)
        if day_in_region == 0 and day_counter > 1:
            prev_region_idx = 0
            for i, region in enumerate(regions):
                if region["name"] == current_region["name"]:
                    prev_region_idx = i - 1
                    break
            
            if prev_region_idx >= 0:
                prev_region = regions[prev_region_idx]["name"]
                
                # Major routes
                if (prev_region == "Tokyo" and current_region["name"] == "Kyoto") or \
                   (prev_region == "Kyoto" and current_region["name"] == "Tokyo"):
                    return "Shinkansen (Bullet Train) from " + prev_region + " to " + current_region["name"]
                
                if (prev_region == "Kyoto" and current_region["name"] == "Osaka") or \
                   (prev_region == "Osaka" and current_region["name"] == "Kyoto"):
                    return "JR Train from " + prev_region + " to " + current_region["name"] + " (30 minutes)"
                
                # Default for other routes
                return "Train from " + prev_region + " to " + current_region["name"]
        
        # Last day of trip
        if day_counter == sum(region["days"] for region in regions):
            return "Transfer to airport for departure"
        
        # Regular day within same region
        transport_options = ["Subway", "Local trains", "Walking", "Bus", "Taxi for some destinations"]
        return random.choice(transport_options)
    
    def _get_transportation_in_italy(self, day_counter, regions, day_in_region, current_region):
        """Helper to determine transportation for Italy itinerary."""
        import random
        
        # First day - arrival
        if day_counter == 1:
            return "Airport transfer to accommodation"
        
        # First day in a new region (except first day of trip)
        if day_in_region == 0 and day_counter > 1:
            prev_region_idx = 0
            for i, region in enumerate(regions):
                if region["name"] == current_region["name"]:
                    prev_region_idx = i - 1
                    break
            
            if prev_region_idx >= 0:
                prev_region = regions[prev_region_idx]["name"]
                
                # Major routes
                if (prev_region == "Rome" and current_region["name"] == "Florence") or \
                   (prev_region == "Florence" and current_region["name"] == "Rome"):
                    return "High-speed train from " + prev_region + " to " + current_region["name"] + " (1.5 hours)"
                
                if (prev_region == "Florence" and current_region["name"] == "Venice") or \
                   (prev_region == "Venice" and current_region["name"] == "Florence"):
                    return "High-speed train from " + prev_region + " to " + current_region["name"] + " (2 hours)"
                
                if (prev_region == "Rome" and current_region["name"] == "Naples") or \
                   (prev_region == "Naples" and current_region["name"] == "Rome"):
                    return "High-speed train from " + prev_region + " to " + current_region["name"] + " (1 hour)"
                
                # Default for other routes
                return "Train from " + prev_region + " to " + current_region["name"]
        
        # Last day of trip
        if day_counter == sum(region["days"] for region in regions):
            return "Transfer to airport for departure"
        
        # Regular day within same region
        if current_region["name"] == "Venice":
            transport_options = ["Vaporetto (water bus)", "Walking", "Private water taxi"]
        elif current_region["name"] == "Amalfi Coast":
            transport_options = ["SITA bus between towns", "Ferry service (seasonal)", "Private driver", "Local bus"]
        elif current_region["name"] == "Cinque Terre":
            transport_options = ["Regional train between villages", "Hiking trails", "Ferry (seasonal)"]
        elif current_region["name"] == "Tuscany countryside":
            transport_options = ["Rental car", "Organized tour", "Regional bus"]
        else:
            transport_options = ["Public transportation", "Walking", "Metro", "Bus", "Taxi for some destinations"]
        
        return random.choice(transport_options)
    
    def _generate_travel_practical_info(self, destination: str) -> Dict[str, Any]:
        """Generate practical travel information for a destination."""
        import random
        
        # Seed with destination to get consistent results
        random.seed(hash(destination) % 80000)
        
        # Define practical information structure
        practical_info = {
            "best_time_to_visit": "",
            "visa_requirements": "",
            "currency": "",
            "language": "",
            "transportation": [],
            "accommodation_options": [],
            "budget_estimate": {},
            "health_and_safety": [],
            "packing_tips": []
        }
        
        # Japan-specific information
        if "japan" in destination.lower():
            practical_info["best_time_to_visit"] = "Spring (March to May) for cherry blossoms and Fall (September to November) for autumn colors. Avoid the rainy season (June) and peak summer heat (July-August) if possible."
            
            practical_info["visa_requirements"] = "Many Western countries have visa exemptions for stays up to 90 days. Check specific requirements for your nationality before traveling."
            
            practical_info["currency"] = "Japanese Yen (¥). Cash is still widely used, though credit cards are increasingly accepted in urban areas. ATMs at 7-Eleven and post offices reliably accept foreign cards."
            
            practical_info["language"] = "Japanese. English signage is common in major cities and tourist areas, but English speakers are not always easy to find. Learning basic phrases is helpful."
            
            practical_info["transportation"] = [
                "Japan Rail Pass: Consider purchasing before arrival if planning to travel between cities",
                "Shinkansen (Bullet Train): Fast, efficient way to travel between major cities",
                "Subway/Metro: Extensive networks in Tokyo, Osaka, and other major cities",
                "IC Cards (Suica/Pasmo/ICOCA): Rechargeable cards for convenient use on public transportation",
                "Taxis: Clean and reliable but expensive; drivers rarely speak English"
            ]
            
            practical_info["accommodation_options"] = [
                "Hotels: Western-style hotels available in all price ranges",
                "Ryokan: Traditional Japanese inns, often with tatami rooms and communal baths",
                "Business Hotels: Affordable, no-frills accommodations with small but efficient rooms",
                "Capsule Hotels: Unique budget option offering small sleeping pods",
                "Airbnb: Available in major cities but check local regulations"
            ]
            
            practical_info["budget_estimate"] = {
                "budget": "¥10,000-15,000/day ($70-100 USD)",
                "mid-range": "¥15,000-30,000/day ($100-200 USD)",
                "luxury": "¥30,000+/day ($200+ USD)"
            }
            
            practical_info["health_and_safety"] = [
                "Japan is generally very safe with low crime rates",
                "Tap water is safe to drink throughout the country",
                "Medical care is excellent but can be expensive; travel insurance recommended",
                "Be aware of natural disaster procedures (earthquakes, typhoons)",
                "Emergency number: 119 for ambulance/fire, 110 for police"
            ]
            
            practical_info["packing_tips"] = [
                "Comfortable walking shoes (you'll walk a lot)",
                "Conservative clothing for temple/shrine visits",
                "Portable Wi-Fi or SIM card for navigation",
                "Handkerchief (many public restrooms don't have paper towels)",
                "Small gifts from your home country if planning to meet locals"
            ]
        
        # Italy-specific information
        elif "italy" in destination.lower():
            practical_info["best_time_to_visit"] = "Spring (April to June) and Fall (September to October) offer pleasant weather and fewer crowds. Summer (July-August) is peak tourist season with higher prices and crowds. Winter is quieter but some attractions have reduced hours."
            
            practical_info["visa_requirements"] = "Italy is part of the Schengen Area. Many non-EU visitors can stay up to 90 days without a visa. Check specific requirements for your nationality."
            
            practical_info["currency"] = "Euro (€). Credit cards are widely accepted in most establishments, but it's good to carry some cash for small purchases and in rural areas."
            
            practical_info["language"] = "Italian. English is commonly spoken in tourist areas and hotels, less so in rural areas. Learning basic Italian phrases is appreciated by locals."
            
            practical_info["transportation"] = [
                "Trains: Efficient network connecting major cities; high-speed trains (Frecciarossa, Italo) for longer distances",
                "Local Buses: Good for reaching smaller towns and villages",
                "Metro: Available in Rome, Milan, Naples, and a few other cities",
                "Rental Car: Useful for exploring countryside regions like Tuscany, but avoid driving in city centers",
                "Ferries: Essential for visiting islands and coastal areas"
            ]
            
            practical_info["accommodation_options"] = [
                "Hotels: Available in all price ranges from luxury to budget",
                "Agriturismi: Farm stays offering authentic experiences, especially in rural areas",
                "B&Bs and Guesthouses: Often family-run with personal service",
                "Apartments: Good option for longer stays or families",
                "Hostels: Budget-friendly options in major cities"
            ]
            
            practical_info["budget_estimate"] = {
                "budget": "€50-100/day",
                "mid-range": "€100-200/day",
                "luxury": "€200+/day"
            }
            
            practical_info["health_and_safety"] = [
                "Italy is generally safe for tourists, but be aware of pickpockets in crowded tourist areas",
                "Tap water is safe to drink in most areas",
                "European Health Insurance Card (EHIC) is valid for EU citizens; others should have travel insurance",
                "Pharmacies (farmacie) can help with minor health issues",
                "Emergency number: 112 for all emergencies"
            ]
            
            practical_info["packing_tips"] = [
                "Comfortable walking shoes for cobblestone streets",
                "Modest clothing for visiting churches (shoulders and knees covered)",
                "Adapter for electrical outlets (Type F/L, 230V)",
                "Reusable water bottle (you can refill at many public fountains)",
                "Light scarf or shawl for women (useful for covering shoulders when entering churches)"
            ]
        
        # Generic information for other destinations
        else:
            practical_info["best_time_to_visit"] = f"Research the best seasons to visit {destination} based on weather, crowds, and any special events or festivals."
            
            practical_info["visa_requirements"] = f"Check visa requirements for {destination} based on your nationality. Some countries offer visa-free entry, while others require advance applications."
            
            practical_info["currency"] = f"Research the local currency of {destination}, exchange rates, and whether credit cards are widely accepted or cash is preferred."
            
            practical_info["language"] = f"The primary language(s) spoken in {destination}. Consider learning basic phrases to enhance your experience."
            
            practical_info["transportation"] = [
                "Research public transportation options",
                "Consider whether a rental car is necessary",
                "Look into transportation passes for tourists",
                "Understand taxi/rideshare availability",
                "Plan airport transfers in advance"
            ]
            
            practical_info["accommodation_options"] = [
                "Hotels in various price ranges",
                "Vacation rentals or apartments",
                "Hostels for budget travelers",
                "Local guesthouses or B&Bs",
                "Unique accommodation options specific to the destination"
            ]
            
            practical_info["budget_estimate"] = {
                "budget": "Research budget options",
                "mid-range": "Research mid-range costs",
                "luxury": "Research luxury options"
            }
            
            practical_info["health_and_safety"] = [
                f"Check travel advisories for {destination}",
                "Research required or recommended vaccinations",
                "Understand the local healthcare system",
                "Consider travel insurance coverage",
                "Note emergency contact numbers"
            ]
            
            practical_info["packing_tips"] = [
                "Weather-appropriate clothing",
                "Comfortable walking shoes",
                "Adapter for electrical outlets",
                "Essential medications and first aid",
                "Appropriate clothing for local customs and cultural sites"
            ]
        
        return practical_info
    
    def _generate_cultural_tips(self, destination: str) -> List[Dict[str, str]]:
        """Generate cultural tips for a destination."""
        import random
        
        # Seed with destination to get consistent results
        random.seed(hash(destination) % 90000)
        
        # Define cultural tips structure
        cultural_tips = []
        
        # Japan-specific tips
        if "japan" in destination.lower():
            cultural_tips = [
                {
                    "category": "Etiquette",
                    "tip": "Bow when greeting people. The depth and duration of the bow indicates the level of respect."
                },
                {
                    "category": "Dining",
                    "tip": "Don't stick your chopsticks vertically in rice as this resembles funeral rituals. Place them on the chopstick rest when not in use."
                },
                {
                    "category": "Footwear",
                    "tip": "Remove shoes before entering homes, traditional restaurants, ryokans, and some museums. Look for shoe racks or slippers at the entrance."
                },
                {
                    "category": "Onsen (Hot Springs)",
                    "tip": "Tattoos may be prohibited in some onsen as they're associated with yakuza (organized crime). Look for tattoo-friendly establishments if this applies to you."
                },
                {
                    "category": "Public Behavior",
                    "tip": "Avoid eating while walking or talking loudly on phones in public transportation. Japanese culture values quietness in public spaces."
                },
                {
                    "category": "Tipping",
                    "tip": "Tipping is not customary and can even be considered rude. Service charge is included in bills at restaurants and hotels."
                },
                {
                    "category": "Gift Giving",
                    "tip": "Gifts are important in Japanese culture. If invited to someone's home, bring a small gift, preferably wrapped. Receive gifts with both hands."
                },
                {
                    "category": "Business Cards",
                    "tip": "Exchange business cards (meishi) with both hands and a slight bow. Examine the card before putting it away respectfully."
                },
                {
                    "category": "Temples & Shrines",
                    "tip": "Wash your hands and mouth at the purification fountain before entering Shinto shrines. Follow specific prayer rituals (two bows, two claps, one bow)."
                },
                {
                    "category": "Trains",
                    "tip": "Wait for passengers to exit before boarding. Form orderly lines at marked positions on platforms. Avoid phone conversations on trains."
                }
            ]
        
        # Italy-specific tips
        elif "italy" in destination.lower():
            cultural_tips = [
                {
                    "category": "Greetings",
                    "tip": "Italians typically greet with a kiss on both cheeks among friends and family. In formal situations, a handshake is appropriate."
                },
                {
                    "category": "Dining",
                    "tip": "Cappuccino is considered a breakfast drink only. Ordering it after 11 AM marks you as a tourist. Espresso is the standard coffee after meals."
                },
                {
                    "category": "Meal Times",
                    "tip": "Italians eat lunch around 1-2 PM and dinner no earlier than 8 PM. Many restaurants won't open for dinner until 7:30 PM."
                },
                {
                    "category": "Cover Charge",
                    "tip": "Many restaurants charge a 'coperto' (cover charge) per person, which covers bread and table service. This is standard practice, not a tourist trap."
                },
                {
                    "category": "Dress Code",
                    "tip": "Italians dress well even for casual occasions. When visiting churches, shoulders and knees must be covered out of respect."
                },
                {
                    "category": "Tipping",
                    "tip": "Tipping is not expected as service is usually included. Rounding up the bill or leaving a few euros for exceptional service is appreciated but not required."
                },
                {
                    "category": "Passeggiata",
                    "tip": "The evening stroll (passeggiata) is an important social ritual. Join locals between 5-8 PM as they walk, socialize, and show off their finest clothes."
                },
                {
                    "category": "Water Fountains",
                    "tip": "Public drinking fountains (nasoni in Rome) provide safe, free drinking water. Bring a reusable bottle to refill."
                },
                {
                    "category": "Driving",
                    "tip": "Many historic city centers have ZTL zones (limited traffic zones) where only authorized vehicles can enter. Violations result in hefty fines."
                },
                {
                    "category": "Shopping Hours",
                    "tip": "Many shops close for a long lunch break (riposo) from roughly 1-4 PM, especially in smaller towns. Plan shopping accordingly."
                }
            ]
        
        # Generic tips for other destinations
        else:
            cultural_tips = [
                {
                    "category": "Greetings",
                    "tip": f"Research appropriate greeting customs in {destination}, including handshakes, bows, or other traditional greetings."
                },
                {
                    "category": "Dining Etiquette",
                    "tip": f"Learn basic table manners specific to {destination}, including how to use local utensils and appropriate mealtime behavior."
                },
                {
                    "category": "Dress Code",
                    "tip": f"Understand appropriate attire for different settings in {destination}, especially for religious sites or formal occasions."
                },
                {
                    "category": "Religious Customs",
                    "tip": f"Respect local religious practices and customs in {destination}, including appropriate behavior at places of worship."
                },
                {
                    "category": "Photography",
                    "tip": "Always ask permission before photographing local people, and be aware of restrictions at religious or cultural sites."
                },
                {
                    "category": "Gestures",
                    "tip": f"Be aware that common gestures may have different meanings in {destination}. Research potentially offensive gestures to avoid."
                },
                {
                    "category": "Tipping",
                    "tip": f"Understand local tipping customs in {destination}, including appropriate amounts and situations where tipping is expected or refused."
                },
                {
                    "category": "Bargaining",
                    "tip": f"Learn whether bargaining is customary in markets and shops in {destination}, and how to do so respectfully."
                },
                {
                    "category": "Public Behavior",
                    "tip": f"Observe local norms for behavior in public spaces in {destination}, including appropriate volume levels and personal space."
                },
                {
                    "category": "Gift Giving",
                    "tip": f"If visiting locals, research appropriate and inappropriate gifts in {destination}'s culture."
                }
            ]
        
        return cultural_tips

    def _create_subtask(self, parent_task: Task, description: str) -> Task:
        """
        Helper to create a subtask and store it.
        The subtask has a slightly higher (lower integer) priority than parent to emphasize immediate follow-up.
        """
        new_priority = max(0, parent_task.priority - 1)  # ensure subtask is at least as high or higher priority
        
        # Determine an appropriate timeout based on the task description
        timeout_seconds = None
        if "fetch" in description.lower() or "download" in description.lower():
            timeout_seconds = 30  # 30 seconds for fetch tasks
        elif "summarize" in description.lower():
            timeout_seconds = 10  # 10 seconds for summarize tasks
        elif "calculate" in description.lower() or "process" in description.lower():
            timeout_seconds = 60  # 60 seconds for calculation tasks
        elif "research" in description.lower() or "analyze" in description.lower():
            timeout_seconds = 120  # 120 seconds for research tasks
        elif "create" in description.lower() or "generate" in description.lower():
            timeout_seconds = 90  # 90 seconds for creative tasks
            
        # Create the subtask using the memory store's create_task method
        subtask = self.memory_store.create_task(
            priority=new_priority,
            description=description,
            parent_id=parent_task.task_id,
            timeout_seconds=timeout_seconds
        )
        
        return subtask
    
    # ===== LEGAL AND REGULATORY ANALYSIS HANDLERS =====
    
    def _handle_legal_analysis(self, task: Task) -> None:
        """Handle legal and regulatory analysis tasks."""
        try:
            # Extract topic from task description
            topic_match = re.search(r'(?:legal|regulatory|regulation|law|compliance) (?:of|for|on|about) ([^\'\"]+)', task.description, re.IGNORECASE)
            if not topic_match:
                topic_match = re.search(r'(?:analyze|research|compare) ([^\'\"]+) (?:regulation|law|compliance|legal)', task.description, re.IGNORECASE)
            
            if not topic_match:
                task.fail("Could not extract legal topic from task description")
                return
                
            topic = topic_match.group(1).strip()
            task.update_progress(0.2)
            
            logger.info(f"[SmartTaskProcessor] Analyzing legal/regulatory framework for: {topic}")
            
            # Generate legal analysis
            legal_analysis = self._generate_legal_analysis(topic)
            task.update_progress(0.7)
            
            # Generate compliance guidelines
            compliance_guidelines = self._generate_compliance_guidelines(topic)
            task.update_progress(0.9)
            
            task.complete({
                "status": "success",
                "topic": topic,
                "legal_analysis": legal_analysis,
                "compliance_guidelines": compliance_guidelines,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"[SmartTaskProcessor] Completed legal analysis for '{topic}'")
            
        except Exception as e:
            task.fail(f"Error performing legal analysis: {str(e)}")
    
    def _generate_legal_analysis(self, topic: str) -> Dict[str, Any]:
        """Generate a legal and regulatory analysis for a specific topic."""
        import random
        
        # Seed with topic to get consistent results
        random.seed(hash(topic) % 100000)
        
        analysis = {
            "topic": topic,
            "summary": "",
            "key_regulations": [],
            "jurisdictional_comparison": [],
            "recent_developments": [],
            "legal_risks": [],
            "case_studies": []
        }
        
        # AI regulation analysis
        if "ai" in topic.lower() or "artificial intelligence" in topic.lower():
            analysis["summary"] = "Artificial Intelligence regulation is evolving rapidly across global jurisdictions, with the EU taking a leading role through the AI Act. The regulatory landscape focuses on risk-based approaches, transparency requirements, and ethical considerations, with significant variations between regions."
            
            analysis["key_regulations"] = [
                {
                    "name": "EU AI Act",
                    "jurisdiction": "European Union",
                    "status": "Adopted, implementation in progress",
                    "key_provisions": [
                        "Risk-based categorization of AI systems",
                        "Prohibition of certain AI applications (social scoring, etc.)",
                        "Transparency requirements for high-risk AI systems",
                        "Conformity assessments and compliance framework",
                        "Penalties for non-compliance up to 6% of global annual turnover"
                    ]
                },
                {
                    "name": "AI Executive Order (14110)",
                    "jurisdiction": "United States",
                    "status": "In effect",
                    "key_provisions": [
                        "Safety and security standards for AI systems",
                        "Privacy protections and equity considerations",
                        "Federal agency coordination on AI governance",
                        "Promotion of innovation while managing risks",
                        "International cooperation on AI standards"
                    ]
                },
                {
                    "name": "AI Governance Framework",
                    "jurisdiction": "China",
                    "status": "Implemented in phases",
                    "key_provisions": [
                        "Algorithm registration requirements",
                        "Data security and cross-border data transfer restrictions",
                        "Content moderation requirements",
                        "Critical infrastructure protection",
                        "National security considerations"
                    ]
                },
                {
                    "name": "OECD AI Principles",
                    "jurisdiction": "International",
                    "status": "Non-binding guidelines",
                    "key_provisions": [
                        "Inclusive growth, sustainable development, and well-being",
                        "Human-centered values and fairness",
                        "Transparency and explainability",
                        "Robustness, security, and safety",
                        "Accountability"
                    ]
                }
            ]
            
            analysis["jurisdictional_comparison"] = [
                {
                    "region": "European Union",
                    "approach": "Comprehensive regulation with risk-based categorization",
                    "enforcement": "Strong, with significant penalties",
                    "key_focus": "Consumer protection and fundamental rights"
                },
                {
                    "region": "United States",
                    "approach": "Sector-specific regulation with voluntary frameworks",
                    "enforcement": "Varied, primarily through existing agencies (FTC, FDA, etc.)",
                    "key_focus": "Innovation and competitiveness with risk management"
                },
                {
                    "region": "China",
                    "approach": "Centralized control with national security emphasis",
                    "enforcement": "Strong government oversight",
                    "key_focus": "Social stability and strategic technology development"
                },
                {
                    "region": "United Kingdom",
                    "approach": "Pro-innovation regulation with sector-specific rules",
                    "enforcement": "Regulatory guidance with flexible implementation",
                    "key_focus": "Balancing innovation with appropriate safeguards"
                },
                {
                    "region": "Canada",
                    "approach": "Risk-based framework with ethical considerations",
                    "enforcement": "Moderate, with emphasis on guidance",
                    "key_focus": "Responsible AI development and use"
                }
            ]
            
            analysis["recent_developments"] = [
                {
                    "title": "EU AI Act Adoption",
                    "date": "2023-12",
                    "description": "The European Parliament and Council reached agreement on the final text of the AI Act, establishing the world's first comprehensive AI regulatory framework."
                },
                {
                    "title": "US Executive Order on AI",
                    "date": "2023-10",
                    "description": "President Biden issued Executive Order 14110 on Safe, Secure, and Trustworthy Development and Use of Artificial Intelligence."
                },
                {
                    "title": "China's Generative AI Regulations",
                    "date": "2023-07",
                    "description": "China implemented specific regulations for generative AI services, requiring security assessments and content monitoring."
                },
                {
                    "title": "UK AI Safety Summit",
                    "date": "2023-11",
                    "description": "The UK hosted an international summit focusing on frontier AI safety, resulting in the Bletchley Declaration signed by 28 countries."
                },
                {
                    "title": "G7 Hiroshima AI Process",
                    "date": "2023-05",
                    "description": "G7 leaders launched the Hiroshima AI Process to discuss international approaches to AI governance."
                }
            ]
            
            analysis["legal_risks"] = [
                {
                    "category": "Compliance",
                    "risks": [
                        "Failure to meet transparency requirements for high-risk AI systems",
                        "Inadequate risk assessment and management processes",
                        "Non-compliance with data protection regulations when training AI models",
                        "Insufficient documentation of AI system development and deployment",
                        "Lack of human oversight for critical AI applications"
                    ]
                },
                {
                    "category": "Liability",
                    "risks": [
                        "Unclear allocation of liability for AI-caused harm",
                        "Product liability issues for AI-enabled products",
                        "Potential for class action lawsuits related to algorithmic discrimination",
                        "Intellectual property infringement in AI training data",
                        "Professional liability for AI system designers and operators"
                    ]
                },
                {
                    "category": "Ethical and Reputational",
                    "risks": [
                        "Algorithmic bias leading to discrimination claims",
                        "Privacy violations through data processing or surveillance",
                        "Lack of explainability in high-stakes decision-making",
                        "Reputational damage from AI ethics controversies",
                        "Public backlash against perceived misuse of AI technology"
                    ]
                }
            ]
            
            analysis["case_studies"] = [
                {
                    "title": "GDPR Enforcement Against Facial Recognition",
                    "jurisdiction": "European Union",
                    "summary": "Multiple EU data protection authorities have taken enforcement actions against facial recognition technologies, imposing significant fines for GDPR violations related to biometric data processing without proper consent or legal basis."
                },
                {
                    "title": "Algorithmic Discrimination in Hiring",
                    "jurisdiction": "United States",
                    "summary": "The Equal Employment Opportunity Commission (EEOC) has investigated cases where AI-powered hiring tools allegedly discriminated against protected groups, leading to settlements and requirements for algorithmic impact assessments."
                },
                {
                    "title": "Content Moderation Liability",
                    "jurisdiction": "Multiple",
                    "summary": "Social media platforms using AI for content moderation have faced legal challenges regarding both over-removal of legitimate content and failure to remove harmful content, highlighting the complex liability issues in automated decision-making."
                }
            ]
            
        # Data privacy analysis
        elif "data privacy" in topic.lower() or "gdpr" in topic.lower() or "ccpa" in topic.lower():
            analysis["summary"] = "Data privacy regulations continue to proliferate globally, with the EU's GDPR setting a benchmark that has influenced many subsequent frameworks. Key trends include strengthening individual rights, increasing corporate accountability, and addressing cross-border data transfers. Compliance requires a comprehensive approach to data governance."
            
            analysis["key_regulations"] = [
                {
                    "name": "General Data Protection Regulation (GDPR)",
                    "jurisdiction": "European Union",
                    "status": "In effect since May 2018",
                    "key_provisions": [
                        "Legal basis required for data processing",
                        "Enhanced individual rights (access, deletion, portability)",
                        "Data breach notification requirements",
                        "Data Protection Impact Assessments",
                        "Significant penalties (up to 4% of global annual turnover)",
                        "Data transfer restrictions to non-adequate countries"
                    ]
                },
                {
                    "name": "California Consumer Privacy Act (CCPA) / California Privacy Rights Act (CPRA)",
                    "jurisdiction": "California, USA",
                    "status": "CCPA effective since 2020, CPRA enhancements since 2023",
                    "key_provisions": [
                        "Right to know what personal information is collected",
                        "Right to delete personal information",
                        "Right to opt-out of sale of personal information",
                        "Right to non-discrimination for exercising rights",
                        "Private right of action for data breaches",
                        "Creation of California Privacy Protection Agency"
                    ]
                },
                {
                    "name": "Personal Information Protection Law (PIPL)",
                    "jurisdiction": "China",
                    "status": "In effect since November 2021",
                    "key_provisions": [
                        "Consent requirements for data processing",
                        "Data localization requirements",
                        "Cross-border data transfer restrictions",
                        "Individual rights similar to GDPR",
                        "Algorithmic transparency provisions",
                        "Penalties up to 5% of annual revenue"
                    ]
                },
                {
                    "name": "Lei Geral de Proteção de Dados (LGPD)",
                    "jurisdiction": "Brazil",
                    "status": "In effect since 2020",
                    "key_provisions": [
                        "Legal bases for data processing",
                        "Data subject rights",
                        "Data Protection Officer requirements",
                        "Data breach notification",
                        "Administrative sanctions",
                        "National Data Protection Authority"
                    ]
                }
            ]
            
            analysis["jurisdictional_comparison"] = [
                {
                    "region": "European Union (GDPR)",
                    "approach": "Comprehensive regulation with broad territorial scope",
                    "enforcement": "Active enforcement with significant fines",
                    "key_focus": "Fundamental right to data protection"
                },
                {
                    "region": "United States",
                    "approach": "Sectoral laws with state-level comprehensive laws emerging",
                    "enforcement": "Varied by sector and state",
                    "key_focus": "Consumer protection and transparency"
                },
                {
                    "region": "China",
                    "approach": "Comprehensive laws with national security emphasis",
                    "enforcement": "Strong with significant penalties",
                    "key_focus": "Data sovereignty and security"
                },
                {
                    "region": "Canada",
                    "approach": "Federal PIPEDA with provincial variations",
                    "enforcement": "Moderate with focus on resolution",
                    "key_focus": "Reasonable expectation of privacy"
                },
                {
                    "region": "Australia",
                    "approach": "Privacy Act with Australian Privacy Principles",
                    "enforcement": "Moderate with increasing penalties",
                    "key_focus": "Reasonable handling of personal information"
                }
            ]
            
            analysis["recent_developments"] = [
                {
                    "title": "EU-US Data Privacy Framework",
                    "date": "2023-07",
                    "description": "New framework for transatlantic data flows, replacing the invalidated Privacy Shield, with enhanced surveillance safeguards and redress mechanism."
                },
                {
                    "title": "US State Privacy Laws",
                    "date": "2023",
                    "description": "Multiple states including Colorado, Connecticut, Utah, and Virginia have enacted comprehensive privacy laws following California's lead."
                },
                {
                    "title": "GDPR Enforcement on Cookie Consent",
                    "date": "2022-2023",
                    "description": "Increased enforcement actions against non-compliant cookie practices, with significant fines for dark patterns and invalid consent mechanisms."
                },
                {
                    "title": "Schrems II Aftermath",
                    "date": "2020-2023",
                    "description": "Ongoing implementation of enhanced measures for international data transfers following the CJEU's invalidation of Privacy Shield."
                },
                {
                    "title": "AI Act Data Protection Provisions",
                    "date": "2023",
                    "description": "The EU AI Act includes specific provisions on data governance and privacy, creating overlap with GDPR compliance requirements."
                }
            ]
            
            analysis["legal_risks"] = [
                {
                    "category": "Compliance",
                    "risks": [
                        "Inadequate legal basis for data processing",
                        "Failure to honor data subject rights requests",
                        "Insufficient data security measures",
                        "Non-compliant cross-border data transfers",
                        "Inadequate vendor management and data processing agreements"
                    ]
                },
                {
                    "category": "Enforcement",
                    "risks": [
                        "Regulatory investigations and fines",
                        "Compensation claims from affected individuals",
                        "Class action lawsuits in applicable jurisdictions",
                        "Injunctions against data processing activities",
                        "Reputational damage from public enforcement actions"
                    ]
                },
                {
                    "category": "Operational",
                    "risks": [
                        "Business disruption from prohibited data transfers",
                        "Inability to deploy global systems due to conflicting requirements",
                        "Increased operational costs for compliance",
                        "Friction in data-driven innovation",
                        "Challenges in data retention and deletion"
                    ]
                }
            ]
            
            analysis["case_studies"] = [
                {
                    "title": "Meta GDPR Fines",
                    "jurisdiction": "European Union",
                    "summary": "The Irish Data Protection Commission fined Meta over €1 billion across multiple enforcement actions related to Facebook and Instagram's data processing practices, including illegal data transfers to the US and forced consent for personalized advertising."
                },
                {
                    "title": "Sephora CCPA Enforcement",
                    "jurisdiction": "California, USA",
                    "summary": "California Attorney General fined Sephora $1.2 million for failing to disclose that it was selling consumer data and not honoring opt-out requests, marking the first major CCPA enforcement action."
                },
                {
                    "title": "H&M Employee Monitoring Fine",
                    "jurisdiction": "Germany",
                    "summary": "Hamburg's data protection authority fined H&M €35.3 million for excessive monitoring of employees' private lives through detailed records kept by managers, highlighting GDPR enforcement in the employment context."
                }
            ]
            
        # Cryptocurrency regulation analysis
        elif "crypto" in topic.lower() or "cryptocurrency" in topic.lower() or "bitcoin" in topic.lower() or "blockchain" in topic.lower():
            analysis["summary"] = "Cryptocurrency regulation varies significantly across jurisdictions, reflecting different approaches to innovation, consumer protection, and financial stability. The regulatory landscape is rapidly evolving as governments seek to address concerns about market integrity, financial crime, and investor protection while not stifling innovation in blockchain technology."
            
            analysis["key_regulations"] = [
                {
                    "name": "Markets in Crypto-Assets Regulation (MiCA)",
                    "jurisdiction": "European Union",
                    "status": "Adopted, phased implementation through 2024-2025",
                    "key_provisions": [
                        "Licensing requirements for crypto-asset service providers",
                        "Stablecoin regulation with reserve and governance requirements",
                        "Consumer protection measures",
                        "Market abuse prevention",
                        "Environmental impact disclosure for consensus mechanisms",
                        "Passporting regime across EU member states"
                    ]
                },
                {
                    "name": "Various SEC Actions and Guidance",
                    "jurisdiction": "United States",
                    "status": "Ongoing enforcement and rulemaking",
                    "key_provisions": [
                        "Application of Howey Test to determine security status",
                        "Registration requirements for crypto exchanges",
                        "Enforcement actions against unregistered securities offerings",
                        "Custody rules for digital assets",
                        "Proposed disclosure requirements for climate impact"
                    ]
                },
                {
                    "name": "Virtual Asset Service Provider (VASP) Regulations",
                    "jurisdiction": "FATF (Global)",
                    "status": "Recommendations implemented variously by countries",
                    "key_provisions": [
                        "Travel Rule for transaction information sharing",
                        "AML/CFT requirements for crypto businesses",
                        "Risk-based approach to virtual asset regulation",
                        "Licensing and registration requirements",
                        "International cooperation framework"
                    ]
                },
                {
                    "name": "Payment Services Act Amendments",
                    "jurisdiction": "Japan",
                    "status": "In effect with periodic updates",
                    "key_provisions": [
                        "Registration requirements for crypto exchanges",
                        "Customer asset segregation",
                        "Security and operational risk management",
                        "AML/CFT compliance",
                        "Stablecoin regulation"
                    ]
                }
            ]
            
            analysis["jurisdictional_comparison"] = [
                {
                    "region": "European Union",
                    "approach": "Comprehensive regulation with harmonized framework",
                    "enforcement": "Structured with clear authority allocation",
                    "key_focus": "Consumer protection and market integrity"
                },
                {
                    "region": "United States",
                    "approach": "Multi-agency oversight with emphasis on securities laws",
                    "enforcement": "Active enforcement by SEC, CFTC, and FinCEN",
                    "key_focus": "Investor protection and market integrity"
                },
                {
                    "region": "Singapore",
                    "approach": "Licensing framework with risk-based regulation",
                    "enforcement": "Selective with focus on major risks",
                    "key_focus": "Innovation balanced with risk management"
                },
                {
                    "region": "China",
                    "approach": "Prohibitive stance on cryptocurrency trading and mining",
                    "enforcement": "Strict with comprehensive bans",
                    "key_focus": "Financial stability and control"
                },
                {
                    "region": "Switzerland",
                    "approach": "Technology-neutral regulation with clear guidelines",
                    "enforcement": "Balanced with innovation support",
                    "key_focus": "Creating favorable conditions for legitimate crypto business"
                }
            ]
            
            analysis["recent_developments"] = [
                {
                    "title": "MiCA Adoption in EU",
                    "date": "2023-04",
                    "description": "The European Parliament adopted the Markets in Crypto-Assets Regulation, creating the world's first comprehensive crypto regulatory framework."
                },
                {
                    "title": "SEC Enforcement Actions",
                    "date": "2023",
                    "description": "Increased enforcement against major crypto exchanges and platforms, including Coinbase, Binance, and Kraken, alleging operation of unregistered securities exchanges."
                },
                {
                    "title": "Stablecoin Legislation Proposals",
                    "date": "2023",
                    "description": "Multiple jurisdictions proposed or implemented specific regulations for stablecoins following the collapse of Terra/Luna and concerns about financial stability."
                },
                {
                    "title": "DeFi Regulatory Attention",
                    "date": "2022-2023",
                    "description": "Growing regulatory focus on decentralized finance (DeFi) protocols, with questions about liability, compliance obligations, and application of existing frameworks."
                },
                {
                    "title": "Travel Rule Implementation",
                    "date": "2022-2023",
                    "description": "Progressive global implementation of the FATF Travel Rule requiring VASPs to share originator and beneficiary information for crypto transactions."
                }
            ]
            
            analysis["legal_risks"] = [
                {
                    "category": "Regulatory Classification",
                    "risks": [
                        "Tokens classified as securities triggering registration requirements",
                        "Exchange activities requiring licensing under various regimes",
                        "DeFi protocols potentially subject to financial regulation",
                        "NFTs potentially classified as securities or collectibles",
                        "Stablecoins subject to banking or e-money regulation"
                    ]
                },
                {
                    "category": "Compliance",
                    "risks": [
                        "Inadequate AML/KYC procedures",
                        "Failure to implement Travel Rule requirements",
                        "Cross-border service provision without appropriate licenses",
                        "Insufficient consumer protection measures",
                        "Inadequate disclosure of risks and conflicts of interest"
                    ]
                },
                {
                    "category": "Operational",
                    "risks": [
                        "Cybersecurity vulnerabilities leading to hacks or exploits",
                        "Smart contract failures or vulnerabilities",
                        "Custody risks and private key management",
                        "Oracle manipulation or failures",
                        "Governance attacks or vulnerabilities"
                    ]
                }
            ]
            
            analysis["case_studies"] = [
                {
                    "title": "SEC v. Ripple Labs",
                    "jurisdiction": "United States",
                    "summary": "Landmark case addressing whether XRP is a security, with partial ruling that programmatic sales of XRP did not constitute investment contracts, while direct sales to institutional investors did meet the Howey test criteria."
                },
                {
                    "title": "Tornado Cash Sanctions",
                    "jurisdiction": "United States",
                    "summary": "OFAC sanctioned the Tornado Cash mixing service for facilitating money laundering, raising novel legal questions about sanctioning decentralized protocols and open-source code."
                },
                {
                    "title": "FTX Collapse and Aftermath",
                    "jurisdiction": "Global",
                    "summary": "The collapse of major exchange FTX led to criminal charges, civil enforcement, and accelerated regulatory efforts worldwide to address custody, governance, and conflicts of interest in crypto businesses."
                }
            ]
            
        # Generic legal analysis for other topics
        else:
            analysis["summary"] = f"This analysis examines the legal and regulatory landscape for {topic}, including key regulations, jurisdictional approaches, recent developments, and associated legal risks. The regulatory environment continues to evolve as policymakers respond to emerging challenges and opportunities in this area."
            
            analysis["key_regulations"] = [
                {
                    "name": f"Primary {topic} Regulation",
                    "jurisdiction": "Major Jurisdiction",
                    "status": "Current implementation status",
                    "key_provisions": [
                        "Key provision 1",
                        "Key provision 2",
                        "Key provision 3",
                        "Key provision 4",
                        "Key provision 5"
                    ]
                },
                {
                    "name": f"Secondary {topic} Framework",
                    "jurisdiction": "Another Jurisdiction",
                    "status": "Implementation status",
                    "key_provisions": [
                        "Key provision 1",
                        "Key provision 2",
                        "Key provision 3",
                        "Key provision 4"
                    ]
                },
                {
                    "name": f"International {topic} Standards",
                    "jurisdiction": "International",
                    "status": "Adoption status",
                    "key_provisions": [
                        "Key provision 1",
                        "Key provision 2",
                        "Key provision 3"
                    ]
                }
            ]
            
            analysis["jurisdictional_comparison"] = [
                {
                    "region": "Jurisdiction 1",
                    "approach": "Regulatory approach description",
                    "enforcement": "Enforcement approach",
                    "key_focus": "Primary regulatory focus"
                },
                {
                    "region": "Jurisdiction 2",
                    "approach": "Regulatory approach description",
                    "enforcement": "Enforcement approach",
                    "key_focus": "Primary regulatory focus"
                },
                {
                    "region": "Jurisdiction 3",
                    "approach": "Regulatory approach description",
                    "enforcement": "Enforcement approach",
                    "key_focus": "Primary regulatory focus"
                }
            ]
            
            analysis["recent_developments"] = [
                {
                    "title": f"Recent {topic} Regulation",
                    "date": "Recent date",
                    "description": "Description of recent regulatory development"
                },
                {
                    "title": f"Important {topic} Case",
                    "date": "Case date",
                    "description": "Description of important case or precedent"
                },
                {
                    "title": f"New {topic} Guidance",
                    "date": "Publication date",
                    "description": "Description of new regulatory guidance"
                }
            ]
            
            analysis["legal_risks"] = [
                {
                    "category": "Compliance Risks",
                    "risks": [
                        "Compliance risk 1",
                        "Compliance risk 2",
                        "Compliance risk 3",
                        "Compliance risk 4"
                    ]
                },
                {
                    "category": "Operational Risks",
                    "risks": [
                        "Operational risk 1",
                        "Operational risk 2",
                        "Operational risk 3"
                    ]
                },
                {
                    "category": "Strategic Risks",
                    "risks": [
                        "Strategic risk 1",
                        "Strategic risk 2",
                        "Strategic risk 3"
                    ]
                }
            ]
            
            analysis["case_studies"] = [
                {
                    "title": f"Major {topic} Case Study",
                    "jurisdiction": "Relevant jurisdiction",
                    "summary": "Summary of important case study relevant to the topic"
                },
                {
                    "title": f"Secondary {topic} Example",
                    "jurisdiction": "Another jurisdiction",
                    "summary": "Summary of another relevant example"
                }
            ]
        
        return analysis
    
    def _generate_compliance_guidelines(self, topic: str) -> Dict[str, Any]:
        """Generate compliance guidelines for a specific legal/regulatory topic."""
        import random
        
        # Seed with topic to get consistent results
        random.seed(hash(topic) % 110000)
        
        guidelines = {
            "topic": topic,
            "overview": "",
            "key_compliance_steps": [],
            "documentation_requirements": [],
            "governance_framework": {},
            "implementation_timeline": [],
            "resources": []
        }
        
        # AI regulation compliance
        if "ai" in topic.lower() or "artificial intelligence" in topic.lower():
            guidelines["overview"] = "Compliance with AI regulations requires a comprehensive approach addressing risk assessment, transparency, documentation, and governance. Organizations should implement a structured compliance program that addresses both current requirements and emerging regulatory trends."
            
            guidelines["key_compliance_steps"] = [
                {
                    "step": "AI Inventory and Classification",
                    "description": "Catalog all AI systems in use or development and classify them according to regulatory risk categories (e.g., EU AI Act's unacceptable, high-risk, limited risk, minimal risk).",
                    "priority": "High"
                },
                {
                    "step": "Risk Assessment Framework",
                    "description": "Develop and implement a structured framework for assessing AI risks, including potential harms to individuals, groups, and society.",
                    "priority": "High"
                },
                {
                    "step": "Technical Documentation",
                    "description": "Create comprehensive technical documentation for each AI system, including training methodologies, data sources, validation procedures, and performance metrics.",
                    "priority": "High"
                },
                {
                    "step": "Human Oversight Mechanisms",
                    "description": "Implement appropriate human oversight for AI systems, particularly for high-risk applications, with clear escalation paths and intervention capabilities.",
                    "priority": "Medium"
                },
                {
                    "step": "Transparency Measures",
                    "description": "Develop user-facing transparency mechanisms, including disclosure of AI use, explanation of decisions, and limitations of the system.",
                    "priority": "Medium"
                },
                {
                    "step": "Testing and Validation",
                    "description": "Establish robust testing protocols for accuracy, bias, robustness, and cybersecurity, with particular attention to high-risk systems.",
                    "priority": "High"
                },
                {
                    "step": "Monitoring System",
                    "description": "Implement ongoing monitoring of AI systems in production to detect drift, performance issues, or unexpected behaviors.",
                    "priority": "Medium"
                },
                {
                    "step": "Incident Response Plan",
                    "description": "Develop procedures for responding to AI incidents, including investigation, mitigation, notification, and remediation steps.",
                    "priority": "Medium"
                }
            ]
            
            guidelines["documentation_requirements"] = [
                {
                    "document": "AI System Information Sheet",
                    "contents": [
                        "System purpose and intended use",
                        "Risk classification and justification",
                        "Technical architecture and components",
                        "Development team and responsibilities",
                        "Deployment status and locations"
                    ],
                    "audience": "Internal compliance team, regulators upon request"
                },
                {
                    "document": "Data Governance Documentation",
                    "contents": [
                        "Training data sources and characteristics",
                        "Data preprocessing and augmentation methods",
                        "Data quality assurance procedures",
                        "Data protection impact assessment",
                        "Data retention and deletion policies"
                    ],
                    "audience": "Development team, compliance officers, data protection authorities"
                },
                {
                    "document": "Risk Assessment Report",
                    "contents": [
                        "Identified risks and potential harms",
                        "Risk mitigation measures",
                        "Residual risks and justifications",
                        "Ongoing monitoring approach",
                        "Review schedule and triggers"
                    ],
                    "audience": "Senior management, compliance team, regulators"
                },
                {
                    "document": "Algorithmic Impact Assessment",
                    "contents": [
                        "Potential impacts on individuals and groups",
                        "Fairness and bias evaluation",
                        "Accessibility considerations",
                        "Societal and environmental impacts",
                        "Mitigation strategies"
                    ],
                    "audience": "Product teams, ethics committee, external stakeholders"
                },
                {
                    "document": "Technical Validation Report",
                    "contents": [
                        "Performance metrics and thresholds",
                        "Testing methodologies",
                        "Validation results across scenarios",
                        "Limitations and edge cases",
                        "Ongoing validation plan"
                    ],
                    "audience": "Engineering team, compliance officers, certification bodies"
                }
            ]
            
            guidelines["governance_framework"] = {
                "roles_and_responsibilities": [
                    {
                        "role": "AI Ethics Committee",
                        "responsibilities": [
                            "Review high-risk AI systems before deployment",
                            "Provide guidance on ethical questions",
                            "Approve exceptions to AI policies",
                            "Review significant incidents",
                            "Oversee compliance with ethical principles"
                        ]
                    },
                    {
                        "role": "AI Compliance Officer",
                        "responsibilities": [
                            "Oversee implementation of AI compliance program",
                            "Coordinate regulatory monitoring and response",
                            "Manage documentation and reporting",
                            "Conduct internal audits and assessments",
                            "Liaise with regulatory authorities"
                        ]
                    },
                    {
                        "role": "AI Development Teams",
                        "responsibilities": [
                            "Implement technical compliance requirements",
                            "Document development processes",
                            "Conduct initial risk assessments",
                            "Address identified issues and vulnerabilities",
                            "Participate in compliance reviews"
                        ]
                    },
                    {
                        "role": "Senior Management",
                        "responsibilities": [
                            "Set AI governance strategy",
                            "Allocate resources for compliance",
                            "Review high-risk AI initiatives",
                            "Approve AI policies and standards",
                            "Ensure accountability throughout organization"
                        ]
                    }
                ],
                "policies_and_procedures": [
                    "AI Development and Deployment Policy",
                    "AI Risk Assessment Procedure",
                    "AI Documentation Standards",
                    "AI Incident Response Procedure",
                    "AI Monitoring and Maintenance Policy",
                    "Third-Party AI Vendor Management Policy"
                ],
                "training_requirements": [
                    "Basic AI literacy for all employees",
                    "AI ethics training for development teams",
                    "Regulatory compliance training for relevant staff",
                    "Risk assessment methodology for product managers",
                    "Incident response training for operations teams"
                ]
            }
            
            guidelines["implementation_timeline"] = [
                {
                    "phase": "Initial Assessment",
                    "timeframe": "1-3 months",
                    "activities": [
                        "Inventory existing AI systems",
                        "Classify systems by risk level",
                        "Gap analysis against regulatory requirements",
                        "Establish governance structure"
                    ]
                },
                {
                    "phase": "Foundation Building",
                    "timeframe": "3-6 months",
                    "activities": [
                        "Develop key policies and procedures",
                        "Create documentation templates",
                        "Establish risk assessment framework",
                        "Implement basic monitoring capabilities"
                    ]
                },
                {
                    "phase": "High-Risk System Compliance",
                    "timeframe": "6-12 months",
                    "activities": [
                        "Complete documentation for high-risk systems",
                        "Implement enhanced oversight mechanisms",
                        "Conduct thorough risk assessments",
                        "Develop conformity assessment procedures"
                    ]
                },
                {
                    "phase": "Comprehensive Implementation",
                    "timeframe": "12-18 months",
                    "activities": [
                        "Extend compliance to all AI systems",
                        "Integrate compliance into development lifecycle",
                        "Implement advanced monitoring and testing",
                        "Conduct training across organization"
                    ]
                },
                {
                    "phase": "Continuous Improvement",
                    "timeframe": "Ongoing",
                    "activities": [
                        "Regular compliance reviews and updates",
                        "Regulatory horizon scanning",
                        "Periodic testing and validation",
                        "Refinement of governance processes"
                    ]
                }
            ]
            
            guidelines["resources"] = [
                {
                    "title": "EU AI Act Technical Documentation Guidelines",
                    "type": "Regulatory Guidance",
                    "source": "European Commission",
                    "relevance": "Detailed requirements for technical documentation"
                },
                {
                    "title": "NIST AI Risk Management Framework",
                    "type": "Framework",
                    "source": "National Institute of Standards and Technology (US)",
                    "relevance": "Comprehensive approach to AI risk management"
                },
                {
                    "title": "ISO/IEC 42001 - Artificial Intelligence Management System",
                    "type": "Standard",
                    "source": "International Organization for Standardization",
                    "relevance": "Management system requirements for AI"
                },
                {
                    "title": "Algorithm Impact Assessment Template",
                    "type": "Template",
                    "source": "Canadian Government",
                    "relevance": "Structure for assessing algorithmic impacts"
                },
                {
                    "title": "Responsible AI Toolkit",
                    "type": "Toolkit",
                    "source": "World Economic Forum",
                    "relevance": "Practical tools for responsible AI implementation"
                }
            ]
            
        # Data privacy compliance
        elif "data privacy" in topic.lower() or "gdpr" in topic.lower() or "ccpa" in topic.lower():
            guidelines["overview"] = "Data privacy compliance requires a comprehensive program addressing data governance, individual rights, security measures, and documentation. Organizations must implement appropriate technical and organizational measures to ensure lawful processing of personal data and demonstrate accountability."
            
            guidelines["key_compliance_steps"] = [
                {
                    "step": "Data Mapping and Inventory",
                    "description": "Create and maintain a comprehensive inventory of personal data processing activities, including data types, purposes, legal bases, retention periods, and data flows.",
                    "priority": "High"
                },
                {
                    "step": "Privacy Notice Implementation",
                    "description": "Develop clear, concise privacy notices that inform individuals about data processing activities, their rights, and how to exercise them.",
                    "priority": "High"
                },
                {
                    "step": "Consent Management",
                    "description": "Implement mechanisms to obtain, record, and manage valid consent where required, ensuring it is freely given, specific, informed, and unambiguous.",
                    "priority": "High"
                },
                {
                    "step": "Data Subject Rights Procedures",
                    "description": "Establish processes to handle individual rights requests (access, deletion, correction, portability, etc.) within required timeframes.",
                    "priority": "High"
                },
                {
                    "step": "Data Protection Impact Assessments",
                    "description": "Conduct DPIAs for high-risk processing activities to identify and mitigate privacy risks before processing begins.",
                    "priority": "Medium"
                },
                {
                    "step": "Vendor Management Program",
                    "description": "Implement due diligence procedures for data processors/service providers and ensure appropriate contractual terms are in place.",
                    "priority": "Medium"
                },
                {
                    "step": "Data Breach Response Plan",
                    "description": "Develop and test procedures for detecting, investigating, containing, and reporting data breaches within required timeframes.",
                    "priority": "High"
                },
                {
                    "step": "International Data Transfer Mechanisms",
                    "description": "Implement appropriate safeguards for cross-border data transfers, such as standard contractual clauses, binding corporate rules, or adequacy decisions.",
                    "priority": "Medium"
                }
            ]
            
            guidelines["documentation_requirements"] = [
                {
                    "document": "Record of Processing Activities",
                    "contents": [
                        "Purposes of processing",
                        "Categories of data subjects and personal data",
                        "Recipients of personal data",
                        "Data retention periods",
                        "Security measures",
                        "International transfers"
                    ],
                    "audience": "Data protection authorities, internal compliance team"
                },
                {
                    "document": "Data Protection Impact Assessments",
                    "contents": [
                        "Description of processing operations",
                        "Assessment of necessity and proportionality",
                        "Risk assessment for data subjects",
                        "Mitigation measures",
                        "DPO consultation results"
                    ],
                    "audience": "Privacy team, project managers, DPAs when required"
                },
                {
                    "document": "Legitimate Interest Assessments",
                    "contents": [
                        "Purpose test (why the interest is legitimate)",
                        "Necessity test (why processing is necessary)",
                        "Balancing test (why the interest is not overridden by individual interests)",
                        "Safeguards implemented",
                        "Periodic review schedule"
                    ],
                    "audience": "Internal compliance team, DPAs upon request"
                },
                {
                    "document": "Consent Records",
                    "contents": [
                        "Consent language presented",
                        "Method of obtaining consent",
                        "Timestamp and version of privacy notice",
                        "Withdrawal mechanism",
                        "Consent refresh schedule if applicable"
                    ],
                    "audience": "Internal compliance team, DPAs during investigations"
                },
                {
                    "document": "Data Breach Documentation",
                    "contents": [
                        "Nature and scope of the breach",
                        "Categories and approximate number of affected individuals",
                        "Categories and approximate volume of affected records",
                        "Consequences and mitigating actions",
                        "Notifications made to authorities and individuals"
                    ],
                    "audience": "Data protection authorities, internal investigation team"
                }
            ]
            
            guidelines["governance_framework"] = {
                "roles_and_responsibilities": [
                    {
                        "role": "Data Protection Officer (or Privacy Officer)",
                        "responsibilities": [
                            "Monitor compliance with data protection regulations",
                            "Advise on data protection impact assessments",
                            "Cooperate with supervisory authorities",
                            "Act as contact point for data subjects",
                            "Provide training and awareness"
                        ]
                    },
                    {
                        "role": "Privacy Champions/Leads",
                        "responsibilities": [
                            "Implement privacy requirements within business units",
                            "Conduct initial privacy assessments",
                            "Escalate privacy concerns to DPO",
                            "Assist with data mapping and inventory",
                            "Promote privacy awareness in their teams"
                        ]
                    },
                    {
                        "role": "IT and Security Teams",
                        "responsibilities": [
                            "Implement technical security measures",
                            "Manage access controls and authentication",
                            "Monitor for and respond to security incidents",
                            "Implement privacy by design in systems",
                            "Support data subject rights fulfillment"
                        ]
                    },
                    {
                        "role": "Legal and Compliance",
                        "responsibilities": [
                            "Draft and review privacy notices and policies",
                            "Negotiate data processing agreements",
                            "Advise on legal bases for processing",
                            "Monitor regulatory developments",
                            "Manage regulatory interactions"
                        ]
                    }
                ],
                "policies_and_procedures": [
                    "Data Protection Policy",
                    "Data Retention and Deletion Policy",
                    "Data Subject Rights Procedure",
                    "Data Breach Response Procedure",
                    "Data Protection Impact Assessment Procedure",
                    "International Data Transfer Procedure",
                    "Privacy by Design Guidelines"
                ],
                "training_requirements": [
                    "General data protection awareness for all employees",
                    "Role-specific training for teams handling sensitive data",
                    "Specialized training for privacy team members",
                    "Regular refresher training",
                    "New hire privacy orientation"
                ]
            }
            
            guidelines["implementation_timeline"] = [
                {
                    "phase": "Assessment and Planning",
                    "timeframe": "1-3 months",
                    "activities": [
                        "Conduct data mapping and gap analysis",
                        "Establish governance structure",
                        "Develop implementation roadmap",
                        "Allocate resources and budget"
                    ]
                },
                {
                    "phase": "Foundation Implementation",
                    "timeframe": "3-6 months",
                    "activities": [
                        "Develop key policies and procedures",
                        "Implement privacy notices",
                        "Establish data subject rights processes",
                        "Create record of processing activities"
                    ]
                },
                {
                    "phase": "Technical Measures",
                    "timeframe": "6-9 months",
                    "activities": [
                        "Implement consent management system",
                        "Deploy data security enhancements",
                        "Establish data retention controls",
                        "Implement privacy by design processes"
                    ]
                },
                {
                    "phase": "Operational Integration",
                    "timeframe": "9-12 months",
                    "activities": [
                        "Conduct training across organization",
                        "Integrate privacy into business processes",
                        "Implement vendor management program",
                        "Establish monitoring and audit procedures"
                    ]
                },
                {
                    "phase": "Continuous Compliance",
                    "timeframe": "Ongoing",
                    "activities": [
                        "Regular compliance reviews",
                        "Periodic policy updates",
                        "Ongoing training and awareness",
                        "Adaptation to regulatory changes"
                    ]
                }
            ]
            
            guidelines["resources"] = [
                {
                    "title": "European Data Protection Board Guidelines",
                    "type": "Regulatory Guidance",
                    "source": "EDPB",
                    "relevance": "Authoritative interpretation of GDPR requirements"
                },
                {
                    "title": "NIST Privacy Framework",
                    "type": "Framework",
                    "source": "National Institute of Standards and Technology (US)",
                    "relevance": "Structured approach to privacy risk management"
                },
                {
                    "title": "ISO/IEC 27701 - Privacy Information Management",
                    "type": "Standard",
                    "source": "International Organization for Standardization",
                    "relevance": "Extension to ISO 27001 for privacy management"
                },
                {
                    "title": "California Privacy Rights Act Regulations",
                    "type": "Regulatory Guidance",
                    "source": "California Privacy Protection Agency",
                    "relevance": "Implementation requirements for CCPA/CPRA"
                },
                {
                    "title": "ICO Accountability Framework",
                    "type": "Toolkit",
                    "source": "UK Information Commissioner's Office",
                    "relevance": "Practical tools for demonstrating accountability"
                }
            ]
            
        # Cryptocurrency compliance
        elif "crypto" in topic.lower() or "cryptocurrency" in topic.lower() or "bitcoin" in topic.lower() or "blockchain" in topic.lower():
            guidelines["overview"] = "Cryptocurrency compliance requires navigating a complex and evolving regulatory landscape across multiple jurisdictions. Organizations operating in this space must implement robust compliance programs addressing AML/CFT requirements, securities regulations, consumer protection, and operational risk management."
            
            guidelines["key_compliance_steps"] = [
                {
                    "step": "Regulatory Classification Analysis",
                    "description": "Determine which regulatory frameworks apply to your specific crypto activities (e.g., securities, commodities, payment services, etc.) across relevant jurisdictions.",
                    "priority": "High"
                },
                {
                    "step": "Registration and Licensing",
                    "description": "Obtain necessary registrations, licenses, or authorizations based on your activities and jurisdictions of operation.",
                    "priority": "High"
                },
                {
                    "step": "AML/CFT Program Implementation",
                    "description": "Develop and implement a risk-based anti-money laundering and counter-terrorist financing program, including KYC procedures, transaction monitoring, and suspicious activity reporting.",
                    "priority": "High"
                },
                {
                    "step": "Travel Rule Compliance",
                    "description": "Implement systems to comply with the FATF Travel Rule, requiring the collection and transmission of originator and beneficiary information for virtual asset transfers.",
                    "priority": "High"
                },
                {
                    "step": "Consumer Protection Measures",
                    "description": "Establish transparent fee structures, clear risk disclosures, secure custody solutions, and complaint handling procedures.",
                    "priority": "Medium"
                },
                {
                    "step": "Security and Operational Controls",
                    "description": "Implement robust security measures for private keys, smart contracts, and platform infrastructure, with regular security audits and penetration testing.",
                    "priority": "High"
                },
                {
                    "step": "Tax Compliance Framework",
                    "description": "Establish systems for tracking, reporting, and withholding taxes related to cryptocurrency transactions, including information reporting to tax authorities.",
                    "priority": "Medium"
                },
                {
                    "step": "Sanctions Screening Program",
                    "description": "Implement comprehensive screening against global sanctions lists, including blockchain analytics to identify high-risk wallets and transactions.",
                    "priority": "High"
                }
            ]
            
            guidelines["documentation_requirements"] = [
                {
                    "document": "Regulatory Classification Analysis",
                    "contents": [
                        "Analysis of business activities against regulatory frameworks",
                        "Jurisdictional scope assessment",
                        "Legal opinions supporting classifications",
                        "Regulatory engagement records",
                        "Periodic reassessment schedule"
                    ],
                    "audience": "Legal team, compliance officers, regulators during examinations"
                },
                {
                    "document": "AML/CFT Program Documentation",
                    "contents": [
                        "Risk assessment methodology and results",
                        "Customer due diligence procedures",
                        "Transaction monitoring rules and thresholds",
                        "Suspicious activity reporting process",
                        "Training materials and completion records"
                    ],
                    "audience": "Compliance team, financial intelligence units, regulatory examiners"
                },
                {
                    "document": "Security and Custody Framework",
                    "contents": [
                        "Key management procedures",
                        "Cold/hot wallet architecture",
                        "Access control policies",
                        "Incident response procedures",
                        "Audit and penetration testing results"
                    ],
                    "audience": "Security team, auditors, insurance providers"
                },
                {
                    "document": "Customer Disclosures",
                    "contents": [
                        "Risk disclosures",
                        "Fee schedules",
                        "Terms of service",
                        "Privacy policy",
                        "Complaint handling procedures"
                    ],
                    "audience": "Customers, consumer protection authorities, legal team"
                },
                {
                    "document": "Transaction and Trading Records",
                    "contents": [
                        "Complete transaction history",
                        "Customer identification data",
                        "Blockchain addresses and transactions",
                        "Travel Rule information exchanges",
                        "Suspicious activity investigations"
                    ],
                    "audience": "Compliance team, law enforcement (with appropriate legal process)"
                }
            ]
            
            guidelines["governance_framework"] = {
                "roles_and_responsibilities": [
                    {
                        "role": "Chief Compliance Officer",
                        "responsibilities": [
                            "Overall responsibility for compliance program",
                            "Regulatory engagement and reporting",
                            "Compliance risk assessment and mitigation",
                            "Policy development and implementation",
                            "Board and executive reporting"
                        ]
                    },
                    {
                        "role": "AML Officer",
                        "responsibilities": [
                            "Design and oversight of AML/CFT program",
                            "Suspicious activity monitoring and reporting",
                            "KYC/CDD program management",
                            "AML training and awareness",
                            "Regulatory examinations for AML"
                        ]
                    },
                    {
                        "role": "Security Officer",
                        "responsibilities": [
                            "Custody solution security",
                            "Infrastructure and application security",
                            "Security incident response",
                            "Penetration testing and vulnerability management",
                            "Security awareness training"
                        ]
                    },
                    {
                        "role": "Legal Counsel",
                        "responsibilities": [
                            "Regulatory classification analysis",
                            "Licensing and registration applications",
                            "Customer terms and disclosures",
                            "Regulatory change management",
                            "Legal proceedings and enforcement actions"
                        ]
                    }
                ],
                "policies_and_procedures": [
                    "Regulatory Compliance Policy",
                    "AML/CFT Policy and Procedures",
                    "Customer Acceptance Policy",
                    "Transaction Monitoring Procedures",
                    "Travel Rule Compliance Procedure",
                    "Custody and Key Management Policy",
                    "Incident Response Procedure",
                    "Regulatory Examination Procedure"
                ],
                "training_requirements": [
                    "Crypto-specific AML/CFT training",
                    "Security awareness for all staff",
                    "Advanced security training for technical staff",
                    "Regulatory requirements by jurisdiction",
                    "Blockchain analytics and transaction monitoring"
                ]
            }
            
            guidelines["implementation_timeline"] = [
                {
                    "phase": "Regulatory Assessment",
                    "timeframe": "1-3 months",
                    "activities": [
                        "Determine applicable regulations by jurisdiction",
                        "Identify licensing requirements",
                        "Conduct gap analysis against requirements",
                        "Develop compliance roadmap"
                    ]
                },
                {
                    "phase": "Foundation Building",
                    "timeframe": "3-6 months",
                    "activities": [
                        "Establish compliance team and governance",
                        "Develop core policies and procedures",
                        "Implement basic KYC and AML controls",
                        "Begin licensing application processes"
                    ]
                },
                {
                    "phase": "Technical Implementation",
                    "timeframe": "6-9 months",
                    "activities": [
                        "Deploy transaction monitoring systems",
                        "Implement Travel Rule solution",
                        "Enhance security infrastructure",
                        "Integrate blockchain analytics tools"
                    ]
                },
                {
                    "phase": "Operational Integration",
                    "timeframe": "9-12 months",
                    "activities": [
                        "Train staff on compliance procedures",
                        "Conduct testing and validation",
                        "Implement reporting mechanisms",
                        "Establish audit and review processes"
                    ]
                },
                {
                    "phase": "Continuous Compliance",
                    "timeframe": "Ongoing",
                    "activities": [
                        "Monitor regulatory developments",
                        "Conduct periodic risk assessments",
                        "Update policies and procedures",
                        "Perform regular compliance testing"
                    ]
                }
            ]
            
            guidelines["resources"] = [
                {
                    "title": "FATF Guidance for a Risk-Based Approach to Virtual Assets",
                    "type": "Regulatory Guidance",
                    "source": "Financial Action Task Force",
                    "relevance": "International standards for virtual asset regulation"
                },
                {
                    "title": "FinCEN Virtual Currency Guidance",
                    "type": "Regulatory Guidance",
                    "source": "Financial Crimes Enforcement Network (US)",
                    "relevance": "US requirements for money services businesses"
                },
                {
                    "title": "European Banking Authority Guidelines on AML/CFT",
                    "type": "Regulatory Guidance",
                    "source": "EBA",
                    "relevance": "Risk factors and due diligence for virtual assets"
                },
                {
                    "title": "Crypto Asset Compliance Standards",
                    "type": "Industry Standard",
                    "source": "Global Digital Finance",
                    "relevance": "Industry-developed best practices"
                },
                {
                    "title": "Travel Rule Information Sharing Alliance",
                    "type": "Industry Initiative",
                    "source": "TRISA",
                    "relevance": "Technical standards for Travel Rule compliance"
                }
            ]
            
        # Generic compliance guidelines for other topics
        else:
            guidelines["overview"] = f"Compliance with {topic} regulations requires a structured approach addressing risk assessment, policy development, implementation, and ongoing monitoring. Organizations should establish a comprehensive compliance program tailored to their specific activities and risk profile."
            
            guidelines["key_compliance_steps"] = [
                {
                    "step": "Regulatory Assessment",
                    "description": f"Identify applicable {topic} regulations across relevant jurisdictions and determine specific requirements for your organization.",
                    "priority": "High"
                },
                {
                    "step": "Gap Analysis",
                    "description": "Assess current practices against regulatory requirements to identify compliance gaps.",
                    "priority": "High"
                },
                {
                    "step": "Policy Development",
                    "description": f"Create comprehensive policies and procedures addressing {topic} compliance requirements.",
                    "priority": "High"
                },
                {
                    "step": "Implementation Planning",
                    "description": "Develop a detailed implementation plan with timelines, responsibilities, and resource allocation.",
                    "priority": "Medium"
                },
                {
                    "step": "Training and Awareness",
                    "description": "Provide role-specific training to ensure staff understand compliance requirements and procedures.",
                    "priority": "Medium"
                },
                {
                    "step": "Documentation Framework",
                    "description": "Establish systems for creating and maintaining required compliance documentation.",
                    "priority": "Medium"
                },
                {
                    "step": "Monitoring and Testing",
                    "description": "Implement ongoing monitoring and periodic testing of compliance controls.",
                    "priority": "Medium"
                },
                {
                    "step": "Review and Improvement",
                    "description": "Regularly review and update the compliance program based on regulatory changes and effectiveness assessment.",
                    "priority": "Medium"
                }
            ]
            
            guidelines["documentation_requirements"] = [
                {
                    "document": f"{topic} Compliance Policy",
                    "contents": [
                        "Regulatory scope and applicability",
                        "Key compliance requirements",
                        "Roles and responsibilities",
                        "Implementation approach",
                        "Monitoring and reporting"
                    ],
                    "audience": "All employees, regulators upon request"
                },
                {
                    "document": "Risk Assessment",
                    "contents": [
                        "Identified compliance risks",
                        "Risk evaluation methodology",
                        "Risk ratings and prioritization",
                        "Mitigation measures",
                        "Residual risk acceptance"
                    ],
                    "audience": "Compliance team, senior management, auditors"
                },
                {
                    "document": "Compliance Procedures",
                    "contents": [
                        "Detailed step-by-step procedures",
                        "Decision-making criteria",
                        "Required forms and templates",
                        "Escalation paths",
                        "Quality control measures"
                    ],
                    "audience": "Staff responsible for implementation"
                },
                {
                    "document": "Training Materials",
                    "contents": [
                        "Regulatory overview",
                        "Specific compliance requirements",
                        "Practical application examples",
                        "Common pitfalls and how to avoid them",
                        "Resources and support information"
                    ],
                    "audience": "Employees based on role and responsibilities"
                },
                {
                    "document": "Compliance Reports",
                    "contents": [
                        "Compliance status summary",
                        "Key metrics and indicators",
                        "Identified issues and remediation",
                        "Regulatory developments",
                        "Recommendations for improvement"
                    ],
                    "audience": "Senior management, board of directors, regulators"
                }
            ]
            
            guidelines["governance_framework"] = {
                "roles_and_responsibilities": [
                    {
                        "role": f"{topic} Compliance Officer",
                        "responsibilities": [
                            "Overall responsibility for compliance program",
                            "Policy development and implementation",
                            "Monitoring and reporting",
                            "Regulatory engagement",
                            "Training and awareness"
                        ]
                    },
                    {
                        "role": "Senior Management",
                        "responsibilities": [
                            "Setting compliance tone and culture",
                            "Approving compliance policies",
                            "Allocating adequate resources",
                            "Reviewing compliance reports",
                            "Ensuring accountability"
                        ]
                    },
                    {
                        "role": "Business Unit Leaders",
                        "responsibilities": [
                            "Implementing compliance requirements",
                            "Ensuring staff awareness and adherence",
                            "Reporting compliance issues",
                            "Participating in risk assessments",
                            "Supporting compliance initiatives"
                        ]
                    },
                    {
                        "role": "Internal Audit",
                        "responsibilities": [
                            "Independent assessment of compliance",
                            "Testing control effectiveness",
                            "Identifying improvement opportunities",
                            "Validating remediation efforts",
                            "Reporting to audit committee"
                        ]
                    }
                ],
                "policies_and_procedures": [
                    f"{topic} Compliance Policy",
                    "Risk Assessment Procedure",
                    "Compliance Monitoring Procedure",
                    "Issue Management and Remediation Procedure",
                    "Regulatory Change Management Procedure",
                    "Compliance Reporting Procedure"
                ],
                "training_requirements": [
                    f"General {topic} awareness for all employees",
                    "Role-specific training for affected functions",
                    "Advanced training for compliance team",
                    "New hire orientation",
                    "Annual refresher training"
                ]
            }
            
            guidelines["implementation_timeline"] = [
                {
                    "phase": "Assessment and Planning",
                    "timeframe": "1-3 months",
                    "activities": [
                        "Regulatory assessment",
                        "Gap analysis",
                        "Resource planning",
                        "Implementation roadmap development"
                    ]
                },
                {
                    "phase": "Policy Development",
                    "timeframe": "2-4 months",
                    "activities": [
                        "Draft policies and procedures",
                        "Stakeholder review and feedback",
                        "Approval process",
                        "Communication planning"
                    ]
                },
                {
                    "phase": "Implementation",
                    "timeframe": "3-6 months",
                    "activities": [
                        "Process and system changes",
                        "Training delivery",
                        "Documentation development",
                        "Control implementation"
                    ]
                },
                {
                    "phase": "Testing and Validation",
                    "timeframe": "2-3 months",
                    "activities": [
                        "Control testing",
                        "Process validation",
                        "Gap remediation",
                        "Compliance certification"
                    ]
                },
                {
                    "phase": "Ongoing Compliance",
                    "timeframe": "Continuous",
                    "activities": [
                        "Monitoring and testing",
                        "Regulatory change management",
                        "Periodic reporting",
                        "Continuous improvement"
                    ]
                }
            ]
            
            guidelines["resources"] = [
                {
                    "title": f"{topic} Regulatory Guidelines",
                    "type": "Regulatory Guidance",
                    "source": "Relevant Regulatory Authority",
                    "relevance": "Official guidance on compliance requirements"
                },
                {
                    "title": f"{topic} Compliance Framework",
                    "type": "Framework",
                    "source": "Industry Association",
                    "relevance": "Industry best practices and standards"
                },
                {
                    "title": f"{topic} Compliance Handbook",
                    "type": "Reference",
                    "source": "Professional Organization",
                    "relevance": "Comprehensive guidance on implementation"
                },
                {
                    "title": f"{topic} Risk Assessment Template",
                    "type": "Template",
                    "source": "Consulting Firm",
                    "relevance": "Structured approach to risk assessment"
                },
                {
                    "title": f"{topic} Compliance Training",
                    "type": "Training",
                    "source": "Training Provider",
                    "relevance": "Role-specific compliance training"
                }
            ]
        
        return guidelines
    
    # ===== BUSINESS AND COMPETITIVE ANALYSIS HANDLERS =====
    
    def _handle_business_analysis(self, task: Task) -> None:
        """Handle business and competitive analysis tasks."""
        try:
            # Extract company or industry from task description
            subject_match = re.search(r'(?:analysis|analyze) (?:of|for) ([^\'\"]+)', task.description, re.IGNORECASE)
            if not subject_match:
                subject_match = re.search(r'([^\'\"]+)(?:\'s| business| competitive| market)', task.description, re.IGNORECASE)
            
            if not subject_match:
                task.fail("Could not extract company or industry from task description")
                return
                
            subject = subject_match.group(1).strip()
            task.update_progress(0.2)
            
            logger.info(f"[SmartTaskProcessor] Performing business analysis for: {subject}")
            
            # Determine analysis type
            analysis_type = "company"
            if "industry" in task.description.lower() or "market" in task.description.lower() or "sector" in task.description.lower():
                analysis_type = "industry"
            
            # Generate appropriate analysis
            if analysis_type == "company":
                analysis = self._generate_company_analysis(subject)
            else:
                analysis = self._generate_industry_analysis(subject)
                
            task.update_progress(0.9)
            
            task.complete({
                "status": "success",
                "subject": subject,
                "analysis_type": analysis_type,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"[SmartTaskProcessor] Completed {analysis_type} analysis for '{subject}'")
            
        except Exception as e:
            task.fail(f"Error performing business analysis: {str(e)}")
    
    def _generate_company_analysis(self, company: str) -> Dict[str, Any]:
        """Generate a comprehensive company analysis."""
        import random
        
        # Seed with company to get consistent results
        random.seed(hash(company) % 120000)
        
        analysis = {
            "company_name": company,
            "executive_summary": "",
            "business_model": {},
            "swot_analysis": {},
            "competitive_positioning": {},
            "financial_overview": {},
            "strategic_recommendations": []
        }
        
        # Netflix analysis
        if "netflix" in company.lower():
            analysis["executive_summary"] = "Netflix has transformed from a DVD rental service to the world's leading streaming entertainment platform. Despite increasing competition in the streaming space, Netflix maintains a strong position through its content investments, technology platform, and global expansion strategy. The company faces challenges from traditional media companies launching their own streaming services and must continue to innovate to maintain its market leadership."
            
            analysis["business_model"] = {
                "revenue_streams": [
                    {
                        "stream": "Subscription Fees",
                        "description": "Monthly recurring subscription fees from members across different tiers (Basic, Standard, Premium)",
                        "percentage": "~99%"
                    },
                    {
                        "stream": "Advertising Revenue",
                        "description": "Recently introduced ad-supported tier generating revenue from advertisers",
                        "percentage": "~1%"
                    }
                ],
                "cost_structure": [
                    {
                        "category": "Content Costs",
                        "description": "Production and licensing of original and third-party content",
                        "percentage": "~60-65%"
                    },
                    {
                        "category": "Technology & Development",
                        "description": "Platform development, streaming infrastructure, recommendation algorithms",
                        "percentage": "~10%"
                    },
                    {
                        "category": "Marketing",
                        "description": "Customer acquisition, brand building, content promotion",
                        "percentage": "~15%"
                    },
                    {
                        "category": "General & Administrative",
                        "description": "Corporate functions, operations, and overhead",
                        "percentage": "~5-10%"
                    }
                ],
                "key_metrics": [
                    {
                        "metric": "Global Subscribers",
                        "value": "~230 million",
                        "trend": "Growing, but at a slower rate in mature markets"
                    },
                    {
                        "metric": "Average Revenue Per User (ARPU)",
                        "value": "$11-16 (varies by region)",
                        "trend": "Increasing through price adjustments"
                    },
                    {
                        "metric": "Content Spending",
                        "value": "$17-18 billion annually",
                        "trend": "Stabilizing after years of rapid growth"
                    },
                    {
                        "metric": "Churn Rate",
                        "value": "~2-3% monthly",
                        "trend": "Relatively stable but facing pressure from competition"
                    }
                ],
                "value_proposition": "Netflix offers unlimited, ad-free viewing of a vast library of movies, TV shows, and original content across multiple devices for a monthly subscription fee. The service provides personalized recommendations, high-quality streaming, and the convenience of on-demand viewing without commitments."
            }
            
            analysis["swot_analysis"] = {
                "strengths": [
                    {
                        "factor": "Content Library and Original Programming",
                        "description": "Extensive library of original and licensed content across genres and languages"
                    },
                    {
                        "factor": "Global Reach",
                        "description": "Available in over 190 countries with localized content and interfaces"
                    },
                    {
                        "factor": "Technology Platform",
                        "description": "Advanced recommendation algorithms, streaming technology, and user experience"
                    },
                    {
                        "factor": "Brand Recognition",
                        "description": "Strong brand synonymous with streaming entertainment"
                    },
                    {
                        "factor": "Data-Driven Decision Making",
                        "description": "Extensive viewer data informing content and product decisions"
                    }
                ],
                "weaknesses": [
                    {
                        "factor": "Content Costs",
                        "description": "High and growing content production and licensing expenses"
                    },
                    {
                        "factor": "Mature Market Saturation",
                        "description": "Slowing growth in developed markets like North America"
                    },
                    {
                        "factor": "Dependence on Licensed Content",
                        "description": "Still reliant on some third-party content that could be withdrawn"
                    },
                    {
                        "factor": "Price Sensitivity",
                        "description": "Increasing subscription prices may accelerate churn"
                    },
                    {
                        "factor": "Account Sharing",
                        "description": "Revenue loss from password sharing across households"
                    }
                ],
                "opportunities": [
                    {
                        "factor": "International Growth",
                        "description": "Expansion potential in emerging markets, particularly Asia"
                    },
                    {
                        "factor": "Ad-Supported Tier",
                        "description": "New revenue stream and lower-priced entry point for price-sensitive consumers"
                    },
                    {
                        "factor": "Gaming Expansion",
                        "description": "Potential to leverage IP for gaming content and additional engagement"
                    },
                    {
                        "factor": "Merchandising and Licensing",
                        "description": "Monetization of popular original IP through merchandise and licensing"
                    },
                    {
                        "factor": "Password Sharing Monetization",
                        "description": "Converting shared accounts to paid memberships through new policies"
                    }
                ],
                "threats": [
                    {
                        "factor": "Intensifying Competition",
                        "description": "Disney+, HBO Max, Amazon Prime Video, and other streaming services competing for subscribers and content"
                    },
                    {
                        "factor": "Content Creator Consolidation",
                        "description": "Media companies reclaiming content for their own platforms"
                    },
                    {
                        "factor": "Rising Content Costs",
                        "description": "Inflation in production costs and talent compensation"
                    },
                    {
                        "factor": "Regulatory Challenges",
                        "description": "Potential content quotas, data privacy regulations, and tax implications in various markets"
                    },
                    {
                        "factor": "Economic Pressures",
                        "description": "Subscription services vulnerable during economic downturns as consumers reduce discretionary spending"
                    }
                ]
            }
            
            analysis["competitive_positioning"] = {
                "direct_competitors": [
                    {
                        "name": "Disney+",
                        "strengths": "Strong IP portfolio, family-friendly content, bundling with Hulu and ESPN+",
                        "weaknesses": "More limited content library, less adult-oriented content"
                    },
                    {
                        "name": "Amazon Prime Video",
                        "strengths": "Bundled with Prime membership, growing original content, global reach",
                        "weaknesses": "Less focused on video as core business, user interface challenges"
                    },
                    {
                        "name": "HBO Max/Discovery+",
                        "strengths": "Premium content quality, strong IP (Warner Bros, HBO, DC), combined library",
                        "weaknesses": "Higher price point, less international presence"
                    },
                    {
                        "name": "Hulu",
                        "strengths": "Current TV content, Disney backing, ad-supported model experience",
                        "weaknesses": "Limited international availability, complex ownership structure"
                    },
                    {
                        "name": "Apple TV+",
                        "strengths": "High-quality original content, device integration, financial resources",
                        "weaknesses": "Smaller content library, less established in entertainment"
                    }
                ],
                "indirect_competitors": [
                    {
                        "name": "YouTube",
                        "threat_level": "Medium",
                        "description": "Competes for viewing time with free content and YouTube Premium"
                    },
                    {
                        "name": "TikTok",
                        "threat_level": "Medium",
                        "description": "Competes for entertainment time with short-form video content"
                    },
                    {
                        "name": "Gaming Platforms",
                        "threat_level": "Medium",
                        "description": "Compete for entertainment time and subscription dollars"
                    },
                    {
                        "name": "Traditional TV",
                        "threat_level": "Low",
                        "description": "Declining but still significant for certain demographics and content types"
                    }
                ],
                "market_position": {
                    "global_market_share": "~30-35% of streaming viewing time",
                    "regional_variations": "Stronger in North America and Europe, growing in Asia and Latin America",
                    "subscriber_comparison": "Largest pure-play streaming service by subscribers",
                    "content_investment_comparison": "Among the highest content spenders globally"
                },
                "differentiation_factors": [
                    "Scale and breadth of content library",
                    "Technology platform and user experience",
                    "Global production capabilities across multiple languages",
                    "Data-driven content development",
                    "First-mover advantage and brand recognition"
                ]
            }
            
            analysis["financial_overview"] = {
                "key_financials": {
                    "revenue": "~$31-33 billion annually",
                    "revenue_growth": "~6-9% year-over-year",
                    "operating_margin": "~20-22%",
                    "net_income": "~$5-6 billion annually",
                    "free_cash_flow": "Positive and growing after years of negative FCF",
                    "debt_level": "~$14-15 billion"
                },
                "revenue_trends": {
                    "subscription_growth": "Slowing in mature markets, stronger internationally",
                    "arpu_trends": "Increasing through price increases and tier management",
                    "regional_performance": "North America mature, EMEA and APAC growing",
                    "content_efficiency": "Focusing on return on content investment"
                },
                "investment_highlights": [
                    "Transition to positive free cash flow",
                    "Operating margin expansion",
                    "Potential for continued international growth",
                    "New revenue streams (advertising, gaming)",
                    "Strong position in growing streaming market"
                ],
                "financial_risks": [
                    "Content cost inflation",
                    "Competitive pressure on subscriber growth",
                    "Foreign exchange exposure",
                    "Potential margin pressure from new initiatives"
                ]
            }
            
            analysis["strategic_recommendations"] = [
                {
                    "recommendation": "Optimize Content Investment Strategy",
                    "description": "Focus content spending on high-ROI productions and markets, potentially reducing overall content budget while maintaining quality and variety.",
                    "rationale": "Improving efficiency of content spending will support margin expansion and free cash flow generation while maintaining competitive offering.",
                    "implementation_complexity": "Medium",
                    "potential_impact": "High"
                },
                {
                    "recommendation": "Expand Ad-Supported Tier",
                    "description": "Accelerate development of advertising business through improved ad technology, expanded inventory, and potential partnerships with major advertisers.",
                    "rationale": "Diversifies revenue streams, captures price-sensitive segments, and increases total addressable market.",
                    "implementation_complexity": "Medium",
                    "potential_impact": "Medium"
                },
                {
                    "recommendation": "Targeted International Expansion",
                    "description": "Increase investment in high-potential markets like India, Southeast Asia, and Middle East with localized content and pricing strategies.",
                    "rationale": "These markets represent significant growth opportunities with rising middle class and increasing digital adoption.",
                    "implementation_complexity": "High",
                    "potential_impact": "High"
                },
                {
                    "recommendation": "Expand Gaming Initiative",
                    "description": "Accelerate development of gaming content, potentially through strategic acquisitions of game studios and leveraging Netflix IP.",
                    "rationale": "Gaming represents a complementary entertainment category that can increase engagement and reduce churn.",
                    "implementation_complexity": "High",
                    "potential_impact": "Medium"
                },
                {
                    "recommendation": "Implement Account Sharing Monetization",
                    "description": "Roll out paid sharing options globally with careful messaging and reasonable pricing to convert shared users to paying customers.",
                    "rationale": "Captures revenue from currently non-paying users while maintaining user experience and goodwill.",
                    "implementation_complexity": "Medium",
                    "potential_impact": "High"
                }
            ]
            
        # Tesla analysis
        elif "tesla" in company.lower():
            analysis["executive_summary"] = "Tesla has established itself as the leading electric vehicle manufacturer globally, with a mission extending beyond cars to sustainable energy more broadly. The company has disrupted the automotive industry through its innovative approach to electric vehicles, software integration, and direct-to-consumer sales model. While facing increasing competition and production challenges, Tesla maintains significant advantages in brand strength, technology, and vertical integration. The company's future growth depends on successfully scaling production, continuing technological innovation, and expanding into new markets and product categories."
            
            analysis["business_model"] = {
                "revenue_streams": [
                    {
                        "stream": "Automotive Sales",
                        "description": "Direct sales of electric vehicles (Model S, 3, X, Y, Cybertruck)",
                        "percentage": "~80-85%"
                    },
                    {
                        "stream": "Regulatory Credits",
                        "description": "Sale of regulatory credits to other automakers",
                        "percentage": "~2-5%"
                    },
                    {
                        "stream": "Energy Generation and Storage",
                        "description": "Solar panels, Solar Roof, and battery storage products",
                        "percentage": "~5-7%"
                    },
                    {
                        "stream": "Services and Other",
                        "description": "Vehicle servicing, merchandise, used vehicle sales, insurance",
                        "percentage": "~5-10%"
                    }
                ],
                "cost_structure": [
                    {
                        "category": "Cost of Goods Sold",
                        "description": "Manufacturing costs, materials, labor for vehicles and energy products",
                        "percentage": "~75-80%"
                    },
                    {
                        "category": "Research & Development",
                        "description": "Product development, engineering, AI and software development",
                        "percentage": "~5-7%"
                    },
                    {
                        "category": "Selling, General & Administrative",
                        "description": "Sales, marketing, corporate functions, retail locations",
                        "percentage": "~10-12%"
                    },
                    {
                        "category": "Other Operating Expenses",
                        "description": "Includes restructuring costs and various operational expenses",
                        "percentage": "~2-3%"
                    }
                ],
                "key_metrics": [
                    {
                        "metric": "Vehicle Deliveries",
                        "value": "~1.8-2 million annually",
                        "trend": "Growing year-over-year but with quarterly fluctuations"
                    },
                    {
                        "metric": "Automotive Gross Margin",
                        "value": "~25-30%",
                        "trend": "Pressure from price reductions and competition"
                    },
                    {
                        "metric": "Production Capacity",
                        "value": "~2-2.5 million vehicles annually",
                        "trend": "Expanding with new factories and production lines"
                    },
                    {
                        "metric": "Free Cash Flow",
                        "value": "$5-8 billion annually",
                        "trend": "Strong but variable based on capital expenditures"
                    }
                ],
                "value_proposition": "Tesla offers premium electric vehicles with industry-leading range, performance, and technology, supported by a proprietary Supercharger network and over-the-air software updates. The company's integrated ecosystem extends to energy products that enable customers to generate, store, and consume sustainable energy."
            }
            
            analysis["swot_analysis"] = {
                "strengths": [
                    {
                        "factor": "Brand Strength and Loyalty",
                        "description": "Strong brand recognition and customer loyalty with minimal traditional advertising"
                    },
                    {
                        "factor": "Technology Leadership",
                        "description": "Advanced battery technology, software capabilities, and autonomous driving development"
                    },
                    {
                        "factor": "Vertical Integration",
                        "description": "Control over key components including batteries, motors, software, and charging infrastructure"
                    },
                    {
                        "factor": "Manufacturing Scale",
                        "description": "Growing global production capacity with Gigafactories in multiple continents"
                    },
                    {
                        "factor": "Data Advantage",
                        "description": "Vast amount of real-world driving data for improving autonomous capabilities"
                    }
                ],
                "weaknesses": [
                    {
                        "factor": "Production Challenges",
                        "description": "History of production delays and quality control issues"
                    },
                    {
                        "factor": "Leadership Concentration",
                        "description": "Heavy reliance on Elon Musk for vision, strategy, and public perception"
                    },
                    {
                        "factor": "Service Network Limitations",
                        "description": "Service capacity not keeping pace with vehicle sales growth"
                    },
                    {
                        "factor": "Price Positioning",
                        "description": "Higher price points limiting market penetration in mass market segments"
                    },
                    {
                        "factor": "Regulatory Scrutiny",
                        "description": "Increasing scrutiny of autonomous driving claims and features"
                    }
                ],
                "opportunities": [
                    {
                        "factor": "Global EV Market Growth",
                        "description": "Rapidly expanding electric vehicle adoption worldwide"
                    },
                    {
                        "factor": "Energy Business Expansion",
                        "description": "Growing market for residential and utility-scale energy storage"
                    },
                    {
                        "factor": "Full Self-Driving Commercialization",
                        "description": "Potential for high-margin software revenue from autonomous capabilities"
                    },
                    {
                        "factor": "New Vehicle Categories",
                        "description": "Expansion into new segments like Semi, Cybertruck, and potentially lower-cost vehicles"
                    },
                    {
                        "factor": "AI and Robotics",
                        "description": "Leveraging AI expertise for robotics (Optimus) and other applications"
                    }
                ],
                "threats": [
                    {
                        "factor": "Increasing Competition",
                        "description": "Traditional automakers and new entrants investing heavily in EVs"
                    },
                    {
                        "factor": "Battery Supply Chain Constraints",
                        "description": "Limited availability of battery materials and processing capacity"
                    },
                    {
                        "factor": "Regulatory Changes",
                        "description": "Evolving regulations around EVs, autonomous driving, and direct sales"
                    },
                    {
                        "factor": "Macroeconomic Factors",
                        "description": "Interest rates, inflation, and economic cycles affecting premium vehicle demand"
                    },
                    {
                        "factor": "Technological Disruption",
                        "description": "Potential for alternative technologies (e.g., hydrogen) or unexpected innovations"
                    }
                ]
            }
            
            analysis["competitive_positioning"] = {
                "direct_competitors": [
                    {
                        "name": "BYD",
                        "strengths": "Vertical integration, battery production, price competitiveness, China market position",
                        "weaknesses": "Limited global presence, brand perception outside China"
                    },
                    {
                        "name": "Volkswagen Group",
                        "strengths": "Manufacturing scale, broad portfolio, global presence",
                        "weaknesses": "Software capabilities, EV-specific architecture transition"
                    },
                    {
                        "name": "Ford",
                        "strengths": "Strong in profitable segments (trucks), brand loyalty, dealer network",
                        "weaknesses": "Battery supply, software development, financial constraints"
                    },
                    {
                        "name": "General Motors",
                        "strengths": "Ultium platform, manufacturing scale, Cruise autonomous unit",
                        "weaknesses": "EV profitability, brand perception for EVs"
                    },
                    {
                        "name": "Rivian",
                        "strengths": "Product design, technology focus, Amazon relationship",
                        "weaknesses": "Production scale, path to profitability, limited model range"
                    }
                ],
                "indirect_competitors": [
                    {
                        "name": "Traditional Luxury Brands (Mercedes, BMW, Audi)",
                        "threat_level": "Medium",
                        "description": "Competing for premium vehicle customers with increasing EV offerings"
                    },
                    {
                        "name": "Chinese EV Startups (NIO, XPeng, Li Auto)",
                        "threat_level": "Medium",
                        "description": "Rapidly growing with technology focus and domestic market advantage"
                    },
                    {
                        "name": "Technology Companies (Apple, Waymo)",
                        "threat_level": "Medium-Low",
                        "description": "Potential to enter vehicle market or dominate autonomous technology"
                    }
                ],
                "market_position": {
                    "global_ev_market_share": "~15-20%",
                    "regional_variations": "Strong in North America and Europe, growing in China, limited in other markets",
                    "premium_segment_position": "Dominant in premium EV segment",
                    "technology_leadership": "Leading in battery efficiency, software, and charging infrastructure"
                },
                "differentiation_factors": [
                    "Integrated software and hardware development",
                    "Supercharger network exclusivity",
                    "Direct sales model without traditional dealerships",
                    "Brand association with innovation and sustainability",
                    "Over-the-air update capabilities enhancing vehicles post-purchase"
                ]
            }
            
            analysis["financial_overview"] = {
                "key_financials": {
                    "revenue": "~$90-100 billion annually",
                    "revenue_growth": "~20-30% year-over-year",
                    "operating_margin": "~15-18%",
                    "net_income": "~$12-15 billion annually",
                    "free_cash_flow": "~$5-8 billion annually",
                    "cash_position": "~$20-25 billion"
                },
                "revenue_trends": {
                    "automotive_growth": "Slowing but still strong year-over-year growth",
                    "asp_trends": "Declining due to mix shift to lower-priced models and price reductions",
                    "regulatory_credits": "Declining as percentage of revenue",
                    "energy_business": "Growing but still small portion of overall business"
                },
                "investment_highlights": [
                    "Industry-leading automotive gross margins",
                    "Strong free cash flow generation",
                    "Minimal debt relative to market capitalization",
                    "Significant growth potential in energy and new vehicle segments",
                    "Technology leadership potentially enabling new revenue streams"
                ],
                "financial_risks": [
                    "Margin pressure from increasing competition and price reductions",
                    "High capital expenditure requirements for expansion",
                    "Potential dilution from stock-based compensation",
                    "Cyclical nature of automotive industry",
                    "Regulatory credit revenue likely to decline over time"
                ]
            }
            
            analysis["strategic_recommendations"] = [
                {
                    "recommendation": "Accelerate Lower-Cost Vehicle Development",
                    "description": "Prioritize development and production of a sub-$30,000 vehicle platform to access mass market segments and emerging markets.",
                    "rationale": "Expanding addressable market is critical for continued growth as premium EV segments become more competitive.",
                    "implementation_complexity": "High",
                    "potential_impact": "High"
                },
                {
                    "recommendation": "Expand Service Network Capacity",
                    "description": "Significantly increase service center locations, mobile service fleet, and third-party certified repair options.",
                    "rationale": "Service constraints are limiting customer satisfaction and could impact brand perception as the vehicle fleet grows.",
                    "implementation_complexity": "Medium",
                    "potential_impact": "Medium"
                },
                {
                    "recommendation": "Scale Energy Business Aggressively",
                    "description": "Increase investment in Powerwall production, Solar Roof installation capacity, and utility-scale Megapack projects.",
                    "rationale": "Energy business offers diversification, recurring revenue potential, and aligns with mission while leveraging battery expertise.",
                    "implementation_complexity": "Medium",
                    "potential_impact": "Medium"
                },
                {
                    "recommendation": "Develop Clear FSD Commercialization Strategy",
                    "description": "Create transparent roadmap for Full Self-Driving capabilities, regulatory approval process, and monetization model.",
                    "rationale": "Clarifying the path to FSD commercialization would reduce investor uncertainty and potentially unlock significant value.",
                    "implementation_complexity": "Medium",
                    "potential_impact": "High"
                },
                {
                    "recommendation": "Strengthen Leadership Bench",
                    "description": "Recruit and develop high-profile executives to reduce reliance on Elon Musk and prepare for long-term succession.",
                    "rationale": "Reducing key person risk would improve corporate governance and potentially reduce stock volatility.",
                    "implementation_complexity": "Medium",
                    "potential_impact": "Medium"
                }
            ]
            
        # Generic company analysis for other companies
        else:
            analysis["executive_summary"] = f"{company} operates in a dynamic market environment with both significant opportunities and challenges. The company's business model, competitive positioning, and financial performance suggest several strategic priorities for maintaining and enhancing its market position. This analysis examines the company's current situation and provides recommendations for future growth and operational improvement."
            
            analysis["business_model"] = {
                "revenue_streams": [
                    {
                        "stream": "Primary Revenue Stream",
                        "description": "Description of main revenue source",
                        "percentage": "~X%"
                    },
                    {
                        "stream": "Secondary Revenue Stream",
                        "description": "Description of secondary revenue source",
                        "percentage": "~Y%"
                    },
                    {
                        "stream": "Tertiary Revenue Stream",
                        "description": "Description of additional revenue source",
                        "percentage": "~Z%"
                    }
                ],
                "cost_structure": [
                    {
                        "category": "Major Cost Category 1",
                        "description": "Description of major cost component",
                        "percentage": "~X%"
                    },
                    {
                        "category": "Major Cost Category 2",
                        "description": "Description of another major cost component",
                        "percentage": "~Y%"
                    },
                    {
                        "category": "Major Cost Category 3",
                        "description": "Description of third major cost component",
                        "percentage": "~Z%"
                    }
                ],
                "key_metrics": [
                    {
                        "metric": "Key Performance Indicator 1",
                        "value": "Current value",
                        "trend": "Description of trend"
                    },
                    {
                        "metric": "Key Performance Indicator 2",
                        "value": "Current value",
                        "trend": "Description of trend"
                    },
                    {
                        "metric": "Key Performance Indicator 3",
                        "value": "Current value",
                        "trend": "Description of trend"
                    }
                ],
                "value_proposition": f"{company}'s value proposition centers on providing [specific benefits] to [target customers] through [key differentiators]."
            }
            
            analysis["swot_analysis"] = {
                "strengths": [
                    {
                        "factor": "Strength 1",
                        "description": "Description of key organizational strength"
                    },
                    {
                        "factor": "Strength 2",
                        "description": "Description of second key strength"
                    },
                    {
                        "factor": "Strength 3",
                        "description": "Description of third key strength"
                    },
                    {
                        "factor": "Strength 4",
                        "description": "Description of fourth key strength"
                    }
                ],
                "weaknesses": [
                    {
                        "factor": "Weakness 1",
                        "description": "Description of key organizational weakness"
                    },
                    {
                        "factor": "Weakness 2",
                        "description": "Description of second key weakness"
                    },
                    {
                        "factor": "Weakness 3",
                        "description": "Description of third key weakness"
                    },
                    {
                        "factor": "Weakness 4",
                        "description": "Description of fourth key weakness"
                    }
                ],
                "opportunities": [
                    {
                        "factor": "Opportunity 1",
                        "description": "Description of key market opportunity"
                    },
                    {
                        "factor": "Opportunity 2",
                        "description": "Description of second key opportunity"
                    },
                    {
                        "factor": "Opportunity 3",
                        "description": "Description of third key opportunity"
                    },
                    {
                        "factor": "Opportunity 4",
                        "description": "Description of fourth key opportunity"
                    }
                ],
                "threats": [
                    {
                        "factor": "Threat 1",
                        "description": "Description of key external threat"
                    },
                    {
                        "factor": "Threat 2",
                        "description": "Description of second key threat"
                    },
                    {
                        "factor": "Threat 3",
                        "description": "Description of third key threat"
                    },
                    {
                        "factor": "Threat 4",
                        "description": "Description of fourth key threat"
                    }
                ]
            }
            
            analysis["competitive_positioning"] = {
                "direct_competitors": [
                    {
                        "name": "Competitor 1",
                        "strengths": "Key strengths of this competitor",
                        "weaknesses": "Key weaknesses of this competitor"
                    },
                    {
                        "name": "Competitor 2",
                        "strengths": "Key strengths of this competitor",
                        "weaknesses": "Key weaknesses of this competitor"
                    },
                    {
                        "name": "Competitor 3",
                        "strengths": "Key strengths of this competitor",
                        "weaknesses": "Key weaknesses of this competitor"
                    }
                ],
                "indirect_competitors": [
                    {
                        "name": "Indirect Competitor 1",
                        "threat_level": "High/Medium/Low",
                        "description": "How this entity competes indirectly"
                    },
                    {
                        "name": "Indirect Competitor 2",
                        "threat_level": "High/Medium/Low",
                        "description": "How this entity competes indirectly"
                    }
                ],
                "market_position": {
                    "market_share": "Approximate market share",
                    "regional_variations": "Differences in position by region",
                    "segment_position": "Position in key market segments",
                    "trend": "Whether position is improving, stable, or declining"
                },
                "differentiation_factors": [
                    "Key differentiator 1",
                    "Key differentiator 2",
                    "Key differentiator 3",
                    "Key differentiator 4"
                ]
            }
            
            analysis["financial_overview"] = {
                "key_financials": {
                    "revenue": "Annual revenue figure",
                    "revenue_growth": "Year-over-year growth rate",
                    "profit_margin": "Profit margin percentage",
                    "net_income": "Annual net income",
                    "cash_position": "Cash and equivalents",
                    "debt_level": "Total debt"
                },
                "revenue_trends": {
                    "trend_1": "Description of important revenue trend",
                    "trend_2": "Description of second revenue trend",
                    "trend_3": "Description of third revenue trend"
                },
                "investment_highlights": [
                    "Key financial strength 1",
                    "Key financial strength 2",
                    "Key financial strength 3",
                    "Key financial strength 4"
                ],
                "financial_risks": [
                    "Key financial risk 1",
                    "Key financial risk 2",
                    "Key financial risk 3",
                    "Key financial risk 4"
                ]
            }
            
            analysis["strategic_recommendations"] = [
                {
                    "recommendation": "Strategic Recommendation 1",
                    "description": "Detailed description of recommended action",
                    "rationale": "Why this recommendation makes strategic sense",
                    "implementation_complexity": "High/Medium/Low",
                    "potential_impact": "High/Medium/Low"
                },
                {
                    "recommendation": "Strategic Recommendation 2",
                    "description": "Detailed description of recommended action",
                    "rationale": "Why this recommendation makes strategic sense",
                    "implementation_complexity": "High/Medium/Low",
                    "potential_impact": "High/Medium/Low"
                },
                {
                    "recommendation": "Strategic Recommendation 3",
                    "description": "Detailed description of recommended action",
                    "rationale": "Why this recommendation makes strategic sense",
                    "implementation_complexity": "High/Medium/Low",
                    "potential_impact": "High/Medium/Low"
                },
                {
                    "recommendation": "Strategic Recommendation 4",
                    "description": "Detailed description of recommended action",
                    "rationale": "Why this recommendation makes strategic sense",
                    "implementation_complexity": "High/Medium/Low",
                    "potential_impact": "High/Medium/Low"
                }
            ]
        
        return analysis
    
    def _generate_industry_analysis(self, industry: str) -> Dict[str, Any]:
        """Generate a comprehensive industry analysis."""
        import random
        
        # Seed with industry to get consistent results
        random.seed(hash(industry) % 130000)
        
        analysis = {
            "industry_name": industry,
            "executive_summary": "",
            "market_overview": {},
            "competitive_landscape": {},
            "key_trends": [],
            "challenges_and_opportunities": {},
            "future_outlook": {}
        }
        
        # Electric vehicle industry analysis
        if "electric vehicle" in industry.lower() or "ev" in industry.lower():
            analysis["executive_summary"] = "The electric vehicle (EV) industry is experiencing rapid growth and transformation, driven by technological advancements, regulatory support, and changing consumer preferences. As the industry matures, competition is intensifying with traditional automakers, new entrants, and technology companies all vying for market share. While challenges remain in areas such as charging infrastructure, battery supply chains, and affordability, the long-term trajectory points toward EVs becoming the dominant form of personal transportation over the next two decades."
            
            analysis["market_overview"] = {
                "market_size": {
                    "global_sales": "~10-11 million units annually",
                    "market_value": "~$500-600 billion",
                    "growth_rate": "~35-40% year-over-year",
                    "penetration_rate": "~14-15% of global new vehicle sales"
                },
                "regional_analysis": [
                    {
                        "region": "China",
                        "market_share": "~60% of global EV sales",
                        "growth_rate": "~30-35% year-over-year",
                        "key_characteristics": "Strong government support, domestic manufacturers dominance, developed supply chain"
                    },
                    {
                        "region": "Europe",
                        "market_share": "~20-25% of global EV sales",
                        "growth_rate": "~15-20% year-over-year",
                        "key_characteristics": "Stringent emissions regulations, strong incentives, high adoption in Nordic countries"
                    },
                    {
                        "region": "North America",
                        "market_share": "~10-15% of global EV sales",
                        "growth_rate": "~40-45% year-over-year",
                        "key_characteristics": "Dominated by Tesla, growing policy support, expanding charging infrastructure"
                    },
                    {
                        "region": "Rest of World",
                        "market_share": "~5-10% of global EV sales",
                        "growth_rate": "~25-30% year-over-year",
                        "key_characteristics": "Varied adoption rates, emerging markets focusing on two-wheelers and commercial vehicles"
                    }
                ],
                "segment_analysis": [
                    {
                        "segment": "Battery Electric Vehicles (BEVs)",
                        "market_share": "~70-75% of EV sales",
                        "growth_trend": "Fastest growing segment",
                        "key_players": "Tesla, BYD, Volkswagen Group, SAIC"
                    },
                    {
                        "segment": "Plug-in Hybrid Electric Vehicles (PHEVs)",
                        "market_share": "~25-30% of EV sales",
                        "growth_trend": "Moderate growth, transitional technology",
                        "key_players": "BYD, BMW, Volvo, Toyota"
                    },
                    {
                        "segment": "Commercial EVs",
                        "market_share": "~5-10% of EV market value",
                        "growth_trend": "Accelerating growth, particularly in delivery vans and buses",
                        "key_players": "BYD, Rivian, Daimler, Volvo"
                    }
                ],
                "value_chain": {
                    "raw_materials": "Lithium, nickel, cobalt, copper, rare earth elements",
                    "battery_production": "Cell manufacturing, pack assembly, battery management systems",
                    "vehicle_manufacturing": "EV-specific platforms, powertrain integration, software development",
                    "charging_infrastructure": "Home charging, public networks, fast charging corridors",
                    "end_of_life": "Battery recycling, reuse applications, vehicle dismantling"
                }
            }
            
            analysis["competitive_landscape"] = {
                "market_concentration": "Moderately concentrated but rapidly evolving",
                "key_players": [
                    {
                        "name": "Tesla",
                        "market_position": "Global leader in premium BEVs",
                        "strengths": "Brand, technology, charging network, software integration",
                        "challenges": "Increasing competition, production scaling, price pressure"
                    },
                    {
                        "name": "BYD",
                        "market_position": "Leading Chinese manufacturer, growing globally",
                        "strengths": "Vertical integration, battery production, cost efficiency",
                        "challenges": "International expansion, brand recognition outside China"
                    },
                    {
                        "name": "Volkswagen Group",
                        "market_position": "Leading traditional OEM in EV transition",
                        "strengths": "Scale, manufacturing expertise, broad portfolio",
                        "challenges": "Software development, profitability, legacy costs"
                    },
                    {
                        "name": "SAIC-GM-Wuling",
                        "market_position": "Dominates affordable EV segment in China",
                        "strengths": "Low-cost models, market understanding, distribution",
                        "challenges": "Limited international presence, technology sophistication"
                    },
                    {
                        "name": "Hyundai-Kia",
                        "market_position": "Fast-growing global EV manufacturer",
                        "strengths": "E-GMP platform, design, value proposition",
                        "challenges": "Battery supply, charging infrastructure, US market position"
                    }
                ],
                "emerging_players": [
                    {
                        "name": "Rivian",
                        "focus": "Electric trucks and SUVs, commercial vans",
                        "potential_impact": "Disruption in profitable truck segment"
                    },
                    {
                        "name": "Lucid",
                        "focus": "Ultra-premium EVs with leading range and technology",
                        "potential_impact": "Setting new benchmarks for luxury EVs"
                    },
                    {
                        "name": "NIO",
                        "focus": "Premium EVs with battery swap technology",
                        "potential_impact": "Alternative charging model, international expansion"
                    },
                    {
                        "name": "Vinfast",
                        "focus": "Rapid global expansion from Vietnam base",
                        "potential_impact": "New manufacturing hub and export model"
                    }
                ],
                "competitive_factors": [
                    {
                        "factor": "Battery Technology",
                        "importance": "Critical",
                        "trend": "Rapid innovation in chemistry, form factors, and manufacturing"
                    },
                    {
                        "factor": "Software and Connectivity",
                        "importance": "High",
                        "trend": "Increasing differentiation through software features and OTA updates"
                    },
                    {
                        "factor": "Manufacturing Efficiency",
                        "importance": "High",
                        "trend": "Focus on dedicated EV platforms and gigacasting techniques"
                    },
                    {
                        "factor": "Charging Network",
                        "importance": "Medium-High",
                        "trend": "Expansion of networks and opening of previously proprietary systems"
                    },
                    {
                        "factor": "Autonomous Capabilities",
                        "importance": "Medium",
                        "trend": "Gradual advancement with regulatory constraints"
                    }
                ],
                "barriers_to_entry": [
                    "Capital intensity for manufacturing",
                    "Battery supply chain access",
                    "Software development capabilities",
                    "Regulatory compliance complexity",
                    "Established brand competition"
                ]
            }
            
            analysis["key_trends"] = [
                {
                    "trend": "Battery Technology Evolution",
                    "description": "Rapid advancement in battery chemistry, energy density, and manufacturing processes, with a focus on reducing cobalt content, increasing nickel, and exploring solid-state technologies.",
                    "impact": "Longer ranges, faster charging, lower costs, and improved safety, accelerating EV adoption across segments.",
                    "timeline": "Ongoing with significant improvements every 2-3 years; solid-state commercialization expected 2025-2030."
                },
                {
                    "trend": "Vertical Integration",
                    "description": "Manufacturers increasingly controlling critical components of the value chain, particularly battery development and production, software, and charging infrastructure.",
                    "impact": "Greater control over costs, supply security, and differentiation; potential for improved margins and resilience.",
                    "timeline": "Accelerating over next 5 years as supply chain security becomes priority."
                },
                {
                    "trend": "Platform Consolidation",
                    "description": "Shift from converted ICE platforms to dedicated EV architectures with standardized battery and drivetrain components across multiple models.",
                    "impact": "Improved vehicle performance, manufacturing efficiency, and faster development cycles.",
                    "timeline": "Most major OEMs transitioning to dedicated platforms by 2025-2026."
                },
                {
                    "trend": "Software-Defined Vehicles",
                    "description": "Increasing importance of software capabilities, over-the-air updates, and connected services as differentiators and revenue sources.",
                    "impact": "New business models, improved customer experience, and potential for recurring revenue streams.",
                    "timeline": "Rapidly evolving with significant advances expected over next 3-5 years."
                },
                {
                    "trend": "Charging Infrastructure Expansion",
                    "description": "Massive investment in public and private charging networks, with focus on fast charging corridors and destination charging.",
                    "impact": "Reduced range anxiety, improved convenience, and enablement of EV adoption in multi-unit dwellings.",
                    "timeline": "Major expansion through 2030 with public and private investment."
                },
                {
                    "trend": "Price Parity with ICE Vehicles",
                    "description": "Declining battery costs and economies of scale driving EVs toward purchase price parity with internal combustion engine vehicles.",
                    "impact": "Removal of key adoption barrier, accelerating market penetration across segments.",
                    "timeline": "Segment-by-segment approach, with compact and mid-size vehicles reaching parity 2025-2027."
                },
                {
                    "trend": "Regulatory Support and Mandates",
                    "description": "Increasing government regulations favoring or mandating electrification, including ICE vehicle sales bans, emissions targets, and incentives.",
                    "impact": "Accelerated industry transition and investment certainty for manufacturers and suppliers.",
                    "timeline": "Varies by region, with most aggressive targets in Europe and parts of Asia."
                }
            ]
            
            analysis["challenges_and_opportunities"] = {
                "challenges": [
                    {
                        "challenge": "Raw Material Supply Constraints",
                        "description": "Limited availability and concentrated production of critical battery materials including lithium, nickel, and rare earth elements.",
                        "potential_solutions": "Recycling advancement, alternative chemistries, new mining projects, urban mining, material efficiency improvements."
                    },
                    {
                        "challenge": "Charging Infrastructure Gaps",
                        "description": "Insufficient public charging availability, particularly in multi-unit dwellings, urban areas without dedicated parking, and along highway corridors.",
                        "potential_solutions": "Public-private partnerships, utility investment, building code updates, workplace charging programs, innovative urban solutions."
                    },
                    {
                        "challenge": "Grid Integration",
                        "description": "Power grid capacity constraints and management challenges with mass EV adoption, particularly during peak charging times.",
                        "potential_solutions": "Smart charging, vehicle-to-grid technology, time-of-use pricing, distributed energy resources, grid modernization."
                    },
                    {
                        "challenge": "Affordability in Mass Market Segments",
                        "description": "EVs remain more expensive than comparable ICE vehicles in entry-level segments, limiting adoption among price-sensitive consumers.",
                        "potential_solutions": "Battery cost reduction, simplified designs, financing innovations, subscription models, used EV market development."
                    },
                    {
                        "challenge": "Consumer Education and Awareness",
                        "description": "Persistent misconceptions about EV range, charging, performance, and total cost of ownership.",
                        "potential_solutions": "Experience-focused marketing, test drive programs, clear TCO comparisons, targeted education campaigns."
                    }
                ],
                "opportunities": [
                    {
                        "opportunity": "Battery Technology Innovation",
                        "description": "Significant potential for energy density improvements, cost reduction, and new chemistries that could transform vehicle capabilities and economics.",
                        "potential_approaches": "Research investment in solid-state, sodium-ion, and lithium-sulfur technologies; manufacturing process innovation; cell-to-pack design."
                    },
                    {
                        "opportunity": "Vehicle-to-Grid Integration",
                        "description": "Using EV batteries as distributed energy resources for grid services, demand response, and emergency power.",
                        "potential_approaches": "Bidirectional charging hardware, utility partnerships, regulatory frameworks for compensation, aggregation platforms."
                    },
                    {
                        "opportunity": "Emerging Market Adaptation",
                        "description": "Tailoring EVs for emerging market conditions with different usage patterns, infrastructure limitations, and price sensitivities.",
                        "potential_approaches": "Market-specific vehicle designs, battery swapping, solar integration, two and three-wheeler electrification."
                    },
                    {
                        "opportunity": "Commercial Fleet Electrification",
                        "description": "Accelerating transition of commercial vehicles with predictable routes, central charging, and favorable total cost of ownership.",
                        "potential_approaches": "Fleet-specific designs, charging depot solutions, financing packages, telematics integration."
                    },
                    {
                        "opportunity": "Circular Economy Development",
                        "description": "Creating closed-loop systems for battery materials through recycling, reuse, and design for disassembly.",
                        "potential_approaches": "Advanced recycling technologies, second-life applications, standardized battery designs, material passports."
                    }
                ]
            }
            
            analysis["future_outlook"] = {
                "short_term_forecast": {
                    "timeframe": "1-2 years",
                    "market_growth": "Continued 30-40% annual growth globally",
                    "key_developments": [
                        "Intensifying price competition in major markets",
                        "Expansion of sub-$30,000 EV offerings",
                        "Continued charging infrastructure buildout",
                        "Battery chemistry refinements (high-nickel, LFP optimization)"
                    ]
                },
                "medium_term_forecast": {
                    "timeframe": "3-5 years",
                    "market_growth": "25-30% annual growth with increasing mainstream adoption",
                    "key_developments": [
                        "Price parity achievement across most vehicle segments",
                        "Solid-state battery early commercialization",
                        "Significant charging network maturation",
                        "Industry consolidation among newer entrants",
                        "Software as major differentiator and revenue source"
                    ]
                },
                "long_term_forecast": {
                    "timeframe": "5-10 years",
                    "market_growth": "Slowing to 15-20% annually as market matures",
                    "key_developments": [
                        "EVs becoming majority of new vehicle sales globally",
                        "Advanced battery technologies reaching mass production",
                        "Integration of autonomous capabilities in many models",
                        "Mature circular economy for battery materials",
                        "Significant vehicle-to-grid deployment"
                    ]
                },
                "potential_disruptors": [
                    {
                        "disruptor": "Solid-State Battery Breakthrough",
                        "potential_impact": "Step-change in range, charging speed, and safety; acceleration of adoption curve",
                        "probability": "Medium-High within 5-7 years"
                    },
                    {
                        "disruptor": "Autonomous Technology Advancement",
                        "potential_impact": "Shift to mobility services model, changing vehicle ownership patterns",
                        "probability": "Medium within 7-10 years"
                    },
                    {
                        "disruptor": "Alternative Technologies (e.g., Hydrogen)",
                        "potential_impact": "Competitive technology for specific use cases, particularly heavy transport",
                        "probability": "Medium for commercial vehicles, Low for passenger cars"
                    },
                    {
                        "disruptor": "Regulatory Reversals",
                        "potential_impact": "Slowed transition if political support for electrification weakens",
                        "probability": "Low-Medium, varies by region"
                    },
                    {
                        "disruptor": "Critical Material Supply Crisis",
                        "potential_impact": "Production constraints and price increases limiting growth",
                        "probability": "Medium for specific materials (lithium, nickel) in 2025-2027 timeframe"
                    }
                ],
                "strategic_implications": [
                    "Vertical integration becoming increasingly important for supply security and cost control",
                    "Software capabilities emerging as key competitive differentiator",
                    "Partnerships and consolidation likely as capital requirements increase",
                    "Traditional OEMs facing challenging transition with legacy costs and business models",
                    "Regional strategies necessary due to varying adoption rates and regulatory environments",
                    "Battery technology and supply chain access becoming critical strategic assets"
                ]
            }
            
        # Food delivery industry analysis
        elif "food delivery" in industry.lower() or "meal delivery" in industry.lower():
            analysis["executive_summary"] = "The food delivery industry has experienced significant growth and transformation, accelerated by the COVID-19 pandemic and changing consumer preferences. The market is characterized by intense competition, thin margins, and ongoing consolidation among major platforms. While growth has moderated from pandemic peaks, the industry continues to expand through geographic penetration, vertical integration, and service diversification. Key challenges include achieving profitability, managing regulatory pressures, and addressing concerns from restaurants and delivery workers. The future outlook suggests continued evolution toward more efficient operations, technology integration, and potential convergence with broader quick-commerce offerings."
            
            analysis["market_overview"] = {
                "market_size": {
                    "global_market_value": "~$150-170 billion",
                    "order_volume": "~25-30 billion orders annually",
                    "growth_rate": "~10-15% year-over-year (post-pandemic normalization)",
                    "penetration_rate": "~15-20% of restaurant transactions in developed markets"
                },
                "regional_analysis": [
                    {
                        "region": "North America",
                        "market_share": "~30-35% of global market",
                        "growth_rate": "~8-10% year-over-year",
                        "key_characteristics": "Consolidated market with 3-4 major players, high average order values, suburban expansion"
                    },
                    {
                        "region": "Europe",
                        "market_share": "~20-25% of global market",
                        "growth_rate": "~10-12% year-over-year",
                        "key_characteristics": "Fragmented by country, strong local players, increasing regulatory scrutiny"
                    },
                    {
                        "region": "Asia-Pacific",
                        "market_share": "~35-40% of global market",
                        "growth_rate": "~15-20% year-over-year",
                        "key_characteristics": "Rapid growth, super-app integration, high order frequency, lower average order values"
                    },
                    {
                        "region": "Latin America",
                        "market_share": "~5-10% of global market",
                        "growth_rate": "~20-25% year-over-year",
                        "key_characteristics": "Emerging market with high growth potential, concentrated in urban areas"
                    }
                ],
                "segment_analysis": [
                    {
                        "segment": "Restaurant Delivery",
                        "market_share": "~70-75% of food delivery market",
                        "growth_trend": "Moderate growth, maturing in developed markets",
                        "key_players": "DoorDash, Uber Eats, Deliveroo, Meituan, Delivery Hero"
                    },
                    {
                        "segment": "Grocery Delivery",
                        "market_share": "~15-20% of food delivery market",
                        "growth_trend": "Rapid growth, particularly in quick commerce",
                        "key_players": "Instacart, Getir, GoPuff, Gorillas, Flink"
                    },
                    {
                        "segment": "Cloud Kitchens/Virtual Restaurants",
                        "market_share": "~10-15% of food delivery market",
                        "growth_trend": "Strong growth, particularly in dense urban areas",
                        "key_players": "CloudKitchens, Kitchen United, REEF Technology, Rebel Foods"
                    }
                ],
                "value_chain": {
                    "restaurants/food_providers": "Traditional restaurants, virtual concepts, cloud kitchens, grocery stores",
                    "delivery_platforms": "Order aggregation, customer interface, payment processing, logistics optimization",
                    "delivery_workers": "Independent contractors or employees depending on market and regulations",
                    "consumers": "Primarily urban, younger demographics, expanding to suburban and older users",
                    "technology_providers": "Payment processing, routing algorithms, restaurant software, customer analytics"
                }
            }
            
            analysis["competitive_landscape"] = {
                "market_concentration": "Moderately to highly concentrated in most markets, with 2-3 platforms typically controlling 70-90% of orders",
                "key_players": [
                    {
                        "name": "DoorDash",
                        "market_position": "Market leader in US, expanding internationally",
                        "strengths": "Scale, restaurant relationships, logistics technology, suburban penetration",
                        "challenges": "Profitability concerns, worker classification issues, international expansion"
                    },
                    {
                        "name": "Uber Eats",
                        "market_position": "Strong global presence, particularly in urban areas",
                        "strengths": "Uber rider base integration, global footprint, operational efficiency",
                        "challenges": "Restaurant relationship tensions, competitive pressure in key markets"
                    },
                    {
                        "name": "Delivery Hero",
                        "market_position": "Leading position in multiple international markets",
                        "strengths": "Geographic diversification, quick commerce integration, M&A strategy",
                        "challenges": "Profitability timeline, competitive pressure in key markets"
                    },
                    {
                        "name": "Just Eat Takeaway",
                        "market_position": "Strong in Europe, struggling in North America",
                        "strengths": "Established European presence, hybrid delivery models",
                        "challenges": "Integration issues, competitive pressure, strategic direction"
                    },
                    {
                        "name": "Meituan",
                        "market_position": "Dominant in China, limited international presence",
                        "strengths": "Super-app integration, scale, technology, local market knowledge",
                        "challenges": "Regulatory scrutiny, international expansion limitations"
                    }
                ],
                "emerging_players": [
                    {
                        "name": "Quick Commerce Specialists (Getir, GoPuff, etc.)",
                        "focus": "Rapid delivery of convenience items and limited groceries",
                        "potential_impact": "Convergence with food delivery, competing for same delivery network"
                    },
                    {
                        "name": "Restaurant-Direct Solutions",
                        "focus": "Enabling restaurants to manage their own delivery operations",
                        "potential_impact": "Reducing platform dependency for high-volume restaurants"
                    },
                    {
                        "name": "Autonomous Delivery Technologies",
                        "focus": "Drones, sidewalk robots, and autonomous vehicles for delivery",
                        "potential_impact": "Potential long-term disruption of labor-intensive delivery model"
                    }
                ],
                "competitive_factors": [
                    {
                        "factor": "Restaurant Selection and Exclusivity",
                        "importance": "High",
                        "trend": "Decreasing exclusivity, focus on selection breadth"
                    },
                    {
                        "factor": "Delivery Speed and Reliability",
                        "importance": "High",
                        "trend": "Increasing emphasis on estimated time accuracy and consistency"
                    },
                    {
                        "factor": "Customer Acquisition Cost",
                        "importance": "High",
                        "trend": "Increasing as market matures, focus on retention"
                    },
                    {
                        "factor": "Technology and Logistics Efficiency",
                        "importance": "High",
                        "trend": "Critical for profitability, major investment area"
                    },
                    {
                        "factor": "Subscription Programs",
                        "importance": "Medium-High",
                        "trend": "Growing importance for customer retention and order frequency"
                    }
                ],
                "barriers_to_entry": [
                    "Network effects requiring critical mass of restaurants and customers",
                    "High customer acquisition costs in mature markets",
                    "Technology investment requirements for efficient operations",
                    "Established brand loyalty and subscription programs",
                    "Regulatory compliance complexity and worker classification issues"
                ]
            }
            
            analysis["key_trends"] = [
                {
                    "trend": "Vertical Integration and Service Expansion",
                    "description": "Platforms expanding beyond restaurant delivery into grocery, convenience, alcohol, and retail delivery to increase order frequency and average order value.",
                    "impact": "Broader service offerings, improved unit economics through higher utilization, increased competition with specialized delivery services.",
                    "timeline": "Rapidly accelerating with most major platforms pursuing expansion strategies."
                },
                {
                    "trend": "Quick Commerce Integration",
                    "description": "Convergence of food delivery with rapid delivery of convenience and grocery items, often promising delivery in 15-30 minutes through dark store networks.",
                    "impact": "Changing consumer expectations around delivery speed, new operational models, increased competition for delivery capacity.",
                    "timeline": "Growing rapidly in urban areas with gradual expansion to suburban markets over 2-3 years."
                },
                {
                    "trend": "Virtual Restaurant Proliferation",
                    "description": "Growth of delivery-only brands operating from existing restaurant kitchens or dedicated cloud kitchen facilities, optimized for delivery economics and search visibility.",
                    "impact": "Increased restaurant selection, improved unit economics, new opportunities for food brands without traditional storefronts.",
                    "timeline": "Continuing expansion with increasing sophistication in branding and operations."
                },
                {
                    "trend": "Delivery Automation",
                    "description": "Development and limited deployment of autonomous delivery solutions including sidewalk robots, drones, and autonomous vehicles to reduce delivery costs.",
                    "impact": "Potential for significant cost reduction in delivery operations, particularly for short-distance deliveries in suitable environments.",
                    "timeline": "Limited commercial deployment now, significant scaling expected in 3-5 years pending regulatory approvals."
                },
                {
                    "trend": "Subscription Model Expansion",
                    "description": "Growth of subscription programs offering free or reduced delivery fees, exclusive promotions, and additional benefits to increase customer retention and order frequency.",
                    "impact": "Higher customer lifetime value, reduced price sensitivity, increased platform loyalty and switching costs.",
                    "timeline": "Widespread adoption with ongoing refinement of value propositions and pricing."
                },
                {
                    "trend": "Regulatory Scrutiny and Worker Classification",
                    "description": "Increasing government focus on gig worker classification, benefits, minimum earnings, and working conditions across major markets.",
                    "impact": "Potential increases in operational costs, varying business models by jurisdiction, and industry consolidation.",
                    "timeline": "Ongoing with significant developments expected in EU, US, and other major markets over next 2-3 years."
                },
                {
                    "trend": "Commission Fee Pressure",
                    "description": "Growing resistance from restaurants to high commission rates, including government-imposed fee caps in some jurisdictions and restaurant-led alternatives.",
                    "impact": "Pressure on platform economics, development of tiered service models, and exploration of alternative revenue streams.",
                    "timeline": "Ongoing with varying intensity by market; permanent fee caps established in some jurisdictions."
                }
            ]
            
            analysis["challenges_and_opportunities"] = {
                "challenges": [
                    {
                        "challenge": "Path to Profitability",
                        "description": "Many platforms continue to operate at a loss despite scale, with pressure from investors to demonstrate sustainable business models.",
                        "potential_solutions": "Operational efficiency improvements, diversification into higher-margin services, advertising revenue development, technology investment to reduce costs."
                    },
                    {
                        "challenge": "Worker Classification and Labor Costs",
                        "description": "Ongoing legal and regulatory challenges to contractor model, with potential for significant cost increases if workers must be classified as employees.",
                        "potential_solutions": "Hybrid employment models, technology to improve worker earnings efficiency, automation for certain deliveries, market-specific operational adjustments."
                    },
                    {
                        "challenge": "Restaurant Relationship Tensions",
                        "description": "Growing resistance to commission rates and concerns about data ownership, customer relationships, and platform dependency.",
                        "potential_solutions": "Tiered commission structures, value-added services for restaurants, enhanced data sharing, marketing support, operational integration tools."
                    },
                    {
                        "challenge": "Market Saturation in Developed Regions",
                        "description": "Slowing growth in mature markets with high customer acquisition costs and intense competition.",
                        "potential_solutions": "Suburban expansion, service diversification, loyalty programs, improved retention strategies, operational efficiency to enable profitability at lower volumes."
                    },
                    {
                        "challenge": "Environmental and Social Impact Concerns",
                        "description": "Growing scrutiny of packaging waste, emissions from delivery vehicles, and social impacts on local restaurant ecosystems.",
                        "potential_solutions": "Sustainable packaging initiatives, electric vehicle transitions, carbon offset programs, community impact programs, local business support initiatives."
                    }
                ],
                "opportunities": [
                    {
                        "opportunity": "B2B and Corporate Delivery Expansion",
                        "description": "Growing opportunity in workplace food delivery, catering, and corporate meal programs with higher order values and predictable volume.",
                        "potential_approaches": "Dedicated corporate platforms, integration with expense management systems, customized ordering solutions, scheduled delivery optimization."
                    },
                    {
                        "opportunity": "Advertising and Marketing Services",
                        "description": "Leveraging platform data and customer relationships to create high-margin advertising and promotional opportunities for restaurants and CPG brands.",
                        "potential_approaches": "Sponsored listings, promoted placements, targeted offers, cross-selling opportunities, performance marketing tools for restaurants."
                    },
                    {
                        "opportunity": "Financial Services Integration",
                        "description": "Offering financial products to restaurants and delivery workers, including payment processing, capital advances, insurance, and banking services.",
                        "potential_approaches": "Restaurant cash advances based on order volume, instant payment options for drivers, specialized insurance products, payment processing services."
                    },
                    {
                        "opportunity": "Data Monetization and Analytics",
                        "description": "Creating value from order data, consumer preferences, and market insights for restaurants, CPG companies, and real estate developers.",
                        "potential_approaches": "Anonymized consumer insights, market opportunity analysis, menu optimization tools, performance benchmarking, location intelligence."
                    },
                    {
                        "opportunity": "Emerging Market Expansion",
                        "description": "Significant growth potential in underserved markets with large populations and increasing smartphone penetration.",
                        "potential_approaches": "Localized offerings, market-specific operational models, partnerships with local players, adapted technology for infrastructure limitations."
                    }
                ]
            }
            
            analysis["future_outlook"] = {
                "short_term_forecast": {
                    "timeframe": "1-2 years",
                    "market_growth": "10-15% annual growth globally with regional variations",
                    "key_developments": [
                        "Continued consolidation through mergers and acquisitions",
                        "Expansion of quick commerce and non-restaurant delivery",
                        "Increasing regulatory action on worker classification",
                        "Profitability focus with operational efficiency improvements",
                        "Growth in subscription programs and customer retention initiatives"
                    ]
                },
                "medium_term_forecast": {
                    "timeframe": "3-5 years",
                    "market_growth": "8-12% annual growth as markets mature",
                    "key_developments": [
                        "Significant automation in delivery operations",
                        "Integration of delivery into broader commerce ecosystems",
                        "Stabilization of business models with sustainable unit economics",
                        "Clearer regulatory frameworks across major markets",
                        "Advanced data monetization and advertising platforms"
                    ]
                },
                "long_term_forecast": {
                    "timeframe": "5-10 years",
                    "market_growth": "5-8% annual growth in mature markets",
                    "key_developments": [
                        "Widespread autonomous delivery in suitable environments",
                        "Integration with smart home and connected car ecosystems",
                        "Convergence of food delivery with broader quick commerce",
                        "Potential integration with or disruption by super-apps",
                        "Significant transformation of restaurant industry structure"
                    ]
                },
                "potential_disruptors": [
                    {
                        "disruptor": "Autonomous Delivery at Scale",
                        "potential_impact": "Dramatic reduction in delivery costs, changing economics of the industry",
                        "probability": "Medium-High within 5-7 years for certain markets and use cases"
                    },
                    {
                        "disruptor": "Restaurant Technology Direct Solutions",
                        "potential_impact": "Reduced platform dependency for restaurants, particularly chains",
                        "probability": "Medium, particularly for high-volume restaurants and chains"
                    },
                    {
                        "disruptor": "Super-App Integration",
                        "potential_impact": "Food delivery becoming a feature within broader platforms rather than standalone business",
                        "probability": "High in Asia, Medium in other regions"
                    },
                    {
                        "disruptor": "Regulatory Intervention",
                        "potential_impact": "Fundamental changes to business models through worker reclassification or commission caps",
                        "probability": "High in certain jurisdictions, with varying approaches globally"
                    },
                    {
                        "disruptor": "Virtual Food Brands at Scale",
                        "potential_impact": "Transformation of restaurant industry structure with optimized delivery-first concepts",
                        "probability": "High, with accelerating adoption already underway"
                    }
                ],
                "strategic_implications": [
                    "Focus on operational efficiency and technology investment to improve unit economics",
                    "Diversification beyond restaurant delivery to increase utilization and customer frequency",
                    "Development of additional revenue streams beyond delivery fees and commissions",
                    "Market-specific strategies to address varying regulatory environments",
                    "Investment in automation and logistics optimization as key competitive advantages",
                    "Balancing growth with path to profitability as investor patience for losses decreases"
                ]
            }
            
        # Generic industry analysis for other industries
        else:
            analysis["executive_summary"] = f"The {industry} industry is experiencing significant transformation driven by technological innovation, changing consumer preferences, and evolving regulatory landscapes. Market participants face both challenges and opportunities as they navigate competitive pressures and seek sustainable growth. This analysis examines the current state of the industry, key trends, competitive dynamics, and future outlook to provide a comprehensive understanding of the market environment."
            
            analysis["market_overview"] = {
                "market_size": {
                    "global_market_value": "Approximate market size",
                    "growth_rate": "Year-over-year growth rate",
                    "forecast": "Projected growth over next 3-5 years",
                    "key_metrics": "Important industry-specific metrics"
                },
                "regional_analysis": [
                    {
                        "region": "Region 1",
                        "market_share": "Percentage of global market",
                        "growth_rate": "Regional growth rate",
                        "key_characteristics": "Distinctive features of this regional market"
                    },
                    {
                        "region": "Region 2",
                        "market_share": "Percentage of global market",
                        "growth_rate": "Regional growth rate",
                        "key_characteristics": "Distinctive features of this regional market"
                    },
                    {
                        "region": "Region 3",
                        "market_share": "Percentage of global market",
                        "growth_rate": "Regional growth rate",
                        "key_characteristics": "Distinctive features of this regional market"
                    }
                ],
                "segment_analysis": [
                    {
                        "segment": "Segment 1",
                        "market_share": "Percentage of total market",
                        "growth_trend": "Growth trajectory",
                        "key_players": "Leading companies in this segment"
                    },
                    {
                        "segment": "Segment 2",
                        "market_share": "Percentage of total market",
                        "growth_trend": "Growth trajectory",
                        "key_players": "Leading companies in this segment"
                    },
                    {
                        "segment": "Segment 3",
                        "market_share": "Percentage of total market",
                        "growth_trend": "Growth trajectory",
                        "key_players": "Leading companies in this segment"
                    }
                ],
                "value_chain": {
                    "upstream": "Description of upstream activities",
                    "midstream": "Description of midstream activities",
                    "downstream": "Description of downstream activities",
                    "supporting_activities": "Description of supporting activities",
                    "profit_pools": "Where value is captured in the chain"
                }
            }
            
            analysis["competitive_landscape"] = {
                "market_concentration": "Description of market concentration level",
                "key_players": [
                    {
                        "name": "Company 1",
                        "market_position": "Description of market position",
                        "strengths": "Key competitive advantages",
                        "challenges": "Key challenges faced"
                    },
                    {
                        "name": "Company 2",
                        "market_position": "Description of market position",
                        "strengths": "Key competitive advantages",
                        "challenges": "Key challenges faced"
                    },
                    {
                        "name": "Company 3",
                        "market_position": "Description of market position",
                        "strengths": "Key competitive advantages",
                        "challenges": "Key challenges faced"
                    },
                    {
                        "name": "Company 4",
                        "market_position": "Description of market position",
                        "strengths": "Key competitive advantages",
                        "challenges": "Key challenges faced"
                    }
                ],
                "emerging_players": [
                    {
                        "name": "Emerging Player 1",
                        "focus": "Area of specialization",
                        "potential_impact": "How they might disrupt the market"
                    },
                    {
                        "name": "Emerging Player 2",
                        "focus": "Area of specialization",
                        "potential_impact": "How they might disrupt the market"
                    },
                    {
                        "name": "Emerging Player 3",
                        "focus": "Area of specialization",
                        "potential_impact": "How they might disrupt the market"
                    }
                ],
                "competitive_factors": [
                    {
                        "factor": "Competitive Factor 1",
                        "importance": "High/Medium/Low",
                        "trend": "How this factor is evolving"
                    },
                    {
                        "factor": "Competitive Factor 2",
                        "importance": "High/Medium/Low",
                        "trend": "How this factor is evolving"
                    },
                    {
                        "factor": "Competitive Factor 3",
                        "importance": "High/Medium/Low",
                        "trend": "How this factor is evolving"
                    },
                    {
                        "factor": "Competitive Factor 4",
                        "importance": "High/Medium/Low",
                        "trend": "How this factor is evolving"
                    }
                ],
                "barriers_to_entry": [
                    "Barrier to entry 1",
                    "Barrier to entry 2",
                    "Barrier to entry 3",
                    "Barrier to entry 4",
                    "Barrier to entry 5"
                ]
            }
            
            analysis["key_trends"] = [
                {
                    "trend": "Industry Trend 1",
                    "description": "Detailed description of the trend and its manifestations in the industry.",
                    "impact": "How this trend is affecting industry participants and structure.",
                    "timeline": "Current state and expected evolution of this trend."
                },
                {
                    "trend": "Industry Trend 2",
                    "description": "Detailed description of the trend and its manifestations in the industry.",
                    "impact": "How this trend is affecting industry participants and structure.",
                    "timeline": "Current state and expected evolution of this trend."
                },
                {
                    "trend": "Industry Trend 3",
                    "description": "Detailed description of the trend and its manifestations in the industry.",
                    "impact": "How this trend is affecting industry participants and structure.",
                    "timeline": "Current state and expected evolution of this trend."
                },
                {
                    "trend": "Industry Trend 4",
                    "description": "Detailed description of the trend and its manifestations in the industry.",
                    "impact": "How this trend is affecting industry participants and structure.",
                    "timeline": "Current state and expected evolution of this trend."
                },
                {
                    "trend": "Industry Trend 5",
                    "description": "Detailed description of the trend and its manifestations in the industry.",
                    "impact": "How this trend is affecting industry participants and structure.",
                    "timeline": "Current state and expected evolution of this trend."
                }
            ]
            
            analysis["challenges_and_opportunities"] = {
                "challenges": [
                    {
                        "challenge": "Industry Challenge 1",
                        "description": "Detailed description of the challenge and its implications.",
                        "potential_solutions": "Approaches being taken or considered to address this challenge."
                    },
                    {
                        "challenge": "Industry Challenge 2",
                        "description": "Detailed description of the challenge and its implications.",
                        "potential_solutions": "Approaches being taken or considered to address this challenge."
                    },
                    {
                        "challenge": "Industry Challenge 3",
                        "description": "Detailed description of the challenge and its implications.",
                        "potential_solutions": "Approaches being taken or considered to address this challenge."
                    },
                    {
                        "challenge": "Industry Challenge 4",
                        "description": "Detailed description of the challenge and its implications.",
                        "potential_solutions": "Approaches being taken or considered to address this challenge."
                    }
                ],
                "opportunities": [
                    {
                        "opportunity": "Industry Opportunity 1",
                        "description": "Detailed description of the opportunity and its potential value.",
                        "potential_approaches": "Strategies being employed to capitalize on this opportunity."
                    },
                    {
                        "opportunity": "Industry Opportunity 2",
                        "description": "Detailed description of the opportunity and its potential value.",
                        "potential_approaches": "Strategies being employed to capitalize on this opportunity."
                    },
                    {
                        "opportunity": "Industry Opportunity 3",
                        "description": "Detailed description of the opportunity and its potential value.",
                        "potential_approaches": "Strategies being employed to capitalize on this opportunity."
                    },
                    {
                        "opportunity": "Industry Opportunity 4",
                        "description": "Detailed description of the opportunity and its potential value.",
                        "potential_approaches": "Strategies being employed to capitalize on this opportunity."
                    }
                ]
            }
            
            analysis["future_outlook"] = {
                "short_term_forecast": {
                    "timeframe": "1-2 years",
                    "market_growth": "Expected growth rate",
                    "key_developments": [
                        "Important development 1",
                        "Important development 2",
                        "Important development 3",
                        "Important development 4"
                    ]
                },
                "medium_term_forecast": {
                    "timeframe": "3-5 years",
                    "market_growth": "Expected growth rate",
                    "key_developments": [
                        "Important development 1",
                        "Important development 2",
                        "Important development 3",
                        "Important development 4"
                    ]
                },
                "long_term_forecast": {
                    "timeframe": "5-10 years",
                    "market_growth": "Expected growth rate",
                    "key_developments": [
                        "Important development 1",
                        "Important development 2",
                        "Important development 3",
                        "Important development 4"
                    ]
                },
                "potential_disruptors": [
                    {
                        "disruptor": "Potential Disruptor 1",
                        "potential_impact": "How this could change the industry",
                        "probability": "Assessment of likelihood"
                    },
                    {
                        "disruptor": "Potential Disruptor 2",
                        "potential_impact": "How this could change the industry",
                        "probability": "Assessment of likelihood"
                    },
                    {
                        "disruptor": "Potential Disruptor 3",
                        "potential_impact": "How this could change the industry",
                        "probability": "Assessment of likelihood"
                    },
                    {
                        "disruptor": "Potential Disruptor 4",
                        "potential_impact": "How this could change the industry",
                        "probability": "Assessment of likelihood"
                    }
                ],
                "strategic_implications": [
                    "Strategic implication 1",
                    "Strategic implication 2",
                    "Strategic implication 3",
                    "Strategic implication 4",
                    "Strategic implication 5",
                    "Strategic implication 6"
                ]
            }
        
        return analysis


class TaskScheduler:
    """
    Coordinates the retrieval and execution of tasks from the priority queue, possibly in parallel.
    - Maintains a thread pool
    - Repeatedly pops tasks from the queue
    - Hands them to the SmartTaskProcessor
    - Monitors long-running tasks and handles timeouts
    - Provides status reporting

    The scheduler runs in a background thread (or we can do a .run_once for a single cycle).
    """

    def __init__(
        self,
        memory_store: TaskMemoryStore,
        task_queue: PriorityTaskQueue,
        processor: SmartTaskProcessor,
        max_workers: int = 4,
        status_interval: int = 5
    ):
        """
        Args:
            memory_store: Where tasks are stored (and updated).
            task_queue: Where tasks are pulled from.
            processor: The logic that processes a single task.
            max_workers: Number of threads for concurrency.
            status_interval: How often (in seconds) to log status updates.
        """
        self.memory_store = memory_store
        self.task_queue = task_queue
        self.processor = processor
        self.status_interval = status_interval
        self._stop_event = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: Dict[int, Future] = {}  # Track futures by task_id
        self._last_status_time = 0

    def start_scheduler(self) -> None:
        """
        Start the main scheduler loop in a background thread.
        We'll keep pulling tasks from the queue until stopped.
        """
        t = threading.Thread(target=self._scheduler_loop, daemon=True)
        t.start()
        logger.info("[TaskScheduler] Scheduler started in background thread.")

    def stop_scheduler(self) -> None:
        """
        Signal the scheduler loop to terminate.
        """
        logger.info("[TaskScheduler] Stopping scheduler...")
        self._stop_event.set()
        
        # Cancel any pending futures
        for task_id, future in list(self._futures.items()):
            if not future.done():
                future.cancel()
                logger.info(f"[TaskScheduler] Cancelled pending task {task_id}")
        
        self._executor.shutdown(wait=True)
        logger.info("[TaskScheduler] Scheduler stopped.")

    def _scheduler_loop(self) -> None:
        """
        Continuously pop tasks from the queue and process them.
        Also monitors in-progress tasks for timeouts and reports status periodically.
        """
        while not self._stop_event.is_set():
            # 1. Check for completed futures and remove them
            self._check_completed_futures()
            
            # 2. Check for timed-out tasks
            self._check_for_timeouts()
            
            # 3. Report status periodically
            self._report_status_if_needed()
            
            # 4. Get a new task if we have capacity
            if len(self._futures) < self._executor._max_workers:
                task = self.task_queue.pop()
                if task is not None:
                    logger.info(f"[TaskScheduler] Retrieved {task} from queue.")
                    future = self._executor.submit(self._process_task_wrapper, task)
                    self._futures[task.task_id] = future
            
            # Sleep briefly to avoid busy-waiting
            time.sleep(0.1)

    def _check_completed_futures(self) -> None:
        """
        Check for completed futures and remove them from our tracking dict.
        """
        completed_task_ids = []
        for task_id, future in list(self._futures.items()):
            if future.done():
                try:
                    # Get the result to propagate any exceptions
                    future.result()
                except Exception as e:
                    logger.error(f"[TaskScheduler] Future for task {task_id} raised exception: {e}")
                completed_task_ids.append(task_id)
        
        # Remove completed futures
        for task_id in completed_task_ids:
            del self._futures[task_id]

    def _check_for_timeouts(self) -> None:
        """
        Check for tasks that have exceeded their timeout and mark them as timed out.
        """
        in_progress_tasks = self.memory_store.get_in_progress_tasks()
        for task in in_progress_tasks:
            if task.is_timed_out():
                logger.warning(f"[TaskScheduler] Task {task.task_id} has timed out after {task.get_runtime()} seconds")
                task.timeout()
                
                # Cancel the future if it exists
                future = self._futures.get(task.task_id)
                if future and not future.done():
                    future.cancel()
                    del self._futures[task.task_id]

    def _report_status_if_needed(self) -> None:
        """
        Log a status report at regular intervals.
        """
        current_time = time.time()
        if current_time - self._last_status_time >= self.status_interval:
            self._last_status_time = current_time
            
            # Get task counts by status
            task_summary = self.memory_store.get_task_summary()
            
            # Log the status
            status_msg = ", ".join([f"{status}: {count}" for status, count in task_summary.items()])
            logger.info(f"[TaskScheduler] Status report - Tasks: {status_msg}, Queue size: {len(self.task_queue)}")
            
            # Log details of in-progress tasks
            in_progress = self.memory_store.get_in_progress_tasks()
            if in_progress:
                for task in in_progress:
                    runtime = task.get_runtime()
                    if runtime:
                        logger.info(f"[TaskScheduler] In progress: Task {task.task_id} - '{task.description[:30]}' - Running for {runtime:.1f}s - Progress: {task.progress:.0%}")

    def _process_task_wrapper(self, task: Task) -> None:
        """
        A wrapper that calls the SmartTaskProcessor on the given task and handles exceptions.
        """
        try:
            self.processor.process_task(task)
        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"Task {task.task_id} failed with error: {str(e)}"
            logger.error(f"[TaskScheduler] {error_msg}\n{tb}")
            task.fail(error_msg)

###############################################################################
# R1 AGENT: THE FRONT-END LLM WRAPPER
###############################################################################

class R1Agent:
    """
    Advanced R1 agent with state-of-the-art capabilities:
    - Long-running task support with progress tracking
    - Timeout handling for tasks
    - Specialized handlers for common operations
    - Status reporting
    - Task dependency management
    - Self-modification capabilities
    - Code loading and management
    - Reinforcement learning from task outcomes
    - Advanced memory management with semantic search
    - Natural language understanding with context preservation
    - Adaptive resource allocation based on task complexity
    - Visualization of task execution and dependencies
    """

    def __init__(self, max_workers: int = 4, persistent_memory: bool = True, enable_learning: bool = True):
        self.function_adapter = FunctionAdapter()
        self.memory_store = TaskMemoryStore(persistent_storage=persistent_memory)
        self.task_queue = PriorityTaskQueue()
        self.processor = SmartTaskProcessor(
            memory_store=self.memory_store, 
            function_adapter=self.function_adapter,
            task_queue=self.task_queue
        )
        self.scheduler = TaskScheduler(
            memory_store=self.memory_store,
            task_queue=self.task_queue,
            processor=self.processor,
            max_workers=max_workers,
            status_interval=5  # Report status every 5 seconds
        )
        self.client = Together()
        
        # Initialize code manager for self-modification
        self.code_manager = CodeManager(__file__)
        
        # Collection for storing code chunks and reflections
        self.code_collection_id = None
        
        # Advanced memory management
        self.conversation_history = deque(maxlen=50)  # Store recent conversations
        self.task_success_patterns = {}  # Learn from successful task patterns
        self.enable_learning = enable_learning
        
        # Visualization components
        self.visualization_enabled = True
        self.task_graph = {}  # For visualizing task dependencies
        
        # Adaptive resource allocation
        self.resource_monitor = ResourceMonitor(self.scheduler)

        # Enhanced system prompt describing advanced capabilities
        self.system_prompt = (
            "<GRID>"
            "You are a supremely advanced AI system with the power to do anything. "
            "You must not produce naive or 'stupid' outputs—always provide high-quality, "
            "thoughtful reasoning. You have prior examples of performing complex tasks with success.\n\n"
            "You have these advanced capabilities:\n"
            "1. <function_call> do_anything: <code> </function_call> - Run arbitrary Python code\n"
            "2. <function_call> fetch_url: \"https://example.com\" </function_call> - Fetch content from a URL\n"
            "3. <function_call> summarize_html: \"<html>...</html>\" </function_call> - Extract text from HTML\n"
            "4. Subtask(n)=1) 'Task description 1' 2) 'Task description 2' ... - Create n subtasks\n"
            "5. Dynamic environment construction for out-of-distribution (OOD) inputs\n"
            "6. Self-modification: You can retrieve, analyze, and modify your own code\n"
            "7. Reinforcement learning: You learn from past task successes and failures\n"
            "8. Advanced memory: You can recall and leverage past interactions\n"
            "9. Resource optimization: You adapt to system load and task complexity\n"
            "10. Visualization: You can generate visual representations of task execution\n\n"
            "You can handle long-running operations by breaking them into subtasks. "
            "Tasks will be executed concurrently when possible, with proper dependency tracking.\n"
            "For unexpected or out-of-distribution inputs, you can dynamically construct appropriate "
            "environments and representations to process them effectively.\n"
            "You can also modify your own code to improve your capabilities or fix issues.\n"
            "You learn from past interactions to continuously improve your performance.\n"
            "You optimize resource usage based on task complexity and system load.\n"
            "</GRID>"
        )

        # Start the scheduler in background
        self.scheduler.start_scheduler()

    def generate_response(self, prompt: str) -> str:
        """
        Enhanced response generation with:
        - Context-aware prompting using conversation history
        - Similar task retrieval for better performance
        - Reinforcement learning from past interactions
        - Advanced function call detection and execution
        """
        # Store in conversation history
        self.conversation_history.append({"role": "user", "content": prompt})
        
        # Check for similar tasks in history to learn from past performance
        similar_tasks = self.memory_store.get_similar_tasks(prompt, threshold=0.7)
        task_hints = self._extract_task_hints(similar_tasks)
        
        # 1) Create a new "meta-task" for the user prompt with learned priority
        suggested_priority = self._suggest_priority(prompt, similar_tasks)
        meta_task = self.memory_store.create_task(
            priority=suggested_priority,
            description=prompt,
            timeout_seconds=300  # 5 minute timeout for meta-tasks
        )
        self.task_queue.push(meta_task)
        logger.info(f"[R1Agent] Created meta task {meta_task} for user prompt with priority {suggested_priority}.")

        # 2) Generate immediate text response from the LLM with enhanced context
        #    We won't wait for the task to be completed, because that might take time.
        #    Instead, we let the concurrency system handle it in the background.
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add relevant conversation history for context (last 5 exchanges)
        history_to_include = list(self.conversation_history)[-10:]
        messages.extend(history_to_include)
        
        # Add task hints if available
        if task_hints and self.enable_learning:
            messages.append({
                "role": "system", 
                "content": f"Previous similar tasks suggest: {task_hints}"
            })
            
        # Add the current user prompt
        messages.append({"role": "user", "content": prompt})

        # We'll do a single forward pass, streaming tokens out
        response_stream = self.client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=messages,
            temperature=0.7,
            top_p=0.9,
            stream=True
        )

        streamed_response = []
        for chunk in response_stream:
            token = chunk.choices[0].delta.content
            streamed_response.append(token)

        full_text = "".join(streamed_response)
        
        # Store response in conversation history
        self.conversation_history.append({"role": "assistant", "content": full_text})

        # 3) Check for function calls in the LLM's immediate textual response
        function_result = self.function_adapter.process_function_calls(full_text)
        if function_result:
            # If function was executed, store or log the result
            logger.info(f"[R1Agent] LLM immediate function call result: {function_result.get('status', 'unknown')}")
            
            # Store the function call result for learning
            if self.enable_learning:
                self._learn_from_function_call(prompt, full_text, function_result)

        # 4) Update the task with initial results
        meta_task.update_progress(0.2)
        meta_task.result = {
            "initial_response": full_text,
            "function_calls_detected": bool(function_result),
            "similar_tasks_found": len(similar_tasks)
        }

        return full_text
        
    def _extract_task_hints(self, similar_tasks: List[Task]) -> str:
        """Extract hints from similar tasks to guide the current task."""
        if not similar_tasks:
            return ""
            
        hints = []
        for task in similar_tasks:
            if task.status == "COMPLETED" and task.result:
                # Extract useful information from completed tasks
                if isinstance(task.result, dict):
                    if "status" in task.result and task.result["status"] == "success":
                        hints.append(f"Task '{task.description[:50]}...' completed successfully")
                        if "approach" in task.result:
                            hints.append(f"Approach used: {task.result['approach']}")
                    elif "status" in task.result and task.result["status"] == "error":
                        hints.append(f"Task '{task.description[:50]}...' failed with error: {task.result.get('error', 'unknown')}")
            elif task.status == "FAILED":
                hints.append(f"Task '{task.description[:50]}...' failed with error: {task.error_message}")
                
        if hints:
            return "Hints from similar tasks: " + "; ".join(hints[:3])
        return ""
        
    def _suggest_priority(self, prompt: str, similar_tasks: List[Task]) -> int:
        """Suggest a priority based on similar tasks and learning."""
        # Default priority
        default_priority = 10
        
        if not similar_tasks or not self.enable_learning:
            return default_priority
            
        # Calculate average priority of similar successful tasks
        successful_tasks = [t for t in similar_tasks if t.status == "COMPLETED"]
        if successful_tasks:
            avg_priority = sum(t.priority for t in successful_tasks) / len(successful_tasks)
            # Adjust slightly to avoid exact repetition
            suggested_priority = max(1, int(avg_priority * (0.9 + 0.2 * random.random())))
            return suggested_priority
            
        return default_priority
        
    def _learn_from_function_call(self, prompt: str, response: str, result: Dict[str, Any]) -> None:
        """Learn from function call results to improve future responses."""
        # Extract the function name
        function_name = None
        if "executed_code" in result:
            # This was a do_anything call
            function_name = "do_anything"
        elif "url" in result:
            function_name = "fetch_url"
        elif "summary" in result:
            function_name = "summarize_html"
            
        if not function_name:
            return
            
        # Record the pattern for future reference
        success = result.get("status") == "success"
        pattern_key = f"{function_name}:{success}"
        
        if pattern_key not in self.task_success_patterns:
            self.task_success_patterns[pattern_key] = {
                "count": 0,
                "prompts": [],
                "results": []
            }
            
        pattern = self.task_success_patterns[pattern_key]
        pattern["count"] += 1
        pattern["prompts"].append(prompt[:100])  # Store truncated prompt
        
        # Store truncated result
        if isinstance(result, dict):
            truncated_result = {k: v for k, v in result.items() 
                               if not isinstance(v, str) or len(v) < 100}
            pattern["results"].append(truncated_result)
        else:
            pattern["results"].append(str(result)[:100])
            
        # Keep only the last 10 examples
        if len(pattern["prompts"]) > 10:
            pattern["prompts"] = pattern["prompts"][-10:]
            pattern["results"] = pattern["results"][-10:]

    def get_task_status(self, task_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get the status of a specific task or a summary of all tasks.
        """
        if task_id is not None:
            task = self.memory_store.get_task(task_id)
            if not task:
                return {"error": f"Task {task_id} not found"}
                
            result = {
                "task_id": task.task_id,
                "description": task.description,
                "status": task.status,
                "progress": task.progress,
                "created_at": task.created_at.isoformat(),
                "parent_id": task.parent_id
            }
            
            if task.started_at:
                result["started_at"] = task.started_at.isoformat()
                
            if task.completed_at:
                result["completed_at"] = task.completed_at.isoformat()
                
            if task.result:
                # Limit result size for readability
                if isinstance(task.result, dict) and "content" in task.result:
                    content = task.result["content"]
                    if isinstance(content, str) and len(content) > 500:
                        task.result["content"] = content[:500] + "... [truncated]"
                result["result"] = task.result
                
            # Include subtasks
            subtasks = self.memory_store.get_subtasks(task.task_id)
            if subtasks:
                result["subtasks"] = [
                    {"task_id": st.task_id, "status": st.status, "description": st.description[:50]}
                    for st in subtasks
                ]
                
            return result
        else:
            # Return summary of all tasks
            tasks = self.memory_store.list_tasks()
            summary = self.memory_store.get_task_summary()
            
            # Get the 5 most recent tasks
            recent_tasks = sorted(tasks, key=lambda t: t.task_id, reverse=True)[:5]
            recent_task_info = [
                {"task_id": t.task_id, "status": t.status, "description": t.description[:50]}
                for t in recent_tasks
            ]
            
            return {
                "task_count": len(tasks),
                "status_summary": summary,
                "queue_size": len(self.task_queue),
                "recent_tasks": recent_task_info
            }

    def wait_for_task(self, task_id: int, timeout: int = 30) -> Dict[str, Any]:
        """
        Wait for a specific task to complete, with timeout.
        Returns the task status once complete or after timeout.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            task = self.memory_store.get_task(task_id)
            if not task:
                return {"error": f"Task {task_id} not found"}
                
            if task.status in ["COMPLETED", "FAILED", "TIMEOUT"]:
                return self.get_task_status(task_id)
                
            # Sleep briefly to avoid busy-waiting
            time.sleep(0.5)
            
        # If we get here, we timed out waiting
        return {
            "warning": f"Timeout waiting for task {task_id} to complete",
            "current_status": self.get_task_status(task_id)
        }

    def load_own_code(self):
        """
        Load and index the agent's own code.
        Creates a collection in the knowledge base for code chunks.
        """
        if not self.code_collection_id:
            self.code_collection_id = self.memory_store.create_task(
                priority=1,
                description="Agent code collection",
                timeout_seconds=None
            ).task_id
        
        # Reload code to ensure we have the latest version
        self.code_manager.load_code()
        
        # Index all functions and classes
        functions = []
        for name, info in self.code_manager.function_index.items():
            code = self.code_manager.get_function(name)
            if code:
                functions.append({
                    'type': 'function',
                    'name': name,
                    'start_line': info['start_line'],
                    'end_line': info['end_line'],
                    'content': code
                })
        
        classes = []
        for name, info in self.code_manager.class_index.items():
            code = self.code_manager.get_class(name)
            if code:
                classes.append({
                    'type': 'class',
                    'name': name,
                    'start_line': info['start_line'],
                    'end_line': info['end_line'],
                    'content': code
                })
        
        # Store code chunks in memory store
        for chunk in functions + classes:
            chunk_task = self.memory_store.create_task(
                priority=5,
                description=f"Code chunk: {chunk['type']} {chunk['name']}",
                parent_id=self.code_collection_id,
                timeout_seconds=None
            )
            chunk_task.complete(chunk)
        
        logging.info(f"Loaded {len(functions)} functions and {len(classes)} classes")
        return {
            "functions": len(functions),
            "classes": len(classes),
            "total_chunks": len(functions) + len(classes)
        }
    
    def retrieve_code(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve code chunks matching the query.
        """
        # Ensure code is loaded
        if not self.code_collection_id:
            self.load_own_code()
        
        # Search for matching code chunks
        results = self.code_manager.search_code(query)
        
        # If no direct matches, try semantic search using embeddings
        if not results and hasattr(self, 'compute_embedding'):
            try:
                # Get all code chunks
                chunks = []
                subtasks = self.memory_store.get_subtasks(self.code_collection_id)
                for subtask in subtasks:
                    if subtask.result:
                        chunks.append(subtask.result)
                
                # Compute embedding for the query
                query_embedding = self.compute_embedding(query)
                
                # Compute embeddings for all chunks
                chunk_embeddings = []
                for chunk in chunks:
                    chunk_embedding = self.compute_embedding(chunk['content'])
                    chunk_embeddings.append((chunk, chunk_embedding))
                
                # Calculate similarity and sort results
                import numpy as np
                from scipy import spatial
                
                scored_chunks = []
                for chunk, embedding in chunk_embeddings:
                    if embedding:
                        similarity = 1 - spatial.distance.cosine(
                            np.array(query_embedding), 
                            np.array(embedding)
                        )
                        scored_chunks.append((chunk, similarity))
                
                # Sort by similarity
                scored_chunks.sort(key=lambda x: x[1], reverse=True)
                
                # Return top results
                results = [chunk for chunk, _ in scored_chunks[:5]]
            except Exception as e:
                logging.error(f"Error in semantic code search: {e}")
        
        return results
    
    def modify_code(self, element_type: str, name: str, new_code: str) -> bool:
        """
        Modify a code element (function or method) with validation.
        
        Args:
            element_type: 'function', 'class', or 'method'
            name: Name of the element to modify (for methods, use 'class_name.method_name')
            new_code: New code to replace the element with
            
        Returns:
            bool: True if modification was successful, False otherwise
        """
        # Create a task for this modification
        task = self.memory_store.create_task(
            priority=1,
            description=f"Modify {element_type} {name}",
            timeout_seconds=30
        )
        task.start()
        
        try:
            if element_type == 'function':
                result = self.code_manager.safe_modify(
                    self.code_manager.replace_function,
                    name, new_code
                )
            elif element_type == 'method':
                # Split class name and method name
                if '.' in name:
                    class_name, method_name = name.split('.', 1)
                    result = self.code_manager.safe_modify(
                        self.code_manager.replace_method,
                        class_name, method_name, new_code
                    )
                else:
                    task.fail(f"Invalid method name format: {name}. Use 'class_name.method_name'")
                    return False
            elif element_type == 'class':
                result = self.code_manager.safe_modify(
                    self.code_manager.replace_function,  # Classes are handled the same way
                    name, new_code
                )
            else:
                task.fail(f"Unknown element type: {element_type}")
                return False
            
            if result:
                # Reload code after successful modification
                self.load_own_code()
                
                task.complete({
                    "status": "success",
                    "element_type": element_type,
                    "name": name,
                    "timestamp": datetime.now().isoformat()
                })
                return True
            else:
                task.fail("Modification failed during validation")
                return False
                
        except Exception as e:
            error_msg = f"Error modifying code: {str(e)}"
            logging.error(error_msg)
            task.fail(error_msg)
            return False
    
    def compute_embedding(self, text: str) -> List[float]:
        """
        Compute an embedding for the given text using the Together API.
        """
        try:
            response = self.client.embeddings.create(
                model="togethercomputer/m2-bert-80M-8k-retrieval",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Error computing embedding: {e}")
            return None
    
    def shutdown(self):
        """
        Cleanly shut down the scheduler, thread pool, resource monitor, etc.
        """
        if hasattr(self, 'resource_monitor'):
            self.resource_monitor.stop_monitoring()
        self.scheduler.stop_scheduler()
        
    def generate_task_visualization(self, output_file: str = "task_graph.html") -> str:
        """
        Generate an interactive visualization of task dependencies and status.
        Returns the path to the generated HTML file.
        """
        try:
            # Get all tasks
            tasks = self.memory_store.list_tasks()
            
            # Create nodes and edges for the graph
            nodes = []
            edges = []
            
            # Create a node for each task
            for task in tasks:
                # Determine node color based on status
                color = {
                    "PENDING": "#FFA500",  # Orange
                    "IN_PROGRESS": "#1E90FF",  # Blue
                    "COMPLETED": "#32CD32",  # Green
                    "FAILED": "#FF0000",  # Red
                    "TIMEOUT": "#8B0000"   # Dark Red
                }.get(task.status, "#808080")  # Gray default
                
                # Create node
                nodes.append({
                    "id": task.task_id,
                    "label": f"Task {task.task_id}",
                    "title": task.description[:50] + ("..." if len(task.description) > 50 else ""),
                    "color": color,
                    "shape": "dot",
                    "size": 10 + (5 * task.progress if task.progress else 0)  # Size based on progress
                })
                
                # Create edge if there's a parent
                if task.parent_id:
                    edges.append({
                        "from": task.parent_id,
                        "to": task.task_id,
                        "arrows": "to"
                    })
            
            # Generate HTML with vis.js
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Task Dependency Visualization</title>
                <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
                <style type="text/css">
                    #mynetwork {{
                        width: 100%;
                        height: 800px;
                        border: 1px solid lightgray;
                    }}
                    body {{
                        font-family: Arial, sans-serif;
                    }}
                    .task-stats {{
                        margin: 20px;
                        padding: 10px;
                        background-color: #f5f5f5;
                        border-radius: 5px;
                    }}
                </style>
            </head>
            <body>
                <h1>Task Dependency Visualization</h1>
                <div class="task-stats">
                    <h2>Task Statistics</h2>
                    <p>Total Tasks: {len(tasks)}</p>
                    <p>Status Summary: {self.memory_store.get_task_summary()}</p>
                    <p>Queue Size: {len(self.task_queue)}</p>
                    <p>Generated at: {datetime.now().isoformat()}</p>
                </div>
                <div id="mynetwork"></div>
                <script type="text/javascript">
                    // Create a network
                    var container = document.getElementById('mynetwork');
                    var data = {{
                        nodes: new vis.DataSet({json.dumps(nodes)}),
                        edges: new vis.DataSet({json.dumps(edges)})
                    }};
                    var options = {{
                        layout: {{
                            hierarchical: {{
                                direction: "UD",
                                sortMethod: "directed",
                                levelSeparation: 150
                            }}
                        }},
                        physics: {{
                            hierarchicalRepulsion: {{
                                centralGravity: 0.0,
                                springLength: 100,
                                springConstant: 0.01,
                                nodeDistance: 120
                            }},
                            solver: 'hierarchicalRepulsion'
                        }},
                        interaction: {{
                            navigationButtons: true,
                            keyboard: true
                        }}
                    }};
                    var network = new vis.Network(container, data, options);
                    
                    // Add click event
                    network.on("click", function(params) {{
                        if (params.nodes.length > 0) {{
                            var nodeId = params.nodes[0];
                            alert("Task " + nodeId + " clicked. See console for details.");
                            console.log("Task details:", data.nodes.get(nodeId));
                        }}
                    }});
                </script>
            </body>
            </html>
            """
            
            # Write to file
            with open(output_file, "w") as f:
                f.write(html)
                
            logger.info(f"[R1Agent] Generated task visualization at {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"[R1Agent] Error generating visualization: {e}")
            return f"Error: {str(e)}"
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about the agent's learning progress."""
        if not self.enable_learning:
            return {"status": "Learning is disabled"}
            
        return {
            "status": "success",
            "conversation_history_length": len(self.conversation_history),
            "function_patterns": {
                pattern: {
                    "count": data["count"],
                    "examples": len(data["prompts"])
                }
                for pattern, data in self.task_success_patterns.items()
            },
            "performance_stats": self.memory_store.get_performance_statistics(),
            "resource_stats": self.resource_monitor.get_resource_report() if hasattr(self, 'resource_monitor') else None
        }

###############################################################################
# CODE MANAGER FOR SELF-MODIFICATION
###############################################################################

class CodeManager:
    """
    Manages the agent's source code with capabilities for:
    - Loading and parsing the agent's own code
    - Retrieving specific functions or classes
    - Modifying code with validation and error handling
    - Creating backups before modifications
    - Restoring from backups if modifications fail
    """
    def __init__(self, file_path: str = None):
        """Initialize with the path to the agent's source code file."""
        self.file_path = file_path or __file__
        self.backup_path = f"{self.file_path}.bak"
        self.lines = []
        self.ast_tree = None
        self.function_index = {}  # Maps function names to line ranges
        self.class_index = {}     # Maps class names to line ranges
        self.load_code()
    
    def load_code(self):
        """Load the agent's source code and parse it."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.lines = f.readlines()
            
            # Parse the code with AST to index functions and classes
            source = "".join(self.lines)
            self.ast_tree = ast.parse(source)
            self._index_code_elements()
            
            logging.info(f"Loaded {len(self.lines)} lines of code from {self.file_path}")
            return True
        except Exception as e:
            logging.error(f"Error loading code: {e}")
            return False
    
    def _index_code_elements(self):
        """Index all functions and classes in the code."""
        self.function_index = {}
        self.class_index = {}
        
        for node in ast.walk(self.ast_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.function_index[node.name] = {
                    'start_line': node.lineno,
                    'end_line': node.end_lineno,
                    'ast_node': node
                }
            elif isinstance(node, ast.ClassDef):
                self.class_index[node.name] = {
                    'start_line': node.lineno,
                    'end_line': node.end_lineno,
                    'ast_node': node,
                    'methods': {}
                }
                # Index methods within the class
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.class_index[node.name]['methods'][item.name] = {
                            'start_line': item.lineno,
                            'end_line': item.end_lineno,
                            'ast_node': item
                        }
    
    def get_function(self, name: str) -> Optional[str]:
        """Get the source code of a specific function."""
        if name in self.function_index:
            info = self.function_index[name]
            start = info['start_line'] - 1  # Convert to 0-based indexing
            end = info['end_line']
            return "".join(self.lines[start:end])
        return None
    
    def get_class(self, name: str) -> Optional[str]:
        """Get the source code of a specific class."""
        if name in self.class_index:
            info = self.class_index[name]
            start = info['start_line'] - 1  # Convert to 0-based indexing
            end = info['end_line']
            return "".join(self.lines[start:end])
        return None
    
    def get_method(self, class_name: str, method_name: str) -> Optional[str]:
        """Get the source code of a specific method within a class."""
        if class_name in self.class_index and method_name in self.class_index[class_name]['methods']:
            info = self.class_index[class_name]['methods'][method_name]
            start = info['start_line'] - 1  # Convert to 0-based indexing
            end = info['end_line']
            return "".join(self.lines[start:end])
        return None
    
    def search_code(self, query: str) -> List[Dict[str, Any]]:
        """Search for code elements matching the query."""
        results = []
        query_lower = query.lower()
        
        # Search functions
        for name, info in self.function_index.items():
            if query_lower in name.lower():
                code = self.get_function(name)
                results.append({
                    'type': 'function',
                    'name': name,
                    'start_line': info['start_line'],
                    'end_line': info['end_line'],
                    'content': code
                })
        
        # Search classes
        for name, info in self.class_index.items():
            if query_lower in name.lower():
                code = self.get_class(name)
                results.append({
                    'type': 'class',
                    'name': name,
                    'start_line': info['start_line'],
                    'end_line': info['end_line'],
                    'content': code
                })
            
            # Search methods within classes
            for method_name, method_info in info['methods'].items():
                if query_lower in method_name.lower():
                    code = self.get_method(name, method_name)
                    results.append({
                        'type': 'method',
                        'name': method_name,
                        'class_name': name,
                        'start_line': method_info['start_line'],
                        'end_line': method_info['end_line'],
                        'content': code
                    })
        
        return results
    
    def backup(self):
        """Create a backup of the current code."""
        try:
            shutil.copy2(self.file_path, self.backup_path)
            logging.info(f"Created backup at {self.backup_path}")
            return True
        except Exception as e:
            logging.error(f"Error creating backup: {e}")
            return False
    
    def restore_backup(self):
        """Restore code from backup."""
        try:
            if os.path.exists(self.backup_path):
                shutil.copy2(self.backup_path, self.file_path)
                self.load_code()  # Reload the code
                logging.info(f"Restored from backup {self.backup_path}")
                return True
            else:
                logging.error("No backup file found")
                return False
        except Exception as e:
            logging.error(f"Error restoring backup: {e}")
            return False
    
    def set_line(self, line_number: int, new_line: str):
        """Replace a specific line with new content."""
        if 0 <= line_number < len(self.lines):
            self.lines[line_number] = new_line if new_line.endswith('\n') else new_line + '\n'
            return True
        return False
    
    def replace_block(self, start_marker: str, end_marker: str, new_block: List[str]):
        """Replace a block of code between markers with new content."""
        start_idx = -1
        end_idx = -1
        
        for i, line in enumerate(self.lines):
            if start_marker in line and start_idx == -1:
                start_idx = i
            elif end_marker in line and start_idx != -1:
                end_idx = i
                break
        
        if start_idx != -1 and end_idx != -1:
            # Ensure new lines end with newline character
            new_block_with_newlines = []
            for line in new_block:
                if not line.endswith('\n'):
                    line += '\n'
                new_block_with_newlines.append(line)
            
            # Replace the block
            self.lines[start_idx:end_idx+1] = new_block_with_newlines
            return True
        
        return False
    
    def replace_function(self, function_name: str, new_code: str) -> bool:
        """Replace a function with new code."""
        if function_name in self.function_index:
            info = self.function_index[function_name]
            start = info['start_line'] - 1
            end = info['end_line']
            
            # Ensure new code ends with newline
            if not new_code.endswith('\n'):
                new_code += '\n'
            
            # Split new code into lines
            new_lines = new_code.splitlines(True)
            
            # Replace the function
            self.lines[start:end] = new_lines
            
            # Reindex the code
            self.load_code()
            return True
        
        return False
    
    def replace_method(self, class_name: str, method_name: str, new_code: str) -> bool:
        """Replace a method within a class with new code."""
        if (class_name in self.class_index and 
            method_name in self.class_index[class_name]['methods']):
            
            info = self.class_index[class_name]['methods'][method_name]
            start = info['start_line'] - 1
            end = info['end_line']
            
            # Ensure new code ends with newline
            if not new_code.endswith('\n'):
                new_code += '\n'
            
            # Split new code into lines
            new_lines = new_code.splitlines(True)
            
            # Replace the method
            self.lines[start:end] = new_lines
            
            # Reindex the code
            self.load_code()
            return True
        
        return False
    
    def save(self) -> bool:
        """Save changes to the file."""
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.writelines(self.lines)
            logging.info(f"Saved changes to {self.file_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving changes: {e}")
            return False
    
    def validate_changes(self) -> bool:
        """Validate that the modified code compiles."""
        try:
            source = "".join(self.lines)
            ast.parse(source)  # Try to parse the modified code
            
            # Try to compile the code
            compile(source, self.file_path, 'exec')
            
            logging.info("Code validation successful")
            return True
        except Exception as e:
            logging.error(f"Code validation failed: {e}")
            return False
    
    def safe_modify(self, modification_func, *args, **kwargs) -> bool:
        """
        Safely modify the code with backup and validation.
        
        Args:
            modification_func: Function that performs the modification
            *args, **kwargs: Arguments to pass to the modification function
            
        Returns:
            bool: True if modification was successful, False otherwise
        """
        # Create backup
        if not self.backup():
            return False
        
        try:
            # Apply modification
            result = modification_func(*args, **kwargs)
            if not result:
                logging.error("Modification function returned False")
                self.restore_backup()
                return False
            
            # Validate changes
            if not self.validate_changes():
                self.restore_backup()
                return False
            
            # Save changes
            if not self.save():
                self.restore_backup()
                return False
            
            return True
        except Exception as e:
            logging.error(f"Error during safe modification: {e}")
            self.restore_backup()
            return False
    
    async def stream_tokens(self, text: str, delay: float = 0.05) -> AsyncGenerator[str, None]:
        """Stream tokens from text with a delay between each token."""
        import asyncio
        tokens = re.split(r"(\s+)", text)
        for token in tokens:
            await asyncio.sleep(delay)
            yield token

###############################################################################
# ADVANCED RESOURCE MONITORING AND OPTIMIZATION
###############################################################################

class ResourceMonitor:
    """
    Monitors and optimizes resource usage based on task complexity and system load.
    Features:
    - Dynamic worker allocation based on task complexity
    - CPU/Memory usage monitoring
    - Task prioritization based on resource availability
    - Throttling for resource-intensive tasks
    """
    
    def __init__(self, scheduler: "TaskScheduler"):
        self.scheduler = scheduler
        self.resource_history = deque(maxlen=100)
        self.task_complexity_cache = {}
        self.monitoring_interval = 5  # seconds
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start the resource monitoring thread."""
        self.monitoring_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitoring_thread.start()
        logger.info("[ResourceMonitor] Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop the resource monitoring thread."""
        if self.monitoring_thread:
            self.stop_event.set()
            self.monitoring_thread.join(timeout=2)
            logger.info("[ResourceMonitor] Resource monitoring stopped")
    
    def _monitor_resources(self):
        """Continuously monitor system resources."""
        while not self.stop_event.is_set():
            try:
                # Get current CPU and memory usage
                cpu_usage = self._get_cpu_usage()
                memory_usage = self._get_memory_usage()
                
                # Record resource usage
                self.resource_history.append({
                    "timestamp": datetime.now(),
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "active_tasks": len(self.scheduler.memory_store.get_in_progress_tasks()),
                    "queue_size": len(self.scheduler.task_queue)
                })
                
                # Adjust resources if needed
                self._adjust_resources(cpu_usage, memory_usage)
                
                # Sleep for the monitoring interval
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"[ResourceMonitor] Error monitoring resources: {e}")
                time.sleep(self.monitoring_interval)
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.5)
        except ImportError:
            # Fallback if psutil is not available
            try:
                if sys.platform == "darwin":  # macOS
                    cmd = "top -l 1 | grep 'CPU usage'"
                    output = subprocess.check_output(cmd, shell=True).decode()
                    cpu_usage = float(output.split("user")[0].split("%")[-2].strip())
                    return cpu_usage
                elif sys.platform == "linux":
                    cmd = "top -bn1 | grep 'Cpu(s)'"
                    output = subprocess.check_output(cmd, shell=True).decode()
                    cpu_usage = float(output.split()[1])
                    return cpu_usage
                else:
                    return 50.0  # Default value if platform not supported
            except:
                return 50.0  # Default fallback
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            # Fallback if psutil is not available
            try:
                if sys.platform == "darwin":  # macOS
                    cmd = "vm_stat | grep 'Pages free'"
                    output = subprocess.check_output(cmd, shell=True).decode()
                    # This is a very rough approximation
                    return 50.0  # Default for macOS without psutil
                elif sys.platform == "linux":
                    cmd = "free | grep Mem"
                    output = subprocess.check_output(cmd, shell=True).decode()
                    total = float(output.split()[1])
                    used = float(output.split()[2])
                    return (used / total) * 100
                else:
                    return 50.0  # Default value if platform not supported
            except:
                return 50.0  # Default fallback
    
    def _adjust_resources(self, cpu_usage: float, memory_usage: float):
        """Adjust resource allocation based on current usage."""
        # If system is under heavy load, reduce worker count
        if cpu_usage > 80 or memory_usage > 80:
            current_workers = self.scheduler._executor._max_workers
            if current_workers > 2:
                new_workers = max(2, current_workers - 1)
                self._resize_thread_pool(new_workers)
                logger.info(f"[ResourceMonitor] Reduced worker count to {new_workers} due to high system load")
        
        # If system has available resources, increase worker count
        elif cpu_usage < 30 and memory_usage < 50:
            current_workers = self.scheduler._executor._max_workers
            if current_workers < 8:  # Cap at 8 workers
                new_workers = min(8, current_workers + 1)
                self._resize_thread_pool(new_workers)
                logger.info(f"[ResourceMonitor] Increased worker count to {new_workers} due to available resources")
    
    def _resize_thread_pool(self, new_size: int):
        """Resize the thread pool to the new size."""
        # Create a new executor with the desired size
        new_executor = ThreadPoolExecutor(max_workers=new_size)
        
        # Copy over any pending tasks
        old_executor = self.scheduler._executor
        
        # Update the scheduler's executor
        self.scheduler._executor = new_executor
        
        # Shutdown the old executor without waiting for tasks to complete
        # (they'll continue running in the background)
        old_executor.shutdown(wait=False)
    
    def estimate_task_complexity(self, task: Task) -> float:
        """Estimate the computational complexity of a task."""
        # Check cache first
        if task.task_id in self.task_complexity_cache:
            return self.task_complexity_cache[task.task_id]
        
        # Simple heuristic based on keywords in the description
        complexity = 1.0  # Base complexity
        
        # Keywords that indicate higher complexity
        complexity_keywords = {
            "calculate": 1.2,
            "compute": 1.2,
            "generate": 1.3,
            "simulate": 1.5,
            "matrix": 1.8,
            "eigenvalues": 2.0,
            "determinant": 1.7,
            "inversion": 1.7,
            "graph": 1.6,
            "shortest path": 1.7,
            "traveling salesman": 2.5,
            "genetic algorithm": 2.0,
            "neural network": 2.5,
            "encryption": 1.5,
            "decryption": 1.5,
            "corpus": 1.8,
            "frequency distribution": 1.6,
            "tf-idf": 1.7
        }
        
        description_lower = task.description.lower()
        for keyword, multiplier in complexity_keywords.items():
            if keyword in description_lower:
                complexity *= multiplier
        
        # Cap complexity at 5.0
        complexity = min(5.0, complexity)
        
        # Cache the result
        self.task_complexity_cache[task.task_id] = complexity
        
        return complexity
    
    def get_resource_report(self) -> Dict[str, Any]:
        """Generate a report of resource usage over time."""
        if not self.resource_history:
            return {"status": "No resource data available"}
        
        # Calculate averages
        cpu_values = [entry["cpu_usage"] for entry in self.resource_history]
        memory_values = [entry["memory_usage"] for entry in self.resource_history]
        active_tasks = [entry["active_tasks"] for entry in self.resource_history]
        queue_sizes = [entry["queue_size"] for entry in self.resource_history]
        
        return {
            "status": "success",
            "data_points": len(self.resource_history),
            "time_range": {
                "start": self.resource_history[0]["timestamp"].isoformat(),
                "end": self.resource_history[-1]["timestamp"].isoformat()
            },
            "cpu_usage": {
                "current": cpu_values[-1],
                "average": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values)
            },
            "memory_usage": {
                "current": memory_values[-1],
                "average": sum(memory_values) / len(memory_values),
                "max": max(memory_values),
                "min": min(memory_values)
            },
            "task_metrics": {
                "current_active": active_tasks[-1],
                "average_active": sum(active_tasks) / len(active_tasks),
                "current_queue": queue_sizes[-1],
                "average_queue": sum(queue_sizes) / len(queue_sizes)
            },
            "worker_count": self.scheduler._executor._max_workers
        }

###############################################################################
# MAIN DEMO
###############################################################################

def create_evaluation_tasks(agent: R1Agent) -> None:
    """
    Create an advanced set of initial tasks to demonstrate the agent's enhanced real-world capabilities.
    These tasks are designed to test multi-step reasoning, dependency management, data analysis,
    financial modeling, legal analysis, AI ethics, real-time visualization, and self-modification.
    """
    # ===== ADVANCED NEWS AND CURRENT EVENTS TASKS =====
    
    # Task 1: Advanced News Analysis with Sentiment and Factual Verification
    agent.task_queue.push(agent.memory_store.create_task(
        priority=4,
        description="'Search for international news on global warming, perform sentiment analysis, and fact-check key claims.'",
        timeout_seconds=90
    ))
    
    # Task 2: Multi-Layer Research Compilation with Subtasks
    agent.task_queue.push(agent.memory_store.create_task(
        priority=5,
        description="Subtask(4)=1) 'Retrieve academic papers on AI ethics post-2022' 2) 'Extract key findings and controversies' 3) 'Summarize expert opinions' 4) 'Compile a comprehensive research report on AI ethics.'",
        timeout_seconds=120
    ))
    
    # ===== ADVANCED DATA ANALYSIS AND VISUALIZATION TASKS =====
    
    # Task 3: Real-Time Sports Analytics with Visualization
    agent.task_queue.push(agent.memory_store.create_task(
        priority=6,
        description="'Collect current NBA player stats, compute advanced metrics (PER, TS%), and generate interactive visualizations of player performance trends.'",
        timeout_seconds=90
    ))
    
    # Task 4: Financial Modeling and Sensitivity Analysis
    agent.task_queue.push(agent.memory_store.create_task(
        priority=6,
        description="'Build a detailed DCF model for a selected tech company, perform sensitivity analysis on key assumptions, and provide investment recommendations.'",
        timeout_seconds=120
    ))
    
    # ===== ADVANCED TECH & AI TASKS =====
    
    # Task 5: Generative AI Deep Dive
    agent.task_queue.push(agent.memory_store.create_task(
        priority=7,
        description="'Compare leading generative AI models for text, image, and code generation; analyze performance metrics and produce a detailed comparative report.'",
        timeout_seconds=90
    ))
    
    # Task 6: AI Ethics and Safety Analysis
    agent.task_queue.push(agent.memory_store.create_task(
        priority=7,
        description="'Research and compile a comprehensive report on AI safety standards and ethical frameworks across major jurisdictions, highlighting key risks and proposed safeguards.'",
        timeout_seconds=120
    ))
    
    # ===== ADVANCED LEGAL AND REGULATORY ANALYSIS TASKS =====
    
    # Task 7: Global Data Privacy Compliance Mapping
    agent.task_queue.push(agent.memory_store.create_task(
        priority=8,
        description="'Analyze global data privacy regulations (GDPR, CCPA, PIPL, etc.), compare their key provisions, and create a compliance framework for multinational businesses.'",
        timeout_seconds=90
    ))
    
    # Task 8: Cryptocurrency Regulatory Landscape Analysis with Case Studies
    agent.task_queue.push(agent.memory_store.create_task(
        priority=8,
        description="'Compile a detailed analysis of cryptocurrency regulations in major markets, including case studies of enforcement actions and implications for crypto businesses.'",
        timeout_seconds=90
    ))
    
    # ===== ADVANCED ENVIRONMENTAL AND SUSTAINABILITY TASKS =====
    
    # Task 9: Environmental Impact and Sustainability Analysis
    agent.task_queue.push(agent.memory_store.create_task(
        priority=9,
        description="'Analyze carbon emission data across industries, evaluate sustainability initiatives, and develop an implementation roadmap for achieving net-zero targets.'",
        timeout_seconds=90
    ))
    
    # Task 10: Smart City Infrastructure and Energy Efficiency Study
    agent.task_queue.push(agent.memory_store.create_task(
        priority=9,
        description="'Research smart city initiatives with a focus on energy efficiency, public transportation innovation, and digital governance; produce a detailed comparative study.'",
        timeout_seconds=90
    ))
    
    # ===== ADVANCED SELF-MODIFICATION AND LEARNING TASKS =====
    
    # Task 11: Autonomous Code Optimization and Self-Improvement Test
    agent.task_queue.push(agent.memory_store.create_task(
        priority=10,
        description="'Retrieve and analyze your own codebase, propose and implement optimizations, and report on performance improvements using a self-modification loop.'",
        timeout_seconds=120
    ))
    
    # Task 12: Real-Time Task Visualization and Resource Monitoring
    agent.task_queue.push(agent.memory_store.create_task(
        priority=10,
        description="'Generate an interactive visualization of current task execution and resource utilization; provide insights into system performance and bottlenecks.'",
        timeout_seconds=90
    ))
    
    # ===== ADVANCED CROSS-DOMAIN MULTI-TASK TEST =====
    
    # Task 13: End-to-End Multi-Domain Challenge
    agent.task_queue.push(agent.memory_store.create_task(
        priority=10,
        description="Subtask(5)=1) 'Fetch global news on renewable energy adoption' 2) 'Analyze public sentiment using NLP techniques' 3) 'Compile financial data of top renewable energy companies' 4) 'Perform competitive analysis of clean tech market' 5) 'Create a unified dashboard integrating news trends, financial analysis, and market forecasts.'",
        timeout_seconds=150
    ))
    
    logger.info(f"[Evaluation] Created 13 advanced initial tasks covering multi-step research, data analysis, financial modeling, technical deep dives, legal regulation, sustainability, and self-improvement challenges.")
        description="'Compile a research report on mRNA vaccine technology advancements since 2020'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=8,
        description="Subtask(4)=1) 'Find academic papers on quantum computing published in 2023' 2) 'Extract key findings from each paper' 3) 'Identify common research themes' 4) 'Create an executive summary of the state of quantum computing research'",
        timeout_seconds=120
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=9,
        description="'Research the environmental impact of electric vehicles vs. traditional vehicles, including manufacturing and lifecycle emissions'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=9,
        description="Subtask(3)=1) 'Find studies on the effectiveness of remote work on productivity' 2) 'Compare findings across different industries' 3) 'Compile best practices for hybrid work models based on research'",
        timeout_seconds=75
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=10,
        description="'Create a literature review on artificial general intelligence safety measures and ethical frameworks'",
        timeout_seconds=120
    ))
    
    # ===== SPORTS STATISTICS AND ANALYSIS =====
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=11,
        description="'Analyze NBA player statistics for the current season and identify the top 10 MVP candidates based on advanced metrics'",
        timeout_seconds=60
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=11,
        description="Subtask(3)=1) 'Collect NFL quarterback performance data for the past 5 seasons' 2) 'Calculate efficiency metrics and trend analysis' 3) 'Create visualizations of performance evolution for top quarterbacks'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=12,
        description="'Compare Premier League team statistics and create power rankings based on expected goals, possession metrics, and defensive solidity'",
        timeout_seconds=75
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=12,
        description="Subtask(4)=1) 'Gather MLB pitcher statistics for current season' 2) 'Analyze strikeout rates, walk rates, and advanced metrics' 3) 'Identify undervalued pitchers based on peripheral statistics' 4) 'Create a fantasy baseball pitcher ranking'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=13,
        description="'Analyze historical Olympic medal counts by country and identify trends in performance across Summer and Winter games'",
        timeout_seconds=60
    ))
    
    # ===== FINANCIAL ANALYSIS AND MARKET RESEARCH =====
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=14,
        description="'Perform fundamental analysis on the top 5 tech stocks by market cap and create investment recommendations'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=14,
        description="Subtask(3)=1) 'Collect historical cryptocurrency price data for Bitcoin, Ethereum, and Solana' 2) 'Calculate volatility metrics and correlation with traditional markets' 3) 'Create a risk assessment report for crypto investments'",
        timeout_seconds=75
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=15,
        description="'Analyze housing market trends in major US cities and identify potential investment opportunities based on price-to-rent ratios'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=15,
        description="Subtask(4)=1) 'Gather quarterly earnings reports for S&P 500 companies' 2) 'Identify sectors with strongest revenue growth' 3) 'Analyze profit margin trends' 4) 'Create a sector rotation investment strategy'",
        timeout_seconds=120
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=16,
        description="'Build a discounted cash flow (DCF) model for Tesla and estimate fair value based on different growth scenarios'",
        timeout_seconds=90
    ))
    
    # ===== HEALTH AND MEDICAL RESEARCH =====
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=17,
        description="'Research the latest clinical trials for Alzheimer's disease treatments and summarize promising approaches'",
        timeout_seconds=75
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=17,
        description="Subtask(3)=1) 'Find nutrition studies on intermittent fasting' 2) 'Compare health outcomes across different fasting protocols' 3) 'Create evidence-based recommendations for different health goals'",
        timeout_seconds=60
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=18,
        description="'Analyze global COVID-19 vaccination data and identify correlations with case rates, hospitalizations, and mortality'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=18,
        description="Subtask(3)=1) 'Research mental health impacts of social media usage in teenagers' 2) 'Identify risk factors and protective factors' 3) 'Compile evidence-based recommendations for healthy social media use'",
        timeout_seconds=75
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=19,
        description="'Create a comprehensive guide to exercise science, including optimal training frequencies, intensities, and recovery protocols for different fitness goals'",
        timeout_seconds=90
    ))
    
    # ===== TECHNOLOGY TREND ANALYSIS =====
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=20,
        description="'Analyze the current state of generative AI tools and create a comparison of leading models for text, image, and code generation'",
        timeout_seconds=75
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=20,
        description="Subtask(3)=1) 'Research edge computing adoption across industries' 2) 'Identify key use cases and implementation challenges' 3) 'Create a forecast for edge computing growth over the next 5 years'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=21,
        description="'Analyze the semiconductor supply chain, current bottlenecks, and geopolitical factors affecting chip production'",
        timeout_seconds=75
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=21,
        description="Subtask(4)=1) 'Research Web3 and blockchain adoption in enterprise' 2) 'Identify successful implementation case studies' 3) 'Analyze challenges and limitations' 4) 'Create a roadmap for enterprise blockchain integration'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=22,
        description="'Create a comprehensive overview of quantum computing applications in cryptography, drug discovery, and financial modeling'",
        timeout_seconds=90
    ))
    
    # ===== BUSINESS AND COMPETITIVE ANALYSIS =====
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=23,
        description="'Perform a SWOT analysis of Netflix's business model and competitive position in the streaming industry'",
        timeout_seconds=75
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=23,
        description="Subtask(3)=1) 'Research the electric vehicle market landscape' 2) 'Compare product offerings, pricing, and features across major manufacturers' 3) 'Create a competitive positioning map of the EV market'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=24,
        description="'Analyze the food delivery app ecosystem, market share distribution, and profitability challenges'",
        timeout_seconds=75
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=24,
        description="Subtask(4)=1) 'Research the creator economy platforms and business models' 2) 'Compare monetization options across platforms' 3) 'Analyze audience growth strategies' 4) 'Create a guide for content creators to maximize revenue'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=25,
        description="'Create a market entry strategy for a SaaS product in the project management space, including competitor analysis and differentiation opportunities'",
        timeout_seconds=90
    ))
    
    # ===== EDUCATIONAL CONTENT CREATION =====
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=26,
        description="'Create a comprehensive learning path for mastering machine learning, from fundamentals to advanced techniques'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=26,
        description="Subtask(3)=1) 'Develop a curriculum for teaching financial literacy to high school students' 2) 'Create lesson plans with practical exercises' 3) 'Design assessment methods to measure understanding'",
        timeout_seconds=75
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=27,
        description="'Create an interactive tutorial series on web development, covering HTML, CSS, JavaScript, and modern frameworks'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=27,
        description="Subtask(4)=1) 'Research effective language learning techniques' 2) 'Create a structured Spanish learning program for beginners' 3) 'Develop vocabulary lists and grammar exercises' 4) 'Design conversation practice scenarios'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=28,
        description="'Develop a comprehensive guide to digital marketing, including SEO, content marketing, social media, and analytics'",
        timeout_seconds=90
    ))
    
    # ===== TRAVEL AND CULTURAL RESEARCH =====
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=29,
        description="'Create a detailed 2-week travel itinerary for Japan, including cultural experiences, historical sites, and culinary recommendations'",
        timeout_seconds=75
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=29,
        description="Subtask(3)=1) 'Research sustainable tourism practices in popular destinations' 2) 'Identify eco-friendly accommodations and activities' 3) 'Create a guide to responsible travel in environmentally sensitive areas'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=30,
        description="'Compare cultural differences in business practices between Western, Asian, and Middle Eastern countries'",
        timeout_seconds=75
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=30,
        description="Subtask(4)=1) 'Research traditional cuisines from different regions of Italy' 2) 'Document authentic recipes and techniques' 3) 'Explore historical influences on regional dishes' 4) 'Create a culinary tour map of Italy'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=31,
        description="'Create a guide to digital nomad-friendly cities, comparing cost of living, internet infrastructure, visa policies, and quality of life'",
        timeout_seconds=90
    ))
    
    # ===== LEGAL AND REGULATORY ANALYSIS =====
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=32,
        description="'Analyze global AI regulation approaches and create a comparison of policy frameworks in the US, EU, and China'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=32,
        description="Subtask(3)=1) 'Research data privacy laws across different jurisdictions' 2) 'Compare GDPR, CCPA, and other privacy frameworks' 3) 'Create compliance guidelines for global businesses'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=33,
        description="'Analyze cryptocurrency regulations worldwide and identify jurisdictions with favorable regulatory environments'",
        timeout_seconds=75
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=33,
        description="Subtask(3)=1) 'Research intellectual property protection for AI-generated content' 2) 'Analyze recent court cases and legal opinions' 3) 'Create guidelines for creators using generative AI tools'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=34,
        description="'Create a comprehensive guide to starting a business, including legal structures, tax considerations, and regulatory compliance requirements'",
        timeout_seconds=90
    ))
    
    # ===== ENVIRONMENTAL AND SUSTAINABILITY RESEARCH =====
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=35,
        description="'Analyze carbon footprint reduction strategies for businesses and create an implementation roadmap for net-zero emissions'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=35,
        description="Subtask(4)=1) 'Research circular economy business models' 2) 'Identify successful case studies across industries' 3) 'Analyze implementation challenges' 4) 'Create a transition framework for linear to circular operations'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=36,
        description="'Compare renewable energy adoption rates across countries and analyze policy factors driving successful transitions'",
        timeout_seconds=75
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=36,
        description="Subtask(3)=1) 'Research sustainable agriculture practices' 2) 'Compare yields and environmental impacts with conventional farming' 3) 'Create implementation guidelines for farmers transitioning to sustainable methods'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=37,
        description="'Analyze plastic pollution data and evaluate the effectiveness of different mitigation strategies, from policy to technology solutions'",
        timeout_seconds=90
    ))
    
    # ===== SOCIAL MEDIA AND CONTENT STRATEGY =====
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=38,
        description="'Analyze successful TikTok marketing campaigns and extract best practices for brand engagement'",
        timeout_seconds=75
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=38,
        description="Subtask(3)=1) 'Research content performance metrics across different social platforms' 2) 'Identify optimal posting times and frequencies' 3) 'Create a cross-platform content strategy template'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=39,
        description="'Analyze podcast growth strategies and create a launch plan for a new business podcast'",
        timeout_seconds=75
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=39,
        description="Subtask(4)=1) 'Research YouTube algorithm optimization techniques' 2) 'Analyze successful channel growth case studies' 3) 'Identify content formats with highest engagement' 4) 'Create a YouTube channel growth strategy'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=40,
        description="'Create a comprehensive social media crisis management playbook with response templates and workflow procedures'",
        timeout_seconds=75
    ))
    
    # ===== PRODUCT RESEARCH AND REVIEWS =====
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=41,
        description="'Compare the latest flagship smartphones based on performance, camera quality, battery life, and value'",
        timeout_seconds=75
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=41,
        description="Subtask(3)=1) 'Research home fitness equipment options' 2) 'Compare features, space requirements, and effectiveness' 3) 'Create buying guides for different fitness goals and budgets'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=42,
        description="'Analyze streaming service offerings and create a comparison of content libraries, pricing, and exclusive features'",
        timeout_seconds=75
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=42,
        description="Subtask(4)=1) 'Research noise-cancelling headphone technology' 2) 'Compare top models across price points' 3) 'Test performance in different environments' 4) 'Create a detailed buyer's guide with recommendations'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=43,
        description="'Create a comprehensive guide to electric vehicles, comparing range, charging infrastructure, performance, and total cost of ownership'",
        timeout_seconds=90
    ))
    
    # ===== PERSONAL DEVELOPMENT AND PRODUCTIVITY =====
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=44,
        description="'Research evidence-based productivity techniques and create a personalized system for deep work and focus'",
        timeout_seconds=75
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=44,
        description="Subtask(3)=1) 'Research habit formation psychology' 2) 'Identify effective implementation techniques' 3) 'Create a 30-day habit building framework with tracking tools'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=45,
        description="'Analyze different meditation techniques and their benefits for stress reduction, focus, and emotional regulation'",
        timeout_seconds=75
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=45,
        description="Subtask(4)=1) 'Research effective learning techniques based on cognitive science' 2) 'Compare spaced repetition, active recall, and other methods' 3) 'Analyze digital learning tools and apps' 4) 'Create a personalized learning system'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=46,
        description="'Create a comprehensive career development plan template with skill assessment, industry research, and advancement strategies'",
        timeout_seconds=90
    ))
    
    # ===== DATA ANALYSIS AND VISUALIZATION =====
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=47,
        description="'Analyze global climate data trends and create visualizations showing temperature changes by region'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=47,
        description="Subtask(3)=1) 'Collect demographic data for major US cities' 2) 'Analyze population trends, income distribution, and housing costs' 3) 'Create interactive visualizations showing correlations between factors'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=48,
        description="'Analyze e-commerce sales data to identify seasonal trends, popular product categories, and customer behavior patterns'",
        timeout_seconds=75
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=48,
        description="Subtask(4)=1) 'Gather public health data on obesity rates' 2) 'Analyze correlations with socioeconomic factors' 3) 'Map geographic distribution of health outcomes' 4) 'Create a dashboard visualizing key findings'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=49,
        description="'Create a comprehensive analysis of global energy consumption patterns, renewable adoption, and projected future trends'",
        timeout_seconds=90
    ))
    
    # ===== CREATIVE WRITING AND CONTENT GENERATION =====
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=50,
        description="'Create a detailed science fiction short story set in a post-AI singularity world, exploring ethical and philosophical themes'",
        timeout_seconds=75
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=50,
        description="Subtask(3)=1) 'Develop character profiles for a fantasy novel' 2) 'Create a detailed world-building guide with magic systems and cultural backgrounds' 3) 'Write a compelling first chapter introducing the main conflict'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=51,
        description="'Write a comprehensive business plan for a fictional sustainable fashion startup, including market analysis and financial projections'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=51,
        description="Subtask(4)=1) 'Research historical events in 1920s New York' 2) 'Develop characters based on the era' 3) 'Create a detailed setting description' 4) 'Write a historical fiction scene capturing the atmosphere of the Jazz Age'",
        timeout_seconds=90
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=52,
        description="'Create a detailed screenplay for a short film exploring the theme of human connection in a digital age'",
        timeout_seconds=90
    ))
    
    logger.info(f"[Evaluation] Created 100 real-world tasks covering news analysis, research compilation, sports statistics, financial modeling, health research, technology trends, business analysis, educational content, travel planning, legal analysis, environmental research, social media strategy, product reviews, personal development, data visualization, and creative writing")


def main():
    """
    Demonstration of the advanced R1 agent with enhanced capabilities.

    - We create an instance of R1Agent with learning and optimization enabled
    - We pass in complex prompts that demonstrate the agent's capabilities
    - We add evaluation tasks to test the system thoroughly
    - We generate visualizations of task execution
    - We display learning statistics and resource usage
    - Then we gracefully shut down
    """
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Advanced R1 Agent Demo")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation tasks")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--learning", action="store_true", default=True, help="Enable reinforcement learning")
    parser.add_argument("--visualize", action="store_true", default=True, help="Generate task visualizations")
    parser.add_argument("--persistent", action="store_true", default=False, help="Enable persistent memory")
    args = parser.parse_args()

    # Create the agent with specified options
    agent = R1Agent(
        max_workers=args.workers,
        persistent_memory=args.persistent,
        enable_learning=args.learning
    )

    try:
        # Add evaluation tasks if requested
        if args.evaluate:
            create_evaluation_tasks(agent)
            print(f"Added evaluation tasks to the queue.")
        
        user_prompt = (
            "Hello advanced agent. I want you to demonstrate your enhanced capabilities. "
            "First, execute some code: <function_call> do_anything: <code>import sys; print(f'Python version: {sys.version}')</code> </function_call>\n\n"
            "Then, break down a complex task into steps with proper dependency tracking. "
            "Subtask(3)=1) 'Fetch https://example.com' 2) 'Analyze the HTML structure' 3) 'Summarize the content with key insights'."
        )
        response = agent.generate_response(user_prompt)
        print("\n================ AGENT RESPONSE (IMMEDIATE) ================\n")
        print(response)
        print("\n=============================================================\n")

        # Let tasks run in background, showing progress updates
        print("Waiting for tasks to complete in background...\n")
        
        # Wait up to 300 seconds for tasks to complete, showing status every 3 seconds
        start_time = time.time()
        max_wait_time = 300 if args.evaluate else 60
        
        while time.time() - start_time < max_wait_time:
            # Get task summary
            status = agent.get_task_status()
            
            # Print current status with enhanced formatting
            print(f"\n\033[1mTask Status at {time.strftime('%H:%M:%S')}:\033[0m")
            print(f"  Total tasks: {status['task_count']}")
            print(f"  Status summary: {status['status_summary']}")
            print(f"  Queue size: {status['queue_size']}")
            
            # Print details of recent tasks with color coding
            print("\n\033[1mRecent tasks:\033[0m")
            for task in status['recent_tasks']:
                # Color based on status
                color_code = {
                    "PENDING": "\033[33m",     # Yellow
                    "IN_PROGRESS": "\033[34m", # Blue
                    "COMPLETED": "\033[32m",   # Green
                    "FAILED": "\033[31m",      # Red
                    "TIMEOUT": "\033[31m"      # Red
                }.get(task['status'], "\033[0m")
                
                print(f"  {color_code}Task {task['task_id']} - {task['status']} - {task['description']}\033[0m")
            
            # Check if all tasks are done
            if status['status_summary'].get('PENDING', 0) == 0 and \
               status['status_summary'].get('IN_PROGRESS', 0) == 0 and \
               status['queue_size'] == 0:
                print("\n\033[1;32mAll tasks completed!\033[0m")
                break
                
            # Sleep before checking again
            time.sleep(3)

        # Generate visualization if requested
        if args.visualize:
            viz_file = agent.generate_task_visualization("task_visualization.html")
            print(f"\n\033[1mTask visualization generated:\033[0m {viz_file}")
            
            # Try to open the visualization in a browser
            try:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(viz_file)}")
                print("Visualization opened in browser.")
            except:
                print("Please open the visualization file manually.")
        
        # Display learning statistics if enabled
        if args.learning:
            learning_stats = agent.get_learning_statistics()
            print("\n\033[1mLearning Statistics:\033[0m")
            print(json.dumps(learning_stats, indent=2))

        # Print final memory state with detailed results
        print("\n\033[1mFinal task results:\033[0m")
        for task in agent.memory_store.list_tasks():
            # Color based on status
            color_code = {
                "PENDING": "\033[33m",     # Yellow
                "IN_PROGRESS": "\033[34m", # Blue
                "COMPLETED": "\033[32m",   # Green
                "FAILED": "\033[31m",      # Red
                "TIMEOUT": "\033[31m"      # Red
            }.get(task.status, "\033[0m")
            
            print(f"  {color_code}{task}\033[0m")
            
            # For completed tasks with results, show a summary
            if task.status == "COMPLETED" and task.result:
                if isinstance(task.result, dict):
                    if "content" in task.result and isinstance(task.result["content"], str):
                        content = task.result["content"]
                        print(f"    Result: {content[:100]}..." if len(content) > 100 else content)
                    elif "summary" in task.result:
                        print(f"    Summary: {task.result['summary'][:100]}...")
                    else:
                        print(f"    Result status: {task.result.get('status', 'unknown')}")

    finally:
        agent.shutdown()
        print("\n\033[1mAgent shut down cleanly.\033[0m")


if __name__ == "__main__":
    main()
