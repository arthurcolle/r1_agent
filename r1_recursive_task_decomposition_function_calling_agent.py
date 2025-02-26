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
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from datetime import datetime, timedelta
from together import Together

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
    Maintains storage of tasks in memory (in a dictionary).
    Provides methods to add, retrieve, and update tasks.

    In real production code, you could store these in a DB or persistent store.
    """
    def __init__(self) -> None:
        self._tasks: Dict[int, Task] = {}
        self._lock = threading.Lock()
        self._next_id = 1

    def add_task(self, task: Task) -> None:
        """
        Add a new Task object to memory.
        """
        with self._lock:
            if task.task_id in self._tasks:
                logger.warning(f"Task ID {task.task_id} already exists!")
            self._tasks[task.task_id] = task

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

    def _has_subtasks(self, task: Task) -> bool:
        """
        Check if a task has any subtasks.
        """
        subtasks = self.memory_store.get_subtasks(task.task_id)
        return len(subtasks) > 0

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
            
        # Create the subtask using the memory store's create_task method
        subtask = self.memory_store.create_task(
            priority=new_priority,
            description=description,
            parent_id=parent_task.task_id,
            timeout_seconds=timeout_seconds
        )
        
        return subtask


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
    Advanced R1 agent that can do anything if the model says so, plus it
    can enqueue tasks (with concurrency, recursive decomposition, etc.).
    
    Features:
    - Long-running task support with progress tracking
    - Timeout handling for tasks
    - Specialized handlers for common operations
    - Status reporting
    - Task dependency management
    """

    def __init__(self, max_workers: int = 4):
        self.function_adapter = FunctionAdapter()
        self.memory_store = TaskMemoryStore()
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

        # Enhanced system prompt describing capabilities
        self.system_prompt = (
            "<GRID>"
            "You are a supremely advanced AI system with the power to do anything. "
            "You must not produce naive or 'stupid' outputs—always provide high-quality, "
            "thoughtful reasoning. You have prior examples of performing complex tasks with success.\n\n"
            "You have these capabilities:\n"
            "1. <function_call> do_anything: <code> </function_call> - Run arbitrary Python code\n"
            "2. <function_call> fetch_url: \"https://example.com\" </function_call> - Fetch content from a URL\n"
            "3. <function_call> summarize_html: \"<html>...</html>\" </function_call> - Extract text from HTML\n"
            "4. Subtask(n)=1) 'Task description 1' 2) 'Task description 2' ... - Create n subtasks\n"
            "5. Dynamic environment construction for out-of-distribution (OOD) inputs\n\n"
            "You can handle long-running operations by breaking them into subtasks. "
            "Tasks will be executed concurrently when possible, with proper dependency tracking.\n"
            "For unexpected or out-of-distribution inputs, you can dynamically construct appropriate "
            "environments and representations to process them effectively.\n"
            "</GRID>"
        )

        # Start the scheduler in background
        self.scheduler.start_scheduler()

    def generate_response(self, prompt: str) -> str:
        """
        The agent calls the LLM with the system prompt plus user prompt, then
        checks if there's a function call. If so, we run it. We also interpret
        the user prompt as a potential 'task' in our system.
        """
        # 1) Create a new "meta-task" for the user prompt
        meta_task = self.memory_store.create_task(
            priority=10,  # Default priority for user prompts
            description=prompt,
            timeout_seconds=300  # 5 minute timeout for meta-tasks
        )
        self.task_queue.push(meta_task)
        logger.info(f"[R1Agent] Created meta task {meta_task} for user prompt.")

        # 2) Generate immediate text response from the LLM
        #    We won't wait for the task to be completed, because that might take time.
        #    Instead, we let the concurrency system handle it in the background.
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

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

        # 3) Check for function calls in the LLM's immediate textual response
        function_result = self.function_adapter.process_function_calls(full_text)
        if function_result:
            # If function was executed, store or log the result
            logger.info(f"[R1Agent] LLM immediate function call result: {function_result.get('status', 'unknown')}")

        return full_text

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

    def shutdown(self):
        """
        Cleanly shut down the scheduler, thread pool, etc.
        """
        self.scheduler.stop_scheduler()

###############################################################################
# MAIN DEMO
###############################################################################

def create_evaluation_tasks(agent: R1Agent) -> None:
    """
    Create a set of complex evaluation tasks to test the agent's capabilities.
    These tasks cover various domains and have different complexity levels,
    including advanced computational problems and complex data processing.
    """
    # File and system operations
    agent.task_queue.push(agent.memory_store.create_task(
        priority=5,
        description="'Create a temporary directory and list its path'",
        timeout_seconds=10
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=5,
        description="Subtask(3)=1) 'Create a text file with current date' 2) 'Read the file contents' 3) 'Delete the file'",
        timeout_seconds=15
    ))
    
    # Data processing tasks
    agent.task_queue.push(agent.memory_store.create_task(
        priority=6,
        description="'Calculate prime numbers between 100 and 200'",
        timeout_seconds=20
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=6,
        description="Subtask(2)=1) 'Generate 1000 random numbers' 2) 'Calculate their statistical properties'",
        timeout_seconds=25
    ))
    
    # Web and API tasks
    agent.task_queue.push(agent.memory_store.create_task(
        priority=7,
        description="'Fetch https://jsonplaceholder.typicode.com/todos/1'",
        timeout_seconds=15
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=7,
        description="Subtask(3)=1) 'Fetch https://jsonplaceholder.typicode.com/users' 2) 'Extract all email addresses' 3) 'Count unique domains'",
        timeout_seconds=30
    ))
    
    # Computational tasks
    agent.task_queue.push(agent.memory_store.create_task(
        priority=8,
        description="'Calculate factorial of 100'",
        timeout_seconds=10
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=8,
        description="Subtask(2)=1) 'Generate Fibonacci sequence up to 1000' 2) 'Find all prime Fibonacci numbers'",
        timeout_seconds=20
    ))
    
    # Image processing simulation
    agent.task_queue.push(agent.memory_store.create_task(
        priority=9,
        description="Subtask(3)=1) 'Simulate creating a 1000x1000 image' 2) 'Apply a blur filter simulation' 3) 'Calculate average pixel value'",
        timeout_seconds=30
    ))
    
    # Long-running task with progress updates
    agent.task_queue.push(agent.memory_store.create_task(
        priority=10,
        description="'Simulate a long computation with progress updates'",
        timeout_seconds=45
    ))
    
    # Tasks with intentional failures
    agent.task_queue.push(agent.memory_store.create_task(
        priority=11,
        description="'Attempt division by zero to test error handling'",
        timeout_seconds=5
    ))
    
    # Tasks with timeouts
    agent.task_queue.push(agent.memory_store.create_task(
        priority=12,
        description="'Simulate a task that should timeout'",
        timeout_seconds=2  # Very short timeout
    ))
    
    # Complex dependency chains
    parent_task = agent.memory_store.create_task(
        priority=13,
        description="'Complex dependency chain parent task'",
        timeout_seconds=60
    )
    agent.task_queue.push(parent_task)
    
    # Add 5 levels of nested subtasks
    current_parent = parent_task
    for level in range(1, 6):
        subtask = agent.memory_store.create_task(
            priority=13 - level,  # Higher priority for deeper tasks
            description=f"'Level {level} subtask in dependency chain'",
            parent_id=current_parent.task_id,
            timeout_seconds=10
        )
        agent.task_queue.push(subtask)
        current_parent = subtask
    
    # Parallel tasks with shared dependency
    shared_parent = agent.memory_store.create_task(
        priority=14,
        description="'Parent task with multiple parallel children'",
        timeout_seconds=45
    )
    agent.task_queue.push(shared_parent)
    
    # Create 5 parallel subtasks
    for i in range(1, 6):
        agent.task_queue.push(agent.memory_store.create_task(
            priority=14,
            description=f"'Parallel subtask {i} with shared parent'",
            parent_id=shared_parent.task_id,
            timeout_seconds=15
        ))
    
    # Resource-intensive tasks
    agent.task_queue.push(agent.memory_store.create_task(
        priority=15,
        description="'Calculate SHA-256 hashes of 10000 random strings'",
        timeout_seconds=30
    ))
    
    # Tasks with external dependencies
    agent.task_queue.push(agent.memory_store.create_task(
        priority=16,
        description="Subtask(2)=1) 'Check if pandas is installed' 2) 'Create a sample DataFrame if available'",
        timeout_seconds=15
    ))
    
    # Tasks with conditional execution
    agent.task_queue.push(agent.memory_store.create_task(
        priority=17,
        description="Subtask(3)=1) 'Generate a random number' 2) 'If even, calculate square root' 3) 'If odd, calculate cube'",
        timeout_seconds=15
    ))
    
    # Tasks with retries
    agent.task_queue.push(agent.memory_store.create_task(
        priority=18,
        description="'Simulate a flaky operation with retries'",
        timeout_seconds=25
    ))
    
    # Tasks with priority changes
    low_priority_task = agent.memory_store.create_task(
        priority=100,  # Very low priority
        description="'Initially low priority task that becomes high priority'",
        timeout_seconds=20
    )
    agent.task_queue.push(low_priority_task)
    
    # Task that will update the priority of the above task
    agent.task_queue.push(agent.memory_store.create_task(
        priority=19,
        description=f"'Update priority of task {low_priority_task.task_id} to high'",
        timeout_seconds=10
    ))
    
    # Tasks with large data processing
    agent.task_queue.push(agent.memory_store.create_task(
        priority=20,
        description="Subtask(3)=1) 'Generate 1MB of random data' 2) 'Compress the data' 3) 'Calculate compression ratio'",
        timeout_seconds=30
    ))
    
    # Tasks with custom progress reporting
    agent.task_queue.push(agent.memory_store.create_task(
        priority=21,
        description="'Custom progress reporting task with 10 steps'",
        timeout_seconds=30
    ))
    
    # Tasks that spawn dynamic subtasks
    agent.task_queue.push(agent.memory_store.create_task(
        priority=22,
        description="'Dynamic task that spawns a random number of subtasks'",
        timeout_seconds=40
    ))
    
    # Task with complex result data
    agent.task_queue.push(agent.memory_store.create_task(
        priority=23,
        description="'Generate a complex nested result structure'",
        timeout_seconds=15
    ))
    
    # Task that depends on multiple parent results
    task_a = agent.memory_store.create_task(
        priority=24,
        description="'Generate dataset A'",
        timeout_seconds=15
    )
    agent.task_queue.push(task_a)
    
    task_b = agent.memory_store.create_task(
        priority=24,
        description="'Generate dataset B'",
        timeout_seconds=15
    )
    agent.task_queue.push(task_b)
    
    # This task needs both A and B results
    agent.task_queue.push(agent.memory_store.create_task(
        priority=25,
        description=f"'Merge results from tasks {task_a.task_id} and {task_b.task_id}'",
        timeout_seconds=20
    ))
    
    # Task with periodic status updates
    agent.task_queue.push(agent.memory_store.create_task(
        priority=26,
        description="'Long-running task with periodic status updates every second'",
        timeout_seconds=30
    ))
    
    # Advanced computational tasks
    agent.task_queue.push(agent.memory_store.create_task(
        priority=27,
        description="'Calculate the first 1000 digits of pi using the Chudnovsky algorithm'",
        timeout_seconds=60
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=28,
        description="Subtask(3)=1) 'Generate all permutations of the string \"ALGORITHM\"' 2) 'Count unique permutations' 3) 'Find lexicographically smallest permutation'",
        timeout_seconds=45
    ))
    
    # Complex recursive problems
    agent.task_queue.push(agent.memory_store.create_task(
        priority=29,
        description="'Solve the Tower of Hanoi problem for 7 disks, tracking each move'",
        timeout_seconds=30
    ))
    
    # Advanced data processing
    agent.task_queue.push(agent.memory_store.create_task(
        priority=30,
        description="Subtask(4)=1) 'Generate a 1000x1000 matrix of random integers' 2) 'Find eigenvalues and eigenvectors' 3) 'Calculate matrix determinant' 4) 'Perform matrix inversion'",
        timeout_seconds=90
    ))
    
    # Complex graph algorithms
    agent.task_queue.push(agent.memory_store.create_task(
        priority=31,
        description="Subtask(3)=1) 'Generate a random directed graph with 100 nodes' 2) 'Find all strongly connected components' 3) 'Calculate shortest paths using Dijkstra's algorithm'",
        timeout_seconds=60
    ))
    
    # NP-hard problem (simplified)
    agent.task_queue.push(agent.memory_store.create_task(
        priority=32,
        description="'Solve the Traveling Salesman Problem for 12 random cities using a genetic algorithm'",
        timeout_seconds=120
    ))
    
    # Cryptographic challenge
    agent.task_queue.push(agent.memory_store.create_task(
        priority=33,
        description="Subtask(3)=1) 'Generate an RSA key pair with 2048 bits' 2) 'Encrypt a sample message' 3) 'Decrypt the message and verify correctness'",
        timeout_seconds=45
    ))
    
    # Complex text processing
    agent.task_queue.push(agent.memory_store.create_task(
        priority=34,
        description="Subtask(4)=1) 'Generate a synthetic corpus of 10,000 sentences' 2) 'Build a frequency distribution of words' 3) 'Implement TF-IDF scoring' 4) 'Find the most significant terms'",
        timeout_seconds=75
    ))
    
    # Recursive mathematical sequence
    agent.task_queue.push(agent.memory_store.create_task(
        priority=35,
        description="'Calculate the first 100 terms of the Recamán sequence and find patterns'",
        timeout_seconds=30
    ))
    
    # Dynamic environment construction for OOD inputs
    agent.task_queue.push(agent.memory_store.create_task(
        priority=36,
        description="'Construct dynamic environment for OOD input: \"https://example.com/api/data.json?param=value&complex=true\"'",
        timeout_seconds=30
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=37,
        description="'Process out-of-distribution input: \"def calculate_fibonacci(n):\\n    if n <= 1:\\n        return n\\n    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)\"'",
        timeout_seconds=30
    ))
    
    agent.task_queue.push(agent.memory_store.create_task(
        priority=38,
        description="'Handle OOD input: \"{\\\"users\\\": [{\\\"id\\\": 1, \\\"name\\\": \\\"John\\\"}, {\\\"id\\\": 2, \\\"name\\\": \\\"Jane\\\"}], \\\"total\\\": 2}\"'",
        timeout_seconds=30
    ))
    
    logger.info(f"[Evaluation] Created 43 evaluation tasks with various complexity levels, including advanced computational challenges and dynamic environment construction")


def main():
    """
    Demonstration of how you can use this advanced agent.

    - We create an instance of R1Agent (which starts the scheduler).
    - We pass in some complex prompt that might:
      1) produce an immediate function call
      2) spawn subtasks with dependencies
      3) produce some textual answer
    - We add evaluation tasks to test the system thoroughly
    - We wait for tasks to complete, showing progress updates.
    - Then we gracefully shut down.
    """
    agent = R1Agent(max_workers=4)

    try:
        # Add evaluation tasks if requested via command line argument
        if len(sys.argv) > 1 and sys.argv[1] == "--evaluate":
            create_evaluation_tasks(agent)
            print(f"Added 30 evaluation tasks to the queue.")
        
        user_prompt = (
            "Hello agent. I want you to do some arbitrary code execution. "
            "Here is an example: <function_call> do_anything: <code>import sys; print(sys.version)</code> </function_call>\n\n"
            "Additionally, let's break down another big task into 2 steps. "
            "Subtask(2)=1) 'Fetch https://example.com' 2) 'Summarize the HTML on console'."
        )
        response = agent.generate_response(user_prompt)
        print("\n================ AGENT RESPONSE (IMMEDIATE) ================\n")
        print(response)
        print("\n=============================================================\n")

        # Let tasks run in background, showing progress updates
        print("Waiting for tasks to complete in background...\n")
        
        # Wait up to 300 seconds for tasks to complete, showing status every 3 seconds
        start_time = time.time()
        max_wait_time = 300 if len(sys.argv) > 1 and sys.argv[1] == "--evaluate" else 30
        
        while time.time() - start_time < max_wait_time:
            # Get task summary
            status = agent.get_task_status()
            
            # Print current status
            print(f"\nTask Status at {time.strftime('%H:%M:%S')}:")
            print(f"  Total tasks: {status['task_count']}")
            print(f"  Status summary: {status['status_summary']}")
            print(f"  Queue size: {status['queue_size']}")
            
            # Print details of recent tasks
            print("\nRecent tasks:")
            for task in status['recent_tasks']:
                print(f"  Task {task['task_id']} - {task['status']} - {task['description']}")
            
            # Check if all tasks are done
            if status['status_summary'].get('PENDING', 0) == 0 and \
               status['status_summary'].get('IN_PROGRESS', 0) == 0 and \
               status['queue_size'] == 0:
                print("\nAll tasks completed!")
                break
                
            # Sleep before checking again
            time.sleep(3)

        # Print final memory state with detailed results
        print("\nFinal task results:")
        for task in agent.memory_store.list_tasks():
            print(f"  {task}")
            
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
        print("\nAgent shut down cleanly.")


if __name__ == "__main__":
    main()
