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
    """

    def __init__(self, memory_store: TaskMemoryStore, function_adapter: FunctionAdapter, task_queue: "PriorityTaskQueue"):
        self.memory_store = memory_store
        self.function_adapter = function_adapter
        self.task_queue = task_queue

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
            
        if task.description.startswith("'Calculate") or task.description.startswith("'Process"):
            self._handle_calculation_task(task)
            return

        # 2) Check for <function_call> usage in the description
        result = self.function_adapter.process_function_calls(task.description)
        if result:
            # If function was executed, store the result
            self.memory_store.update_task_result(task.task_id, result)
            logger.info(f"[SmartTaskProcessor] Task {task.task_id} executed function with result status: {result.get('status')}")
            task.update_progress(1.0)  # Mark as 100% complete
            return

        # 3) Potentially do "recursive decomposition"
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
            "4. Subtask(n)=1) 'Task description 1' 2) 'Task description 2' ... - Create n subtasks\n\n"
            "You can handle long-running operations by breaking them into subtasks. "
            "Tasks will be executed concurrently when possible, with proper dependency tracking.\n"
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

def main():
    """
    Demonstration of how you can use this advanced agent.

    - We create an instance of R1Agent (which starts the scheduler).
    - We pass in some complex prompt that might:
      1) produce an immediate function call
      2) spawn subtasks with dependencies
      3) produce some textual answer
    - We wait for tasks to complete, showing progress updates.
    - Then we gracefully shut down.
    """
    agent = R1Agent(max_workers=4)

    try:
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
        
        # Wait up to 30 seconds for tasks to complete, showing status every 3 seconds
        start_time = time.time()
        while time.time() - start_time < 30:
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
