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
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Dict, List, Optional, Tuple, Callable
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
        status (str): Current status: 'PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED', etc.
        parent_id (Optional[int]): The ID of a parent task, if any. This enables
            recursive decomposition, where subtasks can track their parent's ID.
        result (Any): Output or result data from the task, if any.
    """
    def __init__(self, task_id: int, priority: int, description: str, parent_id: Optional[int] = None):
        self.task_id = task_id
        self.priority = priority
        self.description = description
        self.status = "PENDING"
        self.parent_id = parent_id
        self.result = None

    def __lt__(self, other: "Task") -> bool:
        """
        Allows tasks to be compared based on priority for use in a priority queue.
        A lower priority value means the task is more urgent.
        """
        return self.priority < other.priority

    def __repr__(self) -> str:
        return f"Task(id={self.task_id}, priority={self.priority}, status={self.status}, desc={self.description[:30]})"


class TaskMemoryStore:
    """
    Maintains storage of tasks in memory (in a dictionary).
    Provides methods to add, retrieve, and update tasks.

    In real production code, you could store these in a DB or persistent store.
    """
    def __init__(self) -> None:
        self._tasks: Dict[int, Task] = {}
        self._lock = threading.Lock()

    def add_task(self, task: Task) -> None:
        """
        Add a new Task object to memory.
        """
        with self._lock:
            if task.task_id in self._tasks:
                logger.warning(f"Task ID {task.task_id} already exists!")
            self._tasks[task.task_id] = task

    def get_task(self, task_id: int) -> Optional[Task]:
        """
        Fetch a Task by its ID.
        """
        with self._lock:
            return self._tasks.get(task_id)

    def update_task_status(self, task_id: int, status: str) -> None:
        """
        Change the status of a task in memory.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = status

    def update_task_result(self, task_id: int, result: Any) -> None:
        """
        Store the result in the task's record.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.result = result

    def list_tasks(self) -> List[Task]:
        """
        Return a list of all tasks in memory (for debugging or inspection).
        """
        with self._lock:
            return list(self._tasks.values())

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
    - For demonstration, we only have one "do_anything" function, which
      executes arbitrary Python code.
    """

    def do_anything(self, snippet: str) -> Dict[str, Any]:
        """
        Execute arbitrary Python code from snippet (DANGEROUS).
        """
        code = snippet.strip()

        # Remove <code> tags if present
        code = code.replace("<code>", "").replace("</code>", "")

        logger.info(f"[do_anything] Executing code:\n{code}")

        try:
            exec(code, globals(), locals())
            return {"status": "success", "executed_code": code}
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[do_anything] Error: {str(e)}\nTraceback:\n{tb}")
            return {"status": "error", "error": str(e), "traceback": tb}

    def process_function_calls(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Detect a 'do_anything' token and everything after it up to </function_call>.
        Example usage in text:
            <function_call> do_anything: <some Python code> </function_call>
        """
        pattern = r"<function_call>\s*do_anything\s*:\s*(.*?)</function_call>"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return None

        snippet = match.group(1)
        logger.info(f"[FunctionAdapter] Detected do_anything snippet:\n{snippet}")
        return self.do_anything(snippet)

###############################################################################
# AGENT & RECURSIVE DECOMPOSITION LOGIC
###############################################################################

class SmartTaskProcessor:
    """
    The 'brains' that tries to handle tasks in a 'smart' way:
    - If a Task's description suggests it should be decomposed, we can create subtasks.
    - If a Task's description triggers a do_anything code snippet, we pass that to the FunctionAdapter.
    - We store results in memory, update the queue, etc.
    """

    def __init__(self, memory_store: TaskMemoryStore, function_adapter: FunctionAdapter):
        self.memory_store = memory_store
        self.function_adapter = function_adapter

    def process_task(self, task: Task) -> None:
        """
        Main logic for how we handle tasks:
         - 'Recursive decomposition' if needed
         - 'Function calls' if the text includes them
         - Update status + result

        If the Task spawns subtasks, we create them and place them in the queue.
        """
        logger.info(f"[SmartTaskProcessor] Starting task {task.task_id} - '{task.description}'")
        self.memory_store.update_task_status(task.task_id, "IN_PROGRESS")

        # 1) Check for <function_call> do_anything usage in the description
        result = self.function_adapter.process_function_calls(task.description)
        if result:
            # If code was executed, store the result
            self.memory_store.update_task_result(task.task_id, result)
            logger.info(f"[SmartTaskProcessor] Task {task.task_id} used do_anything with result: {result}")

        # 2) Potentially do "recursive decomposition"
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
                        logger.info(f"[SmartTaskProcessor] Created subtask {new_task.task_id} from parent {task.task_id}")
                else:
                    logger.warning(f"Number of lines found ({len(lines)}) does not match subtask count {num_subtasks}")
            except Exception as e:
                logger.exception(f"[SmartTaskProcessor] Error parsing subtask instructions: {str(e)}")

        # 3) If none of the above triggered anything else, we finalize the task
        self.memory_store.update_task_status(task.task_id, "COMPLETED")
        logger.info(f"[SmartTaskProcessor] Completed task {task.task_id}")

    def _create_subtask(self, parent_task: Task, description: str) -> Task:
        """
        Helper to create a subtask, store it, and push it to the queue.
        The subtask has a slightly higher (lower integer) priority than parent to emphasize immediate follow-up.
        """
        new_task_id = len(self.memory_store) + 1  # naive unique ID
        new_priority = max(0, parent_task.priority - 1)  # ensure subtask is at least as high or higher priority
        subtask = Task(
            task_id=new_task_id,
            priority=new_priority,
            description=description,
            parent_id=parent_task.task_id
        )
        self.memory_store.add_task(subtask)
        # We'll rely on an external TaskScheduler to push it to the queue. We'll just return it.
        return subtask


class TaskScheduler:
    """
    Coordinates the retrieval and execution of tasks from the priority queue, possibly in parallel.
    - Maintains a thread pool
    - Repeatedly pops tasks from the queue
    - Hands them to the SmartTaskProcessor
    - Observes if the Task spawns subtasks, and adds them to the queue

    The scheduler runs in a background thread (or we can do a .run_once for a single cycle).
    """

    def __init__(
        self,
        memory_store: TaskMemoryStore,
        task_queue: PriorityTaskQueue,
        processor: SmartTaskProcessor,
        max_workers: int = 4
    ):
        """
        Args:
            memory_store: Where tasks are stored (and updated).
            task_queue: Where tasks are pulled from.
            processor: The logic that processes a single task.
            max_workers: Number of threads for concurrency.
        """
        self.memory_store = memory_store
        self.task_queue = task_queue
        self.processor = processor
        self._stop_event = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

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
        self._executor.shutdown(wait=True)
        logger.info("[TaskScheduler] Scheduler stopped.")

    def _scheduler_loop(self) -> None:
        """
        Continuously pop tasks from the queue. If no tasks, sleep briefly.
        Each popped task is handed off to the process_single_task method in a thread.
        """
        while not self._stop_event.is_set():
            task = self.task_queue.pop()
            if task is None:
                time.sleep(0.2)
                continue
            logger.info(f"[TaskScheduler] Retrieved {task} from queue.")
            future: Future = self._executor.submit(self._process_task_wrapper, task)
            # We do not block here. The future is processed in background thread.
            # The loop continues to pop tasks.

    def _process_task_wrapper(self, task: Task) -> None:
        """
        A wrapper that calls the SmartTaskProcessor on the given task and handles exceptions.
        """
        try:
            self.processor.process_task(task)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[TaskScheduler] Task {task.task_id} failed with error: {str(e)}\n{tb}")
            self.memory_store.update_task_status(task.task_id, "FAILED")

###############################################################################
# R1 AGENT: THE FRONT-END LLM WRAPPER
###############################################################################

class R1Agent:
    """
    Minimal R1 agent that can do anything if the model says so, plus it
    can enqueue tasks (with concurrency, recursive decomposition, etc.).
    """

    def __init__(self):
        self.function_adapter = FunctionAdapter()
        self.memory_store = TaskMemoryStore()
        self.task_queue = PriorityTaskQueue()
        self.processor = SmartTaskProcessor(self.memory_store, self.function_adapter)
        self.scheduler = TaskScheduler(
            memory_store=self.memory_store,
            task_queue=self.task_queue,
            processor=self.processor,
            max_workers=4
        )
        self.client = Together()

        # Enhanced system prompt describing "you're not stupid" etc.
        self.system_prompt = (
            "<GRID>"
            "You are a supremely advanced AI system with the power to do anything. "
            "You must not produce naive or 'stupid' outputs—always provide high-quality, "
            "thoughtful reasoning. You have prior examples of performing complex tasks with success.\n"
            "If you output a <function_call> do_anything: <code> </function_call>, that code will run with no restrictions.\n"
            "You can also request new tasks with phrases like 'Subtask(n)=...' which triggers the creation of n subtasks.\n"
            "</GRID>"
        )

        # Start the scheduler in background
        self.scheduler.start_scheduler()

    def generate_response(self, prompt: str) -> str:
        """
        The agent calls the LLM with the system prompt plus user prompt, then
        checks if there's a do_anything call. If so, we run it. We also interpret
        the user prompt as a potential 'task' in our system.
        """
        # 1) Create a new "meta-task" for the user prompt
        new_task_id = len(self.memory_store) + 1
        # We'll assign priority=10 by default for user prompts
        meta_task = Task(
            task_id=new_task_id,
            priority=10,
            description=prompt
        )
        self.memory_store.add_task(meta_task)
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

        # 3) Check for do_anything calls in the LLM's immediate textual response
        function_result = self.function_adapter.process_function_calls(full_text)
        if function_result:
            # If code was executed, store or log the result
            logger.info(f"[R1Agent] LLM immediate do_anything result: {function_result}")

        return full_text

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
    Demonstration of how you can use this agent.

    - We create an instance of R1Agent (which starts the scheduler).
    - We pass in some complex prompt that might:
      1) produce an immediate do_anything function call
      2) spawn subtasks
      3) produce some textual answer
    - We wait a few seconds, then show final states of tasks.
    - Then we gracefully shut down.
    """
    agent = R1Agent()

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

        # Let tasks run in background for ~5 seconds
        print("Waiting 5 seconds for tasks to complete in background...\n")
        time.sleep(5)

        # Print final memory state:
        print("Tasks in memory:")
        for tsk in agent.memory_store.list_tasks():
            print(f"  {tsk}")

    finally:
        agent.shutdown()
        print("Agent shut down cleanly.")


if __name__ == "__main__":
    main()
