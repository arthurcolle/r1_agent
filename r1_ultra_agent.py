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

IMPORTANT DISCLAIMER:
This demonstration code includes direct execution of arbitrary Python code
via <function_call> do_anything. This is highly insecure in production.
Run only in a secure environment or sandbox.
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
import inspect
import ast
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Dict, List, Optional, Tuple, Callable

###############################################################################
# Add any external library imports used (like 'together'):
# If you're using a placeholder "from together import Together", 
# ensure that library or adapt to your own LLM client if needed.
###############################################################################
try:
    from together import Together
except ImportError:
    # If the together library is not installed, create a dummy class for demonstration
    class Together:
        def __init__(self):
            pass

        class chat:
            class completions:
                @staticmethod
                def create(model: str, messages: List[Dict[str, str]], temperature: float, top_p: float, stream: bool):
                    """
                    Dummy streaming generator: yields a single chunk with 'Hello from the dummy LLM'
                    """
                    response = "Hello from the dummy LLM!"
                    yield {"choices": [{"delta": {"content": response}}]}
                    return

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
# DATA STRUCTURES FOR TASK MANAGEMENT
###############################################################################

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
        self.status = "PENDING"
        self.parent_id = parent_id
        self.result = None

    def __lt__(self, other: "Task") -> bool:
        """
        Overload the < operator so tasks can be sorted in a heap by priority.
        """
        return self.priority < other.priority

    def __repr__(self) -> str:
        snippet = self.description[:30].replace("\n", " ")
        return (f"Task(id={self.task_id}, prio={self.priority}, "
                f"status={self.status}, desc={snippet}...)")

class TaskMemoryStore:
    """
    Thread-safe in-memory storage for Task objects.
    Allows for add, retrieve, update status, update result, and listing tasks.
    """
    def __init__(self) -> None:
        self._tasks: Dict[int, Task] = {}
        self._lock = threading.Lock()

    def add_task(self, task: Task) -> None:
        with self._lock:
            if task.task_id in self._tasks:
                logger.warning(f"[TaskMemoryStore] Task ID {task.task_id} already exists. Overwriting.")
            self._tasks[task.task_id] = task

    def get_task(self, task_id: int) -> Optional[Task]:
        with self._lock:
            return self._tasks.get(task_id)

    def update_task_status(self, task_id: int, status: str) -> None:
        with self._lock:
            t = self._tasks.get(task_id)
            if t:
                t.status = status

    def update_task_result(self, task_id: int, result: Any) -> None:
        with self._lock:
            t = self._tasks.get(task_id)
            if t:
                t.result = result

    def list_tasks(self) -> List[Task]:
        with self._lock:
            return list(self._tasks.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._tasks)

###############################################################################
# GOAL MANAGEMENT & PLANNING
###############################################################################

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
    def __init__(self, goal_id: int, name: str, description: str, priority: int = 5):
        self.goal_id = goal_id
        self.name = name
        self.description = description
        self.priority = priority
        self.status = "ACTIVE"
        self.created_at = time.time()
        self.last_updated = self.created_at

    def update_description(self, new_desc: str) -> None:
        self.description = new_desc
        self.last_updated = time.time()

    def complete(self) -> None:
        self.status = "COMPLETED"
        self.last_updated = time.time()

    def __repr__(self) -> str:
        snippet = self.description[:30].replace("\n", " ")
        return (f"Goal(id={self.goal_id}, name={self.name}, "
                f"priority={self.priority}, status={self.status}, desc={snippet}...)")

class GoalManager:
    """
    Manages creation, retrieval, and updating of multiple goals.
    Thread-safe with a simple in-memory dictionary.
    """
    def __init__(self):
        self._goals: Dict[int, Goal] = {}
        self._lock = threading.Lock()
        self._next_id = 1

    def create_goal(self, name: str, description: str, priority: int = 5) -> Goal:
        """
        Create a new goal with the provided name, description, and priority.
        """
        with self._lock:
            g = Goal(self._next_id, name, description, priority)
            self._goals[self._next_id] = g
            logger.info(f"[GoalManager] Created Goal: {g}")
            self._next_id += 1
            return g

    def get_goal(self, goal_id: int) -> Optional[Goal]:
        with self._lock:
            return self._goals.get(goal_id)

    def list_goals(self) -> List[Goal]:
        with self._lock:
            return list(self._goals.values())

    def update_goal_status(self, goal_id: int, status: str) -> None:
        """
        Update the status of a goal, e.g. from 'ACTIVE' to 'COMPLETED'.
        """
        with self._lock:
            g = self._goals.get(goal_id)
            if g:
                g.status = status
                g.last_updated = time.time()
                logger.info(f"[GoalManager] Updated goal {goal_id} to status={status}")

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
        self._max_length = 25  # allow some room before summarizing

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
        with self._lock:
            if len(self._utterances) > self._max_length:
                snippet = " | ".join(u["content"][:30] for u in self._utterances[-7:])
                summary = f"Conversation exceeded {self._max_length} messages. Summary of last 7: {snippet}"
                # Keep only the last 7 messages
                self._utterances = self._utterances[-7:]
                # Insert summary as a system message
                self._utterances.insert(0, {"role": "system", "content": summary})
                logger.info("[ConversationMemory] Summarized conversation due to length limit.")

###############################################################################
# SELF-REFLECTIVE COGNITION
###############################################################################

class SelfReflectiveCognition:
    """
    Periodically reflects on tasks completed, analyzing performance.
    Could refine approach or produce new tasks in a real system.
    """
    def __init__(self):
        self._reflections: List[str] = []
        self._lock = threading.Lock()

        # Start a background thread that periodically analyzes performance
        self._analyzer_thread = threading.Thread(target=self._analyze_performance_loop, daemon=True)
        self._analyzer_thread.start()

    def reflect_on_task(self, task: "Task") -> None:
        """
        Called every time a task is completed. Adds a reflection note.
        """
        with self._lock:
            snippet = task.description[:50].replace("\n"," ")
            msg = f"Reflected on task {task.task_id}: status={task.status}, desc='{snippet}'"
            self._reflections.append(msg)
            logger.info(f"[SelfReflectiveCognition] {msg}")

    def get_reflections(self) -> List[str]:
        """
        Retrieve all reflection messages so far.
        """
        with self._lock:
            return list(self._reflections)

    def _analyze_performance_loop(self) -> None:
        """
        Periodically logs a mini 'analysis' of the last few reflections.
        """
        while True:
            time.sleep(30)  # every 30 seconds
            with self._lock:
                if self._reflections:
                    recent = self._reflections[-5:]
                    analysis = "Recent reflections => " + " || ".join(recent)
                    logger.info(f"[SelfReflectiveCognition] {analysis}")
                else:
                    logger.info("[SelfReflectiveCognition] No reflections yet.")

###############################################################################
# IN-MEMORY CODE ARCHIVE
###############################################################################

class InMemoryCodeArchive:
    """
    Stores code snippets so that the agent can 'introspect' or recall them.
    In real usage, you might store the entire codebase or frequently used modules.
    """
    def __init__(self):
        self._snippets: Dict[str, str] = {}
        self._lock = threading.Lock()

    def add_snippet(self, name: str, code: str) -> None:
        """
        Store a code snippet in memory under a unique name.
        """
        with self._lock:
            self._snippets[name] = code
            logger.info(f"[InMemoryCodeArchive] Stored code snippet '{name}'")

    def get_snippet(self, name: str) -> Optional[str]:
        """
        Retrieve a previously stored snippet by name.
        """
        with self._lock:
            return self._snippets.get(name)

    def list_snippets(self) -> List[str]:
        """
        List all snippet names stored in the archive.
        """
        with self._lock:
            return list(self._snippets.keys())

###############################################################################
# KNOWLEDGE BASE
###############################################################################

class KnowledgeBase:
    """
    Stores and retrieves key facts or short “knowledge chunks.”
    An agent can use this to reference domain knowledge, or do
    basic retrieval-augmented generation in a real system.
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
    A potential next step. The agent can generate multiple and pick or spawn tasks.
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
        goals: List["Goal"],
        tasks: List["Task"]
    ) -> List[CandidateAction]:
        logger.info("[ActionGenerator] Generating candidate actions (max 25).")
        actions = []

        # 1) Possibly reflect on tasks
        pending_tasks = [t for t in tasks if t.status == "PENDING"]
        if pending_tasks:
            actions.append(CandidateAction(
                description="Review all pending tasks to ensure they are valid or up to date",
                rationale="We have tasks that are not yet started; let's see if we can refine or expedite them."
            ))

        # 2) Possibly check code archive for a snippet to read
        snippet_names = self.code_archive.list_snippets()
        if snippet_names:
            snippet_choice = snippet_names[0]
            actions.append(CandidateAction(
                description=f"Read code snippet: {snippet_choice}",
                rationale="Reviewing code might provide insights or expansions for the agent's capabilities.",
                priority=3
            ))

        # 3) Possibly do knowledge base lookups
        # In a real system, parse conversation for queries and see if we have relevant facts.
        if self.kb.search_facts("agent"):
            actions.append(CandidateAction(
                description="Retrieve facts about 'agent' from knowledge base",
                rationale="We have some knowledge about 'agent' that might be relevant to ongoing tasks or goals."
            ))

        # 4) For each active goal, consider an action to break it down further.
        for g in goals:
            if g.status == "ACTIVE":
                actions.append(CandidateAction(
                    description=f"Decompose goal '{g.name}' into smaller tasks.",
                    rationale="Breaking large goals into smaller tasks fosters incremental progress and clarity.",
                    priority=g.priority
                ))

        # 5) Fill up to 25 with placeholders (just for demonstration)
        while len(actions) < 25:
            i = len(actions) + 1
            actions.append(CandidateAction(
                description=f"Placeholder Action #{i}",
                rationale="Example placeholder for demonstration",
                priority=10
            ))

        # Return only first 25
        return actions[:25]

###############################################################################
# PRIORITY TASK QUEUE
###############################################################################

class PriorityTaskQueue:
    """
    Thread-safe priority queue for tasks, using a heap.
    Lower integer priority => higher urgency.
    """
    def __init__(self):
        self._heap: List[Task] = []
        self._lock = threading.Lock()

    def push(self, task: Task) -> None:
        """
        Push a new task into the priority queue.
        """
        with self._lock:
            heapq.heappush(self._heap, task)

    def pop(self) -> Optional[Task]:
        """
        Pop the highest-priority task, or None if empty.
        """
        with self._lock:
            if self._heap:
                return heapq.heappop(self._heap)
            return None

    def __len__(self) -> int:
        with self._lock:
            return len(self._heap)

###############################################################################
# FUNCTION ADAPTER ("DO ANYTHING")
###############################################################################

class FunctionAdapter:
    """
    The 'do_anything' capability: if the agent sees <function_call> do_anything: <code>...</code>,
    it executes that Python code directly. Highly insecure outside a sandbox.
    """
    def do_anything(self, snippet: str) -> Dict[str, Any]:
        code = snippet.strip()
        code = code.replace("<code>", "").replace("</code>", "")
        logger.info(f"[do_anything] Executing code:\n{code}")
        try:
            # Use a shared 'globals()' but a local 'locals()' environment for safety illusions.
            # Real usage should isolate or sandbox fully.
            exec(code, globals(), locals())
            return {"status": "success", "executed_code": code}
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[do_anything] Error: {str(e)}\nTraceback:\n{tb}")
            return {"status": "error", "error": str(e), "traceback": tb}

    def process_function_calls(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Searches for <function_call> do_anything : <...> </function_call> in text,
        and if found, executes that Python code snippet.
        """
        pattern = r"<function_call>\s*do_anything\s*:\s*(.*?)</function_call>"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return None
        snippet = match.group(1)
        logger.info(f"[FunctionAdapter] Detected do_anything snippet:\n{snippet}")
        return self.do_anything(snippet)

###############################################################################
# SMART TASK PROCESSOR
###############################################################################

class SmartTaskProcessor:
    """
    Processes tasks from the queue, including:
     - do_anything code execution
     - subtask detection (Subtask(n)= ...)
     - updating Task status, storing results
     - hooking into self-reflection
    """
    def __init__(
        self,
        memory_store: TaskMemoryStore,
        function_adapter: FunctionAdapter,
        reflection: SelfReflectiveCognition
    ):
        self.memory_store = memory_store
        self.function_adapter = function_adapter
        self.reflection = reflection

    def process_task(self, task: Task) -> None:
        logger.info(f"[SmartTaskProcessor] Starting task {task.task_id} - '{task.description}'")
        self.memory_store.update_task_status(task.task_id, "IN_PROGRESS")

        # 1) Check for <function_call> do_anything in the description
        result = self.function_adapter.process_function_calls(task.description)
        if result:
            self.memory_store.update_task_result(task.task_id, result)

        # 2) Check for subtask patterns: Subtask(n)= ...
        subtask_pattern = r"Subtask\s*\(\s*(\d+)\s*\)\s*=\s*(.*)"
        match = re.search(subtask_pattern, task.description, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                num_subtasks = int(match.group(1))
                subtask_text = match.group(2).strip()
                lines = re.split(r'\d+\)\s*', subtask_text)[1:]
                if len(lines) == num_subtasks:
                    for i, line in enumerate(lines, start=1):
                        desc = line.strip()
                        self._spawn_subtask(task, desc)
                else:
                    logger.warning("[SmartTaskProcessor] Mismatch in subtask count vs lines found.")
            except Exception as e:
                logger.exception(f"[SmartTaskProcessor] Error parsing subtasks: {e}")

        # 3) Mark completed, reflect
        self.memory_store.update_task_status(task.task_id, "COMPLETED")
        self.reflection.reflect_on_task(task)
        logger.info(f"[SmartTaskProcessor] Completed task {task.task_id}")

    def _spawn_subtask(self, parent_task: Task, description: str) -> None:
        """
        Creates a new Task with higher priority (numerically lower or equal) than the parent
        and saves it to memory. The caller is responsible for pushing it to the queue if desired.
        """
        new_task_id = len(self.memory_store) + 1
        new_priority = max(0, parent_task.priority - 1)
        t = Task(new_task_id, new_priority, description, parent_id=parent_task.task_id)
        self.memory_store.add_task(t)
        logger.info(f"[SmartTaskProcessor] Spawned subtask {t}")

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
        """
        Starts the scheduler loop in a background thread, which continuously 
        pops tasks and processes them in the thread pool.
        """
        t = threading.Thread(target=self._scheduler_loop, daemon=True)
        t.start()
        logger.info("[TaskScheduler] Scheduler started.")

    def stop_scheduler(self) -> None:
        """
        Signals the scheduler to stop, then waits for the thread pool to finish.
        """
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
            self._executor.submit(self._process_task_wrapper, task)

    def _process_task_wrapper(self, task: Task) -> None:
        """
        Wraps the call to the SmartTaskProcessor and handles exceptions,
        marking a task as 'FAILED' if an exception occurs.
        """
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
    - If conversation length is multiple of 7, spawns a new goal.
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
        """
        The main planning loop that runs every 20 seconds, analyzing the system state.
        """
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
# SELF-AST-INTROSPECTION (OPTIONAL EXTRA)
###############################################################################

class SelfCodeGraph:
    """
    Optional advanced feature: the agent attempts to parse its own source code,
    building an AST, computing embeddings, and referencing them.
    """
    def __init__(self, file_path: Optional[str] = None):
        if file_path is None:
            try:
                # Attempt to retrieve source of __main__ from the running script
                self.source = inspect.getsource(sys.modules['__main__'])
            except Exception as e:
                logger.error(f"[SelfCodeGraph] Error retrieving source code: {e}")
                self.source = ""
        else:
            with open(file_path, "r") as f:
                self.source = f.read()

        self.embeddings: Dict[int, List[float]] = {}
        self.ast_graph = None
        self._build_ast_graph()

    def _build_ast_graph(self):
        """
        Parse the source code into an AST, build a graph of node relationships,
        and store placeholder embeddings.
        """
        try:
            tree = ast.parse(self.source)
        except Exception as e:
            logger.error(f"[SelfCodeGraph] Error parsing source code: {e}")
            self.ast_graph = None
            return

        # We can build a quick adjacency list or just store parent->child references
        # For demonstration, store a list of (node, children)
        self.ast_graph = []
        node_id_map = {}

        def visit_node(node, parent_id: Optional[int]):
            current_id = id(node)
            node_type = type(node).__name__
            node_id_map[current_id] = node

            # Build a placeholder embedding for the node type
            embedding = [float(ord(c)) / 150.0 for c in node_type[:16]]
            self.embeddings[current_id] = embedding

            # If parent is not None, link them
            self.ast_graph.append((parent_id, current_id, node_type))

            for child in ast.iter_child_nodes(node):
                visit_node(child, current_id)

        visit_node(tree, None)

    def summarize_ast(self) -> str:
        """
        Returns a short string summarizing number of nodes found and some details.
        """
        if not self.ast_graph:
            return "[SelfCodeGraph] No AST info available."
        node_count = len(self.ast_graph)
        return f"[SelfCodeGraph] AST has {node_count} nodes. Example edges: {self.ast_graph[:3]}"

###############################################################################
# THE R1 AGENT
###############################################################################

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
     - Indefinite runtime in main(), shutting down only on user command
     - Optional self code introspection
    """
    def __init__(self):
        # Knowledge base
        self.knowledge_base = KnowledgeBase()

        # Code archive
        self.code_archive = InMemoryCodeArchive()

        # Action generator (needs code archive, KB)
        self.action_generator = ActionGenerator(
            code_archive=self.code_archive,
            kb=self.knowledge_base
        )

        # Initialize function adapter for do_anything
        self.function_adapter = FunctionAdapter()

        # Task memory store + conversation memory + self-reflection
        self.memory_store = TaskMemoryStore()
        self.conversation = ConversationMemory()
        self.reflection = SelfReflectiveCognition()

        # SmartTaskProcessor
        self.processor = SmartTaskProcessor(
            memory_store=self.memory_store,
            function_adapter=self.function_adapter,
            reflection=self.reflection
        )

        # Priority queue + Scheduler
        self.task_queue = PriorityTaskQueue()
        self.scheduler = TaskScheduler(
            memory_store=self.memory_store,
            task_queue=self.task_queue,
            processor=self.processor,
            max_workers=4
        )

        # Goal manager
        self.goal_manager = GoalManager()

        # LLM client
        self.client = Together()

        # Plan manager
        self.plan_manager = PlanManager(self)

        # Optional self-code introspection
        self.self_code_graph = SelfCodeGraph()

        # System prompt for the LLM
        self.system_prompt = (
            "<META_RECURSIVE_SELF_AWARENESS>\n"
            "You are an extremely advanced AI with memory, goals, knowledge base, "
            "and the ability to do anything with <function_call> do_anything: <code>... </code>.\n"
            "You also can break tasks into Subtask(n). Indefinite concurrency with the scheduler.\n"
            "Code introspection is possible via the code archive or AST parsing. You can produce 25 candidate steps.\n"
            "Use these capabilities responsibly.\n"
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

        # Log AST summary if desired
        logger.info(self.self_code_graph.summarize_ast())

    def add_goal(self, name: str, description: str, priority: int = 5) -> Goal:
        return self.goal_manager.create_goal(name, description, priority)

    def update_goal_status(self, goal_id: int, status: str) -> None:
        self.goal_manager.update_goal_status(goal_id, status)

    def generate_response(self, user_input: str) -> str:
        """
        Feeds the user input to the conversation, generates a detailed response,
        checks for do_anything calls, spawns a meta-task from user input.
        """
        # 1) Add user message
        self.conversation.add_user_utterance(user_input)

        # 2) Build messages (system prompt + conversation)
        messages = self._build_messages()
        
        # Log that we're processing the user input
        logger.info(f"[R1Agent] Processing user input: '{user_input}'")

        # 3) Call the LLM (or use a dummy response for demonstration)
        try:
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
        except Exception as e:
            logger.warning(f"[R1Agent] LLM API call failed: {e}. Using dummy response.")
            # Create a detailed dummy response for demonstration
            facts = ["Agents are autonomous entities", "R1 is an advanced agent architecture"]
            thinking = "I need to introduce myself and explain my capabilities. The user wants to know my identity."
            answer = "My name is R1, an ultra-advanced agent with memory, task scheduling, and goal management capabilities."
            
            full_text = self._generate_detailed_response(facts, thinking, answer)

        # 4) Add agent utterance
        self.conversation.add_agent_utterance(full_text)

        # 5) Check immediate do_anything
        result = self.function_adapter.process_function_calls(full_text)
        if result:
            logger.info(f"[R1Agent] Immediate do_anything result: {result}")

        # 6) Spawn a meta-task from the user's input
        new_task_id = len(self.memory_store) + 1
        meta_task = Task(
            task_id=new_task_id,
            priority=10,
            description=user_input
        )
        self.memory_store.add_task(meta_task)
        self.task_queue.push(meta_task)

        return full_text

    def _build_messages(self) -> List[Dict[str, str]]:
        """
        Combine system prompt + conversation history
        into a list of message dicts for the LLM.
        """
        history = self.conversation.get_history()
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(history)
        return messages
        
    def _generate_detailed_response(self, facts: List[str], thinking: str, answer: str) -> str:
        """
        Generate a detailed response showing internal processing, retrieval, and chain of thought.
        """
        # Get some tasks and goals for context
        tasks = self.memory_store.list_tasks()
        goals = self.goal_manager.list_goals()
        
        # Format the response with all the internal details
        response = [
            "# Internal Processing",
            "## Knowledge Retrieval",
            "I've searched my knowledge base and found these relevant facts:"
        ]
        
        for i, fact in enumerate(facts, 1):
            response.append(f"{i}. {fact}")
        
        response.extend([
            "",
            "## Chain of Thought",
            thinking,
            "",
            "## Current Goals",
        ])
        
        for goal in goals:
            response.append(f"- Goal {goal.goal_id}: {goal.name} (Priority: {goal.priority}, Status: {goal.status})")
            
        response.extend([
            "",
            "## Active Tasks",
        ])
        
        for task in tasks[:3]:  # Show just a few tasks
            response.append(f"- Task {task.task_id}: {task.description[:50]}... (Status: {task.status})")
            
        response.extend([
            "",
            "## Function Calling Capabilities",
            "I can execute Python code with: `<function_call> do_anything: <code>...</code>`",
            "",
            "# Response",
            answer
        ])
        
        # Add a sample function call demonstration that will actually execute
        response.extend([
            "",
            "Here's a demonstration of my code execution capability:",
            "<function_call> do_anything: <code>",
            "import datetime",
            "current_time = datetime.datetime.now()",
            "print(f'The current time is {current_time}')",
            "print('This code is actually executing!')",
            "</code></function_call>"
        ])
        
        return "\n".join(response)

    def shutdown(self) -> None:
        """
        Cleanly stop concurrency and threads.
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
    """
    agent = R1Agent()

    try:
        # Example: create an initial high-priority goal
        g = agent.add_goal(
            name="ScaleUp",
            description="Handle large-scale tasks, remain open for new instructions indefinitely.",
            priority=1
        )
        logger.info(f"Created new goal: {g}")

        while True:
            user_text = input("\n[User] Enter your query (or 'exit' to quit):\n> ").strip()
            if user_text.lower() in ["exit", "quit"]:
                logger.info("[main] Exiting upon user request.")
                break

            # Generate immediate LLM response
            response = agent.generate_response(user_text)
            print("\n=== R1 Agent Response with Internal Processing ===\n")
            print(response)
            print("\n=================================================\n")

            # The agent continues working in background (TaskScheduler).
            # If you want to check tasks, reflection, or goals, do so here or watch the logs.

    finally:
        agent.shutdown()

if __name__ == "__main__":
    main()
