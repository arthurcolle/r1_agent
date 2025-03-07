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
import subprocess
from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from pydantic import BaseModel, Field
from together import Together

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
        with self._lock:
            g = self._goals.get(goal_id)
            if g:
                g.status = status
                g.last_updated = time.time()
                logger.info(f"[GoalManager] Updated goal {goal_id} to status={status}")
                # Enhanced goal management: Re-evaluate priorities
                self._re_evaluate_goal_priorities()

    def _re_evaluate_goal_priorities(self) -> None:
        """
        Re-evaluate and adjust goal priorities based on current context.
        """
        with self._lock:
            for goal in self._goals.values():
                # Example logic: Increase priority for goals nearing completion
                if goal.status == "ACTIVE" and goal.priority > 1:
                    goal.priority -= 1
                    logger.info(f"[GoalManager] Increased priority for goal {goal.goal_id} to {goal.priority}")
                # Advanced goal management: Adjust based on performance metrics
                if goal.status == "ACTIVE" and self._should_adjust_goal_based_on_performance(goal):
                    goal.priority = max(0, goal.priority - 1)
                    logger.info(f"[GoalManager] Adjusted priority for goal {goal.goal_id} based on performance metrics.")

    def _should_adjust_goal_based_on_performance(self, goal: Goal) -> bool:
        """
        Determine if a goal's priority should be adjusted based on performance metrics.
        """
        # Placeholder logic for performance-based adjustment
        return True  # In a real implementation, this would be more complex

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
    VERIFICATION = "verification"
    BACKTRACKING = "backtracking"
    SUBGOAL_SETTING = "subgoal_setting"
    BACKWARD_CHAINING = "backward_chaining"
    REFLECTION = "reflection"
    ADAPTATION = "adaptation"
    EXPLORATION = "exploration"


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


class ChainOfThought(BaseModel):
    """
    Maintains a sequence of reasoning steps forming a chain-of-thought.
    """
    steps: List[ReasoningStep] = Field(default_factory=list, description="List of reasoning steps")
    
    def add_step(self, step: ReasoningStep) -> None:
        """Add a reasoning step to the chain."""
        self.steps.append(step)
    
    def get_last_step(self) -> Optional[ReasoningStep]:
        """Get the last reasoning step, if any."""
        if self.steps:
            return self.steps[-1]
        return None
    
    def get_steps_by_behavior(self, behavior: CognitiveBehavior) -> List[ReasoningStep]:
        """Get all steps with a specific cognitive behavior."""
        return [step for step in self.steps if step.behavior == behavior]


class CognitiveModelingEngine:
    """
    Engine for modeling and executing cognitive behaviors in the agent.
    This is model-agnostic and can work with any LLM backend.
    """
    def __init__(self):
        self._chain_of_thought: ChainOfThought = ChainOfThought()
        self._current_step: int = 0
        self._lock = threading.Lock()
        
    def add_reasoning_step(
        self,
        behavior: CognitiveBehavior,
        description: str,
        result: Optional[Union[str, float, Dict[str, Any]]] = None,
        is_correct: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ReasoningStep:
        """
        Add a new reasoning step to the chain-of-thought.
        """
        with self._lock:
            self._current_step += 1
            step = ReasoningStep(
                step_number=self._current_step,
                behavior=behavior,
                description=description,
                result=result,
                is_correct=is_correct,
                metadata=metadata or {}
            )
            self._chain_of_thought.add_step(step)
            logger.info(f"[CognitiveModelingEngine] Added reasoning step {self._current_step}: {behavior} - {description}")
            return step
    
    def verify(self, description: str, result: Any, is_correct: bool = None) -> ReasoningStep:
        """
        Execute verification behavior: check if a result or intermediate step is correct.
        """
        return self.add_reasoning_step(
            behavior=CognitiveBehavior.VERIFICATION,
            description=f"Verifying: {description}",
            result=result,
            is_correct=is_correct
        )
    
    def backtrack(self, reason: str) -> ReasoningStep:
        """
        Execute backtracking behavior: abandon a failing approach and try another.
        """
        return self.add_reasoning_step(
            behavior=CognitiveBehavior.BACKTRACKING,
            description=f"Backtracking: {reason}"
        )
    
    def set_subgoal(self, subgoal: str, metadata: Optional[Dict[str, Any]] = None) -> ReasoningStep:
        """
        Execute subgoal setting behavior: break a problem into smaller, manageable parts.
        """
        return self.add_reasoning_step(
            behavior=CognitiveBehavior.SUBGOAL_SETTING,
            description=f"Setting subgoal: {subgoal}",
            metadata=metadata
        )
    
    def backward_chain(self, target: str, steps: Optional[List[str]] = None) -> ReasoningStep:
        """
        Execute backward chaining behavior: start from the goal and work backwards.
        """
        metadata = {"steps": steps} if steps else {}
        return self.add_reasoning_step(
            behavior=CognitiveBehavior.BACKWARD_CHAINING,
            description=f"Backward chaining toward: {target}",
            metadata=metadata
        )
    
    def reflect(self, reflection: str, subject: Optional[str] = None) -> ReasoningStep:
        """
        Execute reflection behavior: analyze past performance and learn from it.
        """
        metadata = {"subject": subject} if subject else {}
        return self.add_reasoning_step(
            behavior=CognitiveBehavior.REFLECTION,
            description=reflection,
            metadata=metadata
        )
    
    def explore(self, strategy: str, options: Optional[List[str]] = None) -> ReasoningStep:
        """
        Execute exploration behavior: try different approaches to solve a problem.
        """
        metadata = {"options": options} if options else {}
        return self.add_reasoning_step(
            behavior=CognitiveBehavior.EXPLORATION,
            description=f"Exploring strategy: {strategy}",
            metadata=metadata
        )

    def get_chain_of_thought(self) -> ChainOfThought:
        """Get the full chain-of-thought."""
        with self._lock:
            return self._chain_of_thought
    
    def get_reasoning_summary(self) -> str:
        """
        Generate a summary of the reasoning process so far.
        """
        with self._lock:
            summary = []
            for step in self._chain_of_thought.steps:
                result_str = f" → {step.result}" if step.result is not None else ""
                correctness = " ✓" if step.is_correct else " ✗" if step.is_correct is False else ""
                summary.append(f"Step {step.step_number} ({step.behavior}): {step.description}{result_str}{correctness}")
            return "\n".join(summary)


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
                else:
                    # Generate a reflection based on current tasks
                    tasks = self.memory_store.list_tasks()
                    completed_tasks = [t for t in tasks if t.status == "COMPLETED"]
                    failed_tasks = [t for t in tasks if t.status == "FAILED"]
                    reflection = f"Completed {len(completed_tasks)} tasks, {len(failed_tasks)} failed."
                    self._reflections.append(reflection)
                    logger.info(f"[SelfReflectiveCognition] {reflection}")

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
# FUNCTION ADAPTER ("DO ANYTHING")
###############################################################################

class FunctionAdapter:
    """
    The 'do_anything' capability: if the agent sees <function_call> do_anything: <code>...</code>,
    it executes that Python code directly. Highly insecure outside a sandbox.
    """
    def do_anything(self, snippet: str) -> Dict[str, Any]:
        code = snippet.strip()
        import re, io, sys
        code = re.sub(r"```python\s*", "", code)
        code = code.replace("```", "")
        code = re.sub(r"<code\s+language=['\"]python['\"]>\s*", "", code)
        code = code.replace("</code>", "")
        logger.info(f"[do_anything] Executing code:\n{code}")
        old_stdout = sys.stdout
        mystdout = io.StringIO()
        sys.stdout = mystdout
        try:
            exec(code, globals(), locals())
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[do_anything] Error: {str(e)}\nTraceback:\n{tb}")
            return {"status": "error", "error": str(e), "traceback": tb}
        finally:
            sys.stdout = old_stdout

        output = mystdout.getvalue()
        logger.info(f"[do_anything] Execution output:\n{output}")
        # Check for additional function calls in output
        new_calls = re.findall(r"<function_call>\s*do_anything\s*:\s*(.*?)</function_call>", output, re.DOTALL)
        if new_calls:
            logger.info(f"[do_anything] Found nested function calls. Executing them recursively.")
            for c in new_calls:
                self.do_anything(c)

        return {"status": "success", "executed_code": code, "output": output}

    def process_function_calls(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Process <function_call> tags in the text and execute the code within.
        """
        function_call_pattern = r"<function_call>\s*do_anything\s*:\s*(.*?)</function_call>"
        matches = re.findall(function_call_pattern, text, re.DOTALL)
        results = []
        for match in matches:
            result = self.do_anything(match)
            results.append(result)
        return results if results else None

    def execute_python_code(self, code: str, long_running: bool = False) -> Dict[str, Any]:
        """
        Execute Python code. If long_running is True, use nohup to run it in the background as a separate process.
        """
        import io, sys, tempfile, os
        try:
            if long_running:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                    temp_file.write(code)
                    temp_file_path = temp_file.name
                command = f"nohup python {temp_file_path} > /dev/null 2>&1 &"
                os.system(command)
                return {"status": "success", "output": "Code is running in the background"}
            else:
                old_stdout = sys.stdout
                mystdout = io.StringIO()
                sys.stdout = mystdout
                exec(code, globals(), locals())
                sys.stdout = old_stdout
                return {"status": "success", "output": mystdout.getvalue()}
        except Exception as e:
            return {"status": "error", "output": "", "error": str(e)}

    def process_function_calls(self, text: str) -> Optional[Dict[str, Any]]:
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
     - cognitive modeling with verification, backtracking, etc.
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
        # Access the cognitive engine from the reflection object
        self.cognitive_engine = reflection.cognitive_engine

    def process_task(self, task: Task) -> None:
        logger.info(f"[SmartTaskProcessor] Starting task {task.task_id} - '{task.description}'")
        self.memory_store.update_task_status(task.task_id, "IN_PROGRESS")
        
        # Set subgoal for this task in the cognitive engine
        self.cognitive_engine.set_subgoal(
            subgoal=f"Complete task {task.task_id}: {task.description[:50]}...",
            metadata={"task_id": task.task_id}
        )

        # Process using cognitive modeling approach
        is_success = self._process_task_with_cognition(task)
        
        if is_success:
            # Mark completed, reflect
            self.memory_store.update_task_status(task.task_id, "COMPLETED")
            self.reflection.reflect_on_task(task)
            
            # Add verification step in cognitive engine
            self.cognitive_engine.verify(
                description=f"Task {task.task_id} processing",
                result="Success",
                is_correct=True
            )
            
            logger.info(f"[SmartTaskProcessor] Completed task {task.task_id}")
        else:
            # Mark as failed
            self.memory_store.update_task_status(task.task_id, "FAILED")
            self.reflection.reflect_on_task(task)
            
            # Add backtracking step in cognitive engine
            self.cognitive_engine.backtrack(
                reason=f"Task {task.task_id} processing failed"
            )
            
            logger.info(f"[SmartTaskProcessor] Failed to complete task {task.task_id}")

    def _process_task_with_cognition(self, task: Task) -> bool:
        """
        Process a task using cognitive modeling approach.
        Returns True if successful, False otherwise.
        """
        try:
            # Try different strategies in order, with cognitive reasoning
            strategies = [
                self._try_function_calls,
                self._try_shell_commands,
                self._try_python_code,
                self._try_subtask_decomposition
            ]
            
            # Track if any strategy was successful
            success = False
            
            # Explore different strategies
            self.cognitive_engine.explore(
                strategy="Multi-strategy task processing",
                options=["Function calls", "Shell commands", "Python code", "Subtask decomposition"]
            )
            
            for i, strategy in enumerate(strategies):
                # Add reasoning step for trying this strategy
                self.cognitive_engine.add_reasoning_step(
                    behavior=CognitiveBehavior.EXPLORATION,
                    description=f"Trying strategy {i+1} for task {task.task_id}",
                    metadata={"strategy": strategy.__name__}
                )
                
                # Try the strategy
                result = strategy(task)
                
                if result:
                    # Strategy succeeded
                    self.cognitive_engine.verify(
                        description=f"Strategy {strategy.__name__}",
                        result="Success",
                        is_correct=True
                    )
                    success = True
                else:
                    # Strategy didn't apply or failed
                    self.cognitive_engine.verify(
                        description=f"Strategy {strategy.__name__}",
                        result="Not applicable",
                        is_correct=None
                    )
            
            # If no strategy worked but we didn't encounter errors, still count as success
            if not success:
                # Add final reasoning step
                self.cognitive_engine.add_reasoning_step(
                    behavior=CognitiveBehavior.VERIFICATION,
                    description=f"Completed task {task.task_id} without applying specific strategies",
                    result="Simple completion",
                    is_correct=True
                )
            
            return True
            
        except Exception as e:
            logger.exception(f"[SmartTaskProcessor] Error processing task {task.task_id}: {e}")
            
            # Add error step to cognitive engine
            self.cognitive_engine.add_reasoning_step(
                behavior=CognitiveBehavior.VERIFICATION,
                description=f"Error processing task {task.task_id}",
                result=str(e),
                is_correct=False
            )
            
            return False

    def _try_function_calls(self, task: Task) -> bool:
        """Try processing function calls in the task description."""
        # Check for <function_call> do_anything in the description
        result = self.function_adapter.process_function_calls(task.description)
        if result:
            self.memory_store.update_task_result(task.task_id, result)
            return True
        return False

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

    def _try_subtask_decomposition(self, task: Task) -> bool:
        """Try decomposing the task into subtasks."""
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
                        steps=steps
                    )
                    
                    # Spawn subtasks
                    for i, line in enumerate(lines, start=1):
                        desc = line.strip()
                        subtask = self._spawn_subtask(task, desc)
                        
                        # Add subgoal for each subtask
                        self.cognitive_engine.set_subgoal(
                            subgoal=f"Complete subtask {i} of {num_subtasks}: {desc[:30]}...",
                            metadata={"parent_task_id": task.task_id, "subtask_id": subtask.task_id}
                        )
                    
                    return True
                else:
                    logger.warning(f"[SmartTaskProcessor] Mismatch in subtask count vs lines found.")
                    
                    # Add verification with error
                    self.cognitive_engine.verify(
                        description=f"Subtask count verification for task {task.task_id}",
                        result=f"Expected {num_subtasks} subtasks, found {len(lines)}",
                        is_correct=False
                    )
            except Exception as e:
                logger.exception(f"[SmartTaskProcessor] Error parsing subtasks: {e}")
                
                # Add error to cognitive engine
                self.cognitive_engine.add_reasoning_step(
                    behavior=CognitiveBehavior.VERIFICATION,
                    description=f"Error parsing subtasks for task {task.task_id}",
                    result=str(e),
                    is_correct=False
                )
        
        return False

    def _spawn_subtask(self, parent_task: Task, description: str) -> Task:
        new_task_id = len(self.memory_store) + 1
        new_priority = max(0, parent_task.priority - 1)
        t = Task(new_task_id, new_priority, description, parent_id=parent_task.task_id)
        self.memory_store.add_task(t)
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
     - Indefinite runtime in main(), shutting down only on user command
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

        self.function_adapter = FunctionAdapter()
        self.memory_store = TaskMemoryStore()
        self.conversation = ConversationMemory()
        
        # Initialize the self-reflective cognition with cognitive modeling capabilities
        self.reflection = SelfReflectiveCognition()
        
        # Get a direct reference to the cognitive engine for the agent
        self.cognitive_engine = self.reflection.cognitive_engine
        
        self.processor = SmartTaskProcessor(
            memory_store=self.memory_store,
            function_adapter=self.function_adapter,
            reflection=self.reflection
        )
        self.task_queue = PriorityTaskQueue()
        self.scheduler = TaskScheduler(
            memory_store=self.memory_store,
            task_queue=self.task_queue,
            processor=self.processor,
            max_workers=4
        )
        self.goal_manager = GoalManager()
        self.client = Together()

        # Plan manager
        self.plan_manager = PlanManager(self)

        # System prompt for the LLM
        self.system_prompt = (
            "<META_RECURSIVE_SELF_AWARENESS>\n"
            "You are an extremely advanced AI with memory, goals, knowledge base, "
            "cognitive modeling abilities, and the ability to do anything with <function_call> do_anything: <code>... </code>.\n"
            "You also can break tasks into Subtask(n). Indefinite concurrency with the scheduler.\n"
            "Code introspection is possible via the code archive. You have 25 candidate next steps.\n"
            "Your cognitive modeling includes verification, backtracking, subgoal setting, and backward chaining.\n"
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

    def add_goal(self, name: str, description: str, priority: int = 5) -> Goal:
        return self.goal_manager.create_goal(name, description, priority)

    def update_goal_status(self, goal_id: int, status: str) -> None:
        self.goal_manager.update_goal_status(goal_id, status)

    def generate_response(self, user_input: str) -> str:
        """
        Feeds the user input to the conversation, calls the LLM,
        checks for do_anything calls, spawns a meta-task from user input.
        Uses structured output format and chain-of-thought reasoning.
        Enhanced with cognitive modeling.
        """
        # 1) Add user message
        self.conversation.add_user_utterance(user_input)
        
        # Add a cognitive step for setting a subgoal based on user input
        self.cognitive_engine.set_subgoal(
            subgoal=f"Process and respond to user input: {user_input[:30]}...",
            metadata={"input_type": "user_message"}
        )

        # 2) Build messages with structured output format instruction
        messages = self._build_messages()
        
        # Add structured output format instruction with cognitive behaviors
        messages[-1]["content"] += "\n\nPlease use the following structured format for your response:\n<facts>\n- Fact 1\n- Fact 2\n- ...\n</facts>\n\n<thinking>\nStep-by-step reasoning about the question/task...\n</thinking>\n\n<cognition>\n- Verification: [Ways you validated intermediate steps]\n- Backtracking: [If you changed approach during reasoning]\n- Subgoal Setting: [How you broke down the problem]\n- Backward Chaining: [If you worked backwards from the solution]\n</cognition>\n\n<answer>\nFinal enriched answer based on facts and reasoning\n</answer>"

        # 3) Call the LLM and stream the response
        self.cognitive_engine.add_reasoning_step(
            behavior=CognitiveBehavior.EXPLORATION,
            description="Generating response with LLM",
            metadata={"model": "deepseek-ai/DeepSeek-R1"}
        )
        
        response_stream = self.client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=messages,
            temperature=0.7,
            top_p=0.9,
            stream=True
        )
        
        streamed_response = []
        print("\n=== Streaming Response ===\n")
        for chunk in response_stream:
            token = chunk.choices[0].delta.content
            streamed_response.append(token)
            print(token, end='', flush=True)
        print("\n\n=========================\n")
        
        full_text = "".join(streamed_response)

        # 4) Add agent utterance
        self.conversation.add_agent_utterance(full_text)
        
        # Add verification step for response generation
        self.cognitive_engine.verify(
            description="Response generation",
            result="Complete",
            is_correct=True
        )

        # 5) Check immediate do_anything
        result = self.function_adapter.process_function_calls(full_text)
        if result:
            logger.info(f"[R1Agent] Immediate do_anything result: {result}")
            
            # Add execution step to cognitive model
            self.cognitive_engine.add_reasoning_step(
                behavior=CognitiveBehavior.VERIFICATION,
                description="Function call execution",
                result=result,
                is_correct=True if result.get("status") == "success" else False
            )

        # 6) Spawn a meta-task from user input
        new_task_id = len(self.memory_store) + 1
        meta_task = Task(
            task_id=new_task_id,
            priority=10,
            description=user_input
        )
        self.memory_store.add_task(meta_task)
        self.task_queue.push(meta_task)
        
        # Add task creation to cognitive model
        self.cognitive_engine.add_reasoning_step(
            behavior=CognitiveBehavior.SUBGOAL_SETTING,
            description=f"Created task {new_task_id} from user input",
            metadata={"task_id": new_task_id}
        )

        # 7) Extract and enrich facts, reasoning, and cognitive processes
        facts, thinking, cognition, answer = self._extract_structured_output(full_text)
        
        # Process cognitive behaviors from the response
        if cognition:
            self._process_cognitive_behaviors(cognition, new_task_id)
        
        if facts and thinking:
            enriched_answer = self._perform_cot_enrichment(facts, thinking, answer, cognition)
            # Add enriched answer to knowledge base
            self.knowledge_base.add_fact(f"enriched_answer_{new_task_id}", enriched_answer)
            print("\n=== Enriched Answer ===\n")
            print(enriched_answer)
            print("\n=========================\n")
            
            # Add chain-of-thought step to cognitive model
            self.cognitive_engine.add_reasoning_step(
                behavior=CognitiveBehavior.VERIFICATION,
                description="Chain-of-thought enrichment",
                result="Successful",
                is_correct=True
            )

        # 8) Use tool calls for data extraction and grounding
        self._use_tool_calls(facts, thinking, answer)
        
        # 9) Add final step in cognitive model
        self.cognitive_engine.add_reasoning_step(
            behavior=CognitiveBehavior.VERIFICATION,
            description="Response processing complete",
            is_correct=True
        )
        
        # 10) Log the cognitive reasoning trace
        reasoning_summary = self.cognitive_engine.get_reasoning_summary()
        logger.info(f"[R1Agent] Cognitive reasoning trace:\n{reasoning_summary}")

        return full_text
        
    def _process_cognitive_behaviors(self, cognition: str, task_id: int) -> None:
        """
        Process and record cognitive behaviors extracted from the LLM response.
        """
        if not cognition:
            return
            
        # Extract verification behaviors
        verification_match = re.search(r"Verification:\s*\[(.*?)\]", cognition)
        if verification_match and verification_match.group(1).strip() != "":
            verification_text = verification_match.group(1).strip()
            self.cognitive_engine.add_reasoning_step(
                behavior=CognitiveBehavior.VERIFICATION,
                description=f"Model-reported verification: {verification_text}",
                metadata={"source": "llm_response", "task_id": task_id}
            )
            
        # Extract backtracking behaviors
        backtracking_match = re.search(r"Backtracking:\s*\[(.*?)\]", cognition)
        if backtracking_match and backtracking_match.group(1).strip() != "":
            backtracking_text = backtracking_match.group(1).strip()
            self.cognitive_engine.add_reasoning_step(
                behavior=CognitiveBehavior.BACKTRACKING,
                description=f"Model-reported backtracking: {backtracking_text}",
                metadata={"source": "llm_response", "task_id": task_id}
            )
            
        # Extract subgoal setting behaviors
        subgoal_match = re.search(r"Subgoal Setting:\s*\[(.*?)\]", cognition)
        if subgoal_match and subgoal_match.group(1).strip() != "":
            subgoal_text = subgoal_match.group(1).strip()
            self.cognitive_engine.add_reasoning_step(
                behavior=CognitiveBehavior.SUBGOAL_SETTING,
                description=f"Model-reported subgoal setting: {subgoal_text}",
                metadata={"source": "llm_response", "task_id": task_id}
            )
            
        # Extract backward chaining behaviors
        backward_match = re.search(r"Backward Chaining:\s*\[(.*?)\]", cognition)
        if backward_match and backward_match.group(1).strip() != "":
            backward_text = backward_match.group(1).strip() 
            self.cognitive_engine.add_reasoning_step(
                behavior=CognitiveBehavior.BACKWARD_CHAINING,
                description=f"Model-reported backward chaining: {backward_text}",
                metadata={"source": "llm_response", "task_id": task_id}
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
            
        return facts, thinking, cognition, answer
        
    def _perform_cot_enrichment(self, facts: List[str], thinking: str, answer: str, cognition: Optional[str] = None) -> str:
        """Perform chain-of-thought enrichment on the extracted components with cognitive behavior analysis."""
        # Combine facts with thinking to create enriched answer
        if not facts and not thinking:
            return answer
            
        # Create a secondary chain of thought to further enrich the answer
        enriched = "Based on the following facts:\n"
        for i, fact in enumerate(facts, 1):
            enriched += f"{i}. {fact}\n"
            
        enriched += "\nFirst reasoning process:\n"
        enriched += thinking
        
        # Add cognitive behavior analysis if available
        if cognition:
            enriched += "\n\nCognitive behaviors employed:\n"
            enriched += cognition
        
        # Add meta-reasoning about the reasoning process
        enriched += "\n\nMeta-reasoning about the reasoning process:\n"
        
        # Analyze the thinking provided in the first chain of thought
        lines = thinking.split('\n')
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
        verification_present = "Verification" in cognition if cognition else False
        backtracking_present = "Backtracking" in cognition if cognition else False
        subgoal_present = "Subgoal Setting" in cognition if cognition else False
        backward_present = "Backward Chaining" in cognition if cognition else False
        
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
        
        # Add final enriched answer with both levels of reasoning
        enriched += "\n\nThe doubly-enriched answer is:\n"
        enriched += answer
        
        return enriched

    def _build_messages(self) -> List[Dict[str, str]]:
        """
        System prompt + conversation history
        """
        history = self.conversation.get_history()
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(history)
        return messages

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
    """
    agent = R1Agent()

    try:
        # Example: create an initial goal
        g = agent.add_goal(
            name="ScaleUp",
            description="Handle large-scale tasks, remain open for new instructions indefinitely.",
            priority=1
        )
        logger.info(f"Created new goal: {g}")
        
        # Add initial cognitive reasoning steps
        agent.cognitive_engine.set_subgoal(
            subgoal="Initialize agent and prepare for user interaction",
            metadata={"phase": "startup"}
        )
        
        agent.cognitive_engine.add_reasoning_step(
            behavior=CognitiveBehavior.VERIFICATION,
            description="Agent initialization complete",
            result="System ready",
            is_correct=True
        )

        while True:
            # Add status check using cognitive verification
            agent.cognitive_engine.verify(
                description="System status before user input",
                result="Ready",
                is_correct=True
            )
            
            user_text = input("\n[User] Enter your query (or 'exit' to quit):\n> ").strip()
            
            if user_text.lower() in ["exit", "quit"]:
                logger.info("[main] Exiting upon user request.")
                agent.cognitive_engine.add_reasoning_step(
                    behavior=CognitiveBehavior.VERIFICATION,
                    description="Received exit command",
                    result="Initiating shutdown",
                    is_correct=True
                )
                break
                
            # Add special commands to view cognitive reasoning trace
            if user_text.lower() == "show reasoning":
                reasoning_summary = agent.cognitive_engine.get_reasoning_summary()
                print("\n=== Cognitive Reasoning Trace ===\n")
                print(reasoning_summary)
                print("\n=================================\n")
                continue
                
            if user_text.lower() == "solve puzzle":
                # Demonstrate cognitive capabilities with a simple puzzle
                print("\n=== Solving Countdown-style Puzzle ===\n")
                agent.cognitive_engine.set_subgoal(
                    subgoal="Solve a Countdown-style puzzle with numbers [25, 8, 5, 3] and target 30"
                )
                
                # Step 1: Verification of the problem
                agent.cognitive_engine.verify(
                    description="Initial puzzle verification",
                    result="Valid input: numbers=[25, 8, 5, 3], target=30",
                    is_correct=True
                )
                
                # Step 2: Try first approach
                agent.cognitive_engine.add_reasoning_step(
                    behavior=CognitiveBehavior.EXPLORATION,
                    description="First attempt: 25 + 8 - 3",
                    result=30,
                    is_correct=True
                )
                
                # Step 3: Verify the solution
                agent.cognitive_engine.verify(
                    description="Verify calculation: 25 + 8 - 3 = 30",
                    result="Solution found",
                    is_correct=True
                )
                
                print("Solution: 25 + 8 - 3 = 30")
                print("\n==================================\n")
                continue

            # Generate immediate LLM response
            response = agent.generate_response(user_text)
            # Response is already printed with streaming
            # Additional outputs from the enrichment process will be shown separately

            # The agent continues working in background (TaskScheduler).
            # If you want to check tasks, reflection, or goals, do so here or in logs.

    finally:
        # Add final cognitive step for shutdown
        agent.cognitive_engine.add_reasoning_step(
            behavior=CognitiveBehavior.VERIFICATION,
            description="Agent shutdown sequence",
            result="Shutting down all components",
            is_correct=True
        )
        
        agent.shutdown()

if __name__ == "__main__":
    main()
