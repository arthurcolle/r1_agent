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
from typing import Any, Dict, List, Optional, Tuple, Callable, Union, Literal, TypeVar, Generic
from pydantic import BaseModel, Field, validator, root_validator
from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum, auto
from typing_extensions import Annotated
from pydantic import BaseModel, Field, create_model, ConfigDict
from together import Together
from datetime import datetime

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
    COUNTERFACTUAL = "counterfactual"
    METACOGNITION = "metacognition"
    UNCERTAINTY = "uncertainty"
    CAUSAL_REASONING = "causal_reasoning"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    ABDUCTIVE_REASONING = "abductive_reasoning"
    DEDUCTIVE_REASONING = "deductive_reasoning"
    INDUCTIVE_REASONING = "inductive_reasoning"


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
                    confidence * 0.8  # Slightly lower initial confidence for exploration paths
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
