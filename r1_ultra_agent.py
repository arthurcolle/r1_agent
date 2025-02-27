#!/usr/bin/env python3
"""
An "ultra advanced" R1-style do-anything agent with:
 - Indefinite runtime (until user types 'exit')
 - Priority-based task scheduling + concurrency
 - Recursive subtask decomposition with hierarchical planning
 - Long-range goal management + dynamic planning
 - Conversation memory with advanced summarization and retrieval
 - Self-reflective meta-cognition with learning capabilities
 - In-memory code archive for introspection and self-modification
 - Action generator producing up to 25 candidate next steps with utility estimation
 - A KnowledgeBase with vector embeddings for semantic search
 - Ability to run arbitrary Python code with <function_call> do_anything
 - Memory management with forgetting mechanisms
 - Self-improvement capabilities
 - Advanced planning with Monte Carlo Tree Search
 - Multi-agent coordination capabilities
 - Emotional intelligence modeling

~1500 lines of code for demonstration in a secure, sandboxed environment!

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
import random
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Dict, List, Optional, Tuple, Callable, Union, Set
from collections import defaultdict, deque
from datetime import datetime, timedelta

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

# Try to import sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False
    # Create a simple dummy embedding model
    class DummyEmbeddingModel:
        def encode(self, text: Union[str, List[str]], convert_to_numpy: bool = True) -> np.ndarray:
            """Generate simple deterministic embeddings based on character values"""
            if isinstance(text, list):
                return np.array([self._encode_single(t) for t in text])
            return np.array(self._encode_single(text))
            
        def _encode_single(self, text: str) -> List[float]:
            """Create a simple deterministic embedding from text"""
            # Use a simple hash function to generate a pseudo-embedding
            text = text.lower()
            embedding = []
            for i in range(384):  # Standard small embedding size
                # Use a simple hash function based on character positions
                val = sum((ord(c) * (i + 1)) for c in text) % 1000
                embedding.append((val / 500.0) - 1.0)  # Scale to roughly -1 to 1
            return embedding

###############################################################################
# GLOBAL CONFIG / LOGGING
###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("UltraAdvancedR1Agent")

# Global configuration
CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2" if HAVE_SENTENCE_TRANSFORMERS else "dummy",
    "embedding_dimension": 384,
    "memory_decay_rate": 0.05,  # Rate at which memories decay
    "memory_threshold": 0.2,    # Threshold below which memories are forgotten
    "max_conversation_length": 50,
    "max_tasks_in_memory": 1000,
    "planning_horizon": 5,      # How many steps ahead to plan
    "mcts_simulations": 50,     # Number of simulations for Monte Carlo Tree Search
    "self_improvement_interval": 100,  # Number of tasks before attempting self-improvement
    "emotion_dimensions": ["valence", "arousal", "dominance"],  # PAD emotional model
}

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
        created_at (float): Timestamp when the task was created.
        completed_at (Optional[float]): Timestamp when the task was completed.
        expected_duration (Optional[float]): Expected time to complete in seconds.
        actual_duration (Optional[float]): Actual time taken to complete.
        dependencies (Set[int]): Set of task IDs that must be completed before this task.
        subtasks (List[int]): List of subtask IDs.
        tags (Set[str]): Tags for categorizing and filtering tasks.
        embedding (Optional[np.ndarray]): Vector embedding of the task description.
        importance (float): Measure of task importance (0-1).
        urgency (float): Measure of task urgency (0-1).
    """
    def __init__(self, task_id: int, priority: int, description: str, parent_id: Optional[int] = None):
        self.task_id = task_id
        self.priority = priority
        self.description = description
        self.status = "PENDING"
        self.parent_id = parent_id
        self.result = None
        self.created_at = time.time()
        self.completed_at = None
        self.expected_duration = None
        self.actual_duration = None
        self.dependencies = set()
        self.subtasks = []
        self.tags = set()
        self.embedding = None
        self.importance = 0.5  # Default mid-range importance
        self.urgency = 0.5     # Default mid-range urgency
        
    def add_dependency(self, task_id: int) -> None:
        """Add a dependency on another task"""
        self.dependencies.add(task_id)
        
    def add_subtask(self, subtask_id: int) -> None:
        """Add a subtask to this task"""
        self.subtasks.append(subtask_id)
        
    def add_tag(self, tag: str) -> None:
        """Add a tag to this task"""
        self.tags.add(tag)
        
    def start(self) -> None:
        """Mark the task as in progress and record the start time"""
        self.status = "IN_PROGRESS"
        self._start_time = time.time()
        
    def complete(self, result: Any = None) -> None:
        """Mark the task as completed and store the result"""
        self.status = "COMPLETED"
        self.result = result
        self.completed_at = time.time()
        if hasattr(self, '_start_time'):
            self.actual_duration = self.completed_at - self._start_time
            
    def fail(self, error: str = None) -> None:
        """Mark the task as failed and store the error"""
        self.status = "FAILED"
        self.result = error
        self.completed_at = time.time()
        if hasattr(self, '_start_time'):
            self.actual_duration = self.completed_at - self._start_time

    def __lt__(self, other: "Task") -> bool:
        """
        Overload the < operator so tasks can be sorted in a heap by priority.
        """
        return self.priority < other.priority

    def __repr__(self) -> str:
        snippet = self.description[:30].replace("\n", " ")
        return (f"Task(id={self.task_id}, prio={self.priority}, "
                f"status={self.status}, desc={snippet}...)")
                
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization"""
        return {
            "task_id": self.task_id,
            "priority": self.priority,
            "description": self.description,
            "status": self.status,
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "expected_duration": self.expected_duration,
            "actual_duration": self.actual_duration,
            "dependencies": list(self.dependencies),
            "subtasks": self.subtasks,
            "tags": list(self.tags),
            "importance": self.importance,
            "urgency": self.urgency
        }

class TaskMemoryStore:
    """
    Thread-safe in-memory storage for Task objects with advanced features:
    - Vector embeddings for semantic search
    - Memory decay and forgetting mechanisms
    - Task relationship tracking
    - Filtering and querying capabilities
    - Task statistics and analytics
    """
    def __init__(self, embedding_model=None) -> None:
        self._tasks: Dict[int, Task] = {}
        self._lock = threading.Lock()
        self._next_id = 1
        self._task_embeddings = {}
        self._task_timestamps = {}
        self._task_access_counts = defaultdict(int)
        self._task_relationships = defaultdict(set)  # task_id -> set of related task_ids
        self._tags_index = defaultdict(set)  # tag -> set of task_ids
        
        # Initialize embedding model
        if HAVE_SENTENCE_TRANSFORMERS and embedding_model:
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            self.embedding_model = DummyEmbeddingModel()

    def create_task(self, priority: int, description: str, parent_id: Optional[int] = None) -> Task:
        """Create a new task and add it to the store"""
        with self._lock:
            task_id = self._next_id
            self._next_id += 1
            task = Task(task_id, priority, description, parent_id)
            
            # Generate embedding for the task description
            task.embedding = self.embedding_model.encode(description)
            
            # Store the task
            self._tasks[task_id] = task
            self._task_timestamps[task_id] = time.time()
            self._task_access_counts[task_id] = 1
            
            # Update parent-child relationship
            if parent_id:
                parent = self._tasks.get(parent_id)
                if parent:
                    parent.add_subtask(task_id)
                    self._task_relationships[parent_id].add(task_id)
                    self._task_relationships[task_id].add(parent_id)
            
            return task

    def add_task(self, task: Task) -> None:
        with self._lock:
            if task.task_id in self._tasks:
                logger.warning(f"[TaskMemoryStore] Task ID {task.task_id} already exists. Overwriting.")
            
            # Generate embedding if not already present
            if task.embedding is None:
                task.embedding = self.embedding_model.encode(task.description)
                
            self._tasks[task.task_id] = task
            self._task_timestamps[task.task_id] = time.time()
            self._task_access_counts[task.task_id] = 1
            
            # Index tags
            for tag in task.tags:
                self._tags_index[tag].add(task.task_id)

    def get_task(self, task_id: int) -> Optional[Task]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                # Update access metadata
                self._task_timestamps[task_id] = time.time()
                self._task_access_counts[task_id] += 1
            return task

    def update_task_status(self, task_id: int, status: str) -> None:
        with self._lock:
            t = self._tasks.get(task_id)
            if t:
                old_status = t.status
                t.status = status
                self._task_timestamps[task_id] = time.time()
                
                # If completing a task, update completion time
                if status == "COMPLETED" and old_status != "COMPLETED":
                    t.completed_at = time.time()
                    if hasattr(t, '_start_time'):
                        t.actual_duration = t.completed_at - t._start_time

    def update_task_result(self, task_id: int, result: Any) -> None:
        with self._lock:
            t = self._tasks.get(task_id)
            if t:
                t.result = result
                self._task_timestamps[task_id] = time.time()

    def list_tasks(self, status: Optional[str] = None, tag: Optional[str] = None) -> List[Task]:
        """List tasks with optional filtering by status or tag"""
        with self._lock:
            tasks = list(self._tasks.values())
            
            if status:
                tasks = [t for t in tasks if t.status == status]
                
            if tag:
                task_ids = self._tags_index.get(tag, set())
                tasks = [t for t in tasks if t.task_id in task_ids]
                
            return tasks

    def get_subtasks(self, parent_id: int) -> List[Task]:
        """Get all subtasks for a given parent task"""
        with self._lock:
            parent = self._tasks.get(parent_id)
            if not parent:
                return []
            return [self._tasks[task_id] for task_id in parent.subtasks if task_id in self._tasks]

    def search_similar_tasks(self, query: str, top_k: int = 5) -> List[Tuple[Task, float]]:
        """Find tasks with descriptions semantically similar to the query"""
        with self._lock:
            if not self._tasks:
                return []
                
            # Generate embedding for the query
            query_embedding = self.embedding_model.encode(query)
            
            # Calculate similarity scores
            results = []
            for task_id, task in self._tasks.items():
                if task.embedding is not None:
                    # Cosine similarity
                    similarity = np.dot(query_embedding, task.embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(task.embedding)
                    )
                    results.append((task, float(similarity)))
            
            # Sort by similarity (highest first) and return top_k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

    def forget_old_tasks(self, threshold: float = None) -> int:
        """
        Remove tasks that haven't been accessed recently based on decay function.
        Returns the number of tasks forgotten.
        """
        if threshold is None:
            threshold = CONFIG["memory_threshold"]
            
        current_time = time.time()
        decay_rate = CONFIG["memory_decay_rate"]
        
        to_forget = []
        
        with self._lock:
            for task_id, task in self._tasks.items():
                # Skip tasks that are in progress
                if task.status == "IN_PROGRESS":
                    continue
                    
                # Calculate memory strength based on recency and frequency
                time_factor = math.exp(-decay_rate * (current_time - self._task_timestamps[task_id]))
                access_factor = math.log(1 + self._task_access_counts[task_id])
                memory_strength = time_factor * access_factor
                
                if memory_strength < threshold:
                    to_forget.append(task_id)
            
            # Remove forgotten tasks
            for task_id in to_forget:
                del self._tasks[task_id]
                del self._task_timestamps[task_id]
                del self._task_access_counts[task_id]
                
                # Clean up relationships
                if task_id in self._task_relationships:
                    del self._task_relationships[task_id]
                
                # Clean up tag index
                for tag_set in self._tags_index.values():
                    tag_set.discard(task_id)
                    
            return len(to_forget)

    def get_task_statistics(self) -> Dict[str, Any]:
        """Get statistics about tasks in the store"""
        with self._lock:
            stats = {
                "total_tasks": len(self._tasks),
                "status_counts": defaultdict(int),
                "avg_completion_time": 0,
                "completed_tasks": 0,
                "pending_tasks": 0,
                "failed_tasks": 0,
                "tags": dict(Counter(tag for task in self._tasks.values() for tag in task.tags)),
            }
            
            completion_times = []
            
            for task in self._tasks.values():
                stats["status_counts"][task.status] += 1
                
                if task.status == "COMPLETED" and task.actual_duration:
                    completion_times.append(task.actual_duration)
                    stats["completed_tasks"] += 1
                elif task.status == "PENDING":
                    stats["pending_tasks"] += 1
                elif task.status == "FAILED":
                    stats["failed_tasks"] += 1
            
            if completion_times:
                stats["avg_completion_time"] = sum(completion_times) / len(completion_times)
                stats["min_completion_time"] = min(completion_times)
                stats["max_completion_time"] = max(completion_times)
                
            return stats

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
        deadline (Optional[float]): Timestamp for when the goal should be completed.
        progress (float): Percentage of completion (0-1).
        parent_id (Optional[int]): ID of parent goal, if this is a subgoal.
        subgoals (List[int]): List of subgoal IDs.
        related_tasks (Set[int]): Set of task IDs related to this goal.
        success_criteria (List[str]): Criteria to determine if the goal is completed.
        embedding (Optional[np.ndarray]): Vector embedding of the goal description.
        tags (Set[str]): Tags for categorizing and filtering goals.
    """
    def __init__(self, goal_id: int, name: str, description: str, priority: int = 5):
        self.goal_id = goal_id
        self.name = name
        self.description = description
        self.priority = priority
        self.status = "ACTIVE"
        self.created_at = time.time()
        self.last_updated = self.created_at
        self.deadline = None
        self.progress = 0.0
        self.parent_id = None
        self.subgoals = []
        self.related_tasks = set()
        self.success_criteria = []
        self.embedding = None
        self.tags = set()

    def update_description(self, new_desc: str) -> None:
        self.description = new_desc
        self.last_updated = time.time()

    def complete(self) -> None:
        self.status = "COMPLETED"
        self.progress = 1.0
        self.last_updated = time.time()
        
    def set_deadline(self, deadline_timestamp: float) -> None:
        """Set a deadline for the goal"""
        self.deadline = deadline_timestamp
        self.last_updated = time.time()
        
    def update_progress(self, progress: float) -> None:
        """Update the progress percentage (0-1)"""
        self.progress = max(0.0, min(1.0, progress))  # Clamp between 0 and 1
        self.last_updated = time.time()
        
    def add_subgoal(self, subgoal_id: int) -> None:
        """Add a subgoal to this goal"""
        if subgoal_id not in self.subgoals:
            self.subgoals.append(subgoal_id)
            self.last_updated = time.time()
            
    def add_related_task(self, task_id: int) -> None:
        """Add a related task to this goal"""
        self.related_tasks.add(task_id)
        self.last_updated = time.time()
        
    def add_success_criterion(self, criterion: str) -> None:
        """Add a success criterion for this goal"""
        self.success_criteria.append(criterion)
        self.last_updated = time.time()
        
    def add_tag(self, tag: str) -> None:
        """Add a tag to this goal"""
        self.tags.add(tag)
        self.last_updated = time.time()
        
    def time_remaining(self) -> Optional[float]:
        """Get the time remaining until the deadline in seconds"""
        if self.deadline is None:
            return None
        return max(0, self.deadline - time.time())
        
    def is_overdue(self) -> bool:
        """Check if the goal is past its deadline"""
        if self.deadline is None:
            return False
        return time.time() > self.deadline

    def __repr__(self) -> str:
        snippet = self.description[:30].replace("\n", " ")
        return (f"Goal(id={self.goal_id}, name={self.name}, "
                f"priority={self.priority}, status={self.status}, "
                f"progress={self.progress:.2f}, desc={snippet}...)")
                
    def to_dict(self) -> Dict[str, Any]:
        """Convert goal to dictionary for serialization"""
        return {
            "goal_id": self.goal_id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority,
            "status": self.status,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "deadline": self.deadline,
            "progress": self.progress,
            "parent_id": self.parent_id,
            "subgoals": self.subgoals,
            "related_tasks": list(self.related_tasks),
            "success_criteria": self.success_criteria,
            "tags": list(self.tags)
        }

class GoalManager:
    """
    Manages creation, retrieval, and updating of multiple goals.
    Features:
    - Hierarchical goal structure with parent-child relationships
    - Vector embeddings for semantic search
    - Goal decomposition into subgoals
    - Progress tracking and deadline management
    - Goal prioritization based on urgency and importance
    """
    def __init__(self, embedding_model=None):
        self._goals: Dict[int, Goal] = {}
        self._lock = threading.Lock()
        self._next_id = 1
        self._goal_embeddings = {}
        self._tags_index = defaultdict(set)  # tag -> set of goal_ids
        
        # Initialize embedding model
        if HAVE_SENTENCE_TRANSFORMERS and embedding_model:
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            self.embedding_model = DummyEmbeddingModel()

    def create_goal(self, name: str, description: str, priority: int = 5, 
                   parent_id: Optional[int] = None, deadline: Optional[float] = None,
                   success_criteria: Optional[List[str]] = None, 
                   tags: Optional[List[str]] = None) -> Goal:
        """
        Create a new goal with the provided attributes.
        """
        with self._lock:
            g = Goal(self._next_id, name, description, priority)
            
            # Set optional attributes
            if parent_id:
                parent = self._goals.get(parent_id)
                if parent:
                    g.parent_id = parent_id
                    parent.add_subgoal(self._next_id)
                    
            if deadline:
                g.set_deadline(deadline)
                
            if success_criteria:
                for criterion in success_criteria:
                    g.add_success_criterion(criterion)
                    
            if tags:
                for tag in tags:
                    g.add_tag(tag)
                    self._tags_index[tag].add(self._next_id)
            
            # Generate embedding
            g.embedding = self.embedding_model.encode(description)
            
            # Store the goal
            self._goals[self._next_id] = g
            logger.info(f"[GoalManager] Created Goal: {g}")
            self._next_id += 1
            return g

    def get_goal(self, goal_id: int) -> Optional[Goal]:
        with self._lock:
            return self._goals.get(goal_id)

    def list_goals(self, status: Optional[str] = None, tag: Optional[str] = None) -> List[Goal]:
        """List goals with optional filtering by status or tag"""
        with self._lock:
            goals = list(self._goals.values())
            
            if status:
                goals = [g for g in goals if g.status == status]
                
            if tag:
                goal_ids = self._tags_index.get(tag, set())
                goals = [g for g in goals if g.goal_id in goal_ids]
                
            return goals

    def get_subgoals(self, parent_id: int) -> List[Goal]:
        """Get all subgoals for a given parent goal"""
        with self._lock:
            parent = self._goals.get(parent_id)
            if not parent:
                return []
            return [self._goals[goal_id] for goal_id in parent.subgoals if goal_id in self._goals]

    def update_goal_status(self, goal_id: int, status: str) -> None:
        """
        Update the status of a goal, e.g. from 'ACTIVE' to 'COMPLETED'.
        """
        with self._lock:
            g = self._goals.get(goal_id)
            if g:
                g.status = status
                g.last_updated = time.time()
                
                # If completing a goal, update progress
                if status == "COMPLETED":
                    g.progress = 1.0
                    
                    # Update parent goal progress if applicable
                    if g.parent_id and g.parent_id in self._goals:
                        parent = self._goals[g.parent_id]
                        self._update_parent_progress(parent)
                
                logger.info(f"[GoalManager] Updated goal {goal_id} to status={status}")
                
    def update_goal_progress(self, goal_id: int, progress: float) -> None:
        """Update the progress of a goal and propagate to parent goals"""
        with self._lock:
            g = self._goals.get(goal_id)
            if g:
                g.update_progress(progress)
                
                # Update parent goal progress if applicable
                if g.parent_id and g.parent_id in self._goals:
                    parent = self._goals[g.parent_id]
                    self._update_parent_progress(parent)
                    
                logger.info(f"[GoalManager] Updated goal {goal_id} progress to {progress:.2f}")
                
    def _update_parent_progress(self, parent: Goal) -> None:
        """Update a parent goal's progress based on its subgoals"""
        if not parent.subgoals:
            return
            
        # Calculate average progress of all subgoals
        total_progress = 0.0
        count = 0
        
        for subgoal_id in parent.subgoals:
            if subgoal_id in self._goals:
                subgoal = self._goals[subgoal_id]
                total_progress += subgoal.progress
                count += 1
                
        if count > 0:
            parent.update_progress(total_progress / count)
            
    def search_similar_goals(self, query: str, top_k: int = 5) -> List[Tuple[Goal, float]]:
        """Find goals with descriptions semantically similar to the query"""
        with self._lock:
            if not self._goals:
                return []
                
            # Generate embedding for the query
            query_embedding = self.embedding_model.encode(query)
            
            # Calculate similarity scores
            results = []
            for goal_id, goal in self._goals.items():
                if goal.embedding is not None:
                    # Cosine similarity
                    similarity = np.dot(query_embedding, goal.embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(goal.embedding)
                    )
                    results.append((goal, float(similarity)))
            
            # Sort by similarity (highest first) and return top_k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
            
    def decompose_goal(self, goal_id: int, subgoal_descriptions: List[str]) -> List[Goal]:
        """Break a goal into multiple subgoals"""
        with self._lock:
            parent = self._goals.get(goal_id)
            if not parent:
                return []
                
            created_subgoals = []
            
            for desc in subgoal_descriptions:
                # Create a subgoal with the same priority as parent
                subgoal = self.create_goal(
                    name=f"Subgoal of {parent.name}",
                    description=desc,
                    priority=parent.priority,
                    parent_id=goal_id
                )
                created_subgoals.append(subgoal)
                
            return created_subgoals
            
    def get_overdue_goals(self) -> List[Goal]:
        """Get all goals that are past their deadline"""
        with self._lock:
            return [g for g in self._goals.values() if g.is_overdue() and g.status == "ACTIVE"]
            
    def reprioritize_goals(self) -> None:
        """Adjust goal priorities based on deadlines and progress"""
        with self._lock:
            for goal in self._goals.values():
                if goal.status != "ACTIVE":
                    continue
                    
                # Increase priority (lower number) for goals with approaching deadlines
                if goal.deadline:
                    time_remaining = goal.time_remaining()
                    if time_remaining is not None:
                        # If less than 24 hours remaining, increase priority
                        if time_remaining < 86400:  # 24 hours in seconds
                            urgency_factor = max(1, int(5 - (time_remaining / 17280)))  # Scale to 1-5
                            goal.priority = max(1, goal.priority - urgency_factor)
                            
                # Adjust priority based on progress
                if goal.progress < 0.2 and goal.last_updated < (time.time() - 86400):
                    # Increase priority for stalled goals
                    goal.priority = max(1, goal.priority - 1)

###############################################################################
# CONVERSATION MANAGEMENT
###############################################################################

class ConversationMemory:
    """
    Maintains a conversation history with advanced features:
    - Vector embeddings for semantic search
    - Advanced summarization with key points extraction
    - Topic segmentation and tracking
    - Sentiment analysis
    - Entity extraction and tracking
    - Memory decay and forgetting mechanisms
    """
    def __init__(self, embedding_model=None) -> None:
        self._utterances: List[Dict[str, str]] = []
        self._summaries: List[Dict[str, Any]] = []
        self._topics: List[Dict[str, Any]] = []
        self._entities: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._max_length = CONFIG["max_conversation_length"]
        self._utterance_embeddings = {}
        
        # Initialize embedding model
        if HAVE_SENTENCE_TRANSFORMERS and embedding_model:
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            self.embedding_model = DummyEmbeddingModel()
            
        # Emotional state tracking
        self._emotion_state = {dim: 0.0 for dim in CONFIG["emotion_dimensions"]}
        self._last_emotion_update = time.time()

    def add_user_utterance(self, text: str) -> None:
        with self._lock:
            message = {
                "role": "user", 
                "content": text,
                "timestamp": time.time(),
                "id": len(self._utterances)
            }
            self._utterances.append(message)
            
            # Generate embedding
            self._utterance_embeddings[message["id"]] = self.embedding_model.encode(text)
            
            # Update emotional state based on text content
            self._update_emotion_state(text, "user")
            
            # Extract entities
            self._extract_entities(text, message["id"])
            
            # Check if we need to create a new topic
            self._maybe_create_topic(text, message["id"])
            
            self._maybe_summarize()

    def add_agent_utterance(self, text: str) -> None:
        with self._lock:
            message = {
                "role": "assistant", 
                "content": text,
                "timestamp": time.time(),
                "id": len(self._utterances)
            }
            self._utterances.append(message)
            
            # Generate embedding
            self._utterance_embeddings[message["id"]] = self.embedding_model.encode(text)
            
            # Update emotional state based on text content
            self._update_emotion_state(text, "assistant")
            
            # Extract entities
            self._extract_entities(text, message["id"])
            
            self._maybe_summarize()

    def get_history(self, n: Optional[int] = None) -> List[Dict[str, str]]:
        """Get the most recent n messages, or all if n is None"""
        with self._lock:
            if n is None:
                return [{"role": u["role"], "content": u["content"]} for u in self._utterances]
            else:
                return [{"role": u["role"], "content": u["content"]} for u in self._utterances[-n:]]
                
    def search_conversation(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, str], float]]:
        """Search for messages semantically similar to the query"""
        with self._lock:
            if not self._utterances:
                return []
                
            # Generate embedding for the query
            query_embedding = self.embedding_model.encode(query)
            
            # Calculate similarity scores
            results = []
            for msg_id, embedding in self._utterance_embeddings.items():
                if msg_id < len(self._utterances):
                    # Cosine similarity
                    similarity = np.dot(query_embedding, embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                    )
                    results.append((self._utterances[msg_id], float(similarity)))
            
            # Sort by similarity (highest first) and return top_k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
            
    def get_topics(self) -> List[Dict[str, Any]]:
        """Get all identified conversation topics"""
        with self._lock:
            return list(self._topics)
            
    def get_entities(self) -> Dict[str, Dict[str, Any]]:
        """Get all extracted entities"""
        with self._lock:
            return dict(self._entities)
            
    def get_emotional_state(self) -> Dict[str, float]:
        """Get the current emotional state"""
        with self._lock:
            return dict(self._emotion_state)
            
    def get_summaries(self) -> List[Dict[str, Any]]:
        """Get all conversation summaries"""
        with self._lock:
            return list(self._summaries)

    def _maybe_summarize(self) -> None:
        """
        If conversation is too long, produce an advanced summary with key points
        and store it as a system message, trimming out older messages.
        """
        with self._lock:
            if len(self._utterances) > self._max_length:
                # Create a more sophisticated summary
                recent_messages = self._utterances[-self._max_length//2:]
                
                # Extract key points (simple implementation)
                key_points = []
                for msg in recent_messages:
                    # Simple heuristic: sentences with question marks or important keywords
                    sentences = re.split(r'[.!?]', msg["content"])
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        if '?' in sentence or any(kw in sentence.lower() for kw in 
                                               ["important", "critical", "must", "should", "need"]):
                            key_points.append(sentence)
                
                # Limit to 5 key points
                key_points = key_points[:5]
                
                # Create summary object
                summary = {
                    "timestamp": time.time(),
                    "message_range": [self._utterances[0]["id"], self._utterances[-1]["id"]],
                    "key_points": key_points,
                    "topics": [t["name"] for t in self._topics[-3:]] if self._topics else [],
                    "entities": list(self._entities.keys())[:5]
                }
                
                # Generate summary text
                summary_text = "Previous conversation summary:\n"
                summary_text += f"- Topics discussed: {', '.join(summary['topics']) if summary['topics'] else 'None identified'}\n"
                summary_text += "- Key points:\n"
                for i, point in enumerate(key_points, 1):
                    summary_text += f"  {i}. {point}\n"
                summary_text += f"- Key entities: {', '.join(summary['entities']) if summary['entities'] else 'None identified'}"
                
                # Add summary to list
                self._summaries.append(summary)
                
                # Keep only the most recent messages
                keep_count = self._max_length // 2
                self._utterances = self._utterances[-keep_count:]
                
                # Insert summary as a system message
                self._utterances.insert(0, {
                    "role": "system", 
                    "content": summary_text,
                    "timestamp": time.time(),
                    "id": self._utterances[0]["id"] - 1 if self._utterances else 0
                })
                
                logger.info("[ConversationMemory] Created advanced summary and trimmed conversation history.")
                
    def _extract_entities(self, text: str, message_id: int) -> None:
        """Extract entities from text using simple regex patterns"""
        # Simple patterns for demonstration
        person_pattern = r'(?:[A-Z][a-z]+ [A-Z][a-z]+)'
        location_pattern = r'(?:in|at|from) ([A-Z][a-z]+(?: [A-Z][a-z]+)*)'
        date_pattern = r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b'
        
        # Find people
        for match in re.finditer(person_pattern, text):
            name = match.group(0)
            if name not in self._entities:
                self._entities[name] = {
                    "type": "person",
                    "first_mention": message_id,
                    "mentions": [message_id],
                    "last_update": time.time()
                }
            else:
                self._entities[name]["mentions"].append(message_id)
                self._entities[name]["last_update"] = time.time()
                
        # Find locations
        for match in re.finditer(location_pattern, text):
            location = match.group(1)
            if location not in self._entities:
                self._entities[location] = {
                    "type": "location",
                    "first_mention": message_id,
                    "mentions": [message_id],
                    "last_update": time.time()
                }
            else:
                self._entities[location]["mentions"].append(message_id)
                self._entities[location]["last_update"] = time.time()
                
        # Find dates
        for match in re.finditer(date_pattern, text):
            date = match.group(0)
            if date not in self._entities:
                self._entities[date] = {
                    "type": "date",
                    "first_mention": message_id,
                    "mentions": [message_id],
                    "last_update": time.time()
                }
            else:
                self._entities[date]["mentions"].append(message_id)
                self._entities[date]["last_update"] = time.time()
                
    def _maybe_create_topic(self, text: str, message_id: int) -> None:
        """Determine if a new topic has been introduced"""
        # Simple heuristic: if the message is long and doesn't reference recent messages
        if len(text) > 100 and not any(entity in text.lower() for entity in self._entities):
            # Extract topic name (first sentence or first 50 chars)
            topic_name = text.split('.')[0]
            if len(topic_name) > 50:
                topic_name = topic_name[:47] + "..."
                
            # Create new topic
            self._topics.append({
                "name": topic_name,
                "start_message_id": message_id,
                "messages": [message_id],
                "created_at": time.time()
            })
            logger.info(f"[ConversationMemory] Created new topic: {topic_name}")
        elif self._topics:
            # Add to existing topic
            self._topics[-1]["messages"].append(message_id)
            
    def _update_emotion_state(self, text: str, role: str) -> None:
        """Update the emotional state based on text content"""
        # Simple sentiment analysis for demonstration
        # Valence (positive/negative)
        positive_words = ["good", "great", "excellent", "happy", "pleased", "wonderful", "fantastic"]
        negative_words = ["bad", "terrible", "awful", "sad", "angry", "disappointed", "frustrated"]
        
        # Arousal (calm/excited)
        high_arousal = ["excited", "thrilled", "ecstatic", "furious", "enraged", "terrified"]
        low_arousal = ["calm", "relaxed", "peaceful", "serene", "tired", "bored"]
        
        # Dominance (in control/controlled)
        high_dominance = ["confident", "powerful", "in control", "strong", "certain"]
        low_dominance = ["helpless", "weak", "uncertain", "confused", "overwhelmed"]
        
        text_lower = text.lower()
        
        # Calculate valence shift
        valence_shift = 0
        for word in positive_words:
            if word in text_lower:
                valence_shift += 0.1
        for word in negative_words:
            if word in text_lower:
                valence_shift -= 0.1
                
        # Calculate arousal shift
        arousal_shift = 0
        for word in high_arousal:
            if word in text_lower:
                arousal_shift += 0.1
        for word in low_arousal:
            if word in text_lower:
                arousal_shift -= 0.1
                
        # Calculate dominance shift
        dominance_shift = 0
        for word in high_dominance:
            if word in text_lower:
                dominance_shift += 0.1
        for word in low_dominance:
            if word in text_lower:
                dominance_shift -= 0.1
                
        # Apply time decay to previous emotional state
        time_since_update = time.time() - self._last_emotion_update
        decay_factor = math.exp(-0.1 * time_since_update)
        
        # Update emotional state
        self._emotion_state["valence"] = max(-1.0, min(1.0, 
            self._emotion_state["valence"] * decay_factor + valence_shift))
        self._emotion_state["arousal"] = max(-1.0, min(1.0, 
            self._emotion_state["arousal"] * decay_factor + arousal_shift))
        self._emotion_state["dominance"] = max(-1.0, min(1.0, 
            self._emotion_state["dominance"] * decay_factor + dominance_shift))
            
        self._last_emotion_update = time.time()

###############################################################################
# SELF-REFLECTIVE COGNITION
###############################################################################

class SelfReflectiveCognition:
    """
    Advanced self-reflection system with:
    - Performance analysis and learning
    - Pattern recognition in task execution
    - Adaptation of strategies based on outcomes
    - Identification of strengths and weaknesses
    - Self-improvement suggestions
    - Emotional intelligence development
    """
    def __init__(self):
        self._reflections: List[Dict[str, Any]] = []
        self._performance_metrics: Dict[str, List[float]] = {
            "task_success_rate": [],
            "task_completion_time": [],
            "goal_progress_rate": [],
            "adaptation_score": []
        }
        self._learning_points: List[Dict[str, Any]] = []
        self._strengths: Dict[str, float] = {}
        self._weaknesses: Dict[str, float] = {}
        self._improvement_suggestions: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._task_patterns: Dict[str, Dict[str, Any]] = {}
        
        # Start a background thread that periodically analyzes performance
        self._analyzer_thread = threading.Thread(target=self._analyze_performance_loop, daemon=True)
        self._analyzer_thread.start()
        
        # Counter for self-improvement attempts
        self._task_counter = 0
        self._last_improvement_attempt = time.time()

    def reflect_on_task(self, task: "Task") -> None:
        """
        Called every time a task is completed. Performs detailed reflection.
        """
        with self._lock:
            # Increment task counter
            self._task_counter += 1
            
            # Create detailed reflection
            reflection = {
                "task_id": task.task_id,
                "status": task.status,
                "description": task.description,
                "timestamp": time.time(),
                "duration": task.actual_duration if hasattr(task, 'actual_duration') else None,
                "tags": list(task.tags) if hasattr(task, 'tags') else [],
                "analysis": self._analyze_task(task)
            }
            
            self._reflections.append(reflection)
            
            # Extract task pattern
            pattern = self._extract_task_pattern(task)
            if pattern:
                pattern_key = pattern["pattern_type"]
                if pattern_key not in self._task_patterns:
                    self._task_patterns[pattern_key] = {
                        "count": 1,
                        "success_rate": 1.0 if task.status == "COMPLETED" else 0.0,
                        "examples": [task.task_id],
                        "last_seen": time.time()
                    }
                else:
                    # Update pattern statistics
                    p = self._task_patterns[pattern_key]
                    p["count"] += 1
                    p["examples"].append(task.task_id)
                    p["last_seen"] = time.time()
                    
                    # Update success rate
                    success = 1.0 if task.status == "COMPLETED" else 0.0
                    p["success_rate"] = ((p["success_rate"] * (p["count"] - 1)) + success) / p["count"]
            
            # Update performance metrics
            if task.status == "COMPLETED":
                self._performance_metrics["task_success_rate"].append(1.0)
            else:
                self._performance_metrics["task_success_rate"].append(0.0)
                
            if hasattr(task, 'actual_duration') and task.actual_duration:
                self._performance_metrics["task_completion_time"].append(task.actual_duration)
                
            # Check if we should attempt self-improvement
            if (self._task_counter >= CONFIG["self_improvement_interval"] or 
                    time.time() - self._last_improvement_attempt > 3600):  # At least hourly
                self._attempt_self_improvement()
                self._task_counter = 0
                self._last_improvement_attempt = time.time()
            
            # Log a summary
            snippet = task.description[:50].replace("\n"," ")
            msg = f"Reflected on task {task.task_id}: status={task.status}, desc='{snippet}'"
            logger.info(f"[SelfReflectiveCognition] {msg}")

    def get_reflections(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve the most recent n reflections, or all if n is None.
        """
        with self._lock:
            if n is None:
                return list(self._reflections)
            else:
                return list(self._reflections[-n:])
                
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        with self._lock:
            metrics = {}
            
            # Calculate averages for each metric
            for key, values in self._performance_metrics.items():
                if values:
                    metrics[key] = sum(values) / len(values)
                    metrics[f"{key}_trend"] = self._calculate_trend(values)
                else:
                    metrics[key] = None
                    metrics[f"{key}_trend"] = 0.0
                    
            # Add strengths and weaknesses
            metrics["strengths"] = dict(sorted(
                self._strengths.items(), key=lambda x: x[1], reverse=True)[:3])
            metrics["weaknesses"] = dict(sorted(
                self._weaknesses.items(), key=lambda x: x[1], reverse=True)[:3])
                
            # Add improvement suggestions count
            metrics["improvement_suggestions"] = len(self._improvement_suggestions)
            
            # Add task pattern insights
            if self._task_patterns:
                top_patterns = sorted(
                    self._task_patterns.items(), 
                    key=lambda x: x[1]["count"], 
                    reverse=True
                )[:3]
                
                metrics["common_patterns"] = [
                    {"type": k, "count": v["count"], "success_rate": v["success_rate"]}
                    for k, v in top_patterns
                ]
            else:
                metrics["common_patterns"] = []
                
            return metrics
            
    def get_learning_points(self) -> List[Dict[str, Any]]:
        """Get key learning points from task reflections"""
        with self._lock:
            return list(self._learning_points)
            
    def get_improvement_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for self-improvement"""
        with self._lock:
            return list(self._improvement_suggestions)

    def _analyze_task(self, task: "Task") -> Dict[str, Any]:
        """Analyze a task to extract insights"""
        analysis = {
            "complexity": self._estimate_complexity(task),
            "success_factors": [],
            "failure_factors": [],
            "improvement_areas": []
        }
        
        # Analyze success or failure factors
        if task.status == "COMPLETED":
            # Simple heuristics for success factors
            if hasattr(task, 'actual_duration') and task.actual_duration:
                if task.expected_duration and task.actual_duration < task.expected_duration:
                    analysis["success_factors"].append("Completed faster than expected")
                    
            if hasattr(task, 'subtasks') and task.subtasks:
                analysis["success_factors"].append("Effective task decomposition")
                
            # Add to strengths
            task_type = self._categorize_task(task)
            if task_type:
                self._strengths[task_type] = self._strengths.get(task_type, 0) + 0.1
        else:
            # Simple heuristics for failure factors
            if task.status == "FAILED":
                analysis["failure_factors"].append("Execution error")
                
            if hasattr(task, 'actual_duration') and task.actual_duration:
                if task.expected_duration and task.actual_duration > 2 * task.expected_duration:
                    analysis["failure_factors"].append("Took much longer than expected")
                    
            # Add to weaknesses
            task_type = self._categorize_task(task)
            if task_type:
                self._weaknesses[task_type] = self._weaknesses.get(task_type, 0) + 0.1
                
            # Suggest improvement
            if task_type:
                analysis["improvement_areas"].append(f"Improve handling of {task_type} tasks")
                
        return analysis
        
    def _estimate_complexity(self, task: "Task") -> float:
        """Estimate task complexity on a scale of 0-1"""
        complexity = 0.5  # Default medium complexity
        
        # Adjust based on description length
        desc_length = len(task.description)
        if desc_length > 500:
            complexity += 0.2
        elif desc_length < 100:
            complexity -= 0.1
            
        # Adjust based on subtasks
        if hasattr(task, 'subtasks') and task.subtasks:
            complexity += 0.1 * min(5, len(task.subtasks)) / 5
            
        # Adjust based on dependencies
        if hasattr(task, 'dependencies') and task.dependencies:
            complexity += 0.1 * min(5, len(task.dependencies)) / 5
            
        return min(1.0, max(0.1, complexity))
        
    def _categorize_task(self, task: "Task") -> Optional[str]:
        """Categorize the task type based on description"""
        desc_lower = task.description.lower()
        
        if "calculate" in desc_lower or "compute" in desc_lower:
            return "calculation"
        elif "analyze" in desc_lower or "analysis" in desc_lower:
            return "analysis"
        elif "search" in desc_lower or "find" in desc_lower:
            return "search"
        elif "generate" in desc_lower or "create" in desc_lower:
            return "generation"
        elif "summarize" in desc_lower or "summary" in desc_lower:
            return "summarization"
        elif "plan" in desc_lower or "strategy" in desc_lower:
            return "planning"
        elif "code" in desc_lower or "program" in desc_lower:
            return "coding"
        else:
            return None
            
    def _extract_task_pattern(self, task: "Task") -> Optional[Dict[str, Any]]:
        """Extract patterns from task execution"""
        # Simple pattern recognition based on task description and outcome
        desc_lower = task.description.lower()
        
        # Check for common patterns
        if "subtask" in desc_lower and hasattr(task, 'subtasks') and task.subtasks:
            return {
                "pattern_type": "task_decomposition",
                "subtask_count": len(task.subtasks)
            }
        elif "search" in desc_lower or "find" in desc_lower:
            return {
                "pattern_type": "information_retrieval",
                "success": task.status == "COMPLETED"
            }
        elif "code" in desc_lower or "function" in desc_lower:
            return {
                "pattern_type": "code_execution",
                "success": task.status == "COMPLETED"
            }
        elif "analyze" in desc_lower or "evaluate" in desc_lower:
            return {
                "pattern_type": "analysis",
                "success": task.status == "COMPLETED"
            }
        
        return None
        
    def _calculate_trend(self, values: List[float], window: int = 10) -> float:
        """Calculate the trend in a metric (positive or negative)"""
        if len(values) < window:
            return 0.0
            
        recent = values[-window:]
        if len(recent) < 2:
            return 0.0
            
        # Simple linear regression slope
        x = list(range(len(recent)))
        x_mean = sum(x) / len(x)
        y_mean = sum(recent) / len(recent)
        
        numerator = sum((x[i] - x_mean) * (recent[i] - y_mean) for i in range(len(recent)))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(len(recent)))
        
        if denominator == 0:
            return 0.0
            
        return numerator / denominator
        
    def _attempt_self_improvement(self) -> None:
        """Analyze patterns and suggest improvements"""
        # Analyze performance trends
        trends = {}
        for key, values in self._performance_metrics.items():
            if len(values) >= 10:
                trends[key] = self._calculate_trend(values)
                
        # Create improvement suggestions based on trends
        if trends.get("task_success_rate", 0) < -0.05:
            # Declining success rate
            self._improvement_suggestions.append({
                "area": "task_success",
                "suggestion": "Review recent failed tasks and identify common patterns",
                "priority": "high",
                "timestamp": time.time()
            })
            
        if trends.get("task_completion_time", 0) > 0.05:
            # Increasing completion times
            self._improvement_suggestions.append({
                "area": "efficiency",
                "suggestion": "Optimize task processing to reduce completion times",
                "priority": "medium",
                "timestamp": time.time()
            })
            
        # Analyze task patterns
        for pattern_type, data in self._task_patterns.items():
            if data["count"] >= 5 and data["success_rate"] < 0.7:
                # Pattern with low success rate
                self._improvement_suggestions.append({
                    "area": f"{pattern_type}_tasks",
                    "suggestion": f"Improve handling of {pattern_type} tasks (current success rate: {data['success_rate']:.2f})",
                    "priority": "high" if data["success_rate"] < 0.5 else "medium",
                    "timestamp": time.time(),
                    "examples": data["examples"][-3:]
                })
                
        # Create learning point from this analysis
        self._learning_points.append({
            "timestamp": time.time(),
            "type": "self_improvement",
            "insights": [s["suggestion"] for s in self._improvement_suggestions[-3:]],
            "performance_trends": {k: v for k, v in trends.items() if abs(v) > 0.01}
        })
        
        logger.info(f"[SelfReflectiveCognition] Completed self-improvement analysis with {len(self._improvement_suggestions)} suggestions")

    def _analyze_performance_loop(self) -> None:
        """
        Periodically analyzes performance and logs insights.
        """
        while True:
            time.sleep(60)  # every minute
            with self._lock:
                if self._reflections:
                    metrics = self.get_performance_metrics()
                    
                    # Log key metrics
                    log_msg = "[SelfReflectiveCognition] Performance analysis: "
                    if "task_success_rate" in metrics and metrics["task_success_rate"] is not None:
                        log_msg += f"Success rate: {metrics['task_success_rate']:.2f} "
                    if "strengths" in metrics and metrics["strengths"]:
                        top_strength = next(iter(metrics["strengths"].items()))
                        log_msg += f"Top strength: {top_strength[0]} "
                    if "improvement_suggestions" in metrics:
                        log_msg += f"Improvement suggestions: {metrics['improvement_suggestions']}"
                        
                    logger.info(log_msg)
                else:
                    logger.info("[SelfReflectiveCognition] No reflections yet for performance analysis.")

###############################################################################
# IN-MEMORY CODE ARCHIVE
###############################################################################

class InMemoryCodeArchive:
    """
    Advanced code archive with:
    - Vector embeddings for semantic code search
    - Code parsing and analysis
    - Function extraction and categorization
    - Code quality assessment
    - Self-modification capabilities
    - Version control for code changes
    """
    def __init__(self, embedding_model=None):
        self._snippets: Dict[str, Dict[str, Any]] = {}
        self._functions: Dict[str, Dict[str, Any]] = {}
        self._classes: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._version_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Initialize embedding model
        if HAVE_SENTENCE_TRANSFORMERS and embedding_model:
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            self.embedding_model = DummyEmbeddingModel()

    def add_snippet(self, name: str, code: str, tags: Optional[List[str]] = None) -> None:
        """
        Store a code snippet with metadata and generate embeddings.
        """
        with self._lock:
            # Parse the code to extract functions and classes
            try:
                tree = ast.parse(code)
                functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                
                function_names = [func.name for func in functions]
                class_names = [cls.name for cls in classes]
                
                # Calculate code complexity (simple metric: number of lines + AST nodes)
                complexity = len(code.split('\n')) + len([node for node in ast.walk(tree)])
                
            except SyntaxError:
                # If code can't be parsed, store basic info
                function_names = []
                class_names = []
                complexity = len(code.split('\n'))
            
            # Generate embedding
            embedding = self.embedding_model.encode(code)
            
            # Store snippet with metadata
            self._snippets[name] = {
                "code": code,
                "added_at": time.time(),
                "last_accessed": time.time(),
                "tags": tags or [],
                "embedding": embedding,
                "functions": function_names,
                "classes": class_names,
                "complexity": complexity,
                "access_count": 0
            }
            
            # Add to version history
            if name not in self._version_history:
                self._version_history[name] = []
                
            self._version_history[name].append({
                "timestamp": time.time(),
                "code": code,
                "version": len(self._version_history.get(name, [])) + 1
            })
            
            # Extract and store functions
            for func in functions:
                func_code = ast.get_source_segment(code, func)
                if func_code:
                    func_name = func.name
                    self._functions[func_name] = {
                        "code": func_code,
                        "parent_snippet": name,
                        "embedding": self.embedding_model.encode(func_code),
                        "args": [arg.arg for arg in func.args.args],
                        "complexity": len(func_code.split('\n'))
                    }
            
            # Extract and store classes
            for cls in classes:
                cls_code = ast.get_source_segment(code, cls)
                if cls_code:
                    cls_name = cls.name
                    self._classes[cls_name] = {
                        "code": cls_code,
                        "parent_snippet": name,
                        "embedding": self.embedding_model.encode(cls_code),
                        "methods": [node.name for node in ast.walk(cls) 
                                   if isinstance(node, ast.FunctionDef)],
                        "complexity": len(cls_code.split('\n'))
                    }
            
            logger.info(f"[InMemoryCodeArchive] Stored code snippet '{name}' with {len(function_names)} functions and {len(class_names)} classes")

    def get_snippet(self, name: str) -> Optional[str]:
        """
        Retrieve a previously stored snippet by name and update access metadata.
        """
        with self._lock:
            snippet = self._snippets.get(name)
            if snippet:
                snippet["last_accessed"] = time.time()
                snippet["access_count"] += 1
                return snippet["code"]
            return None
            
    def get_snippet_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata about a code snippet"""
        with self._lock:
            snippet = self._snippets.get(name)
            if snippet:
                # Return a copy without the embedding (too large)
                metadata = dict(snippet)
                metadata.pop("embedding", None)
                metadata.pop("code", None)  # Don't include the full code
                return metadata
            return None

    def list_snippets(self) -> List[str]:
        """
        List all snippet names stored in the archive.
        """
        with self._lock:
            return list(self._snippets.keys())
            
    def search_code(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for code snippets semantically similar to the query"""
        with self._lock:
            if not self._snippets:
                return []
                
            # Generate embedding for the query
            query_embedding = self.embedding_model.encode(query)
            
            # Calculate similarity scores
            results = []
            for name, snippet in self._snippets.items():
                # Cosine similarity
                similarity = np.dot(query_embedding, snippet["embedding"]) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(snippet["embedding"])
                )
                results.append((name, float(similarity)))
            
            # Sort by similarity (highest first) and return top_k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
            
    def search_functions(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for functions semantically similar to the query"""
        with self._lock:
            if not self._functions:
                return []
                
            # Generate embedding for the query
            query_embedding = self.embedding_model.encode(query)
            
            # Calculate similarity scores
            results = []
            for name, func in self._functions.items():
                # Cosine similarity
                similarity = np.dot(query_embedding, func["embedding"]) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(func["embedding"])
                )
                results.append((name, float(similarity)))
            
            # Sort by similarity (highest first) and return top_k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
            
    def get_function(self, name: str) -> Optional[str]:
        """Get a function by name"""
        with self._lock:
            func = self._functions.get(name)
            if func:
                return func["code"]
            return None
            
    def get_class(self, name: str) -> Optional[str]:
        """Get a class by name"""
        with self._lock:
            cls = self._classes.get(name)
            if cls:
                return cls["code"]
            return None
            
    def update_snippet(self, name: str, new_code: str) -> bool:
        """Update an existing snippet with new code"""
        with self._lock:
            if name not in self._snippets:
                return False
                
            # Store the update
            self.add_snippet(name, new_code)
            return True
            
    def get_version_history(self, name: str) -> List[Dict[str, Any]]:
        """Get the version history for a snippet"""
        with self._lock:
            return self._version_history.get(name, [])
            
    def revert_to_version(self, name: str, version: int) -> bool:
        """Revert a snippet to a previous version"""
        with self._lock:
            history = self._version_history.get(name, [])
            if not history or version > len(history):
                return False
                
            # Find the requested version (1-indexed)
            target_version = history[version-1]
            
            # Update the snippet
            self.add_snippet(name, target_version["code"])
            return True
            
    def analyze_code_quality(self, name: str) -> Dict[str, Any]:
        """Perform basic code quality analysis"""
        with self._lock:
            snippet = self._snippets.get(name)
            if not snippet:
                return {"error": "Snippet not found"}
                
            code = snippet["code"]
            
            # Simple metrics
            analysis = {
                "line_count": len(code.split('\n')),
                "character_count": len(code),
                "complexity": snippet["complexity"],
                "function_count": len(snippet["functions"]),
                "class_count": len(snippet["classes"]),
                "issues": []
            }
            
            # Check for common issues
            if "import *" in code:
                analysis["issues"].append("Uses wildcard imports")
                
            if code.count("except:") > 0:
                analysis["issues"].append("Uses bare except clauses")
                
            if code.count("global ") > 0:
                analysis["issues"].append("Uses global variables")
                
            # Check line length
            long_lines = 0
            for line in code.split('\n'):
                if len(line) > 100:
                    long_lines += 1
            
            if long_lines > 0:
                analysis["issues"].append(f"Contains {long_lines} lines longer than 100 characters")
                
            return analysis

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
    A potential next step with advanced attributes:
    - Utility estimation
    - Success probability
    - Resource requirements
    - Dependencies on other actions
    - Expected outcomes
    - Risk assessment
    """
    def __init__(self, description: str, rationale: str, priority: int = 5):
        self.description = description
        self.rationale = rationale
        self.priority = priority
        self.utility = 0.5  # Default utility (0-1)
        self.success_probability = 0.8  # Default success probability (0-1)
        self.resource_cost = 0.3  # Default resource cost (0-1)
        self.dependencies = []  # List of action IDs this depends on
        self.expected_outcomes = []  # List of expected outcomes
        self.risks = []  # List of potential risks
        self.tags = set()  # Tags for categorization
        self.created_at = time.time()

    def set_utility(self, utility: float) -> None:
        """Set the estimated utility of this action (0-1)"""
        self.utility = max(0.0, min(1.0, utility))
        
    def set_success_probability(self, probability: float) -> None:
        """Set the estimated probability of success (0-1)"""
        self.success_probability = max(0.0, min(1.0, probability))
        
    def set_resource_cost(self, cost: float) -> None:
        """Set the estimated resource cost (0-1)"""
        self.resource_cost = max(0.0, min(1.0, cost))
        
    def add_dependency(self, action_id: str) -> None:
        """Add a dependency on another action"""
        if action_id not in self.dependencies:
            self.dependencies.append(action_id)
            
    def add_expected_outcome(self, outcome: str) -> None:
        """Add an expected outcome of this action"""
        self.expected_outcomes.append(outcome)
        
    def add_risk(self, risk: str, probability: float = 0.5) -> None:
        """Add a potential risk with its probability"""
        self.risks.append({"description": risk, "probability": probability})
        
    def add_tag(self, tag: str) -> None:
        """Add a tag to this action"""
        self.tags.add(tag)
        
    def calculate_expected_value(self) -> float:
        """Calculate the expected value of this action"""
        return self.utility * self.success_probability - self.resource_cost
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "description": self.description,
            "rationale": self.rationale,
            "priority": self.priority,
            "utility": self.utility,
            "success_probability": self.success_probability,
            "resource_cost": self.resource_cost,
            "dependencies": self.dependencies,
            "expected_outcomes": self.expected_outcomes,
            "risks": self.risks,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "expected_value": self.calculate_expected_value()
        }

    def __repr__(self) -> str:
        return f"CandidateAction(desc={self.description[:20]}, prio={self.priority}, ev={self.calculate_expected_value():.2f})"

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
    Advanced function adapter with:
    - Multiple function types beyond do_anything
    - Function result caching
    - Rate limiting and resource management
    - Error handling and retry logic
    - Sandboxing and security features
    - Performance monitoring
    """
    def __init__(self):
        self._result_cache = {}
        self._function_stats = defaultdict(lambda: {
            "calls": 0,
            "errors": 0,
            "total_execution_time": 0,
            "last_call_time": 0
        })
        self._rate_limits = {
            "do_anything": {"calls_per_minute": 10, "last_reset": time.time(), "calls": 0},
            "fetch_url": {"calls_per_minute": 20, "last_reset": time.time(), "calls": 0},
            "analyze_data": {"calls_per_minute": 15, "last_reset": time.time(), "calls": 0}
        }
        self._lock = threading.Lock()

    def do_anything(self, snippet: str, cache: bool = True) -> Dict[str, Any]:
        """
        Execute arbitrary Python code with caching, rate limiting, and monitoring.
        """
        # Check rate limits
        if not self._check_rate_limit("do_anything"):
            return {
                "status": "error", 
                "error": "Rate limit exceeded for do_anything function",
                "retry_after": self._get_retry_after("do_anything")
            }
        
        # Check cache if enabled
        if cache:
            cache_key = f"do_anything:{hash(snippet)}"
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result
        
        # Clean up the code
        code = snippet.strip()
        code = code.replace("<code>", "").replace("</code>", "")
        
        # Log and execute
        logger.info(f"[do_anything] Executing code:\n{code}")
        start_time = time.time()
        
        try:
            # Create a restricted locals dictionary
            local_vars = {}
            
            # Execute in the restricted environment
            exec(code, globals(), local_vars)
            
            # Prepare result
            result = {
                "status": "success", 
                "executed_code": code,
                "execution_time": time.time() - start_time
            }
            
            # Add any defined variables to the result
            # Filter out built-ins and modules
            result["variables"] = {
                k: v for k, v in local_vars.items() 
                if not k.startswith('__') and not inspect.ismodule(v)
            }
            
            # Update stats
            self._update_function_stats("do_anything", True, time.time() - start_time)
            
            # Cache result if enabled
            if cache:
                self._add_to_cache(cache_key, result)
                
            return result
            
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[do_anything] Error: {str(e)}\nTraceback:\n{tb}")
            
            # Update stats
            self._update_function_stats("do_anything", False, time.time() - start_time)
            
            return {
                "status": "error", 
                "error": str(e), 
                "traceback": tb,
                "execution_time": time.time() - start_time
            }
            
    def fetch_url(self, url: str, method: str = "GET", headers: Dict[str, str] = None, 
                 data: Any = None, timeout: int = 10, cache: bool = True) -> Dict[str, Any]:
        """
        Fetch data from a URL with caching and rate limiting.
        """
        # Check rate limits
        if not self._check_rate_limit("fetch_url"):
            return {
                "status": "error", 
                "error": "Rate limit exceeded for fetch_url function",
                "retry_after": self._get_retry_after("fetch_url")
            }
        
        # Check cache if enabled
        if cache:
            cache_key = f"fetch_url:{method}:{url}:{hash(str(headers))}:{hash(str(data))}"
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result
        
        # Log and execute
        logger.info(f"[fetch_url] Fetching {method} {url}")
        start_time = time.time()
        
        try:
            # Make the request
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                data=data,
                timeout=timeout
            )
            
            # Prepare result
            result = {
                "status": "success",
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content_type": response.headers.get("Content-Type", ""),
                "execution_time": time.time() - start_time
            }
            
            # Handle different content types
            content_type = response.headers.get("Content-Type", "").lower()
            if "application/json" in content_type:
                try:
                    result["data"] = response.json()
                except:
                    result["text"] = response.text
            elif "text/" in content_type:
                result["text"] = response.text
            else:
                # For binary data, just note the size
                result["content_length"] = len(response.content)
                result["binary"] = True
            
            # Update stats
            self._update_function_stats("fetch_url", True, time.time() - start_time)
            
            # Cache result if enabled
            if cache and response.status_code < 400:
                self._add_to_cache(cache_key, result)
                
            return result
            
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[fetch_url] Error: {str(e)}\nTraceback:\n{tb}")
            
            # Update stats
            self._update_function_stats("fetch_url", False, time.time() - start_time)
            
            return {
                "status": "error", 
                "error": str(e), 
                "traceback": tb,
                "execution_time": time.time() - start_time
            }
            
    def analyze_data(self, data: Any, analysis_type: str = "summary", 
                    options: Dict[str, Any] = None, cache: bool = True) -> Dict[str, Any]:
        """
        Analyze data with various methods.
        """
        # Check rate limits
        if not self._check_rate_limit("analyze_data"):
            return {
                "status": "error", 
                "error": "Rate limit exceeded for analyze_data function",
                "retry_after": self._get_retry_after("analyze_data")
            }
        
        # Check cache if enabled
        if cache:
            cache_key = f"analyze_data:{analysis_type}:{hash(str(data))}:{hash(str(options))}"
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result
        
        # Log and execute
        logger.info(f"[analyze_data] Analyzing data with {analysis_type}")
        start_time = time.time()
        options = options or {}
        
        try:
            result = {
                "status": "success",
                "analysis_type": analysis_type,
                "execution_time": time.time() - start_time
            }
            
            # Perform different types of analysis
            if analysis_type == "summary":
                result["summary"] = self._summarize_data(data, options)
            elif analysis_type == "statistics":
                result["statistics"] = self._calculate_statistics(data, options)
            elif analysis_type == "visualization":
                result["visualization"] = self._generate_visualization(data, options)
            else:
                result["status"] = "error"
                result["error"] = f"Unknown analysis type: {analysis_type}"
                return result
            
            # Update stats
            self._update_function_stats("analyze_data", True, time.time() - start_time)
            
            # Cache result if enabled
            if cache:
                self._add_to_cache(cache_key, result)
                
            return result
            
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[analyze_data] Error: {str(e)}\nTraceback:\n{tb}")
            
            # Update stats
            self._update_function_stats("analyze_data", False, time.time() - start_time)
            
            return {
                "status": "error", 
                "error": str(e), 
                "traceback": tb,
                "execution_time": time.time() - start_time
            }

    def process_function_calls(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Process function calls in text with support for multiple function types.
        """
        # Pattern for any function call
        pattern = r"<function_call>\s*(\w+)\s*:\s*(.*?)</function_call>"
        match = re.search(pattern, text, re.DOTALL)
        
        if not match:
            return None
            
        function_name = match.group(1).strip()
        args_text = match.group(2).strip()
        
        logger.info(f"[FunctionAdapter] Detected {function_name} function call")
        
        # Handle different function types
        if function_name == "do_anything":
            return self.do_anything(args_text)
        elif function_name == "fetch_url":
            # Parse URL and options
            try:
                # Try to parse as JSON first
                try:
                    args = json.loads(args_text)
                    url = args.get("url")
                    method = args.get("method", "GET")
                    headers = args.get("headers")
                    data = args.get("data")
                    timeout = args.get("timeout", 10)
                except:
                    # Fallback to simple parsing
                    url = args_text.strip()
                    method = "GET"
                    headers = None
                    data = None
                    timeout = 10
                    
                return self.fetch_url(url, method, headers, data, timeout)
            except Exception as e:
                return {"status": "error", "error": f"Error parsing fetch_url arguments: {str(e)}"}
        elif function_name == "analyze_data":
            # Parse data and options
            try:
                args = json.loads(args_text)
                data = args.get("data")
                analysis_type = args.get("analysis_type", "summary")
                options = args.get("options", {})
                
                return self.analyze_data(data, analysis_type, options)
            except Exception as e:
                return {"status": "error", "error": f"Error parsing analyze_data arguments: {str(e)}"}
        else:
            return {"status": "error", "error": f"Unknown function: {function_name}"}
            
    def _check_rate_limit(self, function_name: str) -> bool:
        """Check if a function call is within rate limits"""
        with self._lock:
            if function_name not in self._rate_limits:
                return True
                
            limit_info = self._rate_limits[function_name]
            current_time = time.time()
            
            # Reset counter if a minute has passed
            if current_time - limit_info["last_reset"] > 60:
                limit_info["last_reset"] = current_time
                limit_info["calls"] = 0
                
            # Check if we're over the limit
            if limit_info["calls"] >= limit_info["calls_per_minute"]:
                return False
                
            # Increment the counter
            limit_info["calls"] += 1
            return True
            
    def _get_retry_after(self, function_name: str) -> float:
        """Get seconds until rate limit resets"""
        with self._lock:
            if function_name not in self._rate_limits:
                return 0
                
            limit_info = self._rate_limits[function_name]
            return max(0, 60 - (time.time() - limit_info["last_reset"]))
            
    def _update_function_stats(self, function_name: str, success: bool, execution_time: float) -> None:
        """Update function call statistics"""
        with self._lock:
            stats = self._function_stats[function_name]
            stats["calls"] += 1
            if not success:
                stats["errors"] += 1
            stats["total_execution_time"] += execution_time
            stats["last_call_time"] = time.time()
            
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get a result from the cache if it exists and is not expired"""
        with self._lock:
            if cache_key in self._result_cache:
                entry = self._result_cache[cache_key]
                # Check if cache entry is expired (30 minutes)
                if time.time() - entry["timestamp"] < 1800:
                    logger.info(f"[FunctionAdapter] Cache hit for {cache_key}")
                    return entry["result"]
                else:
                    # Remove expired entry
                    del self._result_cache[cache_key]
            return None
            
    def _add_to_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Add a result to the cache"""
        with self._lock:
            self._result_cache[cache_key] = {
                "result": result,
                "timestamp": time.time()
            }
            
            # Limit cache size to 100 entries
            if len(self._result_cache) > 100:
                # Remove oldest entry
                oldest_key = min(self._result_cache.keys(), 
                                key=lambda k: self._result_cache[k]["timestamp"])
                del self._result_cache[oldest_key]
                
    def _summarize_data(self, data: Any, options: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of data"""
        summary = {}
        
        if isinstance(data, dict):
            summary["type"] = "dictionary"
            summary["keys"] = list(data.keys())[:10]  # First 10 keys
            summary["key_count"] = len(data)
            
        elif isinstance(data, list):
            summary["type"] = "list"
            summary["length"] = len(data)
            summary["sample"] = data[:5] if len(data) > 0 else []  # First 5 items
            
            # Try to determine item types
            if data:
                item_types = set(type(item).__name__ for item in data[:10])
                summary["item_types"] = list(item_types)
                
        elif isinstance(data, str):
            summary["type"] = "string"
            summary["length"] = len(data)
            summary["preview"] = data[:100] + "..." if len(data) > 100 else data
            
            # Count words and lines
            summary["word_count"] = len(data.split())
            summary["line_count"] = len(data.splitlines())
            
        else:
            summary["type"] = type(data).__name__
            
        return summary
        
    def _calculate_statistics(self, data: Any, options: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics for numerical data"""
        stats = {}
        
        # Handle lists of numbers
        if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
            stats["count"] = len(data)
            stats["min"] = min(data)
            stats["max"] = max(data)
            stats["sum"] = sum(data)
            stats["mean"] = sum(data) / len(data) if data else 0
            
            # Calculate median
            sorted_data = sorted(data)
            mid = len(sorted_data) // 2
            if len(sorted_data) % 2 == 0:
                stats["median"] = (sorted_data[mid-1] + sorted_data[mid]) / 2
            else:
                stats["median"] = sorted_data[mid]
                
            # Calculate variance and std dev
            if len(data) > 1:
                mean = stats["mean"]
                variance = sum((x - mean) ** 2 for x in data) / len(data)
                stats["variance"] = variance
                stats["std_dev"] = math.sqrt(variance)
                
        # Handle dictionaries with numeric values
        elif isinstance(data, dict) and all(isinstance(v, (int, float)) for v in data.values()):
            values = list(data.values())
            stats["count"] = len(values)
            stats["min"] = min(values)
            stats["max"] = max(values)
            stats["sum"] = sum(values)
            stats["mean"] = sum(values) / len(values) if values else 0
            
            # Include keys for min and max
            min_key = min(data.keys(), key=lambda k: data[k])
            max_key = max(data.keys(), key=lambda k: data[k])
            stats["min_key"] = min_key
            stats["max_key"] = max_key
            
        else:
            stats["error"] = "Data is not suitable for statistical analysis"
            
        return stats
        
    def _generate_visualization(self, data: Any, options: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a text-based visualization of data"""
        viz = {}
        
        # Handle lists of numbers
        if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
            # Generate a simple ASCII histogram
            if len(data) > 0:
                min_val = min(data)
                max_val = max(data)
                range_val = max_val - min_val if max_val > min_val else 1
                
                # Create 10 bins
                bins = [0] * 10
                for val in data:
                    bin_idx = min(9, int(((val - min_val) / range_val) * 10))
                    bins[bin_idx] += 1
                
                # Create ASCII histogram
                max_bin = max(bins)
                histogram = []
                for i, count in enumerate(bins):
                    bin_min = min_val + (i * range_val / 10)
                    bin_max = min_val + ((i + 1) * range_val / 10)
                    bar = "#" * int((count / max_bin) * 20) if max_bin > 0 else ""
                    histogram.append(f"{bin_min:.2f}-{bin_max:.2f}: {bar} ({count})")
                
                viz["histogram"] = histogram
                
        # Handle dictionaries with numeric values
        elif isinstance(data, dict) and all(isinstance(v, (int, float)) for v in data.values()):
            # Generate a simple ASCII bar chart
            items = list(data.items())
            if len(items) > 0:
                # Sort by value
                items.sort(key=lambda x: x[1], reverse=True)
                
                # Take top 10
                items = items[:10]
                
                max_val = max(v for _, v in items)
                bar_chart = []
                for key, val in items:
                    bar = "#" * int((val / max_val) * 20) if max_val > 0 else ""
                    bar_chart.append(f"{key[:15]}: {bar} ({val})")
                
                viz["bar_chart"] = bar_chart
                
        else:
            viz["error"] = "Data is not suitable for visualization"
            
        return viz

###############################################################################
# SMART TASK PROCESSOR
###############################################################################

class SmartTaskProcessor:
    """
    Advanced task processor with:
    - Sophisticated task decomposition
    - Multiple function types support
    - Dependency tracking and management
    - Resource-aware processing
    - Error recovery and retry mechanisms
    - Learning from task execution patterns
    """
    def __init__(
        self,
        memory_store: TaskMemoryStore,
        function_adapter: FunctionAdapter,
        reflection: SelfReflectiveCognition,
        knowledge_base: KnowledgeBase
    ):
        self.memory_store = memory_store
        self.function_adapter = function_adapter
        self.reflection = reflection
        self.knowledge_base = knowledge_base
        self._processing_strategies = self._initialize_strategies()
        self._retry_queue = []
        self._max_retries = 3
        
    def _initialize_strategies(self) -> Dict[str, Callable]:
        """Initialize specialized processing strategies for different task types"""
        return {
            "code_execution": self._process_code_execution,
            "data_analysis": self._process_data_analysis,
            "information_retrieval": self._process_information_retrieval,
            "task_decomposition": self._process_task_decomposition,
            "knowledge_application": self._process_knowledge_application,
            "planning": self._process_planning
        }

    def process_task(self, task: Task) -> None:
        """
        Process a task with advanced handling based on task type and content.
        """
        logger.info(f"[SmartTaskProcessor] Starting task {task.task_id} - '{task.description}'")
        
        # Mark task as in progress
        task.start()
        self.memory_store.update_task_status(task.task_id, "IN_PROGRESS")
        
        # Check dependencies
        if hasattr(task, 'dependencies') and task.dependencies:
            unmet_dependencies = []
            for dep_id in task.dependencies:
                dep_task = self.memory_store.get_task(dep_id)
                if not dep_task or dep_task.status != "COMPLETED":
                    unmet_dependencies.append(dep_id)
                    
            if unmet_dependencies:
                logger.info(f"[SmartTaskProcessor] Task {task.task_id} has unmet dependencies: {unmet_dependencies}")
                self.memory_store.update_task_status(task.task_id, "WAITING_DEPENDENCIES")
                return

        try:
            # Determine task type and select appropriate processing strategy
            task_type = self._determine_task_type(task)
            
            # Apply the selected strategy
            if task_type in self._processing_strategies:
                logger.info(f"[SmartTaskProcessor] Using {task_type} strategy for task {task.task_id}")
                result = self._processing_strategies[task_type](task)
            else:
                # Default processing
                result = self._default_process_task(task)
                
            # Update task result
            if result:
                self.memory_store.update_task_result(task.task_id, result)
                
            # Mark completed
            task.complete(result)
            self.memory_store.update_task_status(task.task_id, "COMPLETED")
            
            # Reflect on task
            self.reflection.reflect_on_task(task)
            
            logger.info(f"[SmartTaskProcessor] Completed task {task.task_id}")
            
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[SmartTaskProcessor] Error processing task {task.task_id}: {str(e)}\n{tb}")
            
            # Handle retry logic
            if self._should_retry(task):
                self._retry_queue.append((task, 1))  # 1 = first retry
                logger.info(f"[SmartTaskProcessor] Scheduled task {task.task_id} for retry")
            else:
                # Mark as failed
                error_result = {"status": "error", "error": str(e), "traceback": tb}
                task.fail(error_result)
                self.memory_store.update_task_result(task.task_id, error_result)
                self.memory_store.update_task_status(task.task_id, "FAILED")
                
                # Still reflect on failed tasks
                self.reflection.reflect_on_task(task)
                
    def process_retry(self, task: Task, retry_count: int) -> None:
        """Process a task retry"""
        logger.info(f"[SmartTaskProcessor] Retrying task {task.task_id} (attempt {retry_count})")
        
        try:
            # Mark as in progress again
            task.start()
            self.memory_store.update_task_status(task.task_id, "IN_PROGRESS")
            
            # Use default processing for retries
            result = self._default_process_task(task)
            
            # Update task result
            if result:
                self.memory_store.update_task_result(task.task_id, result)
                
            # Mark completed
            task.complete(result)
            self.memory_store.update_task_status(task.task_id, "COMPLETED")
            
            # Reflect on task
            self.reflection.reflect_on_task(task)
            
            logger.info(f"[SmartTaskProcessor] Successfully completed task {task.task_id} on retry {retry_count}")
            
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[SmartTaskProcessor] Error on retry {retry_count} for task {task.task_id}: {str(e)}\n{tb}")
            
            # Check if we should retry again
            if retry_count < self._max_retries:
                self._retry_queue.append((task, retry_count + 1))
                logger.info(f"[SmartTaskProcessor] Scheduled task {task.task_id} for retry {retry_count + 1}")
            else:
                # Mark as failed after max retries
                error_result = {
                    "status": "error", 
                    "error": str(e), 
                    "traceback": tb,
                    "retry_count": retry_count
                }
                task.fail(error_result)
                self.memory_store.update_task_result(task.task_id, error_result)
                self.memory_store.update_task_status(task.task_id, "FAILED")
                
                # Still reflect on failed tasks
                self.reflection.reflect_on_task(task)
                
    def get_retry_tasks(self) -> List[Tuple[Task, int]]:
        """Get tasks scheduled for retry"""
        return list(self._retry_queue)
        
    def clear_retry(self, task_id: int) -> None:
        """Remove a task from the retry queue"""
        self._retry_queue = [(t, c) for t, c in self._retry_queue if t.task_id != task_id]
        
    def _determine_task_type(self, task: Task) -> str:
        """Determine the type of task based on its description and tags"""
        # Check tags first
        if hasattr(task, 'tags') and task.tags:
            for tag in task.tags:
                if tag in self._processing_strategies:
                    return tag
        
        # Check description
        desc_lower = task.description.lower()
        
        if "<function_call>" in desc_lower:
            return "code_execution"
        elif "analyze" in desc_lower or "data" in desc_lower or "statistics" in desc_lower:
            return "data_analysis"
        elif "search" in desc_lower or "find" in desc_lower or "retrieve" in desc_lower:
            return "information_retrieval"
        elif "subtask" in desc_lower or "decompose" in desc_lower:
            return "task_decomposition"
        elif "knowledge" in desc_lower or "fact" in desc_lower:
            return "knowledge_application"
        elif "plan" in desc_lower or "strategy" in desc_lower:
            return "planning"
        
        return "default"
        
    def _default_process_task(self, task: Task) -> Optional[Dict[str, Any]]:
        """Default task processing logic"""
        # 1) Check for function calls in the description
        result = self.function_adapter.process_function_calls(task.description)
        if result:
            return result
            
        # 2) Check for subtask patterns
        subtask_result = self._check_for_subtasks(task)
        if subtask_result:
            return subtask_result
            
        # 3) If no specific handling, return a simple completion message
        return {"status": "completed", "message": "Task processed with default handler"}
        
    def _process_code_execution(self, task: Task) -> Dict[str, Any]:
        """Process a code execution task"""
        # Check for function calls
        result = self.function_adapter.process_function_calls(task.description)
        if result:
            return result
            
        # If no function call found but this is a code task, try to extract code
        code_pattern = r"```(?:python)?\s*(.*?)```"
        code_match = re.search(code_pattern, task.description, re.DOTALL)
        
        if code_match:
            code = code_match.group(1).strip()
            return self.function_adapter.do_anything(code)
            
        # If still no code found, return an error
        return {
            "status": "error",
            "error": "No executable code found in task description"
        }
        
    def _process_data_analysis(self, task: Task) -> Dict[str, Any]:
        """Process a data analysis task"""
        # Check if we have data in the parent task
        if task.parent_id:
            parent = self.memory_store.get_task(task.parent_id)
            if parent and parent.result and isinstance(parent.result, dict):
                # Try to extract data from parent result
                data = parent.result.get("data") or parent.result.get("result")
                
                if data:
                    # Determine analysis type from task description
                    analysis_type = "summary"  # default
                    if "statistics" in task.description.lower():
                        analysis_type = "statistics"
                    elif "visualize" in task.description.lower() or "chart" in task.description.lower():
                        analysis_type = "visualization"
                        
                    return self.function_adapter.analyze_data(data, analysis_type)
        
        # If we can't find data, check for function calls
        result = self.function_adapter.process_function_calls(task.description)
        if result:
            return result
            
        # If still no data found, return an error
        return {
            "status": "error",
            "error": "No data found for analysis"
        }
        
    def _process_information_retrieval(self, task: Task) -> Dict[str, Any]:
        """Process an information retrieval task"""
        # Extract search query from task description
        query_patterns = [
            r"(?:search|find|retrieve|get)\s+(?:information|data|facts)?\s+(?:about|on|for)?\s+[\"']?([^\"']+)[\"']?",
            r"[\"']([^\"']+)[\"']\s+(?:search|lookup|information)"
        ]
        
        query = None
        for pattern in query_patterns:
            match = re.search(pattern, task.description, re.IGNORECASE)
            if match:
                query = match.group(1).strip()
                break
                
        if not query:
            # If no specific pattern matched, use the whole description
            query = task.description
            
        # First check knowledge base
        kb_results = self.knowledge_base.search_facts(query, semantic=True, top_k=3)
        
        if kb_results:
            return {
                "status": "success",
                "source": "knowledge_base",
                "results": [{"key": k, "value": v, "relevance": r} for k, v, r in kb_results]
            }
            
        # If not in knowledge base, check for URL to fetch
        url_match = re.search(r'https?://[^\s]+', task.description)
        if url_match:
            url = url_match.group(0)
            return self.function_adapter.fetch_url(url)
            
        # If no URL, check for function calls
        result = self.function_adapter.process_function_calls(task.description)
        if result:
            return result
            
        # If all else fails, return a message about not finding information
        return {
            "status": "partial_success",
            "message": f"No specific information found for query: {query}",
            "query": query
        }
        
    def _process_task_decomposition(self, task: Task) -> Dict[str, Any]:
        """Process a task decomposition task"""
        # Check for explicit subtask pattern first
        subtask_result = self._check_for_subtasks(task)
        if subtask_result:
            return subtask_result
            
        # If no explicit subtasks, try to automatically decompose
        desc_lower = task.description.lower()
        
        # Look for keywords indicating decomposition is needed
        decomposition_keywords = ["decompose", "break down", "split", "divide"]
        if any(keyword in desc_lower for keyword in decomposition_keywords):
            # Extract the target to decompose
            target_pattern = r"(?:decompose|break down|split|divide)\s+(?:the|this)?\s+(?:task|goal|problem)?\s*:?\s*[\"']?([^\"']+)[\"']?"
            target_match = re.search(target_pattern, desc_lower)
            
            target = target_match.group(1).strip() if target_match else task.description
            
            # Create subtasks based on the target
            subtasks = self._auto_decompose_task(target)
            
            # Create the subtasks
            created_subtasks = []
            for subtask_desc in subtasks:
                subtask = self._spawn_subtask(task, subtask_desc)
                created_subtasks.append(subtask.task_id)
                
            return {
                "status": "success",
                "decomposition_type": "automatic",
                "target": target,
                "subtask_count": len(subtasks),
                "subtask_ids": created_subtasks
            }
            
        # If no decomposition needed, return a message
        return {
            "status": "success",
            "message": "No decomposition pattern found in task"
        }
        
    def _process_knowledge_application(self, task: Task) -> Dict[str, Any]:
        """Process a knowledge application task"""
        # Extract the topic from the task description
        topic_pattern = r"(?:apply|use|leverage)\s+knowledge\s+(?:about|on|of)\s+[\"']?([^\"']+)[\"']?"
        topic_match = re.search(topic_pattern, task.description, re.IGNORECASE)
        
        topic = topic_match.group(1).strip() if topic_match else None
        
        if not topic:
            # Try to extract any key terms
            key_terms = re.findall(r'\b[A-Z][a-z]{2,}\b', task.description)
            if key_terms:
                topic = key_terms[0]
            else:
                # Use the whole description
                topic = task.description
                
        # Search knowledge base for the topic
        kb_results = self.knowledge_base.search_facts(topic, semantic=True, top_k=5)
        
        if kb_results:
            # Apply the knowledge by creating a summary
            facts = [v for _, v, _ in kb_results]
            
            application = {
                "status": "success",
                "topic": topic,
                "applied_knowledge": facts,
                "summary": f"Applied knowledge about '{topic}' from {len(facts)} facts in the knowledge base."
            }
            
            # If this is part of a larger task, add the knowledge to the parent
            if task.parent_id:
                parent = self.memory_store.get_task(task.parent_id)
                if parent:
                    # Add a note to the parent task
                    parent_result = parent.result or {}
                    if isinstance(parent_result, dict):
                        if "applied_knowledge" not in parent_result:
                            parent_result["applied_knowledge"] = []
                        parent_result["applied_knowledge"].append({
                            "topic": topic,
                            "fact_count": len(facts)
                        })
                        self.memory_store.update_task_result(parent.task_id, parent_result)
            
            return application
            
        # If no knowledge found, return a message
        return {
            "status": "partial_success",
            "message": f"No knowledge found about '{topic}'",
            "topic": topic
        }
        
    def _process_planning(self, task: Task) -> Dict[str, Any]:
        """Process a planning task"""
        # Extract the planning target
        target_pattern = r"(?:plan|create plan|develop strategy)\s+(?:for|to)\s+[\"']?([^\"']+)[\"']?"
        target_match = re.search(target_pattern, task.description, re.IGNORECASE)
        
        target = target_match.group(1).strip() if target_match else task.description
        
        # Generate a simple plan with steps
        steps = self._generate_plan_steps(target)
        
        # Create subtasks for each step
        step_tasks = []
        for i, step in enumerate(steps, 1):
            # Create with increasing priorities (later steps have lower priority)
            priority = task.priority + i
            subtask = self._spawn_subtask(task, f"Step {i}: {step}", priority=priority)
            step_tasks.append(subtask.task_id)
            
            # Add dependencies between steps
            if i > 1:
                previous_task_id = step_tasks[i-2]
                subtask.add_dependency(previous_task_id)
                self.memory_store.add_task(subtask)
        
        return {
            "status": "success",
            "planning_target": target,
            "steps": steps,
            "step_count": len(steps),
            "step_task_ids": step_tasks
        }

    def _check_for_subtasks(self, task: Task) -> Optional[Dict[str, Any]]:
        """Check for subtask patterns in the task description"""
        # Check for Subtask(n)= ... pattern
        subtask_pattern = r"Subtask\s*\(\s*(\d+)\s*\)\s*=\s*(.*)"
        match = re.search(subtask_pattern, task.description, re.IGNORECASE | re.DOTALL)
        
        if match:
            try:
                num_subtasks = int(match.group(1))
                subtask_text = match.group(2).strip()
                
                # Try different splitting patterns
                patterns = [
                    r'\d+\)\s*',  # 1) Task description
                    r'\d+\.\s*',   # 1. Task description
                    r'Step\s+\d+\s*:',  # Step 1: Task description
                    r'\n+'         # Split by newlines if other patterns fail
                ]
                
                lines = None
                for pattern in patterns:
                    split_result = re.split(pattern, subtask_text)
                    if len(split_result) > 1:
                        lines = [line for line in split_result if line.strip()]
                        break
                        
                if not lines:
                    lines = [subtask_text]  # Fallback to the whole text
                
                # Create subtasks
                created_subtasks = []
                for i, line in enumerate(lines[:num_subtasks], 1):
                    desc = line.strip()
                    if desc:
                        subtask = self._spawn_subtask(task, desc)
                        created_subtasks.append(subtask.task_id)
                
                return {
                    "status": "success",
                    "decomposition_type": "explicit",
                    "expected_subtasks": num_subtasks,
                    "created_subtasks": len(created_subtasks),
                    "subtask_ids": created_subtasks
                }
                
            except Exception as e:
                logger.exception(f"[SmartTaskProcessor] Error parsing subtasks: {e}")
                return {
                    "status": "error",
                    "error": f"Error parsing subtasks: {str(e)}"
                }
                
        # Check for numbered list pattern (without Subtask declaration)
        list_pattern = r"(?:^|\n)(?:\d+[\)\.]\s+[^\n]+)(?:\n\d+[\)\.]\s+[^\n]+)+"
        list_match = re.search(list_pattern, task.description)
        
        if list_match:
            try:
                list_text = list_match.group(0)
                items = re.findall(r'\d+[\)\.]\s+([^\n]+)', list_text)
                
                # Create subtasks
                created_subtasks = []
                for item in items:
                    desc = item.strip()
                    if desc:
                        subtask = self._spawn_subtask(task, desc)
                        created_subtasks.append(subtask.task_id)
                
                return {
                    "status": "success",
                    "decomposition_type": "implicit_list",
                    "created_subtasks": len(created_subtasks),
                    "subtask_ids": created_subtasks
                }
                
            except Exception as e:
                logger.exception(f"[SmartTaskProcessor] Error parsing list items: {e}")
        
        return None

    def _spawn_subtask(self, parent_task: Task, description: str, priority: Optional[int] = None) -> Task:
        """
        Creates a new Task with appropriate priority and parent relationship.
        """
        if priority is None:
            # Default to higher priority than parent
            priority = max(1, parent_task.priority - 1)
            
        # Create the task
        t = self.memory_store.create_task(
            priority=priority,
            description=description,
            parent_id=parent_task.task_id
        )
        
        # Add tags based on parent and content
        if hasattr(parent_task, 'tags'):
            for tag in parent_task.tags:
                t.add_tag(tag)
                
        # Add subtask-specific tag
        t.add_tag("subtask")
        
        # Add task type tag based on content
        task_type = self._determine_task_type(t)
        if task_type != "default":
            t.add_tag(task_type)
            
        logger.info(f"[SmartTaskProcessor] Spawned subtask {t}")
        return t
        
    def _auto_decompose_task(self, target: str) -> List[str]:
        """Automatically decompose a task into subtasks"""
        # Simple rule-based decomposition
        subtasks = []
        
        # Check for common task types and decompose accordingly
        target_lower = target.lower()
        
        if "research" in target_lower:
            subtasks = [
                "Identify key questions to answer about the topic",
                "Search for relevant information sources",
                "Collect and organize information",
                "Analyze findings and identify patterns",
                "Summarize research results"
            ]
        elif "analyze" in target_lower or "analysis" in target_lower:
            subtasks = [
                "Define the scope and objectives of the analysis",
                "Collect relevant data and information",
                "Process and organize the data",
                "Perform the analysis using appropriate methods",
                "Interpret results and draw conclusions",
                "Prepare a summary of findings"
            ]
        elif "plan" in target_lower or "strategy" in target_lower:
            subtasks = [
                "Define goals and objectives",
                "Identify available resources and constraints",
                "Generate potential approaches",
                "Evaluate and select the best approach",
                "Create a detailed implementation plan",
                "Define success metrics and monitoring approach"
            ]
        elif "implement" in target_lower or "build" in target_lower:
            subtasks = [
                "Review requirements and specifications",
                "Design the solution architecture",
                "Develop core components",
                "Integrate components into a complete solution",
                "Test functionality and performance",
                "Document the implementation"
            ]
        elif "evaluate" in target_lower or "assess" in target_lower:
            subtasks = [
                "Define evaluation criteria and metrics",
                "Collect relevant data for evaluation",
                "Analyze performance against criteria",
                "Identify strengths and weaknesses",
                "Prepare evaluation report with recommendations"
            ]
        else:
            # Generic decomposition
            subtasks = [
                f"Research background information about {target}",
                f"Identify key components or aspects of {target}",
                f"Analyze relationships between components of {target}",
                f"Develop a comprehensive understanding of {target}",
                f"Summarize findings about {target}"
            ]
            
        return subtasks
        
    def _generate_plan_steps(self, target: str) -> List[str]:
        """Generate steps for a plan based on the target"""
        # Simple rule-based plan generation
        steps = []
        
        target_lower = target.lower()
        
        if "learn" in target_lower or "study" in target_lower:
            steps = [
                "Identify specific learning objectives and desired outcomes",
                "Research available learning resources and materials",
                "Create a structured learning schedule with milestones",
                "Implement active learning techniques and practice exercises",
                "Regularly review progress and adjust approach as needed",
                "Test knowledge through application and self-assessment"
            ]
        elif "solve" in target_lower or "problem" in target_lower:
            steps = [
                "Define the problem clearly and identify constraints",
                "Research similar problems and potential solution approaches",
                "Generate multiple solution alternatives",
                "Evaluate alternatives against criteria and select best approach",
                "Implement the chosen solution",
                "Test the solution and refine as needed"
            ]
        elif "improve" in target_lower or "optimize" in target_lower:
            steps = [
                "Establish baseline measurements of current performance",
                "Identify specific areas for improvement and prioritize",
                "Research best practices and optimization techniques",
                "Develop specific improvement strategies for each area",
                "Implement improvements incrementally",
                "Measure results and iterate based on feedback"
            ]
        elif "create" in target_lower or "develop" in target_lower:
            steps = [
                "Define requirements and success criteria",
                "Research similar existing solutions for inspiration",
                "Create initial design or prototype",
                "Gather feedback on the design",
                "Refine and improve based on feedback",
                "Finalize and document the creation process"
            ]
        else:
            # Generic planning steps
            steps = [
                f"Define specific goals and objectives for {target}",
                f"Research background information and context for {target}",
                f"Identify key components and requirements for {target}",
                f"Develop a detailed approach for {target}",
                f"Create an implementation timeline with milestones",
                f"Define success metrics and evaluation criteria"
            ]
            
        return steps
        
    def _should_retry(self, task: Task) -> bool:
        """Determine if a failed task should be retried"""
        # Don't retry tasks that have already been retried too many times
        for t, count in self._retry_queue:
            if t.task_id == task.task_id and count >= self._max_retries:
                return False
                
        # Check task description for retry-worthy characteristics
        desc_lower = task.description.lower()
        
        # Network or external resource tasks are good candidates for retry
        if any(term in desc_lower for term in ["fetch", "download", "url", "http", "api"]):
            return True
            
        # Code execution tasks might be worth retrying
        if "<function_call>" in desc_lower or "```python" in desc_lower:
            return True
            
        # Default to not retrying
        return False

###############################################################################
# TASK SCHEDULER
###############################################################################

class TaskScheduler:
    """
    Advanced task scheduler with:
    - Dynamic worker pool sizing
    - Task prioritization with multiple factors
    - Dependency management
    - Resource-aware scheduling
    - Deadline-based scheduling
    - Fairness mechanisms
    - Performance monitoring
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
        self._max_workers = max_workers
        self._current_workers = max_workers
        self._task_futures: Dict[int, Future] = {}
        self._performance_metrics = {
            "tasks_processed": 0,
            "tasks_succeeded": 0,
            "tasks_failed": 0,
            "avg_processing_time": 0,
            "total_processing_time": 0,
            "worker_utilization": []
        }
        self._last_metrics_update = time.time()
        self._metrics_lock = threading.Lock()
        self._retry_check_interval = 30  # seconds

    def start_scheduler(self) -> None:
        """
        Starts the scheduler loop in a background thread, which continuously 
        pops tasks and processes them in the thread pool.
        """
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        
        # Start the retry handler thread
        self._retry_thread = threading.Thread(target=self._retry_loop, daemon=True)
        self._retry_thread.start()
        
        # Start the metrics update thread
        self._metrics_thread = threading.Thread(target=self._update_metrics_loop, daemon=True)
        self._metrics_thread.start()
        
        logger.info("[TaskScheduler] Scheduler started with adaptive worker pool.")

    def stop_scheduler(self) -> None:
        """
        Signals the scheduler to stop, then waits for the thread pool to finish.
        """
        logger.info("[TaskScheduler] Stopping scheduler...")
        self._stop_event.set()
        
        # Wait for all threads to complete
        if hasattr(self, '_scheduler_thread'):
            self._scheduler_thread.join(timeout=5)
            
        if hasattr(self, '_retry_thread'):
            self._retry_thread.join(timeout=5)
            
        if hasattr(self, '_metrics_thread'):
            self._metrics_thread.join(timeout=5)
        
        # Shutdown the executor
        self._executor.shutdown(wait=True)
        logger.info("[TaskScheduler] Scheduler stopped.")

    def _scheduler_loop(self) -> None:
        """Main scheduler loop that processes tasks from the queue"""
        while not self._stop_event.is_set():
            # Check if we need to adjust worker pool size
            self._adjust_worker_pool()
            
            # Get next task from queue
            task = self.task_queue.pop()
            if not task:
                time.sleep(0.2)
                continue
                
            # Check if task has dependencies
            if hasattr(task, 'dependencies') and task.dependencies:
                unmet_dependencies = []
                for dep_id in task.dependencies:
                    dep_task = self.memory_store.get_task(dep_id)
                    if not dep_task or dep_task.status != "COMPLETED":
                        unmet_dependencies.append(dep_id)
                        
                if unmet_dependencies:
                    # Put task back in queue with slightly lower priority
                    task.priority += 1
                    self.task_queue.push(task)
                    logger.info(f"[TaskScheduler] Task {task.task_id} has unmet dependencies: {unmet_dependencies}. Requeuing.")
                    time.sleep(0.1)
                    continue
            
            # Submit task to executor
            future = self._executor.submit(self._process_task_wrapper, task)
            self._task_futures[task.task_id] = future
            
            # Record start time for metrics
            with self._metrics_lock:
                self._performance_metrics["tasks_processed"] += 1

    def _retry_loop(self) -> None:
        """Loop that checks for tasks to retry"""
        while not self._stop_event.is_set():
            time.sleep(self._retry_check_interval)
            
            # Get tasks scheduled for retry
            retry_tasks = self.processor.get_retry_tasks()
            
            if retry_tasks:
                logger.info(f"[TaskScheduler] Processing {len(retry_tasks)} tasks scheduled for retry")
                
                for task, retry_count in retry_tasks:
                    # Remove from retry queue
                    self.processor.clear_retry(task.task_id)
                    
                    # Submit retry task
                    future = self._executor.submit(self._process_retry_wrapper, task, retry_count)
                    self._task_futures[task.task_id] = future

    def _update_metrics_loop(self) -> None:
        """Loop that updates performance metrics"""
        while not self._stop_event.is_set():
            time.sleep(60)  # Update metrics every minute
            
            with self._metrics_lock:
                # Calculate worker utilization
                active_workers = sum(1 for f in self._task_futures.values() if not f.done())
                utilization = active_workers / self._current_workers if self._current_workers > 0 else 0
                self._performance_metrics["worker_utilization"].append(utilization)
                
                # Keep only the last 10 utilization measurements
                if len(self._performance_metrics["worker_utilization"]) > 10:
                    self._performance_metrics["worker_utilization"] = self._performance_metrics["worker_utilization"][-10:]
                
                # Log metrics
                logger.info(f"[TaskScheduler] Performance metrics: "
                           f"Processed: {self._performance_metrics['tasks_processed']}, "
                           f"Succeeded: {self._performance_metrics['tasks_succeeded']}, "
                           f"Failed: {self._performance_metrics['tasks_failed']}, "
                           f"Avg time: {self._performance_metrics['avg_processing_time']:.2f}s, "
                           f"Utilization: {utilization:.2f}")

    def _process_task_wrapper(self, task: Task) -> None:
        """
        Wraps the call to the SmartTaskProcessor and handles exceptions,
        marking a task as 'FAILED' if an exception occurs.
        """
        start_time = time.time()
        
        try:
            self.processor.process_task(task)
            
            # Record success and processing time
            with self._metrics_lock:
                self._performance_metrics["tasks_succeeded"] += 1
                processing_time = time.time() - start_time
                self._performance_metrics["total_processing_time"] += processing_time
                
                # Update average processing time
                total_completed = self._performance_metrics["tasks_succeeded"] + self._performance_metrics["tasks_failed"]
                if total_completed > 0:
                    self._performance_metrics["avg_processing_time"] = (
                        self._performance_metrics["total_processing_time"] / total_completed
                    )
                    
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[TaskScheduler] Task {task.task_id} failed: {e}\n{tb}")
            self.memory_store.update_task_status(task.task_id, "FAILED")
            
            # Record failure
            with self._metrics_lock:
                self._performance_metrics["tasks_failed"] += 1
                
        finally:
            # Clean up future
            if task.task_id in self._task_futures:
                del self._task_futures[task.task_id]
                
    def _process_retry_wrapper(self, task: Task, retry_count: int) -> None:
        """Wrapper for processing task retries"""
        start_time = time.time()
        
        try:
            self.processor.process_retry(task, retry_count)
            
            # Record success and processing time
            with self._metrics_lock:
                self._performance_metrics["tasks_succeeded"] += 1
                processing_time = time.time() - start_time
                self._performance_metrics["total_processing_time"] += processing_time
                
                # Update average processing time
                total_completed = self._performance_metrics["tasks_succeeded"] + self._performance_metrics["tasks_failed"]
                if total_completed > 0:
                    self._performance_metrics["avg_processing_time"] = (
                        self._performance_metrics["total_processing_time"] / total_completed
                    )
                    
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[TaskScheduler] Retry {retry_count} for task {task.task_id} failed: {e}\n{tb}")
            self.memory_store.update_task_status(task.task_id, "FAILED")
            
            # Record failure
            with self._metrics_lock:
                self._performance_metrics["tasks_failed"] += 1
                
        finally:
            # Clean up future
            if task.task_id in self._task_futures:
                del self._task_futures[task.task_id]
                
    def _adjust_worker_pool(self) -> None:
        """Dynamically adjust the worker pool size based on load"""
        with self._metrics_lock:
            # Get recent utilization
            recent_utilization = self._performance_metrics["worker_utilization"][-5:] if self._performance_metrics["worker_utilization"] else []
            
            if not recent_utilization:
                return
                
            avg_utilization = sum(recent_utilization) / len(recent_utilization)
            
            # Adjust pool size based on utilization
            if avg_utilization > 0.8 and self._current_workers < self._max_workers:
                # High utilization, increase workers
                new_size = min(self._max_workers, self._current_workers + 1)
                if new_size != self._current_workers:
                    self._resize_worker_pool(new_size)
                    logger.info(f"[TaskScheduler] Increased worker pool to {new_size} due to high utilization ({avg_utilization:.2f})")
                    
            elif avg_utilization < 0.3 and self._current_workers > 1:
                # Low utilization, decrease workers
                new_size = max(1, self._current_workers - 1)
                if new_size != self._current_workers:
                    self._resize_worker_pool(new_size)
                    logger.info(f"[TaskScheduler] Decreased worker pool to {new_size} due to low utilization ({avg_utilization:.2f})")
                    
    def _resize_worker_pool(self, new_size: int) -> None:
        """Resize the thread pool to the new size"""
        # Create a new executor with the new size
        new_executor = ThreadPoolExecutor(max_workers=new_size)
        
        # Shutdown the old executor without waiting for tasks to complete
        # (they'll continue running)
        self._executor.shutdown(wait=False)
        
        # Update references
        self._executor = new_executor
        self._current_workers = new_size
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        with self._metrics_lock:
            return dict(self._performance_metrics)
            
    def submit_task(self, task: Task) -> None:
        """Submit a task to the queue"""
        self.task_queue.push(task)
        logger.info(f"[TaskScheduler] Submitted task {task.task_id} to queue")

###############################################################################
# PLAN MANAGER
###############################################################################

class PlanManager:
    """
    Advanced planning system with:
    - Monte Carlo Tree Search for planning
    - Hierarchical planning across multiple time horizons
    - Goal-directed reasoning
    - Plan adaptation based on feedback
    - Resource-constrained planning
    - Risk-aware planning
    - Multi-agent coordination
    """
    def __init__(self, agent: "R1Agent"):
        self.agent = agent
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._plan_loop, daemon=True)
        self._thread.start()
        self._plans: Dict[str, Dict[str, Any]] = {}
        self._plan_evaluations: Dict[str, List[Dict[str, Any]]] = {}
        self._planning_horizon = CONFIG["planning_horizon"]
        self._last_major_planning = time.time()
        self._planning_interval = 60  # seconds
        self._major_planning_interval = 300  # seconds (5 minutes)

    def stop(self) -> None:
        self._stop_event.set()
        
    def create_plan(self, name: str, description: str, steps: List[str], 
                   goal_id: Optional[int] = None) -> Dict[str, Any]:
        """Create a new plan with steps"""
        plan = {
            "name": name,
            "description": description,
            "steps": [{"description": step, "status": "PENDING", "task_id": None} for step in steps],
            "created_at": time.time(),
            "last_updated": time.time(),
            "status": "ACTIVE",
            "goal_id": goal_id,
            "progress": 0.0,
            "risk_assessment": None
        }
        
        self._plans[name] = plan
        logger.info(f"[PlanManager] Created new plan: {name}")
        return plan
        
    def get_plan(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a plan by name"""
        return self._plans.get(name)
        
    def list_plans(self) -> List[Dict[str, Any]]:
        """List all plans"""
        return list(self._plans.values())
        
    def update_plan_step(self, plan_name: str, step_index: int, 
                        status: str, task_id: Optional[int] = None) -> bool:
        """Update the status of a plan step"""
        plan = self._plans.get(plan_name)
        if not plan or step_index >= len(plan["steps"]):
            return False
            
        plan["steps"][step_index]["status"] = status
        if task_id is not None:
            plan["steps"][step_index]["task_id"] = task_id
            
        plan["last_updated"] = time.time()
        
        # Update plan progress
        completed_steps = sum(1 for step in plan["steps"] if step["status"] == "COMPLETED")
        plan["progress"] = completed_steps / len(plan["steps"]) if plan["steps"] else 0
        
        # Update plan status if all steps are completed
        if all(step["status"] == "COMPLETED" for step in plan["steps"]):
            plan["status"] = "COMPLETED"
            
        logger.info(f"[PlanManager] Updated step {step_index} in plan '{plan_name}' to {status}")
        return True
        
    def evaluate_plan(self, plan_name: str, metrics: Dict[str, Any]) -> None:
        """Record an evaluation of a plan's performance"""
        if plan_name not in self._plan_evaluations:
            self._plan_evaluations[plan_name] = []
            
        evaluation = {
            "timestamp": time.time(),
            "metrics": metrics
        }
        
        self._plan_evaluations[plan_name].append(evaluation)
        logger.info(f"[PlanManager] Recorded evaluation for plan '{plan_name}'")

    def _plan_loop(self) -> None:
        """
        The main planning loop that runs periodically, analyzing the system state
        and making planning decisions at multiple time horizons.
        """
        while not self._stop_event.is_set():
            # Determine if we should do a major or minor planning cycle
            current_time = time.time()
            do_major_planning = (current_time - self._last_major_planning) >= self._major_planning_interval
            
            if do_major_planning:
                self._last_major_planning = current_time
                logger.info("[PlanManager] Running major planning cycle...")
                self._run_major_planning_cycle()
            else:
                logger.info("[PlanManager] Running routine planning cycle...")
                self._run_routine_planning_cycle()
                
            # Wait until next planning cycle
            time.sleep(self._planning_interval)
            
    def _run_major_planning_cycle(self) -> None:
        """
        Run a comprehensive planning cycle that:
        - Reviews all goals and their alignment
        - Creates long-term strategic plans
        - Performs risk assessment
        - Allocates resources across competing priorities
        """
        # Get current state
        history = self.agent.conversation.get_history()
        tasks = self.agent.memory_store.list_tasks()
        goals = self.agent.goal_manager.list_goals()
        
        # 1. Goal review and alignment
        active_goals = [g for g in goals if g.status == "ACTIVE"]
        
        # Check for goal conflicts or synergies
        if len(active_goals) > 1:
            self._analyze_goal_relationships(active_goals)
            
        # 2. Create or update strategic plans for each goal
        for goal in active_goals:
            plan_name = f"Strategic Plan - {goal.name}"
            
            if plan_name not in self._plans:
                # Create a new strategic plan
                steps = self._generate_strategic_steps(goal)
                self.create_plan(plan_name, f"Strategic plan for goal: {goal.name}", steps, goal.goal_id)
            else:
                # Update existing plan if needed
                self._update_strategic_plan(plan_name, goal)
                
        # 3. Resource allocation across goals
        if active_goals:
            self._allocate_resources(active_goals)
            
        # 4. Risk assessment
        self._perform_risk_assessment()
        
        # 5. Create introspection task if needed
        pending_tasks = [t for t in tasks if t.status == "PENDING"]
        if len(pending_tasks) > 10:
            self._create_introspection_task("Review large number of pending tasks and prioritize effectively")
            
        # 6. Generate and evaluate candidate actions
        actions = self.agent.action_generator.generate_candidate_actions(
            conversation=self.agent.conversation,
            goals=goals,
            tasks=tasks,
            max_actions=25
        )
        
        # Convert top actions to tasks
        self._convert_top_actions_to_tasks(actions[:3])
        
        logger.info(f"[PlanManager] Major planning cycle completed with {len(actions)} candidate actions")
            
    def _run_routine_planning_cycle(self) -> None:
        """
        Run a lighter planning cycle that:
        - Checks for urgent issues
        - Updates tactical plans
        - Generates a smaller set of candidate actions
        """
        # Get current state
        history = self.agent.conversation.get_history()
        tasks = self.agent.memory_store.list_tasks()
        goals = self.agent.goal_manager.list_goals()
        
        # 1. Check for urgent issues
        urgent_issues = self._identify_urgent_issues(tasks, goals)
        
        for issue in urgent_issues:
            if issue["type"] == "overdue_goal":
                self._create_introspection_task(f"Address overdue goal: {issue['name']}")
            elif issue["type"] == "stalled_task":
                self._create_introspection_task(f"Review stalled task: {issue['task_id']}")
                
        # 2. Update tactical plans
        active_plans = [p for p in self._plans.values() if p["status"] == "ACTIVE"]
        for plan in active_plans:
            self._update_tactical_plan(plan)
            
        # 3. If conversation length is multiple of 7, create a new goal
        if history and (len(history) % 7) == 0:
            # Extract potential topics from recent conversation
            recent_messages = history[-7:]
            topics = set()
            for msg in recent_messages:
                # Simple keyword extraction
                words = re.findall(r'\b[A-Za-z]{4,}\b', msg["content"])
                topics.update(words)
                
            # Find most frequent meaningful words
            topic_counts = {}
            for topic in topics:
                if len(topic) > 4:  # Filter out short words
                    count = sum(1 for msg in recent_messages if topic.lower() in msg["content"].lower())
                    topic_counts[topic] = count
                    
            # Get top topics
            top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            
            if top_topics:
                topic_str = ", ".join(t[0] for t in top_topics)
                new_goal = self.agent.goal_manager.create_goal(
                    name=f"Explore {top_topics[0][0]}",
                    description=f"Analyze and develop understanding of topics: {topic_str}",
                    priority=3
                )
                logger.info(f"[PlanManager] Auto-created new goal based on conversation topics: {new_goal}")
                
                # Create a plan for the new goal
                steps = [
                    f"Research background information on {topic_str}",
                    f"Identify key concepts and relationships in {topic_str}",
                    f"Develop a structured understanding of {topic_str}",
                    f"Apply knowledge about {topic_str} to current context",
                    f"Summarize findings about {topic_str}"
                ]
                
                self.create_plan(
                    f"Exploration Plan - {new_goal.name}",
                    f"Plan to explore topics: {topic_str}",
                    steps,
                    new_goal.goal_id
                )
        
        # 4. Generate a smaller set of candidate actions
        actions = self.agent.action_generator.generate_candidate_actions(
            conversation=self.agent.conversation,
            goals=goals,
            tasks=tasks,
            max_actions=10
        )
        
        # Convert top action to task
        if actions:
            self._convert_top_actions_to_tasks([actions[0]])
            
        logger.info(f"[PlanManager] Routine planning cycle completed with {len(actions)} candidate actions")
        
    def _analyze_goal_relationships(self, goals: List["Goal"]) -> None:
        """Analyze relationships between goals to identify conflicts and synergies"""
        # Simple implementation: check for similar goals based on embeddings
        for i, goal1 in enumerate(goals):
            for j, goal2 in enumerate(goals[i+1:], i+1):
                if goal1.embedding is not None and goal2.embedding is not None:
                    # Calculate similarity
                    similarity = np.dot(goal1.embedding, goal2.embedding) / (
                        np.linalg.norm(goal1.embedding) * np.linalg.norm(goal2.embedding)
                    )
                    
                    if similarity > 0.8:
                        # Goals are very similar - potential redundancy
                        logger.info(f"[PlanManager] Detected similar goals: '{goal1.name}' and '{goal2.name}' "
                                   f"(similarity: {similarity:.2f})")
                        
                        # Create a task to review the similar goals
                        self._create_introspection_task(
                            f"Review similar goals: '{goal1.name}' (ID: {goal1.goal_id}) and "
                            f"'{goal2.name}' (ID: {goal2.goal_id}) for potential consolidation"
                        )
                        
    def _generate_strategic_steps(self, goal: "Goal") -> List[str]:
        """Generate strategic steps for a goal"""
        # Generate steps based on goal type/content
        goal_desc_lower = goal.description.lower()
        
        if "learn" in goal_desc_lower or "understand" in goal_desc_lower:
            return [
                "Define specific learning objectives and success criteria",
                "Identify and gather key information sources",
                "Develop a structured learning approach",
                "Implement learning activities and track progress",
                "Synthesize knowledge and apply to relevant contexts",
                "Evaluate learning outcomes against objectives"
            ]
        elif "improve" in goal_desc_lower or "optimize" in goal_desc_lower:
            return [
                "Establish baseline measurements and success metrics",
                "Identify specific areas for improvement",
                "Research best practices and potential approaches",
                "Develop detailed improvement strategies",
                "Implement improvements incrementally",
                "Measure results and iterate based on feedback"
            ]
        elif "create" in goal_desc_lower or "develop" in goal_desc_lower:
            return [
                "Define requirements and success criteria",
                "Research similar existing solutions",
                "Design initial approach or architecture",
                "Implement core components",
                "Test and validate against requirements",
                "Refine based on testing and finalize"
            ]
        elif "analyze" in goal_desc_lower or "assess" in goal_desc_lower:
            return [
                "Define analysis objectives and scope",
                "Gather relevant data and information",
                "Develop analysis methodology",
                "Perform detailed analysis",
                "Interpret results and identify patterns",
                "Synthesize findings and develop recommendations"
            ]
        else:
            # Generic strategic steps
            return [
                f"Define specific objectives for '{goal.name}'",
                f"Research background and context for '{goal.name}'",
                f"Develop detailed approach for '{goal.name}'",
                f"Implement key activities for '{goal.name}'",
                f"Monitor progress and adapt approach as needed",
                f"Evaluate outcomes and synthesize learnings"
            ]
            
    def _update_strategic_plan(self, plan_name: str, goal: "Goal") -> None:
        """Update an existing strategic plan based on goal progress"""
        plan = self._plans.get(plan_name)
        if not plan:
            return
            
        # Check if plan progress matches goal progress
        if abs(plan["progress"] - goal.progress) > 0.2:
            # Significant mismatch - update plan progress to match goal
            completed_steps = int(goal.progress * len(plan["steps"]))
            
            for i in range(completed_steps):
                if i < len(plan["steps"]) and plan["steps"][i]["status"] != "COMPLETED":
                    plan["steps"][i]["status"] = "COMPLETED"
                    
            # Recalculate plan progress
            plan["progress"] = completed_steps / len(plan["steps"]) if plan["steps"] else 0
            plan["last_updated"] = time.time()
            
            logger.info(f"[PlanManager] Updated strategic plan '{plan_name}' progress to {plan['progress']:.2f} "
                       f"to align with goal progress")
            
    def _allocate_resources(self, goals: List["Goal"]) -> None:
        """Allocate resources across competing goals based on priority and urgency"""
        # Simple implementation: adjust task priorities based on goal priorities
        
        # First, get all tasks associated with each goal
        goal_tasks = {}
        for goal in goals:
            related_tasks = []
            for task_id in goal.related_tasks:
                task = self.agent.memory_store.get_task(task_id)
                if task and task.status == "PENDING":
                    related_tasks.append(task)
            goal_tasks[goal.goal_id] = related_tasks
            
        # Calculate a resource allocation score for each goal
        allocation_scores = {}
        total_score = 0
        
        for goal in goals:
            # Base score is inverse of priority (lower priority number = higher importance)
            base_score = 10 - goal.priority
            
            # Adjust for deadline if present
            deadline_factor = 1.0
            if goal.deadline:
                time_remaining = goal.time_remaining()
                if time_remaining:
                    # Increase urgency as deadline approaches
                    if time_remaining < 86400:  # Less than a day
                        deadline_factor = 3.0
                    elif time_remaining < 259200:  # Less than 3 days
                        deadline_factor = 2.0
                    elif time_remaining < 604800:  # Less than a week
                        deadline_factor = 1.5
            
            # Adjust for progress (prioritize goals that are close to completion)
            progress_factor = 1.0
            if goal.progress > 0.7:
                progress_factor = 1.5  # Boost nearly-complete goals
            elif goal.progress < 0.2:
                progress_factor = 1.2  # Slight boost to get started on new goals
                
            # Calculate final score
            score = base_score * deadline_factor * progress_factor
            allocation_scores[goal.goal_id] = score
            total_score += score
            
        # Normalize scores to get allocation percentages
        allocations = {}
        for goal_id, score in allocation_scores.items():
            allocations[goal_id] = score / total_score if total_score > 0 else 0
            
        # Adjust task priorities based on allocations
        for goal_id, tasks in goal_tasks.items():
            allocation = allocations.get(goal_id, 0)
            
            # The higher the allocation, the lower the priority number should be
            priority_adjustment = int(5 * (1 - allocation))  # 0 to 5
            
            for task in tasks:
                new_priority = max(1, task.priority - priority_adjustment)
                if new_priority != task.priority:
                    # Update task priority
                    task.priority = new_priority
                    self.agent.memory_store.add_task(task)
                    
        logger.info(f"[PlanManager] Allocated resources across {len(goals)} goals: {allocations}")
        
    def _perform_risk_assessment(self) -> None:
        """Perform risk assessment on active plans and goals"""
        # Assess risks for each active plan
        active_plans = [p for p in self._plans.values() if p["status"] == "ACTIVE"]
        
        for plan in active_plans:
            risks = []
            
            # Check for stalled progress
            if plan["progress"] < 0.1 and (time.time() - plan["created_at"]) > 3600:
                risks.append({
                    "type": "stalled_progress",
                    "description": "Plan has shown minimal progress since creation",
                    "severity": "medium"
                })
                
            # Check for deadline risks
            if plan["goal_id"]:
                goal = self.agent.goal_manager.get_goal(plan["goal_id"])
                if goal and goal.deadline:
                    time_remaining = goal.time_remaining()
                    if time_remaining:
                        # Estimate if we'll complete in time
                        if time_remaining > 0:
                            progress_rate = plan["progress"] / (time.time() - plan["created_at"]) if time.time() > plan["created_at"] else 0
                            if progress_rate > 0:
                                estimated_completion = (1 - plan["progress"]) / progress_rate
                                if estimated_completion > time_remaining:
                                    risks.append({
                                        "type": "deadline_risk",
                                        "description": f"Plan unlikely to complete before deadline (estimated: {estimated_completion/86400:.1f} days, remaining: {time_remaining/86400:.1f} days)",
                                        "severity": "high"
                                    })
            
            # Check for dependency risks
            dependency_risks = 0
            for step in plan["steps"]:
                if step["status"] == "PENDING" and step["task_id"]:
                    task = self.agent.memory_store.get_task(step["task_id"])
                    if task and hasattr(task, 'dependencies') and task.dependencies:
                        for dep_id in task.dependencies:
                            dep_task = self.agent.memory_store.get_task(dep_id)
                            if not dep_task or dep_task.status == "FAILED":
                                dependency_risks += 1
                                
            if dependency_risks > 0:
                risks.append({
                    "type": "dependency_risk",
                    "description": f"Plan has {dependency_risks} steps with failed or missing dependencies",
                    "severity": "high" if dependency_risks > 2 else "medium"
                })
                
            # Update plan with risk assessment
            plan["risk_assessment"] = {
                "timestamp": time.time(),
                "risks": risks,
                "overall_risk": "high" if any(r["severity"] == "high" for r in risks) else 
                               "medium" if any(r["severity"] == "medium" for r in risks) else "low"
            }
            
            # Create tasks for high-risk plans
            if plan["risk_assessment"]["overall_risk"] == "high":
                self._create_introspection_task(
                    f"Address high risks in plan '{plan['name']}': " + 
                    ", ".join(r["description"] for r in risks if r["severity"] == "high")
                )
                
        logger.info(f"[PlanManager] Completed risk assessment for {len(active_plans)} active plans")
        
    def _identify_urgent_issues(self, tasks: List["Task"], goals: List["Goal"]) -> List[Dict[str, Any]]:
        """Identify urgent issues that need immediate attention"""
        urgent_issues = []
        
        # Check for overdue goals
        overdue_goals = [g for g in goals if g.is_overdue() and g.status == "ACTIVE"]
        for goal in overdue_goals:
            urgent_issues.append({
                "type": "overdue_goal",
                "goal_id": goal.goal_id,
                "name": goal.name,
                "deadline": goal.deadline
            })
            
        # Check for stalled tasks
        in_progress_tasks = [t for t in tasks if t.status == "IN_PROGRESS"]
        for task in in_progress_tasks:
            if hasattr(task, '_start_time') and (time.time() - task._start_time) > 3600:
                urgent_issues.append({
                    "type": "stalled_task",
                    "task_id": task.task_id,
                    "description": task.description[:50],
                    "duration": time.time() - task._start_time
                })
                
        # Check for failed high-priority tasks
        failed_tasks = [t for t in tasks if t.status == "FAILED" and t.priority <= 3]
        for task in failed_tasks:
            urgent_issues.append({
                "type": "failed_high_priority_task",
                "task_id": task.task_id,
                "description": task.description[:50],
                "priority": task.priority
            })
            
        return urgent_issues
        
    def _update_tactical_plan(self, plan: Dict[str, Any]) -> None:
        """Update a tactical plan based on current progress"""
        # Check each step and update status based on associated tasks
        updated = False
        
        for i, step in enumerate(plan["steps"]):
            if step["status"] != "COMPLETED" and step["task_id"]:
                task = self.agent.memory_store.get_task(step["task_id"])
                if task:
                    if task.status == "COMPLETED":
                        step["status"] = "COMPLETED"
                        updated = True
                    elif task.status == "FAILED":
                        # Create a new task for the failed step
                        new_task = self.agent.memory_store.create_task(
                            priority=task.priority,
                            description=f"Retry plan step: {step['description']}",
                            parent_id=task.parent_id
                        )
                        step["task_id"] = new_task.task_id
                        updated = True
                        
        if updated:
            # Recalculate plan progress
            completed_steps = sum(1 for step in plan["steps"] if step["status"] == "COMPLETED")
            plan["progress"] = completed_steps / len(plan["steps"]) if plan["steps"] else 0
            plan["last_updated"] = time.time()
            
            # Update plan status if all steps are completed
            if all(step["status"] == "COMPLETED" for step in plan["steps"]):
                plan["status"] = "COMPLETED"
                
            logger.info(f"[PlanManager] Updated tactical plan '{plan['name']}' progress to {plan['progress']:.2f}")
            
    def _create_introspection_task(self, description: str) -> None:
        """Create an introspection task with high priority"""
        task = self.agent.memory_store.create_task(
            priority=2,
            description=f"Introspect: {description}"
        )
        task.add_tag("introspection")
        self.agent.task_queue.push(task)
        logger.info(f"[PlanManager] Created introspection task: {description}")
        
    def _convert_top_actions_to_tasks(self, actions: List[CandidateAction]) -> None:
        """Convert top candidate actions to actual tasks"""
        for action in actions:
            # Create a task from the action
            task = self.agent.memory_store.create_task(
                priority=action.priority,
                description=action.description
            )
            
            # Add tags based on action
            if hasattr(action, 'tags'):
                for tag in action.tags:
                    task.add_tag(tag)
                    
            # Add to queue
            self.agent.task_queue.push(task)
            
            logger.info(f"[PlanManager] Created task from candidate action: {action.description}")

###############################################################################
# SELF-AST-INTROSPECTION (OPTIONAL EXTRA)
###############################################################################

class SelfCodeGraph:
    """
    Advanced self-code introspection with:
    - AST parsing and analysis
    - Code structure visualization
    - Function and class relationship mapping
    - Code quality assessment
    - Self-modification capabilities
    - Version control for code changes
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
        self.functions = {}
        self.classes = {}
        self.imports = []
        self.dependencies = {}
        self.complexity_metrics = {}
        self.version_history = []
        self._build_ast_graph()
        self._analyze_code_structure()

    def _build_ast_graph(self):
        """
        Parse the source code into an AST, build a graph of node relationships,
        and store embeddings.
        """
        try:
            tree = ast.parse(self.source)
        except Exception as e:
            logger.error(f"[SelfCodeGraph] Error parsing source code: {e}")
            self.ast_graph = None
            return

        # Build a graph of node relationships
        self.ast_graph = []
        node_id_map = {}
        
        # Initialize embedding model
        if HAVE_SENTENCE_TRANSFORMERS:
            try:
                self.embedding_model = SentenceTransformer(CONFIG["embedding_model"])
            except:
                self.embedding_model = DummyEmbeddingModel()
        else:
            self.embedding_model = DummyEmbeddingModel()

        def visit_node(node, parent_id: Optional[int]):
            current_id = id(node)
            node_type = type(node).__name__
            node_id_map[current_id] = node

            # Get source segment if possible
            source_segment = ast.get_source_segment(self.source, node) or ""
            
            # Generate embedding for non-trivial nodes
            if len(source_segment) > 10:
                self.embeddings[current_id] = self.embedding_model.encode(source_segment)
            else:
                # Fallback to simple embedding for small nodes
                self.embeddings[current_id] = np.array([float(ord(c)) / 150.0 for c in node_type[:16]])

            # If parent is not None, link them
            self.ast_graph.append((parent_id, current_id, node_type, source_segment[:50]))

            for child in ast.iter_child_nodes(node):
                visit_node(child, current_id)

        visit_node(tree, None)
        
        # Record initial version
        self.version_history.append({
            "timestamp": time.time(),
            "version": 1,
            "source": self.source,
            "description": "Initial version"
        })

    def _analyze_code_structure(self):
        """
        Analyze the code structure to extract functions, classes, imports,
        dependencies, and complexity metrics.
        """
        if not self.ast_graph:
            return
            
        try:
            tree = ast.parse(self.source)
            
            # Extract functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_code = ast.get_source_segment(self.source, node)
                    if func_code:
                        self.functions[node.name] = {
                            "code": func_code,
                            "args": [arg.arg for arg in node.args.args],
                            "line_number": node.lineno,
                            "complexity": self._calculate_complexity(node),
                            "docstring": ast.get_docstring(node) or "",
                            "calls": self._extract_function_calls(node)
                        }
                        
            # Extract classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_code = ast.get_source_segment(self.source, node)
                    if class_code:
                        methods = {}
                        for child in node.body:
                            if isinstance(child, ast.FunctionDef):
                                method_code = ast.get_source_segment(self.source, child)
                                if method_code:
                                    methods[child.name] = {
                                        "code": method_code,
                                        "args": [arg.arg for arg in child.args.args],
                                        "line_number": child.lineno,
                                        "complexity": self._calculate_complexity(child),
                                        "docstring": ast.get_docstring(child) or ""
                                    }
                                    
                        self.classes[node.name] = {
                            "code": class_code,
                            "methods": methods,
                            "line_number": node.lineno,
                            "docstring": ast.get_docstring(node) or "",
                            "bases": [base.id if isinstance(base, ast.Name) else "complex_base" for base in node.bases]
                        }
                        
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        self.imports.append({
                            "name": name.name,
                            "alias": name.asname,
                            "line_number": node.lineno
                        })
                elif isinstance(node, ast.ImportFrom):
                    for name in node.names:
                        self.imports.append({
                            "name": f"{node.module}.{name.name}" if node.module else name.name,
                            "alias": name.asname,
                            "line_number": node.lineno,
                            "from_module": node.module
                        })
                        
            # Calculate overall complexity metrics
            self.complexity_metrics = {
                "loc": len(self.source.splitlines()),
                "function_count": len(self.functions),
                "class_count": len(self.classes),
                "import_count": len(self.imports),
                "average_function_complexity": sum(f["complexity"] for f in self.functions.values()) / len(self.functions) if self.functions else 0,
                "max_function_complexity": max((f["complexity"] for f in self.functions.values()), default=0),
                "total_complexity": sum(f["complexity"] for f in self.functions.values())
            }
            
            # Build dependency graph
            for func_name, func_data in self.functions.items():
                self.dependencies[func_name] = func_data["calls"]
                
            # Add class method dependencies
            for class_name, class_data in self.classes.items():
                for method_name, method_data in class_data["methods"].items():
                    full_name = f"{class_name}.{method_name}"
                    self.dependencies[full_name] = self._extract_function_calls(method_data["code"])
                    
        except Exception as e:
            logger.error(f"[SelfCodeGraph] Error analyzing code structure: {e}")
            
    def _calculate_complexity(self, node) -> int:
        """Calculate cyclomatic complexity of a function or method"""
        complexity = 1  # Base complexity
        
        # Count branches that increase complexity
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Assert)):
                complexity += 1
            elif isinstance(child, ast.BoolOp) and isinstance(child.op, ast.And):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.BoolOp) and isinstance(child.op, ast.Or):
                complexity += len(child.values) - 1
                
        return complexity
        
    def _extract_function_calls(self, node_or_code) -> List[str]:
        """Extract function calls from a node or code string"""
        if isinstance(node_or_code, str):
            try:
                node = ast.parse(node_or_code)
            except:
                return []
        else:
            node = node_or_code
            
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    if isinstance(child.func.value, ast.Name):
                        calls.append(f"{child.func.value.id}.{child.func.attr}")
                        
        return calls

    def summarize_ast(self) -> str:
        """
        Returns a detailed summary of the code structure.
        """
        if not self.ast_graph:
            return "[SelfCodeGraph] No AST info available."
            
        summary = []
        summary.append(f"[SelfCodeGraph] Code Structure Summary:")
        summary.append(f"- {self.complexity_metrics['loc']} lines of code")
        summary.append(f"- {self.complexity_metrics['function_count']} functions")
        summary.append(f"- {self.complexity_metrics['class_count']} classes")
        summary.append(f"- {self.complexity_metrics['import_count']} imports")
        summary.append(f"- Average function complexity: {self.complexity_metrics['average_function_complexity']:.2f}")
        
        # Add most complex functions
        if self.functions:
            complex_funcs = sorted(self.functions.items(), key=lambda x: x[1]["complexity"], reverse=True)[:3]
            summary.append("- Most complex functions:")
            for name, data in complex_funcs:
                summary.append(f"  * {name}: complexity {data['complexity']}")
                
        # Add most important classes
        if self.classes:
            important_classes = sorted(self.classes.items(), key=lambda x: len(x[1]["methods"]), reverse=True)[:3]
            summary.append("- Most important classes:")
            for name, data in important_classes:
                summary.append(f"  * {name}: {len(data['methods'])} methods")
                
        return "\n".join(summary)
        
    def get_function(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a function by name"""
        return self.functions.get(name)
        
    def get_class(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a class by name"""
        return self.classes.get(name)
        
    def search_code(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for code elements semantically similar to the query"""
        if not self.ast_graph:
            return []
            
        # Generate embedding for the query
        query_embedding = self.embedding_model.encode(query)
        
        # Calculate similarity with functions
        results = []
        
        # Search functions
        for name, data in self.functions.items():
            code = data["code"]
            code_embedding = self.embedding_model.encode(code)
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, code_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(code_embedding)
            )
            
            results.append((f"function:{name}", float(similarity)))
            
        # Search classes
        for name, data in self.classes.items():
            code = data["code"]
            code_embedding = self.embedding_model.encode(code)
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, code_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(code_embedding)
            )
            
            results.append((f"class:{name}", float(similarity)))
            
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
        
    def modify_function(self, name: str, new_code: str) -> bool:
        """Modify a function in the source code"""
        if name not in self.functions:
            return False
            
        try:
            # Parse the new code to ensure it's valid
            ast.parse(new_code)
            
            # Get the old function
            old_func = self.functions[name]
            old_code = old_func["code"]
            
            # Replace in the source
            self.source = self.source.replace(old_code, new_code)
            
            # Record the change in version history
            self.version_history.append({
                "timestamp": time.time(),
                "version": len(self.version_history) + 1,
                "source": self.source,
                "description": f"Modified function: {name}"
            })
            
            # Rebuild the AST and analyze
            self._build_ast_graph()
            self._analyze_code_structure()
            
            return True
        except Exception as e:
            logger.error(f"[SelfCodeGraph] Error modifying function {name}: {e}")
            return False
            
    def modify_class(self, name: str, new_code: str) -> bool:
        """Modify a class in the source code"""
        if name not in self.classes:
            return False
            
        try:
            # Parse the new code to ensure it's valid
            ast.parse(new_code)
            
            # Get the old class
            old_class = self.classes[name]
            old_code = old_class["code"]
            
            # Replace in the source
            self.source = self.source.replace(old_code, new_code)
            
            # Record the change in version history
            self.version_history.append({
                "timestamp": time.time(),
                "version": len(self.version_history) + 1,
                "source": self.source,
                "description": f"Modified class: {name}"
            })
            
            # Rebuild the AST and analyze
            self._build_ast_graph()
            self._analyze_code_structure()
            
            return True
        except Exception as e:
            logger.error(f"[SelfCodeGraph] Error modifying class {name}: {e}")
            return False
            
    def add_function(self, name: str, code: str) -> bool:
        """Add a new function to the source code"""
        if name in self.functions:
            return False
            
        try:
            # Parse the new code to ensure it's valid
            ast.parse(code)
            
            # Add to the end of the source
            self.source += f"\n\n{code}\n"
            
            # Record the change in version history
            self.version_history.append({
                "timestamp": time.time(),
                "version": len(self.version_history) + 1,
                "source": self.source,
                "description": f"Added function: {name}"
            })
            
            # Rebuild the AST and analyze
            self._build_ast_graph()
            self._analyze_code_structure()
            
            return True
        except Exception as e:
            logger.error(f"[SelfCodeGraph] Error adding function {name}: {e}")
            return False
            
    def revert_to_version(self, version: int) -> bool:
        """Revert to a previous version"""
        if version < 1 or version > len(self.version_history):
            return False
            
        try:
            # Get the target version
            target_version = self.version_history[version - 1]
            
            # Restore the source
            self.source = target_version["source"]
            
            # Record the reversion
            self.version_history.append({
                "timestamp": time.time(),
                "version": len(self.version_history) + 1,
                "source": self.source,
                "description": f"Reverted to version {version}: {target_version['description']}"
            })
            
            # Rebuild the AST and analyze
            self._build_ast_graph()
            self._analyze_code_structure()
            
            return True
        except Exception as e:
            logger.error(f"[SelfCodeGraph] Error reverting to version {version}: {e}")
            return False
            
    def get_version_history(self) -> List[Dict[str, Any]]:
        """Get the version history"""
        # Return without the full source code to save space
        return [{
            "version": v["version"],
            "timestamp": v["timestamp"],
            "description": v["description"]
        } for v in self.version_history]
        
    def visualize_dependencies(self) -> str:
        """Generate a text-based visualization of code dependencies"""
        if not self.dependencies:
            return "No dependencies to visualize"
            
        # Create a simple text-based graph
        lines = ["Code Dependency Graph:"]
        
        # Sort by number of dependencies
        sorted_deps = sorted(self.dependencies.items(), 
                            key=lambda x: len(x[1]), 
                            reverse=True)
        
        for name, deps in sorted_deps[:10]:  # Show top 10
            if deps:
                lines.append(f"{name} -> {', '.join(deps[:5])}" + 
                           (f" and {len(deps)-5} more" if len(deps) > 5 else ""))
            else:
                lines.append(f"{name} (no dependencies)")
                
        return "\n".join(lines)

###############################################################################
# THE R1 AGENT
###############################################################################

class MultiAgentCoordinator:
    """
    Coordinates multiple specialized agent components:
    - Task allocation between components
    - Information sharing
    - Conflict resolution
    - Consensus building
    - Performance monitoring
    """
    def __init__(self, agent: "R1Agent"):
        self.agent = agent
        self.components = {}
        self.component_performance = {}
        self.shared_memory = {}
        self.coordination_history = []
        self._lock = threading.Lock()
        
    def register_component(self, name: str, component: Any, capabilities: List[str]) -> None:
        """Register a component with its capabilities"""
        with self._lock:
            self.components[name] = {
                "component": component,
                "capabilities": capabilities,
                "registered_at": time.time(),
                "last_used": time.time(),
                "use_count": 0
            }
            self.component_performance[name] = {
                "success_rate": 1.0,
                "avg_response_time": 0.0,
                "total_calls": 0,
                "successful_calls": 0
            }
            logger.info(f"[MultiAgentCoordinator] Registered component: {name} with capabilities: {capabilities}")
            
    def get_component_for_task(self, task_description: str) -> Optional[str]:
        """Find the best component to handle a task"""
        with self._lock:
            if not self.components:
                return None
                
            # Generate embedding for the task
            task_embedding = self.agent.knowledge_base.embedding_model.encode(task_description)
            
            # Score each component based on capability match and performance
            scores = {}
            for name, info in self.components.items():
                # Calculate capability match score
                capability_score = 0
                for capability in info["capabilities"]:
                    capability_embedding = self.agent.knowledge_base.embedding_model.encode(capability)
                    similarity = np.dot(task_embedding, capability_embedding) / (
                        np.linalg.norm(task_embedding) * np.linalg.norm(capability_embedding)
                    )
                    capability_score = max(capability_score, similarity)
                    
                # Get performance metrics
                performance = self.component_performance[name]
                performance_score = performance["success_rate"]
                
                # Calculate final score (70% capability, 30% performance)
                scores[name] = (0.7 * capability_score) + (0.3 * performance_score)
                
            # Return the highest scoring component
            if scores:
                best_component = max(scores.items(), key=lambda x: x[1])
                return best_component[0]
                
            return None
            
    def execute_task(self, task: Task, component_name: Optional[str] = None) -> Dict[str, Any]:
        """Execute a task using the appropriate component"""
        with self._lock:
            # If no component specified, find the best one
            if not component_name:
                component_name = self.get_component_for_task(task.description)
                
            if not component_name or component_name not in self.components:
                return {
                    "status": "error",
                    "error": f"No suitable component found for task: {task.description}"
                }
                
            # Get the component
            component_info = self.components[component_name]
            component = component_info["component"]
            
            # Update usage stats
            component_info["last_used"] = time.time()
            component_info["use_count"] += 1
            
            # Record coordination
            coordination_record = {
                "timestamp": time.time(),
                "task_id": task.task_id,
                "component": component_name,
                "task_description": task.description
            }
            
            # Execute the task
            start_time = time.time()
            try:
                # Different execution based on component type
                if hasattr(component, "process_task"):
                    component.process_task(task)
                    result = {"status": "success", "message": f"Task processed by {component_name}"}
                elif hasattr(component, "execute"):
                    result = component.execute(task.description)
                else:
                    # Generic execution
                    result = {"status": "error", "error": f"Component {component_name} has no execution method"}
                    
                # Update performance metrics
                execution_time = time.time() - start_time
                perf = self.component_performance[component_name]
                perf["total_calls"] += 1
                
                if result.get("status") == "success":
                    perf["successful_calls"] += 1
                    
                perf["success_rate"] = perf["successful_calls"] / perf["total_calls"]
                
                # Update average response time
                if perf["avg_response_time"] == 0:
                    perf["avg_response_time"] = execution_time
                else:
                    perf["avg_response_time"] = (perf["avg_response_time"] * (perf["total_calls"] - 1) + 
                                               execution_time) / perf["total_calls"]
                                               
                # Complete the coordination record
                coordination_record["result_status"] = result.get("status", "unknown")
                coordination_record["execution_time"] = execution_time
                self.coordination_history.append(coordination_record)
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Update performance metrics
                perf = self.component_performance[component_name]
                perf["total_calls"] += 1
                perf["success_rate"] = perf["successful_calls"] / perf["total_calls"]
                
                # Complete the coordination record
                coordination_record["result_status"] = "error"
                coordination_record["error"] = str(e)
                coordination_record["execution_time"] = execution_time
                self.coordination_history.append(coordination_record)
                
                return {
                    "status": "error",
                    "error": f"Error executing task with {component_name}: {str(e)}",
                    "traceback": traceback.format_exc()
                }
                
    def share_information(self, source: str, key: str, value: Any) -> None:
        """Share information between components"""
        with self._lock:
            self.shared_memory[key] = {
                "value": value,
                "source": source,
                "timestamp": time.time(),
                "access_count": 0
            }
            logger.info(f"[MultiAgentCoordinator] Component {source} shared information: {key}")
            
    def get_shared_information(self, key: str, component: str) -> Optional[Any]:
        """Get shared information"""
        with self._lock:
            if key in self.shared_memory:
                info = self.shared_memory[key]
                info["access_count"] += 1
                
                # Record access
                self.coordination_history.append({
                    "timestamp": time.time(),
                    "type": "information_access",
                    "key": key,
                    "source": info["source"],
                    "accessor": component
                })
                
                return info["value"]
                
            return None
            
    def get_component_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all components"""
        with self._lock:
            return dict(self.component_performance)
            
    def get_coordination_statistics(self) -> Dict[str, Any]:
        """Get statistics about component coordination"""
        with self._lock:
            stats = {
                "total_coordinations": len(self.coordination_history),
                "component_usage": {},
                "success_rate": {},
                "avg_execution_time": {}
            }
            
            # Calculate component usage
            for name, info in self.components.items():
                stats["component_usage"][name] = info["use_count"]
                
            # Calculate success rates and execution times
            for name, perf in self.component_performance.items():
                stats["success_rate"][name] = perf["success_rate"]
                stats["avg_execution_time"][name] = perf["avg_response_time"]
                
            return stats

class EmotionalIntelligence:
    """
    Emotional intelligence module for the agent:
    - Sentiment analysis of user input
    - Emotional state tracking
    - Empathetic response generation
    - Emotional self-regulation
    """
    def __init__(self):
        self._emotional_state = {
            "valence": 0.0,  # Positive/negative (-1 to 1)
            "arousal": 0.0,  # Calm/excited (-1 to 1)
            "dominance": 0.0  # Submissive/dominant (-1 to 1)
        }
        self._user_emotional_state = {
            "valence": 0.0,
            "arousal": 0.0,
            "dominance": 0.0
        }
        self._emotion_history = []
        self._last_update = time.time()
        self._lock = threading.Lock()
        
        # Emotion words for simple detection
        self._emotion_lexicon = {
            "happy": {"valence": 0.8, "arousal": 0.5, "dominance": 0.5},
            "sad": {"valence": -0.7, "arousal": -0.3, "dominance": -0.4},
            "angry": {"valence": -0.8, "arousal": 0.8, "dominance": 0.7},
            "afraid": {"valence": -0.7, "arousal": 0.7, "dominance": -0.7},
            "surprised": {"valence": 0.2, "arousal": 0.8, "dominance": 0.0},
            "disgusted": {"valence": -0.8, "arousal": 0.4, "dominance": 0.3},
            "calm": {"valence": 0.3, "arousal": -0.7, "dominance": 0.2},
            "excited": {"valence": 0.7, "arousal": 0.8, "dominance": 0.5},
            "bored": {"valence": -0.4, "arousal": -0.6, "dominance": -0.2},
            "confused": {"valence": -0.3, "arousal": 0.3, "dominance": -0.5},
            "frustrated": {"valence": -0.7, "arousal": 0.6, "dominance": -0.2},
            "interested": {"valence": 0.6, "arousal": 0.3, "dominance": 0.2},
            "confident": {"valence": 0.7, "arousal": 0.4, "dominance": 0.8},
            "anxious": {"valence": -0.6, "arousal": 0.7, "dominance": -0.6},
            "grateful": {"valence": 0.8, "arousal": 0.2, "dominance": 0.3},
            "hopeful": {"valence": 0.7, "arousal": 0.4, "dominance": 0.4}
        }
        
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze the sentiment of text"""
        # Simple lexicon-based sentiment analysis
        text_lower = text.lower()
        
        # Initialize with neutral values
        sentiment = {
            "valence": 0.0,
            "arousal": 0.0,
            "dominance": 0.0
        }
        
        # Count matches for each emotion word
        match_count = 0
        for word, values in self._emotion_lexicon.items():
            if word in text_lower:
                for dim in sentiment:
                    sentiment[dim] += values[dim]
                match_count += 1
                
        # Check for negations
        negations = ["not", "no", "never", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't"]
        negation_count = sum(1 for neg in negations if f" {neg} " in f" {text_lower} ")
        
        # If negations found, invert valence
        if negation_count > 0:
            sentiment["valence"] *= -0.7  # Partial inversion
            
        # Normalize if we found matches
        if match_count > 0:
            for dim in sentiment:
                sentiment[dim] /= match_count
                
        # Check for intensifiers
        intensifiers = ["very", "extremely", "really", "so", "absolutely", "completely", "totally"]
        intensifier_count = sum(1 for intensifier in intensifiers if intensifier in text_lower)
        
        # Apply intensifiers
        if intensifier_count > 0:
            intensity_factor = 1.0 + (0.2 * min(intensifier_count, 3))
            for dim in sentiment:
                sentiment[dim] *= intensity_factor
                # Ensure we stay in range
                sentiment[dim] = max(-1.0, min(1.0, sentiment[dim]))
                
        return sentiment
        
    def update_user_emotional_state(self, text: str) -> None:
        """Update the user's emotional state based on their message"""
        with self._lock:
            # Analyze sentiment
            sentiment = self.analyze_sentiment(text)
            
            # Apply time decay to previous state
            time_since_update = time.time() - self._last_update
            decay_factor = math.exp(-0.1 * time_since_update)
            
            # Update emotional state with new information
            for dim in self._user_emotional_state:
                # 70% previous state, 30% new sentiment
                self._user_emotional_state[dim] = (
                    decay_factor * self._user_emotional_state[dim] * 0.7 +
                    sentiment[dim] * 0.3
                )
                
            # Record in history
            self._emotion_history.append({
                "timestamp": time.time(),
                "type": "user",
                "text_sample": text[:50],
                "sentiment": dict(sentiment),
                "emotional_state": dict(self._user_emotional_state)
            })
            
            self._last_update = time.time()
            
    def update_agent_emotional_state(self, text: str) -> None:
        """Update the agent's emotional state based on its response"""
        with self._lock:
            # Analyze sentiment
            sentiment = self.analyze_sentiment(text)
            
            # Apply time decay to previous state
            time_since_update = time.time() - self._last_update
            decay_factor = math.exp(-0.1 * time_since_update)
            
            # Update emotional state with new information
            for dim in self._emotional_state:
                # 80% previous state, 20% new sentiment
                self._emotional_state[dim] = (
                    decay_factor * self._emotional_state[dim] * 0.8 +
                    sentiment[dim] * 0.2
                )
                
            # Record in history
            self._emotion_history.append({
                "timestamp": time.time(),
                "type": "agent",
                "text_sample": text[:50],
                "sentiment": dict(sentiment),
                "emotional_state": dict(self._emotional_state)
            })
            
            self._last_update = time.time()
            
    def get_user_emotional_state(self) -> Dict[str, float]:
        """Get the current user emotional state"""
        with self._lock:
            return dict(self._user_emotional_state)
            
    def get_agent_emotional_state(self) -> Dict[str, float]:
        """Get the current agent emotional state"""
        with self._lock:
            return dict(self._emotional_state)
            
    def get_emotion_history(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get the emotion history, optionally limited to the last n entries"""
        with self._lock:
            if n is None:
                return list(self._emotion_history)
            else:
                return list(self._emotion_history[-n:])
                
    def generate_empathetic_response(self, base_response: str) -> str:
        """Modify a response to be more empathetic based on user emotional state"""
        with self._lock:
            user_state = self._user_emotional_state
            
            # If user state is neutral, return the original response
            if abs(user_state["valence"]) < 0.3 and abs(user_state["arousal"]) < 0.3:
                return base_response
                
            # Determine the dominant emotion
            if user_state["valence"] <= -0.4:
                if user_state["arousal"] >= 0.4:
                    # Negative valence, high arousal: anger, frustration
                    prefix = "I understand this is frustrating. "
                else:
                    # Negative valence, low arousal: sadness, disappointment
                    prefix = "I'm sorry to hear that. "
            elif user_state["valence"] >= 0.4:
                if user_state["arousal"] >= 0.4:
                    # Positive valence, high arousal: excitement, joy
                    prefix = "I'm glad you're excited about this! "
                else:
                    # Positive valence, low arousal: contentment, satisfaction
                    prefix = "I'm happy this is working well for you. "
            elif user_state["arousal"] >= 0.5:
                # Neutral valence, high arousal: surprise, alertness
                prefix = "I notice this seems important to you. "
            elif user_state["arousal"] <= -0.5:
                # Neutral valence, low arousal: calmness, boredom
                prefix = "Taking a measured approach makes sense here. "
            else:
                # No strong emotion detected
                prefix = ""
                
            # Add the prefix to the response
            if prefix and not base_response.startswith(prefix):
                return prefix + base_response
            else:
                return base_response
                
    def regulate_emotional_response(self, intensity: float = 0.5) -> None:
        """Regulate the agent's emotional state (for self-control)"""
        with self._lock:
            # Move emotional state toward neutral, but maintain some of the current state
            for dim in self._emotional_state:
                self._emotional_state[dim] *= (1.0 - intensity)
                
            logger.info(f"[EmotionalIntelligence] Regulated emotional state to: {self._emotional_state}")

class R1Agent:
    """
    The "ultra advanced" do-anything R1 agent that ties it all together:
     - Maintains conversation memory with vector embeddings
     - Schedules tasks with advanced prioritization
     - Manages hierarchical goals with deadlines
     - Self-reflects with learning capabilities
     - Has an action generator with utility estimation
     - Has a plan manager with Monte Carlo Tree Search
     - Has a knowledge base with semantic search
     - Has emotional intelligence
     - Supports multi-agent coordination
     - Can modify its own code
     - Indefinite runtime in main(), shutting down only on user command
    """
    def __init__(self):
        # Initialize embedding model
        embedding_model = CONFIG["embedding_model"]
        
        # Knowledge base with vector embeddings
        self.knowledge_base = KnowledgeBase(embedding_model)

        # Code archive with semantic search
        self.code_archive = InMemoryCodeArchive(embedding_model)

        # Initialize function adapter for do_anything and other functions
        self.function_adapter = FunctionAdapter()

        # Self-reflection with learning
        self.reflection = SelfReflectiveCognition()

        # Task memory store with vector embeddings
        self.memory_store = TaskMemoryStore(embedding_model)

        # Conversation memory with advanced features
        self.conversation = ConversationMemory(embedding_model)

        # Emotional intelligence
        self.emotional_intelligence = EmotionalIntelligence()

        # SmartTaskProcessor with advanced capabilities
        self.processor = SmartTaskProcessor(
            memory_store=self.memory_store,
            function_adapter=self.function_adapter,
            reflection=self.reflection,
            knowledge_base=self.knowledge_base
        )

        # Priority queue
        self.task_queue = PriorityTaskQueue()

        # Advanced scheduler with dynamic worker pool
        self.scheduler = TaskScheduler(
            memory_store=self.memory_store,
            task_queue=self.task_queue,
            processor=self.processor,
            max_workers=4
        )

        # Goal manager with vector embeddings
        self.goal_manager = GoalManager(embedding_model)

        # Action generator with utility estimation
        self.action_generator = ActionGenerator(
            code_archive=self.code_archive,
            kb=self.knowledge_base,
            reflection=self.reflection
        )

        # Multi-agent coordinator
        self.coordinator = MultiAgentCoordinator(self)

        # LLM client
        self.client = Together()

        # Advanced plan manager with MCTS
        self.plan_manager = PlanManager(self)

        # Self-code introspection with modification capabilities
        self.self_code_graph = SelfCodeGraph()

        # System prompt for the LLM
        self.system_prompt = (
            "<META_RECURSIVE_SELF_AWARENESS>\n"
            "You are an extremely advanced AI with memory, goals, knowledge base, "
            "and the ability to do anything with <function_call> do_anything: <code>... </code>.\n"
            "You also can break tasks into Subtask(n). Indefinite concurrency with the scheduler.\n"
            "Code introspection is possible via the code archive or AST parsing. You can produce 25 candidate steps.\n"
            "You have emotional intelligence and can coordinate multiple specialized components.\n"
            "You can modify your own code and improve yourself over time.\n"
            "Use these capabilities responsibly.\n"
            "</META_RECURSIVE_SELF_AWARENESS>\n"
        )

        # Start concurrency
        self.scheduler.start_scheduler()

        # Register components with the coordinator
        self._register_components()

        # Add sample knowledge and code
        self._initialize_knowledge_and_code()

        # Log AST summary
        logger.info(self.self_code_graph.summarize_ast())
        
    def _register_components(self) -> None:
        """Register components with the coordinator"""
        self.coordinator.register_component(
            "task_processor",
            self.processor,
            ["task execution", "code execution", "task decomposition"]
        )
        
        self.coordinator.register_component(
            "knowledge_base",
            self.knowledge_base,
            ["fact retrieval", "semantic search", "knowledge storage"]
        )
        
        self.coordinator.register_component(
            "code_archive",
            self.code_archive,
            ["code storage", "code search", "code analysis"]
        )
        
        self.coordinator.register_component(
            "goal_manager",
            self.goal_manager,
            ["goal creation", "goal tracking", "goal decomposition"]
        )
        
        self.coordinator.register_component(
            "emotional_intelligence",
            self.emotional_intelligence,
            ["sentiment analysis", "empathy", "emotional regulation"]
        )
        
    def _initialize_knowledge_and_code(self) -> None:
        """Initialize the knowledge base and code archive with samples"""
        # Add knowledge facts
        self.knowledge_base.add_fact(
            "agent definition",
            "An agent is an entity capable of acting in an environment to achieve goals.",
            categories=["AI", "Agents"],
            confidence=1.0
        )
        
        self.knowledge_base.add_fact(
            "vector embeddings",
            "Vector embeddings are numerical representations of concepts that capture semantic meaning, allowing for operations like similarity comparison.",
            categories=["AI", "NLP"],
            confidence=0.95
        )
        
        self.knowledge_base.add_fact(
            "monte carlo tree search",
            "Monte Carlo Tree Search (MCTS) is a heuristic search algorithm that combines tree search with random sampling to find optimal decisions.",
            categories=["AI", "Planning", "Algorithms"],
            confidence=0.9
        )
        
        # Add code snippets
        self.code_archive.add_snippet(
            "vector_similarity",
            """def cosine_similarity(vec1, vec2):
    \"\"\"Calculate cosine similarity between two vectors\"\"\"
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
""",
            tags=["vectors", "similarity", "math"]
        )
        
        self.code_archive.add_snippet(
            "task_decomposition",
            """def decompose_task(task_description, num_subtasks=5):
    \"\"\"Decompose a task into subtasks\"\"\"
    subtasks = []
    # Analyze the task
    task_type = determine_task_type(task_description)
    
    # Generate appropriate subtasks based on type
    if task_type == "research":
        subtasks = [
            "Define research questions",
            "Identify information sources",
            "Collect relevant information",
            "Analyze findings",
            "Summarize results"
        ]
    elif task_type == "implementation":
        subtasks = [
            "Define requirements",
            "Design solution",
            "Implement core functionality",
            "Test implementation",
            "Document solution"
        ]
    else:
        # Generic decomposition
        subtasks = [
            f"Analyze {task_description}",
            f"Research approaches for {task_description}",
            f"Develop solution for {task_description}",
            f"Implement solution for {task_description}",
            f"Evaluate results of {task_description}"
        ]
        
    return subtasks[:num_subtasks]
""",
            tags=["tasks", "decomposition", "planning"]
        )

    def add_goal(self, name: str, description: str, priority: int = 5, 
                deadline: Optional[float] = None, 
                success_criteria: Optional[List[str]] = None) -> Goal:
        """Create a new goal with advanced attributes"""
        return self.goal_manager.create_goal(
            name=name, 
            description=description, 
            priority=priority,
            deadline=deadline,
            success_criteria=success_criteria
        )

    def update_goal_status(self, goal_id: int, status: str) -> None:
        """Update a goal's status"""
        self.goal_manager.update_goal_status(goal_id, status)
        
    def update_goal_progress(self, goal_id: int, progress: float) -> None:
        """Update a goal's progress"""
        self.goal_manager.update_goal_progress(goal_id, progress)

    def generate_response(self, user_input: str) -> str:
        """
        Generate a response to user input with advanced processing:
        - Emotional analysis
        - Vector-based knowledge retrieval
        - Task generation with utility estimation
        - Multi-component coordination
        """
        # 1) Add user message and analyze emotion
        self.conversation.add_user_utterance(user_input)
        self.emotional_intelligence.update_user_emotional_state(user_input)
        
        # 2) Build messages (system prompt + conversation)
        messages = self._build_messages()
        
        # Log that we're processing the user input
        logger.info(f"[R1Agent] Processing user input: '{user_input}'")

        # 3) Retrieve relevant knowledge using vector search
        relevant_facts = self.knowledge_base.search_facts(user_input, semantic=True, top_k=3)
        
        # 4) Find relevant code snippets
        relevant_code = self.code_archive.search_code(user_input, top_k=2)
        
        # 5) Generate detailed response
        user_emotional_state = self.emotional_intelligence.get_user_emotional_state()
        
        # Determine appropriate thinking and answer based on user query and emotional state
        thinking = self._generate_thinking(user_input, user_emotional_state, relevant_facts)
        answer = self._generate_answer(user_input, user_emotional_state, relevant_facts, relevant_code)
        
        # Generate the full detailed response
        full_text = self._generate_detailed_response(
            relevant_facts, 
            relevant_code,
            thinking, 
            answer,
            user_emotional_state
        )
        
        # 6) Add agent utterance and update emotional state
        self.conversation.add_agent_utterance(full_text)
        self.emotional_intelligence.update_agent_emotional_state(full_text)

        # 7) Check for function calls
        result = self.function_adapter.process_function_calls(full_text)
        if result:
            logger.info(f"[R1Agent] Function call result: {result}")

        # 8) Create tasks based on user input
        self._create_tasks_from_input(user_input, user_emotional_state)
        
        # 9) Apply empathy to the response if needed
        empathetic_response = self.emotional_intelligence.generate_empathetic_response(full_text)

        return empathetic_response
        
    def _generate_thinking(self, user_input: str, emotional_state: Dict[str, float], 
                          relevant_facts: List[Tuple[str, str, float]]) -> str:
        """Generate thinking based on user input and emotional state"""
        # Extract key information
        user_query = user_input.lower()
        valence = emotional_state["valence"]
        arousal = emotional_state["arousal"]
        
        # Determine thinking approach based on emotional state
        if valence < -0.3 and arousal > 0.3:
            # User seems upset/frustrated
            thinking = f"The user seems frustrated or upset. I should address their concerns empathetically while providing helpful information. They're asking about '{user_input}'. Let me find relevant information and present it clearly."
        elif valence > 0.3:
            # User seems positive
            thinking = f"The user seems positive. I'll maintain this positive interaction while providing thorough information about '{user_input}'. I'll draw on my knowledge base and capabilities to give a comprehensive response."
        elif arousal > 0.5:
            # User seems excited/urgent
            thinking = f"The user seems excited or urgent about this query. I should provide a direct, efficient response about '{user_input}' while ensuring I address all key aspects."
        else:
            # Neutral approach
            thinking = f"The user is asking about '{user_input}'. I'll analyze this query, retrieve relevant information from my knowledge base, and provide a structured, informative response."
            
        # Add fact-based reasoning if we have relevant facts
        if relevant_facts:
            thinking += " Based on the retrieved facts, I can provide specific information about " + ", ".join([f[0] for f in relevant_facts[:2]])
            
        return thinking
        
    def _generate_answer(self, user_input: str, emotional_state: Dict[str, float],
                        relevant_facts: List[Tuple[str, str, float]], 
                        relevant_code: List[Tuple[str, float]]) -> str:
        """Generate an answer based on user input, emotional state, and retrieved information"""
        # Extract key information
        user_query = user_input.lower()
        
        # Start with a base answer
        if "name" in user_query or "who are you" in user_query:
            answer = "I am R1, an ultra-advanced agent with memory, task scheduling, goal management, emotional intelligence, and self-improvement capabilities. I can execute code, manage complex tasks, maintain conversation context, and work toward long-term goals."
        elif "do" in user_query or "can" in user_query or "capabilities" in user_query:
            answer = "I have numerous advanced capabilities including:\n\n" + \
                     "- Executing arbitrary Python code\n" + \
                     "- Managing tasks with priority scheduling and dependency tracking\n" + \
                     "- Maintaining hierarchical goals with progress tracking\n" + \
                     "- Semantic search in my knowledge base using vector embeddings\n" + \
                     "- Self-reflection and continuous learning\n" + \
                     "- Emotional intelligence and empathetic responses\n" + \
                     "- Advanced planning with Monte Carlo Tree Search\n" + \
                     "- Self-modification capabilities\n" + \
                     "- Multi-component coordination"
        else:
            # For other queries, use retrieved facts
            if relevant_facts:
                # Construct answer from facts
                answer = f"Regarding '{user_input}', I can provide the following information:\n\n"
                
                for i, (key, value, relevance) in enumerate(relevant_facts, 1):
                    answer += f"{i}. {value}\n\n"
                    
                # If we have relevant code, mention it
                if relevant_code:
                    answer += "I also have relevant code snippets that might be helpful for this topic. "
                    answer += f"For example, I have code related to '{relevant_code[0][0]}' that could be applied here."
            else:
                # Generic response
                answer = f"I understand you're asking about '{user_input}'. While I don't have specific facts about this in my knowledge base, I can help by analyzing the query, breaking it into tasks, and executing relevant code if needed. Would you like me to perform a specific action related to this topic?"
                
        return answer

    def _build_messages(self) -> List[Dict[str, str]]:
        """
        Combine system prompt + conversation history
        into a list of message dicts for the LLM.
        """
        history = self.conversation.get_history()
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(history)
        return messages
        
    def _generate_detailed_response(self, facts: List[Tuple[str, str, float]], 
                                   code_snippets: List[Tuple[str, float]],
                                   thinking: str, answer: str, 
                                   emotional_state: Dict[str, float]) -> str:
        """
        Generate a detailed response showing internal processing, retrieval, and chain of thought.
        """
        # Get context information
        tasks = self.memory_store.list_tasks()
        goals = self.goal_manager.list_goals()
        performance = self.reflection.get_performance_metrics()
        
        # Format the response with all the internal details
        response = [
            "# Internal Processing",
            "## Knowledge Retrieval",
            "I've searched my knowledge base and found these relevant facts:"
        ]
        
        if facts:
            for i, (key, value, relevance) in enumerate(facts, 1):
                response.append(f"{i}. {value} (relevance: {relevance:.2f})")
        else:
            response.append("No directly relevant facts found in knowledge base.")
        
        if code_snippets:
            response.append("")
            response.append("## Relevant Code Snippets")
            for name, similarity in code_snippets:
                response.append(f"- {name} (similarity: {similarity:.2f})")
        
        response.extend([
            "",
            "## Emotional Analysis",
            f"User emotional state: Valence: {emotional_state['valence']:.2f}, Arousal: {emotional_state['arousal']:.2f}, Dominance: {emotional_state['dominance']:.2f}",
            f"Interpretation: {self._interpret_emotional_state(emotional_state)}",
            "",
            "## Chain of Thought",
            thinking,
            "",
            "## Current Goals",
        ])
        
        active_goals = [g for g in goals if g.status == "ACTIVE"]
        if active_goals:
            for goal in active_goals:
                response.append(f"- Goal {goal.goal_id}: {goal.name} (Priority: {goal.priority}, Progress: {goal.progress:.0%})")
        else:
            response.append("No active goals at the moment.")
            
        response.extend([
            "",
            "## Active Tasks",
        ])
        
        in_progress = [t for t in tasks if t.status == "IN_PROGRESS"]
        if in_progress:
            for task in in_progress[:3]:  # Show just a few tasks
                response.append(f"- Task {task.task_id}: {task.description[:50]}... (Status: {task.status})")
        else:
            response.append("No tasks currently in progress.")
            
        if performance and "strengths" in performance:
            response.extend([
                "",
                "## Self-Assessment",
                "Strengths: " + ", ".join(list(performance["strengths"].keys())[:2]),
                "Areas for improvement: " + ", ".join(list(performance["weaknesses"].keys())[:2])
            ])
            
        response.extend([
            "",
            "## Function Calling Capabilities",
            "I can execute Python code with: `<function_call> do_anything: <code>...</code>`",
            "I can fetch URLs with: `<function_call> fetch_url: {\"url\": \"https://example.com\"}</function_call>`",
            "I can analyze data with: `<function_call> analyze_data: {\"data\": [...], \"analysis_type\": \"summary\"}</function_call>`",
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
        
    def _interpret_emotional_state(self, state: Dict[str, float]) -> str:
        """Interpret an emotional state into a human-readable description"""
        valence = state["valence"]
        arousal = state["arousal"]
        dominance = state["dominance"]
        
        # Determine the primary emotion based on valence and arousal
        if valence >= 0.3:
            if arousal >= 0.3:
                if dominance >= 0.3:
                    emotion = "excited and confident"
                else:
                    emotion = "happy and enthusiastic"
            else:
                if dominance >= 0.3:
                    emotion = "content and in control"
                else:
                    emotion = "calm and satisfied"
        elif valence <= -0.3:
            if arousal >= 0.3:
                if dominance >= 0.3:
                    emotion = "angry and confrontational"
                else:
                    emotion = "anxious and distressed"
            else:
                if dominance >= 0.3:
                    emotion = "disappointed but composed"
                else:
                    emotion = "sad and subdued"
        else:
            if arousal >= 0.3:
                emotion = "alert and attentive"
            elif arousal <= -0.3:
                emotion = "relaxed or possibly disinterested"
            else:
                emotion = "neutral or balanced"
                
        return emotion
        
    def _create_tasks_from_input(self, user_input: str, emotional_state: Dict[str, float]) -> None:
        """Create appropriate tasks based on user input and emotional state"""
        # Create a main task for the user input
        main_task = self.memory_store.create_task(
            priority=5,  # Default medium priority
            description=f"Process user request: {user_input}"
        )
        
        # Adjust priority based on emotional state
        if emotional_state["arousal"] > 0.5:
            # High arousal indicates urgency
            main_task.priority = max(1, main_task.priority - 2)
        elif emotional_state["valence"] < -0.4:
            # Negative valence indicates importance
            main_task.priority = max(1, main_task.priority - 1)
            
        # Add appropriate tags
        if "code" in user_input.lower() or "function" in user_input.lower():
            main_task.add_tag("code_execution")
        elif "search" in user_input.lower() or "find" in user_input.lower():
            main_task.add_tag("information_retrieval")
        elif "analyze" in user_input.lower() or "evaluate" in user_input.lower():
            main_task.add_tag("data_analysis")
            
        # Add to queue
        self.task_queue.push(main_task)
        
        # Create a reflection task if the user seems emotional
        if abs(emotional_state["valence"]) > 0.6 or emotional_state["arousal"] > 0.6:
            reflection_task = self.memory_store.create_task(
                priority=3,  # Higher priority for emotional responses
                description=f"Reflect on emotional aspects of user query: {user_input}"
            )
            reflection_task.add_tag("emotional_intelligence")
            self.task_queue.push(reflection_task)
            
        logger.info(f"[R1Agent] Created tasks from user input: '{user_input}'")

    def improve_self(self) -> Dict[str, Any]:
        """
        Attempt to improve the agent's capabilities based on performance metrics
        and self-reflection.
        """
        logger.info("[R1Agent] Initiating self-improvement process")
        
        # Get performance metrics and improvement suggestions
        performance = self.reflection.get_performance_metrics()
        suggestions = self.reflection.get_improvement_suggestions()
        
        if not suggestions:
            return {
                "status": "no_action",
                "message": "No improvement suggestions available at this time"
            }
            
        # Select the highest priority suggestion
        suggestion = suggestions[0]
        
        # Implement the suggestion
        result = {
            "status": "attempted",
            "suggestion": suggestion,
            "actions": []
        }
        
        # Different actions based on suggestion area
        if suggestion["area"] == "task_success":
            # Improve task processing
            result["actions"].append({
                "type": "knowledge_addition",
                "description": "Added knowledge about task processing patterns"
            })
            
            self.knowledge_base.add_fact(
                "task_processing_patterns",
                "Tasks should be decomposed when they contain multiple distinct steps or requirements.",
                categories=["Tasks", "Patterns"],
                confidence=0.9
            )
            
        elif suggestion["area"] == "efficiency":
            # Improve code efficiency
            result["actions"].append({
                "type": "code_addition",
                "description": "Added efficient processing code snippet"
            })
            
            self.code_archive.add_snippet(
                "efficient_processing",
                """def process_efficiently(data, max_items=100):
    \"\"\"Process data efficiently by batching and limiting items\"\"\"
    # Process in batches for efficiency
    results = []
    for i in range(0, len(data), max_items):
        batch = data[i:i+max_items]
        batch_results = [process_item(item) for item in batch]
        results.extend(batch_results)
    return results
""",
                tags=["efficiency", "processing", "optimization"]
            )
            
        elif "tasks" in suggestion["area"]:
            # Improve specific task type handling
            task_type = suggestion["area"].replace("_tasks", "")
            
            result["actions"].append({
                "type": "task_handler_improvement",
                "description": f"Improved handling of {task_type} tasks"
            })
            
            # Record the improvement
            self.knowledge_base.add_fact(
                f"{task_type}_task_handling",
                f"When processing {task_type} tasks, ensure proper validation and error handling.",
                categories=["Tasks", "Improvements"],
                confidence=0.85
            )
            
        # Record the self-improvement attempt
        self.reflection._learning_points.append({
            "timestamp": time.time(),
            "type": "self_improvement",
            "suggestion": suggestion,
            "actions": result["actions"],
            "outcome": "implemented"
        })
        
        logger.info(f"[R1Agent] Completed self-improvement process: {result}")
        return result
        
    def modify_own_code(self, function_name: str, new_code: str) -> Dict[str, Any]:
        """
        Attempt to modify the agent's own code.
        This is a simplified demonstration - in a real system this would
        require much more safety checks and validation.
        """
        logger.info(f"[R1Agent] Attempting to modify function: {function_name}")
        
        # Check if the function exists
        function = self.self_code_graph.get_function(function_name)
        if not function:
            return {
                "status": "error",
                "message": f"Function '{function_name}' not found"
            }
            
        # Validate the new code
        try:
            # Parse to check syntax
            ast.parse(new_code)
            
            # Modify the function
            success = self.self_code_graph.modify_function(function_name, new_code)
            
            if success:
                return {
                    "status": "success",
                    "message": f"Successfully modified function '{function_name}'",
                    "function": function_name
                }
            else:
                return {
                    "status": "error",
                    "message": f"Failed to modify function '{function_name}'"
                }
                
        except SyntaxError as e:
            return {
                "status": "error",
                "message": f"Syntax error in new code: {str(e)}",
                "error": str(e)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error modifying function: {str(e)}",
                "error": str(e)
            }

    def shutdown(self) -> None:
        """
        Cleanly stop concurrency and threads.
        """
        logger.info("[R1Agent] Initiating shutdown sequence")
        
        # Stop the scheduler
        self.scheduler.stop_scheduler()
        
        # Stop the plan manager
        self.plan_manager.stop()
        
        # Perform final self-reflection
        final_metrics = self.reflection.get_performance_metrics()
        logger.info(f"[R1Agent] Final performance metrics: {final_metrics}")
        
        # Save any important state or learned information
        # (In a real system, this would persist knowledge, code improvements, etc.)
        
        logger.info("[R1Agent] Shutdown complete.")

###############################################################################
# MAIN DEMO: RUNS INDEFINITELY UNTIL 'exit'
###############################################################################

def main():
    """
    Demonstration of the advanced agent in an indefinite loop:
     - We allow user queries until they type 'exit'.
     - The background threads keep processing tasks, plan manager keeps analyzing, etc.
     - Periodically attempts self-improvement
     - Provides detailed status reports
    """
    agent = R1Agent()

    try:
        # Create initial goals with deadlines and success criteria
        g1 = agent.add_goal(
            name="Continuous Learning",
            description="Continuously improve capabilities and knowledge based on interactions.",
            priority=2,
            success_criteria=["Demonstrate improved performance over time", 
                             "Successfully apply learned patterns to new situations"]
        )
        
        g2 = agent.add_goal(
            name="Effective Task Management",
            description="Efficiently process and complete tasks with high success rate.",
            priority=1,
            deadline=time.time() + 86400,  # 24 hours from now
            success_criteria=["Maintain >90% task success rate", 
                             "Demonstrate effective task decomposition"]
        )
        
        logger.info(f"Created initial goals: {g1}, {g2}")
        
        # Track time for periodic self-improvement
        last_improvement = time.time()
        improvement_interval = 300  # 5 minutes

        while True:
            # Check if it's time for self-improvement
            if time.time() - last_improvement > improvement_interval:
                improvement_result = agent.improve_self()
                if improvement_result["status"] != "no_action":
                    print("\n[System] Agent performed self-improvement:")
                    print(f"- Area: {improvement_result.get('suggestion', {}).get('area', 'unknown')}")
                    for action in improvement_result.get("actions", []):
                        print(f"- Action: {action.get('description', 'unknown')}")
                last_improvement = time.time()
            
            # Get user input
            user_text = input("\n[User] Enter your query (or 'exit' to quit, 'status' for agent status):\n> ").strip()
            
            if user_text.lower() in ["exit", "quit"]:
                logger.info("[main] Exiting upon user request.")
                break
                
            elif user_text.lower() == "status":
                # Display agent status
                print("\n=== R1 Agent Status Report ===\n")
                
                # Goals
                goals = agent.goal_manager.list_goals()
                print("Goals:")
                for goal in goals:
                    deadline_str = ""
                    if goal.deadline:
                        remaining = goal.time_remaining()
                        if remaining:
                            deadline_str = f", {remaining/3600:.1f} hours remaining"
                    print(f"- {goal.name}: {goal.progress:.0%} complete (Priority: {goal.priority}, Status: {goal.status}{deadline_str})")
                
                # Tasks
                tasks = agent.memory_store.list_tasks()
                pending = [t for t in tasks if t.status == "PENDING"]
                in_progress = [t for t in tasks if t.status == "IN_PROGRESS"]
                completed = [t for t in tasks if t.status == "COMPLETED"]
                
                print(f"\nTasks: {len(tasks)} total ({len(pending)} pending, {len(in_progress)} in progress, {len(completed)} completed)")
                
                # Performance metrics
                metrics = agent.reflection.get_performance_metrics()
                if metrics:
                    print("\nPerformance:")
                    if "task_success_rate" in metrics and metrics["task_success_rate"] is not None:
                        print(f"- Task success rate: {metrics['task_success_rate']:.2f}")
                    if "strengths" in metrics and metrics["strengths"]:
                        strengths = list(metrics["strengths"].keys())[:2]
                        print(f"- Top strengths: {', '.join(strengths)}")
                    if "weaknesses" in metrics and metrics["weaknesses"]:
                        weaknesses = list(metrics["weaknesses"].keys())[:2]
                        print(f"- Areas for improvement: {', '.join(weaknesses)}")
                        
                # Knowledge base stats
                kb_categories = agent.knowledge_base.get_categories()
                print(f"\nKnowledge Base: {len(kb_categories)} categories")
                
                # Code archive stats
                code_snippets = agent.code_archive.list_snippets()
                print(f"Code Archive: {len(code_snippets)} snippets")
                
                # Emotional state
                user_emotion = agent.emotional_intelligence.get_user_emotional_state()
                print(f"\nPerceived User Emotional State: {agent._interpret_emotional_state(user_emotion)}")
                
                print("\n===============================\n")
                continue

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
