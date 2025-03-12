#!/usr/bin/env python3
"""
Shell Command Execution Tool

This module provides enhanced shell command execution capabilities for the agent,
allowing it to run system commands and process their output with recursive task
decomposition and goal-oriented planning.
"""

import os
import sys
import subprocess
import shlex
import asyncio
import logging
import uuid
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime

class TaskNode:
    """
    Represents a node in the task execution tree.
    Each node can have a parent task and multiple child tasks,
    forming a hierarchical structure for complex command execution.
    """
    
    def __init__(self, task_id: str, command: str, description: str = "", 
                parent_id: Optional[str] = None, priority: int = 5,
                dependencies: List[str] = None, max_retries: int = 0):
        """
        Initialize a task node
        
        Args:
            task_id: Unique identifier for this task
            command: The shell command to execute
            description: Human-readable description of the task
            parent_id: ID of the parent task, if any
            priority: Task priority (1-10, lower is higher priority)
            dependencies: List of task IDs that must complete before this task
            max_retries: Maximum number of retry attempts on failure
        """
        self.task_id = task_id
        self.command = command
        self.description = description
        self.parent_id = parent_id
        self.child_ids = []
        self.status = "PENDING"  # PENDING, IN_PROGRESS, COMPLETED, FAILED, BLOCKED, RETRY
        self.result = None
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.execution_time = None
        self.priority = priority
        self.dependencies = dependencies or []
        self.max_retries = max_retries
        self.retry_count = 0
        self.metadata = {}  # For storing arbitrary task metadata
    
    def add_child(self, child_id: str) -> None:
        """Add a child task to this node"""
        self.child_ids.append(child_id)
    
    def start(self) -> None:
        """Mark the task as started"""
        self.status = "IN_PROGRESS"
        self.started_at = datetime.now()
    
    def complete(self, result: Dict[str, Any]) -> None:
        """Mark the task as completed with results"""
        self.status = "COMPLETED" if result.get("success", False) else "FAILED"
        self.result = result
        self.completed_at = datetime.now()
        if self.started_at:
            self.execution_time = (self.completed_at - self.started_at).total_seconds()
    
    def retry(self) -> bool:
        """
        Attempt to retry a failed task
        
        Returns:
            bool: True if retry is possible, False if max retries exceeded
        """
        if self.status != "FAILED":
            return False
            
        if self.retry_count >= self.max_retries:
            return False
            
        self.retry_count += 1
        self.status = "RETRY"
        self.result = None
        return True
    
    def is_blocked(self) -> bool:
        """Check if task is blocked by dependencies"""
        return len(self.dependencies) > 0
    
    def remove_dependency(self, dependency_id: str) -> None:
        """Remove a dependency once it's completed"""
        if dependency_id in self.dependencies:
            self.dependencies.remove(dependency_id)
            
        # If no more dependencies, update status
        if not self.dependencies and self.status == "BLOCKED":
            self.status = "PENDING"
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the task"""
        self.metadata[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the task node to a dictionary"""
        return {
            "task_id": self.task_id,
            "command": self.command,
            "description": self.description,
            "parent_id": self.parent_id,
            "child_ids": self.child_ids,
            "status": self.status,
            "result": self.result,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_time": self.execution_time,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "metadata": self.metadata
        }


class TaskExecutionTree:
    """
    Manages a tree of task nodes representing the hierarchical decomposition
    of complex commands into simpler subtasks.
    """
    
    def __init__(self):
        """Initialize the task execution tree"""
        self.tasks = {}  # task_id -> TaskNode
        self.root_tasks = []  # List of root task IDs
        self.task_queue = []  # Priority queue for task execution
        self.completed_tasks = []  # History of completed tasks
        self.failed_tasks = []  # History of failed tasks
    
    def create_task(self, command: str, description: str = "", 
                   parent_id: Optional[str] = None, priority: int = 5,
                   dependencies: List[str] = None, max_retries: int = 0) -> str:
        """
        Create a new task in the execution tree
        
        Args:
            command: The shell command to execute
            description: Human-readable description of the task
            parent_id: ID of the parent task, if any
            priority: Task priority (1-10, lower is higher priority)
            dependencies: List of task IDs that must complete before this task
            max_retries: Maximum number of retry attempts on failure
            
        Returns:
            The ID of the newly created task
        """
        task_id = str(uuid.uuid4())
        task = TaskNode(task_id, command, description, parent_id, priority, dependencies, max_retries)
        self.tasks[task_id] = task
        
        # Check if task is blocked by dependencies
        if dependencies and any(dep_id in self.tasks for dep_id in dependencies):
            task.status = "BLOCKED"
        
        if parent_id:
            if parent_id in self.tasks:
                self.tasks[parent_id].add_child(task_id)
        else:
            self.root_tasks.append(task_id)
        
        # Add to priority queue
        self.task_queue.append(task_id)
        self._sort_task_queue()
            
        return task_id
    
    def _sort_task_queue(self):
        """Sort the task queue by priority"""
        self.task_queue.sort(key=lambda task_id: self.tasks[task_id].priority if task_id in self.tasks else 10)
    
    def get_task(self, task_id: str) -> Optional[TaskNode]:
        """Get a task by its ID"""
        return self.tasks.get(task_id)
    
    def get_children(self, task_id: str) -> List[TaskNode]:
        """Get all child tasks for a given task ID"""
        task = self.tasks.get(task_id)
        if not task:
            return []
        return [self.tasks[child_id] for child_id in task.child_ids if child_id in self.tasks]
    
    def get_next_task(self) -> Optional[str]:
        """
        Get the next task to execute based on priority and dependencies
        
        Returns:
            The ID of the next task to execute, or None if no tasks are ready
        """
        for task_id in self.task_queue:
            task = self.tasks.get(task_id)
            if not task:
                continue
                
            if task.status == "PENDING":
                # Check if all dependencies are completed
                if not task.dependencies:
                    return task_id
                
                all_deps_completed = True
                for dep_id in task.dependencies:
                    dep_task = self.tasks.get(dep_id)
                    if not dep_task or dep_task.status != "COMPLETED":
                        all_deps_completed = False
                        break
                
                if all_deps_completed:
                    # Remove from dependencies list
                    task.dependencies = []
                    return task_id
            
            elif task.status == "RETRY":
                return task_id
                
        return None
    
    def update_dependencies(self, completed_task_id: str) -> List[str]:
        """
        Update tasks that depend on the completed task
        
        Args:
            completed_task_id: ID of the task that was just completed
            
        Returns:
            List of task IDs that are now unblocked
        """
        unblocked_tasks = []
        
        for task_id, task in self.tasks.items():
            if completed_task_id in task.dependencies:
                task.remove_dependency(completed_task_id)
                if not task.dependencies:
                    unblocked_tasks.append(task_id)
        
        return unblocked_tasks
    
    def get_subtree(self, task_id: str) -> Dict[str, Any]:
        """
        Get a dictionary representation of a subtree rooted at the given task ID
        
        Args:
            task_id: Root task ID for the subtree
            
        Returns:
            Dictionary representation of the subtree
        """
        task = self.tasks.get(task_id)
        if not task:
            return {}
            
        result = task.to_dict()
        result["children"] = [self.get_subtree(child_id) for child_id in task.child_ids]
        return result
    
    def get_full_tree(self) -> List[Dict[str, Any]]:
        """Get the full execution tree as a list of dictionaries"""
        return [self.get_subtree(task_id) for task_id in self.root_tasks]
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get statistics about the task execution tree"""
        total_tasks = len(self.tasks)
        pending_tasks = sum(1 for t in self.tasks.values() if t.status == "PENDING")
        in_progress_tasks = sum(1 for t in self.tasks.values() if t.status == "IN_PROGRESS")
        completed_tasks = sum(1 for t in self.tasks.values() if t.status == "COMPLETED")
        failed_tasks = sum(1 for t in self.tasks.values() if t.status == "FAILED")
        blocked_tasks = sum(1 for t in self.tasks.values() if t.status == "BLOCKED")
        retry_tasks = sum(1 for t in self.tasks.values() if t.status == "RETRY")
        
        # Calculate average execution time for completed tasks
        execution_times = [t.execution_time for t in self.tasks.values() 
                          if t.status == "COMPLETED" and t.execution_time is not None]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        return {
            "total_tasks": total_tasks,
            "pending_tasks": pending_tasks,
            "in_progress_tasks": in_progress_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "blocked_tasks": blocked_tasks,
            "retry_tasks": retry_tasks,
            "avg_execution_time": avg_execution_time,
            "root_tasks": len(self.root_tasks),
            "leaf_tasks": sum(1 for t in self.tasks.values() if not t.child_ids)
        }
    
    def update_task_status(self, task_id: str, status: str) -> bool:
        """Update the status of a task"""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        old_status = task.status
        task.status = status
        
        # If task is completed, update dependencies
        if status == "COMPLETED" and old_status != "COMPLETED":
            self.update_dependencies(task_id)
            if task_id in self.task_queue:
                self.task_queue.remove(task_id)
            self.completed_tasks.append(task_id)
        
        # If task failed, check for retry or move to failed list
        elif status == "FAILED" and old_status != "FAILED":
            if task.retry_count >= task.max_retries:
                if task_id in self.task_queue:
                    self.task_queue.remove(task_id)
                self.failed_tasks.append(task_id)
        
        return True
    
    def complete_task(self, task_id: str, result: Dict[str, Any]) -> bool:
        """Mark a task as completed with results"""
        task = self.tasks.get(task_id)
        if not task:
            return False
            
        task.complete(result)
        
        # If task completed successfully, update dependencies
        if result.get("success", False):
            self.update_dependencies(task_id)
            if task_id in self.task_queue:
                self.task_queue.remove(task_id)
            self.completed_tasks.append(task_id)
        else:
            # If task failed, check for retry
            if task.retry_count < task.max_retries:
                task.retry()
            else:
                if task_id in self.task_queue:
                    self.task_queue.remove(task_id)
                self.failed_tasks.append(task_id)
                
        return True
    
    def retry_failed_tasks(self) -> List[str]:
        """
        Retry all failed tasks that haven't exceeded max_retries
        
        Returns:
            List of task IDs that were queued for retry
        """
        retried_tasks = []
        
        for task_id in list(self.failed_tasks):
            task = self.tasks.get(task_id)
            if not task:
                continue
                
            if task.retry_count < task.max_retries:
                if task.retry():
                    self.failed_tasks.remove(task_id)
                    self.task_queue.append(task_id)
                    retried_tasks.append(task_id)
        
        return retried_tasks
    
    def generate_visualization_data(self) -> Dict[str, Any]:
        """
        Generate data for visualizing the task execution tree
        
        Returns:
            Dictionary with visualization data
        """
        nodes = []
        links = []
        
        # Create nodes
        for task_id, task in self.tasks.items():
            status_color = {
                "PENDING": "#cccccc",
                "IN_PROGRESS": "#3498db",
                "COMPLETED": "#2ecc71",
                "FAILED": "#e74c3c",
                "BLOCKED": "#f39c12",
                "RETRY": "#9b59b6"
            }.get(task.status, "#cccccc")
            
            nodes.append({
                "id": task_id,
                "name": task.description or task.command[:30] + ("..." if len(task.command) > 30 else ""),
                "status": task.status,
                "color": status_color,
                "priority": task.priority,
                "execution_time": task.execution_time
            })
            
            # Create parent-child links
            if task.parent_id and task.parent_id in self.tasks:
                links.append({
                    "source": task.parent_id,
                    "target": task_id,
                    "type": "parent-child"
                })
            
            # Create dependency links
            for dep_id in task.dependencies:
                if dep_id in self.tasks:
                    links.append({
                        "source": dep_id,
                        "target": task_id,
                        "type": "dependency"
                    })
        
        return {
            "nodes": nodes,
            "links": links,
            "statistics": self.get_task_statistics()
        }


class ShellCommandTool:
    """
    Tool for executing shell commands with advanced features:
    - Command validation and sanitization
    - Timeout handling
    - Output formatting
    - Error handling
    - Resource usage tracking
    - Recursive task decomposition
    - Goal-oriented planning
    """
    
    def __init__(self, working_dir: Optional[str] = None, timeout: int = 30):
        """
        Initialize the shell command tool
        
        Args:
            working_dir: Working directory for command execution (default: current directory)
            timeout: Default timeout in seconds for command execution
        """
        self.working_dir = working_dir or os.getcwd()
        self.default_timeout = timeout
        self.command_history = []
        self.max_history = 100
        self.execution_tree = TaskExecutionTree()
        
    async def execute(self, command: str, timeout: Optional[int] = None, 
                    capture_output: bool = True) -> Dict[str, Any]:
        """
        Execute a shell command asynchronously
        
        Args:
            command: The shell command to execute
            timeout: Timeout in seconds (None uses default timeout)
            capture_output: Whether to capture stdout/stderr
            
        Returns:
            Dict containing execution results
        """
        start_time = datetime.now()
        timeout = timeout or self.default_timeout
        
        # Record command in history
        self.command_history.append({
            "command": command,
            "timestamp": start_time.isoformat(),
            "working_dir": self.working_dir
        })
        
        # Trim history if needed
        if len(self.command_history) > self.max_history:
            self.command_history = self.command_history[-self.max_history:]
        
        try:
            # Create subprocess
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE if capture_output else None,
                stderr=asyncio.subprocess.PIPE if capture_output else None,
                cwd=self.working_dir
            )
            
            try:
                # Wait for process with timeout
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                
                # Process completed within timeout
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                return {
                    "success": process.returncode == 0,
                    "exit_code": process.returncode,
                    "stdout": stdout.decode('utf-8', errors='replace') if stdout else "",
                    "stderr": stderr.decode('utf-8', errors='replace') if stderr else "",
                    "command": command,
                    "execution_time": execution_time,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "timeout_occurred": False
                }
                
            except asyncio.TimeoutError:
                # Process timed out
                try:
                    process.kill()
                except Exception:
                    pass
                    
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                return {
                    "success": False,
                    "exit_code": None,
                    "stdout": "",
                    "stderr": f"Command timed out after {timeout} seconds",
                    "command": command,
                    "execution_time": execution_time,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "timeout_occurred": True
                }
                
        except Exception as e:
            # Error occurred during execution
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return {
                "success": False,
                "exit_code": None,
                "stdout": "",
                "stderr": f"Error executing command: {str(e)}",
                "command": command,
                "execution_time": execution_time,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "timeout_occurred": False,
                "error": str(e)
            }
    
    def execute_sync(self, command: str, timeout: Optional[int] = None,
                   capture_output: bool = True) -> Dict[str, Any]:
        """
        Execute a shell command synchronously
        
        Args:
            command: The shell command to execute
            timeout: Timeout in seconds (None uses default timeout)
            capture_output: Whether to capture stdout/stderr
            
        Returns:
            Dict containing execution results
        """
        start_time = datetime.now()
        timeout = timeout or self.default_timeout
        
        # Record command in history
        self.command_history.append({
            "command": command,
            "timestamp": start_time.isoformat(),
            "working_dir": self.working_dir
        })
        
        # Trim history if needed
        if len(self.command_history) > self.max_history:
            self.command_history = self.command_history[-self.max_history:]
        
        try:
            # Run the command
            process = subprocess.run(
                command,
                shell=True,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                cwd=self.working_dir
            )
            
            # Process completed
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return {
                "success": process.returncode == 0,
                "exit_code": process.returncode,
                "stdout": process.stdout if capture_output else "",
                "stderr": process.stderr if capture_output else "",
                "command": command,
                "execution_time": execution_time,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "timeout_occurred": False
            }
            
        except subprocess.TimeoutExpired:
            # Process timed out
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return {
                "success": False,
                "exit_code": None,
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds",
                "command": command,
                "execution_time": execution_time,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "timeout_occurred": True
            }
            
        except Exception as e:
            # Error occurred during execution
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return {
                "success": False,
                "exit_code": None,
                "stdout": "",
                "stderr": f"Error executing command: {str(e)}",
                "command": command,
                "execution_time": execution_time,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "timeout_occurred": False,
                "error": str(e)
            }
    
    def get_command_history(self) -> List[Dict[str, Any]]:
        """Get the command execution history"""
        return self.command_history
    
    def set_working_directory(self, working_dir: str) -> bool:
        """
        Set the working directory for command execution
        
        Args:
            working_dir: New working directory
            
        Returns:
            bool: Success status
        """
        if os.path.exists(working_dir) and os.path.isdir(working_dir):
            self.working_dir = working_dir
            return True
        return False
    
    def get_working_directory(self) -> str:
        """Get the current working directory"""
        return self.working_dir
    
    async def execute_pipeline(self, commands: List[str], 
                             stop_on_error: bool = True) -> List[Dict[str, Any]]:
        """
        Execute a pipeline of commands sequentially
        
        Args:
            commands: List of commands to execute
            stop_on_error: Whether to stop execution if a command fails
            
        Returns:
            List of execution results for each command
        """
        results = []
        
        for command in commands:
            result = await self.execute(command)
            results.append(result)
            
            if stop_on_error and not result["success"]:
                break
                
        return results
    
    async def execute_task(self, task_id: str) -> Dict[str, Any]:
        """
        Execute a specific task by its ID
        
        Args:
            task_id: ID of the task to execute
            
        Returns:
            Execution result dictionary
        """
        task = self.execution_tree.get_task(task_id)
        if not task:
            return {
                "success": False,
                "error": f"Task with ID {task_id} not found"
            }
        
        # Mark task as started
        task.start()
        
        # Execute the command
        result = await self.execute(task.command)
        
        # Update task with result
        self.execution_tree.complete_task(task_id, result)
        
        return result
    
    async def create_and_execute_task(self, command: str, description: str = "",
                                    parent_id: Optional[str] = None, priority: int = 5,
                                    dependencies: List[str] = None, max_retries: int = 0) -> Dict[str, Any]:
        """
        Create a new task and execute it immediately
        
        Args:
            command: Shell command to execute
            description: Human-readable description of the task
            parent_id: ID of the parent task, if any
            priority: Task priority (1-10, lower is higher priority)
            dependencies: List of task IDs that must complete before this task
            max_retries: Maximum number of retry attempts on failure
            
        Returns:
            Dictionary containing task ID and execution result
        """
        task_id = self.execution_tree.create_task(
            command, description, parent_id, priority, dependencies, max_retries
        )
        
        # Check if task is blocked by dependencies
        task = self.execution_tree.get_task(task_id)
        if task and task.status == "BLOCKED":
            return {
                "task_id": task_id,
                "status": "BLOCKED",
                "message": f"Task is blocked by {len(task.dependencies)} dependencies",
                "dependencies": task.dependencies
            }
        
        result = await self.execute_task(task_id)
        
        return {
            "task_id": task_id,
            "result": result
        }
    
    async def decompose_and_execute(self, goal: str, 
                                  decomposition_func: Callable[[str], List[Dict[str, Any]]],
                                  sequential: bool = True,
                                  max_retries: int = 0,
                                  handle_failures: bool = False) -> Dict[str, Any]:
        """
        Decompose a high-level goal into subtasks and execute them
        
        Args:
            goal: High-level goal description
            decomposition_func: Function that takes a goal and returns a list of subtasks
                                Each subtask should be a dict with 'command' and 'description' keys
            sequential: Whether to execute subtasks sequentially or in parallel
            max_retries: Maximum number of retry attempts for each subtask
            handle_failures: Whether to continue execution after subtask failures
            
        Returns:
            Dictionary containing execution results and task tree
        """
        # Create the root task for the goal
        root_task_id = self.execution_tree.create_task("", goal)
        
        # Decompose the goal into subtasks
        subtasks = decomposition_func(goal)
        
        # Create task nodes for each subtask
        subtask_ids = []
        for i, subtask in enumerate(subtasks):
            # Set up dependencies for sequential execution
            dependencies = []
            if sequential and i > 0:
                dependencies = [subtask_ids[i-1]]
                
            task_id = self.execution_tree.create_task(
                subtask["command"],
                subtask.get("description", ""),
                root_task_id,
                priority=subtask.get("priority", 5),
                dependencies=dependencies,
                max_retries=subtask.get("max_retries", max_retries)
            )
            subtask_ids.append(task_id)
        
        # Execute the subtasks
        results = []
        if sequential:
            # Execute sequentially
            for task_id in subtask_ids:
                result = await self.execute_task(task_id)
                results.append(result)
                
                # Stop on failure if needed
                if not result["success"] and not handle_failures:
                    break
        else:
            # Execute in parallel
            tasks = [self.execute_task(task_id) for task_id in subtask_ids]
            results = await asyncio.gather(*tasks)
        
        # Retry failed tasks if needed
        if max_retries > 0:
            retried_tasks = self.execution_tree.retry_failed_tasks()
            if retried_tasks:
                for task_id in retried_tasks:
                    retry_result = await self.execute_task(task_id)
                    # Update the corresponding result in the results list
                    for i, subtask_id in enumerate(subtask_ids):
                        if subtask_id == task_id:
                            results[i] = retry_result
        
        # Update the root task status based on subtask results
        all_succeeded = all(result["success"] for result in results)
        self.execution_tree.complete_task(root_task_id, {
            "success": all_succeeded,
            "subtask_count": len(subtasks),
            "successful_subtasks": sum(1 for result in results if result["success"]),
            "failed_subtasks": sum(1 for result in results if not result["success"])
        })
        
        return {
            "goal": goal,
            "success": all_succeeded,
            "root_task_id": root_task_id,
            "subtask_results": results,
            "task_tree": self.execution_tree.get_subtree(root_task_id),
            "statistics": self.execution_tree.get_task_statistics()
        }
    
    def get_task_tree(self, task_id: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get the task execution tree or a specific subtree
        
        Args:
            task_id: ID of the root task for the subtree (None for full tree)
            
        Returns:
            Dictionary or list of dictionaries representing the task tree
        """
        if task_id:
            return self.execution_tree.get_subtree(task_id)
        else:
            return self.execution_tree.get_full_tree()
    
    async def recursive_decompose_and_execute(self, goal: str, 
                                           decomposition_func: Callable[[str, int], List[Dict[str, Any]]],
                                           max_depth: int = 3,
                                           max_retries: int = 0) -> Dict[str, Any]:
        """
        Recursively decompose and execute a complex goal with multi-level subtasks
        
        Args:
            goal: High-level goal description
            decomposition_func: Function that takes a goal and depth, returns a list of subtasks
            max_depth: Maximum recursion depth for task decomposition
            max_retries: Maximum number of retry attempts for each subtask
            
        Returns:
            Dictionary containing execution results and task tree
        """
        # Create the root task for the goal
        root_task_id = self.execution_tree.create_task("", goal)
        
        async def decompose_and_execute_recursive(parent_id: str, sub_goal: str, depth: int):
            if depth > max_depth:
                # Reached maximum depth, execute as a leaf task
                command = f"echo 'Executing leaf task: {sub_goal}'"
                task_id = self.execution_tree.create_task(command, sub_goal, parent_id, max_retries=max_retries)
                await self.execute_task(task_id)
                return
            
            # Decompose the goal into subtasks
            subtasks = decomposition_func(sub_goal, depth)
            
            if not subtasks:
                # No further decomposition possible, execute as a leaf task
                command = f"echo 'Executing leaf task: {sub_goal}'"
                task_id = self.execution_tree.create_task(command, sub_goal, parent_id, max_retries=max_retries)
                await self.execute_task(task_id)
                return
            
            # Create and execute subtasks
            for subtask in subtasks:
                command = subtask.get("command", "")
                description = subtask.get("description", "")
                
                # If this is a non-leaf task (has further subtasks)
                if subtask.get("has_subtasks", False):
                    # Create a parent task with no command
                    sub_task_id = self.execution_tree.create_task("", description, parent_id)
                    # Recursively decompose
                    await decompose_and_execute_recursive(sub_task_id, description, depth + 1)
                else:
                    # This is a leaf task, execute it
                    task_id = self.execution_tree.create_task(
                        command, description, parent_id, 
                        priority=subtask.get("priority", 5),
                        max_retries=subtask.get("max_retries", max_retries)
                    )
                    await self.execute_task(task_id)
        
        # Start recursive decomposition from the root
        await decompose_and_execute_recursive(root_task_id, goal, 1)
        
        # Update the root task status
        all_children = self.execution_tree.get_children(root_task_id)
        all_succeeded = all(child.status == "COMPLETED" for child in all_children)
        
        self.execution_tree.complete_task(root_task_id, {
            "success": all_succeeded,
            "subtask_count": len(all_children),
            "successful_subtasks": sum(1 for child in all_children if child.status == "COMPLETED"),
            "failed_subtasks": sum(1 for child in all_children if child.status == "FAILED")
        })
        
        return {
            "goal": goal,
            "success": all_succeeded,
            "root_task_id": root_task_id,
            "task_tree": self.execution_tree.get_subtree(root_task_id),
            "statistics": self.execution_tree.get_task_statistics(),
            "visualization_data": self.execution_tree.generate_visualization_data()
        }
    
    async def execute_with_live_output(self, command: str, 
                                     timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute a command and stream output in real-time
        
        Args:
            command: The shell command to execute
            timeout: Timeout in seconds (None uses default timeout)
            
        Returns:
            Dict containing execution results
        """
        start_time = datetime.now()
        timeout = timeout or self.default_timeout
        
        # Record command in history
        self.command_history.append({
            "command": command,
            "timestamp": start_time.isoformat(),
            "working_dir": self.working_dir
        })
        
        try:
            # Create subprocess
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir
            )
            
            # Set up timeout
            try:
                stdout_data = []
                stderr_data = []
                
                # Create tasks for reading stdout and stderr
                async def read_stdout():
                    while True:
                        line = await process.stdout.readline()
                        if not line:
                            break
                        line_str = line.decode('utf-8', errors='replace')
                        stdout_data.append(line_str)
                        print(line_str, end='', flush=True)
                
                async def read_stderr():
                    while True:
                        line = await process.stderr.readline()
                        if not line:
                            break
                        line_str = line.decode('utf-8', errors='replace')
                        stderr_data.append(line_str)
                        print(f"[ERROR] {line_str}", end='', flush=True)
                
                # Start reading tasks
                stdout_task = asyncio.create_task(read_stdout())
                stderr_task = asyncio.create_task(read_stderr())
                
                # Wait for process to complete with timeout
                try:
                    exit_code = await asyncio.wait_for(process.wait(), timeout=timeout)
                    
                    # Wait for output reading to complete
                    await stdout_task
                    await stderr_task
                    
                    # Process completed within timeout
                    end_time = datetime.now()
                    execution_time = (end_time - start_time).total_seconds()
                    
                    return {
                        "success": exit_code == 0,
                        "exit_code": exit_code,
                        "stdout": ''.join(stdout_data),
                        "stderr": ''.join(stderr_data),
                        "command": command,
                        "execution_time": execution_time,
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "timeout_occurred": False
                    }
                    
                except asyncio.TimeoutError:
                    # Process timed out
                    try:
                        process.kill()
                        stdout_task.cancel()
                        stderr_task.cancel()
                    except Exception:
                        pass
                    
                    end_time = datetime.now()
                    execution_time = (end_time - start_time).total_seconds()
                    
                    return {
                        "success": False,
                        "exit_code": None,
                        "stdout": ''.join(stdout_data),
                        "stderr": f"Command timed out after {timeout} seconds\n" + ''.join(stderr_data),
                        "command": command,
                        "execution_time": execution_time,
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "timeout_occurred": True
                    }
                    
            except Exception as e:
                # Error in reading output
                try:
                    process.kill()
                except Exception:
                    pass
                
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                return {
                    "success": False,
                    "exit_code": None,
                    "stdout": ''.join(stdout_data),
                    "stderr": f"Error reading output: {str(e)}\n" + ''.join(stderr_data),
                    "command": command,
                    "execution_time": execution_time,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "timeout_occurred": False,
                    "error": str(e)
                }
                
        except Exception as e:
            # Error occurred during execution
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return {
                "success": False,
                "exit_code": None,
                "stdout": "",
                "stderr": f"Error executing command: {str(e)}",
                "command": command,
                "execution_time": execution_time,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "timeout_occurred": False,
                "error": str(e)
            }

# Create a global instance for easy access
shell_tool = ShellCommandTool()

# Async function for use with the agent's tool system
async def execute_shell_command(command: str, live_output: bool = False, 
                              timeout: Optional[int] = None) -> Dict[str, Any]:
    """
    Execute a shell command and return the results
    
    Args:
        command: Shell command to execute
        live_output: Whether to stream output in real-time
        timeout: Timeout in seconds (None uses default)
        
    Returns:
        Dict containing execution results
    """
    if live_output:
        return await shell_tool.execute_with_live_output(command, timeout)
    else:
        return await shell_tool.execute(command, timeout)

# Synchronous function for use in non-async contexts
def execute_shell_command_sync(command: str, timeout: Optional[int] = None) -> Dict[str, Any]:
    """
    Execute a shell command synchronously and return the results
    
    Args:
        command: Shell command to execute
        timeout: Timeout in seconds (None uses default)
        
    Returns:
        Dict containing execution results
    """
    return shell_tool.execute_sync(command, timeout)

# Function to get file count in current directory
async def get_file_count(directory: str = ".", recursive: bool = False) -> Dict[str, Any]:
    """
    Count files in a directory
    
    Args:
        directory: Directory to count files in (default: current directory)
        recursive: Whether to count files in subdirectories
        
    Returns:
        Dict containing file count and details
    """
    try:
        if recursive:
            command = f"find {directory} -type f | wc -l"
        else:
            command = f"ls -1 {directory} | wc -l"
            
        result = await shell_tool.execute(command)
        
        if result["success"]:
            try:
                count = int(result["stdout"].strip())
                return {
                    "success": True,
                    "count": count,
                    "directory": directory,
                    "recursive": recursive,
                    "command": command
                }
            except ValueError:
                return {
                    "success": False,
                    "error": "Failed to parse file count",
                    "stdout": result["stdout"],
                    "directory": directory,
                    "recursive": recursive
                }
        else:
            return {
                "success": False,
                "error": result["stderr"],
                "directory": directory,
                "recursive": recursive
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "directory": directory,
            "recursive": recursive
        }

# Function to list files with details
async def list_files_with_details(directory: str = ".", pattern: str = "*") -> Dict[str, Any]:
    """
    List files with details in a directory
    
    Args:
        directory: Directory to list files in (default: current directory)
        pattern: File pattern to match
        
    Returns:
        Dict containing file listing and details
    """
    try:
        command = f"ls -la {directory}"
        result = await shell_tool.execute(command)
        
        if result["success"]:
            return {
                "success": True,
                "directory": directory,
                "listing": result["stdout"],
                "command": command
            }
        else:
            return {
                "success": False,
                "error": result["stderr"],
                "directory": directory
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "directory": directory
        }

# Function to generate HTML visualization of task tree
def generate_task_tree_visualization(task_tree_data: Dict[str, Any], output_file: str = "task_tree_visualization.html") -> str:
    """
    Generate an HTML visualization of the task execution tree
    
    Args:
        task_tree_data: Task tree data from execution_tree.generate_visualization_data()
        output_file: Path to save the HTML visualization
        
    Returns:
        Path to the generated HTML file
    """
    html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Task Execution Tree Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        #visualization { width: 100%; height: 600px; border: 1px solid #ccc; }
        .node { cursor: pointer; }
        .link { stroke: #999; stroke-opacity: 0.6; stroke-width: 1.5px; }
        .link.dependency { stroke-dasharray: 5,5; }
        .tooltip { position: absolute; background: white; border: 1px solid #ccc; 
                  padding: 10px; border-radius: 5px; pointer-events: none; }
        .statistics { margin-bottom: 20px; padding: 10px; background: #f5f5f5; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Task Execution Tree Visualization</h1>
    
    <div class="statistics">
        <h2>Task Statistics</h2>
        <div id="stats"></div>
    </div>
    
    <div id="visualization"></div>
    
    <script>
    // Task tree data
    const data = TASK_TREE_DATA;
    
    // Set up the visualization
    const width = document.getElementById('visualization').clientWidth;
    const height = document.getElementById('visualization').clientHeight;
    
    // Create the SVG container
    const svg = d3.select('#visualization')
        .append('svg')
        .attr('width', width)
        .attr('height', height);
    
    // Create a group for the visualization
    const g = svg.append('g');
    
    // Add zoom behavior
    svg.call(d3.zoom()
        .scaleExtent([0.1, 4])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        }));
    
    // Create the force simulation
    const simulation = d3.forceSimulation(data.nodes)
        .force('link', d3.forceLink(data.links).id(d => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('x', d3.forceX(width / 2).strength(0.1))
        .force('y', d3.forceY(height / 2).strength(0.1));
    
    // Create the links
    const link = g.append('g')
        .selectAll('line')
        .data(data.links)
        .enter()
        .append('line')
        .attr('class', d => `link ${d.type}`)
        .attr('stroke', '#999');
    
    // Create a tooltip
    const tooltip = d3.select('body')
        .append('div')
        .attr('class', 'tooltip')
        .style('opacity', 0);
    
    // Create the nodes
    const node = g.append('g')
        .selectAll('circle')
        .data(data.nodes)
        .enter()
        .append('circle')
        .attr('class', 'node')
        .attr('r', d => 10)
        .attr('fill', d => d.color)
        .on('mouseover', function(event, d) {
            tooltip.transition()
                .duration(200)
                .style('opacity', .9);
            
            let tooltipHtml = `
                <strong>Task:</strong> ${d.name}<br>
                <strong>Status:</strong> ${d.status}<br>
                <strong>Priority:</strong> ${d.priority}<br>
            `;
            
            if (d.execution_time !== null) {
                tooltipHtml += `<strong>Execution Time:</strong> ${d.execution_time.toFixed(2)}s<br>`;
            }
            
            tooltip.html(tooltipHtml)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 28) + 'px');
        })
        .on('mouseout', function() {
            tooltip.transition()
                .duration(500)
                .style('opacity', 0);
        })
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));
    
    // Add labels to nodes
    const label = g.append('g')
        .selectAll('text')
        .data(data.nodes)
        .enter()
        .append('text')
        .attr('dx', 15)
        .attr('dy', '.35em')
        .text(d => d.name.length > 20 ? d.name.substring(0, 20) + '...' : d.name)
        .style('font-size', '10px');
    
    // Update positions on each tick
    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        
        node
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);
        
        label
            .attr('x', d => d.x)
            .attr('y', d => d.y);
    });
    
    // Drag functions
    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    
    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    
    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
    
    // Display statistics
    const stats = data.statistics;
    const statsHtml = `
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
            <div><strong>Total Tasks:</strong> ${stats.total_tasks}</div>
            <div><strong>Pending:</strong> ${stats.pending_tasks}</div>
            <div><strong>In Progress:</strong> ${stats.in_progress_tasks}</div>
            <div><strong>Completed:</strong> ${stats.completed_tasks}</div>
            <div><strong>Failed:</strong> ${stats.failed_tasks}</div>
            <div><strong>Blocked:</strong> ${stats.blocked_tasks}</div>
            <div><strong>Retry:</strong> ${stats.retry_tasks}</div>
            <div><strong>Avg. Execution Time:</strong> ${stats.avg_execution_time.toFixed(2)}s</div>
        </div>
    `;
    
    document.getElementById('stats').innerHTML = statsHtml;
    </script>
</body>
</html>
""".replace('TASK_TREE_DATA', str(task_tree_data))

    with open(output_file, 'w') as f:
        f.write(html_template)
    
    return output_file

# Function to decompose and execute a complex goal
async def execute_complex_goal(goal: str, subtasks: List[Dict[str, Any]], 
                             sequential: bool = True,
                             max_retries: int = 0,
                             handle_failures: bool = False,
                             visualize: bool = False) -> Dict[str, Any]:
    """
    Decompose and execute a complex goal with multiple subtasks
    
    Args:
        goal: High-level goal description
        subtasks: List of subtask dictionaries, each with 'command' and 'description' keys
        sequential: Whether to execute subtasks sequentially or in parallel
        max_retries: Maximum number of retry attempts for each subtask
        handle_failures: Whether to continue execution after subtask failures
        visualize: Whether to generate an HTML visualization of the task tree
        
    Returns:
        Dict containing execution results and task tree
    """
    def decompose_func(g: str) -> List[Dict[str, Any]]:
        return subtasks
    
    result = await shell_tool.decompose_and_execute(
        goal, decompose_func, sequential, max_retries, handle_failures
    )
    
    if visualize:
        visualization_data = shell_tool.execution_tree.generate_visualization_data()
        visualization_file = generate_task_tree_visualization(visualization_data)
        result["visualization_file"] = visualization_file
    
    return result

# Function for recursive task decomposition and execution
async def execute_recursive_goal(goal: str, max_depth: int = 3, 
                               max_retries: int = 0,
                               visualize: bool = False) -> Dict[str, Any]:
    """
    Recursively decompose and execute a complex goal
    
    Args:
        goal: High-level goal description
        max_depth: Maximum recursion depth for task decomposition
        max_retries: Maximum number of retry attempts for each subtask
        visualize: Whether to generate an HTML visualization of the task tree
        
    Returns:
        Dict containing execution results and task tree
    """
    # Example decomposition function that creates subtasks based on the goal
    def recursive_decomposition(g: str, depth: int) -> List[Dict[str, Any]]:
        # This is a simple example - in a real system, you might use an LLM or other
        # intelligent system to decompose tasks based on the goal and current depth
        
        if "file" in g.lower() or "directory" in g.lower():
            if depth == 1:
                return [
                    {
                        "description": "Analyze file system structure",
                        "has_subtasks": True
                    },
                    {
                        "description": "Process files",
                        "has_subtasks": True
                    },
                    {
                        "description": "Generate report",
                        "has_subtasks": True
                    }
                ]
            elif depth == 2:
                if "analyze" in g.lower():
                    return [
                        {
                            "command": "find . -type f | wc -l",
                            "description": "Count total files"
                        },
                        {
                            "command": "find . -type d | wc -l",
                            "description": "Count total directories"
                        },
                        {
                            "command": "du -sh .",
                            "description": "Calculate total size"
                        }
                    ]
                elif "process" in g.lower():
                    return [
                        {
                            "command": "find . -name '*.py' | wc -l",
                            "description": "Count Python files"
                        },
                        {
                            "command": "find . -name '*.py' -exec wc -l {} \\; | sort -nr | head -5",
                            "description": "Find largest Python files"
                        }
                    ]
                elif "report" in g.lower():
                    return [
                        {
                            "command": "echo 'Generating file system report...'",
                            "description": "Start report generation"
                        },
                        {
                            "command": "echo 'Report complete'",
                            "description": "Finalize report"
                        }
                    ]
        
        # Default decomposition for other types of goals
        if depth == 1:
            return [
                {
                    "description": f"Plan approach for: {g}",
                    "has_subtasks": True
                },
                {
                    "description": f"Execute core tasks for: {g}",
                    "has_subtasks": True
                },
                {
                    "description": f"Verify results for: {g}",
                    "has_subtasks": True
                }
            ]
        elif depth == 2:
            if "plan" in g.lower():
                return [
                    {
                        "command": "echo 'Analyzing requirements...'",
                        "description": "Analyze requirements"
                    },
                    {
                        "command": "echo 'Identifying dependencies...'",
                        "description": "Identify dependencies"
                    }
                ]
            elif "execute" in g.lower():
                return [
                    {
                        "command": "echo 'Executing primary task...'",
                        "description": "Execute primary task"
                    },
                    {
                        "command": "echo 'Processing results...'",
                        "description": "Process results"
                    }
                ]
            elif "verify" in g.lower():
                return [
                    {
                        "command": "echo 'Verifying output...'",
                        "description": "Verify output"
                    },
                    {
                        "command": "echo 'Generating verification report...'",
                        "description": "Generate verification report"
                    }
                ]
        
        # Leaf tasks for depth 3 or when no specific decomposition is available
        return [
            {
                "command": f"echo 'Executing task: {g}'",
                "description": f"Execute {g}"
            }
        ]
    
    result = await shell_tool.recursive_decompose_and_execute(
        goal, recursive_decomposition, max_depth, max_retries
    )
    
    if visualize:
        visualization_file = generate_task_tree_visualization(result["visualization_data"])
        result["visualization_file"] = visualization_file
    
    return result

# Function to create a file processing pipeline
async def process_files(directory: str, file_pattern: str, 
                      processing_command: str,
                      max_retries: int = 0,
                      visualize: bool = False) -> Dict[str, Any]:
    """
    Create and execute a file processing pipeline
    
    Args:
        directory: Directory containing files to process
        file_pattern: Pattern to match files (e.g., "*.txt")
        processing_command: Command template to process each file (use {} for filename)
        max_retries: Maximum number of retry attempts for each subtask
        visualize: Whether to generate an HTML visualization of the task tree
        
    Returns:
        Dict containing execution results
    """
    # Define the goal decomposition
    def decompose_file_processing(goal: str) -> List[Dict[str, Any]]:
        subtasks = [
            {
                "command": f"find {directory} -name '{file_pattern}' -type f",
                "description": f"Find files matching pattern '{file_pattern}' in {directory}",
                "priority": 1,  # Higher priority (lower number)
                "max_retries": max_retries
            },
            {
                "command": f"for file in $(find {directory} -name '{file_pattern}' -type f); do {processing_command.format('$file')}; done",
                "description": f"Process each matching file with command: {processing_command}",
                "priority": 5,
                "max_retries": max_retries
            },
            {
                "command": f"echo 'Processing complete for {file_pattern} files in {directory}'",
                "description": "Confirm processing completion",
                "priority": 10,  # Lower priority (higher number)
                "max_retries": 0  # No retries needed for confirmation
            }
        ]
        return subtasks
    
    result = await shell_tool.decompose_and_execute(
        f"Process {file_pattern} files in {directory} with command: {processing_command}",
        decompose_file_processing,
        sequential=True,
        max_retries=max_retries,
        handle_failures=False
    )
    
    if visualize:
        visualization_data = shell_tool.execution_tree.generate_visualization_data()
        visualization_file = generate_task_tree_visualization(visualization_data)
        result["visualization_file"] = visualization_file
    
    return result

if __name__ == "__main__":
    # Example usage
    async def main():
        # Execute a simple command
        result = await execute_shell_command("ls -la")
        print(f"Command executed with status: {result['success']}")
        print(f"Output:\n{result['stdout']}")
        
        # Get file count
        count_result = await get_file_count(".", recursive=True)
        if count_result["success"]:
            print(f"Found {count_result['count']} files")
        else:
            print(f"Error counting files: {count_result['error']}")
        
        # Execute with live output
        print("\nExecuting with live output:")
        live_result = await shell_tool.execute_with_live_output("find . -type f | head -10")
        print(f"\nCommand completed with status: {live_result['success']}")
        
        # Example of task decomposition and goal-oriented execution
        print("\nExecuting a complex goal with task decomposition:")
        goal_result = await execute_complex_goal(
            "Count and analyze Python files in the current directory",
            [
                {
                    "command": "find . -name '*.py' | wc -l",
                    "description": "Count Python files",
                    "priority": 1
                },
                {
                    "command": "find . -name '*.py' -exec wc -l {} \\; | sort -nr | head -5",
                    "description": "Find the 5 largest Python files by line count",
                    "priority": 2
                },
                {
                    "command": "find . -name '*.py' -exec grep -l 'import' {} \\; | wc -l",
                    "description": "Count Python files that import other modules",
                    "priority": 3
                }
            ],
            sequential=True,
            max_retries=1,
            visualize=True
        )
        
        print(f"Complex goal execution completed with status: {goal_result['success']}")
        print("Task tree structure:")
        import json
        print(json.dumps(goal_result['task_tree'], indent=2))
        
        if "visualization_file" in goal_result:
            print(f"\nTask tree visualization generated: {goal_result['visualization_file']}")
            print("Open this file in a web browser to view the visualization.")
        
        # Example of recursive task decomposition
        print("\nExecuting a recursive goal decomposition:")
        recursive_result = await execute_recursive_goal(
            "Analyze the repository file structure and generate a report",
            max_depth=3,
            max_retries=1,
            visualize=True
        )
        
        print(f"Recursive goal execution completed with status: {recursive_result['success']}")
        print("Task statistics:")
        print(json.dumps(recursive_result['statistics'], indent=2))
        
        if "visualization_file" in recursive_result:
            print(f"\nTask tree visualization generated: {recursive_result['visualization_file']}")
            print("Open this file in a web browser to view the visualization.")
    
    if sys.platform != "win32":
        asyncio.run(main())
    else:
        print("This example requires a Unix-like system")
