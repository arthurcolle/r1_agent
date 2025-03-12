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
                parent_id: Optional[str] = None):
        """
        Initialize a task node
        
        Args:
            task_id: Unique identifier for this task
            command: The shell command to execute
            description: Human-readable description of the task
            parent_id: ID of the parent task, if any
        """
        self.task_id = task_id
        self.command = command
        self.description = description
        self.parent_id = parent_id
        self.child_ids = []
        self.status = "PENDING"  # PENDING, IN_PROGRESS, COMPLETED, FAILED
        self.result = None
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.execution_time = None
    
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
            "execution_time": self.execution_time
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
    
    def create_task(self, command: str, description: str = "", 
                   parent_id: Optional[str] = None) -> str:
        """
        Create a new task in the execution tree
        
        Args:
            command: The shell command to execute
            description: Human-readable description of the task
            parent_id: ID of the parent task, if any
            
        Returns:
            The ID of the newly created task
        """
        task_id = str(uuid.uuid4())
        task = TaskNode(task_id, command, description, parent_id)
        self.tasks[task_id] = task
        
        if parent_id:
            if parent_id in self.tasks:
                self.tasks[parent_id].add_child(task_id)
        else:
            self.root_tasks.append(task_id)
            
        return task_id
    
    def get_task(self, task_id: str) -> Optional[TaskNode]:
        """Get a task by its ID"""
        return self.tasks.get(task_id)
    
    def get_children(self, task_id: str) -> List[TaskNode]:
        """Get all child tasks for a given task ID"""
        task = self.tasks.get(task_id)
        if not task:
            return []
        return [self.tasks[child_id] for child_id in task.child_ids if child_id in self.tasks]
    
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
    
    def update_task_status(self, task_id: str, status: str) -> bool:
        """Update the status of a task"""
        task = self.tasks.get(task_id)
        if not task:
            return False
        task.status = status
        return True
    
    def complete_task(self, task_id: str, result: Dict[str, Any]) -> bool:
        """Mark a task as completed with results"""
        task = self.tasks.get(task_id)
        if not task:
            return False
        task.complete(result)
        return True


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
                                    parent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new task and execute it immediately
        
        Args:
            command: Shell command to execute
            description: Human-readable description of the task
            parent_id: ID of the parent task, if any
            
        Returns:
            Dictionary containing task ID and execution result
        """
        task_id = self.execution_tree.create_task(command, description, parent_id)
        result = await self.execute_task(task_id)
        
        return {
            "task_id": task_id,
            "result": result
        }
    
    async def decompose_and_execute(self, goal: str, 
                                  decomposition_func: Callable[[str], List[Dict[str, Any]]],
                                  sequential: bool = True) -> Dict[str, Any]:
        """
        Decompose a high-level goal into subtasks and execute them
        
        Args:
            goal: High-level goal description
            decomposition_func: Function that takes a goal and returns a list of subtasks
                                Each subtask should be a dict with 'command' and 'description' keys
            sequential: Whether to execute subtasks sequentially or in parallel
            
        Returns:
            Dictionary containing execution results and task tree
        """
        # Create the root task for the goal
        root_task_id = self.execution_tree.create_task("", goal)
        
        # Decompose the goal into subtasks
        subtasks = decomposition_func(goal)
        
        # Create task nodes for each subtask
        subtask_ids = []
        for subtask in subtasks:
            task_id = self.execution_tree.create_task(
                subtask["command"],
                subtask.get("description", ""),
                root_task_id
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
                if not result["success"]:
                    break
        else:
            # Execute in parallel
            tasks = [self.execute_task(task_id) for task_id in subtask_ids]
            results = await asyncio.gather(*tasks)
        
        # Update the root task status based on subtask results
        all_succeeded = all(result["success"] for result in results)
        self.execution_tree.complete_task(root_task_id, {
            "success": all_succeeded,
            "subtask_count": len(subtasks),
            "successful_subtasks": sum(1 for result in results if result["success"])
        })
        
        return {
            "goal": goal,
            "success": all_succeeded,
            "root_task_id": root_task_id,
            "subtask_results": results,
            "task_tree": self.execution_tree.get_subtree(root_task_id)
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

# Function to decompose and execute a complex goal
async def execute_complex_goal(goal: str, subtasks: List[Dict[str, Any]], 
                             sequential: bool = True) -> Dict[str, Any]:
    """
    Decompose and execute a complex goal with multiple subtasks
    
    Args:
        goal: High-level goal description
        subtasks: List of subtask dictionaries, each with 'command' and 'description' keys
        sequential: Whether to execute subtasks sequentially or in parallel
        
    Returns:
        Dict containing execution results and task tree
    """
    def decompose_func(g: str) -> List[Dict[str, Any]]:
        return subtasks
    
    return await shell_tool.decompose_and_execute(goal, decompose_func, sequential)

# Function to create a file processing pipeline
async def process_files(directory: str, file_pattern: str, 
                      processing_command: str) -> Dict[str, Any]:
    """
    Create and execute a file processing pipeline
    
    Args:
        directory: Directory containing files to process
        file_pattern: Pattern to match files (e.g., "*.txt")
        processing_command: Command template to process each file (use {} for filename)
        
    Returns:
        Dict containing execution results
    """
    # Define the goal decomposition
    def decompose_file_processing(goal: str) -> List[Dict[str, Any]]:
        subtasks = [
            {
                "command": f"find {directory} -name '{file_pattern}' -type f",
                "description": f"Find files matching pattern '{file_pattern}' in {directory}"
            },
            {
                "command": f"for file in $(find {directory} -name '{file_pattern}' -type f); do {processing_command.format('$file')}; done",
                "description": f"Process each matching file with command: {processing_command}"
            },
            {
                "command": f"echo 'Processing complete for {file_pattern} files in {directory}'",
                "description": "Confirm processing completion"
            }
        ]
        return subtasks
    
    return await shell_tool.decompose_and_execute(
        f"Process {file_pattern} files in {directory} with command: {processing_command}",
        decompose_file_processing
    )

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
                    "description": "Count Python files"
                },
                {
                    "command": "find . -name '*.py' -exec wc -l {} \\; | sort -nr | head -5",
                    "description": "Find the 5 largest Python files by line count"
                },
                {
                    "command": "find . -name '*.py' -exec grep -l 'import' {} \\; | wc -l",
                    "description": "Count Python files that import other modules"
                }
            ]
        )
        
        print(f"Complex goal execution completed with status: {goal_result['success']}")
        print("Task tree structure:")
        import json
        print(json.dumps(goal_result['task_tree'], indent=2))
    
    if sys.platform != "win32":
        asyncio.run(main())
    else:
        print("This example requires a Unix-like system")
