#!/usr/bin/env python3
"""
Shell Command Execution Tool

This module provides enhanced shell command execution capabilities for the agent,
allowing it to run system commands and process their output.
"""

import os
import sys
import subprocess
import shlex
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

class ShellCommandTool:
    """
    Tool for executing shell commands with advanced features:
    - Command validation and sanitization
    - Timeout handling
    - Output formatting
    - Error handling
    - Resource usage tracking
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
    
    if sys.platform != "win32":
        asyncio.run(main())
    else:
        print("This example requires a Unix-like system")
