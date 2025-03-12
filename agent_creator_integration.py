#!/usr/bin/env python3
"""
Agent Creator Integration

This module integrates the Autonomous Agent Creator with the rl_cli.py system,
allowing the main agent to create and manage other agents autonomously.
"""

import os
import sys
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
from pathlib import Path

# Import the autonomous agent creator
from autonomous_agent_creator import AgentCreator, AgentTemplate, SpecializedAgentTemplate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentCreatorIntegration:
    """
    Integrates the Autonomous Agent Creator with the rl_cli.py system
    """
    
    def __init__(self, agent_instance):
        """
        Initialize the integration
        
        Args:
            agent_instance: The main agent instance from rl_cli.py
        """
        self.agent = agent_instance
        self.creator = AgentCreator()
        self.running = False
        self.task_queue = asyncio.Queue()
        self.worker_task = None
        
        # Register tools with the main agent
        self._register_tools()
        
        logger.info("Initialized Agent Creator Integration")
        
    def _register_tools(self):
        """Register agent creation tools with the main agent"""
        try:
            # Import the register_tool function from the main agent
            from rl_cli import register_tool
            
            # Register list_templates tool
            register_tool(
                name="list_agent_templates",
                func=self.list_templates,
                description="List all available agent templates",
                parameters={}
            )
            
            # Register create_agent tool
            register_tool(
                name="create_agent",
                func=self.create_agent,
                description="Create a new agent from a template",
                parameters={
                    "template_id": {
                        "type": "string",
                        "description": "ID of the template to use"
                    },
                    "config": {
                        "type": "object",
                        "description": "Optional configuration for the agent",
                        "default": {}
                    }
                }
            )
            
            # Register create_template tool
            register_tool(
                name="create_agent_template",
                func=self.create_template,
                description="Create a new agent template",
                parameters={
                    "name": {
                        "type": "string",
                        "description": "Name of the template"
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of the template"
                    },
                    "capabilities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of capabilities"
                    },
                    "specialization": {
                        "type": "string",
                        "description": "Optional specialization for the template",
                        "default": None
                    },
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of dependencies",
                        "default": []
                    }
                }
            )
            
            # Register run_agent tool
            register_tool(
                name="run_agent",
                func=self.run_agent,
                description="Run a created agent",
                parameters={
                    "agent_id": {
                        "type": "string",
                        "description": "ID of the agent to run"
                    }
                }
            )
            
            logger.info("Registered agent creation tools")
            
        except ImportError:
            logger.error("Failed to import register_tool from rl_cli")
        except Exception as e:
            logger.error(f"Error registering tools: {e}")
            
    async def start(self):
        """Start the integration"""
        if self.running:
            return
            
        self.running = True
        
        # Start the agent creator
        await self.creator.start()
        
        # Create default templates if none exist
        await self.creator.create_default_templates()
        
        # Start the worker task
        self.worker_task = asyncio.create_task(self._worker_loop())
        
        logger.info("Started Agent Creator Integration")
        
    async def stop(self):
        """Stop the integration"""
        if not self.running:
            return
            
        self.running = False
        
        # Stop the agent creator
        await self.creator.stop()
        
        # Cancel the worker task
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
            
        logger.info("Stopped Agent Creator Integration")
        
    async def _worker_loop(self):
        """Worker loop for processing tasks"""
        try:
            while self.running:
                try:
                    # Get a task from the queue with timeout
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                    
                    # Process the task
                    func, args, kwargs, future = task
                    try:
                        result = await func(*args, **kwargs)
                        future.set_result(result)
                    except Exception as e:
                        future.set_exception(e)
                    finally:
                        self.task_queue.task_done()
                        
                except asyncio.TimeoutError:
                    # No tasks in queue, just continue
                    pass
                    
        except asyncio.CancelledError:
            logger.info("Worker task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in worker loop: {e}")
            
    async def _queue_task(self, func, *args, **kwargs):
        """Queue a task for execution"""
        future = asyncio.Future()
        await self.task_queue.put((func, args, kwargs, future))
        return await future
        
    async def list_templates(self) -> List[Dict[str, Any]]:
        """List all available agent templates"""
        return await self._queue_task(self._list_templates_impl)
        
    async def _list_templates_impl(self) -> List[Dict[str, Any]]:
        """Implementation of list_templates"""
        return self.creator.list_templates()
        
    async def create_agent(self, template_id: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new agent from a template"""
        return await self._queue_task(self._create_agent_impl, template_id, config)
        
    async def _create_agent_impl(self, template_id: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Implementation of create_agent"""
        return await self.creator.create_agent(template_id, config)
        
    async def create_template(self, name: str, description: str, capabilities: List[str],
                            specialization: Optional[str] = None, 
                            dependencies: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create a new agent template"""
        return await self._queue_task(self._create_template_impl, name, description, 
                                    capabilities, specialization, dependencies)
        
    async def _create_template_impl(self, name: str, description: str, capabilities: List[str],
                                  specialization: Optional[str] = None, 
                                  dependencies: Optional[List[str]] = None) -> Dict[str, Any]:
        """Implementation of create_template"""
        if specialization and dependencies:
            template = self.creator.create_specialized_template(
                name=name,
                description=description,
                capabilities=capabilities,
                specialization=specialization,
                dependencies=dependencies or []
            )
        else:
            template = self.creator.create_template(
                name=name,
                description=description,
                capabilities=capabilities
            )
            
        return template.to_dict()
        
    async def run_agent(self, agent_id: str) -> Dict[str, Any]:
        """Run a created agent"""
        return await self._queue_task(self._run_agent_impl, agent_id)
        
    async def _run_agent_impl(self, agent_id: str) -> Dict[str, Any]:
        """Implementation of run_agent"""
        if agent_id not in self.creator.created_agents:
            raise ValueError(f"Agent not found: {agent_id}")
            
        agent_info = self.creator.created_agents[agent_id]
        file_path = agent_info["file_path"]
        
        if not os.path.exists(file_path):
            raise ValueError(f"Agent file not found: {file_path}")
            
        # Create a task to track the agent execution
        task_id = None
        if hasattr(self.agent, 'create_task'):
            task_result = await self.agent.create_task(
                title=f"Run agent: {agent_info['name']}",
                description=f"Execute the agent at {file_path}",
                priority=3,
                tags=["agent", "execution"]
            )
            
            if task_result.get("success", False):
                task_id = task_result.get("task_id")
                await self.agent.update_task(task_id, status="in_progress")
        
        try:
            # Run the agent as a subprocess
            import subprocess
            process = subprocess.Popen(
                [sys.executable, file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Set a timeout for the agent execution
            timeout = 60  # 60 seconds timeout
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                exit_code = process.returncode
                success = exit_code == 0
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                exit_code = -1
                success = False
                stderr += f"\nProcess timed out after {timeout} seconds"
                
            result = {
                "agent_id": agent_id,
                "success": success,
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr
            }
            
            # Update task with results if created
            if task_id:
                await self.agent.update_task(
                    task_id=task_id,
                    status="completed" if success else "failed",
                    result=result
                )
                
            return result
            
        except Exception as e:
            # Update task as failed if created
            if task_id:
                await self.agent.update_task(
                    task_id=task_id,
                    status="failed",
                    result={"error": str(e)}
                )
                
            return {
                "agent_id": agent_id,
                "success": False,
                "error": str(e)
            }

async def initialize_integration(agent_instance):
    """Initialize and start the integration with the main agent"""
    integration = AgentCreatorIntegration(agent_instance)
    await integration.start()
    return integration

def get_integration(agent_instance):
    """Get the integration for the given agent instance"""
    if hasattr(agent_instance, '_extensions'):
        return agent_instance._extensions.get('agent_creator_integration')
    
    # Try to get from global registry
    if hasattr(sys.modules[__name__], '_integration_registry'):
        agent_id = getattr(agent_instance, 'id', id(agent_instance))
        return sys.modules[__name__]._integration_registry.get(agent_id)
    
    return None

def register_with_agent(agent_instance):
    """Register the integration with the main agent"""
    # Create the integration
    integration = asyncio.run(initialize_integration(agent_instance))
    
    # Store the integration in a global variable or attach it to a mutable attribute
    # that already exists on the agent, rather than trying to add a new attribute
    # to a Pydantic model
    if hasattr(agent_instance, '_extensions'):
        agent_instance._extensions['agent_creator_integration'] = integration
    else:
        # Create a global registry if we can't store it on the agent
        if not hasattr(sys.modules[__name__], '_integration_registry'):
            setattr(sys.modules[__name__], '_integration_registry', {})
        
        # Use the agent's id or another unique identifier
        agent_id = getattr(agent_instance, 'id', id(agent_instance))
        sys.modules[__name__]._integration_registry[agent_id] = integration
    
    return integration
