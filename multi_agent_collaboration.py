"""
Multi-Agent Collaboration Framework

This module implements a framework for multiple agents to collaborate on complex tasks.
It provides mechanisms for:
- Agent discovery and registration
- Task distribution and coordination
- Message passing and communication protocols
- Shared knowledge and resource management
- Conflict resolution strategies
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from datetime import datetime
from enum import Enum, auto
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentRole(Enum):
    """Defines possible roles an agent can take in a collaboration"""
    COORDINATOR = auto()  # Manages the overall collaboration
    SPECIALIST = auto()   # Has specialized knowledge or capabilities
    EXECUTOR = auto()     # Executes specific tasks
    VALIDATOR = auto()    # Validates results and ensures quality
    OBSERVER = auto()     # Monitors the collaboration without direct participation

class MessageType(Enum):
    """Types of messages that can be exchanged between agents"""
    TASK_REQUEST = auto()      # Request for task execution
    TASK_RESPONSE = auto()     # Response to a task request
    KNOWLEDGE_QUERY = auto()   # Query for information
    KNOWLEDGE_RESPONSE = auto() # Response to a knowledge query
    STATUS_UPDATE = auto()     # Update on agent status or task progress
    COORDINATION = auto()      # Coordination message between agents
    ERROR = auto()             # Error notification
    HEARTBEAT = auto()         # Periodic heartbeat to confirm agent is active

@dataclass
class AgentMessage:
    """Message exchanged between agents"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: Optional[str] = None  # None for broadcast
    message_type: MessageType = MessageType.COORDINATION
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    conversation_id: Optional[str] = None
    in_reply_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentCapability:
    """Represents a capability that an agent can provide"""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class CollaborationTask:
    """A task within a collaboration that needs to be completed"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    required_capabilities: List[str] = field(default_factory=list)
    assigned_agent_id: Optional[str] = None
    status: str = "pending"  # pending, assigned, in_progress, completed, failed
    priority: int = 5  # 1-10, lower is higher priority
    dependencies: List[str] = field(default_factory=list)  # IDs of tasks this depends on
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    deadline: Optional[str] = None
    result: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class CollaborationAgent:
    """
    Base class for an agent that can participate in collaborations.
    Agents can register capabilities, join collaborations, and exchange messages.
    """
    def __init__(self, agent_id: str, name: str, description: str = ""):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.capabilities: Dict[str, AgentCapability] = {}
        self.collaborations: Set[str] = set()
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.message_queue = asyncio.Queue()
        self.running = False
        self.message_processor_task = None
        self.last_heartbeat = datetime.now().isoformat()
        self.status = "idle"  # idle, busy, offline
        
    def register_capability(self, capability: AgentCapability) -> None:
        """Register a capability that this agent can provide"""
        self.capabilities[capability.name] = capability
        logger.info(f"Agent {self.name} registered capability: {capability.name}")
        
    def register_message_handler(self, message_type: MessageType, 
                               handler: Callable[[AgentMessage], Any]) -> None:
        """Register a handler for a specific message type"""
        self.message_handlers[message_type] = handler
        
    async def start(self) -> None:
        """Start the agent's message processing loop"""
        if self.running:
            return
            
        self.running = True
        self.status = "idle"
        self.message_processor_task = asyncio.create_task(self._process_messages())
        logger.info(f"Agent {self.name} started")
        
    async def stop(self) -> None:
        """Stop the agent's message processing loop"""
        if not self.running:
            return
            
        self.running = False
        self.status = "offline"
        
        if self.message_processor_task:
            self.message_processor_task.cancel()
            try:
                await self.message_processor_task
            except asyncio.CancelledError:
                pass
                
        logger.info(f"Agent {self.name} stopped")
        
    async def send_message(self, message: AgentMessage) -> None:
        """Send a message to another agent or collaboration"""
        # In a real implementation, this would use a message broker or direct communication
        # For now, we'll just log the message
        logger.info(f"Agent {self.name} sending message: {message.message_type.name} to {message.recipient_id}")
        
        # The actual sending would happen here
        # For testing, we can directly put it in the recipient's queue if we have access
        
    async def receive_message(self, message: AgentMessage) -> None:
        """Receive a message from another agent or collaboration"""
        await self.message_queue.put(message)
        
    async def _process_messages(self) -> None:
        """Process incoming messages from the queue"""
        try:
            while self.running:
                try:
                    # Get message with timeout to allow for clean shutdown
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                    
                    # Update status to busy while processing
                    old_status = self.status
                    self.status = "busy"
                    
                    try:
                        # Process the message
                        if message.message_type in self.message_handlers:
                            handler = self.message_handlers[message.message_type]
                            if asyncio.iscoroutinefunction(handler):
                                await handler(message)
                            else:
                                handler(message)
                        else:
                            # Default handling based on message type
                            await self._default_message_handler(message)
                            
                        self.message_queue.task_done()
                    finally:
                        # Restore previous status
                        self.status = old_status
                        
                except asyncio.TimeoutError:
                    # No messages, just continue
                    pass
                    
                # Send periodic heartbeat
                now = datetime.now()
                last_heartbeat = datetime.fromisoformat(self.last_heartbeat)
                if (now - last_heartbeat).total_seconds() > 60:  # Every minute
                    await self._send_heartbeat()
                    
        except asyncio.CancelledError:
            logger.info(f"Message processor for agent {self.name} cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in message processor for agent {self.name}: {e}")
            
    async def _default_message_handler(self, message: AgentMessage) -> None:
        """Default handler for messages without a specific handler"""
        if message.message_type == MessageType.TASK_REQUEST:
            # Check if we can handle this task
            task_info = message.content.get("task", {})
            capabilities_needed = task_info.get("required_capabilities", [])
            
            can_handle = all(cap in self.capabilities for cap in capabilities_needed)
            
            if can_handle:
                # Accept the task
                response = AgentMessage(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type=MessageType.TASK_RESPONSE,
                    content={
                        "task_id": task_info.get("task_id"),
                        "accepted": True,
                        "estimated_completion": datetime.now().isoformat()  # Would be calculated
                    },
                    in_reply_to=message.message_id
                )
            else:
                # Decline the task
                response = AgentMessage(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type=MessageType.TASK_RESPONSE,
                    content={
                        "task_id": task_info.get("task_id"),
                        "accepted": False,
                        "reason": "Missing required capabilities"
                    },
                    in_reply_to=message.message_id
                )
                
            await self.send_message(response)
            
        elif message.message_type == MessageType.HEARTBEAT:
            # Just acknowledge receipt
            logger.debug(f"Agent {self.name} received heartbeat from {message.sender_id}")
            
    async def _send_heartbeat(self) -> None:
        """Send a heartbeat message to all collaborations"""
        self.last_heartbeat = datetime.now().isoformat()
        
        for collab_id in self.collaborations:
            heartbeat = AgentMessage(
                sender_id=self.agent_id,
                recipient_id=collab_id,
                message_type=MessageType.HEARTBEAT,
                content={
                    "status": self.status,
                    "load": len(self.message_queue.qsize())
                }
            )
            await self.send_message(heartbeat)

class AgentCollaboration:
    """
    Manages a collaboration between multiple agents.
    Handles task distribution, message routing, and coordination.
    """
    def __init__(self, collaboration_id: str, name: str, description: str = ""):
        self.collaboration_id = collaboration_id
        self.name = name
        self.description = description
        self.agents: Dict[str, CollaborationAgent] = {}
        self.tasks: Dict[str, CollaborationTask] = {}
        self.message_history: List[AgentMessage] = []
        self.running = False
        self.coordinator_task = None
        self.created_at = datetime.now().isoformat()
        self.status = "initializing"  # initializing, active, completed, failed
        self.shared_knowledge: Dict[str, Any] = {}
        
    async def start(self) -> None:
        """Start the collaboration"""
        if self.running:
            return
            
        self.running = True
        self.status = "active"
        self.coordinator_task = asyncio.create_task(self._coordination_loop())
        logger.info(f"Collaboration {self.name} started")
        
        # Notify all agents that the collaboration has started
        start_message = AgentMessage(
            sender_id=self.collaboration_id,
            message_type=MessageType.COORDINATION,
            content={
                "action": "collaboration_started",
                "collaboration_id": self.collaboration_id,
                "name": self.name
            }
        )
        
        for agent in self.agents.values():
            start_message.recipient_id = agent.agent_id
            await agent.receive_message(start_message)
            
    async def stop(self) -> None:
        """Stop the collaboration"""
        if not self.running:
            return
            
        self.running = False
        self.status = "completed"
        
        if self.coordinator_task:
            self.coordinator_task.cancel()
            try:
                await self.coordinator_task
            except asyncio.CancelledError:
                pass
                
        # Notify all agents that the collaboration has stopped
        stop_message = AgentMessage(
            sender_id=self.collaboration_id,
            message_type=MessageType.COORDINATION,
            content={
                "action": "collaboration_stopped",
                "collaboration_id": self.collaboration_id,
                "name": self.name
            }
        )
        
        for agent in self.agents.values():
            stop_message.recipient_id = agent.agent_id
            await agent.receive_message(stop_message)
            
        logger.info(f"Collaboration {self.name} stopped")
        
    def add_agent(self, agent: CollaborationAgent, role: AgentRole = AgentRole.SPECIALIST) -> None:
        """Add an agent to the collaboration"""
        self.agents[agent.agent_id] = agent
        agent.collaborations.add(self.collaboration_id)
        logger.info(f"Agent {agent.name} added to collaboration {self.name} with role {role.name}")
        
    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the collaboration"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.collaborations.remove(self.collaboration_id)
            del self.agents[agent_id]
            logger.info(f"Agent {agent_id} removed from collaboration {self.name}")
            
    async def add_task(self, task: CollaborationTask) -> str:
        """Add a task to the collaboration"""
        self.tasks[task.task_id] = task
        logger.info(f"Task {task.title} added to collaboration {self.name}")
        
        # If the collaboration is running, try to assign the task
        if self.running:
            await self._assign_task(task)
            
        return task.task_id
        
    async def route_message(self, message: AgentMessage) -> None:
        """Route a message to the appropriate recipient(s)"""
        # Store in message history
        self.message_history.append(message)
        
        if message.recipient_id is None:
            # Broadcast to all agents
            for agent in self.agents.values():
                await agent.receive_message(message)
        elif message.recipient_id in self.agents:
            # Send to specific agent
            await self.agents[message.recipient_id].receive_message(message)
        elif message.recipient_id == self.collaboration_id:
            # Message for the collaboration itself
            await self._handle_collaboration_message(message)
        else:
            logger.warning(f"Message recipient {message.recipient_id} not found in collaboration {self.name}")
            
    async def _handle_collaboration_message(self, message: AgentMessage) -> None:
        """Handle messages directed to the collaboration itself"""
        if message.message_type == MessageType.STATUS_UPDATE:
            # Update task status
            task_id = message.content.get("task_id")
            new_status = message.content.get("status")
            
            if task_id in self.tasks and new_status:
                self.tasks[task_id].status = new_status
                logger.info(f"Task {task_id} status updated to {new_status}")
                
                # If task is completed, check if dependent tasks can now be assigned
                if new_status == "completed":
                    self.tasks[task_id].result = message.content.get("result")
                    await self._check_dependent_tasks(task_id)
                    
        elif message.message_type == MessageType.KNOWLEDGE_QUERY:
            # Handle knowledge query
            query = message.content.get("query")
            if query in self.shared_knowledge:
                # Send knowledge response
                response = AgentMessage(
                    sender_id=self.collaboration_id,
                    recipient_id=message.sender_id,
                    message_type=MessageType.KNOWLEDGE_RESPONSE,
                    content={
                        "query": query,
                        "result": self.shared_knowledge[query]
                    },
                    in_reply_to=message.message_id
                )
                await self.route_message(response)
                
    async def _coordination_loop(self) -> None:
        """Main coordination loop for the collaboration"""
        try:
            while self.running:
                # Check for unassigned tasks
                unassigned_tasks = [t for t in self.tasks.values() if t.status == "pending"]
                for task in unassigned_tasks:
                    await self._assign_task(task)
                    
                # Check for timed-out tasks
                now = datetime.now()
                for task in self.tasks.values():
                    if task.status == "in_progress" and task.deadline:
                        deadline = datetime.fromisoformat(task.deadline)
                        if now > deadline:
                            logger.warning(f"Task {task.task_id} has missed its deadline")
                            # Reassign the task
                            task.status = "pending"
                            task.assigned_agent_id = None
                            
                # Check if all tasks are completed
                if all(t.status == "completed" for t in self.tasks.values()):
                    logger.info(f"All tasks in collaboration {self.name} completed")
                    # Could auto-stop here, but we'll leave that to explicit stop calls
                    
                # Sleep before next check
                await asyncio.sleep(5)
                
        except asyncio.CancelledError:
            logger.info(f"Coordination loop for collaboration {self.name} cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in coordination loop for collaboration {self.name}: {e}")
            self.status = "failed"
            
    async def _assign_task(self, task: CollaborationTask) -> bool:
        """Attempt to assign a task to an appropriate agent"""
        # Check if task dependencies are met
        for dep_id in task.dependencies:
            if dep_id in self.tasks and self.tasks[dep_id].status != "completed":
                # Dependency not yet completed
                return False
                
        # Find agents with the required capabilities
        capable_agents = []
        for agent in self.agents.values():
            if agent.status != "offline" and all(cap in agent.capabilities for cap in task.required_capabilities):
                capable_agents.append(agent)
                
        if not capable_agents:
            logger.warning(f"No capable agents found for task {task.task_id}")
            return False
            
        # For now, just pick the first available agent
        # In a real implementation, we would use a more sophisticated selection algorithm
        selected_agent = capable_agents[0]
        
        # Send task request
        task_request = AgentMessage(
            sender_id=self.collaboration_id,
            recipient_id=selected_agent.agent_id,
            message_type=MessageType.TASK_REQUEST,
            content={
                "task": {
                    "task_id": task.task_id,
                    "title": task.title,
                    "description": task.description,
                    "required_capabilities": task.required_capabilities,
                    "priority": task.priority,
                    "deadline": task.deadline
                }
            }
        )
        
        # Update task status
        task.status = "assigned"
        task.assigned_agent_id = selected_agent.agent_id
        
        await self.route_message(task_request)
        logger.info(f"Task {task.task_id} assigned to agent {selected_agent.name}")
        return True
        
    async def _check_dependent_tasks(self, completed_task_id: str) -> None:
        """Check if any pending tasks depend on the completed task"""
        for task in self.tasks.values():
            if task.status == "pending" and completed_task_id in task.dependencies:
                # Check if all dependencies are now completed
                all_deps_completed = True
                for dep_id in task.dependencies:
                    if dep_id in self.tasks and self.tasks[dep_id].status != "completed":
                        all_deps_completed = False
                        break
                        
                if all_deps_completed:
                    # Try to assign the task
                    await self._assign_task(task)

class CollaborationManager:
    """
    Manages multiple collaborations and provides a registry for agents.
    Acts as a central hub for collaboration creation and discovery.
    """
    def __init__(self):
        self.collaborations: Dict[str, AgentCollaboration] = {}
        self.registered_agents: Dict[str, CollaborationAgent] = {}
        self.running = False
        self.manager_task = None
        
    async def start(self) -> None:
        """Start the collaboration manager"""
        if self.running:
            return
            
        self.running = True
        self.manager_task = asyncio.create_task(self._manager_loop())
        logger.info("Collaboration manager started")
        
    async def stop(self) -> None:
        """Stop the collaboration manager and all collaborations"""
        if not self.running:
            return
            
        self.running = False
        
        # Stop all collaborations
        for collab in self.collaborations.values():
            await collab.stop()
            
        if self.manager_task:
            self.manager_task.cancel()
            try:
                await self.manager_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Collaboration manager stopped")
        
    def register_agent(self, agent: CollaborationAgent) -> None:
        """Register an agent with the collaboration manager"""
        self.registered_agents[agent.agent_id] = agent
        logger.info(f"Agent {agent.name} registered with collaboration manager")
        
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the collaboration manager"""
        if agent_id in self.registered_agents:
            # Remove agent from all collaborations
            agent = self.registered_agents[agent_id]
            for collab_id in list(agent.collaborations):
                if collab_id in self.collaborations:
                    self.collaborations[collab_id].remove_agent(agent_id)
                    
            del self.registered_agents[agent_id]
            logger.info(f"Agent {agent_id} unregistered from collaboration manager")
            
    async def create_collaboration(self, name: str, description: str = "", 
                                 agent_ids: List[str] = None) -> str:
        """Create a new collaboration"""
        collab_id = str(uuid.uuid4())
        collaboration = AgentCollaboration(collab_id, name, description)
        
        # Add specified agents
        if agent_ids:
            for agent_id in agent_ids:
                if agent_id in self.registered_agents:
                    collaboration.add_agent(self.registered_agents[agent_id])
                    
        self.collaborations[collab_id] = collaboration
        logger.info(f"Created collaboration {name} with ID {collab_id}")
        
        # Start the collaboration if the manager is running
        if self.running:
            await collaboration.start()
            
        return collab_id
        
    async def add_agent_to_collaboration(self, agent_id: str, collab_id: str, 
                                       role: AgentRole = AgentRole.SPECIALIST) -> bool:
        """Add an agent to a collaboration"""
        if agent_id not in self.registered_agents or collab_id not in self.collaborations:
            return False
            
        self.collaborations[collab_id].add_agent(self.registered_agents[agent_id], role)
        return True
        
    async def remove_agent_from_collaboration(self, agent_id: str, collab_id: str) -> bool:
        """Remove an agent from a collaboration"""
        if collab_id not in self.collaborations:
            return False
            
        self.collaborations[collab_id].remove_agent(agent_id)
        return True
        
    async def add_task_to_collaboration(self, collab_id: str, task: CollaborationTask) -> Optional[str]:
        """Add a task to a collaboration"""
        if collab_id not in self.collaborations:
            return None
            
        return await self.collaborations[collab_id].add_task(task)
        
    async def route_message(self, message: AgentMessage) -> None:
        """Route a message to the appropriate collaboration or agent"""
        # Check if recipient is a collaboration
        recipient_id = message.recipient_id
        if recipient_id in self.collaborations:
            await self.collaborations[recipient_id].route_message(message)
        elif recipient_id in self.registered_agents:
            await self.registered_agents[recipient_id].receive_message(message)
        else:
            logger.warning(f"Message recipient {recipient_id} not found")
            
    async def _manager_loop(self) -> None:
        """Main management loop"""
        try:
            while self.running:
                # Check for inactive collaborations
                for collab_id, collab in list(self.collaborations.items()):
                    if collab.status == "completed" or collab.status == "failed":
                        # Archive or clean up completed collaborations
                        # For now, we'll just log it
                        logger.info(f"Collaboration {collab.name} is {collab.status}")
                        
                # Check for inactive agents
                for agent_id, agent in list(self.registered_agents.items()):
                    if agent.status == "offline":
                        # Handle offline agents
                        logger.info(f"Agent {agent.name} is offline")
                        
                # Sleep before next check
                await asyncio.sleep(30)
                
        except asyncio.CancelledError:
            logger.info("Collaboration manager loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in collaboration manager loop: {e}")

# Example usage
async def example_usage():
    # Create a collaboration manager
    manager = CollaborationManager()
    await manager.start()
    
    # Create some agents
    agent1 = CollaborationAgent("agent1", "Data Processor")
    agent1.register_capability(AgentCapability("data_processing", "Process data files"))
    
    agent2 = CollaborationAgent("agent2", "Text Analyzer")
    agent2.register_capability(AgentCapability("text_analysis", "Analyze text content"))
    
    agent3 = CollaborationAgent("agent3", "Report Generator")
    agent3.register_capability(AgentCapability("report_generation", "Generate reports"))
    
    # Start the agents
    await agent1.start()
    await agent2.start()
    await agent3.start()
    
    # Register agents with the manager
    manager.register_agent(agent1)
    manager.register_agent(agent2)
    manager.register_agent(agent3)
    
    # Create a collaboration
    collab_id = await manager.create_collaboration(
        "Data Analysis Project",
        "Analyze data and generate reports",
        ["agent1", "agent2", "agent3"]
    )
    
    # Create some tasks
    task1 = CollaborationTask(
        title="Process Raw Data",
        description="Process the raw data files into structured format",
        required_capabilities=["data_processing"]
    )
    
    task2 = CollaborationTask(
        title="Analyze Processed Data",
        description="Analyze the processed data for insights",
        required_capabilities=["text_analysis"],
        dependencies=[task1.task_id]
    )
    
    task3 = CollaborationTask(
        title="Generate Final Report",
        description="Generate a report of the analysis findings",
        required_capabilities=["report_generation"],
        dependencies=[task2.task_id]
    )
    
    # Add tasks to the collaboration
    await manager.add_task_to_collaboration(collab_id, task1)
    await manager.add_task_to_collaboration(collab_id, task2)
    await manager.add_task_to_collaboration(collab_id, task3)
    
    # Let the collaboration run for a while
    await asyncio.sleep(60)
    
    # Stop everything
    await manager.stop()
    await agent1.stop()
    await agent2.stop()
    await agent3.stop()
    
    logger.info("Example completed")

if __name__ == "__main__":
    asyncio.run(example_usage())
