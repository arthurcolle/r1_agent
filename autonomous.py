#!/usr/bin/env python3
"""
Command-line script for autonomous task execution.
This script provides a convenient way to manage autonomous tasks from the command line.
"""

import sys
import os
import argparse
import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from rl_cli import AgentCLI
from network_supervisor import NetworkMonitor, NetworkSupervisor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutonomousManager:
    """
    Manager for autonomous task execution with supervision tree integration.
    """
    def __init__(self):
        self.cli = AgentCLI()
        self.network_monitor = NetworkMonitor(check_interval=60.0)
        self.network_supervisor = NetworkSupervisor(self.network_monitor)
        self.supervision_active = False
        self.config_file = Path.home() / ".autonomous_config.json"
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            "auto_restart": True,
            "network_dependent": True,
            "max_changes": 5,
            "auto_execute_scripts": True,
            "scripts_dir": "./generated_scripts",
            "check_interval": 60.0,
            "persistence_interval": 60,
            "max_continuous_work_time": 3600,
            "response_mode": "normal"
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    config = json.load(f)
                # Update with any missing default values
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                return default_config
        else:
            return default_config
            
    def _save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            
    async def setup_supervision(self):
        """Set up the supervision tree for autonomous execution"""
        if self.supervision_active:
            logger.warning("Supervision is already active")
            return
            
        # Start network monitoring
        await self.network_monitor.start_monitoring()
        
        # Start network supervisor
        await self.network_supervisor.start()
        
        # Register autonomous execution as a service
        await self.network_supervisor.register_service(
            name="autonomous_execution",
            start_func=self._start_autonomous_execution,
            stop_func=self._stop_autonomous_execution,
            check_func=self._check_autonomous_execution,
            requires_network=self.config["network_dependent"],
            auto_restart=self.config["auto_restart"]
        )
        
        self.supervision_active = True
        logger.info("Supervision tree set up for autonomous execution")
        
    async def teardown_supervision(self):
        """Tear down the supervision tree"""
        if not self.supervision_active:
            logger.warning("Supervision is not active")
            return
            
        # Stop network supervisor
        await self.network_supervisor.stop()
        
        # Stop network monitoring
        await self.network_monitor.stop_monitoring()
        
        self.supervision_active = False
        logger.info("Supervision tree torn down")
        
    def _start_autonomous_execution(self):
        """Start autonomous execution (called by supervisor)"""
        # Configure agent based on settings
        self.cli.agent.self_transformation.max_autonomous_changes = self.config["max_changes"]
        self.cli.agent.self_transformation.auto_execute_scripts = self.config["auto_execute_scripts"]
        self.cli.agent.self_transformation.script_output_dir = self.config["scripts_dir"]
        self.cli.agent.max_continuous_work_time = self.config["max_continuous_work_time"]
        self.cli.agent.persistence_interval = self.config["persistence_interval"]
        
        # Set response mode
        asyncio.run(self.cli.agent.set_response_mode(self.config["response_mode"]))
        
        # Enable autonomous mode
        self.cli.do_enable_autonomous(f"{self.config['max_changes']}")
        
        # Start autonomous execution
        asyncio.run(self.cli.agent.start_autonomous_execution())
        
        logger.info("Autonomous execution started")
        return True
        
    def _stop_autonomous_execution(self):
        """Stop autonomous execution (called by supervisor)"""
        # Stop autonomous execution
        asyncio.run(self.cli.agent.stop_autonomous_execution())
        
        # Disable autonomous mode
        self.cli.agent.self_transformation.autonomous_mode = False
        
        logger.info("Autonomous execution stopped")
        return True
        
    def _check_autonomous_execution(self):
        """Check if autonomous execution is running (called by supervisor)"""
        return (
            self.cli.agent.self_transformation.autonomous_mode and
            self.cli.agent.work_continuity_enabled
        )
        
    async def start(self, with_supervision: bool = True):
        """Start autonomous execution"""
        # Start the persistence timer
        await self.cli.agent._start_persistence_timer_async()
        
        if with_supervision:
            # Set up supervision tree
            await self.setup_supervision()
        else:
            # Start directly without supervision
            self._start_autonomous_execution()
            
        logger.info(f"Autonomous execution started with supervision={with_supervision}")
        
    async def stop(self, with_supervision: bool = True):
        """Stop autonomous execution"""
        if with_supervision and self.supervision_active:
            # Tear down supervision tree
            await self.teardown_supervision()
        else:
            # Stop directly without supervision
            self._stop_autonomous_execution()
            
        logger.info(f"Autonomous execution stopped with supervision={with_supervision}")
        
    async def status(self) -> Dict[str, Any]:
        """Get autonomous execution status"""
        # Get status from agent
        agent_status = asyncio.run(self.cli.agent.get_autonomous_execution_status())
        
        # Add supervision status
        status = {
            **agent_status,
            "supervision_active": self.supervision_active
        }
        
        # Add network status if supervision is active
        if self.supervision_active:
            status["network"] = self.network_monitor.get_current_status()
            
            # Get service status
            service = self.network_supervisor.services.get("autonomous_execution", {})
            status["service"] = {
                "running": service.get("running", False),
                "requires_network": service.get("requires_network", True),
                "auto_restart": service.get("auto_restart", True),
                "last_status_change": service.get("last_status_change")
            }
            
        return status
        
    async def create_task(self, title: str, description: str, priority: int = 5,
                        tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create a new autonomous task"""
        # Format tags if provided
        tags_arg = ""
        if tags:
            tags_arg = f" tags={','.join(tags)}"
        
        # Execute create command
        self.cli.do_autonomous(f"create \"{title}\" \"{description}\" priority={priority}{tags_arg}")
        
        # Get the created task (would need to parse output or access directly)
        # For now, just return a success message
        return {
            "success": True,
            "message": f"Created autonomous task: {title}"
        }
        
    async def decompose_task(self, task_id: str, subtasks: List[str]) -> Dict[str, Any]:
        """Break down a task into subtasks"""
        # Format subtasks
        subtasks_str = " ".join([f"\"{subtask}\"" for subtask in subtasks])
        
        # Execute decompose command
        self.cli.do_autonomous(f"decompose {task_id} {subtasks_str}")
        
        # Return success message
        return {
            "success": True,
            "message": f"Decomposed task {task_id} into {len(subtasks)} subtasks"
        }
        
    async def update_config(self, **kwargs) -> Dict[str, Any]:
        """Update configuration settings"""
        # Update config with provided values
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                
        # Save updated config
        self._save_config()
        
        # Apply changes if needed
        if self.supervision_active:
            # Update service configuration
            service = self.network_supervisor.services.get("autonomous_execution", {})
            if service:
                service["requires_network"] = self.config["network_dependent"]
                service["auto_restart"] = self.config["auto_restart"]
                
        # Update agent configuration
        self.cli.agent.self_transformation.max_autonomous_changes = self.config["max_changes"]
        self.cli.agent.self_transformation.auto_execute_scripts = self.config["auto_execute_scripts"]
        self.cli.agent.self_transformation.script_output_dir = self.config["scripts_dir"]
        self.cli.agent.max_continuous_work_time = self.config["max_continuous_work_time"]
        self.cli.agent.persistence_interval = self.config["persistence_interval"]
        
        # Set response mode
        asyncio.run(self.cli.agent.set_response_mode(self.config["response_mode"]))
        
        return {
            "success": True,
            "config": self.config
        }

async def async_main():
    """Async main function for the autonomous command-line tool"""
    parser = argparse.ArgumentParser(description="Autonomous task execution tool")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start autonomous task execution")
    start_parser.add_argument("--no-supervision", action="store_true", help="Start without supervision tree")
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop autonomous task execution")
    stop_parser.add_argument("--no-supervision", action="store_true", help="Stop without supervision tree")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show autonomous execution status")
    status_parser.add_argument("--json", action="store_true", help="Output status as JSON")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new autonomous task")
    create_parser.add_argument("title", help="Short title for the task")
    create_parser.add_argument("description", help="Detailed description of the task")
    create_parser.add_argument("--priority", type=int, default=5, help="Priority level (1-10, lower is higher priority)")
    create_parser.add_argument("--tags", help="Comma-separated list of tags")
    
    # Decompose command
    decompose_parser = subparsers.add_parser("decompose", help="Break down a task into subtasks")
    decompose_parser.add_argument("task_id", help="ID of the task to decompose")
    decompose_parser.add_argument("subtasks", nargs="+", help="List of subtask titles")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="View or update configuration")
    config_parser.add_argument("--set", nargs=2, action="append", metavar=("KEY", "VALUE"), help="Set a configuration value")
    config_parser.add_argument("--json", action="store_true", help="Output config as JSON")
    
    # Network command
    network_parser = subparsers.add_parser("network", help="Network monitoring and supervision")
    network_parser.add_argument("action", choices=["status", "start", "stop"], help="Network action to perform")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create autonomous manager
    manager = AutonomousManager()
    
    # Execute the appropriate command
    if args.command == "start":
        await manager.start(not args.no_supervision)
        print("Autonomous execution started")
        
    elif args.command == "stop":
        await manager.stop(not args.no_supervision)
        print("Autonomous execution stopped")
        
    elif args.command == "status":
        status = await manager.status()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print("\nAutonomous Execution Status:")
            print(f"Enabled: {status['enabled']}")
            print(f"Supervision Active: {status['supervision_active']}")
            
            if status.get('current_task'):
                task = status['current_task']
                print(f"\nCurrent Task: {task['title']}")
                print(f"  ID: {task['task_id']}")
                print(f"  Status: {task['status']}")
                print(f"  Progress: {task['progress']*100:.1f}%")
                
            if status.get('work_session_duration'):
                duration_mins = status['work_session_duration'] / 60
                print(f"\nWork Session Duration: {duration_mins:.1f} minutes")
                print(f"Maximum Work Time: {status['max_continuous_work_time']/60:.1f} minutes")
                
            print("\nTask Statistics:")
            print(f"  Pending: {status['task_stats']['pending']}")
            print(f"  Completed: {status['task_stats']['completed']}")
            print(f"  Paused: {status['task_stats']['paused']}")
            
            if status.get('network'):
                net = status['network']
                print("\nNetwork Status:")
                print(f"  Connected: {net['connected']}")
                if net.get('issues'):
                    print(f"  Issues: {', '.join(net['issues'])}")
                if net.get('latency_ms'):
                    print(f"  Latency: {net['latency_ms']} ms")
                    
    elif args.command == "create":
        # Parse tags
        tags = args.tags.split(",") if args.tags else None
        
        # Create task
        result = await manager.create_task(args.title, args.description, args.priority, tags)
        print(result["message"])
        
    elif args.command == "decompose":
        # Decompose task
        result = await manager.decompose_task(args.task_id, args.subtasks)
        print(result["message"])
        
    elif args.command == "config":
        if args.set:
            # Update config
            config_updates = {}
            for key, value in args.set:
                # Convert value to appropriate type
                if value.lower() in ["true", "yes", "1"]:
                    typed_value = True
                elif value.lower() in ["false", "no", "0"]:
                    typed_value = False
                elif value.isdigit():
                    typed_value = int(value)
                elif value.replace(".", "", 1).isdigit():
                    typed_value = float(value)
                else:
                    typed_value = value
                    
                config_updates[key] = typed_value
                
            result = await manager.update_config(**config_updates)
            print("Configuration updated")
            
        # Show config
        if args.json:
            print(json.dumps(manager.config, indent=2))
        else:
            print("\nConfiguration:")
            for key, value in manager.config.items():
                print(f"  {key}: {value}")
                
    elif args.command == "network":
        if args.action == "status":
            if manager.supervision_active:
                status = manager.network_monitor.get_current_status()
                print("\nNetwork Status:")
                print(f"  Connected: {status['connected']}")
                print(f"  Last Check: {status['last_check']}")
                if status.get('issues'):
                    print(f"  Issues: {', '.join(status['issues'])}")
                if status.get('latency_ms'):
                    print(f"  Latency: {status['latency_ms']} ms")
            else:
                print("Network monitoring is not active")
                
        elif args.action == "start":
            if not manager.supervision_active:
                await manager.network_monitor.start_monitoring()
                print("Network monitoring started")
            else:
                print("Network monitoring is already active")
                
        elif args.action == "stop":
            if manager.supervision_active:
                await manager.network_monitor.stop_monitoring()
                print("Network monitoring stopped")
            else:
                print("Network monitoring is not active")
                
    else:
        parser.print_help()

def main():
    """Main entry point for the autonomous command-line tool"""
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
