#!/usr/bin/env python3
"""
Command-line script for autonomous task execution.
This script provides a convenient way to manage autonomous tasks from the command line.
"""

import sys
import os
import argparse
import asyncio
from rl_cli import AgentCLI

def main():
    """Main entry point for the autonomous command-line tool"""
    parser = argparse.ArgumentParser(description="Autonomous task execution tool")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start autonomous task execution")
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop autonomous task execution")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show autonomous execution status")
    
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
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create CLI instance
    cli = AgentCLI()
    
    # Start the persistence timer with a running event loop
    asyncio.run(cli.agent._start_persistence_timer_async())
    
    # Execute the appropriate command
    if args.command == "start":
        cli.do_autonomous("start")
    elif args.command == "stop":
        cli.do_autonomous("stop")
    elif args.command == "status":
        cli.do_autonomous("status")
    elif args.command == "create":
        # Format tags if provided
        tags_arg = ""
        if args.tags:
            tags_arg = f" tags={args.tags}"
        
        # Execute create command
        cli.do_autonomous(f"create \"{args.title}\" \"{args.description}\" priority={args.priority}{tags_arg}")
    elif args.command == "decompose":
        # Format subtasks
        subtasks_str = " ".join([f"\"{subtask}\"" for subtask in args.subtasks])
        
        # Execute decompose command
        cli.do_autonomous(f"decompose {args.task_id} {subtasks_str}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
