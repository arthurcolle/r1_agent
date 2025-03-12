#!/usr/bin/env python3
"""
Network Supervisor Module

This module provides network monitoring, supervision, and recovery capabilities.
It can detect network issues, attempt recovery, and notify about network status changes.
"""

import os
import sys
import time
import socket
import asyncio
import logging
import json
import subprocess
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NetworkMonitor:
    """
    Monitors network connectivity and provides recovery mechanisms.
    """
    def __init__(self, check_interval: float = 60.0):
        self.check_interval = check_interval
        self.monitoring_task = None
        self.running = False
        self.status_history = []
        self.status_callbacks = []
        self.recovery_strategies = {
            "dns_failure": self._recover_dns,
            "connection_failure": self._recover_connection,
            "gateway_failure": self._recover_gateway
        }
        self.current_status = {
            "connected": False,
            "last_check": None,
            "issues": []
        }
        
    async def start_monitoring(self):
        """Start network monitoring"""
        if self.running:
            logger.warning("Network monitoring is already running")
            return
            
        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Network monitoring started")
        
    async def stop_monitoring(self):
        """Stop network monitoring"""
        if not self.running:
            logger.warning("Network monitoring is not running")
            return
            
        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Network monitoring stopped")
        
    def register_status_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register a callback for network status changes"""
        self.status_callbacks.append(callback)
        
    def get_current_status(self) -> Dict[str, Any]:
        """Get the current network status"""
        return self.current_status.copy()
        
    def get_status_history(self) -> List[Dict[str, Any]]:
        """Get the network status history"""
        return self.status_history.copy()
        
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while self.running:
                status = await self._check_network()
                
                # Check if status changed
                status_changed = (
                    self.current_status.get("connected") != status["connected"] or
                    self.current_status.get("issues") != status["issues"]
                )
                
                # Update current status
                self.current_status = status
                
                # Add to history
                self.status_history.append(status)
                if len(self.status_history) > 100:
                    self.status_history.pop(0)
                    
                # Notify callbacks if status changed
                if status_changed:
                    for callback in self.status_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                asyncio.create_task(callback(status))
                            else:
                                callback(status)
                        except Exception as e:
                            logger.error(f"Error in status callback: {e}")
                
                # If not connected, try recovery
                if not status["connected"] and status["issues"]:
                    await self._attempt_recovery(status["issues"])
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
        except asyncio.CancelledError:
            logger.info("Network monitoring loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in network monitoring loop: {e}")
            logger.error(traceback.format_exc())
            
    async def _check_network(self) -> Dict[str, Any]:
        """Check network connectivity and identify issues"""
        status = {
            "connected": False,
            "last_check": datetime.now().isoformat(),
            "issues": []
        }
        
        # Check DNS resolution
        try:
            socket.gethostbyname("www.google.com")
        except socket.gaierror:
            status["issues"].append("dns_failure")
        
        # Check internet connectivity
        try:
            # Try to connect to Google's DNS server
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            status["connected"] = True
        except OSError:
            status["issues"].append("connection_failure")
            
            # Check if we can reach the gateway
            try:
                # Get default gateway
                if sys.platform.startswith('win'):
                    gateway = subprocess.check_output("ipconfig | findstr Gateway", shell=True).decode()
                    gateway = gateway.split(":")[-1].strip()
                else:
                    gateway = subprocess.check_output("ip route | grep default | awk '{print $3}'", shell=True).decode().strip()
                
                # Try to ping the gateway
                ping_cmd = "ping -n 1 " if sys.platform.startswith('win') else "ping -c 1 "
                subprocess.check_output(ping_cmd + gateway, shell=True)
            except (subprocess.CalledProcessError, Exception):
                status["issues"].append("gateway_failure")
        
        # Add latency information if connected
        if status["connected"]:
            try:
                start_time = time.time()
                socket.create_connection(("8.8.8.8", 53), timeout=3)
                latency = time.time() - start_time
                status["latency_ms"] = round(latency * 1000, 2)
            except OSError:
                pass
                
        return status
        
    async def _attempt_recovery(self, issues: List[str]):
        """Attempt to recover from network issues"""
        logger.info(f"Attempting to recover from network issues: {issues}")
        
        for issue in issues:
            if issue in self.recovery_strategies:
                try:
                    await self.recovery_strategies[issue]()
                except Exception as e:
                    logger.error(f"Error in recovery strategy for {issue}: {e}")
                    
    async def _recover_dns(self):
        """Attempt to recover from DNS issues"""
        logger.info("Attempting DNS recovery")
        
        # Try to set Google DNS servers
        try:
            if sys.platform.startswith('win'):
                # Windows: Use netsh to set DNS servers
                subprocess.run(
                    "netsh interface ip set dns name=\"Wi-Fi\" static 8.8.8.8 primary",
                    shell=True, check=True
                )
                subprocess.run(
                    "netsh interface ip add dns name=\"Wi-Fi\" 8.8.4.4 index=2",
                    shell=True, check=True
                )
            else:
                # Linux: Update resolv.conf
                with open('/etc/resolv.conf', 'w') as f:
                    f.write("nameserver 8.8.8.8\nnameserver 8.8.4.4\n")
                    
            logger.info("DNS recovery: Set Google DNS servers")
        except Exception as e:
            logger.error(f"DNS recovery failed: {e}")
            
    async def _recover_connection(self):
        """Attempt to recover from connection issues"""
        logger.info("Attempting connection recovery")
        
        # Try to restart network interface
        try:
            if sys.platform.startswith('win'):
                # Windows: Disable and re-enable network adapter
                subprocess.run(
                    "netsh interface set interface \"Wi-Fi\" disabled",
                    shell=True, check=True
                )
                await asyncio.sleep(2)
                subprocess.run(
                    "netsh interface set interface \"Wi-Fi\" enabled",
                    shell=True, check=True
                )
            else:
                # Linux: Restart networking service
                subprocess.run(
                    "systemctl restart NetworkManager",
                    shell=True, check=True
                )
                
            logger.info("Connection recovery: Restarted network interface")
        except Exception as e:
            logger.error(f"Connection recovery failed: {e}")
            
    async def _recover_gateway(self):
        """Attempt to recover from gateway issues"""
        logger.info("Attempting gateway recovery")
        
        # Try to release and renew DHCP lease
        try:
            if sys.platform.startswith('win'):
                # Windows: Release and renew IP address
                subprocess.run("ipconfig /release", shell=True, check=True)
                await asyncio.sleep(2)
                subprocess.run("ipconfig /renew", shell=True, check=True)
            else:
                # Linux: Restart DHCP client
                interface = subprocess.check_output(
                    "ip route | grep default | awk '{print $5}'", 
                    shell=True
                ).decode().strip()
                subprocess.run(f"dhclient -r {interface}", shell=True, check=True)
                await asyncio.sleep(2)
                subprocess.run(f"dhclient {interface}", shell=True, check=True)
                
            logger.info("Gateway recovery: Renewed DHCP lease")
        except Exception as e:
            logger.error(f"Gateway recovery failed: {e}")

class NetworkSupervisor:
    """
    Supervises network-dependent services and manages their lifecycle
    based on network availability.
    """
    def __init__(self, monitor: NetworkMonitor):
        self.monitor = monitor
        self.services = {}  # name -> service_info
        self.running = False
        self.supervisor_task = None
        
    async def start(self):
        """Start the network supervisor"""
        if self.running:
            logger.warning("Network supervisor is already running")
            return
            
        self.running = True
        
        # Register for network status updates
        self.monitor.register_status_callback(self._handle_network_status_change)
        
        # Start monitoring if not already running
        if not self.monitor.running:
            await self.monitor.start_monitoring()
            
        self.supervisor_task = asyncio.create_task(self._supervisor_loop())
        logger.info("Network supervisor started")
        
    async def stop(self):
        """Stop the network supervisor"""
        if not self.running:
            logger.warning("Network supervisor is not running")
            return
            
        self.running = False
        
        # Stop all services
        for service_name in list(self.services.keys()):
            await self.stop_service(service_name)
            
        if self.supervisor_task:
            self.supervisor_task.cancel()
            try:
                await self.supervisor_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Network supervisor stopped")
        
    async def register_service(self, name: str, start_func: Callable, 
                             stop_func: Callable, check_func: Callable,
                             requires_network: bool = True,
                             auto_restart: bool = True) -> bool:
        """
        Register a network-dependent service
        
        Args:
            name: Service name
            start_func: Function to start the service
            stop_func: Function to stop the service
            check_func: Function to check if service is running
            requires_network: Whether the service requires network connectivity
            auto_restart: Whether to automatically restart the service on network recovery
            
        Returns:
            bool: Success status
        """
        if name in self.services:
            logger.warning(f"Service {name} is already registered")
            return False
            
        self.services[name] = {
            "name": name,
            "start_func": start_func,
            "stop_func": stop_func,
            "check_func": check_func,
            "requires_network": requires_network,
            "auto_restart": auto_restart,
            "running": False,
            "last_status_change": datetime.now().isoformat(),
            "status_history": []
        }
        
        logger.info(f"Registered service {name}")
        
        # Start the service if appropriate
        current_status = self.monitor.get_current_status()
        if not requires_network or (requires_network and current_status.get("connected", False)):
            await self.start_service(name)
            
        return True
        
    async def start_service(self, name: str) -> bool:
        """Start a registered service"""
        if name not in self.services:
            logger.warning(f"Service {name} is not registered")
            return False
            
        service = self.services[name]
        
        # Check if service requires network
        if service["requires_network"]:
            current_status = self.monitor.get_current_status()
            if not current_status.get("connected", False):
                logger.warning(f"Cannot start service {name}: Network is not connected")
                return False
                
        # Start the service
        try:
            start_func = service["start_func"]
            if asyncio.iscoroutinefunction(start_func):
                await start_func()
            else:
                start_func()
                
            service["running"] = True
            service["last_status_change"] = datetime.now().isoformat()
            service["status_history"].append({
                "status": "started",
                "timestamp": service["last_status_change"]
            })
            
            logger.info(f"Started service {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting service {name}: {e}")
            return False
            
    async def stop_service(self, name: str) -> bool:
        """Stop a registered service"""
        if name not in self.services:
            logger.warning(f"Service {name} is not registered")
            return False
            
        service = self.services[name]
        
        # Stop the service
        try:
            stop_func = service["stop_func"]
            if asyncio.iscoroutinefunction(stop_func):
                await stop_func()
            else:
                stop_func()
                
            service["running"] = False
            service["last_status_change"] = datetime.now().isoformat()
            service["status_history"].append({
                "status": "stopped",
                "timestamp": service["last_status_change"]
            })
            
            logger.info(f"Stopped service {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping service {name}: {e}")
            return False
            
    async def check_service(self, name: str) -> bool:
        """Check if a service is running"""
        if name not in self.services:
            logger.warning(f"Service {name} is not registered")
            return False
            
        service = self.services[name]
        
        try:
            check_func = service["check_func"]
            if asyncio.iscoroutinefunction(check_func):
                is_running = await check_func()
            else:
                is_running = check_func()
                
            # Update service status if it changed
            if service["running"] != is_running:
                service["running"] = is_running
                service["last_status_change"] = datetime.now().isoformat()
                service["status_history"].append({
                    "status": "running" if is_running else "not_running",
                    "timestamp": service["last_status_change"]
                })
                
            return is_running
            
        except Exception as e:
            logger.error(f"Error checking service {name}: {e}")
            return False
            
    async def _supervisor_loop(self):
        """Main supervisor loop"""
        try:
            while self.running:
                # Check all services
                for name, service in list(self.services.items()):
                    try:
                        is_running = await self.check_service(name)
                        
                        # If service should be running but isn't, try to restart it
                        if service["running"] and not is_running:
                            logger.warning(f"Service {name} should be running but isn't, attempting to restart")
                            await self.start_service(name)
                            
                    except Exception as e:
                        logger.error(f"Error in supervisor loop for service {name}: {e}")
                
                # Wait before checking again
                await asyncio.sleep(30)  # Check services every 30 seconds
                
        except asyncio.CancelledError:
            logger.info("Network supervisor loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in network supervisor loop: {e}")
            
    async def _handle_network_status_change(self, status: Dict[str, Any]):
        """Handle network status changes"""
        logger.info(f"Network status changed: connected={status['connected']}")
        
        if status["connected"]:
            # Network is now connected, start services that require network
            for name, service in list(self.services.items()):
                if service["requires_network"] and service["auto_restart"] and not service["running"]:
                    logger.info(f"Network is now connected, starting service {name}")
                    await self.start_service(name)
        else:
            # Network is now disconnected, stop services that require network
            for name, service in list(self.services.items()):
                if service["requires_network"] and service["running"]:
                    logger.info(f"Network is now disconnected, stopping service {name}")
                    await self.stop_service(name)

async def main():
    """Main function for testing the network supervisor"""
    # Create a network monitor
    monitor = NetworkMonitor(check_interval=10.0)
    
    # Create a network supervisor
    supervisor = NetworkSupervisor(monitor)
    
    # Define a simple test service
    def start_test_service():
        logger.info("Test service started")
        return True
        
    def stop_test_service():
        logger.info("Test service stopped")
        return True
        
    def check_test_service():
        # Simulate a service that's running
        return True
    
    # Register the test service
    await supervisor.register_service(
        name="test_service",
        start_func=start_test_service,
        stop_func=stop_test_service,
        check_func=check_test_service,
        requires_network=True
    )
    
    # Start the supervisor
    await supervisor.start()
    
    try:
        # Run for a while
        logger.info("Running network supervisor for 60 seconds...")
        await asyncio.sleep(60)
    finally:
        # Stop the supervisor
        await supervisor.stop()
        
        # Stop the monitor
        await monitor.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())
