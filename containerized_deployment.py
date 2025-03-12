"""
Containerized Agent Deployment

This module provides tools for deploying agents in containerized environments.
It supports Docker, Kubernetes, and cloud container services.

Features:
- Agent containerization with Docker
- Kubernetes deployment configuration
- Resource management and scaling
- Health monitoring and auto-recovery
- Configuration management
- Secure credential handling
- Cross-container communication
"""

import os
import sys
import json
import yaml
import logging
import subprocess
import tempfile
import shutil
import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from enum import Enum, auto
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContainerPlatform(Enum):
    """Supported container platforms"""
    DOCKER = auto()
    KUBERNETES = auto()
    AWS_ECS = auto()
    AZURE_CONTAINER_INSTANCES = auto()
    GCP_CLOUD_RUN = auto()

class ResourceRequirements:
    """Resource requirements for a container"""
    def __init__(self, cpu: str = "0.5", memory: str = "512Mi", 
               gpu: str = None, storage: str = None):
        self.cpu = cpu
        self.memory = memory
        self.gpu = gpu
        self.storage = storage
        
    def to_docker_args(self) -> List[str]:
        """Convert to Docker command-line arguments"""
        args = []
        
        # CPU limits
        if self.cpu:
            args.extend(["--cpus", self.cpu])
            
        # Memory limits
        if self.memory:
            # Convert from Kubernetes-style to Docker style if needed
            if self.memory.endswith("Mi"):
                mem_mb = int(self.memory[:-2])
                args.extend(["--memory", f"{mem_mb}m"])
            elif self.memory.endswith("Gi"):
                mem_mb = int(self.memory[:-2]) * 1024
                args.extend(["--memory", f"{mem_mb}m"])
            else:
                args.extend(["--memory", self.memory])
                
        # GPU limits
        if self.gpu:
            args.extend(["--gpus", self.gpu])
            
        return args
        
    def to_kubernetes_resources(self) -> Dict[str, Any]:
        """Convert to Kubernetes resource requirements"""
        resources = {
            "requests": {},
            "limits": {}
        }
        
        if self.cpu:
            resources["requests"]["cpu"] = self.cpu
            resources["limits"]["cpu"] = self.cpu
            
        if self.memory:
            resources["requests"]["memory"] = self.memory
            resources["limits"]["memory"] = self.memory
            
        if self.gpu:
            resources["limits"]["nvidia.com/gpu"] = self.gpu
            
        return resources
        
    def to_cloud_resources(self, platform: ContainerPlatform) -> Dict[str, Any]:
        """Convert to cloud-specific resource format"""
        if platform == ContainerPlatform.AWS_ECS:
            return {
                "cpu": self.cpu,
                "memory": self.memory,
                # AWS-specific GPU configuration would go here
            }
        elif platform == ContainerPlatform.AZURE_CONTAINER_INSTANCES:
            return {
                "cpu": self.cpu,
                "memoryInGB": self._convert_memory_to_gb(),
                "gpu": {
                    "count": int(self.gpu) if self.gpu else 0,
                    "sku": "K80"  # Default GPU type
                } if self.gpu else None
            }
        elif platform == ContainerPlatform.GCP_CLOUD_RUN:
            return {
                "cpu": self.cpu,
                "memory": self.memory
                # GCP-specific GPU configuration would go here
            }
        else:
            raise ValueError(f"Unsupported platform: {platform}")
            
    def _convert_memory_to_gb(self) -> float:
        """Convert memory string to GB as float"""
        if self.memory.endswith("Mi"):
            return int(self.memory[:-2]) / 1024
        elif self.memory.endswith("Gi"):
            return float(self.memory[:-2])
        elif self.memory.endswith("m"):
            return int(self.memory[:-1]) / 1024 / 1024
        elif self.memory.endswith("g"):
            return float(self.memory[:-1])
        else:
            # Assume MB if no unit
            return int(self.memory) / 1024

@dataclass
class EnvironmentVariable:
    """Environment variable configuration"""
    name: str
    value: Optional[str] = None
    from_secret: Optional[str] = None
    from_config_map: Optional[str] = None
    
    def to_docker_arg(self) -> List[str]:
        """Convert to Docker command-line argument"""
        if self.value:
            return ["-e", f"{self.name}={self.value}"]
        # Docker doesn't directly support secrets in command line
        # For secrets, we'd need to use Docker Compose or other methods
        return []
        
    def to_kubernetes_env(self) -> Dict[str, Any]:
        """Convert to Kubernetes environment variable spec"""
        if self.value:
            return {"name": self.name, "value": self.value}
        elif self.from_secret:
            return {
                "name": self.name,
                "valueFrom": {
                    "secretKeyRef": {
                        "name": self.from_secret,
                        "key": self.name
                    }
                }
            }
        elif self.from_config_map:
            return {
                "name": self.name,
                "valueFrom": {
                    "configMapKeyRef": {
                        "name": self.from_config_map,
                        "key": self.name
                    }
                }
            }
        return {"name": self.name}

@dataclass
class Volume:
    """Volume configuration"""
    name: str
    mount_path: str
    host_path: Optional[str] = None
    persistent: bool = False
    size: Optional[str] = None
    storage_class: Optional[str] = None
    
    def to_docker_args(self) -> List[str]:
        """Convert to Docker command-line arguments"""
        if self.host_path:
            return ["-v", f"{self.host_path}:{self.mount_path}"]
        # For named volumes
        return ["-v", f"{self.name}:{self.mount_path}"]
        
    def to_kubernetes_volume(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Convert to Kubernetes volume and volume mount specs
        
        Returns:
            Tuple of (volume spec, volume mount spec)
        """
        volume_mount = {
            "name": self.name,
            "mountPath": self.mount_path
        }
        
        if self.host_path:
            volume = {
                "name": self.name,
                "hostPath": {
                    "path": self.host_path
                }
            }
        elif self.persistent:
            volume = {
                "name": self.name,
                "persistentVolumeClaim": {
                    "claimName": f"{self.name}-pvc"
                }
            }
        else:
            volume = {
                "name": self.name,
                "emptyDir": {}
            }
            
        return volume, volume_mount
        
    def to_kubernetes_pvc(self) -> Optional[Dict[str, Any]]:
        """Convert to Kubernetes PersistentVolumeClaim if persistent"""
        if not self.persistent:
            return None
            
        return {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": f"{self.name}-pvc"
            },
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "storageClassName": self.storage_class or "standard",
                "resources": {
                    "requests": {
                        "storage": self.size or "1Gi"
                    }
                }
            }
        }

@dataclass
class NetworkConfig:
    """Network configuration"""
    ports: List[Tuple[int, int]] = field(default_factory=list)  # [(container_port, host_port)]
    expose_ports: List[int] = field(default_factory=list)  # Ports to expose without publishing
    network_mode: Optional[str] = None  # e.g., "host", "bridge", "none"
    hostname: Optional[str] = None
    dns: List[str] = field(default_factory=list)
    
    def to_docker_args(self) -> List[str]:
        """Convert to Docker command-line arguments"""
        args = []
        
        # Published ports
        for container_port, host_port in self.ports:
            args.extend(["-p", f"{host_port}:{container_port}"])
            
        # Exposed ports
        for port in self.expose_ports:
            args.extend(["--expose", str(port)])
            
        # Network mode
        if self.network_mode:
            args.extend(["--network", self.network_mode])
            
        # Hostname
        if self.hostname:
            args.extend(["--hostname", self.hostname])
            
        # DNS servers
        for dns_server in self.dns:
            args.extend(["--dns", dns_server])
            
        return args
        
    def to_kubernetes_service(self, name: str) -> Dict[str, Any]:
        """Convert to Kubernetes Service spec"""
        ports = []
        for container_port, host_port in self.ports:
            ports.append({
                "port": host_port,
                "targetPort": container_port,
                "protocol": "TCP"
            })
            
        # Add exposed ports
        for port in self.expose_ports:
            if not any(p["targetPort"] == port for p in ports):
                ports.append({
                    "port": port,
                    "targetPort": port,
                    "protocol": "TCP"
                })
                
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": name
            },
            "spec": {
                "selector": {
                    "app": name
                },
                "ports": ports
            }
        }
        
    def to_kubernetes_container_ports(self) -> List[Dict[str, Any]]:
        """Convert to Kubernetes container ports spec"""
        ports = []
        
        # Add published ports
        for container_port, _ in self.ports:
            ports.append({
                "containerPort": container_port,
                "protocol": "TCP"
            })
            
        # Add exposed ports
        for port in self.expose_ports:
            if not any(p["containerPort"] == port for p in ports):
                ports.append({
                    "containerPort": port,
                    "protocol": "TCP"
                })
                
        return ports

@dataclass
class HealthCheck:
    """Health check configuration"""
    type: str  # "http", "tcp", "exec", "none"
    command: Optional[List[str]] = None  # For exec type
    http_path: Optional[str] = None  # For http type
    port: Optional[int] = None  # For http and tcp types
    initial_delay_seconds: int = 30
    period_seconds: int = 10
    timeout_seconds: int = 5
    success_threshold: int = 1
    failure_threshold: int = 3
    
    def to_docker_args(self) -> List[str]:
        """Convert to Docker command-line arguments"""
        args = []
        
        if self.type == "none":
            return args
            
        health_cmd = ""
        
        if self.type == "http":
            if not self.port:
                raise ValueError("Port is required for HTTP health check")
            health_cmd = f"curl -f http://localhost:{self.port}{self.http_path or '/'} || exit 1"
        elif self.type == "tcp":
            if not self.port:
                raise ValueError("Port is required for TCP health check")
            health_cmd = f"nc -z localhost {self.port} || exit 1"
        elif self.type == "exec":
            if not self.command:
                raise ValueError("Command is required for exec health check")
            health_cmd = " ".join(self.command)
            
        args.extend([
            "--health-cmd", f"CMD-SHELL {health_cmd}",
            "--health-interval", f"{self.period_seconds}s",
            "--health-timeout", f"{self.timeout_seconds}s",
            "--health-retries", str(self.failure_threshold),
            "--health-start-period", f"{self.initial_delay_seconds}s"
        ])
        
        return args
        
    def to_kubernetes_probe(self) -> Dict[str, Any]:
        """Convert to Kubernetes probe spec"""
        probe = {
            "initialDelaySeconds": self.initial_delay_seconds,
            "periodSeconds": self.period_seconds,
            "timeoutSeconds": self.timeout_seconds,
            "successThreshold": self.success_threshold,
            "failureThreshold": self.failure_threshold
        }
        
        if self.type == "http":
            probe["httpGet"] = {
                "path": self.http_path or "/",
                "port": self.port or 80
            }
        elif self.type == "tcp":
            probe["tcpSocket"] = {
                "port": self.port or 80
            }
        elif self.type == "exec":
            probe["exec"] = {
                "command": self.command
            }
        else:
            return None
            
        return probe

@dataclass
class AgentContainerConfig:
    """Configuration for an agent container"""
    name: str
    image: str
    tag: str = "latest"
    command: Optional[List[str]] = None
    args: Optional[List[str]] = None
    working_dir: Optional[str] = None
    env_vars: List[EnvironmentVariable] = field(default_factory=list)
    volumes: List[Volume] = field(default_factory=list)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    health_check: HealthCheck = field(default_factory=lambda: HealthCheck(type="none"))
    restart_policy: str = "always"  # "always", "on-failure", "unless-stopped", "no"
    labels: Dict[str, str] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    
    def get_full_image(self) -> str:
        """Get the full image name with tag"""
        return f"{self.image}:{self.tag}"
        
    def to_docker_run_command(self) -> List[str]:
        """Convert to Docker run command"""
        cmd = ["docker", "run", "-d"]
        
        # Add name
        cmd.extend(["--name", self.name])
        
        # Add restart policy
        if self.restart_policy == "always":
            cmd.append("--restart=always")
        elif self.restart_policy == "on-failure":
            cmd.append("--restart=on-failure")
        elif self.restart_policy == "unless-stopped":
            cmd.append("--restart=unless-stopped")
        elif self.restart_policy == "no":
            cmd.append("--restart=no")
            
        # Add resource requirements
        cmd.extend(self.resources.to_docker_args())
        
        # Add environment variables
        for env_var in self.env_vars:
            cmd.extend(env_var.to_docker_arg())
            
        # Add volumes
        for volume in self.volumes:
            cmd.extend(volume.to_docker_args())
            
        # Add network configuration
        cmd.extend(self.network.to_docker_args())
        
        # Add health check
        cmd.extend(self.health_check.to_docker_args())
        
        # Add working directory
        if self.working_dir:
            cmd.extend(["--workdir", self.working_dir])
            
        # Add labels
        for key, value in self.labels.items():
            cmd.extend(["--label", f"{key}={value}"])
            
        # Add image
        cmd.append(self.get_full_image())
        
        # Add command and args
        if self.command:
            cmd.extend(self.command)
            
        if self.args:
            cmd.extend(self.args)
            
        return cmd
        
    def to_docker_compose_service(self) -> Dict[str, Any]:
        """Convert to Docker Compose service definition"""
        service = {
            "image": self.get_full_image(),
            "container_name": self.name,
            "restart": self.restart_policy
        }
        
        # Add command and args
        if self.command:
            service["command"] = self.command + (self.args or [])
            
        # Add working directory
        if self.working_dir:
            service["working_dir"] = self.working_dir
            
        # Add environment variables
        if self.env_vars:
            service["environment"] = {}
            for env_var in self.env_vars:
                if env_var.value:
                    service["environment"][env_var.name] = env_var.value
                    
        # Add volumes
        if self.volumes:
            service["volumes"] = []
            for volume in self.volumes:
                if volume.host_path:
                    service["volumes"].append(f"{volume.host_path}:{volume.mount_path}")
                else:
                    service["volumes"].append(f"{volume.name}:{volume.mount_path}")
                    
        # Add network configuration
        if self.network.ports:
            service["ports"] = [f"{host}:{container}" for container, host in self.network.ports]
            
        if self.network.expose_ports:
            service["expose"] = self.network.expose_ports
            
        if self.network.network_mode:
            service["network_mode"] = self.network.network_mode
            
        if self.network.hostname:
            service["hostname"] = self.network.hostname
            
        if self.network.dns:
            service["dns"] = self.network.dns
            
        # Add resource requirements
        service["deploy"] = {
            "resources": {
                "limits": {},
                "reservations": {}
            }
        }
        
        if self.resources.cpu:
            service["deploy"]["resources"]["limits"]["cpus"] = self.resources.cpu
            service["deploy"]["resources"]["reservations"]["cpus"] = self.resources.cpu
            
        if self.resources.memory:
            # Convert to bytes for Docker Compose
            if self.resources.memory.endswith("Mi"):
                mem_bytes = int(self.resources.memory[:-2]) * 1024 * 1024
            elif self.resources.memory.endswith("Gi"):
                mem_bytes = int(self.resources.memory[:-2]) * 1024 * 1024 * 1024
            else:
                mem_bytes = int(self.resources.memory)
                
            service["deploy"]["resources"]["limits"]["memory"] = mem_bytes
            service["deploy"]["resources"]["reservations"]["memory"] = mem_bytes
            
        # Add health check
        if self.health_check.type != "none":
            service["healthcheck"] = {
                "interval": f"{self.health_check.period_seconds}s",
                "timeout": f"{self.health_check.timeout_seconds}s",
                "retries": self.health_check.failure_threshold,
                "start_period": f"{self.health_check.initial_delay_seconds}s"
            }
            
            if self.health_check.type == "http":
                service["healthcheck"]["test"] = [
                    "CMD-SHELL",
                    f"curl -f http://localhost:{self.health_check.port}{self.health_check.http_path or '/'} || exit 1"
                ]
            elif self.health_check.type == "tcp":
                service["healthcheck"]["test"] = [
                    "CMD-SHELL",
                    f"nc -z localhost {self.health_check.port} || exit 1"
                ]
            elif self.health_check.type == "exec":
                service["healthcheck"]["test"] = ["CMD"] + self.health_check.command
                
        # Add labels
        if self.labels:
            service["labels"] = self.labels
            
        # Add dependencies
        if self.depends_on:
            service["depends_on"] = self.depends_on
            
        return service
        
    def to_kubernetes_deployment(self) -> Dict[str, Any]:
        """Convert to Kubernetes Deployment spec"""
        # Prepare container spec
        container = {
            "name": self.name,
            "image": self.get_full_image()
        }
        
        # Add command and args
        if self.command:
            container["command"] = self.command
            
        if self.args:
            container["args"] = self.args
            
        # Add working directory
        if self.working_dir:
            container["workingDir"] = self.working_dir
            
        # Add environment variables
        if self.env_vars:
            container["env"] = [env_var.to_kubernetes_env() for env_var in self.env_vars]
            
        # Add resource requirements
        container["resources"] = self.resources.to_kubernetes_resources()
        
        # Add ports
        container["ports"] = self.network.to_kubernetes_container_ports()
        
        # Add health checks
        liveness_probe = self.health_check.to_kubernetes_probe()
        if liveness_probe:
            container["livenessProbe"] = liveness_probe
            container["readinessProbe"] = liveness_probe  # Use same for readiness
            
        # Prepare volumes and volume mounts
        volumes = []
        volume_mounts = []
        
        for volume in self.volumes:
            vol, vol_mount = volume.to_kubernetes_volume()
            volumes.append(vol)
            volume_mounts.append(vol_mount)
            
        if volume_mounts:
            container["volumeMounts"] = volume_mounts
            
        # Create deployment spec
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.name,
                "labels": {
                    "app": self.name
                }
            },
            "spec": {
                "replicas": 1,
                "selector": {
                    "matchLabels": {
                        "app": self.name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": self.name
                        }
                    },
                    "spec": {
                        "containers": [container]
                    }
                }
            }
        }
        
        # Add volumes to pod spec
        if volumes:
            deployment["spec"]["template"]["spec"]["volumes"] = volumes
            
        # Add restart policy
        if self.restart_policy == "always":
            deployment["spec"]["template"]["spec"]["restartPolicy"] = "Always"
        elif self.restart_policy == "on-failure":
            deployment["spec"]["template"]["spec"]["restartPolicy"] = "OnFailure"
        elif self.restart_policy == "no":
            deployment["spec"]["template"]["spec"]["restartPolicy"] = "Never"
            
        # Add labels
        if self.labels:
            for key, value in self.labels.items():
                deployment["metadata"]["labels"][key] = value
                deployment["spec"]["template"]["metadata"]["labels"][key] = value
                
        return deployment
        
    def to_kubernetes_manifests(self) -> List[Dict[str, Any]]:
        """Generate all Kubernetes manifests for this agent"""
        manifests = []
        
        # Add deployment
        manifests.append(self.to_kubernetes_deployment())
        
        # Add service if ports are defined
        if self.network.ports or self.network.expose_ports:
            manifests.append(self.network.to_kubernetes_service(self.name))
            
        # Add PVCs for persistent volumes
        for volume in self.volumes:
            if volume.persistent:
                pvc = volume.to_kubernetes_pvc()
                if pvc:
                    manifests.append(pvc)
                    
        return manifests

class AgentDeployer:
    """
    Handles deployment of agents to various container platforms.
    Provides methods for building, deploying, and managing agent containers.
    """
    def __init__(self, platform: ContainerPlatform = ContainerPlatform.DOCKER):
        self.platform = platform
        self.configs = {}  # name -> AgentContainerConfig
        self.deployed_agents = {}  # name -> container_id/pod_name
        
    def add_agent_config(self, config: AgentContainerConfig) -> None:
        """Add an agent configuration"""
        self.configs[config.name] = config
        
    def remove_agent_config(self, name: str) -> bool:
        """Remove an agent configuration"""
        if name in self.configs:
            del self.configs[name]
            return True
        return False
        
    def get_agent_config(self, name: str) -> Optional[AgentContainerConfig]:
        """Get an agent configuration by name"""
        return self.configs.get(name)
        
    async def build_agent_image(self, name: str, dockerfile_path: str, 
                              context_path: str = ".", build_args: Dict[str, str] = None) -> bool:
        """
        Build a Docker image for an agent
        
        Args:
            name: Agent name
            dockerfile_path: Path to Dockerfile
            context_path: Build context path
            build_args: Build arguments
            
        Returns:
            True if build was successful, False otherwise
        """
        if name not in self.configs:
            logger.error(f"Agent configuration not found: {name}")
            return False
            
        config = self.configs[name]
        image_name = config.get_full_image()
        
        # Prepare build command
        cmd = ["docker", "build", "-t", image_name]
        
        # Add build args
        if build_args:
            for key, value in build_args.items():
                cmd.extend(["--build-arg", f"{key}={value}"])
                
        # Add Dockerfile path
        cmd.extend(["-f", dockerfile_path])
        
        # Add context path
        cmd.append(context_path)
        
        # Run build command
        logger.info(f"Building image for agent {name}: {image_name}")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error building image for agent {name}:")
            logger.error(stderr.decode())
            return False
            
        logger.info(f"Successfully built image for agent {name}: {image_name}")
        return True
        
    async def deploy_agent(self, name: str) -> bool:
        """
        Deploy an agent
        
        Args:
            name: Agent name
            
        Returns:
            True if deployment was successful, False otherwise
        """
        if name not in self.configs:
            logger.error(f"Agent configuration not found: {name}")
            return False
            
        config = self.configs[name]
        
        if self.platform == ContainerPlatform.DOCKER:
            return await self._deploy_docker(config)
        elif self.platform == ContainerPlatform.KUBERNETES:
            return await self._deploy_kubernetes(config)
        elif self.platform == ContainerPlatform.AWS_ECS:
            return await self._deploy_aws_ecs(config)
        elif self.platform == ContainerPlatform.AZURE_CONTAINER_INSTANCES:
            return await self._deploy_azure_container_instances(config)
        elif self.platform == ContainerPlatform.GCP_CLOUD_RUN:
            return await self._deploy_gcp_cloud_run(config)
        else:
            logger.error(f"Unsupported platform: {self.platform}")
            return False
            
    async def _deploy_docker(self, config: AgentContainerConfig) -> bool:
        """Deploy an agent using Docker"""
        # Check if container already exists
        cmd = ["docker", "ps", "-a", "-q", "-f", f"name={config.name}"]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, _ = await process.communicate()
        container_id = stdout.decode().strip()
        
        if container_id:
            # Container exists, remove it
            logger.info(f"Container {config.name} already exists, removing it")
            await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", config.name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
        # Run the container
        run_cmd = config.to_docker_run_command()
        logger.info(f"Deploying agent {config.name} with Docker")
        logger.debug(f"Docker command: {' '.join(run_cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *run_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error deploying agent {config.name}:")
            logger.error(stderr.decode())
            return False
            
        container_id = stdout.decode().strip()
        self.deployed_agents[config.name] = container_id
        
        logger.info(f"Successfully deployed agent {config.name}: {container_id}")
        return True
        
    async def _deploy_kubernetes(self, config: AgentContainerConfig) -> bool:
        """Deploy an agent using Kubernetes"""
        # Generate Kubernetes manifests
        manifests = config.to_kubernetes_manifests()
        
        # Create a temporary file for the manifests
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp_file:
            # Write manifests to file
            for manifest in manifests:
                yaml.dump(manifest, temp_file)
                temp_file.write(b"---\n")
                
            temp_file_path = temp_file.name
            
        try:
            # Apply the manifests
            logger.info(f"Deploying agent {config.name} with Kubernetes")
            process = await asyncio.create_subprocess_exec(
                "kubectl", "apply", "-f", temp_file_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Error deploying agent {config.name} to Kubernetes:")
                logger.error(stderr.decode())
                return False
                
            # Get the pod name
            process = await asyncio.create_subprocess_exec(
                "kubectl", "get", "pods", "-l", f"app={config.name}", "-o", "jsonpath={.items[0].metadata.name}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, _ = await process.communicate()
            pod_name = stdout.decode().strip()
            
            if pod_name:
                self.deployed_agents[config.name] = pod_name
                logger.info(f"Successfully deployed agent {config.name} to Kubernetes: {pod_name}")
                return True
            else:
                logger.warning(f"Deployed agent {config.name} to Kubernetes, but couldn't get pod name")
                return True
                
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
            
    async def _deploy_aws_ecs(self, config: AgentContainerConfig) -> bool:
        """Deploy an agent using AWS ECS"""
        # This is a simplified implementation
        # In a real implementation, we would use the AWS SDK
        
        # Convert resources to ECS format
        resources = config.resources.to_cloud_resources(ContainerPlatform.AWS_ECS)
        
        # Create task definition
        task_def = {
            "family": config.name,
            "containerDefinitions": [
                {
                    "name": config.name,
                    "image": config.get_full_image(),
                    "cpu": int(resources["cpu"]) if resources["cpu"] else 256,
                    "memory": int(resources["memory"]) if resources["memory"] else 512,
                    "essential": True
                }
            ]
        }
        
        # Add command and args
        if config.command:
            task_def["containerDefinitions"][0]["command"] = config.command + (config.args or [])
            
        # Add environment variables
        if config.env_vars:
            task_def["containerDefinitions"][0]["environment"] = []
            for env_var in config.env_vars:
                if env_var.value:
                    task_def["containerDefinitions"][0]["environment"].append({
                        "name": env_var.name,
                        "value": env_var.value
                    })
                    
        # Add port mappings
        if config.network.ports:
            task_def["containerDefinitions"][0]["portMappings"] = []
            for container_port, host_port in config.network.ports:
                task_def["containerDefinitions"][0]["portMappings"].append({
                    "containerPort": container_port,
                    "hostPort": host_port
                })
                
        # Write task definition to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_file.write(json.dumps(task_def).encode())
            temp_file_path = temp_file.name
            
        try:
            # Register task definition
            logger.info(f"Registering ECS task definition for agent {config.name}")
            process = await asyncio.create_subprocess_exec(
                "aws", "ecs", "register-task-definition", "--cli-input-json", f"file://{temp_file_path}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Error registering ECS task definition for agent {config.name}:")
                logger.error(stderr.decode())
                return False
                
            # Run the task
            logger.info(f"Running ECS task for agent {config.name}")
            process = await asyncio.create_subprocess_exec(
                "aws", "ecs", "run-task",
                "--cluster", "default",  # Use default cluster
                "--task-definition", config.name,
                "--count", "1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Error running ECS task for agent {config.name}:")
                logger.error(stderr.decode())
                return False
                
            # Parse task ARN from response
            response = json.loads(stdout.decode())
            if "tasks" in response and response["tasks"]:
                task_arn = response["tasks"][0]["taskArn"]
                self.deployed_agents[config.name] = task_arn
                logger.info(f"Successfully deployed agent {config.name} to AWS ECS: {task_arn}")
                return True
            else:
                logger.error(f"Failed to get task ARN for agent {config.name}")
                return False
                
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
            
    async def _deploy_azure_container_instances(self, config: AgentContainerConfig) -> bool:
        """Deploy an agent using Azure Container Instances"""
        # This is a simplified implementation
        # In a real implementation, we would use the Azure SDK
        
        # Convert resources to Azure format
        resources = config.resources.to_cloud_resources(ContainerPlatform.AZURE_CONTAINER_INSTANCES)
        
        # Create container group definition
        container_group = {
            "name": config.name,
            "location": "eastus",  # Default location
            "properties": {
                "containers": [
                    {
                        "name": config.name,
                        "properties": {
                            "image": config.get_full_image(),
                            "resources": {
                                "requests": {
                                    "cpu": float(resources["cpu"]) if resources["cpu"] else 1.0,
                                    "memoryInGB": resources["memoryInGB"] if "memoryInGB" in resources else 1.5
                                }
                            }
                        }
                    }
                ],
                "osType": "Linux",
                "restartPolicy": "Always"
            }
        }
        
        # Add command and args
        if config.command:
            container_group["properties"]["containers"][0]["properties"]["command"] = config.command + (config.args or [])
            
        # Add environment variables
        if config.env_vars:
            container_group["properties"]["containers"][0]["properties"]["environmentVariables"] = []
            for env_var in config.env_vars:
                if env_var.value:
                    container_group["properties"]["containers"][0]["properties"]["environmentVariables"].append({
                        "name": env_var.name,
                        "value": env_var.value
                    })
                    
        # Add port mappings
        if config.network.ports:
            container_group["properties"]["ipAddress"] = {
                "type": "Public",
                "ports": []
            }
            
            for container_port, _ in config.network.ports:
                container_group["properties"]["ipAddress"]["ports"].append({
                    "port": container_port,
                    "protocol": "TCP"
                })
                
            container_group["properties"]["containers"][0]["properties"]["ports"] = []
            for container_port, _ in config.network.ports:
                container_group["properties"]["containers"][0]["properties"]["ports"].append({
                    "port": container_port
                })
                
        # Write container group definition to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_file.write(json.dumps(container_group).encode())
            temp_file_path = temp_file.name
            
        try:
            # Create container group
            logger.info(f"Creating Azure Container Instance for agent {config.name}")
            process = await asyncio.create_subprocess_exec(
                "az", "container", "create", "--resource-group", "myResourceGroup", "--file", temp_file_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Error creating Azure Container Instance for agent {config.name}:")
                logger.error(stderr.decode())
                return False
                
            # Parse container group ID from response
            response = json.loads(stdout.decode())
            if "id" in response:
                container_group_id = response["id"]
                self.deployed_agents[config.name] = container_group_id
                logger.info(f"Successfully deployed agent {config.name} to Azure Container Instances: {container_group_id}")
                return True
            else:
                logger.error(f"Failed to get container group ID for agent {config.name}")
                return False
                
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
            
    async def _deploy_gcp_cloud_run(self, config: AgentContainerConfig) -> bool:
        """Deploy an agent using GCP Cloud Run"""
        # This is a simplified implementation
        # In a real implementation, we would use the GCP SDK
        
        # Convert resources to GCP format
        resources = config.resources.to_cloud_resources(ContainerPlatform.GCP_CLOUD_RUN)
        
        # Create Cloud Run service
        service_name = config.name.replace("_", "-").lower()  # Cloud Run service names must be lowercase and can't contain underscores
        
        # Prepare command
        cmd = [
            "gcloud", "run", "deploy", service_name,
            "--image", config.get_full_image(),
            "--platform", "managed",
            "--region", "us-central1",  # Default region
            "--allow-unauthenticated"  # Public access
        ]
        
        # Add CPU and memory limits
        if resources.get("cpu"):
            cmd.extend(["--cpu", resources["cpu"]])
            
        if resources.get("memory"):
            cmd.extend(["--memory", resources["memory"]])
            
        # Add environment variables
        for env_var in config.env_vars:
            if env_var.value:
                cmd.extend(["--set-env-vars", f"{env_var.name}={env_var.value}"])
                
        # Add port
        if config.network.ports:
            container_port, _ = config.network.ports[0]  # Cloud Run only supports one port
            cmd.extend(["--port", str(container_port)])
            
        # Deploy the service
        logger.info(f"Deploying agent {config.name} to GCP Cloud Run")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error deploying agent {config.name} to GCP Cloud Run:")
            logger.error(stderr.decode())
            return False
            
        # Get the service URL
        process = await asyncio.create_subprocess_exec(
            "gcloud", "run", "services", "describe", service_name,
            "--platform", "managed",
            "--region", "us-central1",
            "--format", "value(status.url)",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, _ = await process.communicate()
        service_url = stdout.decode().strip()
        
        if service_url:
            self.deployed_agents[config.name] = service_url
            logger.info(f"Successfully deployed agent {config.name} to GCP Cloud Run: {service_url}")
            return True
        else:
            logger.warning(f"Deployed agent {config.name} to GCP Cloud Run, but couldn't get service URL")
            return True
            
    async def stop_agent(self, name: str) -> bool:
        """
        Stop a deployed agent
        
        Args:
            name: Agent name
            
        Returns:
            True if stop was successful, False otherwise
        """
        if name not in self.configs:
            logger.error(f"Agent configuration not found: {name}")
            return False
            
        if name not in self.deployed_agents:
            logger.warning(f"Agent {name} is not deployed")
            return False
            
        if self.platform == ContainerPlatform.DOCKER:
            return await self._stop_docker(name)
        elif self.platform == ContainerPlatform.KUBERNETES:
            return await self._stop_kubernetes(name)
        elif self.platform == ContainerPlatform.AWS_ECS:
            return await self._stop_aws_ecs(name)
        elif self.platform == ContainerPlatform.AZURE_CONTAINER_INSTANCES:
            return await self._stop_azure_container_instances(name)
        elif self.platform == ContainerPlatform.GCP_CLOUD_RUN:
            return await self._stop_gcp_cloud_run(name)
        else:
            logger.error(f"Unsupported platform: {self.platform}")
            return False
            
    async def _stop_docker(self, name: str) -> bool:
        """Stop a Docker container"""
        container_id = self.deployed_agents.get(name)
        
        logger.info(f"Stopping Docker container for agent {name}: {container_id}")
        process = await asyncio.create_subprocess_exec(
            "docker", "stop", container_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        _, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error stopping Docker container for agent {name}:")
            logger.error(stderr.decode())
            return False
            
        # Remove the container
        process = await asyncio.create_subprocess_exec(
            "docker", "rm", container_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        await process.communicate()
        
        del self.deployed_agents[name]
        logger.info(f"Successfully stopped agent {name}")
        return True
        
    async def _stop_kubernetes(self, name: str) -> bool:
        """Stop a Kubernetes deployment"""
        logger.info(f"Stopping Kubernetes deployment for agent {name}")
        process = await asyncio.create_subprocess_exec(
            "kubectl", "delete", "deployment", name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        _, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error stopping Kubernetes deployment for agent {name}:")
            logger.error(stderr.decode())
            return False
            
        # Delete the service if it exists
        process = await asyncio.create_subprocess_exec(
            "kubectl", "delete", "service", name, "--ignore-not-found",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        await process.communicate()
        
        del self.deployed_agents[name]
        logger.info(f"Successfully stopped agent {name}")
        return True
        
    async def _stop_aws_ecs(self, name: str) -> bool:
        """Stop an AWS ECS task"""
        task_arn = self.deployed_agents.get(name)
        
        logger.info(f"Stopping AWS ECS task for agent {name}: {task_arn}")
        process = await asyncio.create_subprocess_exec(
            "aws", "ecs", "stop-task", "--task", task_arn,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        _, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error stopping AWS ECS task for agent {name}:")
            logger.error(stderr.decode())
            return False
            
        del self.deployed_agents[name]
        logger.info(f"Successfully stopped agent {name}")
        return True
        
    async def _stop_azure_container_instances(self, name: str) -> bool:
        """Stop an Azure Container Instance"""
        logger.info(f"Stopping Azure Container Instance for agent {name}")
        process = await asyncio.create_subprocess_exec(
            "az", "container", "delete", "--name", name, "--resource-group", "myResourceGroup", "--yes",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        _, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error stopping Azure Container Instance for agent {name}:")
            logger.error(stderr.decode())
            return False
            
        del self.deployed_agents[name]
        logger.info(f"Successfully stopped agent {name}")
        return True
        
    async def _stop_gcp_cloud_run(self, name: str) -> bool:
        """Stop a GCP Cloud Run service"""
        service_name = name.replace("_", "-").lower()
        
        logger.info(f"Stopping GCP Cloud Run service for agent {name}")
        process = await asyncio.create_subprocess_exec(
            "gcloud", "run", "services", "delete", service_name,
            "--platform", "managed",
            "--region", "us-central1",
            "--quiet",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        _, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error stopping GCP Cloud Run service for agent {name}:")
            logger.error(stderr.decode())
            return False
            
        del self.deployed_agents[name]
        logger.info(f"Successfully stopped agent {name}")
        return True
        
    async def get_agent_status(self, name: str) -> Dict[str, Any]:
        """
        Get the status of a deployed agent
        
        Args:
            name: Agent name
            
        Returns:
            Dictionary with status information
        """
        if name not in self.configs:
            return {"error": f"Agent configuration not found: {name}"}
            
        if name not in self.deployed_agents:
            return {"status": "not_deployed", "name": name}
            
        if self.platform == ContainerPlatform.DOCKER:
            return await self._get_docker_status(name)
        elif self.platform == ContainerPlatform.KUBERNETES:
            return await self._get_kubernetes_status(name)
        elif self.platform == ContainerPlatform.AWS_ECS:
            return await self._get_aws_ecs_status(name)
        elif self.platform == ContainerPlatform.AZURE_CONTAINER_INSTANCES:
            return await self._get_azure_container_instances_status(name)
        elif self.platform == ContainerPlatform.GCP_CLOUD_RUN:
            return await self._get_gcp_cloud_run_status(name)
        else:
            return {"error": f"Unsupported platform: {self.platform}"}
            
    async def _get_docker_status(self, name: str) -> Dict[str, Any]:
        """Get Docker container status"""
        container_id = self.deployed_agents.get(name)
        
        process = await asyncio.create_subprocess_exec(
            "docker", "inspect", container_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error getting Docker container status for agent {name}:")
            logger.error(stderr.decode())
            return {"status": "error", "name": name, "error": stderr.decode()}
            
        try:
            inspect_data = json.loads(stdout.decode())
            if not inspect_data:
                return {"status": "not_found", "name": name}
                
            container_data = inspect_data[0]
            state = container_data.get("State", {})
            
            return {
                "status": state.get("Status", "unknown"),
                "name": name,
                "container_id": container_id,
                "created": container_data.get("Created"),
                "started_at": state.get("StartedAt"),
                "finished_at": state.get("FinishedAt"),
                "health": state.get("Health", {}).get("Status") if "Health" in state else None,
                "exit_code": state.get("ExitCode"),
                "error": state.get("Error")
            }
        except json.JSONDecodeError:
            return {"status": "error", "name": name, "error": "Invalid JSON response"}
            
    async def _get_kubernetes_status(self, name: str) -> Dict[str, Any]:
        """Get Kubernetes deployment status"""
        process = await asyncio.create_subprocess_exec(
            "kubectl", "get", "deployment", name, "-o", "json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error getting Kubernetes deployment status for agent {name}:")
            logger.error(stderr.decode())
            return {"status": "error", "name": name, "error": stderr.decode()}
            
        try:
            deployment_data = json.loads(stdout.decode())
            status = deployment_data.get("status", {})
            
            # Get pod status
            process = await asyncio.create_subprocess_exec(
                "kubectl", "get", "pods", "-l", f"app={name}", "-o", "json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, _ = await process.communicate()
            pods_data = json.loads(stdout.decode())
            pods = pods_data.get("items", [])
            
            pod_statuses = []
            for pod in pods:
                pod_status = pod.get("status", {})
                container_statuses = pod_status.get("containerStatuses", [])
                
                for container in container_statuses:
                    if container.get("name") == name:
                        state = container.get("state", {})
                        state_type = next(iter(state.keys())) if state else "unknown"
                        
                        pod_statuses.append({
                            "pod_name": pod.get("metadata", {}).get("name"),
                            "state": state_type,
                            "ready": container.get("ready", False),
                            "restart_count": container.get("restartCount", 0),
                            "started": container.get("started", False)
                        })
                        
            return {
                "status": "running" if status.get("availableReplicas", 0) > 0 else "not_running",
                "name": name,
                "replicas": status.get("replicas", 0),
                "available_replicas": status.get("availableReplicas", 0),
                "ready_replicas": status.get("readyReplicas", 0),
                "updated_replicas": status.get("updatedReplicas", 0),
                "pods": pod_statuses
            }
        except json.JSONDecodeError:
            return {"status": "error", "name": name, "error": "Invalid JSON response"}
            
    async def _get_aws_ecs_status(self, name: str) -> Dict[str, Any]:
        """Get AWS ECS task status"""
        task_arn = self.deployed_agents.get(name)
        
        process = await asyncio.create_subprocess_exec(
            "aws", "ecs", "describe-tasks", "--tasks", task_arn, "--cluster", "default",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error getting AWS ECS task status for agent {name}:")
            logger.error(stderr.decode())
            return {"status": "error", "name": name, "error": stderr.decode()}
            
        try:
            task_data = json.loads(stdout.decode())
            tasks = task_data.get("tasks", [])
            
            if not tasks:
                return {"status": "not_found", "name": name}
                
            task = tasks[0]
            last_status = task.get("lastStatus")
            
            container_statuses = []
            for container in task.get("containers", []):
                container_statuses.append({
                    "name": container.get("name"),
                    "status": container.get("lastStatus"),
                    "exit_code": container.get("exitCode"),
                    "health_status": container.get("healthStatus")
                })
                
            return {
                "status": last_status.lower() if last_status else "unknown",
                "name": name,
                "task_arn": task_arn,
                "created_at": task.get("createdAt"),
                "started_at": task.get("startedAt"),
                "stopped_at": task.get("stoppedAt"),
                "containers": container_statuses
            }
        except json.JSONDecodeError:
            return {"status": "error", "name": name, "error": "Invalid JSON response"}
            
    async def _get_azure_container_instances_status(self, name: str) -> Dict[str, Any]:
        """Get Azure Container Instance status"""
        process = await asyncio.create_subprocess_exec(
            "az", "container", "show", "--name", name, "--resource-group", "myResourceGroup",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error getting Azure Container Instance status for agent {name}:")
            logger.error(stderr.decode())
            return {"status": "error", "name": name, "error": stderr.decode()}
            
        try:
            container_data = json.loads(stdout.decode())
            
            containers = container_data.get("containers", [])
            container_statuses = []
            
            for container in containers:
                container_statuses.append({
                    "name": container.get("name"),
                    "state": container.get("instanceView", {}).get("currentState", {}).get("state"),
                    "exit_code": container.get("instanceView", {}).get("currentState", {}).get("exitCode"),
                    "start_time": container.get("instanceView", {}).get("currentState", {}).get("startTime"),
                    "finish_time": container.get("instanceView", {}).get("currentState", {}).get("finishTime")
                })
                
            return {
                "status": container_data.get("instanceView", {}).get("state", "unknown").lower(),
                "name": name,
                "ip_address": container_data.get("ipAddress", {}).get("ip"),
                "containers": container_statuses
            }
        except json.JSONDecodeError:
            return {"status": "error", "name": name, "error": "Invalid JSON response"}
            
    async def _get_gcp_cloud_run_status(self, name: str) -> Dict[str, Any]:
        """Get GCP Cloud Run service status"""
        service_name = name.replace("_", "-").lower()
        
        process = await asyncio.create_subprocess_exec(
            "gcloud", "run", "services", "describe", service_name,
            "--platform", "managed",
            "--region", "us-central1",
            "--format", "json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error getting GCP Cloud Run service status for agent {name}:")
            logger.error(stderr.decode())
            return {"status": "error", "name": name, "error": stderr.decode()}
            
        try:
            service_data = json.loads(stdout.decode())
            
            return {
                "status": service_data.get("status", {}).get("conditions", [{}])[0].get("status", "unknown").lower(),
                "name": name,
                "url": service_data.get("status", {}).get("url"),
                "latest_created_revision": service_data.get("status", {}).get("latestCreatedRevisionName"),
                "latest_ready_revision": service_data.get("status", {}).get("latestReadyRevisionName"),
                "traffic": service_data.get("status", {}).get("traffic")
            }
        except json.JSONDecodeError:
            return {"status": "error", "name": name, "error": "Invalid JSON response"}
            
    def generate_docker_compose(self, output_file: str = "docker-compose.yml") -> bool:
        """
        Generate a Docker Compose file for all agent configurations
        
        Args:
            output_file: Output file path
            
        Returns:
            True if generation was successful, False otherwise
        """
        if not self.configs:
            logger.error("No agent configurations found")
            return False
            
        compose = {
            "version": "3",
            "services": {},
            "volumes": {}
        }
        
        # Add services
        for name, config in self.configs.items():
            compose["services"][name] = config.to_docker_compose_service()
            
        # Add volumes
        for config in self.configs.values():
            for volume in config.volumes:
                if volume.persistent and not volume.host_path:
                    compose["volumes"][volume.name] = {"driver": "local"}
                    
        # Write to file
        try:
            with open(output_file, "w") as f:
                yaml.dump(compose, f, default_flow_style=False)
                
            logger.info(f"Generated Docker Compose file: {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error generating Docker Compose file: {e}")
            return False
            
    def generate_kubernetes_manifests(self, output_dir: str = "kubernetes") -> bool:
        """
        Generate Kubernetes manifests for all agent configurations
        
        Args:
            output_dir: Output directory
            
        Returns:
            True if generation was successful, False otherwise
        """
        if not self.configs:
            logger.error("No agent configurations found")
            return False
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        for name, config in self.configs.items():
            # Generate manifests
            manifests = config.to_kubernetes_manifests()
            
            # Write to file
            try:
                output_file = os.path.join(output_dir, f"{name}.yaml")
                with open(output_file, "w") as f:
                    for manifest in manifests:
                        yaml.dump(manifest, f, default_flow_style=False)
                        f.write("---\n")
                        
                logger.info(f"Generated Kubernetes manifests for agent {name}: {output_file}")
            except Exception as e:
                logger.error(f"Error generating Kubernetes manifests for agent {name}: {e}")
                return False
                
        return True
        
    def generate_cloud_templates(self, platform: ContainerPlatform, 
                               output_dir: str = "cloud_templates") -> bool:
        """
        Generate cloud-specific templates for all agent configurations
        
        Args:
            platform: Cloud platform
            output_dir: Output directory
            
        Returns:
            True if generation was successful, False otherwise
        """
        if not self.configs:
            logger.error("No agent configurations found")
            return False
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        if platform == ContainerPlatform.AWS_ECS:
            return self._generate_aws_ecs_templates(output_dir)
        elif platform == ContainerPlatform.AZURE_CONTAINER_INSTANCES:
            return self._generate_azure_container_instances_templates(output_dir)
        elif platform == ContainerPlatform.GCP_CLOUD_RUN:
            return self._generate_gcp_cloud_run_templates(output_dir)
        else:
            logger.error(f"Unsupported platform for template generation: {platform}")
            return False
            
    def _generate_aws_ecs_templates(self, output_dir: str) -> bool:
        """Generate AWS ECS templates"""
        for name, config in self.configs.items():
            # Convert resources to ECS format
            resources = config.resources.to_cloud_resources(ContainerPlatform.AWS_ECS)
            
            # Create task definition
            task_def = {
                "family": name,
                "containerDefinitions": [
                    {
                        "name": name,
                        "image": config.get_full_image(),
                        "cpu": int(resources["cpu"]) if resources["cpu"] else 256,
                        "memory": int(resources["memory"]) if resources["memory"] else 512,
                        "essential": True
                    }
                ]
            }
            
            # Add command and args
            if config.command:
                task_def["containerDefinitions"][0]["command"] = config.command + (config.args or [])
                
            # Add environment variables
            if config.env_vars:
                task_def["containerDefinitions"][0]["environment"] = []
                for env_var in config.env_vars:
                    if env_var.value:
                        task_def["containerDefinitions"][0]["environment"].append({
                            "name": env_var.name,
                            "value": env_var.value
                        })
                        
            # Add port mappings
            if config.network.ports:
                task_def["containerDefinitions"][0]["portMappings"] = []
                for container_port, host_port in config.network.ports:
                    task_def["containerDefinitions"][0]["portMappings"].append({
                        "containerPort": container_port,
                        "hostPort": host_port
                    })
                    
            # Write to file
            try:
                output_file = os.path.join(output_dir, f"{name}-ecs-task-definition.json")
                with open(output_file, "w") as f:
                    json.dump(task_def, f, indent=2)
                    
                logger.info(f"Generated AWS ECS task definition for agent {name}: {output_file}")
            except Exception as e:
                logger.error(f"Error generating AWS ECS task definition for agent {name}: {e}")
                return False
                
        return True
        
    def _generate_azure_container_instances_templates(self, output_dir: str) -> bool:
        """Generate Azure Container Instances templates"""
        for name, config in self.configs.items():
            # Convert resources to Azure format
            resources = config.resources.to_cloud_resources(ContainerPlatform.AZURE_CONTAINER_INSTANCES)
            
            # Create container group definition
            container_group = {
                "name": name,
                "location": "eastus",  # Default location
                "properties": {
                    "containers": [
                        {
                            "name": name,
                            "properties": {
                                "image": config.get_full_image(),
                                "resources": {
                                    "requests": {
                                        "cpu": float(resources["cpu"]) if resources["cpu"] else 1.0,
                                        "memoryInGB": resources["memoryInGB"] if "memoryInGB" in resources else 1.5
                                    }
                                }
                            }
                        }
                    ],
                    "osType": "Linux",
                    "restartPolicy": "Always"
                }
            }
            
            # Add command and args
            if config.command:
                container_group["properties"]["containers"][0]["properties"]["command"] = config.command + (config.args or [])
                
            # Add environment variables
            if config.env_vars:
                container_group["properties"]["containers"][0]["properties"]["environmentVariables"] = []
                for env_var in config.env_vars:
                    if env_var.value:
                        container_group["properties"]["containers"][0]["properties"]["environmentVariables"].append({
                            "name": env_var.name,
                            "value": env_var.value
                        })
                        
            # Add port mappings
            if config.network.ports:
                container_group["properties"]["ipAddress"] = {
                    "type": "Public",
                    "ports": []
                }
                
                for container_port, _ in config.network.ports:
                    container_group["properties"]["ipAddress"]["ports"].append({
                        "port": container_port,
                        "protocol": "TCP"
                    })
                    
                container_group["properties"]["containers"][0]["properties"]["ports"] = []
                for container_port, _ in config.network.ports:
                    container_group["properties"]["containers"][0]["properties"]["ports"].append({
                        "port": container_port
                    })
                    
            # Write to file
            try:
                output_file = os.path.join(output_dir, f"{name}-azure-container-group.json")
                with open(output_file, "w") as f:
                    json.dump(container_group, f, indent=2)
                    
                logger.info(f"Generated Azure Container Instance template for agent {name}: {output_file}")
            except Exception as e:
                logger.error(f"Error generating Azure Container Instance template for agent {name}: {e}")
                return False
                
        return True
        
    def _generate_gcp_cloud_run_templates(self, output_dir: str) -> bool:
        """Generate GCP Cloud Run templates"""
        for name, config in self.configs.items():
            # Convert resources to GCP format
            resources = config.resources.to_cloud_resources(ContainerPlatform.GCP_CLOUD_RUN)
            
            # Create Cloud Run service
            service_name = name.replace("_", "-").lower()
            
            service = {
                "apiVersion": "serving.knative.dev/v1",
                "kind": "Service",
                "metadata": {
                    "name": service_name
                },
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "image": config.get_full_image()
                                }
                            ]
                        }
                    }
                }
            }
            
            # Add command and args
            if config.command:
                service["spec"]["template"]["spec"]["containers"][0]["command"] = config.command
                
            if config.args:
                service["spec"]["template"]["spec"]["containers"][0]["args"] = config.args
                
            # Add environment variables
            if config.env_vars:
                service["spec"]["template"]["spec"]["containers"][0]["env"] = []
                for env_var in config.env_vars:
                    if env_var.value:
                        service["spec"]["template"]["spec"]["containers"][0]["env"].append({
                            "name": env_var.name,
                            "value": env_var.value
                        })
                        
            # Add port
            if config.network.ports:
                container_port, _ = config.network.ports[0]  # Cloud Run only supports one port
                service["spec"]["template"]["spec"]["containers"][0]["ports"] = [
                    {
                        "containerPort": container_port
                    }
                ]
                
            # Add resource limits
            if resources.get("cpu") or resources.get("memory"):
                service["spec"]["template"]["spec"]["containers"][0]["resources"] = {
                    "limits": {}
                }
                
                if resources.get("cpu"):
                    service["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["cpu"] = resources["cpu"]
                    
                if resources.get("memory"):
                    service["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["memory"] = resources["memory"]
                    
            # Write to file
            try:
                output_file = os.path.join(output_dir, f"{name}-cloud-run-service.yaml")
                with open(output_file, "w") as f:
                    yaml.dump(service, f, default_flow_style=False)
                    
                logger.info(f"Generated GCP Cloud Run service template for agent {name}: {output_file}")
            except Exception as e:
                logger.error(f"Error generating GCP Cloud Run service template for agent {name}: {e}")
                return False
                
        return True

# Example usage
async def example_usage():
    # Create a deployer for Docker
    deployer = AgentDeployer(ContainerPlatform.DOCKER)
    
    # Create an agent configuration
    agent_config = AgentContainerConfig(
        name="example-agent",
        image="python",
        tag="3.9-slim",
        command=["python", "-c", "import http.server; http.server.test(HandlerClass=http.server.SimpleHTTPRequestHandler, port=8000)"],
        env_vars=[
            EnvironmentVariable(name="PYTHONUNBUFFERED", value="1")
        ],
        network=NetworkConfig(
            ports=[(8000, 8080)]  # Container port 8000, host port 8080
        ),
        resources=ResourceRequirements(
            cpu="0.5",
            memory="256Mi"
        ),
        health_check=HealthCheck(
            type="http",
            http_path="/",
            port=8000
        )
    )
    
    # Add the agent configuration to the deployer
    deployer.add_agent_config(agent_config)
    
    # Generate Docker Compose file
    deployer.generate_docker_compose("example-docker-compose.yml")
    
    # Generate Kubernetes manifests
    deployer.generate_kubernetes_manifests("example-kubernetes")
    
    # Deploy the agent
    print("Deploying agent...")
    success = await deployer.deploy_agent("example-agent")
    
    if success:
        print("Agent deployed successfully!")
        
        # Get agent status
        print("Getting agent status...")
        status = await deployer.get_agent_status("example-agent")
        print(f"Agent status: {status}")
        
        # Wait for a moment
        print("Waiting for 10 seconds...")
        await asyncio.sleep(10)
        
        # Stop the agent
        print("Stopping agent...")
        await deployer.stop_agent("example-agent")
        print("Agent stopped.")
    else:
        print("Failed to deploy agent.")

if __name__ == "__main__":
    asyncio.run(example_usage())
