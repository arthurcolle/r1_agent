#!/usr/bin/env python3
"""
Code Generator Module

This module provides advanced code generation, transformation, and management capabilities.
It can generate code from specifications, refactor existing code, and manage code repositories.
"""

import os
import sys
import re
import json
import logging
import asyncio
import subprocess
import tempfile
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from pathlib import Path
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI
    HAVE_OPENAI = True
except ImportError:
    HAVE_OPENAI = False
    logger.warning("OpenAI package not installed. Some features will be limited.")

class CodeGenerator:
    """
    Advanced code generation and transformation capabilities.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "o3-mini"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key) if HAVE_OPENAI and self.api_key else None
        self.history = []
        self.templates = {}
        self.load_templates()
        
    def load_templates(self):
        """Load code templates from templates directory"""
        template_dir = Path(__file__).parent / "templates"
        if not template_dir.exists():
            template_dir.mkdir(parents=True)
            
        # Create some default templates if none exist
        if not list(template_dir.glob("*.template")):
            self._create_default_templates(template_dir)
            
        # Load all templates
        for template_file in template_dir.glob("*.template"):
            try:
                template_name = template_file.stem
                with open(template_file, "r") as f:
                    template_content = f.read()
                self.templates[template_name] = template_content
                logger.debug(f"Loaded template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")
                
    def _create_default_templates(self, template_dir: Path):
        """Create default templates"""
        # Python class template
        python_class = """# {{filename}}
# {{description}}
# Created: {{date}}

class {{classname}}:
    """{{description}}"""
    
    def __init__(self{{params}}):
        """Initialize the {{classname}} instance"""
        {{init_body}}
    
    def {{method_name}}(self{{method_params}}):
        """{{method_description}}"""
        {{method_body}}
        
if __name__ == "__main__":
    # Example usage
    {{example_usage}}
"""
        with open(template_dir / "python_class.template", "w") as f:
            f.write(python_class)
            
        # Python script template
        python_script = """#!/usr/bin/env python3
# {{filename}}
# {{description}}
# Created: {{date}}

import os
import sys
import argparse
from typing import List, Dict, Any, Optional

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="{{description}}")
    {{parser_args}}
    args = parser.parse_args()
    
    {{main_body}}
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""
        with open(template_dir / "python_script.template", "w") as f:
            f.write(python_script)
            
        # HTML template
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{title}}</title>
    <style>
        {{css}}
    </style>
</head>
<body>
    <header>
        <h1>{{title}}</h1>
    </header>
    
    <main>
        {{content}}
    </main>
    
    <footer>
        <p>{{footer}}</p>
    </footer>
    
    <script>
        {{javascript}}
    </script>
</body>
</html>
"""
        with open(template_dir / "html_page.template", "w") as f:
            f.write(html_template)
            
    async def generate_code(self, spec: str, language: str = "python", 
                          template: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate code based on a specification
        
        Args:
            spec: Specification for the code to generate
            language: Programming language to use
            template: Optional template name to use
            
        Returns:
            Dict containing the generated code and metadata
        """
        if not HAVE_OPENAI or not self.client:
            return {
                "success": False,
                "error": "OpenAI client not available"
            }
            
        try:
            # Use template if provided
            template_content = None
            if template and template in self.templates:
                template_content = self.templates[template]
                
            # Create a prompt for code generation
            prompt = f"""
            Generate {language} code based on this specification:
            {spec}
            
            Requirements:
            - The code should be complete and runnable
            - Include proper error handling
            - Follow best practices for {language}
            """
            
            if template_content:
                prompt += f"""
                Use this template:
                ```
                {template_content}
                ```
                
                Fill in the template placeholders with appropriate content.
                """
                
            prompt += "\nReturn ONLY the code without any explanations or markdown."
            
            # Use the OpenAI client to generate code
            messages = [
                {"role": "system", "content": f"You are an expert {language} developer."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2
            )
            
            generated_code = response.choices[0].message.content
            
            # Clean up the code (remove markdown code blocks if present)
            code = re.sub(r'^```.*\n|```$', '', generated_code, flags=re.MULTILINE).strip()
            
            # Record in history
            self.history.append({
                "type": "generation",
                "spec": spec,
                "language": language,
                "template": template,
                "code": code,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "success": True,
                "code": code,
                "language": language,
                "template": template
            }
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
            
    async def refactor_code(self, code: str, instructions: str, 
                          language: str = "python") -> Dict[str, Any]:
        """
        Refactor existing code based on instructions
        
        Args:
            code: Existing code to refactor
            instructions: Instructions for refactoring
            language: Programming language of the code
            
        Returns:
            Dict containing the refactored code and metadata
        """
        if not HAVE_OPENAI or not self.client:
            return {
                "success": False,
                "error": "OpenAI client not available"
            }
            
        try:
            # Create a prompt for code refactoring
            prompt = f"""
            Refactor this {language} code according to these instructions:
            
            INSTRUCTIONS:
            {instructions}
            
            CODE TO REFACTOR:
            ```{language}
            {code}
            ```
            
            Return ONLY the refactored code without any explanations or markdown.
            """
            
            # Use the OpenAI client to refactor code
            messages = [
                {"role": "system", "content": f"You are an expert {language} developer specializing in code refactoring."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2
            )
            
            refactored_code = response.choices[0].message.content
            
            # Clean up the code (remove markdown code blocks if present)
            code = re.sub(r'^```.*\n|```$', '', refactored_code, flags=re.MULTILINE).strip()
            
            # Record in history
            self.history.append({
                "type": "refactoring",
                "original_code": code,
                "instructions": instructions,
                "refactored_code": code,
                "language": language,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "success": True,
                "code": code,
                "language": language,
                "diff": self._generate_diff(code, refactored_code)
            }
        except Exception as e:
            logger.error(f"Error refactoring code: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def _generate_diff(self, original: str, modified: str) -> str:
        """Generate a unified diff between original and modified code"""
        import difflib
        
        # Split into lines
        original_lines = original.splitlines(True)
        modified_lines = modified.splitlines(True)
        
        # Generate diff
        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile='original',
            tofile='modified',
            n=3
        )
        
        return ''.join(diff)
            
    async def analyze_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Analyze code for quality, bugs, and improvement opportunities
        
        Args:
            code: Code to analyze
            language: Programming language of the code
            
        Returns:
            Dict containing the analysis results
        """
        if not HAVE_OPENAI or not self.client:
            return {
                "success": False,
                "error": "OpenAI client not available"
            }
            
        try:
            # Create a prompt for code analysis
            prompt = f"""
            Analyze this {language} code for quality, bugs, and improvement opportunities:
            
            ```{language}
            {code}
            ```
            
            Provide a detailed analysis including:
            1. Potential bugs or errors
            2. Code quality issues
            3. Performance concerns
            4. Security vulnerabilities
            5. Improvement suggestions
            
            Format your response as JSON with these sections.
            """
            
            # Use the OpenAI client to analyze code
            messages = [
                {"role": "system", "content": f"You are an expert {language} code reviewer."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2
            )
            
            analysis_text = response.choices[0].message.content
            
            # Extract JSON from the response
            try:
                # Try to parse the entire response as JSON
                analysis = json.loads(analysis_text)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from markdown
                json_match = re.search(r'```json\n(.*?)\n```', analysis_text, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group(1))
                else:
                    # If no JSON found, return the text as is
                    analysis = {"raw_analysis": analysis_text}
            
            # Record in history
            self.history.append({
                "type": "analysis",
                "code": code,
                "language": language,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "success": True,
                "analysis": analysis,
                "language": language
            }
        except Exception as e:
            logger.error(f"Error analyzing code: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    async def generate_from_template(self, template_name: str, 
                                   variables: Dict[str, str]) -> Dict[str, Any]:
        """
        Generate code from a template with variable substitution
        
        Args:
            template_name: Name of the template to use
            variables: Dictionary of variables to substitute in the template
            
        Returns:
            Dict containing the generated code and metadata
        """
        try:
            if template_name not in self.templates:
                return {
                    "success": False,
                    "error": f"Template '{template_name}' not found"
                }
                
            template = self.templates[template_name]
            
            # Add default variables
            variables.setdefault("date", datetime.now().strftime("%Y-%m-%d"))
            
            # Substitute variables
            code = template
            for key, value in variables.items():
                placeholder = f"{{{{{key}}}}}"
                code = code.replace(placeholder, value)
                
            # Record in history
            self.history.append({
                "type": "template",
                "template": template_name,
                "variables": variables,
                "code": code,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "success": True,
                "code": code,
                "template": template_name,
                "variables": variables
            }
        except Exception as e:
            logger.error(f"Error generating from template: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    async def execute_code(self, code: str, language: str = "python", 
                         timeout: float = 5.0) -> Dict[str, Any]:
        """
        Execute code in a sandbox and return the result
        
        Args:
            code: Code to execute
            language: Programming language of the code
            timeout: Maximum execution time in seconds
            
        Returns:
            Dict containing the execution results
        """
        try:
            if language != "python":
                return {
                    "success": False,
                    "error": f"Execution of {language} code is not supported"
                }
                
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp:
                temp_path = temp.name
                temp.write(code.encode())
                
            try:
                # Execute the code in a subprocess
                process = await asyncio.create_subprocess_exec(
                    sys.executable, temp_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                    
                    # Record in history
                    self.history.append({
                        "type": "execution",
                        "code": code,
                        "language": language,
                        "exit_code": process.returncode,
                        "stdout": stdout.decode(),
                        "stderr": stderr.decode(),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    return {
                        "success": process.returncode == 0,
                        "exit_code": process.returncode,
                        "stdout": stdout.decode(),
                        "stderr": stderr.decode()
                    }
                except asyncio.TimeoutError:
                    # Kill the process if it times out
                    process.kill()
                    await process.wait()
                    
                    # Record in history
                    self.history.append({
                        "type": "execution",
                        "code": code,
                        "language": language,
                        "timeout": True,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    return {
                        "success": False,
                        "error": f"Execution timed out after {timeout} seconds"
                    }
            finally:
                # Clean up the temporary file
                os.unlink(temp_path)
                
        except Exception as e:
            logger.error(f"Error executing code: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def save_code_to_file(self, code: str, file_path: str, 
                         make_executable: bool = False) -> Dict[str, Any]:
        """
        Save generated code to a file
        
        Args:
            code: Code to save
            file_path: Path to save the code to
            make_executable: Whether to make the file executable
            
        Returns:
            Dict containing the result
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Write code to file
            with open(file_path, "w") as f:
                f.write(code)
                
            # Make executable if requested
            if make_executable:
                os.chmod(file_path, 0o755)
                
            # Record in history
            self.history.append({
                "type": "save",
                "code": code,
                "file_path": file_path,
                "executable": make_executable,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "success": True,
                "file_path": file_path,
                "size": len(code),
                "executable": make_executable
            }
        except Exception as e:
            logger.error(f"Error saving code to file: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the recent code generation history"""
        return self.history[-limit:] if limit > 0 else self.history.copy()
        
    def clear_history(self) -> None:
        """Clear the code generation history"""
        self.history = []
        
    def add_template(self, name: str, content: str) -> bool:
        """
        Add a new template
        
        Args:
            name: Template name
            content: Template content
            
        Returns:
            bool: Success status
        """
        try:
            # Save template to file
            template_dir = Path(__file__).parent / "templates"
            template_dir.mkdir(parents=True, exist_ok=True)
            
            template_path = template_dir / f"{name}.template"
            with open(template_path, "w") as f:
                f.write(content)
                
            # Add to templates dictionary
            self.templates[name] = content
            
            logger.info(f"Added template: {name}")
            return True
        except Exception as e:
            logger.error(f"Error adding template: {e}")
            return False
            
    def remove_template(self, name: str) -> bool:
        """
        Remove a template
        
        Args:
            name: Template name
            
        Returns:
            bool: Success status
        """
        try:
            if name not in self.templates:
                return False
                
            # Remove from templates dictionary
            del self.templates[name]
            
            # Remove template file
            template_dir = Path(__file__).parent / "templates"
            template_path = template_dir / f"{name}.template"
            if template_path.exists():
                template_path.unlink()
                
            logger.info(f"Removed template: {name}")
            return True
        except Exception as e:
            logger.error(f"Error removing template: {e}")
            return False

class CodeRepository:
    """
    Manages a code repository with version control integration.
    """
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.is_git_repo = self._check_git_repo()
        
    def _check_git_repo(self) -> bool:
        """Check if the repository is a Git repository"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.returncode == 0 and result.stdout.strip() == "true"
        except Exception:
            return False
            
    def initialize_git(self) -> bool:
        """Initialize a Git repository if not already initialized"""
        if self.is_git_repo:
            logger.info(f"Repository at {self.repo_path} is already a Git repository")
            return True
            
        try:
            result = subprocess.run(
                ["git", "init"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.is_git_repo = True
                logger.info(f"Initialized Git repository at {self.repo_path}")
                return True
            else:
                logger.error(f"Failed to initialize Git repository: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error initializing Git repository: {e}")
            return False
            
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the repository"""
        if not self.is_git_repo:
            return {
                "is_git_repo": False,
                "error": "Not a Git repository"
            }
            
        try:
            # Get Git status
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            # Get current branch
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            # Parse status output
            status_lines = status_result.stdout.splitlines()
            modified_files = []
            untracked_files = []
            staged_files = []
            
            for line in status_lines:
                if line.startswith("??"):
                    untracked_files.append(line[3:])
                elif line.startswith(" M"):
                    modified_files.append(line[3:])
                elif line.startswith("M"):
                    staged_files.append(line[2:])
                elif line.startswith("A"):
                    staged_files.append(line[2:])
            
            return {
                "is_git_repo": True,
                "branch": branch_result.stdout.strip(),
                "modified_files": modified_files,
                "untracked_files": untracked_files,
                "staged_files": staged_files,
                "clean": len(status_lines) == 0
            }
        except Exception as e:
            logger.error(f"Error getting repository status: {e}")
            return {
                "is_git_repo": True,
                "error": str(e)
            }
            
    def commit_changes(self, message: str, files: Optional[List[str]] = None) -> bool:
        """
        Commit changes to the repository
        
        Args:
            message: Commit message
            files: Optional list of files to commit (commits all changes if None)
            
        Returns:
            bool: Success status
        """
        if not self.is_git_repo:
            logger.error("Not a Git repository")
            return False
            
        try:
            # Add files
            if files:
                for file in files:
                    add_result = subprocess.run(
                        ["git", "add", file],
                        cwd=self.repo_path,
                        capture_output=True,
                        text=True
                    )
                    if add_result.returncode != 0:
                        logger.error(f"Failed to add file {file}: {add_result.stderr}")
                        return False
            else:
                # Add all changes
                add_result = subprocess.run(
                    ["git", "add", "."],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True
                )
                if add_result.returncode != 0:
                    logger.error(f"Failed to add changes: {add_result.stderr}")
                    return False
            
            # Commit
            commit_result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if commit_result.returncode == 0:
                logger.info(f"Committed changes: {message}")
                return True
            else:
                logger.error(f"Failed to commit changes: {commit_result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error committing changes: {e}")
            return False
            
    def create_branch(self, branch_name: str) -> bool:
        """
        Create a new branch
        
        Args:
            branch_name: Name of the branch to create
            
        Returns:
            bool: Success status
        """
        if not self.is_git_repo:
            logger.error("Not a Git repository")
            return False
            
        try:
            result = subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Created and switched to branch: {branch_name}")
                return True
            else:
                logger.error(f"Failed to create branch: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error creating branch: {e}")
            return False
            
    def switch_branch(self, branch_name: str) -> bool:
        """
        Switch to a different branch
        
        Args:
            branch_name: Name of the branch to switch to
            
        Returns:
            bool: Success status
        """
        if not self.is_git_repo:
            logger.error("Not a Git repository")
            return False
            
        try:
            result = subprocess.run(
                ["git", "checkout", branch_name],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Switched to branch: {branch_name}")
                return True
            else:
                logger.error(f"Failed to switch branch: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error switching branch: {e}")
            return False
            
    def get_file_history(self, file_path: str, max_entries: int = 10) -> List[Dict[str, Any]]:
        """
        Get the commit history for a file
        
        Args:
            file_path: Path to the file
            max_entries: Maximum number of history entries to return
            
        Returns:
            List of commit information dictionaries
        """
        if not self.is_git_repo:
            logger.error("Not a Git repository")
            return []
            
        try:
            result = subprocess.run(
                ["git", "log", f"-{max_entries}", "--pretty=format:%H|%an|%ad|%s", "--", file_path],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                history = []
                for line in result.stdout.splitlines():
                    parts = line.split("|", 3)
                    if len(parts) == 4:
                        commit_hash, author, date, message = parts
                        history.append({
                            "hash": commit_hash,
                            "author": author,
                            "date": date,
                            "message": message
                        })
                return history
            else:
                logger.error(f"Failed to get file history: {result.stderr}")
                return []
        except Exception as e:
            logger.error(f"Error getting file history: {e}")
            return []
            
    def get_diff(self, file_path: Optional[str] = None) -> str:
        """
        Get the diff for a file or the entire repository
        
        Args:
            file_path: Optional path to a specific file
            
        Returns:
            Diff as a string
        """
        if not self.is_git_repo:
            logger.error("Not a Git repository")
            return ""
            
        try:
            cmd = ["git", "diff"]
            if file_path:
                cmd.append(file_path)
                
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                logger.error(f"Failed to get diff: {result.stderr}")
                return ""
        except Exception as e:
            logger.error(f"Error getting diff: {e}")
            return ""

async def main():
    """Main function for testing the code generator"""
    # Create a code generator
    generator = CodeGenerator()
    
    # Generate code from a specification
    spec = """
    Create a function that calculates the Fibonacci sequence up to n terms.
    The function should handle invalid inputs and return a list of integers.
    """
    
    result = await generator.generate_code(spec)
    
    if result["success"]:
        print("Generated code:")
        print(result["code"])
        
        # Save the code to a file
        save_result = generator.save_code_to_file(
            result["code"],
            "fibonacci.py",
            make_executable=True
        )
        
        if save_result["success"]:
            print(f"Saved code to {save_result['file_path']}")
            
            # Execute the code
            exec_result = await generator.execute_code(result["code"])
            
            if exec_result["success"]:
                print("Execution successful:")
                print(exec_result["stdout"])
            else:
                print("Execution failed:")
                print(exec_result["error"] or exec_result["stderr"])
    else:
        print(f"Code generation failed: {result['error']}")

if __name__ == "__main__":
    asyncio.run(main())
