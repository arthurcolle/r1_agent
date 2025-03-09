#!/usr/bin/env python3
"""
Polymorphic Engine for Self-Modifying Agent

This module provides a polymorphic engine that allows the agent to:
1. Dynamically transform its own code structure while maintaining functionality
2. Generate functionally equivalent but structurally different code variants
3. Apply code obfuscation techniques that preserve semantics
4. Self-evolve its transformation capabilities
5. Maintain a history of transformations for rollback if needed
"""

import ast
import astor
import inspect
import random
import hashlib
import base64
import re
import os
import sys
import time
import logging
import importlib
import types
import copy
from typing import Dict, List, Tuple, Set, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransformationType(Enum):
    """Types of code transformations the engine can perform"""
    RENAME_VARIABLES = auto()
    REORDER_STATEMENTS = auto()
    EXTRACT_METHOD = auto()
    INLINE_METHOD = auto()
    CHANGE_CONTROL_FLOW = auto()
    ADD_DEAD_CODE = auto()
    MODIFY_CONSTANTS = auto()
    CHANGE_DATA_STRUCTURES = auto()
    RESTRUCTURE_CLASSES = auto()
    CHANGE_ALGORITHM = auto()

@dataclass
class CodeTransformation:
    """Represents a single code transformation"""
    type: TransformationType
    source_file: str
    node_path: List[int]  # Path to the AST node being transformed
    original_hash: str    # Hash of the original code
    transformed_hash: str  # Hash of the transformed code
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = False

@dataclass
class TransformationHistory:
    """Tracks the history of transformations applied to the codebase"""
    transformations: List[CodeTransformation] = field(default_factory=list)
    current_index: int = -1
    
    def add(self, transformation: CodeTransformation) -> None:
        """Add a transformation to history"""
        # If we're not at the end of history, truncate
        if self.current_index < len(self.transformations) - 1:
            self.transformations = self.transformations[:self.current_index + 1]
        
        self.transformations.append(transformation)
        self.current_index = len(self.transformations) - 1
        
    def can_undo(self) -> bool:
        """Check if undo is possible"""
        return self.current_index >= 0
        
    def can_redo(self) -> bool:
        """Check if redo is possible"""
        return self.current_index < len(self.transformations) - 1
        
    def undo(self) -> Optional[CodeTransformation]:
        """Get the transformation to undo"""
        if not self.can_undo():
            return None
        
        transformation = self.transformations[self.current_index]
        self.current_index -= 1
        return transformation
        
    def redo(self) -> Optional[CodeTransformation]:
        """Get the transformation to redo"""
        if not self.can_redo():
            return None
        
        self.current_index += 1
        return self.transformations[self.current_index]

class ASTVisitor(ast.NodeVisitor):
    """Custom AST visitor to analyze and collect information about the code"""
    
    def __init__(self):
        self.functions = {}
        self.classes = {}
        self.variables = {}
        self.imports = {}
        self.current_scope = None
        self.scope_stack = []
        
    def visit_FunctionDef(self, node):
        """Visit a function definition"""
        self.functions[node.name] = {
            'node': node,
            'args': [arg.arg for arg in node.args.args],
            'line': node.lineno,
            'end_line': node.end_lineno,
            'complexity': self._calculate_complexity(node)
        }
        
        # Track scope
        old_scope = self.current_scope
        self.scope_stack.append(node.name)
        self.current_scope = '.'.join(self.scope_stack)
        
        # Visit children
        self.generic_visit(node)
        
        # Restore scope
        self.scope_stack.pop()
        self.current_scope = old_scope
        
    def visit_ClassDef(self, node):
        """Visit a class definition"""
        self.classes[node.name] = {
            'node': node,
            'bases': [self._get_name(base) for base in node.bases],
            'methods': [],
            'line': node.lineno,
            'end_line': node.end_lineno
        }
        
        # Track scope
        old_scope = self.current_scope
        self.scope_stack.append(node.name)
        self.current_scope = '.'.join(self.scope_stack)
        
        # Visit children
        self.generic_visit(node)
        
        # Collect methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                self.classes[node.name]['methods'].append(item.name)
        
        # Restore scope
        self.scope_stack.pop()
        self.current_scope = old_scope
        
    def visit_Assign(self, node):
        """Visit an assignment"""
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                if self.current_scope not in self.variables:
                    self.variables[self.current_scope] = {}
                self.variables[self.current_scope][var_name] = {
                    'node': node,
                    'line': node.lineno
                }
        self.generic_visit(node)
        
    def visit_Import(self, node):
        """Visit an import statement"""
        for name in node.names:
            self.imports[name.name] = {
                'node': node,
                'alias': name.asname,
                'line': node.lineno
            }
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        """Visit a from-import statement"""
        for name in node.names:
            import_name = f"{node.module}.{name.name}" if node.module else name.name
            self.imports[import_name] = {
                'node': node,
                'alias': name.asname,
                'line': node.lineno
            }
        self.generic_visit(node)
        
    def _calculate_complexity(self, node):
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity
        
        # Count branches
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.Try):
                complexity += len(child.handlers)
                
        return complexity
        
    def _get_name(self, node):
        """Get the name of a node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return str(node)

class CodeTransformer(ast.NodeTransformer):
    """Base class for code transformers"""
    
    def __init__(self, seed=None):
        self.random = random.Random(seed)
        self.transformation_type = None
        
    def transform(self, tree):
        """Transform the AST"""
        return self.visit(tree)

class VariableRenamer(CodeTransformer):
    """Transformer that renames variables while preserving semantics"""
    
    def __init__(self, seed=None):
        super().__init__(seed)
        self.transformation_type = TransformationType.RENAME_VARIABLES
        self.name_map = {}
        self.scope_stack = []
        self.current_scope = None
        self.preserved_names = set()
        
    def visit_FunctionDef(self, node):
        """Visit a function definition"""
        # Track scope
        old_scope = self.current_scope
        self.scope_stack.append(node.name)
        self.current_scope = '.'.join(self.scope_stack)
        
        # Don't rename function name
        self.preserved_names.add(node.name)
        
        # Process arguments
        new_args = copy.deepcopy(node.args)
        for arg in new_args.args:
            if arg.arg != 'self':  # Don't rename 'self'
                old_name = arg.arg
                new_name = self._get_new_name(old_name)
                self._add_mapping(old_name, new_name)
                arg.arg = new_name
        
        # Process body
        new_body = [self.visit(stmt) for stmt in node.body]
        
        # Create new node
        new_node = ast.FunctionDef(
            name=node.name,
            args=new_args,
            body=new_body,
            decorator_list=node.decorator_list,
            returns=node.returns,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset
        )
        
        # Restore scope
        self.scope_stack.pop()
        self.current_scope = old_scope
        
        return new_node
        
    def visit_Name(self, node):
        """Visit a name node"""
        if isinstance(node.ctx, ast.Load):
            # Variable reference
            if node.id in self.name_map.get(self.current_scope, {}):
                return ast.Name(
                    id=self.name_map[self.current_scope][node.id],
                    ctx=node.ctx,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    end_lineno=node.end_lineno,
                    end_col_offset=node.end_col_offset
                )
        elif isinstance(node.ctx, ast.Store):
            # Variable assignment
            if node.id not in self.preserved_names:
                old_name = node.id
                if old_name not in self.name_map.get(self.current_scope, {}):
                    new_name = self._get_new_name(old_name)
                    self._add_mapping(old_name, new_name)
                return ast.Name(
                    id=self.name_map[self.current_scope][node.id],
                    ctx=node.ctx,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    end_lineno=node.end_lineno,
                    end_col_offset=node.end_col_offset
                )
        return node
        
    def _add_mapping(self, old_name, new_name):
        """Add a name mapping for the current scope"""
        if self.current_scope not in self.name_map:
            self.name_map[self.current_scope] = {}
        self.name_map[self.current_scope][old_name] = new_name
        
    def _get_new_name(self, old_name):
        """Generate a new variable name"""
        prefix = ''.join(c for c in old_name if c.isalpha())
        if not prefix:
            prefix = 'var'
        suffix = ''.join(self.random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(5))
        return f"{prefix}_{suffix}"

class StatementReorderer(CodeTransformer):
    """Transformer that reorders independent statements"""
    
    def __init__(self, seed=None):
        super().__init__(seed)
        self.transformation_type = TransformationType.REORDER_STATEMENTS
        
    def visit_FunctionDef(self, node):
        """Visit a function definition"""
        # Process the function body normally
        node = self.generic_visit(node)
        
        # Find blocks of statements that can be reordered
        reorderable_blocks = self._find_reorderable_blocks(node.body)
        
        # Reorder each block
        new_body = []
        i = 0
        while i < len(node.body):
            if i in reorderable_blocks:
                block_size = reorderable_blocks[i]
                block = node.body[i:i+block_size]
                # Shuffle the block
                self.random.shuffle(block)
                new_body.extend(block)
                i += block_size
            else:
                new_body.append(node.body[i])
                i += 1
                
        # Create new node
        return ast.FunctionDef(
            name=node.name,
            args=node.args,
            body=new_body,
            decorator_list=node.decorator_list,
            returns=node.returns,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset
        )
        
    def _find_reorderable_blocks(self, statements):
        """Find blocks of statements that can be safely reordered"""
        reorderable_blocks = {}
        i = 0
        while i < len(statements):
            # Skip non-reorderable statements
            if not self._is_reorderable(statements[i]):
                i += 1
                continue
                
            # Find the end of the reorderable block
            start = i
            i += 1
            while i < len(statements) and self._is_reorderable(statements[i]):
                i += 1
                
            # If block has multiple statements, mark it as reorderable
            if i - start > 1:
                reorderable_blocks[start] = i - start
                
        return reorderable_blocks
        
    def _is_reorderable(self, stmt):
        """Check if a statement can be safely reordered"""
        # Simple assignments can be reordered
        if isinstance(stmt, ast.Assign):
            return True
            
        # Function calls without assignments can be reordered
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            return True
            
        # Other statements are not safe to reorder
        return False

class ControlFlowTransformer(CodeTransformer):
    """Transformer that modifies control flow structures"""
    
    def __init__(self, seed=None):
        super().__init__(seed)
        self.transformation_type = TransformationType.CHANGE_CONTROL_FLOW
        
    def visit_If(self, node):
        """Visit an if statement"""
        # Process children first
        node = self.generic_visit(node)
        
        # Randomly choose a transformation
        transform_type = self.random.choice([
            'invert_condition',
            'add_redundant_condition',
            'split_condition'
        ])
        
        if transform_type == 'invert_condition':
            return self._invert_condition(node)
        elif transform_type == 'add_redundant_condition':
            return self._add_redundant_condition(node)
        elif transform_type == 'split_condition':
            return self._split_condition(node)
            
        return node
        
    def _invert_condition(self, node):
        """Invert the condition and swap the branches"""
        # Create the inverted condition
        inverted_test = ast.UnaryOp(
            op=ast.Not(),
            operand=node.test,
            lineno=node.test.lineno,
            col_offset=node.test.col_offset,
            end_lineno=node.test.end_lineno,
            end_col_offset=node.test.end_col_offset
        )
        
        # Swap the branches
        new_body = node.orelse
        new_orelse = node.body
        
        # Create new node
        return ast.If(
            test=inverted_test,
            body=new_body,
            orelse=new_orelse,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset
        )
        
    def _add_redundant_condition(self, node):
        """Add a redundant condition that doesn't change the outcome"""
        # Create a redundant condition (x == x)
        redundant = ast.Compare(
            left=ast.Constant(value=1, lineno=node.lineno, col_offset=node.col_offset),
            ops=[ast.Eq()],
            comparators=[ast.Constant(value=1, lineno=node.lineno, col_offset=node.col_offset)],
            lineno=node.lineno,
            col_offset=node.col_offset
        )
        
        # Combine with original condition using 'and'
        new_test = ast.BoolOp(
            op=ast.And(),
            values=[node.test, redundant],
            lineno=node.lineno,
            col_offset=node.col_offset
        )
        
        # Create new node
        return ast.If(
            test=new_test,
            body=node.body,
            orelse=node.orelse,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset
        )
        
    def _split_condition(self, node):
        """Split a simple condition into nested if statements"""
        # Only transform if there's no else branch
        if node.orelse:
            return node
            
        # Create a nested if with the same condition and body
        nested_if = ast.If(
            test=ast.Constant(value=True, lineno=node.lineno, col_offset=node.col_offset),
            body=node.body,
            orelse=[],
            lineno=node.lineno,
            col_offset=node.col_offset
        )
        
        # Create the outer if
        return ast.If(
            test=node.test,
            body=[nested_if],
            orelse=[],
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset
        )

class DeadCodeInserter(CodeTransformer):
    """Transformer that inserts dead (unreachable) code"""
    
    def __init__(self, seed=None):
        super().__init__(seed)
        self.transformation_type = TransformationType.ADD_DEAD_CODE
        
    def visit_FunctionDef(self, node):
        """Visit a function definition"""
        # Process children first
        node = self.generic_visit(node)
        
        # Insert dead code at random positions
        new_body = []
        for stmt in node.body:
            # 30% chance to insert dead code before a statement
            if self.random.random() < 0.3:
                new_body.append(self._generate_dead_code(node.lineno))
            new_body.append(stmt)
            
        # Create new node
        return ast.FunctionDef(
            name=node.name,
            args=node.args,
            body=new_body,
            decorator_list=node.decorator_list,
            returns=node.returns,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset
        )
        
    def _generate_dead_code(self, lineno):
        """Generate a dead code block"""
        # Choose a type of dead code
        dead_code_type = self.random.choice([
            'unreachable_if',
            'constant_condition',
            'debug_print'
        ])
        
        if dead_code_type == 'unreachable_if':
            return self._generate_unreachable_if(lineno)
        elif dead_code_type == 'constant_condition':
            return self._generate_constant_condition(lineno)
        elif dead_code_type == 'debug_print':
            return self._generate_debug_print(lineno)
            
    def _generate_unreachable_if(self, lineno):
        """Generate an if statement with an always-false condition"""
        # Create a false condition
        condition = ast.Compare(
            left=ast.Constant(value=1, lineno=lineno, col_offset=0),
            ops=[ast.Eq()],
            comparators=[ast.Constant(value=2, lineno=lineno, col_offset=0)],
            lineno=lineno,
            col_offset=0
        )
        
        # Create a body with a pass statement
        body = [ast.Pass(lineno=lineno, col_offset=4)]
        
        # Create the if statement
        return ast.If(
            test=condition,
            body=body,
            orelse=[],
            lineno=lineno,
            col_offset=0
        )
        
    def _generate_constant_condition(self, lineno):
        """Generate code with a constant condition"""
        # Create a condition that's always false
        condition = ast.Constant(value=False, lineno=lineno, col_offset=0)
        
        # Create a body with a pass statement
        body = [ast.Pass(lineno=lineno, col_offset=4)]
        
        # Create the if statement
        return ast.If(
            test=condition,
            body=body,
            orelse=[],
            lineno=lineno,
            col_offset=0
        )
        
    def _generate_debug_print(self, lineno):
        """Generate a debug print statement inside a false condition"""
        # Create a false condition
        condition = ast.Constant(value=False, lineno=lineno, col_offset=0)
        
        # Create a print call
        print_call = ast.Call(
            func=ast.Name(id='print', ctx=ast.Load(), lineno=lineno, col_offset=4),
            args=[ast.Constant(value='Debug output', lineno=lineno, col_offset=10)],
            keywords=[],
            lineno=lineno,
            col_offset=4
        )
        
        # Wrap in an expression
        print_stmt = ast.Expr(
            value=print_call,
            lineno=lineno,
            col_offset=4
        )
        
        # Create the if statement
        return ast.If(
            test=condition,
            body=[print_stmt],
            orelse=[],
            lineno=lineno,
            col_offset=0
        )

class PolymorphicEngine:
    """
    Main polymorphic engine that orchestrates code transformations
    """
    
    def __init__(self, seed=None):
        self.seed = seed if seed is not None else int(time.time())
        self.random = random.Random(self.seed)
        self.history = TransformationHistory()
        self.transformers = {
            TransformationType.RENAME_VARIABLES: VariableRenamer,
            TransformationType.REORDER_STATEMENTS: StatementReorderer,
            TransformationType.CHANGE_CONTROL_FLOW: ControlFlowTransformer,
            TransformationType.ADD_DEAD_CODE: DeadCodeInserter,
            # Add more transformers as they're implemented
        }
        
    def transform_file(self, file_path: str, transformation_types=None, 
                      intensity: float = 0.5) -> Tuple[bool, str]:
        """
        Transform a Python file using selected transformation types
        
        Args:
            file_path: Path to the Python file
            transformation_types: List of transformation types to apply (None = all)
            intensity: How aggressive the transformations should be (0.0-1.0)
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
                
            # Parse the AST
            tree = ast.parse(source)
            
            # Calculate original hash
            original_hash = hashlib.md5(source.encode()).hexdigest()
            
            # Select transformations to apply
            if transformation_types is None:
                # Use all available transformers
                available_types = list(self.transformers.keys())
            else:
                # Use specified transformers
                available_types = [t for t in transformation_types if t in self.transformers]
                
            if not available_types:
                return False, "No valid transformation types specified"
                
            # Determine how many transformations to apply based on intensity
            num_transformations = max(1, int(len(available_types) * intensity))
            selected_types = self.random.sample(available_types, num_transformations)
            
            # Apply transformations
            for transform_type in selected_types:
                transformer_class = self.transformers[transform_type]
                transformer = transformer_class(seed=self.random.randint(0, 10000))
                tree = transformer.transform(tree)
                
                # Record the transformation
                transformation = CodeTransformation(
                    type=transform_type,
                    source_file=file_path,
                    node_path=[],  # Not tracking specific nodes for now
                    original_hash=original_hash,
                    transformed_hash="",  # Will be updated after all transformations
                    metadata={"transformer": transformer_class.__name__}
                )
                
                # Update the source for the next transformation
                source = astor.to_source(tree)
                
            # Calculate the final hash
            transformed_hash = hashlib.md5(source.encode()).hexdigest()
            
            # Update the transformation record
            transformation.transformed_hash = transformed_hash
            transformation.success = True
            self.history.add(transformation)
            
            # Write the transformed code back to the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(source)
                
            return True, f"Successfully applied {num_transformations} transformations"
            
        except Exception as e:
            logger.error(f"Error transforming file {file_path}: {str(e)}")
            return False, f"Error: {str(e)}"
            
    def transform_code(self, source_code: str, transformation_types=None,
                      intensity: float = 0.5) -> Tuple[bool, str, str]:
        """
        Transform Python code string using selected transformation types
        
        Args:
            source_code: Python source code string
            transformation_types: List of transformation types to apply (None = all)
            intensity: How aggressive the transformations should be (0.0-1.0)
            
        Returns:
            Tuple of (success, transformed_code, message)
        """
        try:
            # Parse the AST
            tree = ast.parse(source_code)
            
            # Calculate original hash
            original_hash = hashlib.md5(source_code.encode()).hexdigest()
            
            # Select transformations to apply
            if transformation_types is None:
                # Use all available transformers
                available_types = list(self.transformers.keys())
            else:
                # Use specified transformers
                available_types = [t for t in transformation_types if t in self.transformers]
                
            if not available_types:
                return False, source_code, "No valid transformation types specified"
                
            # Determine how many transformations to apply based on intensity
            num_transformations = max(1, int(len(available_types) * intensity))
            selected_types = self.random.sample(available_types, num_transformations)
            
            # Apply transformations
            for transform_type in selected_types:
                transformer_class = self.transformers[transform_type]
                transformer = transformer_class(seed=self.random.randint(0, 10000))
                tree = transformer.transform(tree)
                
                # Update the source for the next transformation
                source_code = astor.to_source(tree)
                
            # Calculate the final hash
            transformed_hash = hashlib.md5(source_code.encode()).hexdigest()
            
            # Create a transformation record (but don't add to history since there's no file)
            transformation = CodeTransformation(
                type=selected_types[-1],  # Use the last transformation type
                source_file="<string>",
                node_path=[],  # Not tracking specific nodes for now
                original_hash=original_hash,
                transformed_hash=transformed_hash,
                metadata={"num_transformations": num_transformations},
                success=True
            )
            
            return True, source_code, f"Successfully applied {num_transformations} transformations"
            
        except Exception as e:
            logger.error(f"Error transforming code: {str(e)}")
            return False, source_code, f"Error: {str(e)}"
            
    def undo_last_transformation(self, file_path: str) -> Tuple[bool, str]:
        """
        Undo the last transformation for a specific file
        
        Args:
            file_path: Path to the file to restore
            
        Returns:
            Tuple of (success, message)
        """
        if not self.history.can_undo():
            return False, "No transformations to undo"
            
        # Get the last transformation
        transformation = self.history.undo()
        
        # Check if it's for the requested file
        if transformation.source_file != file_path:
            # Put it back and return error
            self.history.redo()
            return False, f"Last transformation was for {transformation.source_file}, not {file_path}"
            
        try:
            # Read the current file
            with open(file_path, 'r', encoding='utf-8') as f:
                current_source = f.read()
                
            # Check if the file has been modified since the transformation
            current_hash = hashlib.md5(current_source.encode()).hexdigest()
            if current_hash != transformation.transformed_hash:
                return False, "File has been modified since the transformation"
                
            # We don't actually store the original source, so we need to re-transform
            # This is a limitation of the current implementation
            return False, "Undo not fully implemented - original source not stored"
            
        except Exception as e:
            logger.error(f"Error undoing transformation: {str(e)}")
            return False, f"Error: {str(e)}"
            
    def get_transformation_history(self) -> List[CodeTransformation]:
        """Get the history of transformations"""
        return self.history.transformations
        
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a Python file and return information about its structure
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
                
            # Parse the AST
            tree = ast.parse(source)
            
            # Analyze the AST
            visitor = ASTVisitor()
            visitor.visit(tree)
            
            # Prepare results
            results = {
                'functions': {name: {k: v for k, v in info.items() if k != 'node'} 
                             for name, info in visitor.functions.items()},
                'classes': {name: {k: v for k, v in info.items() if k != 'node'} 
                           for name, info in visitor.classes.items()},
                'imports': {name: {k: v for k, v in info.items() if k != 'node'} 
                           for name, info in visitor.imports.items()},
                'variables': {scope: {name: {k: v for k, v in info.items() if k != 'node'} 
                                    for name, info in vars.items()}
                             for scope, vars in visitor.variables.items()},
                'file_hash': hashlib.md5(source.encode()).hexdigest(),
                'loc': len(source.splitlines()),
                'complexity': sum(func['complexity'] for func in visitor.functions.values())
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {str(e)}")
            return {'error': str(e)}
            
    def generate_variant(self, file_path: str, num_variants: int = 1, 
                        intensity: float = 0.5) -> List[str]:
        """
        Generate multiple functionally equivalent variants of a file
        
        Args:
            file_path: Path to the Python file
            num_variants: Number of variants to generate
            intensity: How aggressive the transformations should be (0.0-1.0)
            
        Returns:
            List of file paths to the generated variants
        """
        variants = []
        
        try:
            # Read the original file
            with open(file_path, 'r', encoding='utf-8') as f:
                original_source = f.read()
                
            # Generate variants
            for i in range(num_variants):
                # Create a unique seed for each variant
                variant_seed = self.random.randint(0, 10000)
                
                # Parse the AST
                tree = ast.parse(original_source)
                
                # Apply random transformations
                available_types = list(self.transformers.keys())
                num_transformations = max(1, int(len(available_types) * intensity))
                
                for _ in range(num_transformations):
                    transform_type = self.random.choice(available_types)
                    transformer_class = self.transformers[transform_type]
                    transformer = transformer_class(seed=variant_seed + _)
                    tree = transformer.transform(tree)
                    
                # Generate the variant source
                variant_source = astor.to_source(tree)
                
                # Create a variant file
                base_name, ext = os.path.splitext(file_path)
                variant_path = f"{base_name}_variant_{i+1}{ext}"
                
                with open(variant_path, 'w', encoding='utf-8') as f:
                    f.write(variant_source)
                    
                variants.append(variant_path)
                
            return variants
            
        except Exception as e:
            logger.error(f"Error generating variants for {file_path}: {str(e)}")
            return []
            
    def self_modify(self):
        """
        Apply the polymorphic engine to itself
        
        Returns:
            Tuple of (success, message)
        """
        # Get the path to this file
        self_path = inspect.getfile(self.__class__)
        
        # Transform the file
        return self.transform_file(self_path, intensity=0.3)

# Example usage:
if __name__ == "__main__":
    # Create the polymorphic engine
    engine = PolymorphicEngine()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Polymorphic Engine for Self-Modifying Code")
    parser.add_argument("--file", type=str, help="File to transform")
    parser.add_argument("--intensity", type=float, default=0.5, help="Transformation intensity (0.0-1.0)")
    parser.add_argument("--analyze", action="store_true", help="Analyze file instead of transforming")
    parser.add_argument("--variants", type=int, default=0, help="Generate N variants of the file")
    parser.add_argument("--self-modify", action="store_true", help="Apply the engine to itself")
    
    args = parser.parse_args()
    
    if args.self_modify:
        success, message = engine.self_modify()
        print(f"Self-modification: {message}")
    elif args.file:
        if args.analyze:
            results = engine.analyze_file(args.file)
            print(f"Analysis results for {args.file}:")
            for key, value in results.items():
                print(f"  {key}: {value}")
        elif args.variants > 0:
            variants = engine.generate_variant(args.file, args.variants, args.intensity)
            print(f"Generated {len(variants)} variants:")
            for variant in variants:
                print(f"  {variant}")
        else:
            success, message = engine.transform_file(args.file, intensity=args.intensity)
            print(f"Transformation: {message}")
    else:
        print("No file specified. Use --file to specify a file to transform.")
