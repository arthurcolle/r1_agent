"""
Knowledge Graph Construction and Reasoning

This module implements a knowledge graph system for representing and reasoning
about complex relationships between entities. It provides capabilities for:
- Building knowledge graphs from various data sources
- Querying and traversing the graph
- Inferring new relationships through reasoning
- Visualizing the knowledge graph
- Integrating with external knowledge bases
"""

import os
import re
import json
import logging
import sqlite3
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import networkx as nx
    HAVE_NETWORKX = True
except ImportError:
    HAVE_NETWORKX = False
    logger.warning("NetworkX not found. Graph visualization will be unavailable.")

try:
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False
    logger.warning("Matplotlib not found. Graph visualization will be unavailable.")

try:
    from openai import AsyncOpenAI
    HAVE_OPENAI = True
except ImportError:
    HAVE_OPENAI = False
    logger.warning("OpenAI package not found. Some extraction features will be limited.")

class RelationshipType(Enum):
    """Types of relationships between entities in the knowledge graph"""
    IS_A = auto()           # Inheritance/type relationship
    PART_OF = auto()        # Composition relationship
    HAS_PROPERTY = auto()   # Property relationship
    RELATED_TO = auto()     # Generic relationship
    DEPENDS_ON = auto()     # Dependency relationship
    CAUSES = auto()         # Causal relationship
    PRECEDES = auto()       # Temporal relationship
    LOCATED_IN = auto()     # Spatial relationship
    CREATED_BY = auto()     # Authorship relationship
    INSTANCE_OF = auto()    # Instance relationship
    CONTRADICTS = auto()    # Contradiction relationship
    SUPPORTS = auto()       # Supporting relationship
    CUSTOM = auto()         # Custom relationship type

@dataclass
class Entity:
    """Represents an entity (node) in the knowledge graph"""
    id: str
    name: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    confidence: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Relationship:
    """Represents a relationship (edge) between entities in the knowledge graph"""
    id: str
    source_id: str
    target_id: str
    type: RelationshipType
    name: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    confidence: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    bidirectional: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class KnowledgeGraph:
    """
    Main knowledge graph class that manages entities and relationships.
    Provides methods for adding, querying, and reasoning about knowledge.
    """
    def __init__(self, db_path: str = "knowledge_graph.db"):
        self.db_path = db_path
        self.conn = None
        self.entities = {}  # In-memory cache of entities
        self.relationships = {}  # In-memory cache of relationships
        
        # Initialize database
        self._init_db()
        
        # Initialize NetworkX graph if available
        self.graph = None
        if HAVE_NETWORKX:
            self.graph = nx.DiGraph()
            
    def _init_db(self) -> None:
        """Initialize the SQLite database"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
        # Create tables if they don't exist
        cursor = self.conn.cursor()
        
        # Entities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                properties TEXT,
                sources TEXT,
                confidence REAL,
                created_at TEXT,
                last_updated TEXT,
                embedding BLOB,
                metadata TEXT
            )
        """)
        
        # Relationships table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                type TEXT NOT NULL,
                name TEXT,
                properties TEXT,
                sources TEXT,
                confidence REAL,
                created_at TEXT,
                last_updated TEXT,
                bidirectional INTEGER,
                metadata TEXT,
                FOREIGN KEY (source_id) REFERENCES entities (id),
                FOREIGN KEY (target_id) REFERENCES entities (id)
            )
        """)
        
        # Create indices for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities (type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships (source_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships (target_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships (type)")
        
        # Create a table for inference rules
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inference_rules (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                pattern TEXT NOT NULL,
                confidence REAL,
                created_at TEXT,
                enabled INTEGER
            )
        """)
        
        # Create a table for query history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                results TEXT,
                execution_time REAL
            )
        """)
        
        self.conn.commit()
        
        # Load entities and relationships into memory
        self._load_from_db()
        
    def _load_from_db(self) -> None:
        """Load entities and relationships from the database into memory"""
        cursor = self.conn.cursor()
        
        # Load entities
        cursor.execute("SELECT * FROM entities")
        for row in cursor.fetchall():
            entity = Entity(
                id=row['id'],
                name=row['name'],
                type=row['type'],
                properties=json.loads(row['properties']) if row['properties'] else {},
                sources=json.loads(row['sources']) if row['sources'] else [],
                confidence=row['confidence'],
                created_at=row['created_at'],
                last_updated=row['last_updated'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            self.entities[entity.id] = entity
            
            # Add to NetworkX graph if available
            if self.graph is not None:
                self.graph.add_node(entity.id, **{
                    'name': entity.name,
                    'type': entity.type,
                    'properties': entity.properties
                })
        
        # Load relationships
        cursor.execute("SELECT * FROM relationships")
        for row in cursor.fetchall():
            rel_type = RelationshipType[row['type']] if row['type'] in RelationshipType.__members__ else RelationshipType.CUSTOM
            relationship = Relationship(
                id=row['id'],
                source_id=row['source_id'],
                target_id=row['target_id'],
                type=rel_type,
                name=row['name'],
                properties=json.loads(row['properties']) if row['properties'] else {},
                sources=json.loads(row['sources']) if row['sources'] else [],
                confidence=row['confidence'],
                created_at=row['created_at'],
                last_updated=row['last_updated'],
                bidirectional=bool(row['bidirectional']),
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            self.relationships[relationship.id] = relationship
            
            # Add to NetworkX graph if available
            if self.graph is not None:
                self.graph.add_edge(relationship.source_id, relationship.target_id, **{
                    'id': relationship.id,
                    'type': relationship.type.name,
                    'name': relationship.name,
                    'bidirectional': relationship.bidirectional
                })
                
                # Add reverse edge if bidirectional
                if relationship.bidirectional:
                    self.graph.add_edge(relationship.target_id, relationship.source_id, **{
                        'id': relationship.id + "_reverse",
                        'type': relationship.type.name,
                        'name': relationship.name,
                        'bidirectional': True
                    })
        
        logger.info(f"Loaded {len(self.entities)} entities and {len(self.relationships)} relationships from database")
        
    def add_entity(self, entity: Entity) -> str:
        """
        Add an entity to the knowledge graph
        
        Args:
            entity: The entity to add
            
        Returns:
            The ID of the added entity
        """
        # Generate ID if not provided
        if not entity.id:
            entity.id = str(uuid.uuid4())
            
        # Update timestamps
        now = datetime.now().isoformat()
        entity.created_at = now
        entity.last_updated = now
        
        # Store in memory
        self.entities[entity.id] = entity
        
        # Store in database
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO entities
            (id, name, type, properties, sources, confidence, created_at, last_updated, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entity.id,
            entity.name,
            entity.type,
            json.dumps(entity.properties),
            json.dumps(entity.sources),
            entity.confidence,
            entity.created_at,
            entity.last_updated,
            None,  # embedding (not stored in SQLite for now)
            json.dumps(entity.metadata)
        ))
        self.conn.commit()
        
        # Add to NetworkX graph if available
        if self.graph is not None:
            self.graph.add_node(entity.id, **{
                'name': entity.name,
                'type': entity.type,
                'properties': entity.properties
            })
            
        logger.debug(f"Added entity: {entity.name} ({entity.id})")
        return entity.id
        
    def add_relationship(self, relationship: Relationship) -> str:
        """
        Add a relationship to the knowledge graph
        
        Args:
            relationship: The relationship to add
            
        Returns:
            The ID of the added relationship
        """
        # Check if source and target entities exist
        if relationship.source_id not in self.entities:
            raise ValueError(f"Source entity {relationship.source_id} does not exist")
        if relationship.target_id not in self.entities:
            raise ValueError(f"Target entity {relationship.target_id} does not exist")
            
        # Generate ID if not provided
        if not relationship.id:
            relationship.id = str(uuid.uuid4())
            
        # Update timestamps
        now = datetime.now().isoformat()
        relationship.created_at = now
        relationship.last_updated = now
        
        # Store in memory
        self.relationships[relationship.id] = relationship
        
        # Store in database
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO relationships
            (id, source_id, target_id, type, name, properties, sources, confidence, 
             created_at, last_updated, bidirectional, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            relationship.id,
            relationship.source_id,
            relationship.target_id,
            relationship.type.name,
            relationship.name,
            json.dumps(relationship.properties),
            json.dumps(relationship.sources),
            relationship.confidence,
            relationship.created_at,
            relationship.last_updated,
            1 if relationship.bidirectional else 0,
            json.dumps(relationship.metadata)
        ))
        self.conn.commit()
        
        # Add to NetworkX graph if available
        if self.graph is not None:
            self.graph.add_edge(relationship.source_id, relationship.target_id, **{
                'id': relationship.id,
                'type': relationship.type.name,
                'name': relationship.name,
                'bidirectional': relationship.bidirectional
            })
            
            # Add reverse edge if bidirectional
            if relationship.bidirectional:
                self.graph.add_edge(relationship.target_id, relationship.source_id, **{
                    'id': relationship.id + "_reverse",
                    'type': relationship.type.name,
                    'name': relationship.name,
                    'bidirectional': True
                })
                
        logger.debug(f"Added relationship: {relationship.type.name} from {relationship.source_id} to {relationship.target_id}")
        return relationship.id
        
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID"""
        return self.entities.get(entity_id)
        
    def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """Get a relationship by ID"""
        return self.relationships.get(relationship_id)
        
    def find_entities(self, query: Dict[str, Any]) -> List[Entity]:
        """
        Find entities matching the query
        
        Args:
            query: Dictionary of criteria to match (e.g., {'type': 'Person', 'name': 'John'})
            
        Returns:
            List of matching entities
        """
        results = []
        
        for entity in self.entities.values():
            match = True
            
            for key, value in query.items():
                if key == 'properties':
                    # Check properties
                    for prop_key, prop_value in value.items():
                        if prop_key not in entity.properties or entity.properties[prop_key] != prop_value:
                            match = False
                            break
                elif key == 'name' and isinstance(value, str) and value.startswith('*') and value.endswith('*'):
                    # Wildcard search
                    pattern = value.strip('*')
                    if pattern not in entity.name:
                        match = False
                elif getattr(entity, key, None) != value:
                    match = False
                    break
                    
            if match:
                results.append(entity)
                
        return results
        
    def find_relationships(self, query: Dict[str, Any]) -> List[Relationship]:
        """
        Find relationships matching the query
        
        Args:
            query: Dictionary of criteria to match
            
        Returns:
            List of matching relationships
        """
        results = []
        
        for rel in self.relationships.values():
            match = True
            
            for key, value in query.items():
                if key == 'type' and isinstance(value, str):
                    # Match by type name
                    if rel.type.name != value:
                        match = False
                        break
                elif key == 'properties':
                    # Check properties
                    for prop_key, prop_value in value.items():
                        if prop_key not in rel.properties or rel.properties[prop_key] != prop_value:
                            match = False
                            break
                elif getattr(rel, key, None) != value:
                    match = False
                    break
                    
            if match:
                results.append(rel)
                
        return results
        
    def get_entity_relationships(self, entity_id: str, 
                               direction: str = 'both',
                               rel_type: Optional[Union[RelationshipType, str]] = None) -> List[Relationship]:
        """
        Get relationships for an entity
        
        Args:
            entity_id: ID of the entity
            direction: 'outgoing', 'incoming', or 'both'
            rel_type: Optional relationship type to filter by
            
        Returns:
            List of relationships
        """
        results = []
        
        # Convert string type to enum if needed
        if isinstance(rel_type, str) and rel_type in RelationshipType.__members__:
            rel_type = RelationshipType[rel_type]
            
        for rel in self.relationships.values():
            # Check direction
            if direction == 'outgoing' and rel.source_id != entity_id:
                continue
            if direction == 'incoming' and rel.target_id != entity_id:
                continue
            if direction == 'both' and rel.source_id != entity_id and rel.target_id != entity_id:
                continue
                
            # Check type if specified
            if rel_type is not None and rel.type != rel_type:
                continue
                
            results.append(rel)
            
        return results
        
    def get_connected_entities(self, entity_id: str, 
                             direction: str = 'both',
                             rel_type: Optional[Union[RelationshipType, str]] = None,
                             max_depth: int = 1) -> List[Entity]:
        """
        Get entities connected to the given entity
        
        Args:
            entity_id: ID of the entity
            direction: 'outgoing', 'incoming', or 'both'
            rel_type: Optional relationship type to filter by
            max_depth: Maximum traversal depth
            
        Returns:
            List of connected entities
        """
        if max_depth < 1:
            return []
            
        # Use NetworkX for efficient traversal if available
        if self.graph is not None:
            connected_ids = set()
            
            # Convert string type to enum if needed
            type_name = None
            if isinstance(rel_type, RelationshipType):
                type_name = rel_type.name
            elif isinstance(rel_type, str) and rel_type in RelationshipType.__members__:
                type_name = rel_type
                
            # Get neighbors based on direction
            if direction == 'outgoing' or direction == 'both':
                for _, target, edge_data in self.graph.out_edges(entity_id, data=True):
                    if type_name is None or edge_data.get('type') == type_name:
                        connected_ids.add(target)
                        
            if direction == 'incoming' or direction == 'both':
                for source, _, edge_data in self.graph.in_edges(entity_id, data=True):
                    if type_name is None or edge_data.get('type') == type_name:
                        connected_ids.add(source)
                        
            # Recursively get connections if depth > 1
            if max_depth > 1:
                next_level_ids = set()
                for connected_id in connected_ids:
                    next_level = self.get_connected_entities(
                        connected_id, direction, rel_type, max_depth - 1)
                    next_level_ids.update([e.id for e in next_level])
                    
                connected_ids.update(next_level_ids)
                
            # Remove the original entity
            if entity_id in connected_ids:
                connected_ids.remove(entity_id)
                
            # Convert IDs to entities
            return [self.entities[eid] for eid in connected_ids if eid in self.entities]
        else:
            # Fallback implementation without NetworkX
            connected_ids = set()
            
            # Get relationships
            relationships = self.get_entity_relationships(entity_id, direction, rel_type)
            
            # Get connected entities
            for rel in relationships:
                if rel.source_id == entity_id:
                    connected_ids.add(rel.target_id)
                else:
                    connected_ids.add(rel.source_id)
                    
            # Recursively get connections if depth > 1
            if max_depth > 1:
                next_level_ids = set()
                for connected_id in connected_ids:
                    next_level = self.get_connected_entities(
                        connected_id, direction, rel_type, max_depth - 1)
                    next_level_ids.update([e.id for e in next_level])
                    
                connected_ids.update(next_level_ids)
                
            # Remove the original entity
            if entity_id in connected_ids:
                connected_ids.remove(entity_id)
                
            # Convert IDs to entities
            return [self.entities[eid] for eid in connected_ids if eid in self.entities]
            
    def find_paths(self, source_id: str, target_id: str, 
                 max_length: int = 3) -> List[List[Tuple[Entity, Relationship]]]:
        """
        Find paths between two entities
        
        Args:
            source_id: ID of the source entity
            target_id: ID of the target entity
            max_length: Maximum path length
            
        Returns:
            List of paths, where each path is a list of (entity, relationship) tuples
        """
        if self.graph is not None:
            # Use NetworkX for efficient path finding
            try:
                # Find all simple paths up to max_length
                paths = list(nx.all_simple_paths(
                    self.graph, source_id, target_id, cutoff=max_length))
                
                # Convert paths to (entity, relationship) tuples
                result_paths = []
                for path in paths:
                    result_path = []
                    for i in range(len(path) - 1):
                        entity = self.entities[path[i]]
                        # Find the relationship between this entity and the next
                        for rel in self.relationships.values():
                            if rel.source_id == path[i] and rel.target_id == path[i + 1]:
                                result_path.append((entity, rel))
                                break
                                
                    # Add the final entity
                    result_path.append((self.entities[path[-1]], None))
                    result_paths.append(result_path)
                    
                return result_paths
            except nx.NetworkXNoPath:
                return []
        else:
            # Fallback implementation without NetworkX
            # This is a simplified breadth-first search
            if max_length < 1:
                return []
                
            # Check if source and target exist
            if source_id not in self.entities or target_id not in self.entities:
                return []
                
            # Initialize queue with source entity
            queue = [[(source_id, None)]]
            visited = set([source_id])
            result_paths = []
            
            while queue:
                path = queue.pop(0)
                current_id = path[-1][0]
                
                # Check if we've reached the target
                if current_id == target_id:
                    # Convert IDs to entities and relationships
                    result_path = []
                    for i in range(len(path)):
                        entity_id, rel_id = path[i]
                        entity = self.entities[entity_id]
                        rel = self.relationships.get(rel_id) if rel_id else None
                        result_path.append((entity, rel))
                    result_paths.append(result_path)
                    continue
                    
                # Check if we've reached max length
                if len(path) > max_length:
                    continue
                    
                # Get outgoing relationships
                for rel in self.relationships.values():
                    if rel.source_id == current_id and rel.target_id not in visited:
                        new_path = path + [(rel.target_id, rel.id)]
                        queue.append(new_path)
                        visited.add(rel.target_id)
                        
            return result_paths
            
    def apply_inference_rules(self) -> int:
        """
        Apply inference rules to derive new relationships
        
        Returns:
            Number of new relationships inferred
        """
        # Load inference rules from database
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM inference_rules WHERE enabled = 1")
        rules = cursor.fetchall()
        
        if not rules:
            logger.info("No enabled inference rules found")
            return 0
            
        new_relationships = 0
        
        for rule in rules:
            rule_id = rule['id']
            rule_name = rule['name']
            pattern = json.loads(rule['pattern'])
            confidence = rule['confidence']
            
            logger.info(f"Applying inference rule: {rule_name}")
            
            # Apply the rule based on its pattern
            if pattern['type'] == 'transitive':
                # Transitive inference: if A->B and B->C then A->C
                rel_type_name = pattern['relationship_type']
                rel_type = RelationshipType[rel_type_name] if rel_type_name in RelationshipType.__members__ else None
                
                if not rel_type:
                    logger.warning(f"Invalid relationship type in rule {rule_name}: {rel_type_name}")
                    continue
                    
                # Find all relationships of the specified type
                relationships = self.find_relationships({'type': rel_type_name})
                
                # Build a dictionary of source -> targets
                source_to_targets = {}
                for rel in relationships:
                    if rel.source_id not in source_to_targets:
                        source_to_targets[rel.source_id] = set()
                    source_to_targets[rel.source_id].add(rel.target_id)
                    
                # Find transitive relationships
                for source_id, targets in source_to_targets.items():
                    for target_id in targets:
                        if target_id in source_to_targets:
                            # For each target of the target
                            for transitive_target in source_to_targets[target_id]:
                                # Check if the relationship already exists
                                existing = False
                                for rel in relationships:
                                    if rel.source_id == source_id and rel.target_id == transitive_target:
                                        existing = True
                                        break
                                        
                                if not existing and source_id != transitive_target:
                                    # Create a new inferred relationship
                                    new_rel = Relationship(
                                        id=str(uuid.uuid4()),
                                        source_id=source_id,
                                        target_id=transitive_target,
                                        type=rel_type,
                                        name=f"Inferred {rel_type_name}",
                                        confidence=confidence,
                                        sources=[rule_id],
                                        metadata={"inferred": True, "rule_id": rule_id}
                                    )
                                    self.add_relationship(new_rel)
                                    new_relationships += 1
                                    
            elif pattern['type'] == 'symmetric':
                # Symmetric inference: if A->B then B->A
                rel_type_name = pattern['relationship_type']
                rel_type = RelationshipType[rel_type_name] if rel_type_name in RelationshipType.__members__ else None
                
                if not rel_type:
                    logger.warning(f"Invalid relationship type in rule {rule_name}: {rel_type_name}")
                    continue
                    
                # Find all relationships of the specified type
                relationships = self.find_relationships({'type': rel_type_name})
                
                for rel in relationships:
                    # Check if the symmetric relationship already exists
                    existing = False
                    for other_rel in relationships:
                        if other_rel.source_id == rel.target_id and other_rel.target_id == rel.source_id:
                            existing = True
                            break
                            
                    if not existing:
                        # Create a new inferred relationship
                        new_rel = Relationship(
                            id=str(uuid.uuid4()),
                            source_id=rel.target_id,
                            target_id=rel.source_id,
                            type=rel_type,
                            name=f"Inferred {rel_type_name}",
                            confidence=confidence,
                            sources=[rule_id],
                            metadata={"inferred": True, "rule_id": rule_id}
                        )
                        self.add_relationship(new_rel)
                        new_relationships += 1
                        
            elif pattern['type'] == 'inheritance':
                # Inheritance inference: if A is_a B and B has_property P then A has_property P
                # Find all IS_A relationships
                is_a_rels = self.find_relationships({'type': 'IS_A'})
                
                # Find all HAS_PROPERTY relationships
                has_prop_rels = self.find_relationships({'type': 'HAS_PROPERTY'})
                
                # Build a dictionary of entity -> properties
                entity_to_props = {}
                for rel in has_prop_rels:
                    if rel.source_id not in entity_to_props:
                        entity_to_props[rel.source_id] = set()
                    entity_to_props[rel.source_id].add(rel.target_id)
                    
                # For each IS_A relationship
                for is_a_rel in is_a_rels:
                    child_id = is_a_rel.source_id
                    parent_id = is_a_rel.target_id
                    
                    # If the parent has properties
                    if parent_id in entity_to_props:
                        for prop_id in entity_to_props[parent_id]:
                            # Check if the child already has this property
                            has_prop = False
                            if child_id in entity_to_props:
                                has_prop = prop_id in entity_to_props[child_id]
                                
                            if not has_prop:
                                # Create a new inferred HAS_PROPERTY relationship
                                new_rel = Relationship(
                                    id=str(uuid.uuid4()),
                                    source_id=child_id,
                                    target_id=prop_id,
                                    type=RelationshipType.HAS_PROPERTY,
                                    name="Inferred property",
                                    confidence=confidence,
                                    sources=[rule_id],
                                    metadata={"inferred": True, "rule_id": rule_id}
                                )
                                self.add_relationship(new_rel)
                                new_relationships += 1
                                
        logger.info(f"Inferred {new_relationships} new relationships")
        return new_relationships
        
    def add_inference_rule(self, name: str, description: str, pattern: Dict[str, Any], 
                         confidence: float = 0.8, enabled: bool = True) -> str:
        """
        Add an inference rule
        
        Args:
            name: Rule name
            description: Rule description
            pattern: Rule pattern (e.g., {'type': 'transitive', 'relationship_type': 'IS_A'})
            confidence: Confidence level for inferred relationships
            enabled: Whether the rule is enabled
            
        Returns:
            Rule ID
        """
        rule_id = str(uuid.uuid4())
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO inference_rules
            (id, name, description, pattern, confidence, created_at, enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            rule_id,
            name,
            description,
            json.dumps(pattern),
            confidence,
            datetime.now().isoformat(),
            1 if enabled else 0
        ))
        self.conn.commit()
        
        logger.info(f"Added inference rule: {name} ({rule_id})")
        return rule_id
        
    def visualize(self, output_file: str = "knowledge_graph.png", 
                max_nodes: int = 100, 
                focus_entity_id: Optional[str] = None,
                highlight_path: Optional[List[str]] = None) -> bool:
        """
        Visualize the knowledge graph
        
        Args:
            output_file: Output file path
            max_nodes: Maximum number of nodes to include
            focus_entity_id: Optional entity ID to focus on
            highlight_path: Optional list of entity IDs to highlight
            
        Returns:
            True if visualization was successful, False otherwise
        """
        if not HAVE_NETWORKX or not HAVE_MATPLOTLIB:
            logger.error("NetworkX and/or Matplotlib not available. Cannot visualize graph.")
            return False
            
        if not self.graph:
            logger.error("NetworkX graph not initialized. Cannot visualize.")
            return False
            
        # Create a subgraph if needed
        if len(self.graph) > max_nodes and focus_entity_id:
            # Create a subgraph centered on the focus entity
            entities_to_include = set()
            entities_to_include.add(focus_entity_id)
            
            # Add connected entities up to 2 hops away
            connected = self.get_connected_entities(focus_entity_id, 'both', None, 2)
            entities_to_include.update([e.id for e in connected])
            
            # Limit to max_nodes
            if len(entities_to_include) > max_nodes:
                # Prioritize entities in the highlight path if provided
                if highlight_path:
                    priority_entities = set(highlight_path)
                    remaining = max_nodes - len(priority_entities)
                    if remaining > 0:
                        other_entities = entities_to_include - priority_entities
                        entities_to_include = priority_entities.union(list(other_entities)[:remaining])
                    else:
                        entities_to_include = priority_entities
                else:
                    # Just take the first max_nodes
                    entities_to_include = list(entities_to_include)[:max_nodes]
                    
            # Create the subgraph
            subgraph = self.graph.subgraph(entities_to_include)
        else:
            # Use the full graph if it's small enough
            subgraph = self.graph
            
        # Set up the plot
        plt.figure(figsize=(12, 10))
        
        # Create a layout
        pos = nx.spring_layout(subgraph)
        
        # Prepare node colors based on entity type
        entity_types = set(self.entities[n].type for n in subgraph.nodes() if n in self.entities)
        color_map = {}
        colors = plt.cm.tab10(range(len(entity_types)))
        for i, entity_type in enumerate(entity_types):
            color_map[entity_type] = colors[i]
            
        # Draw nodes
        for entity_type in entity_types:
            nodes = [n for n in subgraph.nodes() if n in self.entities and self.entities[n].type == entity_type]
            nx.draw_networkx_nodes(
                subgraph, pos,
                nodelist=nodes,
                node_color=[color_map[entity_type]] * len(nodes),
                node_size=500,
                alpha=0.8,
                label=entity_type
            )
            
        # Highlight the focus entity if provided
        if focus_entity_id and focus_entity_id in subgraph:
            nx.draw_networkx_nodes(
                subgraph, pos,
                nodelist=[focus_entity_id],
                node_color='red',
                node_size=800,
                alpha=1.0
            )
            
        # Highlight the path if provided
        if highlight_path:
            path_edges = []
            for i in range(len(highlight_path) - 1):
                source = highlight_path[i]
                target = highlight_path[i + 1]
                if subgraph.has_edge(source, target):
                    path_edges.append((source, target))
                    
            nx.draw_networkx_edges(
                subgraph, pos,
                edgelist=path_edges,
                width=3,
                alpha=1.0,
                edge_color='red'
            )
            
        # Draw edges
        nx.draw_networkx_edges(
            subgraph, pos,
            width=1.0,
            alpha=0.5
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            subgraph, pos,
            labels={n: self.entities[n].name for n in subgraph.nodes() if n in self.entities},
            font_size=8
        )
        
        # Draw edge labels
        edge_labels = {}
        for u, v, data in subgraph.edges(data=True):
            edge_labels[(u, v)] = data.get('type', '')
            
        nx.draw_networkx_edge_labels(
            subgraph, pos,
            edge_labels=edge_labels,
            font_size=6
        )
        
        # Add legend for entity types
        plt.legend()
        
        # Remove axis
        plt.axis('off')
        
        # Add title
        if focus_entity_id and focus_entity_id in self.entities:
            plt.title(f"Knowledge Graph centered on '{self.entities[focus_entity_id].name}'")
        else:
            plt.title("Knowledge Graph Visualization")
            
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        logger.info(f"Graph visualization saved to {output_file}")
        return True
        
    def export_to_json(self, output_file: str) -> bool:
        """
        Export the knowledge graph to a JSON file
        
        Args:
            output_file: Output file path
            
        Returns:
            True if export was successful, False otherwise
        """
        try:
            # Prepare data for export
            data = {
                "entities": [],
                "relationships": []
            }
            
            # Export entities
            for entity in self.entities.values():
                entity_dict = {
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.type,
                    "properties": entity.properties,
                    "sources": entity.sources,
                    "confidence": entity.confidence,
                    "created_at": entity.created_at,
                    "last_updated": entity.last_updated,
                    "metadata": entity.metadata
                }
                data["entities"].append(entity_dict)
                
            # Export relationships
            for rel in self.relationships.values():
                rel_dict = {
                    "id": rel.id,
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "type": rel.type.name,
                    "name": rel.name,
                    "properties": rel.properties,
                    "sources": rel.sources,
                    "confidence": rel.confidence,
                    "created_at": rel.created_at,
                    "last_updated": rel.last_updated,
                    "bidirectional": rel.bidirectional,
                    "metadata": rel.metadata
                }
                data["relationships"].append(rel_dict)
                
            # Write to file
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Exported knowledge graph to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting knowledge graph: {e}")
            return False
            
    def import_from_json(self, input_file: str) -> Tuple[int, int]:
        """
        Import a knowledge graph from a JSON file
        
        Args:
            input_file: Input file path
            
        Returns:
            Tuple of (entities_imported, relationships_imported)
        """
        try:
            # Read the file
            with open(input_file, 'r') as f:
                data = json.load(f)
                
            entities_imported = 0
            relationships_imported = 0
            
            # Import entities
            for entity_dict in data.get("entities", []):
                entity = Entity(
                    id=entity_dict["id"],
                    name=entity_dict["name"],
                    type=entity_dict["type"],
                    properties=entity_dict.get("properties", {}),
                    sources=entity_dict.get("sources", []),
                    confidence=entity_dict.get("confidence", 1.0),
                    created_at=entity_dict.get("created_at", datetime.now().isoformat()),
                    last_updated=entity_dict.get("last_updated", datetime.now().isoformat()),
                    metadata=entity_dict.get("metadata", {})
                )
                self.add_entity(entity)
                entities_imported += 1
                
            # Import relationships
            for rel_dict in data.get("relationships", []):
                rel_type = RelationshipType[rel_dict["type"]] if rel_dict["type"] in RelationshipType.__members__ else RelationshipType.CUSTOM
                relationship = Relationship(
                    id=rel_dict["id"],
                    source_id=rel_dict["source_id"],
                    target_id=rel_dict["target_id"],
                    type=rel_type,
                    name=rel_dict.get("name"),
                    properties=rel_dict.get("properties", {}),
                    sources=rel_dict.get("sources", []),
                    confidence=rel_dict.get("confidence", 1.0),
                    created_at=rel_dict.get("created_at", datetime.now().isoformat()),
                    last_updated=rel_dict.get("last_updated", datetime.now().isoformat()),
                    bidirectional=rel_dict.get("bidirectional", False),
                    metadata=rel_dict.get("metadata", {})
                )
                try:
                    self.add_relationship(relationship)
                    relationships_imported += 1
                except ValueError as e:
                    logger.warning(f"Skipping relationship {relationship.id}: {e}")
                    
            logger.info(f"Imported {entities_imported} entities and {relationships_imported} relationships from {input_file}")
            return (entities_imported, relationships_imported)
            
        except Exception as e:
            logger.error(f"Error importing knowledge graph: {e}")
            return (0, 0)
            
    def close(self) -> None:
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None

class KnowledgeExtractor:
    """
    Extracts knowledge from various sources to build the knowledge graph.
    Supports text, structured data, and external APIs.
    """
    def __init__(self, knowledge_graph: KnowledgeGraph, api_key: Optional[str] = None):
        self.knowledge_graph = knowledge_graph
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = None
        
        if HAVE_OPENAI and self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key)
            logger.info("Using OpenAI for knowledge extraction")
            
    async def extract_from_text(self, text: str, source: str = "text") -> Tuple[int, int]:
        """
        Extract entities and relationships from text
        
        Args:
            text: Text to extract from
            source: Source identifier
            
        Returns:
            Tuple of (entities_added, relationships_added)
        """
        entities_added = 0
        relationships_added = 0
        
        if self.client:
            # Use OpenAI to extract entities and relationships
            try:
                # Create a prompt for entity extraction
                entity_prompt = f"""
                Extract entities from the following text. For each entity, provide:
                1. A unique identifier (use format like 'entity1', 'entity2', etc.)
                2. The entity name
                3. The entity type (e.g., Person, Organization, Concept, Location, etc.)
                4. Any properties of the entity mentioned in the text
                
                Format the output as a JSON array of objects.
                
                Text: {text}
                """
                
                entity_response = await self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a knowledge extraction assistant."},
                        {"role": "user", "content": entity_prompt}
                    ],
                    temperature=0.2
                )
                
                entity_content = entity_response.choices[0].message.content
                
                # Extract JSON from the response
                entity_json_match = re.search(r'```json\n(.*?)\n```', entity_content, re.DOTALL)
                if entity_json_match:
                    entity_json = entity_json_match.group(1)
                else:
                    entity_json = entity_content
                    
                try:
                    entities = json.loads(entity_json)
                    
                    # Add entities to the knowledge graph
                    entity_id_map = {}  # Map from extractor IDs to actual entity IDs
                    
                    for entity_data in entities:
                        extractor_id = entity_data.get("id")
                        name = entity_data.get("name")
                        entity_type = entity_data.get("type")
                        properties = entity_data.get("properties", {})
                        
                        if name and entity_type:
                            # Check if entity already exists
                            existing_entities = self.knowledge_graph.find_entities({
                                "name": name,
                                "type": entity_type
                            })
                            
                            if existing_entities:
                                # Use existing entity
                                entity_id = existing_entities[0].id
                                entity_id_map[extractor_id] = entity_id
                            else:
                                # Create new entity
                                entity = Entity(
                                    id="",  # Will be generated
                                    name=name,
                                    type=entity_type,
                                    properties=properties,
                                    sources=[source]
                                )
                                entity_id = self.knowledge_graph.add_entity(entity)
                                entity_id_map[extractor_id] = entity_id
                                entities_added += 1
                                
                    # Create a prompt for relationship extraction
                    relationship_prompt = f"""
                    Extract relationships between entities from the following text. For each relationship, provide:
                    1. The source entity ID
                    2. The target entity ID
                    3. The relationship type (e.g., IS_A, PART_OF, HAS_PROPERTY, RELATED_TO, DEPENDS_ON, CAUSES, etc.)
                    4. A name for the relationship (optional)
                    
                    Use these entity IDs:
                    {json.dumps(entity_id_map)}
                    
                    Format the output as a JSON array of objects.
                    
                    Text: {text}
                    """
                    
                    relationship_response = await self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a knowledge extraction assistant."},
                            {"role": "user", "content": relationship_prompt}
                        ],
                        temperature=0.2
                    )
                    
                    relationship_content = relationship_response.choices[0].message.content
                    
                    # Extract JSON from the response
                    rel_json_match = re.search(r'```json\n(.*?)\n```', relationship_content, re.DOTALL)
                    if rel_json_match:
                        rel_json = rel_json_match.group(1)
                    else:
                        rel_json = relationship_content
                        
                    try:
                        relationships = json.loads(rel_json)
                        
                        # Add relationships to the knowledge graph
                        for rel_data in relationships:
                            source_id = rel_data.get("source_id")
                            target_id = rel_data.get("target_id")
                            rel_type_str = rel_data.get("type")
                            name = rel_data.get("name")
                            
                            # Map source and target IDs
                            if source_id in entity_id_map:
                                source_id = entity_id_map[source_id]
                            if target_id in entity_id_map:
                                target_id = entity_id_map[target_id]
                                
                            # Get relationship type
                            try:
                                rel_type = RelationshipType[rel_type_str] if rel_type_str in RelationshipType.__members__ else RelationshipType.RELATED_TO
                            except (KeyError, TypeError):
                                rel_type = RelationshipType.RELATED_TO
                                
                            if source_id and target_id:
                                try:
                                    # Create relationship
                                    relationship = Relationship(
                                        id="",  # Will be generated
                                        source_id=source_id,
                                        target_id=target_id,
                                        type=rel_type,
                                        name=name,
                                        sources=[source]
                                    )
                                    self.knowledge_graph.add_relationship(relationship)
                                    relationships_added += 1
                                except ValueError as e:
                                    logger.warning(f"Skipping relationship: {e}")
                                    
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing relationship JSON: {e}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing entity JSON: {e}")
                    
            except Exception as e:
                logger.error(f"Error extracting knowledge with OpenAI: {e}")
                
        else:
            # Fallback to simple regex-based extraction
            logger.warning("Using simple regex-based extraction (limited capabilities)")
            
            # Extract potential entities using regex
            # This is a very simplified approach and won't work well for most texts
            entity_patterns = [
                (r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', 'Person'),  # Capitalized words
                (r'\b([A-Z][A-Z]+)\b', 'Organization'),  # Acronyms
                (r'\b(\d+(?:\.\d+)?)\b', 'Number')  # Numbers
            ]
            
            entities = {}
            
            for pattern, entity_type in entity_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    name = match.group(1)
                    
                    # Check if entity already exists
                    existing_entities = self.knowledge_graph.find_entities({
                        "name": name,
                        "type": entity_type
                    })
                    
                    if existing_entities:
                        # Use existing entity
                        entity_id = existing_entities[0].id
                        entities[name] = entity_id
                    else:
                        # Create new entity
                        entity = Entity(
                            id="",  # Will be generated
                            name=name,
                            type=entity_type,
                            sources=[source]
                        )
                        entity_id = self.knowledge_graph.add_entity(entity)
                        entities[name] = entity_id
                        entities_added += 1
                        
            # Extract potential relationships
            # This is extremely simplified and won't work well
            for name1, id1 in entities.items():
                for name2, id2 in entities.items():
                    if name1 != name2:
                        # Check if they appear close to each other in the text
                        if name1 in text and name2 in text:
                            pos1 = text.find(name1)
                            pos2 = text.find(name2)
                            
                            if abs(pos1 - pos2) < 100:  # Arbitrary proximity threshold
                                try:
                                    # Create a generic relationship
                                    relationship = Relationship(
                                        id="",  # Will be generated
                                        source_id=id1,
                                        target_id=id2,
                                        type=RelationshipType.RELATED_TO,
                                        name="related to",
                                        sources=[source],
                                        confidence=0.5  # Low confidence for this simple method
                                    )
                                    self.knowledge_graph.add_relationship(relationship)
                                    relationships_added += 1
                                except ValueError:
                                    pass
                                    
        logger.info(f"Extracted {entities_added} entities and {relationships_added} relationships from text")
        return (entities_added, relationships_added)
        
    async def extract_from_structured_data(self, data: Dict[str, Any], 
                                        entity_type_field: str = "type",
                                        source: str = "structured_data") -> Tuple[int, int]:
        """
        Extract entities and relationships from structured data
        
        Args:
            data: Structured data (e.g., JSON)
            entity_type_field: Field name for entity type
            source: Source identifier
            
        Returns:
            Tuple of (entities_added, relationships_added)
        """
        entities_added = 0
        relationships_added = 0
        
        # Process entities
        if "entities" in data:
            for entity_data in data["entities"]:
                # Extract entity fields
                name = entity_data.get("name")
                entity_type = entity_data.get(entity_type_field)
                
                if not name or not entity_type:
                    continue
                    
                # Extract properties
                properties = {}
                for key, value in entity_data.items():
                    if key not in ["name", entity_type_field, "id"]:
                        properties[key] = value
                        
                # Create entity
                entity = Entity(
                    id=entity_data.get("id", ""),  # Will be generated if empty
                    name=name,
                    type=entity_type,
                    properties=properties,
                    sources=[source]
                )
                self.knowledge_graph.add_entity(entity)
                entities_added += 1
                
        # Process relationships
        if "relationships" in data:
            for rel_data in data["relationships"]:
                source_id = rel_data.get("source_id")
                target_id = rel_data.get("target_id")
                rel_type_str = rel_data.get("type")
                
                if not source_id or not target_id or not rel_type_str:
                    continue
                    
                # Get relationship type
                try:
                    rel_type = RelationshipType[rel_type_str] if rel_type_str in RelationshipType.__members__ else RelationshipType.RELATED_TO
                except (KeyError, TypeError):
                    rel_type = RelationshipType.RELATED_TO
                    
                # Extract properties
                properties = {}
                for key, value in rel_data.items():
                    if key not in ["source_id", "target_id", "type", "id", "name"]:
                        properties[key] = value
                        
                try:
                    # Create relationship
                    relationship = Relationship(
                        id=rel_data.get("id", ""),  # Will be generated if empty
                        source_id=source_id,
                        target_id=target_id,
                        type=rel_type,
                        name=rel_data.get("name"),
                        properties=properties,
                        sources=[source]
                    )
                    self.knowledge_graph.add_relationship(relationship)
                    relationships_added += 1
                except ValueError as e:
                    logger.warning(f"Skipping relationship: {e}")
                    
        logger.info(f"Extracted {entities_added} entities and {relationships_added} relationships from structured data")
        return (entities_added, relationships_added)
        
    async def extract_from_file(self, file_path: str) -> Tuple[int, int]:
        """
        Extract knowledge from a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (entities_added, relationships_added)
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return (0, 0)
            
        # Determine file type
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.json':
            # Process JSON file
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                return await self.extract_from_structured_data(data, source=file_path)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON file: {e}")
                return (0, 0)
                
        elif file_ext in ['.txt', '.md', '.rst']:
            # Process text file
            try:
                with open(file_path, 'r') as f:
                    text = f.read()
                return await self.extract_from_text(text, source=file_path)
            except Exception as e:
                logger.error(f"Error reading text file: {e}")
                return (0, 0)
                
        else:
            logger.warning(f"Unsupported file type: {file_ext}")
            return (0, 0)

# Example usage
async def example_usage():
    # Create a knowledge graph
    kg = KnowledgeGraph("example_knowledge_graph.db")
    
    # Create some entities
    person1 = Entity(
        id="",
        name="John Smith",
        type="Person",
        properties={"age": 30, "occupation": "Software Engineer"}
    )
    person1_id = kg.add_entity(person1)
    
    person2 = Entity(
        id="",
        name="Jane Doe",
        type="Person",
        properties={"age": 28, "occupation": "Data Scientist"}
    )
    person2_id = kg.add_entity(person2)
    
    company = Entity(
        id="",
        name="Tech Corp",
        type="Organization",
        properties={"industry": "Technology", "founded": 2010}
    )
    company_id = kg.add_entity(company)
    
    project = Entity(
        id="",
        name="Knowledge Graph Project",
        type="Project",
        properties={"status": "In Progress", "priority": "High"}
    )
    project_id = kg.add_entity(project)
    
    # Create relationships
    kg.add_relationship(Relationship(
        id="",
        source_id=person1_id,
        target_id=company_id,
        type=RelationshipType.PART_OF,
        name="works at"
    ))
    
    kg.add_relationship(Relationship(
        id="",
        source_id=person2_id,
        target_id=company_id,
        type=RelationshipType.PART_OF,
        name="works at"
    ))
    
    kg.add_relationship(Relationship(
        id="",
        source_id=person1_id,
        target_id=project_id,
        type=RelationshipType.RELATED_TO,
        name="works on"
    ))
    
    kg.add_relationship(Relationship(
        id="",
        source_id=person2_id,
        target_id=project_id,
        type=RelationshipType.RELATED_TO,
        name="works on"
    ))
    
    kg.add_relationship(Relationship(
        id="",
        source_id=project_id,
        target_id=company_id,
        type=RelationshipType.PART_OF,
        name="belongs to"
    ))
    
    # Add an inference rule
    kg.add_inference_rule(
        name="Transitive PART_OF",
        description="If A is part of B and B is part of C, then A is part of C",
        pattern={"type": "transitive", "relationship_type": "PART_OF"}
    )
    
    # Apply inference rules
    new_relationships = kg.apply_inference_rules()
    print(f"Inferred {new_relationships} new relationships")
    
    # Find paths between entities
    paths = kg.find_paths(person1_id, company_id)
    print(f"Found {len(paths)} paths from {person1.name} to {company.name}")
    
    for i, path in enumerate(paths):
        print(f"Path {i+1}:")
        for j, (entity, relationship) in enumerate(path):
            if j < len(path) - 1:
                print(f"  {entity.name} --[{relationship.name}]--> ", end="")
            else:
                print(f"{entity.name}")
                
    # Visualize the knowledge graph
    if HAVE_NETWORKX and HAVE_MATPLOTLIB:
        kg.visualize("example_knowledge_graph.png", focus_entity_id=person1_id)
        print("Knowledge graph visualization saved to example_knowledge_graph.png")
        
    # Export the knowledge graph
    kg.export_to_json("example_knowledge_graph.json")
    print("Knowledge graph exported to example_knowledge_graph.json")
    
    # Close the knowledge graph
    kg.close()

if __name__ == "__main__":
    asyncio.run(example_usage())
