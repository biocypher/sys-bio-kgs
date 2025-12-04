#!/usr/bin/env python3
"""
Similarity scoring module for comparing nodes across multiple SBGN models.

This module generates semantic embeddings for nodes and computes pairwise similarities
between nodes from different models, returning similarity edges in BioCypher format.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Iterator, Tuple, Dict, Any, List
import logging
import re

logger = logging.getLogger(__name__)


def strip_hash_prefix(node_id: str) -> str:
    """
    Strip hash prefix from node ID if present.
    
    The adapter adds a hash prefix (e.g., 'abc123_glyph1') to make IDs unique
    across different model instances. For similarity comparison, we need to
    remove this prefix so that identical nodes from the same model loaded
    multiple times will match perfectly.
    
    Args:
        node_id: Node ID potentially with hash prefix
    
    Returns:
        Node ID without hash prefix
    """
    if '_' in node_id:
        # Check if first part looks like a hex hash (16 characters)
        parts = node_id.split('_', 1)
        if len(parts[0]) == 16 and all(c in '0123456789abcdef' for c in parts[0]):
            return parts[1]
    return node_id


# Regex patterns for extracting database identifiers
DB_ID_PATTERNS = [
    r'(CHEBI:\d+)',
    r'(UniProt[KR]*[:/]\w+)',
    r'(GO:\d+)',
    r'(KEGG[:/]\w+)',
    r'(Reactome[:/]R-[A-Z]+-\d+)',
    r'(Ensembl[:/]ENS\w+)',
    r'(PubChem[:/]\d+)',
    r'(HMDB\d+)',
    r'(SBO:\d+)',
    r'([A-Z]+:\d+)',  # Generic pattern for other ontology IDs
]
DB_ID_REGEX = re.compile('|'.join(DB_ID_PATTERNS), re.IGNORECASE)


def extract_database_ids(text: str) -> List[str]:
    """
    Extract database identifiers from text using regex patterns.
    
    Args:
        text: Text containing potential database identifiers
    
    Returns:
        List of unique database identifiers found
    """
    if not text:
        return []
    
    matches = DB_ID_REGEX.findall(text)
    # Flatten tuples from findall and filter empty strings
    ids = [match for group in matches for match in (group if isinstance(group, tuple) else [group]) if match]
    return list(set(ids))  # Return unique IDs


def extract_node_components(node_type: str, properties: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract different text components from a node for separate embedding generation.
    
    NOTE: Node IDs are intentionally EXCLUDED to avoid including adapter-specific
    hash prefixes or any ID information that would prevent identical nodes from matching.
    
    Components extracted:
    - names: Node name/label (primary biological identifier)
    - annotations: Biological database identifiers (e.g., CHEBI, UniProt)
    - structure: Type, class, SBO term (structural/ontological info)
    - db_ids: Database identifiers extracted via regex
    
    Args:
        node_type: Type of the node
        properties: Dictionary of node properties
    
    Returns:
        Dictionary with keys: names, annotations, structure, db_ids
    """
    components = {
        'names': [],
        'annotations': [],
        'structure': [],
        'db_ids': []
    }
    
    # Extract name/label (most important biological identifier)
    name_value = properties.get('name') or properties.get('label')
    if name_value:
        components['names'].append(str(name_value))
    
    # Extract structural/ontological information
    components['structure'].append(f"Type: {node_type}")
    if 'sbgn_class' in properties:
        components['structure'].append(f"Class: {properties['sbgn_class']}")
    if 'sbo_term' in properties:
        components['structure'].append(f"SBO: {properties['sbo_term']}")
    if 'unit_of_information' in properties:
        info_list = properties['unit_of_information']
        if isinstance(info_list, list):
            components['structure'].append(f"Info: {', '.join(info_list)}")
        else:
            components['structure'].append(f"Info: {info_list}")
    
    # Extract annotations (biological database identifiers)
    for key, value in properties.items():
        if key.startswith('BQB_') or key.startswith('BQM_'):
            value_str = ', '.join(str(v) for v in value) if isinstance(value, list) else str(value)
            components['annotations'].append(f"{key}: {value_str}")
    
    # Extract other properties (excluding spatial/layout info)
    for key, value in properties.items():
        if key not in ['name', 'label', 'sbgn_class', 'sbo_term', 'unit_of_information',
                       'x', 'y', 'width', 'height', 'orientation',  # Exclude layout
                       'sbgn_arc_id', 'sbgn_arc_class',  # Exclude arc-specific
                       'start_x', 'start_y', 'end_x', 'end_y', 'intermediate_points'] and \
           not key.startswith('BQB_') and not key.startswith('BQM_'):
            value_str = ', '.join(str(v) for v in value) if isinstance(value, list) else str(value)
            components['structure'].append(f"{key}: {value_str}")
    
    # Extract database IDs using regex from all text
    all_text = ' '.join(components['names'] + components['annotations'] + components['structure'])
    components['db_ids'] = extract_database_ids(all_text)
    
    # Convert lists to strings
    return {
        'names': ' | '.join(components['names']) if components['names'] else '',
        'annotations': ' | '.join(components['annotations']) if components['annotations'] else '',
        'structure': ' | '.join(components['structure']) if components['structure'] else '',
        'db_ids': ' '.join(components['db_ids']) if components['db_ids'] else ''
    }


def compute_pairwise_similarities(
    node_iterators: List[Iterator[Tuple[str, str, Dict[str, Any]]]],
    edge_iterators: List[Iterator[Tuple[str, str, str, str, Dict[str, Any]]]] = None,
    model_names: List[str] = None,
    embedding_model: str = 'all-mpnet-base-v2',
    similarity_threshold: float = 0.0,
    top_k: int = None,
    weight_names: float = 5.0,
    weight_annotations: float = 1.0,
    weight_db_ids: float = 10,
    weight_process_context: float = 5.0,
    penalty_structure: float = 2.0
) -> Iterator[Tuple[str, str, str, str, Dict[str, Any]]]:
    """
    Compare nodes from multiple iterators and generate similarity edges.
    
    Uses a two-pass approach:
    1. First pass: Compute similarities for all nodes based on intrinsic properties
    2. Second pass: Update process node similarities based on connected node similarities
    
    Uses separate transformer encodings for different text components with two modes:
    - Additive components (weighted average): names, annotations, db_ids
    - Penalty components (multiplicative): structure (node type)
    
    Penalty components reduce similarity when dissimilar but don't increase it when similar.
    This is useful for node types where mismatches should be heavily penalized, but matches
    shouldn't artificially boost the score.
    
    Args:
        node_iterators: List of node iterators from different models
        edge_iterators: Optional list of edge iterators (for process node context)
        model_names: Optional list of model names for better edge IDs
        embedding_model: Name of the sentence-transformers model to use
        similarity_threshold: Minimum similarity score to include (0.0 to 1.0)
        top_k: If specified, only return top k most similar pairs for each node
        weight_names: Weight for name similarity (default: 3.0)
        weight_annotations: Weight for annotation similarity (default: 2.0)
        weight_db_ids: Weight for database ID similarity (default: 2.5)
        weight_process_context: Weight for process context (connected nodes) (default: 2.0)
        penalty_structure: Penalty strength for structural mismatch (default: 2.0).
                          Higher values = stronger penalty for low structural similarity.
                          Applied as: similarity * (structure_sim ** penalty_structure)
    
    Yields:
        Tuples of (edge_id, source_id, target_id, edge_type, properties_dict)
        where edge_type is "similarity" and properties contain the similarity weight
    """
    model = SentenceTransformer(embedding_model)
    
    # Normalize weights for additive components only (structure is penalty-based)
    total_weight = weight_names + weight_annotations + weight_db_ids
    w_names = weight_names / total_weight
    w_annotations = weight_annotations / total_weight
    w_db_ids = weight_db_ids / total_weight
    
    # Convert iterators to lists and extract components for each model
    all_nodes = []  # List of (model_idx, node_id, node_type, properties, component_embeddings_dict)
    
    for model_idx, node_iterator in enumerate(node_iterators):
        model_name = model_names[model_idx] if model_names else f"model_{model_idx}"
        
        nodes_list = list(node_iterator)
        
        # Build edge context for process nodes if edges provided
        process_context = {}
        if edge_iterators and model_idx < len(edge_iterators):
            edges_list = list(edge_iterators[model_idx])
            for edge_id, source_id, target_id, edge_type, props in edges_list:
                # Track reactants/consumption (incoming to process: entity -> process)
                if edge_type in ['reactant', 'consumption']:
                    if target_id not in process_context:
                        process_context[target_id] = {'reactants': [], 'products': [], 'modifiers': []}
                    process_context[target_id]['reactants'].append(source_id)
                # Track products/production (outgoing from process: process -> entity)
                elif edge_type in ['product', 'production']:
                    if source_id not in process_context:
                        process_context[source_id] = {'reactants': [], 'products': [], 'modifiers': []}
                    process_context[source_id]['products'].append(target_id)
                # Track modifiers (all modulation types: entity -> process)
                elif edge_type in ['inhibition', 'catalysis', 'stimulation', 'modulation', 'necessary stimulation', 'modifier']:
                    if target_id not in process_context:
                        process_context[target_id] = {'reactants': [], 'products': [], 'modifiers': []}
                    process_context[target_id]['modifiers'].append(source_id)
        
        # Extract components for all nodes
        node_data = []
        components_dict = {'names': [], 'annotations': [], 'structure': [], 'db_ids': []}
        has_content = []  # Track which components have content for each node
        
        for node_id, node_type, properties in nodes_list:
            # Extract components (IDs are never included)
            components = extract_node_components(node_type, properties)
            
            # Track which components have actual content (not empty)
            node_has_content = {
                'names': bool(components['names']),
                'annotations': bool(components['annotations']),
                'structure': bool(components['structure']),
                'db_ids': bool(components['db_ids'])
            }
            has_content.append(node_has_content)
            
            # Collect texts for batch encoding
            components_dict['names'].append(components['names'] or ' ')  # Avoid empty strings
            components_dict['annotations'].append(components['annotations'] or ' ')
            components_dict['structure'].append(components['structure'] or ' ')
            components_dict['db_ids'].append(components['db_ids'] or ' ')
            
            # Store node data with process context for later use
            context_info = process_context.get(node_id, {})
            node_data.append((node_id, node_type, properties, context_info, node_has_content))
        
        # Generate embeddings for each component separately
        if node_data:
            embeddings_dict = {}
            for component_name, texts in components_dict.items():
                embeddings_dict[component_name] = model.encode(texts, show_progress_bar=False, 
                                                              convert_to_numpy=True)
            
            # Store with model index
            for idx, (node_id, node_type, properties, context_info, node_has_content) in enumerate(node_data):
                component_embeddings = {
                    'names': embeddings_dict['names'][idx],
                    'annotations': embeddings_dict['annotations'][idx],
                    'structure': embeddings_dict['structure'][idx],
                    'db_ids': embeddings_dict['db_ids'][idx]
                }
                all_nodes.append((model_idx, node_id, node_type, properties, component_embeddings, context_info, node_has_content))
    
    # Build component embedding matrices for efficient similarity computation
    n_nodes = len(all_nodes)
    embedding_dim = all_nodes[0][4]['names'].shape[0]  # Get embedding dimension
    
    # Initialize matrices for each component
    emb_matrices = {
        'names': np.zeros((n_nodes, embedding_dim)),
        'annotations': np.zeros((n_nodes, embedding_dim)),
        'structure': np.zeros((n_nodes, embedding_dim)),
        'db_ids': np.zeros((n_nodes, embedding_dim))
    }
    
    # Fill matrices
    for idx, (_, _, _, _, component_embeddings, _, _) in enumerate(all_nodes):
        for component_name in emb_matrices.keys():
            emb_matrices[component_name][idx] = component_embeddings[component_name]
    
    # Normalize embeddings for cosine similarity (makes dot product = cosine similarity)
    for component_name in emb_matrices.keys():
        norms = np.linalg.norm(emb_matrices[component_name], axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        emb_matrices[component_name] = emb_matrices[component_name] / norms
    
    # FIRST PASS: Compute similarity matrices for intrinsic node properties
    sim_matrices = {}
    for component_name, emb_matrix in emb_matrices.items():
        # Matrix multiplication gives all pairwise cosine similarities
        sim_matrices[component_name] = np.dot(emb_matrix, emb_matrix.T)
    
    # Combine similarity matrices with dynamic weights based on content
    # Create a matrix to track which components should contribute to each pair
    # Component contributes only if BOTH nodes have content for that component
    n_nodes = len(all_nodes)
    intrinsic_similarity_matrix = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        has_content_i = all_nodes[i][6]  # has_content dict for node i
        
        for j in range(n_nodes):
            has_content_j = all_nodes[j][6]  # has_content dict for node j
            
            # Determine which additive components both nodes have
            active_components = []
            active_weights = []
            
            if has_content_i['names'] and has_content_j['names']:
                active_components.append('names')
                active_weights.append(weight_names)
            
            if has_content_i['annotations'] and has_content_j['annotations']:
                active_components.append('annotations')
                active_weights.append(weight_annotations)
            
            if has_content_i['db_ids'] and has_content_j['db_ids']:
                active_components.append('db_ids')
                active_weights.append(weight_db_ids)
            
            # Compute weighted similarity only from components both nodes have
            if active_components:
                total_active_weight = sum(active_weights)
                additive_sim = 0.0
                for comp_name, weight in zip(active_components, active_weights):
                    normalized_weight = weight / total_active_weight
                    additive_sim += normalized_weight * sim_matrices[comp_name][i, j]
            else:
                # If no shared additive components, use names only as fallback
                additive_sim = sim_matrices['names'][i, j]
            
            # Apply structure penalty (always applied since structure always has content)
            structure_penalty = np.power(sim_matrices['structure'][i, j], penalty_structure)
            
            intrinsic_similarity_matrix[i, j] = additive_sim * structure_penalty
    
    # SECOND PASS: Update process node similarities based on connected nodes
    final_similarity_matrix = intrinsic_similarity_matrix.copy()
    
    # Identify process nodes
    process_indices = []
    for idx, (_, _, node_type, _, _, context_info, _) in enumerate(all_nodes):
        if node_type == 'process' and context_info:
            process_indices.append(idx)
    
    # For each pair of process nodes, compute similarity based on connected nodes
    updated_count = 0
    for i in process_indices:
        model_idx_i, node_id_i, _, _, _, context_i, _ = all_nodes[i]
        
        # Get connected node IDs
        connected_i = set()
        connected_i.update(context_i.get('reactants', []))
        connected_i.update(context_i.get('products', []))
        connected_i.update(context_i.get('modifiers', []))
        
        # Find indices of connected nodes
        connected_idx_i = []
        for idx, (model_idx, node_id, _, _, _, _, _) in enumerate(all_nodes):
            if model_idx == model_idx_i and node_id in connected_i:
                connected_idx_i.append(idx)
        
        if not connected_idx_i:
            continue
        
        for j in process_indices:
            if i >= j:  # Only compute upper triangle
                continue
            
            model_idx_j, node_id_j, _, _, _, context_j, _ = all_nodes[j]
            
            # Only compare nodes from different models
            if model_idx_i == model_idx_j:
                continue
            
            # Get connected node IDs
            connected_j = set()
            connected_j.update(context_j.get('reactants', []))
            connected_j.update(context_j.get('products', []))
            connected_j.update(context_j.get('modifiers', []))
            
            # Find indices of connected nodes
            connected_idx_j = []
            for idx, (model_idx, node_id, _, _, _, _, _) in enumerate(all_nodes):
                if model_idx == model_idx_j and node_id in connected_j:
                    connected_idx_j.append(idx)
            
            if not connected_idx_j:
                continue
            
            # Compute similarity between connected nodes using optimal matching
            # Strategy: Find the best one-to-one pairing between connected nodes
            # that maximizes total similarity (greedy approach)
            
            # Build similarity matrix for connected nodes
            n_connected_i = len(connected_idx_i)
            n_connected_j = len(connected_idx_j)
            
            # Handle empty sets: if either has no connected nodes, context similarity is 0
            if n_connected_i == 0 and n_connected_j == 0:
                # Both processes have no connected nodes - they're similar in this aspect
                avg_context_sim = 1.0
                n_matched = 0
                n_total = 0
            elif n_connected_i == 0 or n_connected_j == 0:
                # One has connections, the other doesn't - very dissimilar
                avg_context_sim = 0.0
                n_matched = 0
                n_total = max(n_connected_i, n_connected_j)
            else:
                # Both have connected nodes - compute optimal matching
                context_similarities = []
                
                # Create pairwise similarity matrix for connected nodes
                connected_sim_matrix = np.zeros((n_connected_i, n_connected_j))
                for ii, idx_i in enumerate(connected_idx_i):
                    for jj, idx_j in enumerate(connected_idx_j):
                        sim = final_similarity_matrix[min(idx_i, idx_j), max(idx_i, idx_j)]
                        connected_sim_matrix[ii, jj] = sim
                
                # Greedy matching: repeatedly find the highest similarity pair
                # and remove both nodes from consideration
                used_i = set()
                used_j = set()
                matched_pairs = []
                
                # Sort all pairs by similarity (descending)
                pairs = []
                for ii in range(n_connected_i):
                    for jj in range(n_connected_j):
                        pairs.append((connected_sim_matrix[ii, jj], ii, jj))
                pairs.sort(reverse=True)
                
                # Select pairs greedily (each node used only once)
                for sim, ii, jj in pairs:
                    if ii not in used_i and jj not in used_j:
                        matched_pairs.append((ii, jj, sim))
                        used_i.add(ii)
                        used_j.add(jj)
                        context_similarities.append(sim)
                
                # Compute average similarity considering unmatched nodes
                n_matched = len(matched_pairs)
                n_total = max(n_connected_i, n_connected_j)
                n_unmatched = n_total - n_matched
                
                # Unmatched nodes contribute 0 similarity
                # This penalizes processes with different numbers of connections
                total_similarity = sum(context_similarities) + (n_unmatched * 0.0)
                avg_context_sim = total_similarity / n_total if n_total > 0 else 0.0
            
            # Combine intrinsic similarity with context similarity
            intrinsic_sim = intrinsic_similarity_matrix[i, j]
            combined_sim = (intrinsic_sim + weight_process_context * avg_context_sim) / (1 + weight_process_context)
            
            # Update both symmetric positions
            final_similarity_matrix[i, j] = combined_sim
            final_similarity_matrix[j, i] = combined_sim
            updated_count += 1
    
    # Generate edges from similarity matrix
    edge_count = 0
    similarity_counts = {}
    
    for i in range(n_nodes):
        model_idx_i, node_id_i, node_type_i, props_i, _, _, has_content_i = all_nodes[i]
        model_name_i = model_names[model_idx_i] if model_names else f"model_{model_idx_i}"
        
        # Collect similarities for this node if top_k is specified
        node_similarities = []
        
        for j in range(i + 1, n_nodes):
            model_idx_j, node_id_j, node_type_j, props_j, _, _, has_content_j = all_nodes[j]
            
            # Only compare nodes from different models
            if model_idx_i == model_idx_j:
                continue
            
            # Skip empty set nodes
            if node_type_i == 'empty set' or node_type_j == 'empty set':
                continue
            
            model_name_j = model_names[model_idx_j] if model_names else f"model_{model_idx_j}"
            
            # Get similarity from precomputed matrix
            similarity = float(final_similarity_matrix[i, j])
            
            # Apply threshold
            if similarity < similarity_threshold:
                continue
            
            if top_k is not None:
                node_similarities.append((j, similarity, node_id_j))
            else:
                # Generate edge immediately
                edge_id = f"similarity_{model_name_i}_{node_id_i}_{model_name_j}_{node_id_j}"
                
                # Calculate which components are active for this pair
                active_components = []
                if has_content_i['names'] and has_content_j['names']:
                    active_components.append('names')
                if has_content_i['annotations'] and has_content_j['annotations']:
                    active_components.append('annotations')
                if has_content_i['db_ids'] and has_content_j['db_ids']:
                    active_components.append('db_ids')
                
                # Calculate additive similarity before penalty
                active_comp_str = ', '.join(active_components) if active_components else 'names (fallback)'
                structure_pen = float(np.power(sim_matrices['structure'][i, j], penalty_structure))
                additive_sim = float(intrinsic_similarity_matrix[i, j] / structure_pen) if structure_pen > 0 else 0.0
                
                properties = {
                    'weight': similarity,
                    'similarity_score': similarity,
                    'source_model': model_name_i,
                    'target_model': model_name_j,
                    'source_type': node_type_i,
                    'target_type': node_type_j,
                    'sim_names': float(sim_matrices['names'][i, j]),
                    'sim_annotations': float(sim_matrices['annotations'][i, j]),
                    'sim_structure': float(sim_matrices['structure'][i, j]),
                    'sim_db_ids': float(sim_matrices['db_ids'][i, j]),
                    'active_components': active_comp_str,
                    'additive_sim': additive_sim,
                    'structure_penalty': structure_pen,
                    'intrinsic_sim': float(intrinsic_similarity_matrix[i, j])
                }
                
                yield (edge_id, node_id_i, node_id_j, "similarity", properties)
                edge_count += 1
                
                # Track statistics
                model_pair = tuple(sorted([model_name_i, model_name_j]))
                similarity_counts[model_pair] = similarity_counts.get(model_pair, 0) + 1
        
        # If using top_k, yield only the top k similarities for this node
        if top_k is not None and node_similarities:
            # Sort by similarity (descending) and take top k
            node_similarities.sort(key=lambda x: x[1], reverse=True)
            
            for j, similarity, node_id_j in node_similarities[:top_k]:
                model_idx_j, node_id_j, node_type_j, props_j, _, _, has_content_j = all_nodes[j]
                model_name_j = model_names[model_idx_j] if model_names else f"model_{model_idx_j}"
                
                edge_id = f"similarity_{model_name_i}_{node_id_i}_{model_name_j}_{node_id_j}"
                
                # Calculate which components are active for this pair
                active_components = []
                if has_content_i['names'] and has_content_j['names']:
                    active_components.append('names')
                if has_content_i['annotations'] and has_content_j['annotations']:
                    active_components.append('annotations')
                if has_content_i['db_ids'] and has_content_j['db_ids']:
                    active_components.append('db_ids')
                
                # Calculate additive similarity before penalty
                active_comp_str = ', '.join(active_components) if active_components else 'names (fallback)'
                structure_pen = float(np.power(sim_matrices['structure'][i, j], penalty_structure))
                additive_sim = float(intrinsic_similarity_matrix[i, j] / structure_pen) if structure_pen > 0 else 0.0
                
                properties = {
                    'weight': similarity,
                    'similarity_score': similarity,
                    'source_model': model_name_i,
                    'target_model': model_name_j,
                    'source_type': node_type_i,
                    'target_type': node_type_j,
                    'sim_names': float(sim_matrices['names'][i, j]),
                    'sim_annotations': float(sim_matrices['annotations'][i, j]),
                    'sim_structure': float(sim_matrices['structure'][i, j]),
                    'sim_db_ids': float(sim_matrices['db_ids'][i, j]),
                    'active_components': active_comp_str,
                    'additive_sim': additive_sim,
                    'structure_penalty': structure_pen,
                    'intrinsic_sim': float(intrinsic_similarity_matrix[i, j])
                }
                
                yield (edge_id, node_id_i, node_id_j, "similarity", properties)
                edge_count += 1
                
                # Track statistics
                model_pair = tuple(sorted([model_name_i, model_name_j]))
                similarity_counts[model_pair] = similarity_counts.get(model_pair, 0) + 1
