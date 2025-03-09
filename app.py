import networkx as nx
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine

class GraphRAG:
    def __init__(self, api_key):
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-3-7-sonnet-20250219"  # Use the latest available model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_graph = nx.Graph()
        self.node_embeddings = {}
        
    def add_document(self, doc_id, content, metadata=None):
        """Add a document to the knowledge graph."""
        # Add document node
        self.knowledge_graph.add_node(doc_id, type='document', content=content, metadata=metadata)
        # Get embedding for the document
        self.node_embeddings[doc_id] = self.embedding_model.encode(content)
        
        # Extract entities (simplified - in practice use NER models)
        # This is a placeholder for actual entity extraction
        entities = self._extract_entities(content)
        
        # Add entities and relationships to graph
        for entity in entities:
            entity_id = f"entity_{entity['name']}"
            if entity_id not in self.knowledge_graph:
                self.knowledge_graph.add_node(entity_id, type='entity', name=entity['name'])
                self.node_embeddings[entity_id] = self.embedding_model.encode(entity['name'])
            
            # Connect document to entity
            self.knowledge_graph.add_edge(doc_id, entity_id, type='contains')
            
            # Connect entities to each other based on co-occurrence
            for other_entity in entities:
                if entity != other_entity:
                    other_id = f"entity_{other_entity['name']}"
                    self.knowledge_graph.add_edge(entity_id, other_id, type='related')
    
    def _extract_entities(self, content):
        """
        Placeholder for entity extraction.
        In a real implementation, use NER models like spaCy or transformers.
        """
        # Simplified entity extraction - replace with actual NER
        words = content.split()
        # Pretend some words are entities (obviously simplistic)
        entities = [{"name": word} for word in words if len(word) > 5 and word.isalpha()]
        return entities
    
    def retrieve(self, query, max_results=5):
        """Retrieve relevant nodes from the knowledge graph based on the query."""
        query_embedding = self.embedding_model.encode(query)
        
        # Find most similar nodes based on embedding similarity
        similarities = {}
        for node_id, embedding in self.node_embeddings.items():
            similarity = 1 - cosine(query_embedding, embedding)
            similarities[node_id] = similarity
        
        # Get top nodes
        top_nodes = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:max_results]
        
        # Extract subgraph around these nodes (including 1-hop neighbors)
        relevant_nodes = [node_id for node_id, _ in top_nodes]
        for node_id, _ in top_nodes:
            relevant_nodes.extend(list(self.knowledge_graph.neighbors(node_id)))
        
        subgraph = self.knowledge_graph.subgraph(relevant_nodes)
        
        # Format retrieval results
        results = []
        for node in subgraph.nodes(data=True):
            node_id, data = node
            if data.get('type') == 'document':
                results.append({
                    'id': node_id,
                    'content': data.get('content'),
                    'type': 'document',
                    'connections': list(subgraph.neighbors(node_id))
                })
            elif data.get('type') == 'entity':
                results.append({
                    'id': node_id,
                    'name': data.get('name'),
                    'type': 'entity',
                    'connections': list(subgraph.neighbors(node_id))
                })
        
        return results
    
    def generate(self, query, system_prompt="You are a helpful assistant."):
        """Generate a response using Claude with retrieved context."""
        # Retrieve relevant information from the knowledge graph
        retrieved_info = self.retrieve(query)
        
        # Format the retrieved information as context
        context = self._format_context(retrieved_info)
        
        # Generate response from Claude
        response = self.client.messages.create(
            model=self.model,
            system=f"{system_prompt}\n\nRelevant context information: {context}",
            messages=[
                {"role": "user", "content": query}
            ],
            max_tokens=1000
        )
        
        return response.content[0].text
    
    def _format_context(self, retrieved_info):
        """Format retrieved information as context for Claude."""
        context_parts = []
        
        # Add document content
        for item in retrieved_info:
            if item['type'] == 'document':
                context_parts.append(f"Document: {item['content']}")
            elif item['type'] == 'entity':
                # For entities, add information about connections
                connected_docs = [conn for conn in item['connections'] 
                                 if conn.startswith('doc_')]
                entity_info = f"Entity: {item['name']}"
                if connected_docs:
                    entity_info += f" (mentioned in: {', '.join(connected_docs)})"
                context_parts.append(entity_info)
        
        return "\n\n".join(context_parts)


# Example usage
if __name__ == "__main__":
    graph_rag = GraphRAG(api_key="your_anthropic_api_key")
    
    # Add documents to the knowledge graph
    graph_rag.add_document("doc_1", "Claude is an AI assistant created by Anthropic.", 
                          {"source": "website"})
    graph_rag.add_document("doc_2", "Anthropic was founded in 2021 by former OpenAI researchers.", 
                          {"source": "news"})
    graph_rag.add_document("doc_3", "Graph RAG enhances retrieval by using graph structures.", 
                          {"source": "research paper"})
    
    # Generate a response
    response = graph_rag.generate("Tell me about Anthropic and their AI models.")
    print(response)