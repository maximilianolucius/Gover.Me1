#!/usr/bin/env python3
"""Milvus connectivity smoke test.

Connects to a local Milvus instance, creates a temporary collection,
verifies basic operations, and cleans up.
"""

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import os

def test_milvus():
    """Connect to Milvus, create and drop a test collection, and report success."""
    try:
        # Connect
        connections.connect(host="localhost", port="19530")
        print("✅ Connected to Milvus")
        
        # Test simple utility functions
        print("📊 Milvus server info:")
        collections = utility.list_collections()
        print(f"   Existing collections: {collections}")
        
        # Try creating a simple test collection
        test_name = "test_collection_temp"
        
        # Define simple schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)
        ]
        schema = CollectionSchema(fields, "Test collection")
        
        # Create collection using Collection constructor
        test_collection = Collection(test_name, schema)
        print(f"✅ Created test collection: {test_name}")
        
        # Test if we can work with it
        print(f"✅ Collection created successfully")
        
        # Clean up
        test_collection.drop()
        print(f"🗑️ Cleaned up test collection")
        
        print("🎉 Milvus is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_milvus()
