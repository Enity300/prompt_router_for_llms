import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from collections import Counter
from pathlib import Path

def check_evaluation_json(json_path="evaluation_dataset.json"):
    """Check dimensions and structure of evaluation dataset JSON"""
    
    if not os.path.exists(json_path):
        print(f"❌ File not found: {json_path}")
        return
    
    print(f"📊 Analyzing: {json_path}\n")
    print("="*70)
    
    # Load JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Basic dimensions
    print(f"\n📏 BASIC DIMENSIONS:")
    print(f"   Total samples: {len(data)}")
    print(f"   File size: {os.path.getsize(json_path) / (1024*1024):.2f} MB")
    
    # Check structure of first item
    if data:
        first_item = data[0]
        print(f"\n📝 STRUCTURE (first sample):")
        print(f"   Keys: {list(first_item.keys())}")
        
        for key, value in first_item.items():
            value_type = type(value).__name__
            if isinstance(value, str):
                print(f"   - {key}: {value_type} (length: {len(value)})")
            elif isinstance(value, (list, dict)):
                print(f"   - {key}: {value_type} (size: {len(value)})")
            else:
                print(f"   - {key}: {value_type} = {value}")
    
    # Category distribution
    if data and 'category' in data[0]:
        categories = [item['category'] for item in data]
        category_counts = Counter(categories)
        
        print(f"\n📊 CATEGORY DISTRIBUTION:")
        total = len(data)
        for category, count in sorted(category_counts.items()):
            percentage = (count / total) * 100
            print(f"   - {category}: {count} samples ({percentage:.1f}%)")
    
    # Query length statistics
    if data and 'query' in data[0]:
        query_lengths = [len(item['query']) for item in data]
        word_counts = [len(item['query'].split()) for item in data]
        
        print(f"\n📐 QUERY STATISTICS:")
        print(f"   Character length:")
        print(f"      - Min: {min(query_lengths)}")
        print(f"      - Max: {max(query_lengths)}")
        print(f"      - Average: {sum(query_lengths)/len(query_lengths):.1f}")
        print(f"   Word count:")
        print(f"      - Min: {min(word_counts)}")
        print(f"      - Max: {max(word_counts)}")
        print(f"      - Average: {sum(word_counts)/len(word_counts):.1f}")
    
    # Check for embeddings (if present)
    if data and 'embedding' in data[0]:
        embedding_dim = len(data[0]['embedding'])
        print(f"\n🔢 EMBEDDING DIMENSIONS:")
        print(f"   Embedding dimension: {embedding_dim}")
        print(f"   Total embedding vectors: {len(data)}")
    
    # Sample queries from each category
    if data and 'category' in data[0] and 'query' in data[0]:
        print(f"\n📋 SAMPLE QUERIES (first from each category):")
        seen_categories = set()
        for item in data:
            if item['category'] not in seen_categories:
                seen_categories.add(item['category'])
                query_preview = item['query'][:80] + "..." if len(item['query']) > 80 else item['query']
                print(f"   [{item['category']}]: {query_preview}")
                if len(seen_categories) == len(category_counts):
                    break
    
    print("\n" + "="*70)
    print("✅ Analysis complete!\n")


if __name__ == "__main__":
    # Check if a custom path is provided
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        json_path = "evaluation_dataset.json"
    
    check_evaluation_json(json_path)

