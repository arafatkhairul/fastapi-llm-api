#!/usr/bin/env python3
"""
Test script for Database-First Role Play System
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_roleplay_system():
    """Test the complete role play system"""
    print("🧪 Testing Database-First Role Play System")
    print("=" * 50)
    
    # Test client ID
    client_id = "test_client_123"
    
    # Test 1: Check database stats
    print("\n1. Checking database stats...")
    try:
        response = requests.get(f"{BASE_URL}/roleplay/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Database stats: {stats}")
        else:
            print(f"❌ Failed to get stats: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
    
    # Test 2: Configure role play
    print("\n2. Configuring role play...")
    roleplay_config = {
        "role_play_enabled": True,
        "role_play_template": "school",
        "organization_name": "Test International School",
        "organization_details": "A modern school with 500 students, focusing on technology and innovation",
        "role_title": "Technology Teacher"
    }
    
    try:
        # Send config via WebSocket simulation (using test endpoint)
        response = requests.post(f"{BASE_URL}/test-mcp-tools", params={
            "client_id": client_id,
            "question": "Configure role play"
        })
        if response.status_code == 200:
            print(f"✅ Role play configured: {response.json()}")
        else:
            print(f"❌ Failed to configure role play: {response.status_code}")
    except Exception as e:
        print(f"❌ Error configuring role play: {e}")
    
    # Test 3: Test role play with questions
    test_questions = [
        "What is your organization name?",
        "How many students do you have?",
        "What do you teach?",
        "What are your school's facilities?",
        "What is your role here?"
    ]
    
    print("\n3. Testing role play questions...")
    for i, question in enumerate(test_questions, 1):
        print(f"\n   Question {i}: {question}")
        try:
            response = requests.post(f"{BASE_URL}/test-mcp-tools", params={
                "client_id": client_id,
                "question": question
            })
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Response: {result.get('mcp_integration', 'No response')}")
            else:
                print(f"   ❌ Failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        time.sleep(1)  # Small delay between requests
    
    # Test 4: Check stored answers
    print("\n4. Checking stored answers...")
    try:
        response = requests.get(f"{BASE_URL}/roleplay/answers/{client_id}")
        if response.status_code == 200:
            answers = response.json()
            print(f"✅ Found {answers.get('total_answers', 0)} stored answers")
            for answer in answers.get('answers', [])[:3]:  # Show first 3
                print(f"   - Q: {answer['question'][:50]}...")
                print(f"     A: {answer['answer'][:50]}...")
        else:
            print(f"❌ Failed to get answers: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting answers: {e}")
    
    # Test 5: Search for specific answer
    print("\n5. Searching for specific answer...")
    try:
        response = requests.get(f"{BASE_URL}/roleplay/search/{client_id}", params={
            "question": "What is your organization name?"
        })
        if response.status_code == 200:
            search_result = response.json()
            print(f"✅ Search result: {search_result}")
        else:
            print(f"❌ Failed to search: {response.status_code}")
    except Exception as e:
        print(f"❌ Error searching: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Role play system test completed!")

if __name__ == "__main__":
    test_roleplay_system()
