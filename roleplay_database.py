#!/usr/bin/env python3
"""
Database-First Role Play System
Stores and retrieves role play answers from database first, then falls back to LLM
"""

import sqlite3
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

log = logging.getLogger("roleplay_db")

class RolePlayDatabase:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with role play tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create role_play_configs table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS role_play_configs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        client_id TEXT UNIQUE NOT NULL,
                        role_play_enabled BOOLEAN DEFAULT FALSE,
                        role_play_template TEXT DEFAULT 'school',
                        organization_name TEXT DEFAULT '',
                        organization_details TEXT DEFAULT '',
                        role_title TEXT DEFAULT '',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create role_play_answers table for storing Q&A
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS role_play_answers (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        client_id TEXT NOT NULL,
                        question_hash TEXT NOT NULL,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        organization_name TEXT NOT NULL,
                        role_title TEXT NOT NULL,
                        template TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(client_id, question_hash)
                    )
                """)
                
                # Create indexes for faster lookups
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_client_id ON role_play_configs(client_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_answers_client_id ON role_play_answers(client_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_answers_question_hash ON role_play_answers(question_hash)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_answers_org_role ON role_play_answers(organization_name, role_title)")
                
                conn.commit()
                log.info("Role play database initialized successfully")
                
        except Exception as e:
            log.error(f"Database initialization error: {e}")
            raise
    
    def _hash_question(self, question: str) -> str:
        """Create a hash of the question for efficient lookup"""
        return hashlib.md5(question.lower().strip().encode()).hexdigest()
    
    def save_role_play_config(self, client_id: str, config: Dict[str, Any]) -> bool:
        """Save or update role play configuration"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if config exists
                cursor.execute("SELECT id FROM role_play_configs WHERE client_id = ?", (client_id,))
                exists = cursor.fetchone()
                
                if exists:
                    # Update existing config
                    cursor.execute("""
                        UPDATE role_play_configs 
                        SET role_play_enabled = ?, role_play_template = ?, 
                            organization_name = ?, organization_details = ?, role_title = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE client_id = ?
                    """, (
                        config.get('role_play_enabled', False),
                        config.get('role_play_template', 'school'),
                        config.get('organization_name', ''),
                        config.get('organization_details', ''),
                        config.get('role_title', ''),
                        client_id
                    ))
                else:
                    # Insert new config
                    cursor.execute("""
                        INSERT INTO role_play_configs 
                        (client_id, role_play_enabled, role_play_template, 
                         organization_name, organization_details, role_title)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        client_id,
                        config.get('role_play_enabled', False),
                        config.get('role_play_template', 'school'),
                        config.get('organization_name', ''),
                        config.get('organization_details', ''),
                        config.get('role_title', '')
                    ))
                
                conn.commit()
                log.info(f"Saved role play config for client: {client_id}")
                return True
                
        except Exception as e:
            log.error(f"Save role play config error for {client_id}: {e}")
            return False
    
    def get_role_play_config(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get role play configuration for a client"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT role_play_enabled, role_play_template, organization_name, 
                           organization_details, role_title, updated_at
                    FROM role_play_configs 
                    WHERE client_id = ?
                """, (client_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'role_play_enabled': bool(row[0]),
                        'role_play_template': row[1],
                        'organization_name': row[2],
                        'organization_details': row[3],
                        'role_title': row[4],
                        'updated_at': row[5]
                    }
                return None
                
        except Exception as e:
            log.error(f"Get role play config error for {client_id}: {e}")
            return None
    
    def save_role_play_answer(self, client_id: str, question: str, answer: str, 
                            organization_name: str, role_title: str, template: str) -> bool:
        """Save a role play Q&A pair"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                question_hash = self._hash_question(question)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO role_play_answers 
                    (client_id, question_hash, question, answer, organization_name, 
                     role_title, template, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (client_id, question_hash, question, answer, 
                     organization_name, role_title, template))
                
                conn.commit()
                log.info(f"Saved role play answer for client: {client_id}")
                return True
                
        except Exception as e:
            log.error(f"Save role play answer error for {client_id}: {e}")
            return False
    
    def get_role_play_answer(self, client_id: str, question: str) -> Optional[Dict[str, Any]]:
        """Get a stored role play answer for a question"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                question_hash = self._hash_question(question)
                
                cursor.execute("""
                    SELECT question, answer, organization_name, role_title, template, 
                           created_at, updated_at
                    FROM role_play_answers 
                    WHERE client_id = ? AND question_hash = ?
                """, (client_id, question_hash))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'question': row[0],
                        'answer': row[1],
                        'organization_name': row[2],
                        'role_title': row[3],
                        'template': row[4],
                        'created_at': row[5],
                        'updated_at': row[6],
                        'source': 'database'
                    }
                return None
                
        except Exception as e:
            log.error(f"Get role play answer error for {client_id}: {e}")
            return None
    
    def search_similar_answers(self, client_id: str, question: str, 
                             organization_name: str, role_title: str) -> List[Dict[str, Any]]:
        """Search for similar answers from the same organization/role"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Search for answers from same org/role
                cursor.execute("""
                    SELECT question, answer, created_at
                    FROM role_play_answers 
                    WHERE organization_name = ? AND role_title = ?
                    ORDER BY created_at DESC
                    LIMIT 5
                """, (organization_name, role_title))
                
                rows = cursor.fetchall()
                return [
                    {
                        'question': row[0],
                        'answer': row[1],
                        'created_at': row[2]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            log.error(f"Search similar answers error: {e}")
            return []
    
    def get_all_answers_for_client(self, client_id: str) -> List[Dict[str, Any]]:
        """Get all stored answers for a client"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT question, answer, organization_name, role_title, template, 
                           created_at, updated_at
                    FROM role_play_answers 
                    WHERE client_id = ?
                    ORDER BY updated_at DESC
                """, (client_id,))
                
                rows = cursor.fetchall()
                return [
                    {
                        'question': row[0],
                        'answer': row[1],
                        'organization_name': row[2],
                        'role_title': row[3],
                        'template': row[4],
                        'created_at': row[5],
                        'updated_at': row[6]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            log.error(f"Get all answers error for {client_id}: {e}")
            return []
    
    def clear_role_play_config(self, client_id: str) -> bool:
        """Clear role play configuration and all answers for a client"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Clear config
                cursor.execute("""
                    UPDATE role_play_configs 
                    SET role_play_enabled = FALSE, organization_name = '', 
                        organization_details = '', role_title = '',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE client_id = ?
                """, (client_id,))
                
                # Clear all answers
                cursor.execute("DELETE FROM role_play_answers WHERE client_id = ?", (client_id,))
                
                conn.commit()
                log.info(f"Cleared role play config and answers for client: {client_id}")
                return True
                
        except Exception as e:
            log.error(f"Clear role play config error for {client_id}: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count configs
                cursor.execute("SELECT COUNT(*) FROM role_play_configs")
                total_configs = cursor.fetchone()[0]
                
                # Count enabled configs
                cursor.execute("SELECT COUNT(*) FROM role_play_configs WHERE role_play_enabled = 1")
                enabled_configs = cursor.fetchone()[0]
                
                # Count answers
                cursor.execute("SELECT COUNT(*) FROM role_play_answers")
                total_answers = cursor.fetchone()[0]
                
                # Count unique organizations
                cursor.execute("SELECT COUNT(DISTINCT organization_name) FROM role_play_answers")
                unique_orgs = cursor.fetchone()[0]
                
                return {
                    'total_configs': total_configs,
                    'enabled_configs': enabled_configs,
                    'total_answers': total_answers,
                    'unique_organizations': unique_orgs
                }
                
        except Exception as e:
            log.error(f"Get database stats error: {e}")
            return {}
