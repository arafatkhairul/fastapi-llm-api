#!/usr/bin/env python3
"""
MCP Server for Role Play Database Integration
Connects LLM with role play database for intelligent responses
"""

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import ServerSession, StdioServerParameters
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
    StructuredContent
)

# Database configuration
DB_PATH = Path("roleplay.db")

class RolePlayMCPServer:
    def __init__(self):
        self.server = Server("roleplay-db-mcp")
        
        # Register tools
        self.server.list_tools(self.list_tools)
        self.server.call_tool(self.call_tool)
        
        # Database connection
        self.db_path = DB_PATH
    
    def get_db_connection(self):
        """Get SQLite database connection"""
        return sqlite3.connect(self.db_path)
    
    async def list_tools(self, session: ServerSession, params: ListToolsRequest) -> ListToolsResult:
        """List available tools"""
        tools = [
            Tool(
                name="get_roleplay_info",
                description="Get role play information for a specific client",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "client_id": {
                            "type": "string",
                            "description": "Client ID to get role play info for"
                        }
                    },
                    "required": ["client_id"]
                }
            ),
            Tool(
                name="search_roleplay_by_org",
                description="Search role play configurations by organization name",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "organization_name": {
                            "type": "string",
                            "description": "Organization name to search for"
                        }
                    },
                    "required": ["organization_name"]
                }
            ),
            Tool(
                name="get_all_roleplay_configs",
                description="Get all role play configurations",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="create_roleplay_context",
                description="Create a role play context prompt for LLM",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "client_id": {
                            "type": "string",
                            "description": "Client ID to create context for"
                        },
                        "question": {
                            "type": "string",
                            "description": "User's question to answer in role play context"
                        }
                    },
                    "required": ["client_id", "question"]
                }
            )
        ]
        return ListToolsResult(tools=tools)
    
    async def call_tool(self, session: ServerSession, params: CallToolRequest) -> CallToolResult:
        """Execute tool calls"""
        try:
            if params.name == "get_roleplay_info":
                return await self._get_roleplay_info(params.arguments)
            elif params.name == "search_roleplay_by_org":
                return await self._search_roleplay_by_org(params.arguments)
            elif params.name == "get_all_roleplay_configs":
                return await self._get_all_roleplay_configs(params.arguments)
            elif params.name == "create_roleplay_context":
                return await self._create_roleplay_context(params.arguments)
            else:
                return CallToolResult(
                    isError=True,
                    content=[TextContent(text=f"Unknown tool: {params.name}")]
                )
        except Exception as e:
            return CallToolResult(
                isError=True,
                content=[TextContent(text=f"Tool execution failed: {str(e)}")]
            )
    
    async def _get_roleplay_info(self, args: Dict[str, Any]) -> CallToolResult:
        """Get role play info for a specific client"""
        client_id = args.get("client_id")
        if not client_id:
            return CallToolResult(
                isError=True,
                content=[TextContent(text="client_id is required")]
            )
        
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT role_play_enabled, role_play_template, organization_name, 
                           organization_details, role_title, updated_at
                    FROM role_play_configs 
                    WHERE client_id = ?
                """, (client_id,))
                
                row = cursor.fetchone()
                if row:
                    data = {
                        "client_id": client_id,
                        "role_play_enabled": bool(row[0]),
                        "role_play_template": row[1],
                        "organization_name": row[2],
                        "organization_details": row[3],
                        "role_title": row[4],
                        "updated_at": row[5]
                    }
                    return CallToolResult(
                        content=[
                            StructuredContent(data=data),
                            TextContent(text=f"Role play info for {client_id}: {json.dumps(data, indent=2)}")
                        ]
                    )
                else:
                    return CallToolResult(
                        content=[TextContent(text=f"No role play config found for client: {client_id}")]
                    )
        except Exception as e:
            return CallToolResult(
                isError=True,
                content=[TextContent(text=f"Database error: {str(e)}")]
            )
    
    async def _search_roleplay_by_org(self, args: Dict[str, Any]) -> CallToolResult:
        """Search role play by organization name"""
        org_name = args.get("organization_name")
        if not org_name:
            return CallToolResult(
                isError=True,
                content=[TextContent(text="organization_name is required")]
            )
        
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT client_id, role_play_enabled, role_play_template, 
                           organization_name, role_title, updated_at
                    FROM role_play_configs 
                    WHERE organization_name LIKE ? AND role_play_enabled = 1
                """, (f"%{org_name}%",))
                
                rows = cursor.fetchall()
                if rows:
                    results = []
                    for row in rows:
                        results.append({
                            "client_id": row[0],
                            "role_play_enabled": bool(row[1]),
                            "role_play_template": row[2],
                            "organization_name": row[3],
                            "role_title": row[4],
                            "updated_at": row[5]
                        })
                    
                    return CallToolResult(
                        content=[
                            StructuredContent(data={"results": results}),
                            TextContent(text=f"Found {len(results)} role play configs for '{org_name}': {json.dumps(results, indent=2)}")
                        ]
                    )
                else:
                    return CallToolResult(
                        content=[TextContent(text=f"No role play configs found for organization: {org_name}")]
                    )
        except Exception as e:
            return CallToolResult(
                isError=True,
                content=[TextContent(text=f"Database error: {str(e)}")]
            )
    
    async def _get_all_roleplay_configs(self, args: Dict[str, Any]) -> CallToolResult:
        """Get all role play configurations"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT client_id, role_play_enabled, role_play_template, 
                           organization_name, role_title, updated_at
                    FROM role_play_configs 
                    ORDER BY updated_at DESC
                """)
                
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "client_id": row[0],
                        "role_play_enabled": bool(row[1]),
                        "role_play_template": row[2],
                        "organization_name": row[3],
                        "role_title": row[4],
                        "updated_at": row[5]
                    })
                
                return CallToolResult(
                    content=[
                        StructuredContent(data={"total": len(results), "configs": results}),
                        TextContent(text=f"Total role play configs: {len(results)}")
                    ]
                )
        except Exception as e:
            return CallToolResult(
                isError=True,
                content=[TextContent(text=f"Database error: {str(e)}")]
            )
    
    async def _create_roleplay_context(self, args: Dict[str, Any]) -> CallToolResult:
        """Create role play context prompt for LLM"""
        client_id = args.get("client_id")
        question = args.get("question")
        
        if not client_id or not question:
            return CallToolResult(
                isError=True,
                content=[TextContent(text="Both client_id and question are required")]
            )
        
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT role_play_enabled, role_play_template, organization_name, 
                           organization_details, role_title
                    FROM role_play_configs 
                    WHERE client_id = ?
                """, (client_id,))
                
                row = cursor.fetchone()
                if row and row[0]:  # role_play_enabled
                    template = row[1]
                    org_name = row[2]
                    org_details = row[3]
                    role_title = row[4]
                    
                    # Create role play context
                    context = f"""You are a {role_title} at {org_name}. 

Organization Details: {org_details}

User Question: {question}

Please answer this question in character as a {role_title} at {org_name}. Stay in character and provide relevant information based on your role and organization."""
                    
                    return CallToolResult(
                        content=[
                            StructuredContent(data={
                                "client_id": client_id,
                                "role_play_enabled": True,
                                "template": template,
                                "organization_name": org_name,
                                "role_title": role_title,
                                "context": context
                            }),
                            TextContent(text=context)
                        ]
                    )
                else:
                    # No role play config or disabled
                    return CallToolResult(
                        content=[
                            StructuredContent(data={
                                "client_id": client_id,
                                "role_play_enabled": False,
                                "message": "No role play configuration found or role play is disabled"
                            }),
                            TextContent(text=f"No role play configuration found for client {client_id}. Please answer the question normally: {question}")
                        ]
                    )
        except Exception as e:
            return CallToolResult(
                isError=True,
                content=[TextContent(text=f"Database error: {str(e)}")]
            )

async def main():
    """Main entry point"""
    server = RolePlayMCPServer()
    
    # Run with stdio
    async with stdio_server(StdioServerParameters()) as (read, write):
        await server.server.run(
            read,
            write,
            initial_notifications=[
                {
                    "method": "notifications/initialized",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {"listChanged": True},
                            "resources": {"subscribelistChanged": True}
                        },
                        "serverInfo": {
                            "name": "RolePlay Database MCP Server",
                            "version": "1.0.0"
                        }
                    }
                }
            ]
        )

if __name__ == "__main__":
    asyncio.run(main())
