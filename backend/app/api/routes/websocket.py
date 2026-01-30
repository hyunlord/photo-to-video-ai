from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
import redis
import asyncio
import logging

from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Store active WebSocket connections
active_connections: Dict[str, Set[WebSocket]] = {}

# Redis client for pub/sub
redis_client = redis.from_url(settings.REDIS_URL)
pubsub = redis_client.pubsub()

@router.websocket("/projects/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    """
    WebSocket endpoint for real-time project updates

    Clients connect to receive updates about job progress
    """
    await websocket.accept()
    logger.info(f"WebSocket connected for project {project_id}")

    # Add to active connections
    if project_id not in active_connections:
        active_connections[project_id] = set()
    active_connections[project_id].add(websocket)

    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "project_id": project_id,
            "message": "WebSocket connected"
        })

        # Keep connection alive and listen for messages
        while True:
            # Receive message from client (for heartbeat)
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )

                # Handle client messages
                try:
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                except json.JSONDecodeError:
                    pass

            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                try:
                    await websocket.send_json({"type": "ping"})
                except:
                    break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for project {project_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Remove from active connections
        if project_id in active_connections:
            active_connections[project_id].discard(websocket)
            if not active_connections[project_id]:
                del active_connections[project_id]

async def broadcast_to_project(project_id: str, message: dict):
    """
    Broadcast message to all connections for a project
    """
    if project_id in active_connections:
        disconnected = set()

        for connection in active_connections[project_id]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                disconnected.add(connection)

        # Remove disconnected connections
        active_connections[project_id] -= disconnected

async def broadcast_job_update(job_id: str, project_id: str, status: str, **kwargs):
    """
    Broadcast job update to all project connections

    Args:
        job_id: Job ID
        project_id: Project ID
        status: Job status (started, progress, completed, failed)
        **kwargs: Additional data (progress, error_message, result_path, etc.)
    """
    message = {
        "type": f"job.{status}",
        "job_id": job_id,
        **kwargs
    }

    await broadcast_to_project(project_id, message)
