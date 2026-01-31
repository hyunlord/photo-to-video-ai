import { useEffect, useRef, useState, useCallback } from 'react';

interface WebSocketMessage {
  type: string;
  [key: string]: any;
}

export function useWebSocket(projectId: string, onMessage?: (message: WebSocketMessage) => void) {
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const onMessageRef = useRef(onMessage);
  const mountedRef = useRef(true);
  const connectionIdRef = useRef(0);

  // Update the ref when onMessage changes
  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);

  // Track mounted state
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    if (!projectId) return;

    // Increment connection ID to invalidate previous connections
    const currentConnectionId = ++connectionIdRef.current;
    const wsUrl = `${process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'}/ws/projects/${projectId}`;

    const isCurrentConnection = () => {
      return mountedRef.current && connectionIdRef.current === currentConnectionId;
    };

    const connect = () => {
      // Don't connect if this connection is stale
      if (!isCurrentConnection()) return;

      // Close existing connection if any
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        return; // Already connected
      }

      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        if (isCurrentConnection()) {
          setIsConnected(true);
        } else {
          // Stale connection, close it
          ws.close();
        }
      };

      ws.onmessage = (event) => {
        if (!isCurrentConnection()) return;

        try {
          const message = JSON.parse(event.data);

          // Handle ping/pong silently
          if (message.type === 'ping') {
            ws.send(JSON.stringify({ type: 'pong' }));
            return;
          }

          // Call the callback with the message
          if (onMessageRef.current) {
            onMessageRef.current(message);
          }
        } catch (error) {
          // Ignore parse errors silently
        }
      };

      ws.onerror = () => {
        // Errors are expected during cleanup, don't log
      };

      ws.onclose = () => {
        if (!isCurrentConnection()) return;

        setIsConnected(false);

        // Reconnect after 10 seconds (longer delay to reduce spam)
        reconnectTimeoutRef.current = setTimeout(() => {
          if (isCurrentConnection()) {
            connect();
          }
        }, 10000);
      };

      wsRef.current = ws;
    };

    connect();

    // Cleanup
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = undefined;
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [projectId]);

  return { isConnected, ws: wsRef.current };
}
