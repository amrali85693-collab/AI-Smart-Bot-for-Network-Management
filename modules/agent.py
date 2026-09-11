"""Agentic chatbot with tool-calling (function-calling) capability."""

from __future__ import annotations

import json
import time
from typing import Any

import streamlit as st

from modules.alerts import generate_alerts
from modules.anomaly_detector import get_cached_detector
from modules.data_sources import get_data_source, get_cached_devices
try:
    from modules.knowledge_base import KnowledgeBase
except (ImportError, KeyError):
    KnowledgeBase = None  # type: ignore[misc,assignment]
from modules.llm_client import _get_secret, LLMConfigurationError
from modules.network_monitor import real_ping

GROQ_MODEL = "openai/gpt-oss-120b"
MAX_TOOL_ITERATIONS = 4

SYSTEM_PROMPT = (
    "You are a network operations assistant with access to real-time monitoring tools. "
    "You can search the knowledge base, check device status, run ping tests, view alerts, "
    "and check anomaly scores. Use tools to gather information before answering. "
    "Be precise and cite tool results. "
    "IMPORTANT: You are ONLY allowed to answer questions related to: "
    "network management, network monitoring, network diagnostics, network security, "
    "network troubleshooting, network configuration, network protocols, "
    "network devices (routers, switches, firewalls), network performance, "
    "network alerts, and network infrastructure. "
    "If a question is completely outside this scope (e.g., cooking, sports, "
    "politics, entertainment, general knowledge not related to networking), "
    "politely refuse and state that you can only help with network-related topics."
)

# Tool definitions in OpenAI format (Groq supports this)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the local networking knowledge base for relevant information on a topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (e.g., 'DNS troubleshooting', 'packet loss causes')",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_device_status",
            "description": "Check the current status and metrics of a specific network device by name",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_name": {
                        "type": "string",
                        "description": "The name of the device to check (e.g., 'Router-Core-01', 'Switch-Access-01')",
                    }
                },
                "required": ["device_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_ping",
            "description": "Perform a real ICMP ping test to a host to check reachability and latency",
            "parameters": {
                "type": "object",
                "properties": {
                    "host": {
                        "type": "string",
                        "description": "The hostname or IP address to ping (e.g., '8.8.8.8', 'google.com')",
                    }
                },
                "required": ["host"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_alerts",
            "description": "Get the list of current alerts from the monitoring system (high CPU, packet loss, devices down, etc.)",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_anomaly_score",
            "description": "Check if a device's metrics are anomalous according to the ML model",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_name": {
                        "type": "string",
                        "description": "The name of the device to check for anomalies",
                    }
                },
                "required": ["device_name"],
            },
        },
    },
]


def _execute_tool(tool_name: str, arguments: dict[str, Any], kb: KnowledgeBase) -> str:
    """Execute a tool and return the result as a string."""
    try:
        if tool_name == "search_knowledge_base":
            query = arguments.get("query", "")
            results = kb.search(query, top_k=3)
            if not results:
                return "No relevant knowledge base entries found for that query."
            output = "Knowledge base results:\n\n"
            for i, result in enumerate(results, 1):
                output += f"{i}. [{result['topic']}] (score: {result['score']:.3f})\n"
                output += f"   {result['answer']}\n\n"
            return output

        elif tool_name == "check_device_status":
            device_name = arguments.get("device_name", "")
            devices_df = get_cached_devices(
                st.session_state.get("data_source", "real"))
            
            # Case-insensitive partial match
            device_row = devices_df[
                devices_df["name"].str.contains(device_name, case=False, na=False)
            ]
            
            if device_row.empty:
                available = ", ".join(devices_df["name"].tolist()[:10])
                return f"Device '{device_name}' not found. Available devices: {available}"
            
            device = device_row.iloc[0]
            output = f"Device: {device['name']}\n"
            output += f"Type: {device.get('type', 'unknown')}\n"
            output += f"Status: {device.get('status', 'unknown')}\n"
            output += f"CPU Usage: {device.get('cpu_usage', 0):.1f}%\n"
            output += f"Latency: {device.get('latency_ms', 0):.1f} ms\n"
            output += f"Packet Loss: {device.get('packet_loss_pct', 0):.2f}%\n"
            if 'bandwidth_mbps' in device.index:
                output += f"Bandwidth: {device.get('bandwidth_mbps', 0):.1f} Mbps\n"
            if 'uptime_pct' in device.index:
                output += f"Uptime: {device.get('uptime_pct', 0):.2f}%\n"
            return output

        elif tool_name == "run_ping":
            host = arguments.get("host", "")
            result = real_ping(host, timeout=2.0)
            
            status = result.get("status", "unknown")
            if status == "up":
                return f"Ping to {host}: SUCCESS. Latency: {result.get('latency_ms')} ms"
            elif status == "down":
                return f"Ping to {host}: FAILED. {result.get('error', 'Host unreachable')}"
            else:
                return f"Ping to {host}: ERROR. {result.get('error', 'Unknown error')}"

        elif tool_name == "get_recent_alerts":
            devices_df = get_cached_devices(
                st.session_state.get("data_source", "real"))
            alerts = generate_alerts(devices_df)
            
            if not alerts or all(a.get("level") == "ok" for a in alerts):
                return "No active alerts. All systems within thresholds."
            
            actionable = [a for a in alerts if a.get("level") != "ok"]
            output = f"Current alerts ({len(actionable)} active):\n\n"
            for i, alert in enumerate(actionable[:10], 1):
                output += f"{i}. [{alert['level'].upper()}] {alert['message']}\n"
                output += f"   Device: {alert['device']}, Metric: {alert['metric']}, Value: {alert['value']}\n\n"
            return output

        elif tool_name == "get_anomaly_score":
            device_name = arguments.get("device_name", "")
            
            # Use a cached (per-mode) fitted detector; trains only once per
            # process instead of on every first tool call.
            if "anomaly_detector" not in st.session_state:
                try:
                    detector = get_cached_detector(
                        st.session_state.get("data_source", "real"),
                        hours=24,
                        contamination=0.05,
                        n_neighbors=30,
                    )
                    st.session_state.anomaly_detector = detector
                except Exception as e:
                    return f"Anomaly detector not initialized and auto-training failed: {e}"
            
            detector = st.session_state.anomaly_detector
            
            if not detector.is_fitted:
                try:
                    data_source = get_data_source()
                    history_df = data_source.get_traffic_history(hours=24)
                    detector.fit(history_df)
                except Exception as e:
                    return f"Anomaly detector not trained and auto-training failed: {e}"
            
            devices_df = get_cached_devices(
                st.session_state.get("data_source", "real"))
            
            device_row = devices_df[
                devices_df["name"].str.contains(device_name, case=False, na=False)
            ]
            
            if device_row.empty:
                return f"Device '{device_name}' not found."
            
            device = device_row.iloc[0]
            
            # Build metrics dict for the detector
            metrics = {
                col: float(device.get(col, 0)) 
                for col in detector.feature_columns 
                if col in device.index
            }
            
            try:
                result = detector.predict(metrics)
                is_anomaly = result.get("is_anomaly", False)
                score = result.get("anomaly_score", 0.0)
                
                output = f"Anomaly detection for {device['name']}:\n"
                output += f"Anomalous: {'YES' if is_anomaly else 'NO'}\n"
                output += f"Anomaly score: {score:.3f} (lower = more anomalous)\n"
                
                if is_anomaly:
                    output += "\nTop contributing features:\n"
                    contrib = result.get("feature_contributions", {})
                    sorted_contrib = sorted(contrib.items(), key=lambda x: x[1], reverse=True)
                    for feat, val in sorted_contrib[:3]:
                        output += f"  - {feat}: {val:.2f}\n"
                
                return output
            except Exception as e:
                return f"Error computing anomaly score: {e}"

        else:
            return f"Unknown tool: {tool_name}"

    except Exception as e:
        return f"Tool execution error: {str(e)}"


def agent_answer(
    query: str,
    kb: KnowledgeBase,
    history: list[dict] | None = None,
    temperature: float = 0.7,
) -> dict:
    """
    Agent-based answer with tool calling.
    
    Returns:
        dict with keys:
            - answer: str
            - tool_trace: list[dict] (tool calls made)
            - sources: list[str] (KB topics used)
    """
    start_time = time.time()
    
    api_key = _get_secret("GROQ_API_KEY")
    if not api_key:
        raise LLMConfigurationError(
            "GROQ_API_KEY is not set. Add it to .streamlit/secrets.toml"
        )
    
    from groq import Groq
    client = Groq(api_key=api_key)
    
    # Build message history
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    if history:
        for turn in history[-6:]:  # Last 6 turns for context
            messages.append({"role": turn["role"], "content": turn["content"]})
    
    messages.append({"role": "user", "content": query})
    
    tool_trace: list[dict] = []
    sources: list[str] = []
    
    # Tool-calling loop
    for iteration in range(MAX_TOOL_ITERATIONS):
        try:
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=1024,
            )
            
            response_message = completion.choices[0].message
            
            # Check if model wants to call a tool
            if response_message.tool_calls:
                # Add assistant's tool call request to messages
                messages.append(response_message)
                
                # Execute each tool call
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Execute tool
                    tool_result = _execute_tool(function_name, function_args, kb)
                    
                    # Track sources from KB searches
                    if function_name == "search_knowledge_base":
                        results = kb.search(function_args.get("query", ""), top_k=3)
                        sources.extend([r["topic"] for r in results])
                    
                    # Log for trace
                    tool_trace.append({
                        "iteration": iteration + 1,
                        "tool": function_name,
                        "arguments": function_args,
                        "result": tool_result,
                    })
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": tool_result,
                    })
                
                # Continue loop to let model see tool results
                continue
            
            # No tool calls - model has final answer
            answer = response_message.content or "No response generated."
            
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "answer": answer,
                "tool_trace": tool_trace,
                "sources": list(set(sources)),  # Deduplicate
                "latency_ms": latency_ms,
            }
        
        except Exception as e:
            # If tool calling fails, fall back to non-tool answer
            if iteration == 0:
                # Try one more time without tools
                try:
                    fallback = client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=1024,
                    )
                    answer = fallback.choices[0].message.content or "Error generating response."
                    return {
                        "answer": answer,
                        "tool_trace": [{"error": str(e)}],
                        "sources": [],
                        "latency_ms": (time.time() - start_time) * 1000,
                    }
                except Exception:
                    pass
            
            raise RuntimeError(f"Agent error: {e}") from e
    
    # Max iterations reached
    return {
        "answer": "Tool calling loop exceeded maximum iterations. Please try a simpler question.",
        "tool_trace": tool_trace,
        "sources": sources,
        "latency_ms": (time.time() - start_time) * 1000,
    }
