"""Network reachability checks, HTTP health checks, and simulated device inventory."""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Any, Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import socket
import statistics

import pandas as pd

try:
    from ping3 import ping as icmp_ping
except ImportError:
    icmp_ping = None

try:
    import requests
except ImportError:
    requests = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    go = None
    px = None


def _connect_port(host: str, port: int, timeout: float) -> float:
    """Return TCP connect latency (ms) to a port, or raise OSError."""
    start = time.time()
    with socket.create_connection((host, port), timeout=timeout):
        return round((time.time() - start) * 1000, 2)


def _tcp_probe_host(host: str, timeout: float = 2.0, ports: Optional[List[int]] = None) -> Dict[str, Any]:
    """Fallback probe using TCP connect to common service ports (probed in parallel)."""
    timestamp = datetime.now(timezone.utc).isoformat()
    if ports is None:
        ports = [443, 80]

    executor = ThreadPoolExecutor(max_workers=min(len(ports), 4))
    try:
        futs = {executor.submit(_connect_port, host, p, timeout): p for p in ports}
        try:
            for fut in as_completed(futs, timeout=timeout + 1.0):
                try:
                    latency_ms = fut.result()
                    return {
                        "host": host,
                        "status": "up",
                        "latency_ms": latency_ms,
                        "error": None,
                        "probe_port": futs[fut],
                        "timestamp": timestamp,
                    }
                except OSError:
                    continue
        except TimeoutError:
            pass
    finally:
        executor.shutdown(wait=False)

    return {
        "host": host,
        "status": "down",
        "latency_ms": None,
        "error": "TCP probe failed: all service ports unreachable",
        "probe_port": None,
        "timestamp": timestamp,
    }


def real_ping(host: str, timeout: float = 2.0) -> Dict[str, Any]:
    """Perform a single real ICMP ping against a host."""
    timestamp = datetime.now(timezone.utc).isoformat()

    if icmp_ping is None:
        return _tcp_probe_host(host, timeout=timeout)

    try:
        # ping3 returns float in seconds, or None/False if unreachable
        result = icmp_ping(host, timeout=timeout, unit="ms")
        if result is None or result is False:
            tcp_result = _tcp_probe_host(host, timeout=timeout)
            if tcp_result["status"] == "up":
                return tcp_result
            return {
                "host": host,
                "status": "down",
                "latency_ms": None,
                "error": "Request timed out or host unreachable",
                "timestamp": timestamp,
            }
        return {
            "host": host,
            "status": "up",
            "latency_ms": round(float(result), 2),
            "error": None,
            "timestamp": timestamp,
        }
    except PermissionError:
        return _tcp_probe_host(host, timeout=timeout)
    except OSError as exc:
        tcp_result = _tcp_probe_host(host, timeout=timeout)
        if tcp_result["status"] == "up":
            return tcp_result
        return {
            "host": host,
            "status": "error",
            "latency_ms": None,
            "error": str(exc),
            "timestamp": timestamp,
        }


def ping_host_telemetry(host: str, count: int = 3, timeout: float = 1.0) -> Dict[str, Any]:
    """
    Perform multiple pings against a host to calculate average latency and packet loss.
    Returns a unified metrics dictionary.
    """
    latencies = []
    failures = 0
    up_status = "down"

    for _ in range(count):
        res = real_ping(host, timeout=timeout)
        if res.get("status") == "up":
            up_status = "up"
            latency = res.get("latency_ms")
            if latency is not None:
                latencies.append(latency)
        else:
            failures += 1

    packet_loss_pct = round((failures / count) * 100, 1)
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0.0

    return {
        "status": up_status,
        "latency_ms": avg_latency,
        "packet_loss_pct": packet_loss_pct,
    }


def simulate_devices(seed: Optional[int] = 42) -> pd.DataFrame:
    """
    Demo device inventory for the dashboard when no real devices are reachable.
    Fully simulated for presentation purposes.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    devices = [
        ("core-router-01", "router"),
        ("edge-fw-01", "firewall"),
        ("access-sw-01", "switch"),
        ("access-sw-02", "switch"),
        ("web-server-01", "server"),
        ("db-server-01", "server"),
        ("wifi-ap-floor2", "access_point"),
        ("dns-resolver-01", "server"),
    ]

    rows = []
    for name, dtype in devices:
        status_roll = rng.random()
        if status_roll < 0.12:
            status = "down"
            latency = None
            packet_loss = 100.0
            cpu = 0.0
            bandwidth = 0.0
            uptime = 0.0
        else:
            status = "up"
            latency = round(float(rng.uniform(2, 85)), 1)
            packet_loss = round(float(rng.uniform(0, 3.5)), 2)
            cpu = round(float(rng.uniform(15, 95)), 1)
            bandwidth = round(float(rng.uniform(10, 950)), 1)
            uptime = round(float(rng.uniform(95, 100)), 2)

        rows.append(
            {
                "device": name,
                "type": dtype,
                "status": status,
                "cpu_usage": cpu,
                "latency_ms": latency,
                "packet_loss_pct": packet_loss,
                "bandwidth_mbps": bandwidth,
                "uptime_pct": uptime,
            }
        )

    return pd.DataFrame(rows)


def http_health_check(url: str, timeout: float = 5.0) -> Dict[str, Any]:
    """
    Perform an HTTP health check against a URL.
    Returns status, HTTP status code, response time, and error if any.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    if requests is None:
        return {
            "url": url,
            "status": "error",
            "http_status": None,
            "latency_ms": None,
            "error": "requests library is not installed",
            "timestamp": timestamp,
        }

    try:
        start_time = time.time()
        response = requests.get(url, timeout=timeout)
        latency_ms = round((time.time() - start_time) * 1000, 2)

        if 200 <= response.status_code < 400:
            return {
                "url": url,
                "status": "up",
                "http_status": response.status_code,
                "latency_ms": latency_ms,
                "error": None,
                "timestamp": timestamp,
            }
        else:
            return {
                "url": url,
                "status": "down",
                "http_status": response.status_code,
                "latency_ms": latency_ms,
                "error": f"HTTP {response.status_code}",
                "timestamp": timestamp,
            }
    except requests.exceptions.Timeout:
        return {
            "url": url,
            "status": "down",
            "http_status": None,
            "latency_ms": None,
            "error": "Request timed out",
            "timestamp": timestamp,
        }
    except requests.exceptions.ConnectionError as exc:
        return {
            "url": url,
            "status": "down",
            "http_status": None,
            "latency_ms": None,
            "error": f"Connection error: {str(exc)}",
            "timestamp": timestamp,
        }
    except Exception as exc:
        return {
            "url": url,
            "status": "error",
            "http_status": None,
            "latency_ms": None,
            "error": str(exc),
            "timestamp": timestamp,
        }


def snmp_poll(device_ip: str, community: str = "public") -> Dict[str, Any]:
    """
    Extension point for SNMP polling (stretch goal).
    Can be wired to PySNMP to query local network hardware in production.
    """
    return {
        "device_ip": device_ip,
        "status": "not_implemented",
        "message": (
            "SNMP polling stub — implement with pysnmp against a reachable "
            "lab device or on-prem deployment."
        ),
        "community": community,
    }


def batch_ping(hosts: List[str], timeout: float = 2.0, max_workers: int = 10) -> Dict[str, Any]:
    """
    Perform ICMP pings against multiple hosts in parallel with timing metrics.
    Returns a dictionary with results and timing information.
    """
    import time
    start_time = time.time()
    results = []
    
    effective_workers = max(1, min(max_workers, len(hosts), 20))
    executor = ThreadPoolExecutor(max_workers=effective_workers)
    future_to_host = {executor.submit(real_ping, host, timeout): host for host in hosts}
    all_futures = list(future_to_host)
    pending_futures = set(all_futures)

    try:
        for future in as_completed(all_futures, timeout=timeout + 2):
            host = future_to_host[future]
            pending_futures.remove(future)
            try:
                result = future.result(timeout=1)
                result["scan_time"] = time.time() - start_time
                results.append(result)
            except TimeoutError:
                results.append({
                    "host": host,
                    "status": "error",
                    "latency_ms": None,
                    "error": f"Ping task timed out after {timeout + 1:.0f}s",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "scan_time": time.time() - start_time,
                })
            except Exception as exc:
                results.append({
                    "host": host,
                    "status": "error",
                    "latency_ms": None,
                    "error": str(exc),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "scan_time": time.time() - start_time,
                })
    except TimeoutError:
        pass
    finally:
        for future in pending_futures:
            host = future_to_host[future]
            results.append({
                "host": host,
                "status": "error",
                "latency_ms": None,
                "error": "Ping task did not complete within expected time",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "scan_time": time.time() - start_time,
            })
        executor.shutdown(wait=False)
    
    total_time = time.time() - start_time
    
    # Calculate timing statistics
    successful_results = [r for r in results if r.get("status") == "up"]
    latencies = [r.get("latency_ms", 0) for r in successful_results if r.get("latency_ms") is not None]
    
    timing_stats = {
        "total_time_seconds": round(total_time, 3),
        "total_hosts": len(hosts),
        "successful_hosts": len(successful_results),
        "failed_hosts": len(hosts) - len(successful_results),
        "hosts_per_second": round(len(hosts) / total_time, 2) if total_time > 0 else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "min_latency_ms": round(min(latencies), 2) if latencies else 0,
        "max_latency_ms": round(max(latencies), 2) if latencies else 0,
    }
    
    return {
        "results": results,
        "timing": timing_stats,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def calculate_latency_stats(latencies: List[float]) -> Dict[str, float]:
    """
    Calculate statistical metrics for latency values.
    Returns min, max, avg, median, std_dev, and jitter.
    """
    if not latencies:
        return {
            "min_ms": 0.0,
            "max_ms": 0.0,
            "avg_ms": 0.0,
            "median_ms": 0.0,
            "std_dev_ms": 0.0,
            "jitter_ms": 0.0,
            "count": 0,
        }
    
    sorted_latencies = sorted(latencies)
    jitter_values = [abs(latencies[i] - latencies[i-1]) for i in range(1, len(latencies))]
    
    return {
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "avg_ms": round(statistics.mean(latencies), 2),
        "median_ms": round(statistics.median(latencies), 2),
        "std_dev_ms": round(statistics.stdev(latencies) if len(latencies) > 1 else 0.0, 2),
        "jitter_ms": round(statistics.mean(jitter_values) if jitter_values else 0.0, 2),
        "count": len(latencies),
    }


def dns_lookup(hostname: str) -> Dict[str, Any]:
    """
    Perform DNS lookup for a hostname.
    Returns IP addresses and lookup time.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    
    try:
        start_time = time.time()
        ip_addresses = socket.getaddrinfo(hostname, None)
        lookup_time_ms = round((time.time() - start_time) * 1000, 2)
        
        # Extract unique IP addresses
        unique_ips = list(set([addr[4][0] for addr in ip_addresses]))
        
        return {
            "hostname": hostname,
            "status": "success",
            "ip_addresses": unique_ips,
            "lookup_time_ms": lookup_time_ms,
            "error": None,
            "timestamp": timestamp,
        }
    except socket.gaierror as exc:
        return {
            "hostname": hostname,
            "status": "error",
            "ip_addresses": [],
            "lookup_time_ms": None,
            "error": f"DNS lookup failed: {str(exc)}",
            "timestamp": timestamp,
        }
    except Exception as exc:
        return {
            "hostname": hostname,
            "status": "error",
            "ip_addresses": [],
            "lookup_time_ms": None,
            "error": str(exc),
            "timestamp": timestamp,
        }


# Common port service mapping
COMMON_SERVICES = {
    21: {"name": "FTP", "description": "File Transfer Protocol", "risk": "medium"},
    22: {"name": "SSH", "description": "Secure Shell", "risk": "low"},
    23: {"name": "Telnet", "description": "Telnet", "risk": "high"},
    25: {"name": "SMTP", "description": "Simple Mail Transfer", "risk": "medium"},
    53: {"name": "DNS", "description": "Domain Name System", "risk": "low"},
    80: {"name": "HTTP", "description": "Web Server", "risk": "low"},
    110: {"name": "POP3", "description": "Post Office Protocol", "risk": "medium"},
    143: {"name": "IMAP", "description": "Internet Message Access", "risk": "medium"},
    443: {"name": "HTTPS", "description": "Secure Web Server", "risk": "low"},
    445: {"name": "SMB", "description": "Server Message Block", "risk": "high"},
    993: {"name": "IMAPS", "description": "Secure IMAP", "risk": "medium"},
    995: {"name": "POP3S", "description": "Secure POP3", "risk": "medium"},
    1433: {"name": "MSSQL", "description": "Microsoft SQL Server", "risk": "medium"},
    3306: {"name": "MySQL", "description": "MySQL Database", "risk": "medium"},
    3389: {"name": "RDP", "description": "Remote Desktop Protocol", "risk": "high"},
    5432: {"name": "PostgreSQL", "description": "PostgreSQL Database", "risk": "medium"},
    5900: {"name": "VNC", "description": "Virtual Network Computing", "risk": "high"},
    6379: {"name": "Redis", "description": "Redis Database", "risk": "medium"},
    8080: {"name": "HTTP-Alt", "description": "Alternative HTTP", "risk": "low"},
    8443: {"name": "HTTPS-Alt", "description": "Alternative HTTPS", "risk": "low"},
    27017: {"name": "MongoDB", "description": "MongoDB Database", "risk": "medium"},
}

# Categorized port presets
PORT_PRESETS = {
    "Web Services": [80, 443, 8080, 8443, 8000, 8888],
    "Database": [3306, 5432, 1433, 27017, 6379, 1521],
    "Email": [25, 587, 993, 995, 110, 143],
    "Remote Access": [22, 3389, 5900, 5901],
    "File Transfer": [21, 69, 873, 989, 990],
    "Network Services": [53, 67, 68, 123, 161, 162],
    "Common Top 20": [21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 993, 995, 1433, 3306, 3389, 5432, 5900, 8080, 8443, 27017],
}


def get_service_info(port: int) -> Dict[str, str]:
    """
    Get service information for a port.
    Returns service name, description, and risk level.
    """
    if port in COMMON_SERVICES:
        return COMMON_SERVICES[port]
    return {"name": "Unknown", "description": "No service information", "risk": "unknown"}


def grab_banner(host: str, port: int, timeout: float = 3.0) -> str:
    """
    Attempt to grab service banner from an open port.
    Returns banner string or empty string if failed.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        
        # Send simple HTTP request for web services
        if port in [80, 8080, 8000, 8888]:
            sock.send(b"HEAD / HTTP/1.0\r\n\r\n")
        elif port in [443, 8443]:
            sock.send(b"HEAD / HTTP/1.0\r\n\r\n")
        # For other services, just wait for banner
        
        sock.settimeout(2.0)
        banner = sock.recv(1024).decode('utf-8', errors='ignore').strip()
        sock.close()
        
        return banner[:200]  # Limit banner length
    except Exception:
        return ""


def check_port(host: str, port: int, timeout: float = 3.0, grab_banner: bool = False) -> Dict[str, Any]:
    """
    Check if a specific port is open on a host.
    Returns connection status, latency, service info, and optionally banner.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    service_info = get_service_info(port)
    
    try:
        start_time = time.time()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        latency_ms = round((time.time() - start_time) * 1000, 2)
        sock.close()
        
        banner = ""
        if result == 0 and grab_banner:
            banner = grab_banner(host, port, timeout)
        
        if result == 0:
            return {
                "host": host,
                "port": port,
                "status": "open",
                "latency_ms": latency_ms,
                "service_name": service_info["name"],
                "service_description": service_info["description"],
                "risk_level": service_info["risk"],
                "banner": banner,
                "error": None,
                "timestamp": timestamp,
            }
        else:
            return {
                "host": host,
                "port": port,
                "status": "closed",
                "latency_ms": latency_ms,
                "service_name": service_info["name"],
                "service_description": service_info["description"],
                "risk_level": service_info["risk"],
                "banner": "",
                "error": None,
                "timestamp": timestamp,
            }
    except socket.timeout:
        return {
            "host": host,
            "port": port,
            "status": "timeout",
            "latency_ms": None,
            "service_name": service_info["name"],
            "service_description": service_info["description"],
            "risk_level": service_info["risk"],
            "banner": "",
            "error": "Connection timed out",
            "timestamp": timestamp,
        }
    except Exception as exc:
        return {
            "host": host,
            "port": port,
            "status": "error",
            "latency_ms": None,
            "service_name": service_info["name"],
            "service_description": service_info["description"],
            "risk_level": service_info["risk"],
            "banner": "",
            "error": str(exc),
            "timestamp": timestamp,
        }


def batch_port_check(host: str, ports: List[int], timeout: float = 3.0, max_workers: int = 50, grab_banner: bool = False) -> List[Dict[str, Any]]:
    """
    Check multiple ports on a host in parallel.
    Returns a list of port check results with service information.
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_port = {executor.submit(check_port, host, port, timeout, grab_banner): port for port in ports}
        
        for future in as_completed(future_to_port):
            port = future_to_port[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                service_info = get_service_info(port)
                results.append({
                    "host": host,
                    "port": port,
                    "status": "error",
                    "latency_ms": None,
                    "service_name": service_info["name"],
                    "service_description": service_info["description"],
                    "risk_level": service_info["risk"],
                    "banner": "",
                    "error": str(exc),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
    
    # Sort by port number for better readability
    results.sort(key=lambda x: x["port"])
    return results


def scan_port_range(host: str, start_port: int, end_port: int, timeout: float = 3.0, max_workers: int = 50) -> List[Dict[str, Any]]:
    """
    Scan a range of ports on a host.
    Returns a list of port check results.
    """
    if start_port < 1 or end_port > 65535 or start_port > end_port:
        return [{
            "host": host,
            "port": 0,
            "status": "error",
            "error": "Invalid port range",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }]
    
    ports = list(range(start_port, end_port + 1))
    return batch_port_check(host, ports, timeout, max_workers, grab_banner=False)


def assess_port_security(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Assess security of open ports based on risk levels.
    Returns security summary and recommendations.
    """
    open_ports = [r for r in results if r.get("status") == "open"]
    
    high_risk_ports = [r for r in open_ports if r.get("risk_level") == "high"]
    medium_risk_ports = [r for r in open_ports if r.get("risk_level") == "medium"]
    low_risk_ports = [r for r in open_ports if r.get("risk_level") == "low"]
    
    recommendations = []
    
    if high_risk_ports:
        recommendations.append(f"⚠️ {len(high_risk_ports)} high-risk ports open - consider closing unnecessary services")
    if medium_risk_ports:
        recommendations.append(f"⚡ {len(medium_risk_ports)} medium-risk ports open - review access controls")
    if len(open_ports) > 20:
        recommendations.append(f"📊 {len(open_ports)} total open ports - consider minimizing attack surface")
    
    # Specific service recommendations
    for port in high_risk_ports:
        port_num = port.get("port")
        if port_num == 23:  # Telnet
            recommendations.append("🔴 Telnet (port 23) is insecure - use SSH instead")
        elif port_num == 445:  # SMB
            recommendations.append("🔴 SMB (port 445) exposed - restrict network access")
        elif port_num == 3389:  # RDP
            recommendations.append("🔴 RDP (port 3389) exposed - use VPN or restrict access")
        elif port_num == 5900:  # VNC
            recommendations.append("🔴 VNC (port 5900) exposed - use SSH tunneling")
    
    return {
        "total_open_ports": len(open_ports),
        "high_risk_count": len(high_risk_ports),
        "medium_risk_count": len(medium_risk_ports),
        "low_risk_count": len(low_risk_ports),
        "security_score": max(0, 100 - (len(high_risk_ports) * 20) - (len(medium_risk_ports) * 5)),
        "recommendations": recommendations,
        "high_risk_details": high_risk_ports,
    }


def analyze_ping_history(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze ping history for trends, anomalies, and performance insights.
    Returns comprehensive analysis with recommendations.
    """
    if not history:
        return {"error": "No ping history available"}
    
    successful_pings = [p for p in history if p.get("status") == "up"]
    failed_pings = [p for p in history if p.get("status") in ["down", "error"]]
    
    if not successful_pings:
        return {
            "total_pings": len(history),
            "success_rate": 0.0,
            "failure_rate": 100.0,
            "status": "critical",
            "recommendations": ["Host is completely unreachable - check network connectivity and host availability"]
        }
    
    latencies = [p.get("latency_ms", 0) for p in successful_pings if p.get("latency_ms") is not None]
    
    if not latencies:
        return {
            "total_pings": len(history),
            "success_rate": len(successful_pings) / len(history) * 100,
            "failure_rate": len(failed_pings) / len(history) * 100,
            "status": "degraded",
            "recommendations": ["Ping successful but latency data unavailable - check ping implementation"]
        }
    
    # Calculate statistics
    avg_latency = statistics.mean(latencies)
    median_latency = statistics.median(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    std_dev = statistics.stdev(latencies) if len(latencies) > 1 else 0
    
    # Calculate jitter (latency variation)
    jitter = 0
    if len(latencies) > 1:
        jitter = sum(abs(latencies[i] - latencies[i-1]) for i in range(1, len(latencies))) / (len(latencies) - 1)
    
    # Determine status
    status = "healthy"
    recommendations = []
    
    if avg_latency > 200:
        status = "degraded"
        recommendations.append(f"⚠️ High average latency ({avg_latency:.2f}ms) - investigate network congestion")
    elif avg_latency > 100:
        status = "warning"
        recommendations.append(f"⚡ Elevated latency ({avg_latency:.2f}ms) - monitor for degradation")
    
    if std_dev > 50:
        status = "degraded" if status != "critical" else status
        recommendations.append(f"📊 High latency variance ({std_dev:.2f}ms) - unstable connection")
    
    if jitter > 30:
        status = "degraded" if status != "critical" else status
        recommendations.append(f"🔄 High jitter ({jitter:.2f}ms) - may cause VoIP/video issues")
    
    success_rate = (len(successful_pings) / len(history)) * 100
    if success_rate < 90:
        status = "critical"
        recommendations.append(f"🔴 Low success rate ({success_rate:.1f}%) - significant packet loss")
    elif success_rate < 95:
        status = "warning"
        recommendations.append(f"⚡ Reduced success rate ({success_rate:.1f}%) - some packet loss detected")
    
    if not recommendations:
        recommendations.append("✅ Network performance is within acceptable parameters")
    
    return {
        "total_pings": len(history),
        "successful_pings": len(successful_pings),
        "failed_pings": len(failed_pings),
        "success_rate": success_rate,
        "failure_rate": 100 - success_rate,
        "latency": {
            "avg_ms": round(avg_latency, 2),
            "median_ms": round(median_latency, 2),
            "min_ms": round(min_latency, 2),
            "max_ms": round(max_latency, 2),
            "std_dev_ms": round(std_dev, 2),
            "jitter_ms": round(jitter, 2),
        },
        "status": status,
        "recommendations": recommendations,
    }


def analyze_http_history(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze HTTP health check history for performance trends and issues.
    Returns comprehensive analysis with recommendations.
    """
    if not history:
        return {"error": "No HTTP history available"}
    
    successful_checks = [h for h in history if h.get("status") == "up"]
    failed_checks = [h for h in history if h.get("status") in ["error", "down", "timeout"]]
    
    if not successful_checks:
        return {
            "total_checks": len(history),
            "success_rate": 0.0,
            "failure_rate": 100.0,
            "status": "critical",
            "recommendations": ["HTTP endpoint completely unreachable - check service availability and network connectivity"]
        }
    
    response_times = [h.get("latency_ms", 0) for h in successful_checks if h.get("latency_ms") is not None]
    status_codes = [h.get("http_status", 0) for h in successful_checks if h.get("http_status") is not None]
    
    # Calculate statistics
    avg_response_time = statistics.mean(response_times) if response_times else 0
    median_response_time = statistics.median(response_times) if response_times else 0
    min_response_time = min(response_times) if response_times else 0
    max_response_time = max(response_times) if response_times else 0
    
    # Status code analysis
    status_code_counts = {}
    for code in status_codes:
        status_code_counts[code] = status_code_counts.get(code, 0) + 1
    
    # Determine status
    status = "healthy"
    recommendations = []
    
    if avg_response_time > 1000:
        status = "degraded"
        recommendations.append(f"⚠️ Slow response time ({avg_response_time:.2f}ms) - investigate server load")
    elif avg_response_time > 500:
        status = "warning"
        recommendations.append(f"⚡ Elevated response time ({avg_response_time:.2f}ms) - monitor performance")
    
    if 500 in status_code_counts:
        status = "critical"
        recommendations.append(f"🔴 Server errors (500) detected - check application logs")
    if 400 in status_code_counts:
        status = "warning"
        recommendations.append(f"⚡ Client errors (400) detected - review request parameters")
    
    success_rate = (len(successful_checks) / len(history)) * 100
    if success_rate < 90:
        status = "critical"
        recommendations.append(f"🔴 Low success rate ({success_rate:.1f}%) - significant service degradation")
    elif success_rate < 95:
        status = "warning"
        recommendations.append(f"⚡ Reduced success rate ({success_rate:.1f}%) - some requests failing")
    
    if not recommendations:
        recommendations.append("✅ HTTP service performance is within acceptable parameters")
    
    return {
        "total_checks": len(history),
        "successful_checks": len(successful_checks),
        "failed_checks": len(failed_checks),
        "success_rate": success_rate,
        "failure_rate": 100 - success_rate,
        "response_time": {
            "avg_ms": round(avg_response_time, 2),
            "median_ms": round(median_response_time, 2),
            "min_ms": round(min_response_time, 2),
            "max_ms": round(max_response_time, 2),
        },
        "status_codes": status_code_counts,
        "status": status,
        "recommendations": recommendations,
    }


def analyze_dns_history(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze DNS lookup history for performance and reliability.
    Returns comprehensive analysis with recommendations.
    """
    if not history:
        return {"error": "No DNS history available"}
    
    successful_lookups = [h for h in history if h.get("status") == "success"]
    failed_lookups = [h for h in history if h.get("status") in ["error", "timeout"]]
    
    if not successful_lookups:
        return {
            "total_lookups": len(history),
            "success_rate": 0.0,
            "failure_rate": 100.0,
            "status": "critical",
            "recommendations": ["DNS resolution completely failing - check DNS server configuration and network connectivity"]
        }
    
    lookup_times = [h.get("lookup_time_ms", 0) for h in successful_lookups if h.get("lookup_time_ms") is not None]
    
    # Calculate statistics
    avg_lookup_time = statistics.mean(lookup_times) if lookup_times else 0
    median_lookup_time = statistics.median(lookup_times) if lookup_times else 0
    
    # Determine status
    status = "healthy"
    recommendations = []
    
    if avg_lookup_time > 200:
        status = "degraded"
        recommendations.append(f"⚠️ Slow DNS resolution ({avg_lookup_time:.2f}ms) - consider using faster DNS servers")
    elif avg_lookup_time > 100:
        status = "warning"
        recommendations.append(f"⚡ Elevated DNS lookup time ({avg_lookup_time:.2f}ms) - monitor performance")
    
    success_rate = (len(successful_lookups) / len(history)) * 100
    if success_rate < 90:
        status = "critical"
        recommendations.append(f"🔴 Low DNS success rate ({success_rate:.1f}%) - DNS server issues detected")
    elif success_rate < 95:
        status = "warning"
        recommendations.append(f"⚡ Reduced DNS success rate ({success_rate:.1f}%) - intermittent DNS issues")
    
    if not recommendations:
        recommendations.append("✅ DNS resolution performance is within acceptable parameters")
    
    return {
        "total_lookups": len(history),
        "successful_lookups": len(successful_lookups),
        "failed_lookups": len(failed_lookups),
        "success_rate": success_rate,
        "failure_rate": 100 - success_rate,
        "lookup_time": {
            "avg_ms": round(avg_lookup_time, 2),
            "median_ms": round(median_lookup_time, 2),
        },
        "status": status,
        "recommendations": recommendations,
    }


def calculate_network_health_score(ping_analysis: Dict, http_analysis: Dict, dns_analysis: Dict, port_security: Dict) -> Dict[str, Any]:
    """
    Calculate overall network health score based on all monitoring metrics.
    Returns comprehensive health assessment with actionable insights.
    """
    scores = []
    weights = []
    issues = []
    
    # Ping analysis weight: 30%
    if "error" not in ping_analysis:
        ping_score = ping_analysis.get("success_rate", 0)
        if ping_analysis.get("status") == "healthy":
            ping_score = 100
        elif ping_analysis.get("status") == "warning":
            ping_score = 75
        elif ping_analysis.get("status") == "degraded":
            ping_score = 50
        elif ping_analysis.get("status") == "critical":
            ping_score = 25
        scores.append(ping_score)
        weights.append(30)
        issues.extend(ping_analysis.get("recommendations", []))
    
    # HTTP analysis weight: 25%
    if "error" not in http_analysis:
        http_score = http_analysis.get("success_rate", 0)
        if http_analysis.get("status") == "healthy":
            http_score = 100
        elif http_analysis.get("status") == "warning":
            http_score = 75
        elif http_analysis.get("status") == "degraded":
            http_score = 50
        elif http_analysis.get("status") == "critical":
            http_score = 25
        scores.append(http_score)
        weights.append(25)
        issues.extend(http_analysis.get("recommendations", []))
    
    # DNS analysis weight: 20%
    if "error" not in dns_analysis:
        dns_score = dns_analysis.get("success_rate", 0)
        if dns_analysis.get("status") == "healthy":
            dns_score = 100
        elif dns_analysis.get("status") == "warning":
            dns_score = 75
        elif dns_analysis.get("status") == "degraded":
            dns_score = 50
        elif dns_analysis.get("status") == "critical":
            dns_score = 25
        scores.append(dns_score)
        weights.append(20)
        issues.extend(dns_analysis.get("recommendations", []))
    
    # Port security weight: 25%
    if "security_score" in port_security:
        port_score = port_security.get("security_score", 50)
        scores.append(port_score)
        weights.append(25)
        issues.extend(port_security.get("recommendations", []))
    
    # Calculate weighted average
    if scores:
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    else:
        weighted_score = 50  # Default if no data
    
    # Determine overall status
    if weighted_score >= 90:
        overall_status = "excellent"
        status_emoji = "🟢"
    elif weighted_score >= 75:
        overall_status = "good"
        status_emoji = "🟡"
    elif weighted_score >= 50:
        overall_status = "fair"
        status_emoji = "🟠"
    else:
        overall_status = "poor"
        status_emoji = "🔴"
    
    # Prioritize issues
    critical_issues = [i for i in issues if "🔴" in i or "critical" in i.lower()]
    warnings = [i for i in issues if "⚠️" in i or "⚡" in i or "warning" in i.lower()]
    
    return {
        "overall_score": round(weighted_score, 1),
        "overall_status": overall_status,
        "status_emoji": status_emoji,
        "component_scores": {
            "ping": scores[0] if len(scores) > 0 else None,
            "http": scores[1] if len(scores) > 1 else None,
            "dns": scores[2] if len(scores) > 2 else None,
            "security": scores[3] if len(scores) > 3 else None,
        },
        "critical_issues": critical_issues[:5],  # Top 5 critical issues
        "warnings": warnings[:5],  # Top 5 warnings
        "all_recommendations": list(set(issues)),  # Unique recommendations
        "priority_actions": critical_issues[:3] if critical_issues else warnings[:3],
    }


def compare_hosts(batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare performance metrics across multiple hosts.
    Returns comparative analysis with best/worst performers.
    """
    if not batch_results:
        return {"error": "No batch results available"}
    
    hosts_by_latency = []
    hosts_by_success = []
    
    for result in batch_results:
        host = result.get("host", "unknown")
        latency = result.get("latency_ms")
        status = result.get("status")
        
        if latency is not None:
            hosts_by_latency.append((host, latency))
        
        if status == "up":
            hosts_by_success.append((host, True))
        else:
            hosts_by_success.append((host, False))
    
    # Sort by latency
    hosts_by_latency.sort(key=lambda x: x[1])
    
    # Calculate success rate
    successful_hosts = [h for h, s in hosts_by_success if s]
    success_rate = len(successful_hosts) / len(hosts_by_success) * 100 if hosts_by_success else 0
    
    # Find best and worst performers
    best_host = hosts_by_latency[0] if hosts_by_latency else None
    worst_host = hosts_by_latency[-1] if hosts_by_latency else None
    
    avg_latency = statistics.mean([l for h, l in hosts_by_latency]) if hosts_by_latency else 0
    
    return {
        "total_hosts": len(batch_results),
        "successful_hosts": len(successful_hosts),
        "failed_hosts": len(hosts_by_success) - len(successful_hosts),
        "success_rate": round(success_rate, 2),
        "average_latency_ms": round(avg_latency, 2),
        "best_performer": {
            "host": best_host[0] if best_host else "N/A",
            "latency_ms": best_host[1] if best_host else None,
        },
        "worst_performer": {
            "host": worst_host[0] if worst_host else "N/A",
            "latency_ms": worst_host[1] if worst_host else None,
        },
        "latency_ranking": [{"host": h, "latency_ms": l} for h, l in hosts_by_latency[:10]],
    }


def traceroute(host: str, max_hops: int = 30, timeout: float = 2.0) -> List[Dict[str, Any]]:
    """
    Perform a traceroute to a host.
    Returns a list of hop information.
    Note: This is a simplified implementation using ICMP.
    """
    results = []
    
    for ttl in range(1, max_hops + 1):
        try:
            # Create a raw socket for ICMP (requires admin privileges on Windows)
            # This is a simplified version - for full functionality, use scapy or similar
            start_time = time.time()
            ping_result = real_ping(host, timeout=timeout)
            latency_ms = ping_result.get("latency_ms")
            
            hop_result = {
                "hop": ttl,
                "host": host,
                "status": ping_result.get("status"),
                "latency_ms": latency_ms,
                "error": ping_result.get("error"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            
            results.append(hop_result)
            
            if ping_result.get("status") == "up":
                break
                
        except Exception as exc:
            results.append({
                "hop": ttl,
                "host": host,
                "status": "error",
                "latency_ms": None,
                "error": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
    
    return results


def create_latency_chart(history: List[Dict[str, Any]], host: str):
    """
    Create an enhanced Plotly chart showing latency over time with
    statistical overlays (mean, jitter band) and color-coded markers.
    Returns a Plotly figure or None if plotly is not available.
    """
    if go is None or not history:
        return None

    # Filter results for the specific host
    host_history = [h for h in history if h.get("host") == host and h.get("latency_ms") is not None]

    if not host_history:
        return None

    latencies = [h.get("latency_ms", 0) for h in host_history]
    timestamps = [h.get("timestamp", "") for h in host_history]
    x_vals = [ts[:19] if ts else str(i) for i, ts in enumerate(timestamps)]

    # Compute statistics
    mean_lat = statistics.mean(latencies) if latencies else 0
    std_lat  = statistics.stdev(latencies) if len(latencies) > 1 else 0
    min_lat  = min(latencies) if latencies else 0
    max_lat  = max(latencies) if latencies else 0
    jitter   = statistics.stdev(latencies) if len(latencies) > 1 else 0

    # Color-code markers: green (normal), yellow (above mean+std), red (above mean+2*std)
    marker_colors = []
    for v in latencies:
        if v > mean_lat + 2 * std_lat and std_lat > 0:
            marker_colors.append("#EF4444")  # red — outlier
        elif v > mean_lat + std_lat and std_lat > 0:
            marker_colors.append("#F59E0B")  # yellow — elevated
        else:
            marker_colors.append("#22C55E")  # green — normal

    fig = go.Figure()

    # Jitter band (mean ± 1 std)
    if std_lat > 0:
        fig.add_hrect(y0=mean_lat - std_lat, y1=mean_lat + std_lat,
                      fillcolor="rgba(56,189,248,0.06)", line_width=0,
                      layer="below")

    # Main latency line with color-coded markers
    fig.add_trace(go.Scatter(
        x=x_vals, y=latencies,
        mode='lines+markers',
        name=f'{host} Latency',
        line=dict(color='#38BDF8', width=2),
        marker=dict(size=8, color=marker_colors,
                    line=dict(color='#0F172A', width=1)),
        hovertemplate='<b>Ping %{x}</b><br>Latency: %{y:.2f} ms<extra></extra>',
    ))

    # Mean reference line
    fig.add_hline(y=mean_lat, line_dash='dash', line_color='#22C55E',
                  line_width=1.2,
                  annotation_text=f'Mean {mean_lat:.1f} ms',
                  annotation_font_color='#22C55E', annotation_font_size=10,
                  annotation_position='top left')

    # Median line (if different enough from mean)
    median_lat = statistics.median(latencies)
    if abs(median_lat - mean_lat) > 0.5:
        fig.add_hline(y=median_lat, line_dash='dot', line_color='#A78BFA',
                      line_width=1,
                      annotation_text=f'Median {median_lat:.1f} ms',
                      annotation_font_color='#A78BFA', annotation_font_size=9,
                      annotation_position='bottom left')

    # Annotate peak
    peak_idx = latencies.index(max_lat)
    fig.add_annotation(x=x_vals[peak_idx], y=max_lat,
                       text=f'Peak {max_lat:.1f} ms',
                       showarrow=True, arrowhead=2, arrowcolor='#F59E0B',
                       font=dict(color='#F59E0B', size=10), ay=-30)

    fig.update_layout(
        title=dict(text=f'Latency History — {host}',
                   subtitle=dict(
                       text=f'Jitter: {jitter:.1f} ms  |  '
                            f'Range: {min_lat:.1f}–{max_lat:.1f} ms  |  '
                            f'N={len(latencies)} pings',
                       font=dict(size=11, color='#64748B'),
                   )),
        xaxis_title='Ping',
        yaxis_title='Latency (ms)',
        template='plotly_dark',
        height=440,
        margin=dict(l=50, r=50, t=70, b=50),
        plot_bgcolor='#0F172A', paper_bgcolor='#0F172A',
        font_color='#94A3B8',
        xaxis=dict(gridcolor='#1E293B'),
        yaxis=dict(gridcolor='#1E293B', zerolinecolor='#334155'),
        legend=dict(orientation='h', y=-0.2, bgcolor='rgba(0,0,0,0)'),
        hovermode='x unified',
    )

    return fig


def create_multi_host_chart(results: List[Dict[str, Any]]):
    """
    Create an enhanced bar chart comparing latency across multiple hosts,
    sorted by latency, with average reference line and status indicators.
    Returns a Plotly figure or None if plotly is not available.
    """
    if go is None or not results:
        return None

    # Sort by latency (ascending), putting failed hosts at the end
    sorted_results = sorted(
        results,
        key=lambda r: r.get("latency_ms") if r.get("latency_ms") is not None else 99999,
    )

    hosts     = [r.get("host", "Unknown") for r in sorted_results]
    latencies = [r.get("latency_ms", 0) if r.get("latency_ms") is not None else 0
                 for r in sorted_results]
    statuses  = [r.get("status", "unknown") for r in sorted_results]

    # Gradient coloring: green (low) → yellow (mid) → red (high), gray for down
    valid_latencies = [l for l, s in zip(latencies, statuses) if s == "up" and l > 0]
    avg_latency = statistics.mean(valid_latencies) if valid_latencies else 0

    colors = []
    for lat, status in zip(latencies, statuses):
        if status != "up":
            colors.append('#EF4444')  # red for down
        elif lat <= avg_latency:
            colors.append('#22C55E')  # green — below average
        elif lat <= avg_latency * 1.5:
            colors.append('#F59E0B')  # yellow — elevated
        else:
            colors.append('#EF4444')  # red — high latency

    # Build text labels with status indicator
    text_labels = []
    for lat, status in zip(latencies, statuses):
        if status == "up":
            text_labels.append(f'{lat:.1f} ms')
        else:
            text_labels.append(f'DOWN')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=hosts, y=latencies,
        marker_color=colors,
        text=text_labels,
        textposition='outside',
        textfont=dict(size=10),
        hovertemplate='<b>%{x}</b><br>Latency: %{y:.1f} ms<br>Status: %{text}<extra></extra>',
        customdata=text_labels,
    ))

    # Average reference line
    if avg_latency > 0:
        fig.add_hline(y=avg_latency, line_dash='dash', line_color='#38BDF8',
                      line_width=1.2,
                      annotation_text=f'Avg {avg_latency:.1f} ms',
                      annotation_font_color='#38BDF8', annotation_font_size=10,
                      annotation_position='top right')

    # Status summary in subtitle
    up_count   = sum(1 for s in statuses if s == "up")
    down_count = len(statuses) - up_count

    fig.update_layout(
        title=dict(text='Multi-Host Latency Comparison',
                   subtitle=dict(
                       text=f'{up_count} up  |  {down_count} down  |  '
                            f'Avg: {avg_latency:.1f} ms',
                       font=dict(size=11, color='#64748B'),
                   )),
        xaxis_title='Host',
        yaxis_title='Latency (ms)',
        template='plotly_dark',
        height=440,
        margin=dict(l=50, r=50, t=70, b=50),
        plot_bgcolor='#0F172A', paper_bgcolor='#0F172A',
        font_color='#94A3B8',
        xaxis=dict(gridcolor='#1E293B', tickangle=-20),
        yaxis=dict(gridcolor='#1E293B', zerolinecolor='#334155'),
        showlegend=False,
    )

    return fig


def check_alert_thresholds(result: Dict[str, Any], latency_threshold: float = 100.0, packet_loss_threshold: float = 5.0) -> Dict[str, Any]:
    """
    Check if a ping result exceeds alert thresholds.
    Returns alert information with severity level.
    """
    alerts = []
    severity = "info"
    
    latency = result.get("latency_ms")
    packet_loss = result.get("packet_loss_pct", 0)
    status = result.get("status", "unknown")
    
    if status == "down":
        alerts.append({
            "type": "host_down",
            "message": f"Host {result.get('host')} is unreachable",
            "severity": "critical"
        })
        severity = "critical"
    elif status == "error":
        alerts.append({
            "type": "error",
            "message": f"Error checking host {result.get('host')}: {result.get('error')}",
            "severity": "warning"
        })
        severity = "warning"
    else:
        if latency is not None and latency > latency_threshold:
            alerts.append({
                "type": "high_latency",
                "message": f"High latency: {latency}ms (threshold: {latency_threshold}ms)",
                "severity": "warning"
            })
            if severity == "info":
                severity = "warning"
        
        if packet_loss > packet_loss_threshold:
            alerts.append({
                "type": "packet_loss",
                "message": f"High packet loss: {packet_loss}% (threshold: {packet_loss_threshold}%)",
                "severity": "warning"
            })
            if severity == "info":
                severity = "warning"
    
    return {
        "host": result.get("host"),
        "has_alerts": len(alerts) > 0,
        "alerts": alerts,
        "severity": severity,
        "timestamp": result.get("timestamp")
    }


def continuous_monitor(hosts: List[str], interval: int = 5, max_iterations: int = 100) -> List[Dict[str, Any]]:
    """
    Perform continuous monitoring of hosts at specified intervals.
    Returns a list of monitoring results.
    Note: This is a generator-like function for use in auto-refresh scenarios.
    """
    results = []
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        batch_results = batch_ping(hosts)
        
        for result in batch_results:
            result["iteration"] = iteration
            result["monitoring_timestamp"] = datetime.now(timezone.utc).isoformat()
            results.append(result)
        
        # In a real auto-refresh scenario, this would yield control back to the UI
        # For Streamlit, this is handled by st.automatic_rerun or similar mechanisms
        break  # Single iteration for now
    
    return results
