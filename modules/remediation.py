"""Human-in-the-loop remediation workflow with state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import uuid

import streamlit as st

try:
    from modules.storage import save_remediation_incident, get_remediation_incidents
except (ImportError, KeyError):
    def save_remediation_incident(*_a, **_kw): pass  # type: ignore[misc]
    def get_remediation_incidents(*_a, **_kw): return []  # type: ignore[misc]


class RemediationState(Enum):
    """States in the remediation workflow."""
    DETECTED = "detected"
    DIAGNOSED = "diagnosed"
    SUGGESTED = "suggested"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    LOGGED = "logged"


class RemediationActionType(Enum):
    """Types of remediation actions."""
    RESTART_SERVICE = "restart_service"
    FLAG_FOR_REVIEW = "flag_for_review"
    BLOCK_IP = "block_ip"
    ADJUST_THRESHOLD = "adjust_threshold"
    CLEAR_CACHE = "clear_cache"
    INVESTIGATE_FURTHER = "investigate_further"


@dataclass
class RemediationAction:
    """A remediation action proposal."""
    action_type: RemediationActionType
    description: str
    device: str
    severity: str  # low, medium, high
    estimated_impact: str
    rollback_plan: str


@dataclass
class Incident:
    """An incident requiring remediation."""
    incident_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    alert_id: str = ""
    device: str = ""
    issue_type: str = ""
    state: RemediationState = RemediationState.DETECTED
    detected_at: datetime = field(default_factory=datetime.now)
    diagnosed_at: datetime | None = None
    suggested_at: datetime | None = None
    approved_at: datetime | None = None
    rejected_at: datetime | None = None
    diagnosis: str = ""
    suggested_action: RemediationAction | None = None
    approval_notes: str = ""
    rejection_reason: str = ""
    state_history: list[dict[str, Any]] = field(default_factory=list)
    
    def transition_to(self, new_state: RemediationState, notes: str = "") -> None:
        """Transition to a new state with logging."""
        old_state = self.state
        self.state = new_state
        
        # Update timestamps
        now = datetime.now()
        if new_state == RemediationState.DIAGNOSED:
            self.diagnosed_at = now
        elif new_state == RemediationState.SUGGESTED:
            self.suggested_at = now
        elif new_state == RemediationState.APPROVED:
            self.approved_at = now
        elif new_state == RemediationState.REJECTED:
            self.rejected_at = now
        
        # Log state transition
        self.state_history.append({
            "from_state": old_state.value,
            "to_state": new_state.value,
            "timestamp": now.isoformat(),
            "notes": notes,
        })


class RemediationEngine:
    """Engine for managing remediation workflows."""

    # Safe action set - these are simulated, not real infrastructure changes
    SAFE_ACTIONS = {
        RemediationActionType.RESTART_SERVICE: {
            "description": "Simulate service restart",
            "impact": "Brief service interruption",
            "rollback": "Service auto-restart on failure",
        },
        RemediationActionType.FLAG_FOR_REVIEW: {
            "description": "Flag device for human review",
            "impact": "No immediate impact",
            "rollback": "Remove flag from review queue",
        },
        RemediationActionType.BLOCK_IP: {
            "description": "Simulate IP blocking",
            "impact": "Blocked IP cannot communicate",
            "rollback": "Remove IP from blocklist",
        },
        RemediationActionType.ADJUST_THRESHOLD: {
            "description": "Adjust monitoring threshold",
            "impact": "Changes alert sensitivity",
            "rollback": "Restore previous threshold",
        },
        RemediationActionType.CLEAR_CACHE: {
            "description": "Simulate cache clearing",
            "impact": "Temporary performance impact",
            "rollback": "Cache rebuilds automatically",
        },
        RemediationActionType.INVESTIGATE_FURTHER: {
            "description": "Schedule deeper investigation",
            "impact": "No immediate action",
            "rollback": "Cancel investigation",
        },
    }

    def __init__(self):
        self.incidents: dict[str, Incident] = {}
        self._load_from_storage()

    def create_incident(self, alert: dict[str, Any]) -> Incident:
        """Create a new incident from an alert."""
        incident = Incident(
            alert_id=str(alert.get("id", "")),
            device=alert.get("device", "unknown"),
            issue_type=alert.get("metric", "unknown"),
            diagnosis=alert.get("message", ""),
        )
        
        self.incidents[incident.incident_id] = incident
        self._save_to_storage()
        
        return incident

    def diagnose_incident(self, incident_id: str, diagnosis: str) -> Incident:
        """Add diagnosis to an incident."""
        incident = self.incidents.get(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")
        
        incident.diagnosis = diagnosis
        incident.transition_to(RemediationState.DIAGNOSED, f"Diagnosis: {diagnosis}")
        self._save_to_storage()
        
        return incident

    def suggest_action(
        self,
        incident_id: str,
        action_type: RemediationActionType,
        description: str,
        severity: str = "medium",
    ) -> Incident:
        """Suggest a remediation action for an incident."""
        incident = self.incidents.get(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")
        
        action_info = self.SAFE_ACTIONS.get(action_type, {})
        
        incident.suggested_action = RemediationAction(
            action_type=action_type,
            description=description,
            device=incident.device,
            severity=severity,
            estimated_impact=action_info.get("impact", "Unknown"),
            rollback_plan=action_info.get("rollback", "Manual"),
        )
        
        incident.transition_to(RemediationState.SUGGESTED, f"Suggested action: {action_type.value}")
        self._save_to_storage()
        
        return incident

    def submit_for_approval(self, incident_id: str) -> Incident:
        """Submit incident for human approval."""
        incident = self.incidents.get(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")
        
        if not incident.suggested_action:
            raise ValueError(f"Incident {incident_id} has no suggested action")
        
        incident.transition_to(RemediationState.PENDING_APPROVAL, "Submitted for approval")
        self._save_to_storage()
        
        return incident

    def approve_action(self, incident_id: str, notes: str = "") -> Incident:
        """Approve and execute the suggested action."""
        incident = self.incidents.get(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")
        
        if incident.state != RemediationState.PENDING_APPROVAL:
            raise ValueError(f"Incident {incident_id} is not pending approval")
        
        # Simulate action execution
        execution_result = self._execute_action(incident.suggested_action)
        
        incident.approval_notes = f"{notes}\nExecution result: {execution_result}"
        incident.transition_to(RemediationState.APPROVED, f"Approved. Notes: {notes}")
        self._save_to_storage()
        
        return incident

    def reject_action(self, incident_id: str, reason: str) -> Incident:
        """Reject the suggested action."""
        incident = self.incidents.get(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")
        
        incident.rejection_reason = reason
        incident.transition_to(RemediationState.REJECTED, f"Rejected. Reason: {reason}")
        self._save_to_storage()
        
        return incident

    def get_pending_approvals(self) -> list[Incident]:
        """Get all incidents pending approval."""
        return [
            inc for inc in self.incidents.values()
            if inc.state == RemediationState.PENDING_APPROVAL
        ]

    def get_incident_history(self, limit: int = 50) -> list[Incident]:
        """Get incident history."""
        incidents = list(self.incidents.values())
        incidents.sort(key=lambda x: x.detected_at, reverse=True)
        return incidents[:limit]

    def _execute_action(self, action: RemediationAction) -> str:
        """Simulate execution of a remediation action."""
        # This is a simulation - in production, this would execute real actions
        return f"Simulated execution of {action.action_type.value} on {action.device}"

    def _save_to_storage(self) -> None:
        """Save incidents to session state and SQLite database."""
        if "remediation_incidents" not in st.session_state:
            st.session_state.remediation_incidents = {}
        
        # Convert incidents to serializable format
        serializable = {}
        for inc_id, incident in self.incidents.items():
            incident_data = {
                "incident_id": incident.incident_id,
                "alert_id": incident.alert_id,
                "device": incident.device,
                "issue_type": incident.issue_type,
                "state": incident.state.value,
                "detected_at": incident.detected_at.isoformat(),
                "diagnosed_at": incident.diagnosed_at.isoformat() if incident.diagnosed_at else None,
                "suggested_at": incident.suggested_at.isoformat() if incident.suggested_at else None,
                "approved_at": incident.approved_at.isoformat() if incident.approved_at else None,
                "rejected_at": incident.rejected_at.isoformat() if incident.rejected_at else None,
                "diagnosis": incident.diagnosis,
                "suggested_action_type": incident.suggested_action.action_type.value if incident.suggested_action else None,
                "suggested_action_description": incident.suggested_action.description if incident.suggested_action else None,
                "suggested_action_severity": incident.suggested_action.severity if incident.suggested_action else None,
                "suggested_action_impact": incident.suggested_action.estimated_impact if incident.suggested_action else None,
                "suggested_action_rollback": incident.suggested_action.rollback_plan if incident.suggested_action else None,
                "approval_notes": incident.approval_notes,
                "rejection_reason": incident.rejection_reason,
                "state_history": incident.state_history,
            }
            serializable[inc_id] = incident_data
            
            # Save to SQLite
            try:
                save_remediation_incident(incident_data)
            except Exception:
                # Silently fail if SQLite save fails
                pass
        
        st.session_state.remediation_incidents = serializable

    def _load_from_storage(self) -> None:
        """Load incidents from SQLite database and session state."""
        # First load from SQLite for persistence
        try:
            db_incidents = get_remediation_incidents(limit=100)
            for data in db_incidents:
                incident = Incident(
                    incident_id=data["incident_id"],
                    alert_id=data["alert_id"],
                    device=data["device"],
                    issue_type=data["issue_type"],
                    state=RemediationState(data["state"]),
                    detected_at=datetime.fromisoformat(data["detected_at"]),
                    diagnosed_at=datetime.fromisoformat(data["diagnosed_at"]) if data["diagnosed_at"] else None,
                    suggested_at=datetime.fromisoformat(data["suggested_at"]) if data["suggested_at"] else None,
                    approved_at=datetime.fromisoformat(data["approved_at"]) if data["approved_at"] else None,
                    rejected_at=datetime.fromisoformat(data["rejected_at"]) if data["rejected_at"] else None,
                    diagnosis=data["diagnosis"],
                    approval_notes=data["approval_notes"],
                    rejection_reason=data["rejection_reason"],
                    state_history=data["state_history"],
                )
                
                # Reconstruct suggested action
                if data.get("suggested_action_type"):
                    incident.suggested_action = RemediationAction(
                        action_type=RemediationActionType(data["suggested_action_type"]),
                        description=data.get("suggested_action_description", ""),
                        device=data["device"],
                        severity=data.get("suggested_action_severity", "medium"),
                        estimated_impact=data.get("suggested_action_impact", ""),
                        rollback_plan=data.get("suggested_action_rollback", ""),
                    )
                
                self.incidents[incident.incident_id] = incident
        except Exception:
            # Silently fail if SQLite load fails
            pass
        
        # Then merge with session state (session state takes precedence)
        if "remediation_incidents" in st.session_state:
            serializable = st.session_state.remediation_incidents
            
            for inc_id, data in serializable.items():
                incident = Incident(
                    incident_id=data["incident_id"],
                    alert_id=data["alert_id"],
                    device=data["device"],
                    issue_type=data["issue_type"],
                    state=RemediationState(data["state"]),
                    detected_at=datetime.fromisoformat(data["detected_at"]),
                    diagnosed_at=datetime.fromisoformat(data["diagnosed_at"]) if data["diagnosed_at"] else None,
                    suggested_at=datetime.fromisoformat(data["suggested_at"]) if data["suggested_at"] else None,
                    approved_at=datetime.fromisoformat(data["approved_at"]) if data["approved_at"] else None,
                    rejected_at=datetime.fromisoformat(data["rejected_at"]) if data["rejected_at"] else None,
                    diagnosis=data["diagnosis"],
                    approval_notes=data["approval_notes"],
                    rejection_reason=data["rejection_reason"],
                    state_history=data["state_history"],
                )
                
                # Reconstruct suggested action
                if data.get("suggested_action_type"):
                    incident.suggested_action = RemediationAction(
                        action_type=RemediationActionType(data["suggested_action_type"]),
                        description=data.get("suggested_action_description", ""),
                        device=data["device"],
                        severity=data.get("suggested_action_severity", "medium"),
                        estimated_impact=data.get("suggested_action_impact", ""),
                        rollback_plan=data.get("suggested_action_rollback", ""),
                    )
                
                self.incidents[inc_id] = incident


def get_remediation_engine() -> RemediationEngine:
    """Get or create the remediation engine singleton."""
    if "remediation_engine" not in st.session_state:
        st.session_state.remediation_engine = RemediationEngine()
    return st.session_state.remediation_engine
