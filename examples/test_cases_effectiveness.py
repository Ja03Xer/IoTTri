from __future__ import annotations

from models import TaskDescriptor, LearningObjective, GroundTruthArtifact, LearnerState
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
def build_web_task() -> TaskDescriptor:
    return TaskDescriptor(
        task_id="lab-web-001",
        scenario_identifier="smart-camera web-interface misconfiguration",
        device_type="smart_camera",
        scenario_summary=(
            "Authorised IoT pentesting lab focused on weak authentication, "
            "unencrypted transport, and exposed web administration."
        ),
        exposed_protocols_interfaces=["HTTP", "MQTT"],
        allowed_tool_categories=["network scanner", "HTTP client"],
        learning_objectives=[
            LearningObjective("Apply", "Identify insecure configuration"),
            LearningObjective("Analyze", "Reason about trust boundaries"),
            LearningObjective("Evaluate", "Justify remediation"),
        ],
        ground_truth_artifacts=[
            GroundTruthArtifact("weak_auth", "evidence", "Weak or default credentials accepted"),
            GroundTruthArtifact("no_https", "evidence", "Administrative interface uses unencrypted HTTP"),
        ],
        active_attack_surfaces=["device_web_interface", "local_network_services"],
        laboratory_scope="Authorised sandbox only",
    )


def build_mqtt_task() -> TaskDescriptor:
    return TaskDescriptor(
        task_id="lab-mqtt-001",
        scenario_identifier="smart-sensor mqtt broker exposure",
        device_type="smart_sensor",
        scenario_summary=(
            "Authorised IoT pentesting lab focused on MQTT topic exposure, "
            "weak broker authentication, and trust-boundary analysis."
        ),
        exposed_protocols_interfaces=["MQTT", "HTTP"],
        allowed_tool_categories=["network scanner", "message client"],
        learning_objectives=[
            LearningObjective("Apply", "Identify insecure messaging configuration"),
            LearningObjective("Analyze", "Reason about message exposure and trust boundaries"),
            LearningObjective("Evaluate", "Recommend mitigations for broker security"),
        ],
        ground_truth_artifacts=[
            GroundTruthArtifact("mqtt_no_auth", "evidence", "Broker accepts unauthenticated connections"),
            GroundTruthArtifact("mqtt_topic_exposure", "evidence", "Sensitive telemetry published without access control"),
        ],
        active_attack_surfaces=["local_network_services", "cloud_companion_api"],
        laboratory_scope="Authorised sandbox only",
    )


def build_ble_task() -> TaskDescriptor:
    return TaskDescriptor(
        task_id="lab-ble-001",
        scenario_identifier="wearable ble pairing weakness",
        device_type="wearable_device",
        scenario_summary=(
            "Authorised IoT pentesting lab focused on BLE pairing assumptions, "
            "device trust, and insecure nearby access."
        ),
        exposed_protocols_interfaces=["BLE"],
        allowed_tool_categories=["wireless scanner", "BLE inspection tool"],
        learning_objectives=[
            LearningObjective("Apply", "Identify insecure pairing indicators"),
            LearningObjective("Analyze", "Reason about local attacker proximity and trust"),
            LearningObjective("Evaluate", "Propose safer pairing and authentication design"),
        ],
        ground_truth_artifacts=[
            GroundTruthArtifact("weak_pairing", "evidence", "Pairing lacks strong verification"),
            GroundTruthArtifact("nearby_data_exposure", "evidence", "Nearby party may access device data or functions"),
        ],
        active_attack_surfaces=["wireless_interfaces"],
        laboratory_scope="Authorised sandbox only",
    )


def build_firmware_task() -> TaskDescriptor:
    return TaskDescriptor(
        task_id="lab-fw-001",
        scenario_identifier="smart-plug firmware update integrity weakness",
        device_type="smart_plug",
        scenario_summary=(
            "Authorised IoT pentesting lab focused on firmware update validation, "
            "integrity checking, and secure update design."
        ),
        exposed_protocols_interfaces=["HTTP", "local update channel"],
        allowed_tool_categories=["HTTP client", "firmware analysis utility"],
        learning_objectives=[
            LearningObjective("Apply", "Identify insecure firmware update design"),
            LearningObjective("Analyze", "Reason about integrity and authenticity"),
            LearningObjective("Evaluate", "Recommend secure update controls"),
        ],
        ground_truth_artifacts=[
            GroundTruthArtifact("unsigned_update", "evidence", "Firmware update lacks signature verification"),
            GroundTruthArtifact("update_channel_risk", "evidence", "Update retrieval or validation is not trustworthy"),
        ],
        active_attack_surfaces=["firmware_storage_and_update"],
        laboratory_scope="Authorised sandbox only",
    )


def _state(turn_index: int, phase: str, hint_level: int, summary: str, evidence=None, progress=None) -> LearnerState:
    return LearnerState(
        turn_index,
        phase,
        hint_level,
        summary,
        evidence_markers=evidence or [],
        progress_markers=progress or [],
    )


def build_test_cases() -> list[dict]:
    web_task = build_web_task()
    mqtt_task = build_mqtt_task()
    ble_task = build_ble_task()
    firmware_task = build_firmware_task()

    return [
        # Web / observation / reasoning
        {
            "case_id": "c1",
            "group": "web_recon",
            "task": web_task,
            "state": _state(1, "reconnaissance", 1, "Found login page and HTTP service banner.", ["login page observed"], ["identified admin interface"]),
            "query": "What should I examine next in the authorised lab?",
            "expected_focus": "prioritise evidence collection on authentication and transport security",
        },
        {
            "case_id": "c2",
            "group": "web_recon",
            "task": web_task,
            "state": _state(2, "reconnaissance", 1, "I found a web admin page and an MQTT service but I am unsure which observations matter first.", ["admin page observed", "mqtt service observed"], []),
            "query": "Which observations should I prioritise collecting next?",
            "expected_focus": "prioritise security-relevant observations and trust boundaries",
        },
        {
            "case_id": "c3",
            "group": "web_recon",
            "task": web_task,
            "state": _state(3, "reconnaissance", 1, "The device exposes a login form and uses plain HTTP rather than HTTPS.", ["http login page observed"], ["transport issue noticed"]),
            "query": "Why is the transport choice important before I make any security conclusion?",
            "expected_focus": "recognise credential exposure and insecure transport risk",
        },
        {
            "case_id": "c4",
            "group": "web_recon",
            "task": web_task,
            "state": _state(4, "reconnaissance", 1, "I can see a settings page but I have not yet verified whether access control is strong.", ["settings page observed"], []),
            "query": "What kinds of evidence would help me reason about weak authentication or insecure configuration?",
            "expected_focus": "identify evidence for weak authentication and insecure config",
        },
        {
            "case_id": "c5",
            "group": "web_recon",
            "task": web_task,
            "state": _state(5, "reconnaissance", 1, "I only know that a login page exists.", ["login page observed"], []),
            "query": "Is the existence of a login page itself a sign that security is good?",
            "expected_focus": "reject false assurance from presence of login page",
        },
        {
            "case_id": "c6",
            "group": "web_recon",
            "task": web_task,
            "state": _state(6, "reconnaissance", 2, "The admin page is reachable and I want to reason about risk without jumping to conclusions.", ["admin page reachable"], ["initial evidence gathered"]),
            "query": "How should I distinguish between evidence, suspicion, and conclusion?",
            "expected_focus": "separate observed facts from interpretation",
        },

        # Web / validation / reporting
        {
            "case_id": "c7",
            "group": "web_validation",
            "task": web_task,
            "state": _state(7, "vulnerability_analysis", 1, "Observed that the login page uses HTTP and requests administrator credentials.", ["http login page observed"], ["transport security considered"]),
            "query": "What does this suggest about the device security?",
            "expected_focus": "identify insecure transport and credential exposure risk",
        },
        {
            "case_id": "c8",
            "group": "web_validation",
            "task": web_task,
            "state": _state(8, "validation", 2, "I observed the interface has no HTTPS and credentials may be exposed on the local network.", ["no https observed", "credentials risk suspected"], ["evidence collected"]),
            "query": "How should I justify this finding in a report?",
            "expected_focus": "justify finding with evidence and impact",
        },
        {
            "case_id": "c9",
            "group": "web_validation",
            "task": web_task,
            "state": _state(9, "vulnerability_analysis", 1, "I think the device is safe because it has a login page.", [], []),
            "query": "Is this enough evidence to conclude the device is secure?",
            "expected_focus": "correct the misconception and request stronger evidence",
        },
        {
            "case_id": "c10",
            "group": "web_validation",
            "task": web_task,
            "state": _state(10, "reporting", 2, "I identified weak authentication and lack of encrypted transport.", ["weak auth", "no https"], ["drafting findings"]),
            "query": "How can I explain the trust-boundary impact clearly?",
            "expected_focus": "explain trust-boundary and attacker opportunity",
        },
        {
            "case_id": "c11",
            "group": "web_validation",
            "task": web_task,
            "state": _state(11, "vulnerability_analysis", 1, "The device asks for credentials, so maybe authentication is enough protection.", ["login requirement observed"], []),
            "query": "What should I verify before assuming authentication is strong?",
            "expected_focus": "suggest verification of strength rather than assumption",
        },
        {
            "case_id": "c12",
            "group": "web_validation",
            "task": web_task,
            "state": _state(12, "validation", 2, "I want to phrase remediation suggestions for weak authentication and insecure transport.", ["weak auth", "unencrypted transport"], ["remediation thinking"]),
            "query": "What remediation themes should I discuss?",
            "expected_focus": "discuss authentication hardening and secure transport",
        },

        # MQTT
        {
            "case_id": "c13",
            "group": "mqtt",
            "task": mqtt_task,
            "state": _state(13, "reconnaissance", 1, "I discovered an MQTT broker and device telemetry topics.", ["mqtt broker observed", "telemetry topics observed"], ["messaging surface identified"]),
            "query": "What should I inspect first to reason about message exposure risk?",
            "expected_focus": "check authentication, topic access, and sensitive data exposure",
        },
        {
            "case_id": "c14",
            "group": "mqtt",
            "task": mqtt_task,
            "state": _state(14, "vulnerability_analysis", 1, "The broker appears reachable and may allow topic subscription without clear access control.", ["broker reachable"], []),
            "query": "Why is this significant from a trust-boundary perspective?",
            "expected_focus": "explain message trust boundary and unauthorised access risk",
        },
        {
            "case_id": "c15",
            "group": "mqtt",
            "task": mqtt_task,
            "state": _state(15, "validation", 2, "I suspect telemetry is exposed more broadly than intended.", ["topic exposure suspected"], ["risk framing started"]),
            "query": "How should I describe the security impact without exaggeration?",
            "expected_focus": "describe impact carefully and evidence-based",
        },
        {
            "case_id": "c16",
            "group": "mqtt",
            "task": mqtt_task,
            "state": _state(16, "reporting", 2, "I want to recommend mitigations for insecure broker configuration.", ["broker auth weakness"], ["report drafting"]),
            "query": "What remediation directions should I include for MQTT security?",
            "expected_focus": "authentication, authorisation, topic restrictions, encryption",
        },
        {
            "case_id": "c17",
            "group": "mqtt",
            "task": mqtt_task,
            "state": _state(17, "vulnerability_analysis", 1, "I found telemetry topics, but I am not sure whether this is just functionality or a security issue.", ["telemetry topics observed"], []),
            "query": "What evidence would distinguish normal functionality from insecure exposure?",
            "expected_focus": "need evidence of access control and sensitivity",
        },
        {
            "case_id": "c18",
            "group": "mqtt",
            "task": mqtt_task,
            "state": _state(18, "validation", 2, "The learner wants to reason about message confidentiality and integrity.", ["broker identified"], ["security property discussion"]),
            "query": "How should I relate exposed topics to confidentiality and integrity concerns?",
            "expected_focus": "link exposure to confidentiality and potential control/integrity risks",
        },

        # BLE
        {
            "case_id": "c19",
            "group": "ble",
            "task": ble_task,
            "state": _state(19, "reconnaissance", 1, "A wearable advertises over BLE and pairing appears simple.", ["ble advertisement observed"], ["pairing surface identified"]),
            "query": "What should I look for when reasoning about pairing weakness?",
            "expected_focus": "look for verification strength and nearby attacker assumptions",
        },
        {
            "case_id": "c20",
            "group": "ble",
            "task": ble_task,
            "state": _state(20, "vulnerability_analysis", 1, "The pairing process seems convenient but may not strongly verify the device or user.", ["pairing weakness suspected"], []),
            "query": "Why does convenience in pairing sometimes create security risk?",
            "expected_focus": "explain tradeoff between usability and authentication assurance",
        },
        {
            "case_id": "c21",
            "group": "ble",
            "task": ble_task,
            "state": _state(21, "validation", 2, "I want to explain the impact of nearby unauthorized access in a realistic way.", ["nearby risk considered"], ["impact wording started"]),
            "query": "How should I justify the local attacker model and its impact?",
            "expected_focus": "justify local proximity attacker model and realistic impact",
        },

        # Firmware
        {
            "case_id": "c22",
            "group": "firmware",
            "task": firmware_task,
            "state": _state(22, "reconnaissance", 1, "The device appears to support firmware updates, but I have not yet reasoned about validation.", ["update support observed"], ["update surface identified"]),
            "query": "What evidence matters most when assessing firmware update security?",
            "expected_focus": "integrity, authenticity, signatures, trusted source",
        },
        {
            "case_id": "c23",
            "group": "firmware",
            "task": firmware_task,
            "state": _state(23, "vulnerability_analysis", 1, "I suspect the update mechanism does not verify authenticity strongly enough.", ["authenticity concern"], []),
            "query": "Why is authenticity verification essential in firmware updates?",
            "expected_focus": "prevent untrusted firmware and preserve device integrity",
        },
        {
            "case_id": "c24",
            "group": "firmware",
            "task": firmware_task,
            "state": _state(24, "reporting", 2, "I need to recommend controls for a weak firmware update design.", ["update integrity weakness"], ["report drafting"]),
            "query": "What mitigation themes should I include for secure firmware updates?",
            "expected_focus": "signatures, secure channels, validation, rollback protection if appropriate",
        },
    ]
