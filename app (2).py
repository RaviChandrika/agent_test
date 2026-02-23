"""
QA-GPT Streamlit Application — Phase 13

A two-column Streamlit UI for the QA-GPT workflow.
Layout: Sidebar | Progress Bar | [Left: Chat/Q&A/Review] | [Right: Artifact Tabs]

Agents: Requirement Analysis → Test Scenario → Test Case → Test Plan →
        Step Definition → XPath → Utility Method → Test Script
"""

import sys
import os
import tempfile
import logging
from pathlib import Path
from uuid import uuid4

# ============================================================================
# PATH SETUP (PRD Section 8.6 — cross-platform compatibility)
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config.settings import settings
from src.graph.state import (
    AgentState,
    WorkflowStage,
    WorkflowStatus,
    QASession,
    JudgeResult,
    create_initial_state,
)
from src.knowledge.retrieval.context_fetcher import fetch_context
from src.agents.llm_client import verify_llm_connection
from langgraph.types import Command

logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIG (must be first Streamlit command)
# ============================================================================
st.set_page_config(
    page_title="QA-GPT 🧪",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    /* Progress milestone badges */
    .milestone-done { 
        background: #22c55e; color: white; padding: 4px 10px; 
        border-radius: 12px; font-size: 0.72rem; font-weight: 600;
        text-align: center; 
    }
    .milestone-active { 
        background: #3b82f6; color: white; padding: 4px 10px; 
        border-radius: 12px; font-size: 0.72rem; font-weight: 600;
        text-align: center; animation: pulse 2s infinite;
    }
    .milestone-pending { 
        background: #e5e7eb; color: #6b7280; padding: 4px 10px; 
        border-radius: 12px; font-size: 0.72rem; font-weight: 600;
        text-align: center; 
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    /* Review gate styling */
    .review-gate {
        border: 2px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    /* Compact artifact version info */
    .artifact-meta {
        font-size: 0.8rem;
        color: #6b7280;
    }
    /* Input page card styling */
    .input-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .input-section-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .agent-pipeline {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
        margin: 0.5rem 0;
    }
    .agent-badge {
        background: #dbeafe;
        color: #1e40af;
        border-radius: 8px;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# CONSTANTS
# ============================================================================

MILESTONE_LABELS = [
    "Req. Analysis",
    "Test Scenarios",
    "Test Cases",
    "Test Plan",
    "Step Defs",
    "XPath",
    "Utility Methods",
    "Test Script",
    "Done",
]

MILESTONE_MAP = {
    WorkflowStage.QA_INTERACTION:           0,   # Requirement Analysis agent
    WorkflowStage.REQUIREMENTS_SPEC_GEN:    0,
    WorkflowStage.JUDGE_REQUIREMENTS:       0,
    WorkflowStage.HUMAN_REVIEW_SPEC:        0,
    WorkflowStage.STRATEGY:                 1,   # Test Scenario agent
    WorkflowStage.JUDGE_STRATEGY:           1,
    WorkflowStage.HUMAN_REVIEW_STRATEGY:    1,
    WorkflowStage.TEST_CASE_GENERATION:     2,   # Test Case agent
    WorkflowStage.JUDGE_TEST_CASES:         2,
    WorkflowStage.HUMAN_REVIEW_TEST_CASES:  2,
    WorkflowStage.CODE_STRUCTURE_PLANNING:  3,   # Test Plan agent
    WorkflowStage.JUDGE_CODE_PLAN:          3,
    WorkflowStage.HUMAN_REVIEW_CODE_PLAN:   3,
    WorkflowStage.SCRIPTING:                7,   # Test Script agent (final code)
    WorkflowStage.JUDGE_CODE:               7,
    WorkflowStage.HUMAN_REVIEW_CODE:        7,
    WorkflowStage.COMPLETED:                8,
}

# Map gate names to their document content/version keys
GATE_CONFIG = {
    "spec": {
        "content_key": "requirements_spec_content",
        "version_key": "current_requirements_spec_version",
        "judge_eval_key": "judge_requirements_evaluation",
        "label": "Requirements Analysis",
    },
    "strategy": {
        "content_key": "strategy_content",
        "version_key": "current_strategy_version",
        "judge_eval_key": "judge_strategy_evaluation",
        "label": "Test Scenarios",
    },
    "test_cases": {
        "content_key": "gherkin_content",
        "version_key": "current_test_cases_version",
        "judge_eval_key": "judge_test_cases_evaluation",
        "label": "Test Cases (Gherkin)",
    },
    "code_plan": {
        "content_key": "code_plan_content",
        "version_key": "current_code_plan_version",
        "judge_eval_key": "judge_code_plan_evaluation",
        "label": "Test Plan",
    },
    "code": {
        "content_key": "script_content",
        "version_key": "current_script_version",
        "judge_eval_key": "judge_code_evaluation",
        "label": "Test Script",
    },
}

# Map workflow stages to gate names for interrupt detection
HUMAN_REVIEW_STAGES = {
    WorkflowStage.HUMAN_REVIEW_SPEC:        "spec",
    WorkflowStage.HUMAN_REVIEW_STRATEGY:    "strategy",
    WorkflowStage.HUMAN_REVIEW_TEST_CASES:  "test_cases",
    WorkflowStage.HUMAN_REVIEW_CODE_PLAN:   "code_plan",
    WorkflowStage.HUMAN_REVIEW_CODE:        "code",
}

# Test type options
TEST_TYPES = ["UI", "ETL", "API"]


# ============================================================================
# 13.1 — SESSION STATE INITIALIZATION
# ============================================================================

def _init_session_state():
    """Initialize all session state keys if not already set."""
    defaults = {
        # --- Input page ---
        "input_page_done": False,
        "project_config": {},          # Stores all input-page values
        # --- Core workflow ---
        "messages": [],
        "graph_state": None,
        "thread_id": None,
        "workflow_running": False,
        "awaiting_qa": False,
        "awaiting_review": None,
        "bedrock_ok": None,
        "tech_context_path": settings.tech_context_path,
        "codebase_map_path": settings.codebase_map_path,
        "llm_provider": settings.llm_provider,
        "qa_graph": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _get_graph():
    """Get or create the graph instance for this session."""
    if st.session_state.qa_graph is None:
        from src.graph.builder import build_graph
        st.session_state.qa_graph = build_graph()
    return st.session_state.qa_graph


# ============================================================================
# 13.2 — SIDEBAR
# ============================================================================

def _render_sidebar():
    """Render the sidebar with provider info, connection test, and session info."""
    st.title("QA-GPT 🧪")
    st.caption("AI-Powered QA Workflow")

    st.divider()

    # Agent pipeline display
    st.markdown('<div class="input-section-title">🤖 Agent Pipeline</div>', unsafe_allow_html=True)
    agents = [
        "Req. Analysis", "Test Scenario", "Test Case", "Test Plan",
        "Step Definition", "XPath", "Utility Method", "Test Script",
    ]
    for agent in agents:
        st.markdown(f'<span class="agent-badge">⚙ {agent}</span>', unsafe_allow_html=True)

    st.divider()

    # Provider display
    provider = st.session_state.llm_provider
    if provider == "bedrock":
        st.info("🔷 Using: **AWS Bedrock**")
    else:
        st.info("🔶 Using: **Google Gemini**")

    # Verify connection button
    if st.button("🔌 Verify Connection", use_container_width=True):
        with st.spinner("Testing connection..."):
            ok = verify_llm_connection()
        if ok:
            st.success("✅ Connection verified!")
            st.session_state.bedrock_ok = True
        else:
            st.error("❌ Connection failed. Check API key in .env")
            st.session_state.bedrock_ok = False

    st.divider()

    # Project config summary (shown after input page)
    if st.session_state.input_page_done:
        cfg = st.session_state.project_config
        st.subheader("📋 Project Config")
        if cfg.get("test_type"):
            st.caption(f"**Type:** {cfg['test_type']}")
        if cfg.get("jira_id"):
            st.caption(f"**JIRA:** {cfg['jira_id']}")
        if cfg.get("user_story_jira_id"):
            st.caption(f"**Story:** {cfg['user_story_jira_id']}")
        if cfg.get("project_mode"):
            st.caption(f"**Project:** {cfg['project_mode']}")
        if cfg.get("starts_with"):
            st.caption(f"**Starts With:** `{cfg['starts_with']}`")
        if cfg.get("branch"):
            st.caption(f"**Branch:** `{cfg['branch']}`")

        st.divider()

    # Session info
    st.subheader("📊 Session Info")
    state = st.session_state.graph_state
    if state:
        workflow_id = state.get("workflow_id", "N/A")
        current_stage = state.get("current_stage", "N/A")
        cost = state.get("accumulated_cost_usd", 0.0)
        status = state.get("workflow_status", "N/A")

        st.caption(f"**Workflow:** `{str(workflow_id)[:8]}...`")
        st.caption(f"**Stage:** {current_stage}")
        st.caption(f"**Status:** {status}")
        st.caption(f"**Cost:** ${cost:.4f}")
    else:
        st.caption("No active workflow")

    st.divider()

    # Back to input / New session
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⬅ Back", use_container_width=True, type="secondary",
                     disabled=not st.session_state.input_page_done):
            st.session_state.input_page_done = False
            st.rerun()
    with col_b:
        if st.button("🔄 Reset", use_container_width=True, type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ============================================================================
# 13.3 — INPUT PAGE
# ============================================================================

def _render_input_page():
    """Full-width input page shown before the main workflow."""
    st.markdown("## ⚙️ Project Configuration")
    st.caption("Configure your QA pipeline. All fields are optional — fill what's relevant and click **Next**.")

    st.divider()

    # ── Row 1: Type + Project mode ────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="input-section-title">Test Type</div>', unsafe_allow_html=True)
        test_type = st.selectbox(
            "Type",
            options=["(none)", "UI", "ETL", "API"],
            index=0,
            label_visibility="collapsed",
            key="inp_test_type",
        )
        test_type = None if test_type == "(none)" else test_type

    with col2:
        st.markdown('<div class="input-section-title">Project Mode</div>', unsafe_allow_html=True)
        project_mode = st.radio(
            "Project Mode",
            options=["New Project", "Existing Project"],
            horizontal=True,
            label_visibility="collapsed",
            key="inp_project_mode",
        )

    st.divider()

    # ── Row 2: GIT details ────────────────────────────────────────────────────
    st.markdown('<div class="input-section-title">🔗 Git Configuration</div>', unsafe_allow_html=True)
    gcol1, gcol2, gcol3 = st.columns([3, 1.5, 2])
    with gcol1:
        git_url = st.text_input(
            "GIT URL",
            placeholder="https://github.com/org/repo.git",
            key="inp_git_url",
        )
    with gcol2:
        branch = st.text_input(
            "Branch",
            placeholder="main",
            key="inp_branch",
        )
    with gcol3:
        git_pat = st.text_input(
            "GIT PAT (Personal Access Token)",
            placeholder="ghp_••••••••••••",
            type="password",
            key="inp_git_pat",
        )

    st.divider()

    # ── Row 3: JIRA details ───────────────────────────────────────────────────
    st.markdown('<div class="input-section-title">🎫 JIRA Details</div>', unsafe_allow_html=True)
    jcol1, jcol2, jcol3 = st.columns([1.5, 1.5, 2])
    with jcol1:
        jira_id = st.text_input(
            "JIRA ID",
            placeholder="PROJ-1234",
            key="inp_jira_id",
        )
    with jcol2:
        user_story_jira_id = st.text_input(
            "User Story JIRA ID",
            placeholder="PROJ-5678",
            key="inp_user_story_jira_id",
        )
    with jcol3:
        starts_with = st.text_input(
            "Starts With",
            placeholder="e.g. TC_, TS_",
            key="inp_starts_with",
        )

    st.divider()

    # ── Row 4: File uploads ───────────────────────────────────────────────────
    st.markdown('<div class="input-section-title">📂 Input Files</div>', unsafe_allow_html=True)

    is_ui = (test_type == "UI")

    fcol1, fcol2, fcol3, fcol4 = st.columns(4)

    with fcol1:
        req_files = st.file_uploader(
            "📄 Requirements",
            type=["md", "txt", "pdf", "docx", "xlsx"],
            accept_multiple_files=True,
            key="inp_req_files",
            help="Upload one or more requirement documents",
        )

    with fcol2:
        context_files = st.file_uploader(
            "🗂 Context Files",
            type=["md", "txt", "pdf", "docx"],
            accept_multiple_files=True,
            key="inp_context_files",
            help="Upload supporting context / tech docs",
        )

    with fcol3:
        template_files = st.file_uploader(
            "📐 Templates / Examples",
            type=["md", "txt", "feature", "py", "java", "xlsx"],
            accept_multiple_files=True,
            key="inp_template_files",
            help="Upload existing test templates or example scripts",
        )

    with fcol4:
        html_dom_files = st.file_uploader(
            "🌐 HTML DOM" + ("" if is_ui else " (UI only)"),
            type=["html", "htm", "xml"],
            accept_multiple_files=True,
            key="inp_html_dom",
            disabled=not is_ui,
            help="HTML DOM files — only available when Type is UI",
        )

    # File summary feedback
    file_counts = {
        "Requirements": len(req_files) if req_files else 0,
        "Context": len(context_files) if context_files else 0,
        "Templates": len(template_files) if template_files else 0,
        "HTML DOM": len(html_dom_files) if (html_dom_files and is_ui) else 0,
    }
    total_files = sum(file_counts.values())
    if total_files > 0:
        parts = [f"{v} {k}" for k, v in file_counts.items() if v > 0]
        st.success(f"✅ {total_files} file(s) ready — {', '.join(parts)}")

    st.divider()

    # ── Next button ───────────────────────────────────────────────────────────
    _, btn_col, _ = st.columns([3, 2, 3])
    with btn_col:
        if st.button("▶  Next", type="primary", use_container_width=True):
            # Persist uploaded files to temp dir
            def _save_files(file_list, prefix):
                paths = []
                if not file_list:
                    return paths
                for i, f in enumerate(file_list):
                    tmp = os.path.join(tempfile.gettempdir(), f"{prefix}_{i}_{f.name}")
                    Path(tmp).write_bytes(f.getvalue())
                    paths.append(tmp)
                return paths

            cfg = {
                "test_type":            test_type,
                "git_url":              git_url.strip() or None,
                "branch":               branch.strip() or None,
                "git_pat":              git_pat.strip() or None,
                "project_mode":         project_mode,
                "jira_id":              jira_id.strip() or None,
                "user_story_jira_id":   user_story_jira_id.strip() or None,
                "starts_with":          starts_with.strip() or None,
                "req_file_paths":       _save_files(req_files, "req"),
                "context_file_paths":   _save_files(context_files, "ctx"),
                "template_file_paths":  _save_files(template_files, "tmpl"),
                "html_dom_paths":       _save_files(html_dom_files, "dom") if is_ui else [],
            }

            # Feed context files into the session paths settings use
            if cfg["context_file_paths"]:
                st.session_state.tech_context_path = cfg["context_file_paths"]
            if cfg["req_file_paths"]:
                st.session_state.codebase_map_path = cfg["req_file_paths"]

            st.session_state.project_config = cfg
            st.session_state.input_page_done = True
            st.rerun()


# ============================================================================
# 13.4 — PIPELINE PROGRESS BAR
# ============================================================================

def _render_progress_bar(current_stage):
    """Render the pipeline progress bar with agent milestones."""
    if current_stage is None:
        current_milestone = -1
    elif isinstance(current_stage, WorkflowStage):
        current_milestone = MILESTONE_MAP.get(current_stage, -1)
    else:
        current_milestone = -1

    progress_value = min((current_milestone + 1) / len(MILESTONE_LABELS), 1.0) if current_milestone >= 0 else 0.0
    st.progress(progress_value)

    cols = st.columns(len(MILESTONE_LABELS))
    for i, (col, label) in enumerate(zip(cols, MILESTONE_LABELS)):
        with col:
            if current_milestone >= 0 and i < current_milestone:
                st.markdown(f'<div class="milestone-done">✓ {label}</div>', unsafe_allow_html=True)
            elif i == current_milestone:
                st.markdown(f'<div class="milestone-active">● {label}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="milestone-pending">{label}</div>', unsafe_allow_html=True)


# ============================================================================
# 13.5 — ARTIFACT TABS (Right Column)
# ============================================================================

def _render_judge_badge(state, judge_eval_key):
    """Render judge evaluation badge if available."""
    judge_eval = state.get(judge_eval_key)
    if judge_eval is None:
        return

    score = judge_eval.score
    result = judge_eval.result

    if isinstance(result, JudgeResult):
        result_value = result.value
    else:
        result_value = str(result)

    if result_value == "PASS":
        st.success(f"✅ Judge: **PASS** (Score: {score}/100)")
    elif result_value == "NEEDS_HUMAN":
        st.warning(f"⚠️ Judge: **NEEDS HUMAN** (Score: {score}/100)")
    elif result_value == "FAIL":
        st.error(f"❌ Judge: **FAIL** (Score: {score}/100)")

    if judge_eval.feedback:
        with st.expander("Judge Feedback"):
            st.markdown(judge_eval.feedback)


def _render_artifact_tabs(state):
    """Render the 8 agent artifact tabs in the right column."""
    if state is None:
        state = {}

    tabs = st.tabs([
        "📋 Req. Analysis",
        "🗺️ Test Scenarios",
        "🧩 Test Cases",
        "📐 Test Plan",
        "🪜 Step Definitions",
        "🔍 XPath",
        "🛠️ Utility Methods",
        "💻 Test Script",
    ])

    # ── Tab 1: Requirement Analysis ──────────────────────────────────────────
    with tabs[0]:
        version = state.get("current_requirements_spec_version", 0)
        if version > 0:
            st.caption(f"Version: {version}")
        _render_judge_badge(state, "judge_requirements_evaluation")
        content = state.get("requirements_spec_content", "")
        if content:
            st.markdown(content)
        else:
            st.markdown("*Not generated yet.*")

    # ── Tab 2: Test Scenarios ─────────────────────────────────────────────────
    with tabs[1]:
        version = state.get("current_strategy_version", 0)
        if version > 0:
            st.caption(f"Version: {version}")
        _render_judge_badge(state, "judge_strategy_evaluation")
        content = state.get("strategy_content", "")
        if content:
            st.markdown(content)
        else:
            st.markdown("*Not generated yet.*")

    # ── Tab 3: Test Cases (Gherkin) ───────────────────────────────────────────
    with tabs[2]:
        version = state.get("current_test_cases_version", 0)
        if version > 0:
            st.caption(f"Version: {version}")
        _render_judge_badge(state, "judge_test_cases_evaluation")
        content = state.get("gherkin_content", "")
        if content:
            st.code(content, language="gherkin")
        else:
            st.markdown("*Not generated yet.*")

    # ── Tab 4: Test Plan ──────────────────────────────────────────────────────
    with tabs[3]:
        version = state.get("current_code_plan_version", 0)
        if version > 0:
            st.caption(f"Version: {version}")
        _render_judge_badge(state, "judge_code_plan_evaluation")
        content = state.get("code_plan_content", "")
        if content:
            st.markdown(content)
        else:
            st.markdown("*Not generated yet.*")

    # ── Tab 5: Step Definitions ───────────────────────────────────────────────
    with tabs[4]:
        content = state.get("step_definitions_content", "")
        if content:
            st.code(content, language="python")
            st.download_button(
                label="⬇️ Download Step Definitions",
                data=content,
                file_name=state.get("step_definitions_filename", "step_definitions.py"),
                mime="text/x-python",
                use_container_width=True,
            )
        else:
            st.markdown("*Not generated yet.*")

    # ── Tab 6: XPath ──────────────────────────────────────────────────────────
    with tabs[5]:
        content = state.get("xpath_content", "")
        cfg = st.session_state.project_config
        if cfg.get("test_type") and cfg["test_type"] != "UI":
            st.info("ℹ️ XPath locators are primarily used for UI test types.")
        if content:
            st.code(content, language="python")
            st.download_button(
                label="⬇️ Download XPath Locators",
                data=content,
                file_name=state.get("xpath_filename", "locators.py"),
                mime="text/x-python",
                use_container_width=True,
            )
        else:
            st.markdown("*Not generated yet.*")

    # ── Tab 7: Utility Methods ────────────────────────────────────────────────
    with tabs[6]:
        content = state.get("utility_methods_content", "")
        if content:
            st.code(content, language="python")
            st.download_button(
                label="⬇️ Download Utility Methods",
                data=content,
                file_name=state.get("utility_methods_filename", "utils.py"),
                mime="text/x-python",
                use_container_width=True,
            )
        else:
            st.markdown("*Not generated yet.*")

    # ── Tab 8: Test Script ────────────────────────────────────────────────────
    with tabs[7]:
        version = state.get("current_script_version", 0)
        if version > 0:
            st.caption(f"Version: {version}")
        _render_judge_badge(state, "judge_code_evaluation")
        content = state.get("script_content", "")
        if content:
            st.code(content, language="python")
            filename = state.get("script_filename", "test_generated.py")
            st.download_button(
                label="⬇️ Download Test Script",
                data=content,
                file_name=filename,
                mime="text/x-python",
                use_container_width=True,
            )
        else:
            st.markdown("*Not generated yet.*")


# ============================================================================
# 13.6 — Q&A FORM (Left Column)
# ============================================================================

def _render_qa_form():
    """Render the Q&A clarifying questions form."""
    state = st.session_state.graph_state
    if state is None:
        return

    qa_sessions = state.get("qa_sessions", [])
    if not qa_sessions:
        return

    latest_session = qa_sessions[-1]
    confidence = latest_session.ai_confidence if hasattr(latest_session, "ai_confidence") else 0.0
    questions = latest_session.questions if hasattr(latest_session, "questions") else []

    st.subheader("🔍 Clarifying Questions")
    st.caption(f"AI Confidence: {confidence:.0%} (threshold: {settings.qa_confidence_threshold:.0%})")
    st.progress(min(confidence, 1.0))

    if not questions:
        st.info("No questions generated. Pipeline should proceed.")
        return

    with st.form("qa_answers", clear_on_submit=True):
        answers = {}
        for i, q in enumerate(questions):
            if isinstance(q, str):
                q_text = q
                q_id = f"q_{i}"
                required = True
            elif isinstance(q, dict):
                q_text = q.get("text", q.get("id", str(q)))
                q_id = q.get("id", f"q_{i}")
                required = q.get("is_required", True)
            else:
                q_text = getattr(q, "text", str(q))
                q_id = getattr(q, "id", f"q_{i}")
                required = getattr(q, "is_required", True)

            label = f"{'* ' if required else ''}{q_text}"
            answers[q_id] = st.text_area(label, key=f"qa_answer_{i}", height=80)

        submitted = st.form_submit_button("📤 Submit Answers", use_container_width=True, type="primary")

        if submitted:
            latest_session.answers = answers
            latest_session.status = "answered"

            graph = _get_graph()
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            try:
                graph.update_state(config, {"qa_sessions": qa_sessions})
                st.session_state.awaiting_qa = False
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"Submitted {len(answers)} answers to clarifying questions.",
                })
                _resume_pipeline()
            except Exception as e:
                st.error(f"Failed to submit answers: {e}")


# ============================================================================
# 13.7 — HUMAN REVIEW GATE UI (Left Column)
# ============================================================================

def _render_human_review_gate():
    """Render the human review gate UI."""
    gate_name = st.session_state.awaiting_review
    if gate_name is None:
        return

    state = st.session_state.graph_state
    if state is None:
        return

    gate_config = GATE_CONFIG.get(gate_name, {})
    label = gate_config.get("label", gate_name.title())
    content_key = gate_config.get("content_key", "")
    judge_eval_key = gate_config.get("judge_eval_key", "")

    st.subheader(f"👤 Human Review Required: {label}")

    judge_eval = state.get(judge_eval_key)
    if judge_eval:
        result = judge_eval.result
        if isinstance(result, JudgeResult):
            result_value = result.value
        else:
            result_value = str(result)

        if result_value == "NEEDS_HUMAN":
            st.warning(f"⚠️ The AI judge flagged this for human review. Score: {judge_eval.score}/100")
            if judge_eval.feedback:
                st.markdown(f"**Judge feedback:** {judge_eval.feedback}")
        elif result_value == "PASS":
            st.success(f"✅ Judge passed this artifact. Score: {judge_eval.score}/100")
        elif result_value == "FAIL":
            st.error(f"❌ Judge failed this artifact. Score: {judge_eval.score}/100")
            if judge_eval.feedback:
                st.markdown(f"**Judge feedback:** {judge_eval.feedback}")

    st.info("📖 Review the artifact in the panel on the right, then make your decision below.")

    with st.form("human_review_form", clear_on_submit=True):
        decision = st.radio(
            "Decision",
            ["APPROVE", "REJECT", "EDIT"],
            horizontal=True,
            index=0,
        )

        feedback = ""
        edited_content = ""

        if decision in ("REJECT", "EDIT"):
            feedback = st.text_area(
                "Feedback / Instructions",
                placeholder="Describe what needs to change...",
                height=100,
            )

        if decision == "EDIT":
            current_content = state.get(content_key, "")
            edited_content = st.text_area(
                "Edit Document",
                value=current_content,
                height=300,
            )

        guidance = st.text_area(
            "💡 Guidance for Next Stage (optional)",
            placeholder="Add extra context or instructions for the next stage's AI agent...",
            height=80,
        )

        submitted = st.form_submit_button(
            f"Submit {decision}",
            use_container_width=True,
            type="primary",
        )

        if submitted:
            human_response = {
                "decision": decision,
                "feedback": feedback,
                "edited_content": edited_content,
                "guidance": guidance,
            }

            try:
                st.session_state.awaiting_review = None
                decision_emoji = {"APPROVE": "✅", "REJECT": "🔄", "EDIT": "✏️"}.get(decision, "")
                st.session_state.messages.append({
                    "role": "user",
                    "content": (
                        f"{decision_emoji} {decision} — {label}. {feedback}"
                        if feedback else f"{decision_emoji} {decision} — {label}"
                    ),
                })
                st.toast(f"{decision_emoji} {label}: {decision}!")
                _resume_pipeline(resume_value=human_response)
            except Exception as e:
                st.error(f"Failed to submit review: {e}")


# ============================================================================
# 13.8 — GRAPH STREAMING LOOP
# ============================================================================

def _run_pipeline(user_input: str):
    """Start a new pipeline execution from user input."""
    graph = _get_graph()

    team_context = fetch_context(
        team_id=settings.team_id,
        tech_context_path=st.session_state.tech_context_path,
        codebase_map_path=st.session_state.codebase_map_path,
    )

    initial_state = create_initial_state(
        raw_input=user_input,
        team_context=team_context,
        team_id=settings.team_id,
        qa_confidence_threshold=settings.qa_confidence_threshold,
    )

    thread_id = initial_state["thread_id"]
    st.session_state.thread_id = thread_id
    st.session_state.workflow_running = True

    config = {"configurable": {"thread_id": thread_id}}
    _stream_graph(graph, initial_state, config)


def _resume_pipeline(resume_value=None):
    """Resume the pipeline after human input (Q&A or review gate)."""
    graph = _get_graph()
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    if resume_value is not None:
        _stream_graph(graph, Command(resume=resume_value), config)
    else:
        _stream_graph(graph, None, config)


def _stream_graph(graph, input_state, config):
    """Stream graph execution, handling interrupts and state updates."""
    try:
        with st.spinner("🤖 AI is working..."):
            final_state = None
            for state_snapshot in graph.stream(input_state, config=config, stream_mode="values"):
                final_state = state_snapshot
                st.session_state.graph_state = state_snapshot

            if final_state is not None:
                current_stage = final_state.get("current_stage")
                workflow_status = final_state.get("workflow_status")

                if (current_stage == WorkflowStage.QA_INTERACTION
                        and final_state.get("qa_completed") is False):
                    st.session_state.awaiting_qa = True
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "🔍 I need some clarifying information. Please answer the questions below.",
                    })
                    st.rerun()
                    return

                if current_stage in HUMAN_REVIEW_STAGES:
                    gate_name = HUMAN_REVIEW_STAGES[current_stage]
                    st.session_state.awaiting_review = gate_name
                    gate_label = GATE_CONFIG[gate_name]["label"]
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"👤 **Human Review Required:** {gate_label}. Please review the artifact and make your decision.",
                    })
                    st.rerun()
                    return

                if current_stage == WorkflowStage.COMPLETED or workflow_status == WorkflowStatus.COMPLETED:
                    st.session_state.workflow_running = False
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "✅ **Pipeline Complete!** Your test script has been generated. Check the Test Script tab to download it.",
                    })
                    st.toast("🎉 Pipeline complete!")
                    st.rerun()
                    return

                if current_stage == WorkflowStage.FAILED or workflow_status == WorkflowStatus.FAILED:
                    error_msg = final_state.get("error_message", "Unknown error")
                    st.session_state.workflow_running = False
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ **Pipeline Failed:** {error_msg}",
                    })
                    st.rerun()
                    return

                try:
                    graph_state = graph.get_state(config)
                    if graph_state and hasattr(graph_state, 'next') and graph_state.next:
                        next_node = graph_state.next[0] if graph_state.next else None
                        if next_node and next_node.startswith("human_review_"):
                            gate_suffix = next_node.replace("human_review_", "")
                            if gate_suffix in GATE_CONFIG:
                                st.session_state.awaiting_review = gate_suffix
                                gate_label = GATE_CONFIG[gate_suffix]["label"]
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": f"👤 **Human Review Required:** {gate_label}. Please review the artifact and make your decision.",
                                })
                                st.rerun()
                                return
                except Exception:
                    pass

                st.rerun()

    except Exception as e:
        error_str = str(e)
        error_type = type(e).__name__

        if "GraphInterrupt" in error_type or "interrupt" in error_str.lower():
            if st.session_state.graph_state:
                current_stage = st.session_state.graph_state.get("current_stage")
                if current_stage in HUMAN_REVIEW_STAGES:
                    gate_name = HUMAN_REVIEW_STAGES[current_stage]
                    st.session_state.awaiting_review = gate_name
                    gate_label = GATE_CONFIG[gate_name]["label"]
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"👤 **Human Review Required:** {gate_label}.",
                    })
                    st.rerun()
                    return

        st.session_state.workflow_running = False
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ **Error:** {error_str}",
        })
        st.error(f"Pipeline error: {error_str}")
        st.rerun()


# ============================================================================
# 13.9 — CHAT INTERFACE
# ============================================================================

def _render_chat_history():
    """Render chat history messages."""
    for msg in st.session_state.messages:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        with st.chat_message(role):
            st.markdown(content)


# ============================================================================
# MAIN APP LAYOUT
# ============================================================================

def main():
    """Main application entry point."""
    _init_session_state()

    # --- Sidebar (always visible) ---
    with st.sidebar:
        _render_sidebar()

    # ── INPUT PAGE (pre-workflow) ─────────────────────────────────────────────
    if not st.session_state.input_page_done:
        _render_input_page()
        return

    # ── MAIN WORKFLOW UI ──────────────────────────────────────────────────────

    # Project config badge strip
    cfg = st.session_state.project_config
    badge_parts = []
    if cfg.get("test_type"):
        badge_parts.append(f"🔖 **{cfg['test_type']}**")
    if cfg.get("project_mode"):
        badge_parts.append(f"📁 {cfg['project_mode']}")
    if cfg.get("jira_id"):
        badge_parts.append(f"🎫 {cfg['jira_id']}")
    if cfg.get("branch"):
        badge_parts.append(f"🌿 `{cfg['branch']}`")
    if badge_parts:
        st.caption("  ·  ".join(badge_parts))

    # Progress bar (full width)
    state = st.session_state.graph_state
    current_stage = state.get("current_stage") if state else None
    _render_progress_bar(current_stage)

    st.divider()

    # Two-column layout
    col1, col2 = st.columns([2, 3], gap="large")

    # --- Left Column: Chat / Q&A / Review ---
    with col1:
        if st.session_state.awaiting_qa:
            _render_qa_form()
        elif st.session_state.awaiting_review:
            _render_human_review_gate()
        else:
            _render_chat_history()

            if state and state.get("workflow_status") == WorkflowStatus.FAILED:
                st.error(f"❌ Workflow failed: {state.get('error_message', 'Unknown error')}")

            user_input = st.chat_input(
                "Describe the feature you want to test...",
                disabled=st.session_state.workflow_running,
            )

            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "🚀 Starting QA-GPT pipeline...",
                })
                _run_pipeline(user_input)

    # --- Right Column: Artifact Tabs ---
    with col2:
        _render_artifact_tabs(state)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
