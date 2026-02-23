"""
QA-GPT Streamlit Application — Phase 13

Flow: Input Configuration Page → Sidebar | Progress Bar | [Left: Chat/Q&A/Review] | [Right: Artifact Tabs]

New: A pre-flight Input Configuration step collects all run parameters
(testing type, Git details, JIRA info, file uploads, start node) before
the main chat/pipeline workflow begins.
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
    /* ── Progress milestone badges ── */
    .milestone-done { 
        background: #22c55e; color: white; padding: 4px 10px; 
        border-radius: 12px; font-size: 0.75rem; font-weight: 600;
        text-align: center; 
    }
    .milestone-active { 
        background: #3b82f6; color: white; padding: 4px 10px; 
        border-radius: 12px; font-size: 0.75rem; font-weight: 600;
        text-align: center; animation: pulse 2s infinite;
    }
    .milestone-pending { 
        background: #e5e7eb; color: #6b7280; padding: 4px 10px; 
        border-radius: 12px; font-size: 0.75rem; font-weight: 600;
        text-align: center; 
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    /* ── Review gate styling ── */
    .review-gate {
        border: 2px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    /* ── Compact artifact version info ── */
    .artifact-meta {
        font-size: 0.8rem;
        color: #6b7280;
    }

    /* ══════════════════════════════════════════════
       INPUT CONFIGURATION PAGE STYLES
    ══════════════════════════════════════════════ */

    /* Hero header */
    .cfg-hero {
        background: linear-gradient(135deg, #1e3a5f 0%, #1e40af 60%, #1d4ed8 100%);
        border-radius: 12px;
        padding: 32px 36px 28px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .cfg-hero::after {
        content: '🧪';
        position: absolute;
        right: 32px; top: 50%;
        transform: translateY(-50%);
        font-size: 72px;
        opacity: 0.15;
    }
    .cfg-hero h1 {
        color: #fff !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        margin: 0 0 6px 0 !important;
        padding: 0 !important;
    }
    .cfg-hero p {
        color: #bfdbfe !important;
        font-size: 0.95rem !important;
        margin: 0 !important;
    }

    /* Pipeline visual */
    .pipeline-bar {
        display: flex;
        align-items: center;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 20px;
        gap: 0;
        overflow-x: auto;
        margin-bottom: 28px;
    }
    .pipeline-bar::-webkit-scrollbar { height: 4px; }
    .pipeline-bar::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 2px; }
    .p-node {
        display: flex;
        flex-direction: column;
        align-items: center;
        min-width: 80px;
        flex-shrink: 0;
    }
    .p-icon {
        width: 38px; height: 38px;
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 16px;
        margin-bottom: 5px;
        border: 1.5px solid;
    }
    .p-label {
        font-size: 10px;
        color: #64748b;
        text-align: center;
        line-height: 1.3;
        font-weight: 500;
    }
    .p-arrow { color: #cbd5e1; font-size: 18px; padding: 0 4px; margin-bottom: 18px; flex-shrink: 0; }

    /* node colours */
    .pn-req  { background:#ede9fe; border-color:#7c3aed; }
    .pn-scn  { background:#f0f9ff; border-color:#0284c7; }
    .pn-tc   { background:#f0fdf4; border-color:#16a34a; }
    .pn-step { background:#fff7ed; border-color:#ea580c; }
    .pn-xp   { background:#fdf2f8; border-color:#c026d3; }
    .pn-util { background:#eff6ff; border-color:#2563eb; }
    .pn-ts   { background:#fefce8; border-color:#ca8a04; }

    /* Section headers inside config form */
    .cfg-section {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 24px 0 12px;
        padding-bottom: 8px;
        border-bottom: 2px solid #e2e8f0;
    }
    .cfg-section-icon {
        width: 30px; height: 30px;
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 14px;
        flex-shrink: 0;
    }
    .cfg-section-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #1e293b;
    }
    .cfg-section-subtitle {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-left: auto;
    }

    /* DOM disabled overlay */
    .dom-disabled {
        background: #f8fafc;
        border: 1.5px dashed #cbd5e1;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
    }

    /* Summary chips */
    .chip-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0 16px; }
    .chip {
        background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe;
        border-radius: 20px; padding: 3px 12px; font-size: 0.75rem; font-weight: 600;
    }
    .chip.green { background:#f0fdf4; color:#15803d; border-color:#bbf7d0; }
    .chip.amber { background:#fff7ed; color:#c2410c; border-color:#fed7aa; }
    .chip.gray  { background:#f1f5f9; color:#475569; border-color:#e2e8f0; }

    /* Start node selector cards */
    .node-cards { display: flex; flex-direction: column; gap: 8px; margin-top: 4px; }

    /* Config summary box */
    .cfg-summary {
        background: #f0fdf4;
        border: 1.5px solid #86efac;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 20px 0 8px;
    }
    .cfg-summary-title {
        font-weight: 700; font-size: 0.9rem; color: #15803d; margin-bottom: 8px;
    }

    /* Launch button */
    div[data-testid="stButton"] > button.launch-btn {
        background: linear-gradient(135deg, #1d4ed8, #7c3aed) !important;
        color: white !important;
        border: none !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        padding: 16px !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 14px rgba(29, 78, 216, 0.35) !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# CONSTANTS
# ============================================================================

MILESTONE_LABELS = ["Q&A", "Spec", "Strategy", "Test Cases", "Code Plan", "Script", "Done"]

MILESTONE_MAP = {
    WorkflowStage.QA_INTERACTION: 0,
    WorkflowStage.REQUIREMENTS_SPEC_GEN: 1,
    WorkflowStage.JUDGE_REQUIREMENTS: 1,
    WorkflowStage.HUMAN_REVIEW_SPEC: 1,
    WorkflowStage.STRATEGY: 2,
    WorkflowStage.JUDGE_STRATEGY: 2,
    WorkflowStage.HUMAN_REVIEW_STRATEGY: 2,
    WorkflowStage.TEST_CASE_GENERATION: 3,
    WorkflowStage.JUDGE_TEST_CASES: 3,
    WorkflowStage.HUMAN_REVIEW_TEST_CASES: 3,
    WorkflowStage.CODE_STRUCTURE_PLANNING: 4,
    WorkflowStage.JUDGE_CODE_PLAN: 4,
    WorkflowStage.HUMAN_REVIEW_CODE_PLAN: 4,
    WorkflowStage.SCRIPTING: 5,
    WorkflowStage.JUDGE_CODE: 5,
    WorkflowStage.HUMAN_REVIEW_CODE: 5,
    WorkflowStage.COMPLETED: 6,
}

# Map gate names to their document content/version keys
GATE_CONFIG = {
    "spec": {
        "content_key": "requirements_spec_content",
        "version_key": "current_requirements_spec_version",
        "judge_eval_key": "judge_requirements_evaluation",
        "label": "Requirements Specification",
    },
    "strategy": {
        "content_key": "strategy_content",
        "version_key": "current_strategy_version",
        "judge_eval_key": "judge_strategy_evaluation",
        "label": "Test Strategy",
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
        "label": "Code Structure Plan",
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
    WorkflowStage.HUMAN_REVIEW_SPEC: "spec",
    WorkflowStage.HUMAN_REVIEW_STRATEGY: "strategy",
    WorkflowStage.HUMAN_REVIEW_TEST_CASES: "test_cases",
    WorkflowStage.HUMAN_REVIEW_CODE_PLAN: "code_plan",
    WorkflowStage.HUMAN_REVIEW_CODE: "code",
}


# ============================================================================
# 12.1 — SESSION STATE INITIALIZATION
# ============================================================================

def _init_session_state():
    """Initialize all session state keys if not already set."""
    defaults = {
        # ── Existing workflow keys ──
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
        # ── Input configuration page ──
        "config_complete": False,      # False = show config page, True = show pipeline
        "cfg_testing_type": "UI",
        "cfg_project_mode": "New Project",
        "cfg_git_url": "",
        "cfg_git_branch": "main",
        "cfg_git_pat": "",
        "cfg_jira_id": "",
        "cfg_user_story_id": "",
        "cfg_jira_prefix": "",
        "cfg_start_node": "Requirement Structuring",
        "cfg_req_files": [],
        "cfg_context_files": [],
        "cfg_template_files": [],
        "cfg_dom_files": [],
        "cfg_dom_text": "",
        "cfg_extra_instructions": "",
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
# 13.0 — INPUT CONFIGURATION PAGE
# ============================================================================

# Agent pipeline metadata
_PIPELINE_NODES = [
    ("pn-req",  "📋", "Requirement\nAnalysis"),
    ("pn-scn",  "🎯", "Test\nScenario"),
    ("pn-tc",   "🧪", "Test\nCase"),
    ("pn-step", "🔢", "Step\nDefinition"),
    ("pn-xp",   "🔍", "XPath\nAgent"),
    ("pn-util", "🔧", "Utility\nMethod"),
    ("pn-ts",   "⚙️",  "Test\nScript"),
]

_START_NODES = [
    ("📋", "Requirement Structuring",  "Full pipeline — starts from raw requirements"),
    ("🎯", "Test Scenario Generation", "Skip req analysis; provide pre-structured requirements"),
    ("🌿", "Gherkin / BDD",            "Generate Gherkin feature files from existing scenarios"),
    ("🧪", "Test Case Generation",     "Jump directly to test case authoring"),
    ("⚙️", "Test Script Generation",   "Generate scripts from ready-made test cases"),
]


def _render_pipeline_visual():
    """Render the horizontal agent pipeline diagram."""
    nodes_html = ""
    for i, (cls, icon, label) in enumerate(_PIPELINE_NODES):
        lbl = label.replace("\n", "<br/>")
        nodes_html += (
            f'<div class="p-node">'
            f'  <div class="p-icon {cls}">{icon}</div>'
            f'  <div class="p-label">{lbl}</div>'
            f'</div>'
        )
        if i < len(_PIPELINE_NODES) - 1:
            nodes_html += '<div class="p-arrow">›</div>'

    st.markdown(
        f'<div class="pipeline-bar">{nodes_html}</div>',
        unsafe_allow_html=True,
    )


def _section(icon: str, title: str, subtitle: str = "", bg: str = "#eff6ff",
             icon_bg: str = "#dbeafe"):
    """Render a styled section header."""
    st.markdown(
        f'''<div class="cfg-section">
              <div class="cfg-section-icon" style="background:{icon_bg};">{icon}</div>
              <span class="cfg-section-title">{title}</span>
              <span class="cfg-section-subtitle">{subtitle}</span>
            </div>''',
        unsafe_allow_html=True,
    )


def _render_config_page():
    """Full-page input configuration form shown before the pipeline starts."""

    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown(
        '''<div class="cfg-hero">
             <h1>QA-GPT · Run Configuration 🧪</h1>
             <p>Configure your test generation run before launching the AI pipeline.</p>
           </div>''',
        unsafe_allow_html=True,
    )

    # ── Agent Pipeline Visual ─────────────────────────────────────────────
    _render_pipeline_visual()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1 — Testing Type & Project Mode
    # ══════════════════════════════════════════════════════════════════════
    _section("⚗️", "Testing Type & Project Mode",
             "Determines active agents and available inputs",
             bg="#ede9fe", icon_bg="#ddd6fe")

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        testing_type = st.selectbox(
            "Type of Testing",
            ["UI (Browser Automation)", "ETL (Data Pipeline)", "API (REST / SOAP / GraphQL)"],
            index=["UI", "ETL", "API"].index(st.session_state.cfg_testing_type)
                  if st.session_state.cfg_testing_type in ["UI", "ETL", "API"] else 0,
            help="Controls which agents activate and whether HTML DOM inputs are shown.",
        )
        st.session_state.cfg_testing_type = (
            "UI" if "UI" in testing_type else
            "ETL" if "ETL" in testing_type else "API"
        )

    with col2:
        project_mode = st.radio(
            "Project Mode",
            ["New Project", "Existing Project"],
            index=0 if st.session_state.cfg_project_mode == "New Project" else 1,
            horizontal=True,
            help="New project creates a fresh test suite; Existing project extends an existing one.",
        )
        st.session_state.cfg_project_mode = project_mode

    is_ui = st.session_state.cfg_testing_type == "UI"

    if is_ui:
        st.info("ℹ️ **UI mode active** — XPath Agent and HTML DOM inputs are enabled.")
    else:
        st.info(f"ℹ️ **{st.session_state.cfg_testing_type} mode active** — XPath / DOM inputs are disabled.")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2 — Git Repository
    # ══════════════════════════════════════════════════════════════════════
    _section("🔗", "Git Repository", "Source repository for generated test scripts",
             icon_bg="#dbeafe")

    c1, c2, c3 = st.columns([3, 1, 2], gap="medium")
    with c1:
        git_url = st.text_input(
            "Repository URL",
            value=st.session_state.cfg_git_url,
            placeholder="https://github.com/org/repo.git",
        )
        st.session_state.cfg_git_url = git_url
    with c2:
        git_branch = st.text_input(
            "Branch",
            value=st.session_state.cfg_git_branch,
            placeholder="main",
        )
        st.session_state.cfg_git_branch = git_branch
    with c3:
        git_pat = st.text_input(
            "Personal Access Token (PAT)",
            value=st.session_state.cfg_git_pat,
            type="password",
            placeholder="ghp_xxxxxxxxxxxx",
        )
        st.session_state.cfg_git_pat = git_pat

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 3 — JIRA Integration
    # ══════════════════════════════════════════════════════════════════════
    _section("📌", "JIRA Integration", "Link generated artifacts to JIRA issues",
             icon_bg="#fce7f3")

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        jira_id = st.text_input(
            "JIRA Project ID",
            value=st.session_state.cfg_jira_id,
            placeholder="PROJ",
        )
        st.session_state.cfg_jira_id = jira_id
    with c2:
        user_story_id = st.text_input(
            "User Story JIRA ID",
            value=st.session_state.cfg_user_story_id,
            placeholder="PROJ-456",
        )
        st.session_state.cfg_user_story_id = user_story_id
    with c3:
        jira_prefix = st.text_input(
            "Issue ID Prefix",
            value=st.session_state.cfg_jira_prefix,
            placeholder="PROJ-",
            help="Prefix used when auto-generating child issue IDs.",
        )
        st.session_state.cfg_jira_prefix = jira_prefix

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 4 — Input Files
    # ══════════════════════════════════════════════════════════════════════
    _section("📂", "Input Files", "Documents and context for the AI agents",
             icon_bg="#d1fae5")

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown("**📄 Requirements Files**")
        st.caption("BRD, FRD, user stories, acceptance criteria…")
        req_files = st.file_uploader(
            "Upload requirement documents",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt", "md", "xlsx", "csv"],
            key="uploader_req",
            label_visibility="collapsed",
        )
        if req_files:
            st.session_state.cfg_req_files = req_files
            st.success(f"✅ {len(req_files)} requirement file(s) uploaded")

        st.markdown("**🗂️ Context Files**", help=None)
        st.caption("Architecture docs, API specs, data dictionaries…")
        context_files = st.file_uploader(
            "Upload context / reference documents",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt", "md", "json", "yaml", "yml"],
            key="uploader_context",
            label_visibility="collapsed",
        )
        if context_files:
            st.session_state.cfg_context_files = context_files
            st.success(f"✅ {len(context_files)} context file(s) uploaded")

    with col_right:
        st.markdown("**📐 Templates / Examples**")
        st.caption("Existing test scripts, Gherkin templates, coding standards…")
        template_files = st.file_uploader(
            "Upload template or example files",
            accept_multiple_files=True,
            type=["py", "java", "feature", "txt", "md", "json"],
            key="uploader_templates",
            label_visibility="collapsed",
        )
        if template_files:
            st.session_state.cfg_template_files = template_files
            st.success(f"✅ {len(template_files)} template(s) uploaded")

        # HTML DOM — only active in UI mode
        dom_label = "🌐 HTML DOM" + (" ✅" if is_ui else " 🔒 Disabled — UI mode only")
        st.markdown(f"**{dom_label}**")
        st.caption("Page source HTML for the XPath Agent to derive locators.")

        if is_ui:
            dom_files = st.file_uploader(
                "Upload HTML / page source files",
                accept_multiple_files=True,
                type=["html", "htm", "xml", "txt"],
                key="uploader_dom",
                label_visibility="collapsed",
            )
            if dom_files:
                st.session_state.cfg_dom_files = dom_files
                st.success(f"✅ {len(dom_files)} DOM file(s) uploaded")

            dom_text = st.text_area(
                "Or paste HTML snippet directly",
                value=st.session_state.cfg_dom_text,
                height=90,
                placeholder="<html>\n  <body>…</body>\n</html>",
                key="dom_paste",
            )
            st.session_state.cfg_dom_text = dom_text
        else:
            st.markdown(
                '<div class="dom-disabled">🔒 Enable <strong>UI testing</strong> above to unlock HTML DOM inputs.</div>',
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 5 — Pipeline Start Node
    # ══════════════════════════════════════════════════════════════════════
    _section("🚀", "Pipeline Start Node",
             "Choose where the agent pipeline begins execution",
             icon_bg="#fef3c7")

    node_options  = [f"{ic}  {name}" for ic, name, _ in _START_NODES]
    node_captions = [desc for _, _, desc in _START_NODES]

    current_node_idx = next(
        (i for i, (_, name, _) in enumerate(_START_NODES)
         if name == st.session_state.cfg_start_node),
        0,
    )

    selected_node = st.radio(
        "Start at",
        node_options,
        captions=node_captions,
        index=current_node_idx,
        label_visibility="collapsed",
    )
    st.session_state.cfg_start_node = selected_node.split("  ", 1)[1]

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 6 — Additional Instructions (collapsed by default)
    # ══════════════════════════════════════════════════════════════════════
    with st.expander("💡 Additional Instructions (optional)"):
        extra = st.text_area(
            "Extra context or guidance for the agents",
            value=st.session_state.cfg_extra_instructions,
            height=100,
            placeholder=(
                "e.g. 'Use Page Object Model', "
                "'Focus on edge cases', "
                "'Target Chrome and Firefox'…"
            ),
            key="extra_instructions",
        )
        st.session_state.cfg_extra_instructions = extra

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY + LAUNCH
    # ══════════════════════════════════════════════════════════════════════
    st.divider()

    # Build summary chips
    chips = [
        f'<span class="chip">{st.session_state.cfg_testing_type}</span>',
        f'<span class="chip gray">{st.session_state.cfg_project_mode}</span>',
    ]
    if st.session_state.cfg_git_url:
        chips.append('<span class="chip green">Git ✓</span>')
    if st.session_state.cfg_jira_id:
        chips.append(f'<span class="chip green">JIRA {st.session_state.cfg_jira_id}</span>')
    if st.session_state.cfg_user_story_id:
        chips.append(f'<span class="chip">Story {st.session_state.cfg_user_story_id}</span>')
    if req_files:
        chips.append(f'<span class="chip amber">{len(req_files)} Req File(s)</span>')
    if context_files:
        chips.append(f'<span class="chip amber">{len(context_files)} Context File(s)</span>')
    if template_files:
        chips.append(f'<span class="chip amber">{len(template_files)} Template(s)</span>')
    if is_ui and (st.session_state.cfg_dom_files or st.session_state.cfg_dom_text):
        chips.append('<span class="chip green">DOM ✓</span>')
    chips.append(f'<span class="chip">▶ {st.session_state.cfg_start_node}</span>')

    st.markdown(
        f'<div class="cfg-summary">'
        f'  <div class="cfg-summary-title">✅ Run Summary</div>'
        f'  <div class="chip-row">{"".join(chips)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Validation
    errors = []
    if not st.session_state.cfg_git_url:
        errors.append("Git Repository URL is required.")
    if not st.session_state.cfg_req_files and not st.session_state.cfg_extra_instructions:
        errors.append("Upload at least one Requirements file, or provide Additional Instructions.")

    if errors:
        for e in errors:
            st.warning(f"⚠️ {e}")

    col_launch, col_reset = st.columns([3, 1], gap="medium")
    with col_launch:
        launch_disabled = bool(errors)
        if st.button(
            "🚀  Launch Pipeline",
            type="primary",
            use_container_width=True,
            disabled=launch_disabled,
            help="Fix the warnings above before launching." if launch_disabled else "",
        ):
            st.session_state.config_complete = True
            st.rerun()

    with col_reset:
        if st.button("↺  Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ============================================================================
# 12.2 — SIDEBAR
# ============================================================================

def _render_sidebar():
    """Render the sidebar with provider info, connection test, uploaders, and session info."""
    st.title("QA-GPT 🧪")
    st.caption("AI-Powered QA Workflow")

    st.divider()

    # ── Config summary (shown after config is complete) ───────────────────
    if st.session_state.config_complete:
        st.markdown("**⚙️ Run Configuration**")

        cfg_type  = st.session_state.cfg_testing_type
        cfg_mode  = st.session_state.cfg_project_mode
        cfg_start = st.session_state.cfg_start_node
        cfg_jira  = st.session_state.cfg_jira_id or "—"
        cfg_story = st.session_state.cfg_user_story_id or "—"
        cfg_url   = st.session_state.cfg_git_url or "—"
        cfg_branch= st.session_state.cfg_git_branch or "—"

        st.caption(f"🔬 **Type:** {cfg_type}")
        st.caption(f"📁 **Mode:** {cfg_mode}")
        st.caption(f"▶ **Start node:** {cfg_start}")
        st.caption(f"🔗 **Repo:** `{cfg_url[:30]}…`" if len(cfg_url) > 30 else f"🔗 **Repo:** `{cfg_url}`")
        st.caption(f"🌿 **Branch:** `{cfg_branch}`")
        st.caption(f"📌 **JIRA:** {cfg_jira}  |  Story: {cfg_story}")

        file_counts = []
        if st.session_state.cfg_req_files:
            file_counts.append(f"{len(st.session_state.cfg_req_files)} req")
        if st.session_state.cfg_context_files:
            file_counts.append(f"{len(st.session_state.cfg_context_files)} ctx")
        if st.session_state.cfg_template_files:
            file_counts.append(f"{len(st.session_state.cfg_template_files)} tpl")
        if st.session_state.cfg_dom_files:
            file_counts.append(f"{len(st.session_state.cfg_dom_files)} dom")
        st.caption(f"📂 **Files:** {', '.join(file_counts) if file_counts else 'None'}")

        if st.button("← Edit Configuration", use_container_width=True, type="secondary"):
            st.session_state.config_complete = False
            st.rerun()

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

    # Context file uploaders
    st.subheader("📂 Context Files")

    tech_files = st.file_uploader(
        "Upload tech_context.md (one or more)",
        type=["md", "txt"],
        key="tech_upload",
        accept_multiple_files=True,
    )
    if tech_files:
        # Write to temp files and update paths
        paths = []
        for i, file in enumerate(tech_files):
            tmp_path = os.path.join(tempfile.gettempdir(), f"tech_context_{i}_{file.name}")
            Path(tmp_path).write_bytes(file.getvalue())
            paths.append(tmp_path)
            
        st.session_state.tech_context_path = paths
        total_bytes = sum(len(f.getvalue()) for f in tech_files)
        st.success(f"✅ Loaded {len(paths)} file(s) ({total_bytes} bytes)")

    codebase_files = st.file_uploader(
        "Upload codebase_map.md (one or more)",
        type=["md", "txt"],
        key="codebase_upload",
        accept_multiple_files=True,
    )
    if codebase_files:
        paths = []
        for i, file in enumerate(codebase_files):
            tmp_path = os.path.join(tempfile.gettempdir(), f"codebase_{i}_{file.name}")
            Path(tmp_path).write_bytes(file.getvalue())
            paths.append(tmp_path)
            
        st.session_state.codebase_map_path = paths
        total_bytes = sum(len(f.getvalue()) for f in codebase_files)
        st.success(f"✅ Loaded {len(paths)} file(s) ({total_bytes} bytes)")

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

    # New session button
    if st.button("🔄 New Session", use_container_width=True, type="secondary"):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ============================================================================
# 12.3 — PIPELINE PROGRESS BAR
# ============================================================================

def _render_progress_bar(current_stage):
    """Render the pipeline progress bar with 7 milestones."""
    if current_stage is None:
        current_milestone = -1
    elif isinstance(current_stage, WorkflowStage):
        current_milestone = MILESTONE_MAP.get(current_stage, -1)
    else:
        current_milestone = -1

    # Progress value
    progress_value = min((current_milestone + 1) / len(MILESTONE_LABELS), 1.0) if current_milestone >= 0 else 0.0
    st.progress(progress_value)

    # Milestone labels
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
# 12.4 — ARTIFACT TABS (Right Column)
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
    """Render the 5 artifact tabs in the right column."""
    if state is None:
        state = {}

    tabs = st.tabs(["📋 Spec", "🗺️ Strategy", "🧩 Test Cases", "📐 Code Plan", "💻 Script"])

    # Tab 1: Requirements Spec
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

    # Tab 2: Strategy
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

    # Tab 3: Test Cases (Gherkin)
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

    # Tab 4: Code Plan
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

    # Tab 5: Script
    with tabs[4]:
        version = state.get("current_script_version", 0)
        if version > 0:
            st.caption(f"Version: {version}")
        _render_judge_badge(state, "judge_code_evaluation")
        content = state.get("script_content", "")
        if content:
            st.code(content, language="python")
            # Download button
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
# 12.5 — Q&A FORM (Left Column)
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
            # Handle questions as strings or dicts
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
            # Update the latest session with answers
            latest_session.answers = answers
            latest_session.status = "answered"

            # Update state and resume graph
            graph = _get_graph()
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            try:
                graph.update_state(config, {
                    "qa_sessions": qa_sessions,
                })
                st.session_state.awaiting_qa = False
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"Submitted {len(answers)} answers to clarifying questions.",
                })
                # Resume pipeline
                _resume_pipeline()
            except Exception as e:
                st.error(f"Failed to submit answers: {e}")


# ============================================================================
# 12.6 — HUMAN REVIEW GATE UI (Left Column)
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

    # Show judge info if available
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

    # Decision form
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

        # Always-visible guidance for the next stage
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
                    "content": f"{decision_emoji} {decision} — {label}. {feedback}" if feedback else f"{decision_emoji} {decision} — {label}",
                })

                st.toast(f"{decision_emoji} {label}: {decision}!")

                # Resume pipeline with Command(resume=...) to pass response to interrupt()
                _resume_pipeline(resume_value=human_response)
            except Exception as e:
                st.error(f"Failed to submit review: {e}")


# ============================================================================
# 12.7 — GRAPH STREAMING LOOP
# ============================================================================

def _run_pipeline(user_input: str):
    """Start a new pipeline execution from user input."""
    graph = _get_graph()

    # Fetch context
    team_context = fetch_context(
        team_id=settings.team_id,
        tech_context_path=st.session_state.tech_context_path,
        codebase_map_path=st.session_state.codebase_map_path,
    )

    # Create initial state
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

    # Stream the graph
    _stream_graph(graph, initial_state, config)


def _resume_pipeline(resume_value=None):
    """Resume the pipeline after human input (Q&A or review gate)."""
    graph = _get_graph()
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    if resume_value is not None:
        # Resume from interrupt() — pass Command(resume=...) so the interrupted
        # node receives the human response as the return value of interrupt()
        _stream_graph(graph, Command(resume=resume_value), config)
    else:
        # Resume without interrupt response (e.g., after Q&A answers)
        _stream_graph(graph, None, config)


def _stream_graph(graph, input_state, config):
    """Stream graph execution, handling interrupts and state updates."""
    try:
        with st.spinner("🤖 AI is working..."):
            final_state = None
            for state_snapshot in graph.stream(input_state, config=config, stream_mode="values"):
                final_state = state_snapshot
                st.session_state.graph_state = state_snapshot

            # Check what happened after stream ended
            if final_state is not None:
                current_stage = final_state.get("current_stage")
                workflow_status = final_state.get("workflow_status")

                # Check if we're awaiting Q&A (qa_completed=False, still at QA stage)
                if (current_stage == WorkflowStage.QA_INTERACTION
                        and final_state.get("qa_completed") is False):
                    st.session_state.awaiting_qa = True
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "🔍 I need some clarifying information. Please answer the questions below.",
                    })
                    st.rerun()
                    return

                # Check if we hit a human review gate (interrupt)
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

                # Check for completion
                if current_stage == WorkflowStage.COMPLETED or workflow_status == WorkflowStatus.COMPLETED:
                    st.session_state.workflow_running = False
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "✅ **Pipeline Complete!** Your test script has been generated. Check the Script tab to download it.",
                    })
                    st.toast("🎉 Pipeline complete!")
                    st.rerun()
                    return

                # Check for failure
                if current_stage == WorkflowStage.FAILED or workflow_status == WorkflowStatus.FAILED:
                    error_msg = final_state.get("error_message", "Unknown error")
                    st.session_state.workflow_running = False
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ **Pipeline Failed:** {error_msg}",
                    })
                    st.rerun()
                    return

                # Stream ended normally but not at a known stopping point
                # This might happen with LangGraph interrupts
                # Check graph state for pending interrupts
                try:
                    graph_state = graph.get_state(config)
                    if graph_state and hasattr(graph_state, 'next') and graph_state.next:
                        # There's a next node pending — likely an interrupt
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
                    pass  # If we can't check graph state, continue normally

                st.rerun()

    except Exception as e:
        error_str = str(e)
        error_type = type(e).__name__

        # Handle GraphInterrupt (expected at human review gates)
        if "GraphInterrupt" in error_type or "interrupt" in error_str.lower():
            # Try to detect which gate we're at
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

        # Genuine error
        st.session_state.workflow_running = False
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ **Error:** {error_str}",
        })
        st.error(f"Pipeline error: {error_str}")
        st.rerun()


# ============================================================================
# 12.7 — CHAT INTERFACE
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

    # ── Route: Input Configuration page ───────────────────────────────────
    # Show the config form until the user clicks "Launch Pipeline".
    if not st.session_state.config_complete:
        _render_config_page()
        return

    # ── Route: Pipeline (existing chat + artifact workflow) ───────────────

    # --- Sidebar ---
    with st.sidebar:
        _render_sidebar()

    # --- Progress Bar (full width) ---
    state = st.session_state.graph_state
    current_stage = state.get("current_stage") if state else None
    _render_progress_bar(current_stage)

    st.divider()

    # --- Two-column layout ---
    col1, col2 = st.columns([2, 3], gap="large")

    # --- Left Column: Chat / Q&A / Review ---
    with col1:
        # Show the appropriate interface
        if st.session_state.awaiting_qa:
            _render_qa_form()
        elif st.session_state.awaiting_review:
            _render_human_review_gate()
        else:
            # Normal chat interface
            _render_chat_history()

            # Show error state prominently
            if state and state.get("workflow_status") == WorkflowStatus.FAILED:
                st.error(f"❌ Workflow failed: {state.get('error_message', 'Unknown error')}")

            # Chat input — pre-populate hint with config context
            testing_type = st.session_state.cfg_testing_type
            start_node   = st.session_state.cfg_start_node
            placeholder  = (
                f"Describe the {testing_type} feature you want to test "
                f"(pipeline starts at: {start_node})…"
            )

            user_input = st.chat_input(
                placeholder,
                disabled=st.session_state.workflow_running,
            )

            if user_input:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": (
                        f"🚀 Starting QA-GPT pipeline…\n\n"
                        f"**Configuration:** {testing_type} testing · "
                        f"{st.session_state.cfg_project_mode} · "
                        f"Start node: *{start_node}*"
                        + (f" · JIRA: {st.session_state.cfg_jira_id}" if st.session_state.cfg_jira_id else "")
                    ),
                })

                # Run pipeline
                _run_pipeline(user_input)

    # --- Right Column: Artifact Tabs ---
    with col2:
        _render_artifact_tabs(state)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
