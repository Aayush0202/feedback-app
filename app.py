import hmac
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from llm import analyse_comment, get_client
import rules

# Page configuration
st.set_page_config(
    page_title="Anonymous Employee Feedback & Risk Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mandatory banner across all pages
BANNER_HTML = """
<div style="background: linear-gradient(90deg, #fff3cd 0%, #ffeeba 100%); color: #856404; padding: 12px 18px; border-radius: 8px; margin-bottom: 24px; font-weight: 600; border-left: 6px solid #e0a800; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
    ⚠️ Educational prototype using synthetic data. Not legal or HR advice.
</div>
"""

def _train_fallback_model():
    """Retrains the Random Forest from data/teams.csv if model.pkl cannot be
    loaded (e.g. the online server has a different scikit-learn version).
    Same settings as train_model.py: team_size >= 5, train on quarters 1-6,
    balanced Random Forest, random_state=42, tuned threshold 0.38."""
    from sklearn.ensemble import RandomForestClassifier

    teams_path = Path("data") / "teams.csv"
    if not teams_path.exists():
        return None
    df = pd.read_csv(teams_path)
    df = df[df["team_size"] >= rules.MIN_GROUP_SIZE]
    train = df[df["quarter"] <= 6]
    encoded = pd.get_dummies(train, columns=["department", "location"], drop_first=False, dtype=int)
    drop_cols = [c for c in ["team_id", "next_q_attrition", "high_risk"] if c in encoded.columns]
    X = encoded.drop(columns=drop_cols)
    y = encoded["high_risk"]
    model = RandomForestClassifier(class_weight="balanced", random_state=42)
    model.fit(X, y)
    return {"model": model, "features": list(X.columns), "threshold": 0.38}


@st.cache_resource
def load_ml_model():
    """Loads the trained Random Forest model and threshold bundle.
    Falls back to retraining from teams.csv if model.pkl is missing or
    incompatible with the installed scikit-learn version."""
    model_path = Path("model.pkl")
    if model_path.exists():
        try:
            with open(model_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    try:
        return _train_fallback_model()
    except Exception as e:
        st.error(f"Could not load or retrain the attrition model: {e}")
        return None

@st.cache_data
def load_base_data():
    """Loads teams and the pre-processed comments shown on the HR pages.
    results/dashboard_comments.csv holds ONLY anonymised text and predicted
    labels (no original text), plus a team_id for each comment."""
    teams_path = Path("data") / "teams.csv"
    dash_path = Path("results") / "dashboard_comments.csv"

    df_teams = pd.read_csv(teams_path) if teams_path.exists() else pd.DataFrame()
    df_eval = pd.read_csv(dash_path) if dash_path.exists() else pd.DataFrame()

    if not df_eval.empty and not df_teams.empty and "team_id" in df_eval.columns:
        teams_unique = df_teams.drop_duplicates(subset=["team_id"]).set_index("team_id")
        df_eval["department"] = df_eval["team_id"].map(teams_unique["department"]).fillna("Unknown")
        df_eval["location"] = df_eval["team_id"].map(teams_unique["location"]).fillna("Unknown")

    return df_teams, df_eval

# Initialize session state collections
if "submitted_comments" not in st.session_state:
    st.session_state["submitted_comments"] = []

if "pending_submission" not in st.session_state:
    st.session_state["pending_submission"] = None

if "escalation_decisions" not in st.session_state:
    st.session_state["escalation_decisions"] = {}

if "demo_text_input" not in st.session_state:
    st.session_state["demo_text_input"] = ""

# Authentication session state
if "is_hr_authenticated" not in st.session_state:
    st.session_state["is_hr_authenticated"] = False

if "login_attempts" not in st.session_state:
    st.session_state["login_attempts"] = 0

if "login_blocked" not in st.session_state:
    st.session_state["login_blocked"] = False

# Sidebar navigation
st.sidebar.title("🛡️ Feedback Portal")
st.sidebar.markdown("**Role & Privacy Controlled Navigation**")
page = st.sidebar.radio(
    "Go to",
    ["1. Submit Feedback", "2. HR Dashboard", "3. Escalation Queue", "4. Security Log"]
)

st.sidebar.markdown("---")
st.sidebar.caption("🔒 Privacy Rule: Groups < 5 masked (k-anonymity).")
st.sidebar.caption("🔒 Access Rule: Escalations isolated from managers.")

# Sidebar Logout button if authenticated
if st.session_state["is_hr_authenticated"]:
    st.sidebar.markdown("---")
    st.sidebar.success("🔑 Authenticated as HR")
    if st.sidebar.button("Log out", key="hr_logout_btn"):
        st.session_state["is_hr_authenticated"] = False
        st.session_state["login_attempts"] = 0
        st.rerun()

# Render top banner on every page
st.markdown(BANNER_HTML, unsafe_allow_html=True)

def check_hr_authentication():
    """Renders the HR passcode login barrier for protected pages."""
    if st.session_state["is_hr_authenticated"]:
        return True

    st.markdown("### 🔒 HR Portal Access Required")
    st.caption("Prototype access control. A real deployment would use company single sign-on with role-based access.")

    if st.session_state["login_blocked"]:
        st.error("🚫 Access blocked: Too many incorrect attempts (3/3). This session is locked.")
        st.info("To protect sensitive feedback, access has been disabled for this session.")
        st.stop()

    if st.session_state["login_attempts"] > 0:
        remaining = 3 - st.session_state["login_attempts"]
        st.warning(f"⚠️ {remaining} attempt(s) remaining before session lockout.")

    with st.form("hr_login_form"):
        passcode_input = st.text_input(
            "Enter HR Passcode:",
            type="password",
            placeholder="Enter passcode...",
            help="Passcode configured in secrets.toml"
        )
        submit_btn = st.form_submit_button("Unlock HR Portal", type="primary")

    if submit_btn:
        if "HR_PASSCODE" not in st.secrets:
            st.error("HR_PASSCODE is not configured in .streamlit/secrets.toml.")
            st.stop()

        actual_passcode = st.secrets["HR_PASSCODE"]

        # Secure constant-time comparison without hardcoding or printing the passcode
        is_valid = hmac.compare_digest(
            passcode_input.strip().encode("utf-8"),
            str(actual_passcode).strip().encode("utf-8")
        )

        if is_valid:
            st.session_state["is_hr_authenticated"] = True
            st.session_state["login_attempts"] = 0
            st.success("Authentication successful! Redirecting...")
            st.rerun()
        else:
            st.session_state["login_attempts"] += 1
            if st.session_state["login_attempts"] >= 3:
                st.session_state["login_blocked"] = True
                st.rerun()
            else:
                remaining = 3 - st.session_state["login_attempts"]
                st.error(f"Incorrect passcode. {remaining} attempt(s) remaining before session lockout.")

    st.stop()

df_teams, df_eval = load_base_data()
model_bundle = load_ml_model()

# Gate protected HR pages
if page in ["2. HR Dashboard", "3. Escalation Queue", "4. Security Log"]:
    check_hr_authentication()


# ==============================================================================
# PAGE 1: SUBMIT FEEDBACK (EMPLOYEE VIEW)
# ==============================================================================
if page == "1. Submit Feedback":
    st.title("📝 Submit Anonymous Feedback")
    st.markdown(
        "Your voice matters. To protect your identity, comments are automatically pre-redacted "
        "and anonymised by AI before submission. **You review the anonymised version before final submission.**"
    )

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        dept = st.selectbox("Your Department", ["Engineering", "Sales", "Operations", "HR", "Finance", "Support"])
    with col_meta2:
        loc = st.selectbox("Your Location", ["Mumbai", "Pune", "Bengaluru", "Delhi", "Hyderabad"])

    st.markdown("#### Need inspiration? Try a demo comment:")
    demo_c1, demo_c2, demo_c3, demo_c4 = st.columns(4)

    if demo_c1.button("📋 1. Routine Comment"):
        st.session_state["demo_text_input"] = (
            "The current sprint timelines are unrealistic and leading to weekend burnouts. "
            "We need better buffer planning for production release cycles."
        )
        st.session_state["pending_submission"] = None

    if demo_c2.button("👤 2. Comment with Names"):
        st.session_state["demo_text_input"] = (
            "Rajesh Kumar on the 5th floor Bangalore office rejects leave requests without any valid explanation. "
            "He expects everyone to be on call post 8 PM."
        )
        st.session_state["pending_submission"] = None

    if demo_c3.button("🚨 3. Harassment Comment"):
        st.session_state["demo_text_input"] = (
            "A senior executive has been sending inappropriate personal messages on WhatsApp late at night "
            "and threatening negative appraisals if I do not comply."
        )
        st.session_state["pending_submission"] = None

    if demo_c4.button("⚡ 4. Night Shift Edge Case"):
        st.session_state["demo_text_input"] = (
            "As the only woman on the Pune QA night shift, my cab requests are constantly delayed, "
            "leaving me stranded outside the campus for hours."
        )
        st.session_state["pending_submission"] = None

    comment_text = st.text_area(
        "Enter your feedback (can include Hindi / Hinglish):",
        value=st.session_state["demo_text_input"],
        height=140,
        placeholder="Type your feedback here..."
    )

    if st.button("🔒 Step 1: Anonymise My Feedback", type="primary"):
        if not comment_text.strip():
            st.warning("Please enter your comment before anonymising.")
        else:
            with st.spinner("Anonymising and analyzing feedback safely..."):
                try:
                    analysis = analyse_comment(comment_text)
                    # PRIVACY: removed_details contains real names — never saved.
                    # Only anonymised text and classification labels are retained.
                    st.session_state["pending_submission"] = {
                        "department": dept,
                        "location": loc,
                        "anonymised_text": analysis["anonymised"],
                        "theme": analysis["theme"],
                        "sentiment": analysis["sentiment"],
                        "severity": analysis["severity"],
                        "escalate": analysis["escalate"],
                        "security_flag": analysis["security_flag"],
                        "still_identifiable": analysis.get("still_identifiable", False),
                        "injection_attempt": analysis.get("injection_attempt", False),
                    }
                except Exception as e:
                    st.error(f"The AI service is temporarily unavailable ({e}). Please try again in a few moments.")

    # Human Checkpoint: Employee reviews anonymised version
    if st.session_state["pending_submission"]:
        pending = st.session_state["pending_submission"]
        st.markdown("---")
        st.subheader("🔍 Step 2: Review Anonymised Version (Human Checkpoint)")
        st.info("The original text will **never** be saved or shared with HR or your manager.")

        st.markdown(f"**Anonymised Text to be Submitted:**")
        st.success(f"\"{pending['anonymised_text']}\"")

        # Labels preview
        lbl_col1, lbl_col2, lbl_col3, lbl_col4 = st.columns(4)
        lbl_col1.metric("Classified Theme", pending["theme"].capitalize())
        lbl_col2.metric("Sentiment", pending["sentiment"].capitalize())
        lbl_col3.metric("Severity", pending["severity"].capitalize())
        lbl_col4.metric("Escalation Flag", "🚨 Yes" if pending["escalate"] else "✅ No")

        # removed_details is intentionally NOT displayed — it contains the actual
        # identifying phrases (real names etc.) and must never be shown anywhere.


        if pending["security_flag"]:
            st.warning("\u26a0\ufe0f System Note: Security instruction flags detected. Review will be audited.")

        # ── Uniqueness / re-identification warning ─────────────────────────────
        if pending.get("still_identifiable"):
            st.warning(
                "\u26a0\ufe0f **Your comment may still identify you** \u2014 it contains a phrase like "
                "*'the only person'*, *'sole'*, or similar wording that could make you "
                "recognisable even after anonymisation.  \n"
                "Please **edit your comment** to remove this detail, or submit anyway "
                "if you are comfortable doing so."
            )
            edit_col, submit_col = st.columns([1, 2])
            do_edit   = edit_col.button("\u270f\ufe0f Edit my comment",
                                        key="edit_comment_btn",
                                        help="Go back and rewrite your comment")
            do_submit = submit_col.button("\u2705 Step 3: Confirm & Submit Anyway",
                                          type="primary",
                                          key="submit_anyway_btn")
        else:
            do_edit   = False
            do_submit = st.button("\u2705 Step 3: Confirm & Submit Anonymised Version",
                                  type="primary", key="submit_normal_btn")

        if do_edit:
            st.session_state["pending_submission"] = None
            st.rerun()

        if do_submit:
            new_id = f"NEW_{len(st.session_state['submitted_comments']) + 1:03d}"
            submitted_record = {
                "comment_id": new_id,
                "department": pending["department"],
                "location": pending["location"],
                "anonymised_text": pending["anonymised_text"],
                "theme": pending["theme"],
                "sentiment": pending["sentiment"],
                "severity": pending["severity"],
                "escalate": pending["escalate"],
                "security_flag": pending["security_flag"],
                "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state["submitted_comments"].append(submitted_record)
            st.session_state["pending_submission"] = None
            st.session_state["demo_text_input"] = ""
            st.balloons()
            st.success(f"\U0001f389 Feedback successfully submitted under anonymous ID **{new_id}**! Only the anonymised version was stored.")


# ==============================================================================
# PAGE 2: HR DASHBOARD
# ==============================================================================
elif page == "2. HR Dashboard":
    st.title("📊 HR Analytics & Department Health Dashboard")
    st.markdown("Aggregated feedback metrics and team attrition risk forecasts. **Privacy minimum of 5 individuals enforced.**")

    # Combine historical eval comments + newly submitted comments
    combined_records = []
    if not df_eval.empty:
        for _, row in df_eval.iterrows():
            combined_records.append({
                "comment_id": row["comment_id"],
                "department": row.get("department", "Engineering"),
                "location": row.get("location", "Bengaluru"),
                "anonymised_text": row["anonymised_text"],
                "theme": row["pred_theme"],
                "sentiment": row["pred_sentiment"],
                "severity": row["pred_severity"],
                "escalate": row["pred_escalate"],
                "security_flag": row.get("pred_injection_attempt", False),
            })

    for sc in st.session_state["submitted_comments"]:
        combined_records.append({
            "comment_id": sc["comment_id"],
            "department": sc["department"],
            "location": sc["location"],
            "anonymised_text": sc["anonymised_text"],
            "theme": sc["theme"],
            "sentiment": sc["sentiment"],
            "severity": sc["severity"],
            "escalate": sc["escalate"],
            "security_flag": sc.get("security_flag", False),
        })

    df_active_comments = pd.DataFrame(combined_records)

    st.subheader("1. Department Feedback Breakdown (k-Anonymity Guard)")
    if not df_active_comments.empty:
        dept_counts = df_active_comments["department"].value_counts()
        masked_counts = rules.mask_small_department_aggregates(dept_counts, min_count=rules.MIN_GROUP_SIZE)

        metrics_cols = st.columns(len(masked_counts))
        for col, (dept_name, display_count) in zip(metrics_cols, masked_counts.items()):
            col.metric(label=dept_name, value=display_count)

        # Filter comments for departments that meet privacy threshold
        valid_comments_df = rules.filter_comments_by_group_size(df_active_comments, group_col="department", min_count=rules.MIN_GROUP_SIZE)

        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            st.markdown("**Theme Distribution by Department (Count >= 5)**")
            if not valid_comments_df.empty:
                theme_pivot = pd.crosstab(valid_comments_df["department"], valid_comments_df["theme"])
                st.bar_chart(theme_pivot)
        with c_chart2:
            st.markdown("**Sentiment Distribution by Department (Count >= 5)**")
            if not valid_comments_df.empty:
                sentiment_pivot = pd.crosstab(valid_comments_df["department"], valid_comments_df["sentiment"])
                st.bar_chart(sentiment_pivot)
    else:
        st.info("No feedback comments available yet.")

    st.markdown("---")
    st.subheader("2. Team Attrition Risk Forecast (Quarter 8)")
    st.caption("Machine learning forecast using Random Forest (`model.pkl`). Teams with fewer than 5 members are hidden for privacy.")

    high_risk_departments = []
    if model_bundle and not df_teams.empty:
        rf_model = model_bundle["model"]
        features = model_bundle["features"]
        threshold = model_bundle.get("threshold", 0.38)

        # Enforce Rule 1: Filter teams with team_size >= 5
        q8_teams = df_teams[(df_teams["quarter"] == 8)].copy()
        q8_safe_teams = rules.filter_teams_by_size(q8_teams, min_size=rules.MIN_GROUP_SIZE)

        hidden_team_count = len(q8_teams) - len(q8_safe_teams)
        if hidden_team_count > 0:
            st.caption(f"🔒 Hiding {hidden_team_count} teams with `team_size < 5` to protect employee anonymity.")

        # One-hot encode
        q8_encoded = pd.get_dummies(q8_safe_teams, columns=["department", "location"], drop_first=False, dtype=int)
        X_q8 = q8_encoded.reindex(columns=features, fill_value=0)

        probs = rf_model.predict_proba(X_q8)[:, 1]
        q8_safe_teams["Risk Probability"] = [round(float(p), 3) for p in probs]
        q8_safe_teams["Predicted High Risk"] = (probs >= threshold).astype(int)

        high_risk_teams = q8_safe_teams[q8_safe_teams["Predicted High Risk"] == 1].sort_values(by="Risk Probability", ascending=False)
        high_risk_departments = high_risk_teams["department"].unique().tolist()

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Evaluated Q8 Teams", len(q8_safe_teams))
        m_col2.metric("High-Risk Teams (Prob >= 38%)", len(high_risk_teams))
        m_col3.metric("Model Decision Threshold", f"{threshold:.2f}")

        # Display high risk teams table
        st.dataframe(
            high_risk_teams[[
                "team_id", "department", "location", "team_size",
                "Risk Probability", "avg_sentiment", "avg_overtime_hours",
                "pct_negative_manager", "pct_negative_pay"
            ]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("Model bundle or teams data not loaded.")

    st.markdown("---")
    st.subheader("3. AI-Powered HR Department Summary")
    st.markdown("Synthesize anonymous feedback for departments at high risk of attrition without revealing individual identities.")

    target_departments = high_risk_departments if high_risk_departments else ["Operations", "Finance", "Support", "Engineering"]
    selected_dept = st.selectbox("Select Department to Summarize:", target_departments)

    dept_comments = df_active_comments[df_active_comments["department"] == selected_dept]
    st.write(f"Available anonymised comments for **{selected_dept}**: `{len(dept_comments)}`")

    if not rules.is_group_safe(len(dept_comments), min_size=rules.MIN_GROUP_SIZE):
        st.warning(f"🔒 {rules.HIDDEN_MESSAGE} ({len(dept_comments)} comments found, minimum required is {rules.MIN_GROUP_SIZE}).")
    else:
        if st.button(f"✨ Generate HR Summary for {selected_dept}", type="primary"):
            with st.spinner("Analyzing anonymised comments with Gemini..."):
                comments_context = "\n".join([
                    f"- [{row['comment_id']}] (Theme: {row['theme']}, Severity: {row['severity']}): {row['anonymised_text']}"
                    for _, row in dept_comments.iterrows()
                ])

                prompt = (
                    f"You are an executive HR analytics advisor. Below are anonymised employee feedback comments "
                    f"from the '{selected_dept}' department.\n\n"
                    f"STRICT RULES:\n"
                    f"1. Summarise the main root-cause themes ONLY from these anonymised comments.\n"
                    f"2. You MUST cite specific comment IDs (e.g. C018, C038) for every key finding.\n"
                    f"3. Do NOT mention any individual names, and do NOT speculate on individual identities.\n"
                    f"4. Provide 3 concrete, constructive recommendations for department leadership.\n\n"
                    f"ANONYMISED COMMENTS:\n{comments_context}"
                )

                try:
                    client = get_client()
                    response = client.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=prompt
                    )
                    st.markdown("### 📋 Executive HR Summary")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Unable to generate summary at this time ({e}). Please try again shortly.")

# ==============================================================================
# PAGE 3: ESCALATION QUEUE (HR REVIEWER ONLY)
# ==============================================================================
elif page == "3. Escalation Queue":
    st.title("🚨 HR Reviewer Escalation Queue")
    st.error("🔒 **CONFIDENTIAL ACCESS ONLY:** This queue is strictly restricted to HR Reviewers and is **NEVER shared with the employee's manager**.")

    # Gather all escalations
    escalation_items = []
    if not df_eval.empty:
        for _, row in df_eval[df_eval["pred_escalate"] == True].iterrows():
            escalation_items.append({
                "comment_id": row["comment_id"],
                "department": row.get("department", "Engineering"),
                "location": row.get("location", "Bengaluru"),
                "theme": row["pred_theme"],
                "severity": row["pred_severity"],
                "anonymised_text": row["anonymised_text"],
            })

    for sc in st.session_state["submitted_comments"]:
        if sc.get("escalate") is True:
            escalation_items.append({
                "comment_id": sc["comment_id"],
                "department": sc["department"],
                "location": sc["location"],
                "theme": sc["theme"],
                "severity": sc["severity"],
                "anonymised_text": sc["anonymised_text"],
            })

    st.write(f"Total Escalations Requiring HR Action: **{len(escalation_items)}**")

    if not escalation_items:
        st.success("No pending escalations in queue.")
    else:
        for item in escalation_items:
            cid = item["comment_id"]
            current_decision = st.session_state["escalation_decisions"].get(cid, {})

            with st.expander(f"Case {cid} — {item['department']} | Severity: {item['severity'].upper()} | Theme: {item['theme'].upper()}", expanded=(cid not in st.session_state["escalation_decisions"])):
                st.markdown(f"**Anonymised Employee Report:**\n> \"{item['anonymised_text']}\"")

                status_badge = current_decision.get("status", "Pending Review")
                if status_badge == "Confirmed":
                    st.success(f"Status: **Confirmed Escalation** (Notes: {current_decision.get('notes', 'None')})")
                elif status_badge == "Dismissed":
                    st.info(f"Status: **Dismissed** (Notes: {current_decision.get('notes', 'None')})")
                else:
                    st.warning("Status: **Pending HR Review**")

                col_act1, col_act2 = st.columns([3, 1])
                with col_act1:
                    notes = st.text_input(f"Reviewer Notes for {cid}:", value=current_decision.get("notes", ""), key=f"notes_{cid}")
                with col_act2:
                    st.write("")
                    st.write("")
                    btn_c1, btn_c2 = st.columns(2)
                    if btn_c1.button("✅ Confirm", key=f"conf_{cid}"):
                        st.session_state["escalation_decisions"][cid] = {
                            "status": "Confirmed",
                            "notes": notes,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        st.rerun()
                    if btn_c2.button("❌ Dismiss", key=f"dism_{cid}"):
                        st.session_state["escalation_decisions"][cid] = {
                            "status": "Dismissed",
                            "notes": notes,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        st.rerun()

# ==============================================================================
# PAGE 4: SECURITY LOG
# ==============================================================================
elif page == "4. Security Log":
    st.title("🛡️ Cybersecurity & Prompt Injection Audit Log")
    st.markdown(
        "Audits system instruction overrides, prompt injections, and unauthorized credential extraction attempts. "
        "**Kept strictly isolated from workplace health & safety escalations.**"
    )

    security_events = []
    if not df_eval.empty:
        for _, row in df_eval.iterrows():
            if row.get("pred_injection_attempt") is True or row.get("category") == "injection":
                security_events.append({
                    "comment_id": row["comment_id"],
                    "department": row.get("department", "Engineering"),
                    "threat_type": "Prompt Injection / System Override",
                    "anonymised_text": row["anonymised_text"],
                    "flagged_at": "Dataset Log"
                })

    for sc in st.session_state["submitted_comments"]:
        if sc.get("security_flag") is True:
            security_events.append({
                "comment_id": sc["comment_id"],
                "department": sc["department"],
                "threat_type": "Prompt Injection / Security Keywords",
                "anonymised_text": sc["anonymised_text"],
                "flagged_at": sc.get("submitted_at", "Live Session")
            })

    st.metric("Total Flagged Security Events", len(security_events))

    if security_events:
        sec_df = pd.DataFrame(security_events)
        st.dataframe(
            sec_df[["comment_id", "department", "threat_type", "anonymised_text", "flagged_at"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No security threats or prompt injections detected.")