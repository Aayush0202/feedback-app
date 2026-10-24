"""
rules.py: Privacy and Access Control Rules for the Anonymous Employee Feedback System.
Enforces differential privacy thresholds (k-anonymity) and strict role-based queue isolation.
"""

import pandas as pd

# Global privacy threshold: minimum individuals/comments required to display aggregates
MIN_GROUP_SIZE = 5
HIDDEN_MESSAGE = "Hidden: group too small to protect anonymity"

def is_group_safe(count: int, min_size: int = MIN_GROUP_SIZE) -> bool:
    """Returns True if the group size or comment count meets the minimum threshold."""
    return count >= min_size

def filter_teams_by_size(df_teams: pd.DataFrame, min_size: int = MIN_GROUP_SIZE) -> pd.DataFrame:
    """
    Enforces privacy rule: removes any team with fewer than MIN_GROUP_SIZE members
    from all analytics and ML inference pipelines.
    """
    if "team_size" not in df_teams.columns:
        return df_teams
    return df_teams[df_teams["team_size"] >= min_size].copy()

def mask_small_department_aggregates(counts: pd.Series | dict, min_count: int = MIN_GROUP_SIZE) -> dict:
    """
    Given a series or dictionary of counts per department/group:
    Masks any count less than min_count with HIDDEN_MESSAGE to prevent individual identification.
    """
    items = counts.items() if isinstance(counts, (dict, pd.Series)) else []
    sanitized = {}
    for group, count in items:
        if count < min_count:
            sanitized[group] = HIDDEN_MESSAGE
        else:
            sanitized[group] = int(count)
    return sanitized

def filter_comments_by_group_size(
    df_comments: pd.DataFrame,
    group_col: str = "department",
    min_count: int = MIN_GROUP_SIZE
) -> pd.DataFrame:
    """
    Hides groups/departments that have fewer than MIN_GROUP_SIZE comments
    to protect the anonymity of contributing employees.
    """
    if group_col not in df_comments.columns:
        return df_comments

    group_counts = df_comments[group_col].value_counts()
    valid_groups = group_counts[group_counts >= min_count].index
    return df_comments[df_comments[group_col].isin(valid_groups)].copy()

def can_access_escalation_queue(role: str) -> bool:
    """
    Access Control Rule:
    Escalations (harassment, discrimination, safety threats) are strictly confidential
    and routed ONLY to the 'HR Reviewer' queue. They are NEVER accessible in a manager view.
    """
    normalized_role = str(role).strip().lower()
    return normalized_role in {"hr reviewer", "hr_reviewer", "hr_admin", "compliance"}
