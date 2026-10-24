import os
from pathlib import Path
import numpy as np
import pandas as pd

def generate_teams_data():
    # Set seed for reproducible results
    np.random.seed(42)

    n_teams = 300
    n_quarters = 8

    departments = ["Engineering", "Sales", "Operations", "HR", "Finance", "Support"]
    locations = ["Mumbai", "Pune", "Bengaluru", "Delhi", "Hyderabad"]

    # Assign team-level properties
    dept_choices = np.random.choice(departments, size=n_teams)
    loc_choices = np.random.choice(locations, size=n_teams)

    # About 10% of teams smaller than 5 (size 3-4), remaining 90% between 5 and 40
    is_small_team = np.random.rand(n_teams) < 0.10
    team_sizes = np.where(
        is_small_team,
        np.random.randint(3, 5, size=n_teams),
        np.random.randint(5, 41, size=n_teams)
    )

    rows = []

    for i in range(n_teams):
        team_id = f"TEAM_{i+1:03d}"
        dept = dept_choices[i]
        loc = loc_choices[i]
        tsize = int(team_sizes[i])

        # Team baseline traits to keep changes smooth and realistic across quarters
        base_sentiment = np.random.uniform(-0.35, 0.45)
        base_pay_neg = np.random.uniform(0.12, 0.45)
        base_workload_neg = np.random.uniform(0.15, 0.55)
        base_manager_neg = np.random.uniform(0.08, 0.40)
        base_growth_neg = np.random.uniform(0.10, 0.45)
        base_overtime = np.random.uniform(4.0, 16.0)
        base_participation = np.random.uniform(0.60, 0.95)
        base_tenure = np.random.uniform(1.2, 5.5)
        
        # Promotion cycle counter (in months)
        promo_months = np.random.choice([0, 3, 6, 9, 12])

        # Current state values
        current_sentiment = base_sentiment
        current_pay_neg = base_pay_neg
        current_workload_neg = base_workload_neg
        current_manager_neg = base_manager_neg
        current_growth_neg = base_growth_neg
        current_overtime = base_overtime
        current_participation = base_participation
        current_tenure = base_tenure
        last_attrition = np.random.uniform(0.02, 0.12)
        next_attrition = last_attrition

        for q in range(1, n_quarters + 1):
            if q > 1:
                # Smooth quarterly progression using an autoregressive process (AR-1)
                current_sentiment = np.clip(
                    0.72 * current_sentiment + 0.28 * base_sentiment + np.random.normal(0, 0.10),
                    -1.0, 1.0
                )
                current_pay_neg = np.clip(
                    0.72 * current_pay_neg + 0.28 * base_pay_neg + np.random.normal(0, 0.05),
                    0.0, 1.0
                )
                current_workload_neg = np.clip(
                    0.72 * current_workload_neg + 0.28 * base_workload_neg + np.random.normal(0, 0.05),
                    0.0, 1.0
                )
                current_manager_neg = np.clip(
                    0.72 * current_manager_neg + 0.28 * base_manager_neg + np.random.normal(0, 0.04),
                    0.0, 1.0
                )
                current_growth_neg = np.clip(
                    0.72 * current_growth_neg + 0.28 * base_growth_neg + np.random.normal(0, 0.05),
                    0.0, 1.0
                )
                current_overtime = np.clip(
                    0.72 * current_overtime + 0.28 * base_overtime + np.random.normal(0, 1.5),
                    0.0, 25.0
                )
                current_participation = np.clip(
                    0.72 * current_participation + 0.28 * base_participation + np.random.normal(0, 0.03),
                    0.2, 1.0
                )
                
                # Months since promotion cycle advances by 3 months each quarter
                promo_months += 3
                # When reaching 12-24 months, a promotion cycle may reset the timer
                if promo_months > 24 or (promo_months >= 12 and np.random.rand() < 0.28):
                    promo_months = np.random.choice([0, 3])

                # Tenure increases smoothly by ~0.25 years each quarter
                current_tenure = np.clip(
                    current_tenure + 0.25 + np.random.normal(0, 0.05),
                    0.5, 10.0
                )

                # Prior quarter's next_q_attrition becomes this quarter's last_q_attrition
                last_attrition = next_attrition

            # Calculate next_q_attrition:
            # - Rises with: overtime, manager negativity, pay negativity, months since promotion, last_q_attrition
            # - Falls with: sentiment, tenure
            # - Plus random noise (so it is not perfectly predictable)
            attrition_score = (
                0.015
                + 0.085 * (current_overtime / 25.0)
                + 0.110 * current_manager_neg
                + 0.080 * current_pay_neg
                + 0.040 * (promo_months / 24.0)
                + 0.140 * last_attrition
                - 0.060 * current_sentiment
                - 0.030 * (current_tenure / 10.0)
                + np.random.normal(0, 0.032)
            )

            # Bound between 0.0 and 0.45
            next_attrition = float(np.clip(attrition_score, 0.0, 0.45))
            high_risk = 1 if next_attrition >= 0.15 else 0

            rows.append({
                "team_id": team_id,
                "department": dept,
                "location": loc,
                "team_size": tsize,
                "quarter": q,
                "avg_sentiment": round(float(current_sentiment), 3),
                "pct_negative_pay": round(float(current_pay_neg), 3),
                "pct_negative_workload": round(float(current_workload_neg), 3),
                "pct_negative_manager": round(float(current_manager_neg), 3),
                "pct_negative_growth": round(float(current_growth_neg), 3),
                "avg_overtime_hours": round(float(current_overtime), 1),
                "participation_rate": round(float(current_participation), 3),
                "months_since_promotion_cycle": int(promo_months),
                "median_tenure_years": round(float(current_tenure), 2),
                "last_q_attrition": round(float(last_attrition), 3),
                "next_q_attrition": round(float(next_attrition), 3),
                "high_risk": high_risk
            })

    df = pd.DataFrame(rows)

    # Save to data/teams.csv
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "teams.csv"
    df.to_csv(csv_path, index=False)

    # Print summary results
    print(f"Number of rows: {len(df)}")
    high_risk_pct = (df['high_risk'].mean()) * 100
    print(f"Percentage of high_risk: {high_risk_pct:.2f}%")
    print("\nFirst 5 rows:")
    print(df.head().to_string())

if __name__ == "__main__":
    generate_teams_data()
