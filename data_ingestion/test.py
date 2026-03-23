import pandas as pd

def explore_survey_participation(answers_path: str):
    """
    Explore how many unique users answered surveys 3 and 4.

    Prints:
    - number of users per survey
    - overlap between surveys
    """
    df = pd.read_csv(answers_path)

    # Keep only surveys 3 and 4
    df = df[df["survey_id"].isin([3, 4])]

    # Unique users per survey
    users_per_survey = (
        df.groupby("survey_id")["user_id"]
        .nunique()
        .sort_index()
    )

    print("=== Unique users per survey ===")
    for sid, count in users_per_survey.items():
        print(f"Survey {sid}: {count} users")

    # Sets of users
    users_3 = set(df[df["survey_id"] == 3]["user_id"])
    users_4 = set(df[df["survey_id"] == 4]["user_id"])

    # Overlap analysis
    both = users_3 & users_4
    only_3 = users_3 - users_4
    only_4 = users_4 - users_3

    print("\n=== Overlap ===")
    print(f"Answered both: {len(both)}")
    print(f"Only survey 3: {len(only_3)}")
    print(f"Only survey 4: {len(only_4)}")

    return {
        "survey_3": len(users_3),
        "survey_4": len(users_4),
        "both": len(both),
        "only_3": len(only_3),
        "only_4": len(only_4),
    }

if __name__ == "__main__":
    explore_survey_participation("data_ingestion/survey-answers.csv")