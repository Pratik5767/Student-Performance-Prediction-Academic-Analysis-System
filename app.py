# Imports
import pandas as pd
import joblib


# loading model
model = joblib.load('student_pass_fail_model.pkl')


# inputs
FEATURES = ["Study_Hours", "Attendance", "Previous_Marks", "Assignment_Score", "Internal_Marks", "Previous_Failures"]


# functions that makes prediction
def student_performance_recommendation(study_hours, attendance, previous_marks, assignment_score, internal_marks, previous_failures):
    new_student = pd.DataFrame(
        [[study_hours, attendance, previous_marks, assignment_score, internal_marks, previous_failures]],
        columns=FEATURES
    )

    result = model.predict(new_student)[0]
    probability = model.predict_proba(new_student)[0][1] * 100

    if result == 1:
        return "Pass", probability, "Student performance is satisfactory. Continue current study pattern."
    else:
       return "Fail", probability, "Student requires additional academic support. Improve attendance and study hours."



if __name__ == '__main__':
    label, chance, advice = student_performance_recommendation(7, 78, 82, 70, 85, 0)
    print("Predicted Result :", label)
    print(f"Chance of passing: {chance:.1f}%")
    print("Recommendation   :", advice)