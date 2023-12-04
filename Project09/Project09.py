import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

warnings.filterwarnings("ignore")

data = pd.read_csv("StressLevelDataset.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# INFO COLUMN
for i in range(21):
    if i < 20:
        print(f"Cột thứ {i + 1}:\n\t{X.iloc[:, i].value_counts()}")
    else:
        print(f"Cột nhãn:\n{y.value_counts()}")

# DESCRIPTIVE STATISTICS
num_students = len(data)

average_anxiety_level = data['anxiety_level'].mean()

students_with_mental_health_history = data[data['mental_health_history'] == 1]
num_students_with_mental_health_history = len(students_with_mental_health_history)

print("\nDESCRIPTIVE STATISTICS")
print(f"1) Number of students in the dataset: {num_students}")
print(f"2) Average anxiety level of students: {average_anxiety_level}")
print(f"3) Number of students with a history of mental health issues: {num_students_with_mental_health_history}")

psychological_factors = ['anxiety_level', 'self_esteem', 'mental_health_history', 'depression']
physiological_factors = ['headache', 'blood_pressure', 'sleep_quality', 'breathing_problem']
environmental_factors = ['noise_level', 'living_conditions', 'safety', 'basic_needs']
academic_factors = ['academic_performance', 'study_load', 'teacher_student_relationship', 'future_career_concerns']
social_factors = ['social_support', 'peer_pressure', 'extracurricular_activities', 'bullying']

num_students_with_negative_psychological = data[psychological_factors].apply(lambda x: x.lt(3).sum(), axis=1)
num_students_with_negative_physiological = data[physiological_factors].apply(lambda x: x.gt(3).sum(), axis=1)
num_students_with_negative_environmental = data[environmental_factors].apply(lambda x: x.gt(3).sum(), axis=1)
num_students_with_negative_academic = data[academic_factors].apply(lambda x: x.lt(3).sum(), axis=1)
num_students_with_negative_social = data[social_factors].apply(lambda x: x.gt(3).sum(), axis=1)

factors = ['Psychological', 'Physiological', 'Environmental', 'Academic', 'Social']
negative_experiences = [
    num_students_with_negative_psychological.sum(),
    num_students_with_negative_physiological.sum(),
    num_students_with_negative_environmental.sum(),
    num_students_with_negative_academic.sum(),
    num_students_with_negative_social.sum()
]
colors = sb.color_palette("pastel")

plt.figure(figsize=(10, 8))
ax = sb.barplot(x=factors, y=negative_experiences, palette=colors)
plt.title("Number of Students Reporting Negative Experiences or Conditions by Factor")
plt.xlabel("Factors")
plt.xticks(rotation=45)
plt.ylabel("Number of Students")

for i, v in enumerate(negative_experiences):
    ax.text(i, v + 0.1, str(v), ha='center', va='bottom')

plt.show()

# PSYCHOLOGICAL FACTORS
average_self_esteem = data['self_esteem'].mean()
students_below_average_self_esteem = data[data['self_esteem'] < average_self_esteem]
num_students_below_average_self_esteem = len(students_below_average_self_esteem)

threshold = 10
data['depression_binary'] = (data['depression'] > threshold).astype(int)

if 'num_students' not in locals():
    num_students = len(data)

percentage_students_experiencing_depression = (data['depression_binary'].sum() / num_students) * 100
print("\nPSYCHOLOGICAL FACTORS")
print(f"1) Number of students with self-esteem below the average: {num_students_below_average_self_esteem}")
print(f"2) Percentage of students experiencing depression: {percentage_students_experiencing_depression:.2f}%")

# PHYSIOLOGICAL FACTORS
students_with_frequent_headaches = data[data['headache'] == 1]
num_students_with_frequent_headaches = len(students_with_frequent_headaches)

average_blood_pressure = data['blood_pressure'].mean()

students_with_poor_sleep_quality = data[data['sleep_quality'] < 3]
num_students_with_poor_sleep_quality = len(students_with_poor_sleep_quality)

print("\nPHYSIOLOGICAL FACTORS")
print(f"1) Number of students experiencing frequent headaches: {num_students_with_frequent_headaches}")
print(f"2) Average blood pressure reading among the students: {average_blood_pressure}")
print(f"3) Number of students with poor sleep quality: {num_students_with_poor_sleep_quality}")

# ENVIRONMENTAL FACTORS
students_in_high_noise_conditions = data[data['noise_level'] > 3]
num_students_in_high_noise_conditions = len(students_in_high_noise_conditions)

threshold_for_safety = 2
data['feeling_unsafe'] = (data['safety'] <= threshold_for_safety).astype(int)

percentage_students_feeling_unsafe = (data['feeling_unsafe'].sum() / num_students) * 100

students_without_basic_needs = data[data['basic_needs'] == 0]
num_students_without_basic_needs = len(students_without_basic_needs)

print("\nENVIRONMENTAL FACTORS")
print(f"1) Number of students living in high noise conditions: {num_students_in_high_noise_conditions}")
print(f"2) Percentage students feeling unsafe in their living conditions: {percentage_students_feeling_unsafe:.2f}%")
print(f"3) Number of students without their basic needs met: {num_students_without_basic_needs}")

# ACADEMIC FACTORS
students_below_average_academic_performance = data[data['academic_performance'] < 3]
num_students_below_average_academic_performance = len(students_below_average_academic_performance)

average_study_load = data['study_load'].mean()

students_with_future_career_concerns = data[data['future_career_concerns'] == 1]
num_students_with_future_career_concerns = len(students_with_future_career_concerns)

print("\nACADEMIC FACTORS")
print(f"1) Number of students with below-average academic performance: {num_students_below_average_academic_performance}")
print(f"2) Average study load reported by students: {average_study_load}")
print(f"3) Number of students with concerns about their future careers: {num_students_with_future_career_concerns}")

# SOCIAL FACTORS
students_with_strong_social_support = data[data['social_support'] > 3]
num_students_with_strong_social_support = len(students_with_strong_social_support)

bullying_threshold = 3
data['bullying_binary'] = (data['bullying'] > bullying_threshold).astype(int)

if 'num_students' not in locals():
    num_students = len(data)

percentage_students_experiencing_bullying = (data['bullying_binary'].sum() / num_students) * 100

students_pariticipating_in_extracurricular = data[data['extracurricular_activities'] == 1]
num_students_pariticipating_in_extracurricular = len(students_pariticipating_in_extracurricular)

print("\nSOCIAL FACTORS")
print(f"1) Number of students with strong social support: {num_students_with_strong_social_support}")
print(f"2) Percentage of students experiencing bullying: {percentage_students_experiencing_bullying:.2f}%")
print(f"3) Number of students participating in extracurricular activities: {num_students_pariticipating_in_extracurricular}")

# COMPARATIVE ANALYSIS
correlation_anxiety_academic = data['anxiety_level'].corr(data['academic_performance'])
correlation_anxiety_depression = data['sleep_quality'].corr(data['depression'])

students_with_bullying_history = data[data['bullying'] == 1]
students_with_bullying_history_and_metal_health_history = students_with_bullying_history[students_with_bullying_history['mental_health_history'] == 1]

percentage_students_with_bullying_history_and_metal_health_history = (len(students_with_bullying_history_and_metal_health_history) / len(students_with_bullying_history)) * 100

print("\nCOMPARATIVE ANALYSIS")
print(f"1) Correlation between anxiety level and academic performance: {correlation_anxiety_academic}")
print(f"2) Correlation between sleep quality and depression: {correlation_anxiety_depression}")
print(f"3) Percentage of students with bullying history and mental health history: {percentage_students_with_bullying_history_and_metal_health_history}\n")

# GENERAL EXPLORATION
factors = ['Psychological', 'Physiological', 'Environmental', 'Academic', 'Social']
negative_experiences = [num_students_below_average_self_esteem,
                        num_students_with_frequent_headaches,
                        num_students_in_high_noise_conditions,
                        num_students_without_basic_needs,
                        len(students_with_bullying_history)]
factor_with_most_negatives = factors[negative_experiences.index(max(negative_experiences))]

sb.set(style='whitegrid', palette='Set2')
factors_to_plot = data[['anxiety_level', 'self_esteem', 'depression', 'sleep_quality', 'academic_performance']]

g = sb.PairGrid(factors_to_plot)
g.map_upper(sb.scatterplot)
g.map_lower(sb.kdeplot, colors='C0')
g.map_diag(sb.histplot, kde_kws={'color': 'C0'})

plt.subplots_adjust(top=0.95)
g.fig.suptitle("Pairplot of Key Factors", size=16)

plt.show()

# MODEL
model = RandomForestRegressor()
feature_importance = {}

for factor, features in zip(factors, [psychological_factors,
                                      physiological_factors,
                                      environmental_factors,
                                      academic_factors,
                                      social_factors]):
    X = data[features]
    y = data['stress_level']

    model.fit(X, y)

    importance = model.feature_importances_

    feature_importance[factor] = {feature: importance_value for feature, importance_value in zip(features, importance)}

for factor, importance_dict in feature_importance.items():
    print(f"Factor: {factor}")
    for feature, importance_value in importance_dict.items():
        print(f"- Feature: {feature}, Importance: {importance_value}")
    print()

# CORRELATION HEATMAP OF KEY FACTORS
correlation_matrix = data[['anxiety_level', 'self_esteem', 'depression', 'sleep_quality', 'academic_performance']].corr()
plt.figure(figsize=(10, 6))
sb.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Key Factors")
plt.show()

# BOX PLOTS
plt.figure(figsize=(10, 6))
sb.boxplot(data=data[['anxiety_level', 'self_esteem', 'depression', 'sleep_quality', 'academic_performance']], orient='h')
plt.title("Box Plots of Key Factors")
plt.show()

# DETERMINING SPECTIFIC FEATURE IMPACT
psychological_feature_importance = feature_importance['Psychological']
physiological_feature_importance = feature_importance['Physiological']
environmental_feature_importance = feature_importance['Environmental']
academic_feature_importance = feature_importance['Academic']
social_feature_importance = feature_importance['Social']

def plot_feature_importance(factor_name, importance_dict):
    features = list(importance_dict.keys())
    importance = list(importance_dict.values())
    plt.figure(figsize=(10, 6))
    sb.barplot(x=importance, y=features, orient='h')
    plt.title(f"Feature Importance within {factor_name} Factor")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.show()

plot_feature_importance('Psychological', psychological_feature_importance)
plot_feature_importance('Physiological', physiological_feature_importance)
plot_feature_importance('Environmental', environmental_feature_importance)
plot_feature_importance('Academic', academic_feature_importance)
plot_feature_importance('Social', social_feature_importance)