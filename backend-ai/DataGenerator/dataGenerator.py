import random
import pandas as pd

# Define age groups
age_groups = ["Infant", "Child", "Adolescent", "Adult", "Elderly"]

# Function to randomly pick values within ranges
def random_vital(low, high):
    return random.randint(low, high)

# Function to assign label and probable_condition
def assign_labels_and_condition(heart_rate, resp_rate, spo2, temp, glucose):
    label_status = "Normal"

    # Check vitals for abnormal values
    if (
        heart_rate < 60 or heart_rate > 120 or
        resp_rate < 12 or resp_rate > 24 or
        spo2 < 95 or
        temp < 36.1 or temp > 37.2 or
        glucose < 70 or glucose > 125
    ):
        label_status = "Warning"

    if (
        heart_rate < 50 or heart_rate > 140 or
        resp_rate < 10 or resp_rate > 40 or
        spo2 < 90 or
        temp < 35.0 or temp > 38.0 or
        glucose < 60 or glucose > 180
    ):
        label_status = "Critical"

    # Simplified probable_condition
    probable_condition = "Healthy" if label_status == "Normal" else "Not Healthy"

    return label_status, probable_condition

# Generate synthetic dataset
def generate_patient_data(n_samples=100, balance=True):
    data = []
    healthy, not_healthy = 0, 0
    target_each = n_samples // 2 if balance else None

    i = 1
    while len(data) < n_samples:
        age_group = random.choice(age_groups)
        heart_rate = random_vital(50, 160)
        resp_rate = random_vital(10, 30)
        bp_sys = random_vital(90, 140)
        bp_dia = random_vital(60, 90)
        spo2 = random_vital(85, 100)
        temp = round(random.uniform(35.0, 39.0), 1)
        glucose = random_vital(60, 180)

        label_status, probable_condition = assign_labels_and_condition(
            heart_rate, resp_rate, spo2, temp, glucose
        )

        # If balancing, enforce equal class counts
        if balance:
            if probable_condition == "Healthy" and healthy >= target_each:
                continue
            if probable_condition == "Not Healthy" and not_healthy >= target_each:
                continue

        data.append({
            "patient_id": i,
            "age_group": age_group,
            "heart_rate": heart_rate,
            "resp_rate": resp_rate,
            "blood_pressure_systolic": bp_sys,
            "blood_pressure_diastolic": bp_dia,
            "spo2": spo2,
            "temperature_c": temp,
            "glucose_mgdl": glucose,
            "label_status": label_status,
            "probable_condition": probable_condition
        })

        if probable_condition == "Healthy":
            healthy += 1
        else:
            not_healthy += 1
        i += 1

    return pd.DataFrame(data)

# Example usage
df = generate_patient_data(500, balance=True)
print(df["probable_condition"].value_counts())
print(df.head())

# Save to CSV
df.to_csv("Datasets/vitals.csv", index=False)
