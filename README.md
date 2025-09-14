# Infosys_SpringBoard_Project 
Title : Detecting Unusual Health Patterns Using Fitness Watch Data
A Data Science Project for health monitoring
Project Statement:

With the surge in wearable fitness devices, a vast amount of health related time-series data is generated daily. Yet, users and healthcare providers often fail to notice subtle early warning signs of health anomalies like irregular heartbeats, sleep disturbances, or abnormal activity levels. This project leverages Al-based anomaly detection to intelligently analyze fitness watch data and flag unusual health patterns. By enabling proactive health monitoring, it supports preventive healthcare and personalized wellness insights.

Outcomes:

-Accurate detection of anomalies in heart rate, sleep duration, and step count.
-Personalized health alerts using time-series models and clustering techniques.
-Integration with raw fitness watch data (CSV/JSON format).
-Real-time or batch-based anomaly flagging for dynamic use cases.
-Configurable metric thresholds (e.g., HR>120 bpm for>10 mins).
-Interactive dashboards for visual trend and anomaly tracking.
-Exportable reports for users or healthcare professionals.

Modules to be Implemented:

Module 1: Data Collection and Preprocessing
-Import health data (heart rate, steps, sleep) from fitness trackers in CSV/JSON format.
-Clean and normalize timestamps, interpolate missing values.
-Resample to consistent time intervals (e.g., per minute, hourly).

Module 2: Feature Extraction and Modeling
-Extract relevant time-series features using TSFresh (mean, std, kurtosis, etc.).
-Apply Facebook Prophet to model seasonal trends and detect deviations.
-Use clustering (e.g., KMeans, DBSCAN) to group similar behavioral patterns.

Module 3: Anomaly Detection and Visualization
-Rule-based (thresholds) and model-based anomaly detection (Prophet residuals, clustering outliers).
-Visualization with Matplotlib or Plotly: Line charts, anomaly highlights, time windows.

Module 4: Dashboard for Insights
-Build an interactive dashboard using Streamlit.
-Allow users to upload fitness data files, trigger anomaly detection dynamically.
-Generate downloadable export reports(pdf/csv) for anomaly summaries and trends.

Tools & Technology

“Now, what technologies and tools will we use?”
-Python – the main programming language.
 Libraries: Pandas, NumPy, Matplotlib, Plotly, Scikit-learn.
-TSFresh – for feature extraction from time-series data.
-Facebook Prophet – for trend detection and seasonality modeling.
-Clustering Algorithms – KMeans and DBSCAN to detect patterns and outliers.
-Streamlit – to build a simple, interactive web dashboard.
-Data Formats – CSV or JSON files exported from fitness trackers.

Types of Anomalies

-Point Anomaly: A single data point is unusual.
 Example: Heart rate suddenly spikes to 180 bpm while sitting.
-Contextual Anomaly: Data is unusual in a specific context.
 Example: High heart rate is normal during running, but abnormal while sleeping.
-Collective Anomaly: A group of data points together is unusual.
 Example: Sleep is below 3 hours for 7 days continuously.

Why Detect Anomalies?

-In healthcare: Early detection of irregularities → prevent diseases.
-In finance: Detect fraud transactions.
-In cybersecurity: Detect hacking attempts.
-In manufacturing: Detect machine faults early.

Timestamps & Time Zones

What is a timestamp?
 -A datetime value attached to each record.
 -Example: 2025-09-08 10:00:00 vs 2025-09-08T10:00:00Z.
Common timestamp formats:
 -ISO format: YYYY-MM-DDTHH:MM:SSZ.
 -Local format: YYYY/MM/DD HH:MM AM/PM.
2024-01-01 08:00:00    72.0  (heart rate)
2024-01-01 08:01:00    NaN   (missing!)
2024-01-01 08:02:00    NaN   (missing!)
2024-01-01 08:03:00    NaN   (missing!)
2024-01-01 08:04:00    89.0  (back online)

Types of Missing Data
1.Missing Completely at Random (MCAR):
 No pattern to missing values
 Device randomly fails to record
 Safe to use simple imputation methods
2.Missing at Random (MAR):
 Missing depends on observed variables
 Heart rate missing during high activity
 Can use predictive imputation
3.Missing Not at Random (MNAR):
 Missing depends on unobserved factors
 User removes device during private activities
 Most complex to handle properly

08:00:00  Heart rate recorded
08:00:05  Heart rate recorded  
08:00:12  Heart rate recorded (irregular!)
08:00:23  Heart rate recorded (very irregular!)
08:01:00  Steps recorded (different schedule!)
08:01:00  Heart rate recorded

Project Architecture (Deeper Theory)

Break the pipeline like a system design:
1. Data Source: Fitness watch → raw CSV/JSON.
2. Ingestion Layer: Python script to load files.
3. Preprocessing Layer: Cleaning, handling missing values.
4. Feature Extraction Layer: TSFresh generates 100s of features.
5. Model Layer: Prophet & clustering models.
6. Anomaly Engine: Combines thresholds + residuals + clustering outliers.
7. Visualization Layer: Plotly, Matplotlib charts.
8. Application Layer: Streamlit dashboard.

<img width="836" height="364" alt="Screenshot 2025-09-15 010700" src="https://github.com/user-attachments/assets/7f59e475-5852-4e28-862b-d6af29e76292" />

