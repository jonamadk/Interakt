# streamlit_app.py

import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import numpy as np
import os
import requests
from mediapipe.framework.formats import landmark_pb2
import time
import csv
from datetime import datetime
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Hand Interaction Tracker",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS TO INCREASE SIDEBAR WIDTH ---
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        width: 350px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Matplotlib Style ---
plt.style.use('seaborn-v0_8-darkgrid')

# --- CSV Logging Setup ---
LOG_FILENAME = 'activity_log.csv'

@st.cache_data
def setup_log_file():
    """Creates the log file and writes the header if it doesn't exist."""
    if not os.path.exists(LOG_FILENAME) or os.path.getsize(LOG_FILENAME) == 0:
        with open(LOG_FILENAME, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Log Entry"])

def log_activity(object_label, duration_seconds):
    """Formats and appends a new activity row to the CSV log file."""
    if object_label is None or duration_seconds <= 1:
        return
    timestamp = datetime.now().strftime('%m/%d/%Y %I:%M %p')
    duration_str = f"{int(round(duration_seconds)):02d}sec"
    log_string = f"Activity:interaction with {object_label}: time: {duration_str} timestamp: {timestamp}"
    with open(LOG_FILENAME, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([log_string])
    print(f"Logged: {log_string}")

# --- Helper Functions ---
def download_file(url, filename):
    """Downloads a file if it doesn't exist, without a visible progress bar."""
    if not os.path.exists(filename):
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading model: {e}")
            st.stop()

def draw_landmarks_on_image(rgb_image, detection_result):
    """Draws hand landmarks on the image using MediaPipe's drawing utilities."""
    annotated_image = np.copy(rgb_image)
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image, hand_landmarks_proto, mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style())
    return annotated_image

def check_overlap(box1, box2):
    """Checks if two bounding boxes [x1, y1, x2, y2] overlap."""
    return not (box1[2] < box2[0] or box2[2] < box1[0] or box1[3] < box2[1] or box2[3] < box1[1])

# --- Model Loading ---
@st.cache_resource
def load_models():
    """Loads and caches the YOLO and MediaPipe models to avoid reloading."""
    with st.spinner("Downloading and loading AI models... please wait."):
        model_url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
        model_filename = 'hand_landmarker.task'
        download_file(model_url, model_filename)
        base_options = python.BaseOptions(model_asset_path=model_filename)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2, min_hand_detection_confidence=0.6,
                                               min_hand_presence_confidence=0.6, min_tracking_confidence=0.5)
        detector = vision.HandLandmarker.create_from_options(options)
        yolo_model = YOLO('yolov8m.pt')
    return detector, yolo_model

# --- Dashboard Functions ---
def parse_log_file():
    """Reads and parses the activity_log.csv into a pandas DataFrame."""
    try:
        log_pattern = re.compile(r"Activity:interaction with (.*?): time: (\d+)sec timestamp: (.*)")
        with open(LOG_FILENAME, 'r') as f:
            lines = f.readlines()[1:]
        if not lines:
            return pd.DataFrame()
        parsed_data = [match.groups() for line in lines if (match := log_pattern.match(line.strip()))]
        df = pd.DataFrame(parsed_data, columns=["Activity", "time", "timestamp"])
        df["time"] = pd.to_numeric(df["time"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], format='%m/%d/%Y %I:%M %p')
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()

def categorize_behavior(row):
    """Categorizes user behavior based on interaction frequency and duration."""
    avg_duration = row['avg_duration_sec']
    interactions = row['interactions']
    if avg_duration < 4 and interactions > 50:
        return "Habit-driven (short, frequent checks)"
    elif avg_duration > 10:
        return "Intentional (focused engagement)"
    else:
        return "Mixed behavior"

def display_dashboard(charts_placeholder, trend_chart_placeholder, insights_placeholder,
                      plot_type, selected_objects, selected_years, selected_months, selected_days):
    """Renders all data visualizations based on the selected filters."""
    df = parse_log_file() # Always get the latest data

    if df.empty:
        charts_placeholder.warning("No activity data has been logged yet.")
        trend_chart_placeholder.write("")
        insights_placeholder.write("")
        return

    # Add date components for filtering
    df['year'] = df['timestamp'].dt.year
    df['month_name'] = df['timestamp'].dt.strftime('%B')
    df['day'] = df['timestamp'].dt.day # Numerical day
    df['day_of_week'] = df['timestamp'].dt.day_name()

    # Create filter mask
    mask = pd.Series(True, index=df.index)
    if selected_objects:
        mask &= df['Activity'].isin(selected_objects)
    if selected_years:
        mask &= df['year'].isin(selected_years)
    if selected_months:
        mask &= df['month_name'].isin(selected_months)
    if selected_days:
        mask &= df['day'].isin(selected_days)
    df_filtered = df[mask]

    if df_filtered.empty:
        charts_placeholder.warning("No data matches the selected filters.")
        trend_chart_placeholder.write("")
        insights_placeholder.write("")
        return

    # --- Render Charts in the Top-Right Placeholder ---
    with charts_placeholder.container():
        st.subheader("Total Interaction Time")
        activity_summary = df_filtered.groupby("Activity")["time"].sum().reset_index()

        if not activity_summary.empty:
            activity_summary["minutes"] = activity_summary["time"] / 60
            fig1, ax1 = plt.subplots()
            sns.barplot(data=activity_summary, x="Activity", y="minutes", palette="viridis", ax=ax1)
            ax1.set_ylabel("Total Minutes"); ax1.set_xlabel("")
            plt.xticks(rotation=45, ha='right'); st.pyplot(fig1, use_container_width=True)
        else:
            st.info("No activity for this chart.")

        st.markdown("---")

        col_pie, col_heatmap = st.columns(2, gap="large")
        with col_pie:
            st.subheader("Activity Proportions")
            if not activity_summary.empty:
                fig_pie, ax_pie = plt.subplots()
                ax_pie.pie(activity_summary['time'], labels=activity_summary['Activity'], autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis", len(activity_summary)))
                ax_pie.axis('equal'); st.pyplot(fig_pie)
            else:
                st.info("No activity for this chart.")

        with col_heatmap:
            st.subheader("Hourly Usage Heatmap")
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            heatmap_data = df_filtered.pivot_table(index='day_of_week', columns=df_filtered['timestamp'].dt.hour, values='time', aggfunc='sum', fill_value=0).reindex(days_order)
            fig2, ax2 = plt.subplots()
            sns.heatmap(heatmap_data, cmap="YlGnBu", ax=ax2, linewidths=.5)
            ax2.set_title("Total Interaction Seconds per Hour/Weekday"); ax2.set_xlabel("Hour of Day"); ax2.set_ylabel("Day of Week")
            st.pyplot(fig2, use_container_width=True)

    # --- Render Trend Chart in its own placeholder ---
    with trend_chart_placeholder.container():
        if not df_filtered.empty:
            df_filtered['month_period'] = df_filtered['timestamp'].dt.to_period('M').astype(str)
            monthly_summary = df_filtered.groupby(['month_period', 'Activity'])['time'].sum().reset_index()
            monthly_summary['hours'] = monthly_summary['time'] / 3600

            if not monthly_summary.empty and len(monthly_summary['month_period'].unique()) > 1:
                fig_trend, ax_trend = plt.subplots(figsize=(12, 6))

                if plot_type == 'Line Plot':
                    sns.lineplot(data=monthly_summary, x="month_period", y="hours", hue="Activity", marker="o", ax=ax_trend)
                    sns.move_legend(ax_trend, "upper left", bbox_to_anchor=(1, 1))
                elif plot_type == 'Area Plot':
                    pivot_df = monthly_summary.pivot(index='month_period', columns='Activity', values='hours').fillna(0)
                    pivot_df.plot(kind='area', stacked=True, ax=ax_trend, colormap='viridis')
                    ax_trend.legend(title='Activity', bbox_to_anchor=(1.05, 1), loc='upper left')

                ax_trend.set_title("Monthly Interaction Trends by Activity")
                ax_trend.set_ylabel("Total Hours")
                ax_trend.set_xlabel("Month")
                ax_trend.tick_params(axis='x', rotation=45)
                ax_trend.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.tight_layout()
                st.pyplot(fig_trend, use_container_width=True)
            else:
                st.info("Not enough data to plot monthly trends (requires data from more than one month).")
        else:
            st.info("Not enough data to plot monthly trends.")

    # --- Render Insights Table in its own placeholder ---
    with insights_placeholder.container():
        insights = df_filtered.groupby("Activity")["time"].agg(
            total_time_sec="sum", avg_duration_sec="mean", interactions="count"
        ).reset_index().round(1)
        if not insights.empty:
            insights["proportion_%"] = (insights["total_time_sec"] / insights["total_time_sec"].sum() * 100).round(1)
            insights["behavior_type"] = insights.apply(categorize_behavior, axis=1)
            insights = insights[['Activity', 'interactions', 'avg_duration_sec', 'proportion_%', 'behavior_type']]
            st.dataframe(insights, use_container_width=True)

            st.subheader("Interpretation")
            interpretation_text = ""
            for _, row in insights.iterrows():
                interpretation_text += (f"- **{row['Activity']}**: _{row['behavior_type']}_ "
                                        f"({row['interactions']} interactions, avg {row['avg_duration_sec']}s, "
                                        f"{row['proportion_%']}% of total time)\n")
            st.markdown(interpretation_text)
        else:
            st.info("No insight data for the current selection.")

# --- ======================== Main App ======================== ---

st.title("👋 Interakt: Real-time Hand-Object Interaction Tracker")
st.markdown("This app uses your webcam for real-time analysis of your hand-object interactions. The dashboard updates automatically!")
st.markdown("---")

# --- Sidebar Controls & Filters ---
with st.sidebar:

    run_camera = st.toggle("Start Camera Feed", value=False)
    TARGET_OBJECTS = st.multiselect(
        "Objects to track:",
        ['cell phone', 'laptop', 'book', 'cup', 'bottle', 'mouse', 'keyboard', 'remote', 'tv'],
        default=['cell phone', 'laptop', 'book', 'cup']
    )
    
    st.markdown("---")

    # --- UPDATED FILTER LOGIC ---
    df_for_filters = parse_log_file()
    # Initialize all filter variables to prevent errors
    selected_objects, selected_years, selected_months, selected_days = [], [], [], []

    # Use columns to place the button next to the expander title
    col_expander, col_button = st.columns([6, 1])

    with col_expander:
        # The expander is now always visible
        expander = st.expander("📊 Dashboard Filters")
        

    with col_button:
        st.button("🔄") # Refresh button - clicking it re-runs the script

    # The content inside the expander is conditional on data existing
    if not df_for_filters.empty:
        with expander:
            # Add the new object filter
            all_objects = sorted(df_for_filters['Activity'].unique())
            selected_objects = st.multiselect("Filter by Object:", options=all_objects, default=[])

            # Date-based filters
            df_for_filters['year'] = df_for_filters['timestamp'].dt.year
            df_for_filters['month_name'] = df_for_filters['timestamp'].dt.strftime('%B')
            df_for_filters['day'] = df_for_filters['timestamp'].dt.day

            all_years = sorted(df_for_filters['year'].unique(), reverse=True)
            selected_years = st.multiselect("Filter by Year:", options=all_years, default=[])

            if not selected_years:
                filtered_by_year_df = df_for_filters
            else:
                filtered_by_year_df = df_for_filters[df_for_filters['year'].isin(selected_years)]
            
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
            all_months = sorted(filtered_by_year_df['month_name'].unique(), key=lambda m: month_order.index(m))
            selected_months = st.multiselect("Filter by Month:", options=all_months, default=[])

            if not selected_months:
                 filtered_by_month_df = filtered_by_year_df
            else:
                 filtered_by_month_df = filtered_by_year_df[filtered_by_year_df['month_name'].isin(selected_months)]

            all_days = sorted(filtered_by_month_df['day'].unique())
            selected_days = st.multiselect("Filter by Day:", options=all_days, default=[])
    else:
        with expander:
            # Show a warning inside the expander if there is no data
            st.warning("No activity data to filter.")

# --- Load Models & Setup Log ---
detector, yolo_model = load_models()
setup_log_file()

# --- App Layout ---
top_container = st.container()
with top_container:
    col_feed, col_dash = st.columns([2, 1.5], gap="large")
    with col_feed:
        st.info("⬅️ Turn on the 'Start Camera Feed' toggle in the sidebar to begin.")
        st.header("🎥 Live Feed")
        FRAME_WINDOW = st.image([])
    with col_dash:
        st.header("📊 Activity Charts")
        charts_placeholder = st.empty()

st.markdown("---")

# --- Define STATIC UI elements for the bottom section ONCE ---
st.header("📈 Monthly Interaction Trends")
plot_type = st.radio(
    "Select Trend Chart Type:",
    ('Line Plot', 'Area Plot'),
    horizontal=True,
    key="trend_chart_type"
)
trend_chart_placeholder = st.empty()

st.markdown("---")

st.header("🧠 User Behavior Insights")
insights_placeholder = st.empty()

# --- Main Logic ---
def update_dashboard():
    """Wrapper to pass all required arguments to the display function."""
    display_dashboard(charts_placeholder, trend_chart_placeholder, insights_placeholder,
                      plot_type, selected_objects, selected_years, selected_months, selected_days)

update_dashboard() # Initial display

if run_camera:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam. Please grant camera permissions and refresh.")
    else:
        if 'is_using_object' not in st.session_state:
            st.session_state.is_using_object = False
            st.session_state.usage_start_time = None
            st.session_state.current_object_label = None
        last_dashboard_update_time = time.time()
        
        while cap.isOpened() and run_camera:
            success, frame = cap.read()
            if not success:
                st.error("Failed to capture image from camera."); break
            
            frame = cv2.flip(frame, 1)
            
            # --- Detection Logic ---
            yolo_results = yolo_model(frame, verbose=False)
            detected_objects = [{'box': [int(x) for x in box.xyxy[0]], 'label': yolo_model.names[int(box.cls[0])]} for r in yolo_results for box in r.boxes if yolo_model.names[int(box.cls[0])] in TARGET_OBJECTS]
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)
            annotated_image = cv2.cvtColor(draw_landmarks_on_image(rgb_frame, detection_result), cv2.COLOR_RGB2BGR)
            
            hand_boxes = []
            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    h, w, _ = frame.shape
                    x_coords = [landmark.x * w for landmark in hand_landmarks]; y_coords = [landmark.y * h for landmark in hand_landmarks]
                    hand_boxes.append([int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))])
            
            # --- Interaction and Logging Logic ---
            object_held_this_frame, held_object_label_this_frame = False, None
            for obj in detected_objects:
                is_held = any(check_overlap(obj['box'], h_box) for h_box in hand_boxes)
                color = (0, 255, 0) if is_held else (255, 0, 255)
                text = f"INTERACTING: {obj['label']}" if is_held else obj['label']
                cv2.rectangle(annotated_image, (obj['box'][0], obj['box'][1]), (obj['box'][2], obj['box'][3]), color, 3)
                cv2.putText(annotated_image, text, (obj['box'][0], obj['box'][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if is_held:
                    object_held_this_frame, held_object_label_this_frame = True, obj['label']; break
            
            if object_held_this_frame and not st.session_state.is_using_object:
                st.session_state.update(is_using_object=True, usage_start_time=time.time(), current_object_label=held_object_label_this_frame)
            
            elif not object_held_this_frame and st.session_state.is_using_object:
                log_activity(st.session_state.current_object_label, time.time() - st.session_state.usage_start_time)
                st.cache_data.clear()
                update_dashboard()
                st.session_state.update(is_using_object=False, usage_start_time=None, current_object_label=None)
            
            FRAME_WINDOW.image(annotated_image, channels="BGR")
            
            if time.time() - last_dashboard_update_time > 5:
                update_dashboard()
                last_dashboard_update_time = time.time()
        
        cap.release()
        st.info("Camera stopped. Performing final dashboard update.")
        st.cache_data.clear()
        update_dashboard()
else:
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(black_frame, "Camera is OFF", (190, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    FRAME_WINDOW.image(black_frame)