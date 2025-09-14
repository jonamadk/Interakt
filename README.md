# Interakt
Interakt is a real-time hand–object interaction tracker powered by YOLO , MediaPipe and Streamlit. It uses your webcam to detect hands and everyday objects, logs interaction durations automatically, and visualizes behavior patterns through an intuitive dashboard. 

# 👋 Interakt: Real-time Hand-Object Interaction Tracker

**Interakt** is a real-time hand–object interaction tracker powered by **YOLO**, **MediaPipe** and **Streamlit**. It uses your webcam to detect hands and everyday  (Cell Phone, books, bottle, cup, laptop), and visualizes userbehavior patterns through an intuitive dashboard.



---

## ✨ Features

* **🎥 Live Webcam Feed:** Real-time video processing directly in your browser.
* **🤖 AI-Powered Detection:** Utilizes **YOLOv8** for state-of-the-art object detection and **MediaPipe Hand Landmarker** for robust, high-fidelity hand tracking.
* **🤝 Smart Interaction Logic:** Intelligently detects interactions by checking for overlaps between the bounding boxes of your hands and specified objects.
* **💾 Persistent Logging:** All detected interactions are timestamped and saved to a local `activity_log.csv` file, so your data is never lost.
* **📊 Interactive Dashboard:** A comprehensive dashboard built with Matplotlib and Seaborn that visualizes your interaction data through various charts:
    * Total interaction time by object.
    * Proportional activity breakdown (pie chart).
    * Hourly usage patterns (heatmap).
    * Monthly interaction trends (line/area charts).
* **🧠 Behavior Analysis:** Automatically categorizes your interaction patterns (e.g., 'Habit-driven' vs. 'Intentional') based on frequency and duration, providing unique insights into your habits.
* **⚙️ User-Friendly Controls:** Easily toggle the camera, select which objects to track from a predefined list, and filter the dashboard visualizations by year, month, and day.

---

## 🔧 How It Works

The application follows a continuous loop to process webcam frames and generate insights:

1.  **Frame Capture:** The app captures video frames from your webcam using OpenCV.
2.  **Object Detection:** Each frame is passed to a **YOLOv8** model, which identifies and draws bounding boxes around a set of target objects (like 'cell phone', 'cup', 'book').
3.  **Hand Landmark Detection:** Simultaneously, the frame is processed by the **MediaPipe Hand Landmarker** model to detect the presence and precise location of hands, generating a bounding box for each.
4.  **Interaction Check:** The core logic checks if any hand bounding box **overlaps** with any target object's bounding box.
5.  **State Management & Logging:**
    * If an overlap is detected for the first time on an object, a timer starts.
    * When the overlap ceases, the timer stops, and the interaction duration, object name, and current timestamp are logged to `activity_log.csv`.
6.  **Live Dashboard Updates:** The dashboard periodically reads the updated `activity_log.csv` file, processes the data with Pandas, and refreshes all plots and insights to reflect the latest activity.

---

## 🚀 Getting Started

Follow these instructions to get the project running on your local machine.

### Prerequisites

* Python 3.8+
* A webcam connected to your computer.

### Installation

1.  **Clone the repository (or download the script):**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Mac/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    Create a file named `requirements.txt` with the following content:
    ```
    streamlit
    opencv-python
    mediapipe
    ultralytics
    pandas
    matplotlib
    seaborn
    requests
    ```
    Then, run the installation command:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The first time you run the app, it will automatically download the necessary YOLOv8 and MediaPipe model files.*

### Running the Application

1.  Navigate to the project directory in your terminal.
2.  Run the following command:
    ```bash
    streamlit run src/streamlit_app.py
    ```
3.  Your web browser will automatically open a new tab with the application running. Grant camera permissions when prompted.

---

## 💻 Usage Guide

1.  **Start the Camera:** Use the "Start Camera Feed" toggle in the sidebar to begin real-time analysis.
2.  **Select Objects:** In the "Objects to track" multiselect box, choose the items you want the app to monitor.
3.  **Interact:** Hold up one of the selected objects so that both your hand and the object are visible to the camera. You will see a green box around the object when an interaction is detected.
4.  **View the Dashboard:** Watch as the charts and insights on the dashboard update in near real-time based on your interactions.
5.  **Filter Data:** Use the "Dashboard Filters" expander in the sidebar to drill down into specific time periods.

---

## 📁 Project Files

* `streamlit_app.py`: The main Python script containing all the application logic.
* `activity_log.csv`: (Generated on first run) The CSV file where all interaction data is stored.
* `hand_landmarker.task`: (Downloaded on first run) The MediaPipe model file for hand detection.
* `yolov8m.pt`: (Downloaded on first run) The YOLO model file for object detection.