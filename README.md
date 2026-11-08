# AI Face Replacement Studio

A Streamlit-based Python application for replacing a person's face in a video using an uploaded reference image. The project is designed to work locally and includes video validation, reference image verification, face detection, video processing, and MP4 export.

## Features

- Streamlit dashboard with a modern dark theme
- MP4, MOV, and AVI video upload support
- JPG, JPEG, and PNG face image uploads
- Video preview, metadata display, and image preview
- Consent enforcement for lawful, consent-based use
- Face detection and replacement pipeline using OpenCV and InsightFace
- Progress tracking during processing
- Output video preview and download
- Automatic audio preservation and final MP4 export
- Advanced facial motion controls for lip, eye, expression, and head movement synchronization
- Sidebar toggles for face replacement, lip sync, eye sync, expression sync, head motion, lighting correction, and output resolution

## Project Structure

```text
ai-face-replacement/
├── app.py
├── requirements.txt
├── README.md
├── config.py
├── modules/
│   ├── audio_processor.py
│   ├── face_aligner.py
│   ├── face_detector.py
│   ├── face_replacement.py
│   ├── face_tracker.py
│   ├── video_exporter.py
│   └── video_processor.py
├── utils/
│   ├── file_handler.py
│   ├── helpers.py
│   └── validation.py
├── uploads/
├── outputs/
├── assets/
└── temp/
```

## Installation on Windows

1. Install Python 3.11 or newer.
2. Open PowerShell in the project root.
3. Create and activate a virtual environment:

```powershell
cd D:\F\mine\pp\ai-face-replacement
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

5. Install FFmpeg and add it to PATH.
   - Recommended on Windows: `winget install Gyan.Dev.FFmpeg`
   - Or download from https://www.ffmpeg.org/download.html and add the `bin` folder to PATH, for example `C:\ffmpeg\bin`
   - Verify installation with: `ffmpeg -version`

6. Run the Streamlit app:

```powershell
streamlit run app.py
```

## Model Setup and Runtime Notes

This project uses InsightFace for face detection and landmark extraction. On first run, the model may download and cache weights automatically depending on your environment. If the model is unavailable or cannot initialize, the application falls back to OpenCV Haar cascades for basic face detection.

For best results:

- Use a clear front-facing reference portrait with good lighting.
- Ensure the source video contains a visible, consented face.
- Use a static camera shot when possible for cleaner replacement results.
- Keep the reference image and target person consented and legally authorized.

## Security and Compliance

- Uploaded files are stored only temporarily in the local `uploads` folder.
- Temporary files are cleaned up after processing.
- The app requires users to confirm they have permission to use the reference person's likeness.
- Do not apply this tool to impersonation, identity fraud, or deceptive content.

## Usage

1. Launch the app.
2. Upload a source video.
3. Upload a reference face image.
4. Confirm consent.
5. Click "Start Processing".
6. Preview the final output and download the MP4.

## Notes

This implementation is designed for local, consent-based use and can be extended with more advanced face swap algorithms or higher-fidelity models depending on GPU/CPU resources and deployment constraints.
