# ArUco Crack Measurement

Desktop application for measuring fatigue-crack length from specimen images using ArUco markers as the geometric reference.

The project contains:

- a PySide6 application for image-based and live-view crack measurements
- edge-refined ArUco marker rectification for more consistent metrology
- autosaved measurement-session recovery
- crack-growth plotting from current measurements and imported CSV/TSV files
- archived legacy prototype code for reference

## Main Features

- `Live View` tab for real-time visual tracking
- `Image Measurement` tab for saved-image measurements and TSV export
- `Crack Growth` tab for plotting crack length versus cycle count
- configurable ArUco detector settings in the UI
- crash-recovery autosave for the current measurement session
- a committed Windows build under `release/ArucoCrackMeasurement/`

## Project Layout

- `main/Aruco_crack_len_measurement.py`:
  Application entry point and `--self-test` launcher
- `main/aruco_measurement_app/`:
  Main application package
- `main/legacy_Aruco_crack_len_measurement.py`:
  Archived original prototype
- `Test images/` and `Test pics/`:
  Sample images used during development and verification
- `technical specifications/specifications.md`:
  Original project notes and requested functionality
- `build_windows.ps1`:
  Windows packaging script using PyInstaller
- `build_windows.bat`:
  Double-clickable Windows wrapper around the build script
- `release/ArucoCrackMeasurement/`:
  Packaged Windows application bundle to copy to another PC

## Quick Start

Create a virtual environment and install runtime dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the application:

```powershell
.\.venv\Scripts\python.exe main\Aruco_crack_len_measurement.py
```

Run the bundled smoke test:

```powershell
.\.venv\Scripts\python.exe main\Aruco_crack_len_measurement.py --self-test
```

## Using The Packaged App

If you do not want to compile anything, use the already packaged Windows build in:

```text
release/ArucoCrackMeasurement/
```

To run it on another PC:

1. Copy the entire `ArucoCrackMeasurement` folder to that PC.
2. Keep all files inside that folder together.
3. Run `ArucoCrackMeasurement.exe`.

Important:
- Do not copy only the `.exe` by itself.
- The executable depends on the bundled `_internal` folder that PyInstaller creates.

## Building The Windows Executable

In this project, "compiling" means packaging the Python app into a standalone Windows application using PyInstaller.

### Easiest method

If Python is already installed on the machine:

1. Open the project folder.
2. Double-click `build_windows.bat`.
3. Wait while it:
   - creates a local virtual environment if needed
   - installs the required build tools
   - packages the application
4. When it finishes, the built app will be in:

```text
release/ArucoCrackMeasurement/
```

### PowerShell method

From a PowerShell window opened in the project folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_windows.ps1
```

### What the build script does

The build script is meant to be as automatic as possible. It:

1. looks for `.venv`
2. creates `.venv` if it does not exist
3. installs dependencies from `requirements-dev.txt`
4. runs PyInstaller
5. writes the packaged application into `release/ArucoCrackMeasurement/`

### Manual build steps

If you ever want to do the build manually instead of using the script:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m PyInstaller --noconfirm --clean --windowed --name ArucoCrackMeasurement --distpath release --workpath build main\Aruco_crack_len_measurement.py
```

### Build prerequisites

- Windows
- Python 3 installed and available as `python` or `py`
- Internet access the first time dependencies are installed

### Build output

After a successful build, the important file is:

```text
release/ArucoCrackMeasurement/ArucoCrackMeasurement.exe
```

But again, distribute the whole `release/ArucoCrackMeasurement/` folder, not only that one file.

## Dependencies

Runtime dependencies:

```powershell
pip install -r requirements.txt
```

Development and packaging dependencies:

```powershell
pip install -r requirements-dev.txt
```

## Notes

- The app autosaves the current session to the user-specific Qt app-data folder so a crash does not lose day-long measurement work.
- The legacy script is kept for historical reference and is not the maintained application entry point.
- The legacy prototype is not part of the packaged application and may need its own original environment if you want to run it directly.

This project update and packaging workflow were prepared with Codex using the GPT-5.4 model.
