from pathlib import Path
import runpy


if __name__ == "__main__":
    source_script = Path(__file__).with_name("6g_wave_physical_ai.py")
    runpy.run_path(str(source_script), run_name="__main__")
