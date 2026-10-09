"""One-time installer of pinned upstream Iris source into THIS node folder.
Run using ComfyUI portable Python. No system-wide edits or model downloads.
"""
from pathlib import Path
from urllib.request import urlopen
import io
import zipfile
import shutil

ROOT = Path(__file__).resolve().parent
COMMIT = "a8d15239dea469aba042cfa56ca3bb4e450d5ebc"
URL = f"https://github.com/speridlabs/iris-3b/archive/{COMMIT}.zip"
TARGET = ROOT / "third_party" / "iris-3b" / "src"


def main():
    if (TARGET / "iris3b" / "config.py").is_file():
        print("Iris source already installed:", TARGET)
        return
    print("Downloading pinned upstream Iris source:", URL)
    with urlopen(URL, timeout=90) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        wanted = [n for n in archive.namelist() if "/src/iris3b/" in n and not n.endswith("/")]
        if not wanted:
            raise RuntimeError("Expected src/iris3b package not present in upstream archive")
        for name in wanted:
            rel = name.split("/src/", 1)[1]
            out = TARGET / rel
            if not out.resolve().is_relative_to(TARGET.resolve()):
                raise RuntimeError("Unsafe ZIP path " + name)
            out.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as f, out.open("wb") as g:
                shutil.copyfileobj(f, g)
    assert (TARGET / "iris3b" / "config.py").is_file()
    print("Installed source into:", TARGET)

if __name__ == "__main__":
    main()
