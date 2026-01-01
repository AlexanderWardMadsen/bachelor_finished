from pathlib import Path
import urllib.request

class PoseModel:
    def __init__(self, model_path: Path, model_url: str):
        self.model_path = model_path#################
        self.model_url = model_url###################
        self._ensure_model()

    def _ensure_model(self):
        if not self.model_path.exists():
            print(f"[INFO] Downloading model to {self.model_path} ...")
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(self.model_url, str(self.model_path))
            print("[INFO] Download complete.")

    def path(self) -> str:
        return str(self.model_path)
