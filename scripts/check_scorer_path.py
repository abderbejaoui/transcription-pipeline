import sys
from pathlib import Path

# Ensure workspace root is importable when running this script directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from app.pipeline.scorer import _resolve_model_path

print(repr(_resolve_model_path()))
