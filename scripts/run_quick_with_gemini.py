"""Run the quick pipeline with GEMINI env loaded from .env in-process.

This avoids shell quoting issues and ensures the Python process has the
correct GEMINI_API_KEY and GEMINI_MODEL environment variables set.
"""
from __future__ import annotations
import os
from pathlib import Path


def load_dotenv(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            continue
        k, v = line.split('=', 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def main():
    root = Path(__file__).resolve().parents[1]
    dotenv = root / '.env'
    load_dotenv(dotenv)

    # Ensure model selection for the quick pipeline
    os.environ.setdefault('GEMINI_MODEL', os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash'))

    # Run the quick pipeline script main in-process
    import importlib
    mod = importlib.import_module('scripts.run_quick_pipeline')
    if hasattr(mod, 'main'):
        mod.main()
    else:
        # fallback: execute as script
        import runpy
        runpy.run_path(str(root / 'scripts' / 'run_quick_pipeline.py'), run_name='__main__')


if __name__ == '__main__':
    main()
