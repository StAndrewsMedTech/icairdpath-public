from pathlib import Path


def project_root() -> Path:
    return Path(__file__).parent.parent.parent

def results_root(name: str) -> Path:
    return project_root() / 'results' / name
