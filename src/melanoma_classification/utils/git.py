import subprocess


def get_git_commit_hash() -> str | None:
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        return commit_hash
    except subprocess.CalledProcessError:
        return None


def git_changes_detected() -> bool:
    try:
        status = (
            subprocess.check_output(["git", "status", "--porcelain"])
            .strip()
            .decode("utf-8")
        )
        return len(status) > 0
    except subprocess.CalledProcessError:
        return False
