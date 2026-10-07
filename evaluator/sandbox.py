"""Run contributed code as an unprivileged account that cannot touch what the evaluator protects.

Contributed code (a quantizer PR) runs only in four steps, all on the CPU:

    manifest --ids-out   expand the recipe (imports the new quantizer)
    fingerprint          encode the seeded synthetic probe
    build                write the checkpoint
    regenerate           rebuild the audit's secret samples from the sources, with the built
                         checkpoint already sealed away

Everything that judges -- audit comparison, quality, speed, tasks, holdout, ranking -- runs as the
evaluator from trusted `main` code, on the checkpoint as data.

Each step runs through `runuser` with an allow-listed environment and no GPU, in its own session;
afterwards every process the account owns is killed. `problems()` refuses to run anything when the
account can read a secret or write evaluator state. Set it up once with evaluator/setup_sandbox.sh.

Between steps everything the account could have stashed (its home, /tmp, /var/tmp, /dev/shm) is
wiped, and `problems()` requires the account's outbound network to be blocked
(evaluator/setup_sandbox.sh adds the firewall rule): otherwise a build could upload its checkpoint and
the regeneration step download it back. The account still shares the machine's kernel.
"""

from __future__ import annotations

import os
import pwd
import signal
import subprocess
from pathlib import Path

from bittrellis import BLAS_VARS, blas_threads

DEFAULT_USER = "bt-sandbox"
SCRATCH = ("/tmp", "/var/tmp", "/dev/shm")
NET_PROBE = ("import socket\ntry:\n    socket.create_connection(('1.1.1.1', 443), timeout=3)\n    print('reachable')\n"
             "except OSError:\n    print('blocked')")
ENV_ALLOW = ("LANG", "LC_ALL", "TZ")


def command(user: str, cmd: list[str], home: str, path: str, extra_env: dict[str, str] | None = None) -> list[str]:
    env = {k: os.environ[k] for k in ENV_ALLOW if k in os.environ}
    env.update({"HOME": home, "PATH": path, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
                "BITTRELLIS_REPLAY_CACHE": f"{home}/.cache/bittrellis/replay",
                # every sandbox step is CPU-only; a build runs build_jobs() encoders at once (see blas_threads)
                **{v: str(blas_threads()) for v in BLAS_VARS}, **(extra_env or {})})
    return ["runuser", "-u", user, "--", "env", "-i", *[f"{k}={v}" for k, v in sorted(env.items())], *cmd]


class Sandbox:
    def __init__(self, user: str = DEFAULT_USER):
        self.user = user
        entry = pwd.getpwnam(user)
        self.uid, self.gid, self.home = entry.pw_uid, entry.pw_gid, entry.pw_dir

    def _as_user(self, *test: str) -> bool:
        return subprocess.run(["runuser", "-u", self.user, "--", *test], capture_output=True).returncode == 0

    def problems(self, secrets: list[Path], protected_dirs: list[Path], readable: list[Path]) -> list[str]:
        """Reasons it is unsafe to run contributed code now; empty when it is safe."""
        out = []
        if os.geteuid() != 0:
            out.append("the evaluator must run as root to switch to the sandbox account")
            return out
        if self.uid == 0:
            out.append(f"sandbox account {self.user} is root")
        for p in secrets:
            if p.exists() and self._as_user("test", "-r", str(p)):
                out.append(f"{self.user} can read {p}")
        for d in protected_dirs:
            if d.exists() and (self._as_user("test", "-w", str(d)) or self._as_user("test", "-r", str(d))):
                out.append(f"{self.user} can read or write {d}")
        for p in readable:
            if not self._as_user("test", "-r", str(p)):
                out.append(f"{self.user} cannot read {p} (the build needs it)")
        probe = subprocess.run(["runuser", "-u", self.user, "--", "python3", "-c", NET_PROBE], capture_output=True, text=True, timeout=20)
        if "blocked" not in probe.stdout:  # fail closed: a probe that did not run proves nothing
            out.append(f"{self.user} can reach the network, or the check could not run "
                       "(block its outbound traffic: evaluator/setup_sandbox.sh)")
        return out

    def own(self, path: Path) -> None:
        """Give the sandbox account a directory tree it may write."""
        for root, dirs, files in os.walk(path):
            for n in [root, *(os.path.join(root, d) for d in dirs), *(os.path.join(root, f) for f in files)]:
                os.lchown(n, self.uid, self.gid)

    def seal(self, src: Path, dst: Path) -> None:
        """Move a tree the sandbox wrote to a place only the evaluator can read, owned by the evaluator."""
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(dst.parent, 0o700)
        os.rename(src, dst)
        for root, dirs, files in os.walk(dst):
            for n in [root, *(os.path.join(root, d) for d in dirs), *(os.path.join(root, f) for f in files)]:
                os.lchown(n, 0, 0)

    def kill_all(self) -> None:
        subprocess.run(["pkill", "-KILL", "-u", self.user], capture_output=True)

    def wipe(self) -> None:
        """Delete everything the account owns in the places it can write outside a PR's untrusted/ tree."""
        for d in (self.home, *SCRATCH):
            if os.path.isdir(d):
                subprocess.run(["find", d, "-mindepth", "1", "-user", self.user, "-delete"], capture_output=True)

    def run(self, cmd: list[str], cwd: Path, log: Path, python_bin: str, timeout: int = 6 * 3600) -> int:
        full = command(self.user, cmd, self.home, f"{python_bin}:/usr/local/bin:/usr/bin:/bin")
        with open(log, "a") as fh:
            fh.write(f"\n$ [sandbox:{self.user}] {' '.join(cmd)}\n")
            fh.flush()
            proc = subprocess.Popen(full, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                return proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                fh.write(f"[sandbox] timed out after {timeout}s\n")
                return 124
            finally:
                self.kill_all()  # nothing the step started may outlive it
                self.wipe()      # nor anything it stashed for a later step
