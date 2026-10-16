import os
import time

from fdgx import gpulog
from fdgx.config import GpuLogCfg

# Excerpt of the log the DGX writes (processos.usando.GPU.log).
SAMPLE = """Buscando containers para os PIDs:
==============================================
PID 512648: Processo rodando no host (não containerizado)
PID=112372, Memoria=112372, 2120 MiB, Nome=/jupyter-gustavo.hining.1.67njwpmfusgakudks7dfy7t9w
PID=3908736, Memoria=3908736, 41862 MiB, Nome=/jupyter-arthur.leonida.1.icnza6nynxukzj4svoofoh7to
PID=3708875, Memoria=3708875, 30334 MiB, Nome=/jupyter-arthur.leonida.1.icnza6nynxukzj4svoofoh7to
PID=635155, Memoria=635155, 39042 MiB, Nome=/chat_ollama.1.fhmzsrgbvpcqbtupy5hfdj3i7
PID=2819124, Memoria=2819124, 23194 MiB, Nome=/chat_ollama.1.fhmzsrgbvpcqbtupy5hfdj3i7
==============================================
pid, process_name, used_gpu_memory [MiB]
512648, nvidia-cuda-mps-server, 62 MiB
3908736, /home/jovyan/compartilhado/ArthurLeonida/envs/dreamcatalyst_ns/bin/python3.9, 41806 MiB
"""


def test_parse_sample():
    entries = gpulog.parse(SAMPLE)
    assert len(entries) == 5  # host process and nvidia-smi dump lines are ignored
    assert entries[0].pid == 112372 and abs(entries[0].gb - 2120 / 1024) < 1e-9  # MiB from the 3rd field
    log = gpulog.GpuLog(entries)
    top = log.by_container()[0]
    assert top[0] == "jupyter-arthur.leonida.1.icnza6nynxukzj4svoofoh7to" and top[2] == 2
    assert abs(top[1] - (41862 + 30334) / 1024) < 1e-9


def test_read_missing_stale_and_encoding(tmp_path):
    p = tmp_path / "gpu.log"
    assert gpulog.read(GpuLogCfg(p)).error.startswith("cannot read")
    p.write_bytes(SAMPLE.encode("latin-1"))  # not UTF-8: still parsed
    assert len(gpulog.read(GpuLogCfg(p)).entries) == 5
    old = time.time() - 3600
    os.utime(p, (old, old))
    assert "stale" in gpulog.read(GpuLogCfg(p, max_age=600)).error


def test_ghost_warnings():
    log = gpulog.GpuLog(gpulog.parse(SAMPLE))
    assert gpulog.ghost_warnings(log, None, 0) == []
    assert gpulog.ghost_warnings(log, "fabio", 0) == []
    ((key, msg),) = gpulog.ghost_warnings(log, "arthur.leonida", 0)
    assert key.startswith("idle:") and "while fdgx runs no GPU job" in msg
    assert gpulog.ghost_warnings(log, "arthur.leonida", 1) == []
    two = gpulog.GpuLog(gpulog.parse(SAMPLE + "PID=1, Memoria=1, 1024 MiB, Nome=/jupyter-arthur.leonida.1.OLDCONTAINER\n"))
    ((key, msg),) = gpulog.ghost_warnings(two, "arthur.leonida", 1)
    assert key.startswith("multi:") and "ghost from a previous container" in msg
