from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from threading import Thread, Event
import subprocess
import os
import time
import re
import argparse
import sys
import signal
import json
import shlex

# ---------- Config & helpers ----------

STEP_PATTERNS = [
	(r"\[Step 0\]", "step_0_frames"),
	(r"\[Step 1\]", "step_1_landmarks"),
	(r"\[Step 2\]", "step_2_3dmm"),
	(r"\[Stage 1\]", "step_2_stage1"),
	(r"\[Stage 2\]", "step_2_stage2"),
	(r"\[Step 3\]", "step_3_stabilize"),
	(r"\[Step 4\]", "step_4_audio_mel"),
	(r"\[Step 5\]", "step_5_ref_enhance"),
	(r"\[Step 6\]", "step_6_lip_synth"),
]

NVSMI_QUERY = (
	"timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
)

NVSMI_CMD = ["nvidia-smi", f"--query-gpu={NVSMI_QUERY}", "--format=csv", "-l", "1"]


@dataclass
class StepWindow:
	name: str
	start_ts: float
	end_ts: Optional[float] = None

	def to_dict(self) -> Dict[str, float]:
		return {
			"name": self.name,
			"start_ts": self.start_ts,
			"end_ts": self.end_ts if self.end_ts is not None else -1.0,
			"duration_s": (self.end_ts - self.start_ts) if self.end_ts else -1.0,
		}


@dataclass
class GpuSample:
	ts: float
	gpu_index: int
	name: str
	util_gpu: float
	util_mem: float
	mem_used_mb: float
	mem_total_mb: float
	power_w: float


@dataclass
class ProfileResult:
	steps: List[StepWindow] = field(default_factory=list)
	gpu_samples: List[GpuSample] = field(default_factory=list)
	stdout_log_path: Optional[Path] = None
	gpu_log_path: Optional[Path] = None
	cmd: List[str] = field(default_factory=list)
	started_at: float = 0.0
	ended_at: float = 0.0

	def step_stats(self) -> Dict[str, Dict[str, float]]:
		# Aggregate by unique step name across potentially multiple windows
		by_name: Dict[str, List[StepWindow]] = {}
		for w in self.steps:
			if w.end_ts is None:
				continue
			by_name.setdefault(w.name, []).append(w)
		stats: Dict[str, Dict[str, float]] = {}
		for name, windows in by_name.items():
			duration = sum(w.end_ts - w.start_ts for w in windows)
			# Collect all GPU samples that fall within any of these windows
			rel_samples: List[GpuSample] = []
			for w in windows:
				rel_samples.extend([s for s in self.gpu_samples if w.start_ts <= s.ts <= w.end_ts])
			if not rel_samples:
				stats[name] = {
					"duration_s": duration,
					"gpu_util_mean": 0.0,
					"gpu_util_max": 0.0,
					"mem_used_mean_mb": 0.0,
					"mem_used_max_mb": 0.0,
				}
				continue
			gpu_mean = sum(s.util_gpu for s in rel_samples) / len(rel_samples)
			gpu_max = max(s.util_gpu for s in rel_samples)
			mem_mean = sum(s.mem_used_mb for s in rel_samples) / len(rel_samples)
			mem_max = max(s.mem_used_mb for s in rel_samples)
			stats[name] = {
				"duration_s": duration,
				"gpu_util_mean": gpu_mean,
				"gpu_util_max": gpu_max,
				"mem_used_mean_mb": mem_mean,
				"mem_used_max_mb": mem_max,
			}
		return stats


# ---------- Core logic ----------


def find_pair(default_dir: Path, stem: str) -> Tuple[Path, Path]:
	video = default_dir / f"{stem}.mp4"
	audio = default_dir / f"{stem}.wav"
	if not video.exists():
		raise FileNotFoundError(f"Video not found: {video}")
	if not audio.exists():
		raise FileNotFoundError(f"Audio not found: {audio}")
	return video.resolve(), audio.resolve()


def _gpu_reader_thread(proc: subprocess.Popen, log_path: Path, stop_evt: Event, out_list: List[GpuSample]) -> None:
	# Read CSV lines in real-time, attach wallclock ts, write-through to log file
	with open(log_path, "w") as f:
		f.write(f"# nv-smi sampling started at {datetime.now().isoformat()}\n")
		# Drain header if present
		header_seen = False
		while not stop_evt.is_set():
			line = proc.stdout.readline()
			if not line:
				if proc.poll() is not None:
					break
				continue
			line = line.strip()
			if not line:
				continue
			# Write raw line
			f.write(line + "\n")
			f.flush()
			if not header_seen and "timestamp" in line and "utilization.gpu" in line:
				header_seen = True
				continue
			# Parse metrics
			parts = [p.strip() for p in line.split(",")]
			if len(parts) < 8:
				continue
			try:
				# Wallclock timestamp now for correlation
				wts = time.time()
				gpu_index = int(parts[1])
				name = parts[2]
				util_gpu = float(re.sub("[^0-9.]", "", parts[3]))
				util_mem = float(re.sub("[^0-9.]", "", parts[4]))
				mem_used = float(re.sub("[^0-9.]", "", parts[5]))
				mem_total = float(re.sub("[^0-9.]", "", parts[6]))
				power_w = float(re.sub("[^0-9.]", "", parts[7])) if parts[7].strip() else 0.0
				out_list.append(GpuSample(wts, gpu_index, name, util_gpu, util_mem, mem_used, mem_total, power_w))
			except Exception:
				continue


def start_nvidia_smi_stream(log_path: Path) -> Tuple[subprocess.Popen, Thread, Event, List[GpuSample]]:
	proc = subprocess.Popen(
		NVSMI_CMD,
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		text=True,
		bufsize=1,
		preexec_fn=os.setsid,
	)
	stop_evt = Event()
	out_list: List[GpuSample] = []
	thr = Thread(target=_gpu_reader_thread, args=(proc, log_path, stop_evt, out_list), daemon=True)
	thr.start()
	return proc, thr, stop_evt, out_list


def build_inference_cmd(uv_python: Path, inference_py: Path, face: Path, audio: Path, outfile: Path, extra_args: List[str]) -> List[str]:
	cmd = [
		str(uv_python), str(inference_py),
		"--face", str(face),
		"--audio", str(audio),
		"--outfile", str(outfile),
		# Note: --re_preprocess removed to test cached performance
	]
	cmd.extend(extra_args)
	return cmd


def stream_with_timestamps(proc: subprocess.Popen, log_path: Path, step_windows: List[StepWindow]) -> None:
	pattern_map = [(re.compile(pat), name) for pat, name in STEP_PATTERNS]
	current_open: Optional[StepWindow] = None
	current_name: Optional[str] = None
	with open(log_path, "w") as fout:
		fout.write(f"# stdout capture started at {datetime.now().isoformat()}\n")
		while True:
			line = proc.stdout.readline()
			if not line and proc.poll() is not None:
				break
			if not line:
				time.sleep(0.01)
				continue
			line = line.rstrip("\n")
			ts = time.time()
			fout.write(f"{ts:.6f} {line}\n")
			fout.flush()
			for cre, name in pattern_map:
				if cre.search(line):
					# Ignore repeated matches of the same step (tqdm refreshes)
					if current_name == name:
						break
					# Close previous step window if open and different
					if current_open is not None and current_open.end_ts is None:
						current_open.end_ts = ts
					# Start new window
					current_open = StepWindow(name=name, start_ts=ts)
					current_name = name
					step_windows.append(current_open)
					break
		# Close last window if still open
		if current_open is not None and current_open.end_ts is None:
			current_open.end_ts = time.time()


def readable_report(profile: ProfileResult) -> str:
	lines: List[str] = []
	lines.append("VideoReTalking Inference Profiling Report")
	lines.append(f"Generated: {datetime.now().isoformat()}")
	lines.append("")
	lines.append("Command:")
	lines.append(" ".join(shlex.quote(c) for c in profile.cmd))
	lines.append("")
	lines.append(f"Stdout log: {profile.stdout_log_path}")
	lines.append(f"GPU log: {profile.gpu_log_path}")
	lines.append("")
	lines.append(f"Total wall time: {profile.ended_at - profile.started_at:.2f}s")
	lines.append("")
	stats = profile.step_stats()
	lines.append("Per-step timings and GPU metrics:")
	for name, st in stats.items():
		lines.append(
			f"- {name}: duration={st.get('duration_s', 0.0):.2f}s, "
			f"gpu_mean={st.get('gpu_util_mean', 0.0):.1f}%, gpu_max={st.get('gpu_util_max', 0.0):.1f}%, "
			f"mem_mean={st.get('mem_used_mean_mb', 0.0):.0f}MB, mem_max={st.get('mem_used_max_mb', 0.0):.0f}MB"
		)
	return "\n".join(lines)


# ---------- CLI ----------


def main() -> int:
	parser = argparse.ArgumentParser(description="Profile VideoReTalking inference with uv and GPU metrics")
	parser.add_argument("--stem", type=str, default="segment_0045_full_front", help="Test clip stem name without extension")
	parser.add_argument("--clips_dir", type=Path, default=Path("test_clips"), help="Directory containing test clips")
	parser.add_argument("--results_dir", type=Path, default=Path("test_clips/results_uv"), help="Directory to store outputs")
	parser.add_argument("--extra_args", type=str, nargs=argparse.REMAINDER, default=[], help="Extra args to pass to inference.py after --")
	args, unknown = parser.parse_known_args()

	clips_dir: Path = args.clips_dir.resolve()
	results_dir: Path = args.results_dir.resolve()
	results_dir.mkdir(parents=True, exist_ok=True)

	video, audio = find_pair(clips_dir, args.stem)
	outfile = results_dir / f"{args.stem}_result_profiled.mp4"
	stdout_log = results_dir / f"{args.stem}_stdout.log"
	gpu_log = results_dir / f"{args.stem}_nvidia_smi.csv"
	report_txt = results_dir / f"{args.stem}_profile.txt"
	report_json = results_dir / f"{args.stem}_profile.json"

	uv_python = Path.cwd() / ".venv" / "bin" / "python"
	inference_py = Path.cwd() / "inference.py"
	if not uv_python.exists():
		raise RuntimeError(f"uv virtualenv python not found: {uv_python}")
	if not inference_py.exists():
		raise RuntimeError(f"inference.py not found: {inference_py}")

	# Handle optional extra args; accept either --extra_args ... or unknowns after --
	extra_args: List[str] = []
	if getattr(args, 'extra_args', None):
		if len(args.extra_args) >= 1 and args.extra_args[0] == "--":
			extra_args.extend(args.extra_args[1:])
		else:
			extra_args.extend(args.extra_args)
	if unknown:
		if unknown[0] == "--":
			unknown = unknown[1:]
		extra_args.extend(unknown)

	cmd = build_inference_cmd(uv_python, inference_py, video, audio, outfile, extra_args)

	# Start GPU sampler (streaming)
	gpu_proc, gpu_thr, gpu_stop, gpu_samples_live = start_nvidia_smi_stream(gpu_log)

	# Launch inference
	proc = subprocess.Popen(
		cmd,
		cwd=Path.cwd(),
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		bufsize=1,
		text=True,
	)

	profile = ProfileResult(stdout_log_path=stdout_log, gpu_log_path=gpu_log, cmd=cmd)
	profile.started_at = time.time()
	step_windows: List[StepWindow] = []

	try:
		stream_with_timestamps(proc, stdout_log, step_windows)
		ret = proc.wait()
	finally:
		# Stop GPU sampler
		try:
			gpu_stop.set()
			time.sleep(0.2)
			os.killpg(os.getpgid(gpu_proc.pid), signal.SIGTERM)
		except Exception:
			pass
		try:
			gpu_thr.join(timeout=2)
		except Exception:
			pass

	profile.ended_at = time.time()
	profile.steps = step_windows
	profile.gpu_samples = list(gpu_samples_live)

	# Write reports
	report_text = readable_report(profile)
	report_txt.write_text(report_text)

	# JSON
	json_out = {
		"generated_at": datetime.now().isoformat(),
		"cmd": profile.cmd,
		"stdout_log": str(profile.stdout_log_path) if profile.stdout_log_path else None,
		"gpu_log": str(profile.gpu_log_path) if profile.gpu_log_path else None,
		"started_at": profile.started_at,
		"ended_at": profile.ended_at,
		"total_duration_s": profile.ended_at - profile.started_at,
		"steps": [s.to_dict() for s in profile.steps],
		"per_step_stats": profile.step_stats(),
	}
	report_json.write_text(json.dumps(json_out, indent=2))

	print(report_text)
	print(f"\nReport written to: {report_txt}")
	print(f"JSON written to: {report_json}")
	return 0


if __name__ == "__main__":
	sys.exit(main())

