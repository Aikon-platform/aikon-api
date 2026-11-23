"""Pilote M4M en sous-processus (arbitrage 6: "comme on peut" -> subprocess, [A CONFIRMER] Paul).

Sources: scripts/btv1b84472995/line_step_{0,1,2}.sh (Raphael, 37fea71) + reconstruction.py l.311-316.
- Lancer depuis la racine M4M (imports absolus, chemins relatifs datasets/*.json).
- --tag impose le dossier de sortie: logs_reconstruction/<tag> (ecrase --output_dir).
- --wandb jamais passe. --model_config_path et --loss passes a chaque step.
"""
import os
import shutil
import sys
import subprocess
from pathlib import Path

from ..const import M4M_PATH

DEFAULTS = {
    0: dict(max_e=20, batch_size=8),
    1: dict(max_e=100, batch_size=8, learning_rate="1e-4", weight_loss_reconstruction=3),
    2: dict(max_e=140, batch_size=8, learning_rate="1e-2"),
}
MODEL_CONFIG = "config/Latin_accent.py"
LOSS = "L1"
DEFAULT_SEED = 42  # same default as M4M main_synthetic.py; params["seed"]=None disables it


class M4MError(RuntimeError):
    pass


class M4MEntrypoint:
    def __init__(self, params=None, m4m_path=M4M_PATH, python_bin=None, logger=None):
        self.params = params or {}
        self.m4m_path = Path(m4m_path)
        # M4M runs with the API interpreter (single uv env, decided 19/11)
        self.python_bin = python_bin or sys.executable
        self.logger = logger
        self.data_folder = self.params.get("data_folder")  # [A CONFIRMER] resolution via datasets/config.json
        self.run_id = self.params.get("run_id", "paleo")

    def tag(self, step):
        return f"{self.run_id}_s{step}"

    def out_dir(self, step):
        return f"logs_reconstruction/{self.tag(step)}"

    def build_cmd(self, step, **kwargs):
        p = {**DEFAULTS[step], **self.params.get(f"step{step}", {}), **kwargs}
        max_e = int(p["max_e"])
        if max_e < 1:
            raise M4MError(f"step {step}: max_e doit etre >= 1 (recu {max_e})")
        cmd = [self.python_bin, "reconstruction.py",
               "--dataset_file", "dataset",
               "--data_folder", str(self.data_folder),
               "--space_index", "0",
               "--model_config_path", MODEL_CONFIG,
               "--num_fine_classes", "2",
               "--step", str(step),
               "--loss", LOSS,
               "--tag", self.tag(step),
               "--max_e", str(max_e),
               "--batch_size", str(p["batch_size"])]
        seed = self.params.get("seed", DEFAULT_SEED)
        if seed is not None:
            cmd += ["--seed", str(int(seed))]
        if step == 0:
            cmd += ["--init", "--model_checkpoint_path", str(p["checkpoint"])]  # checkpoint HF obligatoire
        elif step == 1:
            s0 = self.out_dir(0)
            cmd += ["--learning_rate", str(p["learning_rate"]),
                    "--weight_loss_reconstruction", str(p["weight_loss_reconstruction"]),
                    "--model_checkpoint_path", f"{s0}/model.pth",
                    "--reconstructor_path", f"{s0}/reconstructor.pth"]
        elif step == 2:
            s1 = self.out_dir(1)
            cmd += ["--documents", "--mask_sprite",
                    "--learning_rate", str(p["learning_rate"]),
                    "--prototypes_only_path", f"{s1}/reconstructor_unfrozen.pth",
                    "--model_checkpoint_path", f"{s1}/model.pth",
                    "--annotation_file", str(p["annotation_file"])]
        return cmd

    def _env(self):
        """Environment of every M4M subprocess: honours params["cuda_device"]."""
        env = {**os.environ}
        if "cuda_device" in self.params:
            env["CUDA_VISIBLE_DEVICES"] = str(self.params["cuda_device"])
        return env

    def run_step(self, step, **kwargs):
        cmd = self.build_cmd(step, **kwargs)
        env = self._env()
        if self.logger:
            self.logger.info(" ".join(cmd))
        r = subprocess.run(cmd, cwd=self.m4m_path, env=env)
        if r.returncode != 0:
            raise M4MError(f"step {step} a echoue (code {r.returncode})")
        return self.out_dir(step)

    def export(self, **kwargs):
        """export_paleography_input.py (--help vu). Sortie: <step1_dir>/paleography_input."""
        cmd = [self.python_bin, "export_paleography_input.py",
               "--step1_dir", self.out_dir(1),
               "--finetune_dir", self.out_dir(2),
               "--data_folder", str(self.data_folder),
               "--dataset_file", "dataset",
               "--model_config_path", MODEL_CONFIG,
               "--num_fine_classes", "2"]
        # charset.json: defaut = racine M4M, origine non verifiee. [A CONFIRMER] Raphael
        if self.params.get("charset"):
            cmd += ["--charset", str(self.params["charset"])]
        if self.params.get("annotation_file"):
            cmd += ["--annotation_file", str(self.params["annotation_file"])]
        r = subprocess.run(cmd, cwd=self.m4m_path, env=self._env())  # export runs predict.py on the GPU too
        if r.returncode != 0:
            raise M4MError(f"export a echoue (code {r.returncode})")
        out = self.m4m_path / self.out_dir(1) / "paleography_input"
        problems = []
        # CER global and per sub-folder (CR 15/10), shipped in the results zip under cer/
        predict = self.m4m_path / self.out_dir(2) / "predict"
        summaries = sorted(predict.glob("*/document_summary.json")) if predict.is_dir() else []
        if (predict / "global_summary.json").is_file():
            summaries.insert(0, predict / "global_summary.json")
        else:
            problems.append(f"no global_summary.json in {predict}")
        for f in summaries:
            dst = out / "cer" / f.relative_to(predict)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
        protos = out / "prototypes"
        proto_dirs = [d for d in protos.iterdir() if d.is_dir()] if protos.is_dir() else []
        if not proto_dirs:
            problems.append("no folder in prototypes/")
        problems += [f"prototypes/{d.name} is empty" for d in proto_dirs if not any(d.iterdir())]
        meas = out / "characters_measurements"
        if not (meas.is_dir() and any(meas.rglob("*.json"))):
            problems.append("no JSON in characters_measurements/")
        if problems:  # export exits 0 with warnings only: do not report an empty result as SUCCESS
            raise M4MError("export incomplet: " + "; ".join(problems))
        return out
