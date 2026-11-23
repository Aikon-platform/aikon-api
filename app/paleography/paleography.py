import gc
from typing import Optional

from .lib.entrypoint import M4MEntrypoint
from ..shared.tasks import LoggedTask
from ..shared.dataset import Dataset
import shutil
from .const import DEMO_NAME, PALEO_RESULTS_PATH


class Paleography(LoggedTask):
    """
    Task running the M4M trainer on a dataset.
    Unlike region_extraction (image by image), M4M steps 0/1 work on the whole
    dataset and step 2 per sub-dataset (folio): run_task is dataset-level.
    """

    def __init__(self, dataset: Dataset, params: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.params = params or {}
        self.results_url = []
        self.entrypoint = None

    def initialize(self):
        # task logger (JobLogger): epoch progress -> /status -> front progress tracker
        self.entrypoint = M4MEntrypoint(self.params, logger=getattr(self, "jlogger", None) or getattr(self, "logger", None))

    def _prepare_d1_input(self):
        """Doit produire <dir>/<folio>/<ligne>.png + .txt (arborescence du zip D1).
        MODE TEST: params['d1_dir'] = dossier D1 deja pret.
        Sinon: un seul document zip, telecharge ici (comme region_extraction/vectorization),
        dont images_path contient les .txt grace a ALLOWED_EXTENSIONS."""
        d1_dir = self.params.get("d1_dir")
        if d1_dir:
            return d1_dir
        docs = self.dataset.documents or []
        if len(docs) != 1 or docs[0].dtype != "zip":
            raise ValueError(
                f"entree D1: un seul document de type zip attendu, recu {[(d.dtype, d.uid) for d in docs]}"
            )  # regle pour plusieurs documents/folios. [A CONFIRMER] Mathieu
        doc = docs[0]
        doc.download()  # sans effet si deja present (Document.sanity_check)
        d1_dir = doc.images_path
        if not any(d1_dir.rglob("*.txt")):
            raise ValueError(
                f"entree D1: aucune transcription .txt dans {d1_dir}. Le zip doit contenir "
                "<folio>/<ligne>.png + .txt. Si ce document a ete telecharge avant l'ajout de .txt "
                "dans ALLOWED_EXTENSIONS, supprimer son dossier pour forcer un nouveau telechargement."
            )
        return d1_dir

    STEPS = ("convert", "0", "1", "2", "export")

    def _get_steps(self):
        """Etapes a lancer, dans l'ordre canonique. Par defaut: toutes."""
        steps = self.params.get("steps")
        if steps is None:
            return list(self.STEPS)
        if not isinstance(steps, (list, tuple)) or not steps:
            raise ValueError("params['steps'] doit etre une liste non vide parmi " + str(list(self.STEPS)))
        wanted = {str(x) for x in steps}
        unknown = wanted - set(self.STEPS)
        if unknown:
            raise ValueError(f"params['steps']: etapes inconnues {sorted(unknown)}, valides: {list(self.STEPS)}")
        return [x for x in self.STEPS if x in wanted]

    def _run_chain(self):
        import json, subprocess, sys
        from pathlib import Path
        from .const import M4M_PATH, CONVERTER_PATH, M4M_CHECKPOINT, PALEO_LOGS_PATH

        ep = self.entrypoint
        params = self.params
        steps = self._get_steps()
        if "convert" not in steps and not params.get("data_folder"):
            raise ValueError("params['data_folder'] requis si 'convert' n'est pas dans steps (dataset deja converti)")
        # --out du converter ne doit pas exister: dossier neuf par experience
        ep.data_folder = params.get("data_folder") or f"paleo_{self.experiment_id}"
        ep.run_id = params.get("run_id") or ep.data_folder
        converter = Path(params.get("converter_path") or CONVERTER_PATH)  # defaut: converter du module
        if "convert" in steps and not converter.is_file():
            raise ValueError(f"converter introuvable: {converter}")
        checkpoint = Path(params.get("checkpoint") or M4M_CHECKPOINT)  # defaut: checkpoint HF (const.py)
        if "0" in steps and not checkpoint.is_file():
            raise ValueError(f"checkpoint introuvable: {checkpoint}")

        # M4M writes to logs_reconstruction/<tag> (cwd = M4M_PATH): keep outputs in the data folder
        logs_link = Path(M4M_PATH) / "logs_reconstruction"
        if not logs_link.exists() and not logs_link.is_symlink():
            logs_link.symlink_to(PALEO_LOGS_PATH, target_is_directory=True)

        cfg_path = Path(M4M_PATH) / "datasets" / "config.json"
        if not cfg_path.exists():  # fichier gitignore cote M4M, propre a chaque deploiement
            from .const import PALEO_DATA_FOLDER
            cfg_path.write_text(json.dumps({"datasets_path": str(Path(PALEO_DATA_FOLDER) / "datasets")}))
        cfg = json.loads(cfg_path.read_text())
        out = Path(cfg["datasets_path"]) / ep.data_folder
        annotation = out / "annotation.json"

        if "convert" in steps:
            d1_dir = self._prepare_d1_input()
            cmd = [sys.executable, str(converter), "--d1", str(d1_dir), "--out", str(out),
                   "--val-ratio", "0"]  # CER complet (CR 08/10)
            if params.get("script"):
                cmd += ["--script", str(params["script"])]
            if params.get("script_map"):
                cmd += ["--script-map", str(params["script_map"])]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:  # codes 1 et 2 = erreur (arbitrage 9). [A CONFIRMER]
                raise RuntimeError(f"converter: code retour {r.returncode}: {r.stderr.strip()[-500:]}")
        elif not out.is_dir():
            raise ValueError(f"dataset converti introuvable: {out}")

        if "2" in steps and not annotation.exists():
            raise ValueError(f"annotation.json introuvable: {annotation}")

        # NB: les steps 1 et 2 supposent les sorties des steps precedents (meme run_id)
        if "0" in steps:
            ep.run_step(0, checkpoint=str(checkpoint))
        if "1" in steps:
            ep.run_step(1)
        if "2" in steps:
            ep.run_step(2, annotation_file=str(annotation))
        return ep.export() if "export" in steps else None

    def store_results(self, src) -> list:
        """Zip paleography_input as <doc.uid>.zip; one entry per document (M4M runs on the whole dataset)."""
        PALEO_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        entries, first_zip = [], None
        for doc in self.dataset.documents:
            zip_path = PALEO_RESULTS_PATH / f"{doc.uid}.zip"
            if first_zip is None:
                shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=str(src))
                first_zip = zip_path
            else:
                shutil.copy2(first_zip, zip_path)
            entries.append({"doc_id": doc.uid, "result_url": doc.get_results_url(DEMO_NAME)})
        return entries

    def terminate(self):
        self.entrypoint = None
        gc.collect()

    def check_doc(self) -> bool:
        return bool(self.dataset.documents)

    def run_task(self) -> bool:
        if not self.check_doc():
            self.print_and_log_warning("[task.paleography] No dataset to process")
            self.task_update(
                "ERROR",
                message=f"[API ERROR] No documents in {self.dataset}",
                exception=Exception("No documents to process"),
            )
            return False

        self.task_update("STARTED")
        try:
            self.initialize()
            result = self._run_chain()
            if result:
                self.results_url.extend(self.store_results(result))
            return True
        except Exception as e:
            self.task_update(
                "ERROR",
                message=[f"Error in paleography task: {e}"] + self.error_list,
                exception=e,
            )
            return False
        finally:
            self.terminate()  # no `return` here, so errors reach @notifying
