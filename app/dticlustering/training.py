"""
Training tools to adapt DTI research lib to the API
"""
import json
import sys
import traceback

from hydra.core.hydra_config import HydraConfig

from yaml import load, Loader, dump, Dumper
from pathlib import Path
import os, torch
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from sklearn.metrics.cluster import normalized_mutual_info_score
from omegaconf import OmegaConf

from .lib.src.dataset import get_dataset
from .lib.src.kmeans_trainer import Trainer as KMeansTrainer
from .lib.src.sprites_trainer import Trainer as SpritesTrainer

# from .lib.src._kmeans_trainer import Trainer as KMeansTrainer
# from .lib.src._sprites_trainer import Trainer as SpritesTrainer
from .const import RUNS_PATH, CONFIGS_PATH
from .lib.src.utils.image import convert_to_img

from ..shared.utils.logging import TLogger, LoggerHelper, serializer

TEMPLATES_DIR = Path(__file__).parent / "templates"
KMEANS_CONFIG_FILE = TEMPLATES_DIR / "kmeans-conf.yml"
SPRITES_CONFIG_FILE = TEMPLATES_DIR / "sprites-conf.yml"
BASE_CONFIG_FILE = TEMPLATES_DIR / "base-conf.yml"


class LoggingTrainerMixin:
    """
    A mixin with hooks to track training progress inside dti Trainers
    """

    output_proto_dir: str = "prototypes"

    def __init__(self, logger: TLogger, *args, **kwargs):
        self.jlogger = logger
        super().__init__(*args, **kwargs)

    def setup_logging(self):
        self.logger = self.jlogger
        self.print_and_log_info(
            f"Trainer initialisation: run directory is {self.run_dir}"
        )

    def print_and_log_info(self, string: str) -> None:
        self.jlogger.info(string)
        # self.logger.info(string)

    def run(self, *args, **kwargs):
        # Log epoch progress start
        self.jlogger.progress(
            self.start_epoch - 1, self.n_epochs, title="Training epoch"
        )

        return super().run(*args, **kwargs)

    def update_scheduler(self, epoch, batch):
        # Log epoch progress
        self.jlogger.progress(epoch - 1, self.n_epochs, title="Training epoch")

        return super().update_scheduler(epoch, batch)

    def log_end(self):
        self.jlogger.progress(
            self.n_epochs, self.n_epochs, title="Training epoch", end=True
        )
        self.jlogger.info("Training over, running evaluation")

    def evaluate_folder_to_cluster_mapping(self, cluster_by_path_df):
        """
        Evaluate how well the clustering aligns with the folder structure.
        """
        if cluster_by_path_df.empty:
            return {}

        # Extract folder from path (part before the last '+' in name)
        def extract_folder(path):
            if "+" in path:
                return "+".join(path.split("+")[:-1])
            return "default"  # Fallback for paths without '+'

        cluster_by_path_df["folder"] = cluster_by_path_df["path"].apply(extract_folder)

        folders = cluster_by_path_df["folder"].unique()
        n_folders = len(folders)

        if n_folders <= 1:
            return {}

        if n_folders != self.n_prototypes:
            self.print_and_log_info(
                f"Number of folders ({n_folders}) doesn't match number of clusters ({self.n_prototypes})"
            )

        # Group by folder and cluster
        folder_counts = cluster_by_path_df.groupby("folder").size()
        folder_to_cluster = (
            cluster_by_path_df.groupby(["folder", "cluster_id"])
            .size()
            .unstack(fill_value=0)
        )

        total_images = len(cluster_by_path_df)
        correct_assignments = 0

        # For each folder, find the cluster with the most images from that folder
        folder_to_best_cluster = {}
        for folder in folders:
            if folder not in folder_to_cluster.index:
                continue

            clusters_for_folder = folder_to_cluster.loc[folder]
            best_cluster = clusters_for_folder.idxmax()
            max_count = clusters_for_folder.max()

            folder_to_best_cluster[folder] = (best_cluster, max_count)
            correct_assignments += max_count

        purity = correct_assignments / total_images if total_images > 0 else 0

        folder_labels = cluster_by_path_df["folder"].map(
            {f: i for i, f in enumerate(folders)}
        )
        nmi = normalized_mutual_info_score(
            folder_labels, cluster_by_path_df["cluster_id"]
        )

        # confusion_matrix = folder_to_cluster.copy()

        # hungarian matching

        metrics_by_folder = {}
        for folder in folders:
            if folder not in folder_to_cluster.index:
                continue

            best_cluster, true_positives = folder_to_best_cluster.get(folder, (None, 0))
            if best_cluster is None:
                continue

            false_positives = folder_to_cluster[best_cluster].sum() - true_positives
            false_negatives = folder_counts[folder] - true_positives
            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else 0
            )
            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0
            )

            metrics_by_folder[folder] = {
                "best_cluster": best_cluster,
                "precision": precision,
                "recall": recall,
                "images_in_folder": folder_counts[folder],
                "images_in_best_cluster": true_positives,
            }
            self.print_and_log_info(
                f"Folder {folder} ({folder_counts[folder]} imgs) best match cluster {best_cluster} ({true_positives} true positive): "
                f"Precision: {precision:.2f}, Recall: {recall:.2f}"
            )

        avg_precision, avg_recall = 0, 0
        if nb_metrics := len(metrics_by_folder):
            avg_precision = (
                sum(m["precision"] for m in metrics_by_folder.values()) / nb_metrics
            )
            avg_recall = (
                sum(m["recall"] for m in metrics_by_folder.values()) / nb_metrics
            )
        self.print_and_log_info(
            f"Average Precision: {avg_precision:.2f}, Average Recall: {avg_recall:.2f}"
        )
        self.print_and_log_info(f"Purity: {purity:.2f}, NMI: {nmi:.2f}")

        # Purity: percentage of images correctly assigned to their best matching cluster
        # NMI: measure of mutual dependence between folder and cluster assignments
        # Precision: of all images in a cluster, how many are from the best matching folder
        # Recall: of all images in a folder, how many are in the best matching cluster
        results = {
            "dataset": self.dataset_name,
            "purity": purity,
            "nmi": nmi,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "n_folders": n_folders,
            "n_clusters": self.n_prototypes,
            "folder_metrics": metrics_by_folder,
            # 'confusion_matrix': confusion_matrix.to_dict(),
        }

        folder_metrics_path = self.run_dir / "folder_cluster_metrics.json"
        with open(folder_metrics_path, "w") as f:
            json.dump(json.loads(json.dumps(results, default=serializer)), f, indent=2)

        return results

    @torch.no_grad()
    def qualitative_eval(self):
        """
        Evaluate model qualitatively by visualizing clusters and saving results
        """
        cluster_path = Path(self.run_dir / "clusters")
        cluster_path.mkdir(parents=True, exist_ok=True)

        # Setup dataset with paths
        dataset = self.train_loader.dataset
        if hasattr(dataset, "output_paths"):
            dataset.output_paths = True

        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=False,
        )

        for k in range(self.n_prototypes):
            path = cluster_path / f"cluster{k}"
            path.mkdir(parents=True, exist_ok=True)

        # Prepare data collection
        cluster_by_path = []
        k_image = 0
        distances_all, cluster_idx_all = np.array([]), np.array([], dtype=np.int32)

        # Process dataset
        for images, _, _, paths in train_loader:
            images = images.to(self.device)

            batch_distances, batch_argmin_idx = self._get_cluster_argmin_idx(images)
            distances_all = np.hstack([distances_all, batch_distances])
            cluster_idx_all = np.hstack([cluster_idx_all, batch_argmin_idx])

            tsf_imgs = self.model.transform(images).cpu()
            # Save individual images
            for b, (img, idx, d, p) in enumerate(
                zip(images, batch_argmin_idx, batch_distances, paths)
            ):
                convert_to_img(img.cpu()).save(
                    cluster_path / f"cluster{idx}" / f"{k_image}_raw.png"
                )

                # trick for non-RGB images
                tsf_idx = min(idx, tsf_imgs.shape[1] - 1)
                try:
                    tsf_img = tsf_imgs[b, tsf_idx]
                except IndexError:
                    tsf_img = tsf_imgs[b]

                convert_to_img(tsf_img).save(
                    cluster_path / f"cluster{idx}" / f"{k_image}_tsf.png"
                )

                rel_path = (
                    os.path.relpath(p, dataset.data_path)
                    if hasattr(dataset, "data_path")
                    else str(p)
                )
                cluster_by_path.append((k_image, rel_path, idx, float(d)))
                k_image += 1

        dataset.output_paths = False

        if cluster_by_path:
            cluster_df = pd.DataFrame(
                cluster_by_path, columns=["image_id", "path", "cluster_id", "distance"]
            ).set_index("image_id")
            cluster_df.to_csv(self.run_dir / "cluster_by_path.csv")
            cluster_df.to_json(self.run_dir / "cluster_by_path.json", orient="index")
            self.evaluate_folder_to_cluster_mapping(cluster_df.reset_index())

            env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
            template = env.get_template("result-template.html")
            output_html = template.render(
                clusters=range(self.n_prototypes),
                images=cluster_df.to_dict(orient="index"),
                proto_dir=self.output_proto_dir,
            )
            with open(self.run_dir / "clusters.html", "w") as fh:
                fh.write(output_html)

        return [np.array([]) for k in range(self.n_prototypes)]

    @torch.no_grad()
    def _get_cluster_argmin_idx(self, images):
        raise NotImplementedError()


class LoggedKMeansTrainer(LoggingTrainerMixin, KMeansTrainer):
    """
    A KMeansTrainer with hooks to track training progress
    """

    output_proto_dir = "prototypes"

    @torch.no_grad()
    def _get_cluster_argmin_idx(self, images):
        distances = self.model(images)[1]
        dist_min_by_sample, argmin_idx = map(
            lambda t: t.cpu().numpy(), distances.min(1)
        )
        return dist_min_by_sample, argmin_idx

    @torch.no_grad()
    def save_training_metrics(self):
        """
        Overwrite original save_training_metrics method for lightweight plots saving
        """
        self.model.eval()
        # Prototypes & transformation predictions
        self.save_prototypes()
        self.save_transformed_images()

        self.log_end()


class LoggedSpritesTrainer(LoggingTrainerMixin, SpritesTrainer):
    """
    A SpritesTrainer with hooks to track training progress
    """

    output_proto_dir = "masked_prototypes"

    @torch.no_grad()
    def _get_cluster_argmin_idx(self, images):
        dist = self.model(images)[1]
        if self.n_backgrounds > 1:
            dist = dist.view(images.size(0), self.n_prototypes, self.n_backgrounds).min(
                2
            )[0]
        dist_min_by_sample, argmin_idx = map(lambda t: t.cpu().numpy(), dist.min(1))
        return dist_min_by_sample, argmin_idx

    @torch.no_grad()
    def save_training_metrics(self):
        """
        Overwrite original save_training_metrics method for lightweight plots saving
        """
        self.model.eval()
        # Prototypes & transformation predictions
        self.save_prototypes()
        if self.learn_masks:
            self.save_masked_prototypes()
            self.save_masks()
        if self.learn_backgrounds:
            self.save_backgrounds()
        self.save_transformed_images()

        self.log_end()


def default_milestones(transforms, epochs):
    n_tsf = len(transforms.split("_"))
    epochs = epochs

    # Set the number of epochs for each transformation
    m1, m2, m3, m4, m5, m6 = (
        int(epochs * 0.1),
        int(epochs * 0.2),
        int(epochs * 0.3),
        int(epochs * 0.4),
        int(epochs * 0.5),
        int(epochs * 0.6),
    )
    milestones = {
        1: [m1],
        2: [m1, m3],
        3: [m1, m3, m5],
        4: [m1, m2, m3, m5],
        5: [m1, m2, m3, m4, m5],
        6: [m1, m2, m3, m4, m5, m6],
    }
    # len(curriculum_learning) == self.n_tsf - 1
    # if sprites:
    #     cfg.model.curriculum_learning_bkg = milestones[n_tsf - 1]
    return milestones[n_tsf - 1]


def set_transformation_sequence(cfg, tsf_seq, sprites=False):
    """
    Set the transformation sequence for the model

    Args:
        cfg: The configuration object.
        tsf_seq: The transformation sequence to set
        sprites: Whether the model is for sprites or not.

    tsf_seq = {
      "transforms": "identity_linearcolor_affine_morpho_tps",
      "iterations": 15000, # total number of iterations to train
      "milestones": [ 3000, 4000, 7000, 11000 ] # those are iterations, not epochs
      "n_batches": 1000, # total number of batches in the dataset
    }

    Transformations can be:
    # COARSE (should be applied early on during training)
    "id"|"identity": IdentityModule, (first)
    "col"|"color": ColorModule,
    "linearcolor": LinearColorModule,

    # SPATIAL
    "aff"|"affine": AffineModule,
    "pos"|"position": PositionModule,
    "proj"|"projective"|"homography": ProjectiveModule,
    "sim"|"similarity": SimilarityModule,
    "rotation": RotationModule,
    "translation": TranslationModule,
    "tps"|"thinplatespline": TPSModule,

    # MORPHOLOGICAL
    "morpho"|"morphological": MorphologicalModule,
    """
    transforms = tsf_seq.get("transforms", "identity_affine_morpho")
    tsf_nb = len(transforms.split("_"))

    iter_nb = tsf_seq.get("iterations", tsf_nb * 1000)
    batch_nb = tsf_seq.get("n_batches", 500)
    epoch_nb = max(iter_nb // batch_nb, 1)
    # cfg.training.n_iterations = iter_nb
    cfg.training.n_epochs = epoch_nb

    cfg.model.transformation_sequence = transforms
    if tsf_nb == 1:
        cfg.model.curriculum_learning = False
    elif milestones := tsf_seq.get("milestones", False):
        # convert iter in epochs
        cfg.model.curriculum_learning = [it // batch_nb + 1 for it in milestones]
    else:
        cfg.model.curriculum_learning = default_milestones(transforms, epoch_nb)

    cfg.model.grid_size = tsf_seq.get("grid_size", 4)

    if sprites:
        # TODO allow to define custom background transformations
        cfg.model.transformation_sequence_bkg = transforms
        cfg.model.curriculum_learning_bkg = cfg.model.curriculum_learning

    # see if reconstruction decreases
    return cfg


def set_scheduler_milestones(cfg):
    """
    Set the scheduler milestones depending on number of epochs
    """
    cfg.training.scheduler.milestones = [
        int(cfg.training.n_epochs * 0.85),
    ]
    return cfg


def get_n_batches(cfg):
    dataset = get_dataset(cfg.dataset.name)("train", **cfg.dataset)
    dataset_size = len(dataset)

    if dataset_size == 0:
        raise ValueError(f"Dataset '{cfg.dataset.name}' is empty")

    batch_size = (
        cfg.training.batch_size
        if cfg.training.batch_size < dataset_size
        else dataset_size
    )

    return (dataset_size + batch_size - 1) // batch_size


def run_training(
    clustering_id: str,
    dataset_uid: str,
    parameters: dict,
    logger: TLogger = LoggerHelper,
    sprites: bool = False,
):
    base_cfg = OmegaConf.load(BASE_CONFIG_FILE)
    specific_cfg = OmegaConf.load(
        SPRITES_CONFIG_FILE if sprites else KMEANS_CONFIG_FILE
    )
    cfg = OmegaConf.merge(base_cfg, specific_cfg)

    cfg.dataset.tag = dataset_uid

    cfg.training.optimizer.lr = parameters.get("lr", 1e-4)
    cfg.model.empty_cluster_threshold = parameters.get("empty_cluster_threshold", 0.025)

    bkg_opt = parameters.get("background_option", {})
    if sprites:
        # constant background = [False, True, False] + ["constant", "constant", "gaussian"] + [0.1, 0.1, 0.0]
        # learned foreground  = [True, True, False]  + ["constant", "constant", "gaussian"] + [0.1, 0.9, 0.0]
        cfg.model.prototype.data.freeze = bkg_opt.get("freeze", [False, False, False])
        cfg.model.prototype.data.init = bkg_opt.get("init", ["mean", "mean", "mean"])
        cfg.model.prototype.data.value = bkg_opt.get("value", [0.1, 0.5, 0.0])
    else:
        cfg.model.prototype.data.init = bkg_opt.get("init", ["gaussian"])[0]

    # Set training parameters from parameters
    if n_proto := parameters.get("n_prototypes"):
        if sprites:
            cfg.model.n_sprites = n_proto
        else:
            cfg.model.n_prototypes = n_proto

    CONFIGS_PATH.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, CONFIGS_PATH / f"{clustering_id}.yml")

    run_dir = RUNS_PATH / clustering_id
    trainer_class = LoggedSpritesTrainer if sprites else LoggedKMeansTrainer

    torch.backends.cudnn.enabled = False

    if tsf_seq := parameters.get("transformation_sequence"):
        tsf_seq["n_batches"] = get_n_batches(cfg)
        cfg = set_transformation_sequence(cfg, tsf_seq, sprites=sprites)

    cfg = set_scheduler_milestones(cfg)

    try:
        print(OmegaConf.to_yaml(cfg))
        seed = cfg.get("training", {}).get("seed", 3407)
        trainer = trainer_class(
            cfg=cfg, run_dir=str(run_dir), seed=seed, save=True, logger=logger
        )
        trainer.run(seed=seed)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise

    return run_dir


def run_kmeans_training(
    clustering_id: str,
    dataset_uid: str,
    parameters: dict,
    logger: TLogger = LoggerHelper,
) -> Path:
    """
    Main function to run DTI clustering training.

    Args:
        clustering_id (str): The ID of the clustering task.
        dataset_uid (str): The ID of the dataset.
        parameters (dict): An object containing the training parameters.
            Expected keys are:

            - n_prototypes: Number of prototypes.
            - transformation_sequence: Sequence of transformations.
        logger (TLogger, optional): A logger object. Defaults to LoggerHelper.

    Returns:
        Path: The path to the output directory.
    """
    return run_training(
        clustering_id=clustering_id,
        dataset_uid=dataset_uid,
        parameters=parameters,
        logger=logger,
        sprites=False,
    )


def run_sprites_training(
    clustering_id: str,
    dataset_uid: str,
    parameters: dict,
    logger: TLogger = LoggerHelper,
) -> Path:
    """
    Main function to run DTI sprites training.

    Args:
        clustering_id (str): The ID of the clustering task.
        dataset_uid (str): The ID of the dataset.
        parameters (dict): An object containing the training parameters.
            Expected keys are:

            - n_prototypes: Number of prototypes.
            - transformation_sequence: Sequence of transformations.
            - background_option: Option for background handling.
        logger (TLogger, optional): A logger object. Defaults to LoggerHelper.

    Returns:
        Path: The path to the output directory.
    """
    return run_training(
        clustering_id=clustering_id,
        dataset_uid=dataset_uid,
        parameters=parameters,
        logger=logger,
        sprites=True,
    )
