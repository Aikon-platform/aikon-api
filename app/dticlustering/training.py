"""
Training tools to adapt DTI research lib to the API
"""
from yaml import load, Loader, dump, Dumper
from pathlib import Path
import os, torch
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from .lib.src.kmeans_trainer import Trainer as KMeansTrainer
from .lib.src.sprites_trainer import Trainer as SpritesTrainer
from .const import RUNS_PATH, CONFIGS_PATH
from .lib.src.utils.image import convert_to_img
from ..shared.dataset import Dataset

from ..shared.utils.logging import TLogger, LoggerHelper

TEMPLATES_DIR = Path(__file__).parent / "templates"
KMEANS_CONFIG_FILE = TEMPLATES_DIR / "kmeans-conf.yml"
SPRITES_CONFIG_FILE = TEMPLATES_DIR / "sprites-conf.yml"


class LoggingTrainerMixin:
    """
    A mixin with hooks to track training progress inside dti Trainers
    """

    output_proto_dir: str = "prototypes"

    def __init__(self, logger: TLogger, *args, **kwargs):
        self.jlogger = logger
        super().__init__(*args, **kwargs)

    def print_and_log_info(self, string: str) -> None:
        self.jlogger.info(string)
        self.logger.info(string)

    def run(self, *args, **kwargs):
        # Log epoch progress start
        self.jlogger.progress(
            self.start_epoch - 1, self.n_epoches, title="Training epoch"
        )

        return super().run(*args, **kwargs)

    def update_scheduler(self, epoch, batch):
        # Log epoch progress
        self.jlogger.progress(epoch - 1, self.n_epoches, title="Training epoch")

        return super().update_scheduler(epoch, batch)

    def save_training_metrics(self):
        # Log epoch progress end
        self.jlogger.progress(
            self.n_epoches, self.n_epoches, title="Training epoch", end=True
        )
        self.jlogger.info("Training over, running evaluation")

        return super().save_training_metrics()

    @torch.no_grad()
    def qualitative_eval(self):
        """
        Overwrite original qualitative_eval method to save images and results
        """
        cluster_path = Path(self.run_dir / "clusters")
        cluster_path.mkdir(parents=True, exist_ok=True)
        dataset = self.train_loader.dataset
        dataset.output_paths = True
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=False,
        )
        cluster_by_path = []
        k_image = 0

        # Create cluster folders
        for k in range(self.n_prototypes):
            path = cluster_path / f"cluster{k}"
            path.mkdir(parents=True, exist_ok=True)

        # Iterate over dataset
        for images, labels, masks, paths in train_loader:
            images = images.to(self.device)
            distances, argmin_idx = self._get_cluster_argmin_idx(
                images
            )  # depends on the method

            transformed_images = self.model.transform(images)
            argmin_idx = argmin_idx.astype(np.int32)

            # Save individual images to their cluster folder
            for img, idx, d, p, tsf_imgs in zip(
                images, argmin_idx, distances, paths, transformed_images
            ):
                convert_to_img(img).save(
                    cluster_path / f"cluster{idx}" / f"{k_image}_raw.png"
                )
                convert_to_img(tsf_imgs[idx]).save(
                    cluster_path / f"cluster{idx}" / f"{k_image}_tsf.png"
                )
                cluster_by_path.append(
                    (k_image, os.path.relpath(p, dataset.data_path), idx, d)
                )
                k_image += 1

        dataset.output_paths = False

        # Save cluster_by_path to csv (for export) and json (for dataviz)
        cluster_by_path = pd.DataFrame(
            cluster_by_path, columns=["image_id", "path", "cluster_id", "distance"]
        ).set_index("image_id")
        cluster_by_path.to_csv(self.run_dir / "cluster_by_path.csv")
        cluster_by_path.to_json(self.run_dir / "cluster_by_path.json", orient="index")

        # Render jinja template
        env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
        template = env.get_template("result-template.html")
        output_from_parsed_template = template.render(
            clusters=range(self.n_prototypes),
            images=cluster_by_path.to_dict(orient="index"),
            proto_dir=self.output_proto_dir,
        )

        with open(self.run_dir / "clusters.html", "w") as fh:
            fh.write(output_from_parsed_template)

        return [np.array([]) for k in range(self.n_prototypes)]

    @torch.no_grad()
    def save_metric_plots(self):
        """
        Overwrite original save_metric_plots method for lightweight plots saving
        # TODO fill this function even for kmeans clustering
        """
        # self.model.eval()
        # # Prototypes & transformation predictions
        # self.save_prototypes()
        # if self.learn_masks:
        #     self.save_masked_prototypes()
        #     self.save_masks()
        # if self.learn_backgrounds:
        #     self.save_backgrounds()
        # self.save_transformed_images()
        pass

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


def setup_conf(cfg, clustering_id):
    """
    Convert a dict to an OmegaConf object
    """
    config_file = CONFIGS_PATH / f"{clustering_id}.yml"
    CONFIGS_PATH.mkdir(parents=True, exist_ok=True)
    dump(cfg, open(config_file, "w"), Dumper=Dumper)

    # from omegaconf import OmegaConf
    # try:
    #     cfg = OmegaConf.create(cfg)
    # except Exception as e:
    #     print(f"Error converting to OmegaConf: {e}")
    #     pass

    return cfg


def run_kmeans_training(
    clustering_id: str,
    # dataset: Dataset, dataset (Dataset): The dataset to be processed
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
    # Load config template
    train_config = load(open(KMEANS_CONFIG_FILE), Loader=Loader)

    # Set dataset tag and run dir
    train_config["dataset"]["tag"] = dataset_uid
    run_dir = RUNS_PATH / clustering_id

    # Set training parameters from parameters
    if "n_prototypes" in parameters:
        train_config["model"]["n_prototypes"] = parameters["n_prototypes"]

    if "transformation_sequence" in parameters:
        train_config["model"]["transformation_sequence"] = parameters[
            "transformation_sequence"
        ]

    cfg = setup_conf(train_config, clustering_id)
    # TODO to remove once hydra is configured
    cfg = CONFIGS_PATH / f"{clustering_id}.yml"

    # Run training
    trainer = LoggedKMeansTrainer(
        logger, cfg, run_dir, seed=train_config["training"]["seed"]
    )
    trainer.run(seed=train_config["training"]["seed"])

    # Return output directory
    return run_dir


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
    # Load config template
    train_config = load(open(SPRITES_CONFIG_FILE), Loader=Loader)

    # Set dataset tag and run dir
    train_config["dataset"]["tag"] = dataset_uid
    run_dir = RUNS_PATH / clustering_id

    # Set background option
    bkg_opt = parameters.get("background_option", "1_learn_bg")

    if bkg_opt == "2_const_bg":
        # Data parameters are respectively [foreground, background, masks]
        train_config["model"]["prototype"]["data"]["freeze"][1] = True
        train_config["model"]["prototype"]["data"]["init"][1] = "constant"
        train_config["model"]["prototype"]["data"]["value"][1] = 0.1
    elif bkg_opt == "3_learn_fg":
        train_config["model"]["prototype"]["data"]["freeze"] = [True, True, False]
        train_config["model"]["prototype"]["data"]["init"] = [
            "constant",
            "constant",
            "gaussian",
        ]
        train_config["model"]["prototype"]["data"]["value"] = [0.1, 0.9, 0.0]

    # Set training parameters from parameters
    if "n_prototypes" in parameters:
        train_config["model"]["n_sprites"] = parameters["n_prototypes"]

    if "transformation_sequence" in parameters:
        train_config["model"]["transformation_sequence"] = parameters[
            "transformation_sequence"
        ]
        train_config["model"]["transformation_sequence_bkg"] = parameters[
            "transformation_sequence"
        ]

    cfg = setup_conf(train_config, clustering_id)

    # Run training
    trainer = LoggedSpritesTrainer(
        logger, cfg, run_dir, seed=train_config["training"]["seed"]
    )
    trainer.run(seed=train_config["training"]["seed"])

    # Return output directory
    return run_dir
