from typing import Optional, List, Dict, Tuple
from torch.utils.data import DataLoader
from safetensors.torch import save_file as sft_save_file, safe_open as sft_safe_open
import numpy as np
from scipy.spatial.distance import cdist
from pathlib import Path
import orjson
import gzip

from ..similarity.lib.features import FeatureExtractor
from ..similarity.lib.dataset import FileListDataset
from ..similarity.lib.const import FEAT_NET, COS_TOPK
from ..similarity.lib.utils import AllTranspose, handle_transpositions, group_by_documents

from ..shared.tasks import LoggedTask
from ..shared.dataset import Dataset, Image
from ..shared.dataset.utils import ImageDict
from ..shared.utils import get_device
from .const import SEARCH_INDEX_PATH, SEARCH_RESULTS_PATH

class DatasetIndex:
    """
    This class represents an index of a dataset

    Building:
    - Initialize the class from a dataset
    - Call .build() to build the index
    - Call .save() to save the index into two files: 
        - index_id.safetensors contains feature vectors
        - index_id.json.gz contains metadata and list of images
    
    Querying:
    - Call .load() to load the index from the index_id.safetensors file
    - Call .query() to query the index with a dataset. Returns a dictionary with the description of the query dataset, and the associated results (as a list of pairs of image indices and scores). The index indices refer to the index_id.json.gz file.

    Utils:
    - .describe_self() returns a dictionary with the description of the index (metadata and list of images)
    """
    def __init__(self, dataset: Dataset, feat_net: str, transpositions: List[str], extra_metadata: Optional[Dict] = None):
        self.dataset = dataset
        self.images = self.dataset.prepare()

        self.feat_net = feat_net

        self.raw_transpositions = transpositions
        self.transpositions = [
            getattr(AllTranspose, t.upper()) for t in self.raw_transpositions
        ]

        self.index_id = self.id_for_dataset(self.dataset, self.feat_net, self.transpositions)
        self.index_path = self.path_for_id(self.index_id)
        
        self.extractor = None
        self.device = get_device()

        self.extra_metadata = extra_metadata

    @property
    def metadata(self):
        return {
            "dataset_uid": self.dataset.uid,
            "index_id": self.index_id,
            "feat_net": self.feat_net,
            "transpositions": self.raw_transpositions,
            **(self.extra_metadata or {}),
        }

    def init_extractor(self):
        """
        Initialize the feature extractor
        """
        self.extractor = FeatureExtractor(
            feat_net=self.feat_net,
            device=self.device,
        )

    @staticmethod
    def id_for_dataset(dataset: Dataset, feat_net: str, transpositions: List[str]):
        """
        Generate an ID for a dataset index, for a given (dataset, feature net, transpositions) tuple
        """
        return f"{dataset.uid}+{feat_net}+{''.join(str(t.value) for t in transpositions)}"

    @staticmethod
    def path_for_id(index_id: str) -> Path:
        """
        Returns the path for a dataset index, for a given index ID
        """
        return SEARCH_INDEX_PATH / f"{index_id}.safetensors"

    @classmethod
    def load(cls, index_id: str, dataset: Optional[Dataset] = None):
        index_path = cls.path_for_id(index_id)
        """
        Prepare the index for a given index ID
        """
        with sft_safe_open(index_path, "torch") as f:
            metadata = f.metadata()
            if dataset is None:
                dataset = Dataset(uid=metadata["dataset_uid"], load=True)
            else:
                if dataset.uid != metadata["dataset_uid"]:
                    raise Exception("Dataset does not match the index")
            
            obj = cls(
                dataset=dataset,
                feat_net=metadata["feat_net"],
                transpositions=metadata["raw_transpositions"],
                extra_metadata=metadata,
            )
            obj.index_features = f["features"]
            return obj

    def describe_images(self, images: List[Image], transpositions: List[str]) -> dict:
        docs = group_by_documents(images)
        return {
            "sources": {
                doc.document.uid: doc.document.to_dict(with_metadata=True)
                for doc in docs
            },
            "images": [
                cast(ImageDict, {**im.to_dict(), "doc_uid": im.document.uid})
                for document in docs
                for im in document.images
            ],
            "transpositions": transpositions,
        }

    def describe_self(self) -> dict:
        return {
            "metadata": self.extra_metadata,
            "index": self.describe_images(self.images, self.transpositions)
        }

    def build(self):
        """
        Build the index
        """
        if self.extractor is None:
            self.init_extractor()

        img_paths = [str(i.path) for i in self.images]

        img_dataset = FileListDataset(
            data_paths=img_paths,
            transform=self.extractor.transforms,
            device=self.device,
            transpositions=self.transpositions,
        )

        data_loader = DataLoader(img_dataset, batch_size=16, shuffle=False)
        cache_id = (
            f"{self.dataset.uid}@{''.join(str(t.value) for t in self.transpositions)}"
        )

        self.index_features = self.extractor.extract_features(
            data_loader,
            cache_dir=self.dataset.path / "features",
            cache_id=cache_id,
        )

    def save(self):
        """
        Save the index to disk
        """
        sft_save_file(
            {"features": self.index_features}, 
            self.index_path, 
            metadata=self.metadata
        )
        with gzip.open(self.index_path.with_suffix(".json.gz"), "wt") as f:
            orjson.dump(self.metadata, f)
    
    def query(self, target_dataset: Dataset, raw_transpositions: List[str], topk: int = COS_TOPK):
        """
        Query the index with a list of image URLs
        """
        if self.extractor is None:
            self.init_extractor()

        query_images = target_dataset.prepare()
        query_transpositions = [
            getattr(AllTranspose, t.upper()) for t in raw_transpositions
        ]

        img_dataset = FileListDataset(
            data_paths=[str(i.path) for i in query_images],
            transform=self.extractor.transforms,
            device=self.device,
            transpositions=query_transpositions,
        )

        data_loader = DataLoader(img_dataset, batch_size=16, shuffle=False)

        query_features = self.extractor.extract_features(
            data_loader # no cache
        )

        pairs = self.compute_cosine_similarity(
            query_features.cpu().numpy(),
            self.index_features.cpu().numpy(),
            topk=self.topk,
            n_query_transpositions=len(query_transpositions),
        )

        return {
            "query": self.describe_images(query_images, query_transpositions),
            "result": {
                "pairs": [
                    (int(i), int(j), float(score), int(tr_i), int(tr_j))
                    for i, (js, scores, tr_is, tr_js) in enumerate(zip(*pairs))
                    for (j, score, tr_i, tr_j) in zip(js, scores, tr_is, tr_js)
                ],
            }
        }

    def compute_cosine_similarity(
        self,
        query_features: np.ndarray,
        topk: int = COS_TOPK,
        n_query_transpositions: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute pairwise cosine similarities between feature vectors, handling transpositions.

        Args:
            query_features: Feature vectors of shape (n_query_samples * n_query_transpositions, n_features)
            topk: Number of most similar matches to return per query vector
            n_query_transpositions: Number of consecutive features representing the same image

        Returns:
            A tuple of (sim_matrix, tr_i, tr_j) where:
                sim_matrix: A matrix of shape (n_query_samples, topk) containing the cosine similarities
                tr_i: A vector of shape (n_query_samples, topk) containing the best transpositions for the query features
                tr_j: A vector of shape (n_index_samples, topk) containing the best transpositions for the index features
        """
        self.log(f"Computing cosine similarity")

        n_index_transpositions = len(self.transpositions)

        sim_matrix = 1.0 - cdist(
            query_features,
            self.index_features,
            metric="cosine",
        )

        if n_query_transpositions > 1 or n_index_transpositions > 1:
            sim_matrix, tr_i, tr_j = handle_transpositions(
                sim_matrix, n_query_transpositions, n_index_transpositions
            )
        else:
            tr_i = tr_j = np.zeros_like(sim_matrix)

        # get topk
        raw_top_indices = np.argpartition(sim_matrix, -topk, axis=1)[:, -topk:]
        sorted_top_indices = np.argsort(np.take_along_axis(sim_matrix, raw_top_indices, axis=1), axis=1)

        top_indices = np.take_along_axis(raw_top_indices, sorted_top_indices, axis=1)
        top_values = np.take_along_axis(sim_matrix, top_indices, axis=1)
        top_tr_i = np.take_along_axis(tr_i, top_indices, axis=1)
        top_tr_j = np.take_along_axis(tr_j, top_indices, axis=1)

        return top_indices, top_values, top_tr_i, top_tr_j


class IndexDataset(LoggedTask):
    """
    Index a dataset of images
    """

    def __init__(
        self, dataset: Dataset, parameters: Optional[dict] = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.dataset = dataset

        self.feat_net = parameters.get("feat_net", FEAT_NET) if parameters else FEAT_NET
        self.algorithm = "cosine"

        self.raw_transpositions: List[str] = parameters.get("transpositions", ["none"])

        self.device = get_device()

    @staticmethod
    def path_for_task(experiment_id: str) -> Path:
        return SEARCH_INDEX_PATH / f"{experiment_id}.json"

    def run_task(self) -> bool:
        if not self.check_dataset():
            self.print_and_log_warning("[task.search] No documents to index")
            self.task_update(
                "ERROR",
                "[API ERROR] No documents to index",
                exception=Exception(f"No images to index"),
            )
            return False

        self.task_update("STARTED")
        self.log(
            f"Dataset indexing task triggered for {self.dataset.uid} with {self.feat_net}!"
        )

        try:
            self.index_info = self.index_dataset()

            task_output_file = self.path_for_task(self.experiment_id)
            task_output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(task_output_file, "wb") as f:
                f.write(orjson.dumps(self.index_info, default=serializer))

            self.log(f"Successfully indexed dataset")
            return True
        except Exception as e:
            self.task_update(
                "ERROR",
                f"[API ERROR] Error indexing dataset: {str(e)}",
                exception=e,
            )
            return False
        
    def check_dataset(self) -> bool:
        return len(self.images) > 0

    def index_dataset(self):
        """
        Create an index of the images
        """

        self.index = DatasetIndex(
            self.dataset, 
            self.feat_net, 
            self.raw_transpositions, 
            {
                "from_experiment": self.experiment_id
            }
        )

        self.log(f"Indexing {len(self.index.images)} images")

        self.index.build()

        self.log("Indexing completed, saving index")
        
        self.index.save()

        self.log("Index saved")

        return self.index.describe_self()

class QueryIndex(LoggedTask):
    """
    Query an index with another dataset
    """

    def __init__(
        self, dataset: Dataset, parameters: Optional[dict] = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.query_dataset = dataset
        self.query_images = self.query_dataset.prepare()

        self.index_id = parameters.get("index_id")

        self.raw_transpositions: List[str] = parameters.get("transpositions", ["none"])
        self.transpositions = [
            getattr(AllTranspose, t.upper()) for t in self.raw_transpositions
        ]

        self.device = get_device()

    def path_for_task(experiment_id: str) -> Path:
        return SEARCH_RESULTS_PATH / f"{experiment_id}.json"

    def run_task(self) -> bool:
        try:
            # Load the index
            self.log(f"Loading index {self.index_id}")

            self.index = DatasetIndex.load(self.index_id)

            self.log(f"Loaded index {self.index_id}; Querying index with {len(self.query_images)} images")

            results = self.index.query(self.query_dataset)

            self.log(f"Query completed, saving results")

            task_output_file = self.path_for_task(self.experiment_id)
            task_output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(task_output_file, "wb") as f:
                f.write(orjson.dumps(results, default=serializer))

            self.task_update("COMPLETED")
            return True
        except Exception as e:
            self.task_update(
                "ERROR",
                f"[API ERROR] Error querying index: {str(e)}",
                exception=e,
            )
            return False
