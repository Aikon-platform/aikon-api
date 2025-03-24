from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from typing import (
    Optional,
    TypedDict,
    Dict,
    List,
    TypeAlias,
    Tuple,
    cast,
    Set,
    TypeVar,
    Union,
)
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Iterable
import orjson
from scipy.spatial.distance import cdist

from .const import SCORES_PATH, DEMO_NAME
from .lib.const import (
    SEG_STRIDE,
    MAX_SIZE,
    COS_TOPK,
    FEAT_NET,
)
from .lib.dataset import FileListDataset
from .lib.features import FeatureExtractor
from .lib import segswap
from .lib.models import get_model_path
from .lib.utils import AllTranspose

from ..shared.dataset import Dataset, Document
from ..shared.dataset.document import DocDict, get_file_url
from ..shared.dataset.utils import ImageDict, Image
from ..shared.utils import get_device
from ..shared.tasks import LoggedTask
from ..shared.utils.logging import serializer, console

SimScore: TypeAlias = Tuple[float, int, int]
PairTuple: TypeAlias = Tuple[int, int, float, int, int]
PairList: TypeAlias = Set[PairTuple] | List[PairTuple]
DocRef: TypeAlias = Tuple[str, str]


@dataclass
class DocInFeatures:
    document: Document
    range: range
    images: List[Image]

    def __eq__(self, value: "DocInFeatures") -> bool:
        return self.document.uid == value.document.uid

    def __hash__(self):
        return hash(self.document.uid)

    def slice(self, scale_by: int) -> slice:
        return slice(self.range.start * scale_by, self.range.stop * scale_by)

    def __str__(self):
        return (
            f"DocInFeatures({self.document.uid}, {self.range.start}-{self.range.stop})"
        )

    def __repr__(self):
        return str(self)


Pair: TypeAlias = Tuple[Tuple[DocInFeatures, int], Tuple[DocInFeatures, int], SimScore]


def _extend_from_dense_scores(
    matrix,
    scores: np.ndarray,
    topk: int,
    min_tr_i: np.ndarray = None,
    min_tr_j: np.ndarray = None,
    min_score: float = 0.0,
) -> None:
    """
    Get top-k matches between two documents. Updates the output matrix with the best matches.
    """
    tops = np.argsort(-scores, axis=1)[:, :topk]  # Negative for descending order

    for i, row in enumerate(tops):
        for j in row:
            if scores[i, j] < min_score:
                break
            matrix[i, j] = (
                round(float(scores[i, j]), 3),
                int(min_tr_i[i, j]),
                int(min_tr_j[i, j]),
            )


class SparseDocSimMatrix:
    def __init__(self, doc1: DocInFeatures, doc2: DocInFeatures):
        """
        Data structure to store the similarity matrix between two documents
        """
        self.data: Dict[Tuple[int, int], SimScore] = OrderedDict()
        self.doc1 = doc1
        self.doc2 = doc2
        assert doc1.document.uid <= doc2.document.uid, "Documents must be sorted by UID"

    def __getitem__(self, key: Tuple[int, int]) -> SimScore:
        i, j = key
        i, j = int(i), int(j)
        return self.data[i, j]

    def __setitem__(self, key: Tuple[int, int], value: SimScore):
        i, j = key
        i, j = int(i), int(j)
        if (
            self.doc1 == self.doc2 and i > j
        ):  # if self-similarity, only store upper triangle
            i, j = j, i
        if (i, j) not in self.data or self.data[i, j][0] < value[0]:
            self.data[i, j] = value

    def __iter__(self) -> Iterable[Pair]:
        for (i, j) in sorted(self.data.keys()):
            yield (self.doc1, i), (self.doc2, j), self.data[i, j]

    def __len__(self) -> int:
        return len(self.data)

    def absolute_pairs(self) -> Iterable[PairTuple]:
        for (i, j) in sorted(self.data.keys()):
            yield self.doc1.range[i], self.doc2.range[j], *self.data[i, j]

    def transposed(self) -> "TransposedSimMatrix":
        return TransposedSimMatrix(self)

    def untransposed(self):
        return self

    def extend_from_dense_scores(
        self, scores: np.ndarray, topk: int, tr_i: np.ndarray, tr_j: np.ndarray
    ):
        _extend_from_dense_scores(self, scores, topk, tr_i, tr_j)


class TransposedSimMatrix:
    def __init__(self, obj: SparseDocSimMatrix):
        self.obj = obj

    def __getitem__(self, key: Tuple[int, int]) -> SimScore:
        i, j = key
        return self.obj[j, i]

    def __setitem__(self, key: Tuple[int, int], value: SimScore) -> None:
        i, j = key
        self.obj[j, i] = value

    def extend_from_dense_scores(
        self, scores: np.ndarray, topk: int, tr_i: np.ndarray, tr_j: np.ndarray
    ):
        _extend_from_dense_scores(self, scores, topk, tr_i, tr_j)

    def untransposed(self) -> SparseDocSimMatrix:
        return self.obj


class BlockSimMatrix:
    def __init__(self):
        """
        Data structure to store pairwise document similarity matrices
        """
        self.data: Dict[Tuple[str, str], SparseDocSimMatrix] = OrderedDict()

    def __getitem__(
        self, docs: Tuple[DocInFeatures, DocInFeatures]
    ) -> Union[SparseDocSimMatrix, TransposedSimMatrix]:
        doc1, doc2 = docs
        if reverse := (doc1.document.uid > doc2.document.uid):
            doc1, doc2 = doc2, doc1
        doc = self.data.setdefault(
            (doc1.document.uid, doc2.document.uid), SparseDocSimMatrix(doc1, doc2)
        )
        if reverse:
            return doc.transposed()
        return doc

    def __iter__(self) -> Iterable[Pair]:
        for matrix in self.data.values():
            yield from matrix

    def __len__(self) -> int:
        return sum(len(matrix) for matrix in self.data.values())

    def absolute_pairs(self) -> Iterable[PairTuple]:
        for matrix in self.data.values():
            yield from matrix.absolute_pairs()

    def blocks(self) -> Iterable[SparseDocSimMatrix]:
        return self.data.values()


class DocIndex(TypedDict):
    sources: Dict[str, DocDict]
    images: List[ImageDict]
    transpositions: List[str]


class SimParameters(TypedDict):
    algorithm: str
    topk: int
    feat_net: str
    segswap_prefilter: bool
    segswap_n: Optional[int]
    raw_transpositions: Optional[List[str]]
    # transpositions: Optional[List[str]]


class SimilarityResults(TypedDict):
    parameters: SimParameters
    index: DocIndex
    pairs: List[PairTuple]


def group_by_documents(images: List[Image]) -> List[DocInFeatures]:
    """
    Identify groups of consecutive images from the same document
    """
    ranges = []
    p = 0
    for k, i in enumerate(images + [None]):
        if i is None or i.document != images[p].document:
            ranges.append(DocInFeatures(images[p].document, range(p, k), images[p:k]))
            p = k
    return ranges


def handle_transpositions(
    sim_matrix: np.ndarray, n_trans_rows: int, n_trans_cols: int = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Handle multiple transpositions per image.
    """
    if n_trans_cols is None:
        n_trans_cols = n_trans_rows

    n_rows, n_cols = sim_matrix.shape
    n_im_rows = n_rows // n_trans_rows
    n_im_cols = n_cols // n_trans_cols
    assert n_rows % n_trans_rows == 0, "Features must be divisible by transpositions"
    assert n_cols % n_trans_cols == 0, "Features must be divisible by transpositions"

    # Reshape to get all transposition combinations
    sim_trans = (
        sim_matrix.reshape(n_im_rows, n_trans_rows, n_im_cols, n_trans_cols)
        .transpose(0, 2, 1, 3)
        .reshape(n_im_rows, n_im_cols, n_trans_rows * n_trans_cols)
    )

    # Find best transposition pairs
    best_trans = sim_trans.argmax(axis=2, keepdims=True)
    sim_matrix = np.take_along_axis(sim_trans, best_trans, axis=2).squeeze(axis=2)
    tr_i, tr_j = np.divmod(best_trans.squeeze(axis=2), n_trans_cols)

    return sim_matrix, tr_i, tr_j


class ComputeSimilarity(LoggedTask):
    """
    Compute the similarity between images inside a dataset
    """

    def __init__(
        self, dataset: Dataset, parameters: Optional[dict] = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.results: SimilarityResults | dict = {}
        self._results_url = []

        self.dataset = dataset
        self.images = self.dataset.prepare()

        # Sequences of indices for each document to compare
        self.doc_images = group_by_documents(self.images)

        self.feat_net = parameters.get("feat_net", FEAT_NET) if parameters else FEAT_NET
        self.topk = parameters.get("topk", COS_TOPK)
        self.algorithm = parameters.get("algorithm", "cosine")

        # Whether to perform pre-filter using cosine similarity to keep only best matches before running segswap
        self.segswap_prefilter = parameters.get("segswap_prefilter", True)
        # If so, how many best matches should be kept
        self.segswap_n = parameters.get("segswap_n", COS_TOPK)

        self.raw_transpositions: List[str] = parameters.get("transpositions", ["none"])
        self.transpositions = [
            getattr(AllTranspose, t.upper()) for t in self.raw_transpositions
        ]

        self.device = get_device()

    @property
    def results_url(self):
        if self._results_url:
            return self._results_url

        self._results_url = self.get_results_url(self.experiment_id)
        return self._results_url

    def get_results_url(self, experiment_id):
        results_url = []
        for path in SCORES_PATH.rglob(f"{experiment_id}/*"):
            if not path.is_file() or path.suffix != ".json":
                continue

            filename = path.stem
            file_path = f"{experiment_id}/{filename}"

            if filename == f"{self.dataset.uid}-scores":
                results_url.append(
                    {
                        "doc_pair": "dataset",
                        "result_url": get_file_url(DEMO_NAME, file_path),
                    }
                )
                continue

            if filename.startswith(self.algorithm):
                results_url.append(
                    {
                        "doc_pair": filename.replace(f"{self.algorithm}-", ""),
                        "result_url": get_file_url(DEMO_NAME, file_path),
                    }
                )

        return results_url

    def add_results_url(self, value):
        self._results_url.append(value)

    @torch.no_grad()
    def get_features(self, img_paths: List[str]):
        """
        Extract features from a list of images
        """
        extractor = FeatureExtractor(
            feat_net=self.feat_net,
            device=self.device,
        )

        img_dataset = FileListDataset(
            data_paths=img_paths,
            transform=extractor.transforms,
            device=self.device,
            transpositions=self.transpositions,
        )

        data_loader = DataLoader(img_dataset, batch_size=16, shuffle=False)
        cache_id = (
            f"{self.dataset.uid}@{''.join(str(t.value) for t in self.transpositions)}"
        )

        features = extractor.extract_features(
            data_loader,
            cache_dir=self.dataset.path / "features",
            cache_id=cache_id,
        )

        if not features.numel():
            self.print_and_log_warning("[task.similarity] No features extracted")
            self.task_update(
                "ERROR",
                "[API ERROR] No features extracted",
                exception=Exception(
                    f"No features were extracted for {self.dataset.uid}"
                ),
            )
            return

        try:
            del extractor
        except Exception:
            self.print_and_log_warning(
                "[task.similarity] Failed to clear memory from extractor"
            )

        return features

    def format_parameters(self) -> SimParameters:
        return {
            "algorithm": self.algorithm,
            "topk": self.topk,
            "feat_net": self.feat_net,
            "segswap_prefilter": self.segswap_prefilter,
            "segswap_n": self.segswap_n,
            "raw_transpositions": self.raw_transpositions,
            # "transpositions": self.transpositions,
        }

    def format_results(self, pairs: Iterable[Pair]) -> SimilarityResults:
        """
        Format the results for output

        Args:
            pairs (Iterable[Pair]): The similarity pairs

        Returns:
            A dictionary with the document index and pairs
        """
        pairs = list(pairs)

        docs = sorted(
            set(pair[0][0] for pair in pairs).union(pair[1][0] for pair in pairs),
            key=lambda d: d.document.uid,
        )
        offsets = {}
        offset = 0

        for doc in docs:
            offsets[doc] = offset
            offset += len(doc.range)
        # print("offsets", offsets)
        # print("pairs", pairs)

        return {
            "parameters": self.format_parameters(),
            "index": {
                "sources": {
                    doc.document.uid: doc.document.to_dict(with_metadata=True)
                    for doc in docs
                },
                "images": [
                    cast(ImageDict, {**im.to_dict(), "doc_uid": im.document.uid})
                    for document in docs
                    for im in document.images
                ],
                "transpositions": self.raw_transpositions,
            },
            "pairs": [
                (offsets[doc1] + i, offsets[doc2] + j, *sim)
                for (doc1, i), (doc2, j), sim in pairs
            ],
        }

    @torch.no_grad()
    def compute_similarity(self) -> SimilarityResults:
        """
        Compute the similarity between images in the dataset and returns the results
        """
        source_paths = [str(i.path) for i in self.images]

        self.log(
            f"Prepared {len(self.images)} images to be processed"
        )
        topk = self.segswap_n if self.algorithm == "segswap" else self.topk
        features = self.get_features(source_paths)

        # TODO skip this step if self.algorithm == "segswap" && self.segswap_prefilter == false
        pairs = self.compute_cosine_similarity(
            features.cpu().numpy(),
            topk=topk,
            n_transpositions=len(self.transpositions),
        )

        if self.algorithm == "segswap":
            pairs = self.compute_segswap_similarity(
                source_paths, pairs, cos_topk=topk, device=self.device
            )

        self.log(
            f"Computed similarity scores for {len(pairs)} pairs"
        )

        return self.format_results(pairs)

    @staticmethod
    def get_docs_ref(uid1, uid2) -> DocRef:
        return tuple(sorted([uid1, uid2]))

    def get_doc_uid(self, idx: int) -> str:
        """Get the document UID for an image index."""
        return self.images[idx].document.uid

    def store(self, matrix: SparseDocSimMatrix, algorithm="cosine"):
        """Store similarity pairs for a document pair and sends results to front"""
        doc_ref = "-".join(
            self.get_docs_ref(matrix.doc1.document.uid, matrix.doc2.document.uid)
        )
        result_name = f"{algorithm}-{doc_ref}"

        score_file = SCORES_PATH / self.experiment_id / f"{result_name}.json"
        score_file.parent.mkdir(parents=True, exist_ok=True)

        res = self.format_results(matrix)
        with open(score_file, "wb") as f:
            f.write(orjson.dumps(res, default=serializer))

        if self.algorithm == algorithm:
            file_path = f"{self.experiment_id}/{result_name}"
            result_url = {
                "doc_pair": doc_ref,
                "result_url": get_file_url(DEMO_NAME, file_path),
            }

            self.add_results_url(result_url)
            self.notifier(
                "PROGRESS",
                output={
                    "dataset_url": self.dataset.get_absolute_url(),
                    "results_url": [result_url],
                },
            )

    def compute_cosine_similarity(
        self,
        features: np.ndarray,
        topk: int = COS_TOPK,
        n_transpositions: int = 1,
    ) -> BlockSimMatrix:
        """
        Compute pairwise cosine similarities between feature vectors, optionally handling transpositions.

        Args:
            features: Feature vectors of shape (n_samples * n_transpositions, n_features)
            topk: Number of most similar matches to return per vector
            n_transpositions: Number of consecutive features representing the same image

        Returns:
            A BlockSimMatrix object with the similarity scores, grouped by pairs of documents
        """
        doc_images = self.doc_images

        self.log(
            f"Computing cosine similarity for {len(doc_images)} pairs"
        )

        all_scores = BlockSimMatrix()
        for doc1 in doc_images:
            for doc2 in doc_images:
                sim_matrix = 1.0 - cdist(
                    features[doc1.slice(scale_by=n_transpositions)],
                    features[doc2.slice(scale_by=n_transpositions)],
                    metric="cosine",
                )

                if n_transpositions > 1:
                    sim_matrix, tr_i, tr_j = handle_transpositions(
                        sim_matrix, n_transpositions, n_transpositions
                    )
                else:
                    tr_i = tr_j = np.zeros_like(sim_matrix)

                if doc1 == doc2:
                    np.fill_diagonal(sim_matrix, -1000)  # Exclude self-matches

                pairs = all_scores[doc1, doc2]
                pairs.extend_from_dense_scores(sim_matrix, topk, tr_i, tr_j)

                self.store(pairs.untransposed(), algorithm="cosine")

        return all_scores

    @torch.no_grad()
    def compute_segswap_similarity(
        self,
        source_images: List[str],
        input_pairs: BlockSimMatrix,
        cos_topk,
        device="cuda",
    ) -> BlockSimMatrix:
        """
        Compute the similarity between pairs of images using the SegSwap algorithm

        Args:
            source_images (List[str]): The list of image paths
            input_pairs (BlockSimMatrix): The input pairs of documents
            cos_topk (int): The number of best matches to return
            device (str): The device to run the computation on

        Returns:
            A list of pairs (k_i, k_j, sim, tr_i, tr_j)
        """
        self.log(
            f"Computing SegSwap similarity for {len(input_pairs.data)} pairs of documents"
        )

        param = torch.load(get_model_path("hard_mining_neg5"), map_location=device)
        backbone = segswap.load_backbone(param).to(device)
        encoder = segswap.load_encoder(param).to(device)

        feat_size = MAX_SIZE // SEG_STRIDE
        y_grid, x_grid = np.where(np.ones((feat_size, feat_size), dtype=bool))

        img_dataset = FileListDataset(
            data_paths=source_images,
            transform=transforms.Compose(
                [
                    transforms.Resize((segswap.MAX_SIZE, segswap.MAX_SIZE)),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            device=device,
        )

        last_img_idx = None
        q_tensor = None
        segswap_scores = BlockSimMatrix()
        # TODO make a dataloader
        for block in input_pairs.blocks():
            doc1 = block.doc1
            doc2 = block.doc2
            doc_scores = segswap_scores[doc1, doc2]

            pairs = list(block)

            batched_pairs = [
                pairs[i : i + cos_topk] for i in range(0, len(pairs), cos_topk)
            ]
            for batch in batched_pairs:
                tensor1, tensor2, batch_pairs = [], [], []
                for (_, rel_i), (_, rel_j), *_ in batch:
                    i = doc1.range[rel_i]
                    j = doc2.range[rel_j]
                    print(("i", rel_i, i, "j", rel_j, j, last_img_idx))

                    # Reuse tensor if same image index (assumes sorted pairs)
                    if last_img_idx != i:
                        q_tensor = img_dataset[i]
                        last_img_idx = i

                    tensor1.append(q_tensor)
                    tensor2.append(img_dataset[j])
                    batch_pairs.append((rel_i, rel_j))

                scores = segswap.compute_score(
                    torch.stack(tensor1),
                    torch.stack(tensor2),
                    backbone,
                    encoder,
                    y_grid,
                    x_grid,
                )

                for (i, j), s in zip(batch_pairs, scores):
                    doc_scores[i, j] = (round(float(s), 2), 0, 0)

            self.store(doc_scores, "segswap")

        return segswap_scores

    def run_task(self):
        if not self.check_dataset():
            self.print_and_log_warning("[task.similarity] No documents to compare")
            self.task_update(
                "ERROR",
                "[API ERROR] No documents to compare",
                exception=Exception(f"No images to compare"),
            )
            return

        self.task_update("STARTED")
        self.log(
            f"Similarity task triggered for {self.dataset.uid} with {self.feat_net}!"
        )

        scores, experiment_id = self.check_already_computed()
        if scores:
            return {
                "dataset_url": self.dataset.get_absolute_url(),
                "results_url": self.get_results_url(experiment_id),
            }

        try:
            self.results = self.compute_similarity()

            (SCORES_PATH / self.experiment_id).parent.mkdir(parents=True, exist_ok=True)
            with open(
                SCORES_PATH / self.experiment_id / f"{self.dataset.uid}-scores.json",
                "wb",
            ) as f:
                f.write(orjson.dumps(self.results, default=serializer))
                self.add_results_url(
                    {
                        "doc_pair": "dataset",
                        "result_url": get_file_url(
                            DEMO_NAME, f"{self.experiment_id}/{self.dataset.uid}-scores"
                        ),
                    }
                )

            self.log(f"Successfully computed similarity scores")
            return True
        except Exception as e:
            self.task_update(
                "ERROR",
                message=[f"Error initializing similarity task: {e}"] + self.error_list,
                exception=e,
            )
            return False
        finally:
            pass

    def check_parameters(self, parameters):
        """
        Return True if all the parameters are the same (meaning that the similarity has already been computed)
        False if one of the parameters is not the same
        """
        if parameters is None:
            return False
        if parameters.get("algorithm", None) != self.algorithm:
            return False
        if parameters.get("topk", None) != self.topk:
            return False
        if parameters.get("feat_net", None) != self.feat_net:
            return False
        if parameters.get("segswap_n", None) != self.segswap_n:
            return False

        # OTHER PARAMETERS TO CHECK
        # "segswap_prefilter": self.segswap_prefilter,
        # "raw_transpositions": self.raw_transpositions,
        # "transpositions": self.transpositions,
        return True

    def check_already_computed(self):
        for path in SCORES_PATH.rglob(f"{self.dataset.uid}-scores.json"):
            if not path.is_file():
                continue

            try:
                scores = orjson.loads(path.read_text())
            except (orjson.JSONDecodeError, OSError) as e:
                self.print_and_log_warning(
                    f"[task.similarity] Error reading scores file {path}: {e}"
                )
                continue

            if self.check_parameters(scores.get("parameters")):
                return scores, path.parent.name

        return False, False

    def check_dataset(self):
        if self.dataset is None:
            return False

        if len(self.dataset.documents) == 0:
            return False

        return True
