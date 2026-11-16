import itertools
from collections import OrderedDict
from functools import cached_property
from typing import (
    Optional,
    TypedDict,
    Dict,
    List,
    TypeAlias,
    Tuple,
    Set,
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
from .lib.utils import AllTranspose, handle_transpositions

from ..shared.dataset import Dataset
from ..shared.dataset.types import ImageDict, DocInRange
from ..shared.dataset.utils import group_by_documents
from ..shared.dataset.document import DocDict, get_file_url, Document
from ..shared.utils import get_device, hash_str
from ..shared.tasks import LoggedTask
from ..shared.utils.logging import serializer

from ..config.base import IS_CUDA

SimScore: TypeAlias = Tuple[float, int, int]
PairTuple: TypeAlias = Tuple[int, int, float, int, int]
PairList: TypeAlias = Set[PairTuple] | List[PairTuple]
DocRef: TypeAlias = Tuple[str, str]
Pair: TypeAlias = Tuple[Tuple[DocInRange, int], Tuple[DocInRange, int], SimScore]


def region_key(img_id: str) -> str:
    """
    Name of a region whatever the writing of its bbox, to match the front names with the API image ids:
    - url_list ids (API) use the aiiinotate xywh as is: "12.6" or "-3" are possible
    - front names clamp negative coordinates at 0 (get_coord_from_annotation) and truncate decimals (parse_bbox, norm_bbox)
    → each coordinate is truncated, then clamped at 0
    """
    img_id = img_id.removesuffix(".jpg")
    stem, _, bbox = img_id.rpartition("_")
    try:
        x, y, w, h = (max(0, int(float(c))) for c in bbox.split(","))
    except ValueError:
        return img_id  # not a region
    return f"{stem}_{x},{y},{w},{h}"


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
    tops = np.argsort(-scores, axis=1)[:, : int(topk)]  # Negative for descending order

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
    def __init__(self, doc1: DocInRange, doc2: DocInRange):
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
        """
        Retain top-K matches from both row and column perspectives, merged into sparse matrix
        A pair (i, j) is kept if j is in row i top-K or i is in column j top-K.
        """
        _extend_from_dense_scores(self, scores, topk, tr_i, tr_j)
        if self.doc1 != self.doc2:
            _extend_from_dense_scores(self.transposed(), scores.T, topk, tr_i.T, tr_j.T)


class TransposedSimMatrix:
    def __init__(self, obj: SparseDocSimMatrix):
        self.obj = obj

    def __getitem__(self, key: Tuple[int, int]) -> SimScore:
        i, j = key
        return self.obj[j, i]

    def __setitem__(self, key: Tuple[int, int], value: SimScore) -> None:
        i, j = key
        self.obj[j, i] = value

    def untransposed(self) -> SparseDocSimMatrix:
        return self.obj


class BlockSimMatrix:
    def __init__(self):
        """
        Data structure to store pairwise document similarity matrices
        """
        self.data: Dict[Tuple[str, str], SparseDocSimMatrix] = OrderedDict()

    def __getitem__(
        self, docs: Tuple[DocInRange, DocInRange]
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

    def __str__(self):
        pairs = self.doc_pairs
        return ", ".join([f"{p[0]}-{p[1]}" for p in pairs])

    def absolute_pairs(self) -> Iterable[PairTuple]:
        for matrix in self.data.values():
            yield from matrix.absolute_pairs()

    def blocks(self) -> Iterable[SparseDocSimMatrix]:
        return self.data.values()

    @property
    def doc_pairs(self) -> Iterable[Tuple[Document, Document]]:
        pairs = []
        for matrix in self.data.values():
            pairs.append((matrix.doc1.document.uid, matrix.doc2.document.uid))
        return pairs


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
        self.topk = int(parameters.get("cosine_n_filter", COS_TOPK))
        self.algorithm = parameters.get("algorithm", "cosine")

        # Whether to perform pre-filter using cosine similarity to keep only best matches before running segswap
        self.segswap_prefilter = parameters.get("segswap_prefilter", True)
        # If so, how many best matches should be kept
        self.segswap_n = int(parameters.get("segswap_n", COS_TOPK))

        self.raw_transpositions: List[str] = parameters.get("transpositions", ["none"])
        self.transpositions = [
            getattr(AllTranspose, t.upper()) for t in self.raw_transpositions
        ]
        self.skip_pairs: set[str] = set(parameters.get("skip_pairs", []))
        # pairs of regions to score whatever their rank or score (e.g. regions remapped after the computation)
        self.pairs: List[Tuple[str, str]] = parameters.get("pairs", [])
        # recompute document pairs even if a dataset score file exists
        self.force: bool = bool(parameters.get("force", False))

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

    def skip(self, uid1: str, uid2: str) -> bool:
        """Check if this pair should be skipped (explicitly listed by the front, in any order)"""
        skipping = (
            f"{uid1}-{uid2}" in self.skip_pairs or f"{uid2}-{uid1}" in self.skip_pairs
        )
        if skipping:
            self.log(f"Skipping {uid1} / {uid2}")
        return skipping

    @torch.no_grad()
    def get_features(self):
        """
        Extract features document by document, in the order of self.images
        Each document caches the features of its current list of images: a document whose images change is re-extracted alone
        """
        extractor = FeatureExtractor(
            feat_net=self.feat_net,
            device=self.device,
        )
        transpositions = "".join(str(t.value) for t in self.transpositions)

        features = torch.cat(
            [
                extractor.extract_features(
                    DataLoader(
                        FileListDataset(
                            data_paths=[str(im.path) for im in doc.images],
                            transform=extractor.transforms,
                            device=self.device,
                            transpositions=self.transpositions,
                        ),
                        batch_size=16,
                        shuffle=False,
                    ),
                    cache_dir=doc.document.path / "features",
                    cache_id=f"{hash_str(chr(10).join(im.id for im in doc.images))[:16]}@{transpositions}",
                ).cpu()
                for doc in self.doc_images
            ]
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
            "index": Dataset.serialize(
                documents=docs, transpositions=self.raw_transpositions
            ),
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
        doc_ids = self.dataset.doc_uid

        if len(self.images) == 0:
            raise Exception(f"No images found for {self.dataset.uid}")

        self.log(
            f"Prepared {len(self.images)} images to be processed from {len(doc_ids)} documents ({', '.join(doc_ids)})"
        )

        topk = self.segswap_n if self.algorithm == "segswap" else self.topk
        features = self.get_features().numpy()

        # TODO skip this step if self.algorithm == "segswap" && self.segswap_prefilter == false
        pairs = self.compute_cosine_similarity(
            features,
            topk=topk,
            n_transpositions=len(self.transpositions),
        )

        if self.algorithm == "segswap":
            if not IS_CUDA:
                self.log(
                    "Warning: SegSwap on CPU will be slow. Consider using cosine similarity."
                )
            pairs = self.compute_segswap_similarity(pairs, cos_topk=topk)

        self.log(f"Computed similarity scores for {len(pairs)} pairs of images")

        if self.pairs:
            self.compute_pairs(features, topk)

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
            self.log(f"Stored similarity scores inside {score_file}")

        if self.algorithm == algorithm:
            file_path = f"{self.experiment_id}/{result_name}"
            result_url = {
                "doc_pair": doc_ref,
                "result_url": get_file_url(DEMO_NAME, file_path),
            }

            self.add_results_url(result_url)

            if self.skip(matrix.doc1.document.uid, matrix.doc2.document.uid):
                # do not notify skipped pairs
                return

            self.log(f"Sending {doc_ref} results to front")  # marker
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
        doc_images = sorted(self.doc_images, key=lambda d: d.document.uid)

        self.log(f"Computing cosine similarity for {len(doc_images)} documents")

        all_scores = BlockSimMatrix()
        for doc1, doc2 in itertools.combinations_with_replacement(doc_images, 2):
            if self.skip(doc1.document.uid, doc2.document.uid):
                self.log(
                    f"Skipping cosine computation for: {doc1.document.uid} - {doc2.document.uid}"
                )
                continue

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
                np.fill_diagonal(sim_matrix, -1000)

            pairs = all_scores[doc1, doc2]
            pairs.extend_from_dense_scores(sim_matrix, topk, tr_i, tr_j)
            self.store(pairs, algorithm="cosine")

        return all_scores

    @cached_property
    def segswap_scorer(self):
        """
        Load SegSwap once, return a function scoring pairs (i, j) of indices of self.images by batches
        """
        device = self.device
        param = torch.load(
            get_model_path("hard_mining_neg5"), map_location=device, weights_only=False
        )
        backbone = segswap.load_backbone(param).to(device)
        encoder = segswap.load_encoder(param).to(device)

        feat_size = MAX_SIZE // SEG_STRIDE
        y_grid, x_grid = np.where(np.ones((feat_size, feat_size), dtype=bool))

        img_dataset = FileListDataset(
            data_paths=[str(i.path) for i in self.images],
            transform=transforms.Compose(
                [
                    transforms.Resize((segswap.MAX_SIZE, segswap.MAX_SIZE)),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            device=device,
        )

        def score(pairs: List[Tuple[int, int]], batch_size: int) -> List[float]:
            """Scores in the order of pairs; pairs are grouped by query image so that it is loaded and encoded once per batch"""
            scores = [0.0] * len(pairs)
            order = sorted(range(len(pairs)), key=lambda k: pairs[k])
            # TODO make a dataloader
            for i, group in itertools.groupby(order, key=lambda k: pairs[k][0]):
                group = list(group)
                q_tensor = img_dataset[i].unsqueeze(0)
                for start in range(0, len(group), batch_size):
                    batch = group[start : start + batch_size]
                    batch_scores = segswap.compute_score(
                        q_tensor,
                        torch.stack([img_dataset[pairs[k][1]] for k in batch]),
                        backbone,
                        encoder,
                        y_grid,
                        x_grid,
                    )
                    for k, s in zip(batch, batch_scores):
                        scores[k] = float(s)
            return scores

        return score

    @torch.no_grad()
    def compute_segswap_similarity(
        self, input_pairs: BlockSimMatrix, cos_topk
    ) -> BlockSimMatrix:
        """
        Compute the similarity between pairs of images using the SegSwap algorithm

        Args:
            input_pairs (BlockSimMatrix): The input pairs of documents
            cos_topk (int): The number of best matches to return

        Returns:
            A list of pairs (k_i, k_j, sim, tr_i, tr_j)
        """
        self.log(
            f"Computing SegSwap similarity for {len(input_pairs.data)} pairs of documents ({input_pairs})"
        )

        segswap_scores = BlockSimMatrix()
        for block in input_pairs.blocks():
            doc1 = block.doc1
            doc2 = block.doc2
            if self.skip(doc1.document.uid, doc2.document.uid):
                self.log(
                    f"Skipping segswap computation for: {doc1.document.uid} - {doc2.document.uid}"
                )
                continue

            doc_scores = segswap_scores[doc1, doc2]
            rel_pairs = [(rel_i, rel_j) for (_, rel_i), (_, rel_j), _ in block]
            scores = self.segswap_scorer(
                [(doc1.range[i], doc2.range[j]) for i, j in rel_pairs], cos_topk
            )
            for (i, j), s in zip(rel_pairs, scores):
                doc_scores[i, j] = (round(s, 2), 0, 0)

            self.store(doc_scores, "segswap")

        return segswap_scores

    @torch.no_grad()
    def compute_pairs(self, features: np.ndarray, batch_size: int):
        """
        Score each pair of self.pairs, whatever its rank or score, and store them in "{algorithm}-pairs.json"
        Pairs are returned with the names sent by the front, so that it updates its rows without renaming
        """
        index = {region_key(im.id): k for k, im in enumerate(self.images)}
        found = [
            (a, b, index[ka], index[kb])
            for a, b in self.pairs
            if (ka := region_key(a)) in index and (kb := region_key(b)) in index
        ]
        if missing := len(self.pairs) - len(found):
            self.print_and_log_warning(
                f"[task.similarity] {missing} requested pairs have an image missing from the dataset"
            )
        if not found:
            return

        if self.algorithm == "segswap":
            scores = [
                round(s, 2)
                for s in self.segswap_scorer([(i, j) for *_, i, j in found], batch_size)
            ]
        else:
            n = len(self.transpositions)
            scores = [
                round(float(1.0 - cdist(features[i * n : (i + 1) * n], features[j * n : (j + 1) * n], metric="cosine").max()), 3)
                for *_, i, j in found
            ]

        names = list(dict.fromkeys(img for a, b, *_ in found for img in (a, b)))
        position = {img: k for k, img in enumerate(names)}
        result_name = f"{self.algorithm}-pairs"
        score_file = SCORES_PATH / self.experiment_id / f"{result_name}.json"
        score_file.parent.mkdir(parents=True, exist_ok=True)
        score_file.write_bytes(
            orjson.dumps(
                {
                    "parameters": self.format_parameters(),
                    "index": {"images": [{"id": img} for img in names]},
                    "pairs": [
                        (position[a], position[b], s, 0, 0)
                        for (a, b, *_), s in zip(found, scores)
                    ],
                }
            )
        )
        self.add_results_url(
            {
                "doc_pair": "pairs",
                "result_url": get_file_url(DEMO_NAME, f"{self.experiment_id}/{result_name}"),
            }
        )
        self.log(f"Scored {len(found)} requested pairs")

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

        scores, experiment_id = (
            (False, False)
            if self.pairs or self.force
            else self.check_already_computed()
        )
        if scores:
            response = {
                "dataset_url": self.dataset.get_absolute_url(),
                "results_url": self.get_results_url(experiment_id),
            }
            self.log(
                f"Similarity scores already computed for {self.dataset.uid} with {self.feat_net}"
            )
            self.log(response)
            return response

        self.log(
            f"Similarity task triggered for {self.dataset.uid} with {self.feat_net}!"
        )

        try:
            self.results = self.compute_similarity()

            if not self.skip_pairs:
                (SCORES_PATH / self.experiment_id).mkdir(parents=True, exist_ok=True)
                with open(SCORES_PATH / self.experiment_id / f"{self.dataset.uid}-scores.json", "wb") as f:
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


    def check_already_computed(self):
        for path in SCORES_PATH.rglob(f"{self.dataset.uid}-scores.json"):
            try:
                scores = orjson.loads(path.read_bytes())
            except (orjson.JSONDecodeError, OSError) as e:
                self.print_and_log_warning(f"[task.similarity] Error reading scores file {path}: {e}")
                continue

            if scores.get("parameters") == self.format_parameters():
                return scores, path.parent.name

        return False, False

    def check_dataset(self):
        if self.dataset is None:
            return False

        if len(self.dataset.documents) == 0:
            return False

        return True
