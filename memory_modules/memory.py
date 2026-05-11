from copy import deepcopy
import json
from abc import ABC, abstractmethod
from pathlib import Path
import threading
from typing import Literal, TypedDict


class MemoryConfig(TypedDict):
    memory_type: str
    memory_params: dict[str, object]


class MemoryContextItem(TypedDict):
    type: Literal["text", "image"]
    value: str


def require(condition: bool, message: str) -> None:
    """Raise a runtime error when a required condition is not met."""
    if not condition:
        raise RuntimeError(message)


class Memory(ABC):
    """Base interface for memory backends used by the evaluation harness."""

    memory_type: str = ""

    def __init__(self, memory_params: dict[str, object]) -> None:
        self.memory_params = dict(memory_params)
        self._query_context_local = threading.local()

    @property
    def memory_config(self) -> MemoryConfig:
        """Return the minimal config needed to reconstruct this memory."""
        return {
            "memory_type": self.memory_type,
            "memory_params": self.memory_params,
        }

    @abstractmethod
    def insert(self, trajectory: dict[str, object]) -> None:
        """Index one full trajectory object into the backend."""
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        query: str,
        query_image: str | None = None,
    ) -> list[MemoryContextItem]:
        """Return a formatted memory context payload for a query."""
        raise NotImplementedError

    def configure_runtime(self, **kwargs: object) -> None:
        """Apply non-persisted runtime overrides after build/load."""
        return None

    def post_query_hook(
        self,
        *,
        query: str,
        query_image: str | None,
        memory_context: list[MemoryContextItem],
    ) -> dict[str, object] | None:
        """Run optional synchronous post-query work after retrieval."""
        return None

    @classmethod
    def reconcile_loaded_memory_config(
        cls,
        saved_config: MemoryConfig,
        requested_config: MemoryConfig | None,
    ) -> MemoryConfig:
        """Resolve the effective config when loading a persisted memory."""
        require(
            saved_config["memory_type"] == cls.memory_type,
            (
                "Saved memory config type does not match the receiving memory class: "
                f"{saved_config['memory_type']} vs {cls.memory_type}"
            ),
        )
        if requested_config is None:
            return deepcopy(saved_config)
        require(
            requested_config["memory_type"] == cls.memory_type,
            (
                "Requested memory config type does not match the saved memory type: "
                f"{requested_config['memory_type']} vs {cls.memory_type}"
            ),
        )
        require(
            requested_config == saved_config,
            (
                f"{cls.memory_type} requires the requested memory config to exactly match "
                "the saved memory config when loading a prebuilt memory artifact"
            ),
        )
        return deepcopy(saved_config)

    def set_query_context(self, **kwargs: object) -> None:
        """Set thread-local context for the next query call."""
        self._query_context_local.context = dict(kwargs)

    def clear_query_context(self) -> None:
        """Clear thread-local query context for this worker thread."""
        if hasattr(self._query_context_local, "context"):
            delattr(self._query_context_local, "context")

    def get_query_context(self) -> dict[str, object]:
        """Return thread-local query context for this worker thread."""
        context = getattr(self._query_context_local, "context", None)
        if isinstance(context, dict):
            return dict(context)
        return {}

    def save_memory(self, output_dir: str | Path) -> None:
        """Persist the memory config and backend state to a directory."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "memory_config.json").write_text(
            json.dumps(self.memory_config, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        self._save_backend(path)

    def _save_backend(self, output_dir: Path) -> None:
        """Persist backend-specific state inside an existing save directory."""
        return None

    def _load_backend(self, input_dir: Path) -> None:
        """Restore backend-specific state from an existing save directory."""
        return None


MEMORY_TYPES: dict[str, type[Memory]] = {}


def register_memory(memory_cls: type[Memory]) -> type[Memory]:
    """Register a concrete memory class under its declared memory_type."""
    require(memory_cls.memory_type, "memory_type must be non-empty")
    MEMORY_TYPES[memory_cls.memory_type] = memory_cls
    return memory_cls


def _validate_memory_config(memory_config_obj: object) -> MemoryConfig:
    """Validate a loaded memory config object and normalize its fields."""
    require(isinstance(memory_config_obj, dict), "Memory config must be a JSON object")
    memory_type = memory_config_obj.get("memory_type")
    memory_params = memory_config_obj.get("memory_params")
    require(
        isinstance(memory_type, str) and memory_type,
        "Memory config missing non-empty memory_type",
    )
    require(
        isinstance(memory_params, dict),
        "Memory config missing object memory_params",
    )
    require(
        all(isinstance(key, str) and key for key in memory_params),
        "memory_params keys must be non-empty strings",
    )
    require(memory_type in MEMORY_TYPES, f"Unknown memory_type: {memory_type}")
    return {
        "memory_type": memory_type,
        "memory_params": dict(memory_params),
    }


def load_memory_config(memory_config_path: str | Path) -> MemoryConfig:
    """Load and validate a memory config JSON file."""
    path = Path(memory_config_path)
    require(path.exists(), f"Missing memory config file: {path}")
    return _validate_memory_config(json.loads(path.read_text(encoding="utf-8")))


def build_memory(memory_config: MemoryConfig | str | Path) -> Memory:
    """Construct a memory instance from a config object or config path."""
    if isinstance(memory_config, (str, Path)):
        config = load_memory_config(memory_config)
    else:
        config = _validate_memory_config(memory_config)
    memory_cls = MEMORY_TYPES[config["memory_type"]]
    return memory_cls(config["memory_params"])


def save_memory(memory: Memory, output_dir: str | Path) -> None:
    """Persist a memory instance to a directory."""
    memory.save_memory(output_dir)


def load_memory(
    input_dir: str | Path,
    requested_config: MemoryConfig | str | Path | None = None,
) -> Memory:
    """Load a persisted memory instance from a save directory."""
    path = Path(input_dir)
    require(path.exists(), f"Missing memory save dir: {path}")
    require(path.is_dir(), f"Memory save path must be a directory: {path}")
    config_path = path / "memory_config.json"
    saved_config = load_memory_config(config_path)
    requested_config_obj: MemoryConfig | None = None
    if requested_config is not None:
        if isinstance(requested_config, (str, Path)):
            requested_config_obj = load_memory_config(requested_config)
        else:
            requested_config_obj = _validate_memory_config(requested_config)
    memory_cls = MEMORY_TYPES[saved_config["memory_type"]]
    effective_config = memory_cls.reconcile_loaded_memory_config(
        saved_config,
        requested_config_obj,
    )
    memory = build_memory(effective_config)
    memory._load_backend(path)
    return memory


from .no_retrieval import NoRetrievalMemory  # noqa: E402,F401
from .codex import CodexMemory  # noqa: E402,F401
from .agentrunbook_c import AgentRunbookC  # noqa: E402,F401
from .agentrunbook_r import AgentRunbookR  # noqa: E402,F401
from .rag import RagMemory  # noqa: E402,F401
