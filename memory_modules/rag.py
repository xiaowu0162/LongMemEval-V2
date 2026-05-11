from __future__ import annotations

from copy import deepcopy
import json
import shutil
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .agentrunbook_r import (
    DEFAULT_CONTROLLER_API_KEY_ENV,
    DEFAULT_CONTROLLER_BASE_URL,
    DEFAULT_CONTROLLER_DISABLE_THINKING,
    DEFAULT_CONTROLLER_MAX_COMPLETION_TOKENS,
    DEFAULT_CONTROLLER_MAX_RETRIES,
    DEFAULT_CONTROLLER_MODEL,
    DEFAULT_CONTROLLER_TEMPERATURE,
    DEFAULT_CONTROLLER_TIMEOUT_SECONDS,
    DEFAULT_CONTROLLER_TOP_K,
    DEFAULT_CONTROLLER_TOP_P,
    DEFAULT_EMBEDDING_API_KEY_ENV,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_MAX_INPUT_TOKENS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_QUERY_INSTRUCTION,
    DEFAULT_NOTE_TOP_K,
    DEFAULT_RAW_STATE_SLICE_RADIUS,
    DEFAULT_RAW_STATE_TOP_K,
    NOTE_GENERATION_PROMPT_VERSION,
    PREVIEW_EXAMPLE_COUNT,
    PREVIEW_TEXT_MAX_CHARS,
    TRAJECTORY_ARTIFACT_DIRNAME,
    TRAJECTORY_ARTIFACT_HINT_EMBEDDING_FILENAME,
    TRAJECTORY_ARTIFACT_HINT_NOTE_FILENAME,
    TRAJECTORY_ARTIFACT_INDEX_FILENAME,
    TRAJECTORY_ARTIFACT_PROCEDURE_EMBEDDING_FILENAME,
    TRAJECTORY_ARTIFACT_PROCEDURE_NOTE_FILENAME,
    TRAJECTORY_ARTIFACT_RAW_STATE_EMBEDDINGS_FILENAME,
    TRAJECTORY_ARTIFACT_RAW_STATE_FILENAME,
    TRAJECTORY_ARTIFACT_SCHEMA_VERSION,
    AgentRunbookR,
    _append_embedding_rows,
    _domain_from_url,
    _first_clause,
    _read_jsonl,
    _truncate_middle,
    _write_jsonl,
)
from .memory import Memory, MemoryConfig, MemoryContextItem, register_memory, require
from .trajectory_store import (
    load_json,
    materialize_prepared_trajectory,
    normalize_trajectory_pool_root,
    prepare_trajectory_insert,
    relative_symlink,
    save_json,
    validate_pooled_trajectory_dir,
)


def _empty_embedding_matrix() -> np.ndarray:
    return np.zeros((0, 0), dtype=np.float32)


@register_memory
class RagMemory(Memory):
    memory_type = "rag"

    _ensure_workspace_layout = AgentRunbookR._ensure_workspace_layout
    _build_controller_request = AgentRunbookR._build_controller_request
    _call_controller_text = AgentRunbookR._call_controller_text
    _call_controller_content_text = AgentRunbookR._call_controller_content_text
    _get_controller_client = AgentRunbookR._get_controller_client
    _get_embedding_client = AgentRunbookR._get_embedding_client
    _get_embedding_tokenizer = AgentRunbookR._get_embedding_tokenizer
    _truncate_for_embedding = AgentRunbookR._truncate_for_embedding
    _format_query_for_embedding = AgentRunbookR._format_query_for_embedding
    _embed_texts = AgentRunbookR._embed_texts
    _build_raw_state_entries = AgentRunbookR._build_raw_state_entries
    _build_note_generation_messages = AgentRunbookR._build_note_generation_messages
    _build_note_repair_messages = AgentRunbookR._build_note_repair_messages
    _fallback_note_entry = AgentRunbookR._fallback_note_entry
    _build_note_entries = AgentRunbookR._build_note_entries
    _search_entries = AgentRunbookR._search_entries
    _build_note_context_items = AgentRunbookR._build_note_context_items
    _build_raw_state_context_items = AgentRunbookR._build_raw_state_context_items
    _load_embedding_matrix_file = AgentRunbookR._load_embedding_matrix_file
    _absolute_screenshot_path = AgentRunbookR._absolute_screenshot_path
    _load_stored_trajectory = AgentRunbookR._load_stored_trajectory

    @classmethod
    def _normalize_reconcilable_memory_config(cls, memory_config: MemoryConfig) -> MemoryConfig:
        require(
            memory_config["memory_type"] == cls.memory_type,
            f"Expected memory_type={cls.memory_type}, got {memory_config['memory_type']}",
        )
        memory_params_obj = memory_config.get("memory_params")
        require(
            isinstance(memory_params_obj, dict),
            "rag memory_params must be an object",
        )
        memory_params = dict(memory_params_obj)
        allowed_top_level_keys = {
            "trajectory_pool_root",
            "workspace_dir",
            "trajectories_root_dir",
            "controller_params",
            "embedding_params",
            "index_params",
            "retrieval_params",
        }
        unexpected_top_level_keys = sorted(set(memory_params) - allowed_top_level_keys)
        require(
            not unexpected_top_level_keys,
            (
                "rag config contains unexpected memory_params keys: "
                f"{unexpected_top_level_keys}"
            ),
        )

        controller_params_obj = memory_params.get("controller_params", {})
        embedding_params_obj = memory_params.get("embedding_params", {})
        index_params_obj = memory_params.get("index_params", {})
        retrieval_params_obj = memory_params.get("retrieval_params", {})
        require(
            isinstance(controller_params_obj, dict),
            "rag controller_params must be an object",
        )
        require(
            isinstance(embedding_params_obj, dict),
            "rag embedding_params must be an object",
        )
        require(
            isinstance(index_params_obj, dict),
            "rag index_params must be an object",
        )
        require(
            isinstance(retrieval_params_obj, dict),
            "rag retrieval_params must be an object",
        )

        controller_params = dict(controller_params_obj)
        embedding_params = dict(embedding_params_obj)
        index_params = dict(index_params_obj)
        retrieval_params = dict(retrieval_params_obj)

        expected_controller_keys = {
            "model",
            "base_url",
            "api_key_env",
            "api_key_file",
            "max_completion_tokens",
            "timeout_seconds",
            "max_retries",
            "disable_thinking",
            "temperature",
            "top_p",
            "top_k",
        }
        expected_embedding_keys = {
            "model",
            "base_url",
            "api_key_env",
            "api_key_file",
            "max_input_tokens",
            "query_instruction",
        }
        expected_index_keys = {"raw_state_slice_radius"}
        expected_retrieval_keys = {
            "enable_notes",
            "raw_state_search_top_k",
            "note_search_top_k_per_type",
        }

        require(
            not (set(controller_params) - expected_controller_keys),
            (
                "rag controller_params contains unexpected keys: "
                f"{sorted(set(controller_params) - expected_controller_keys)}"
            ),
        )
        require(
            not (set(embedding_params) - expected_embedding_keys),
            (
                "rag embedding_params contains unexpected keys: "
                f"{sorted(set(embedding_params) - expected_embedding_keys)}"
            ),
        )
        require(
            not (set(index_params) - expected_index_keys),
            (
                "rag index_params contains unexpected keys: "
                f"{sorted(set(index_params) - expected_index_keys)}"
            ),
        )
        require(
            not (set(retrieval_params) - expected_retrieval_keys),
            (
                "rag retrieval_params contains unexpected keys: "
                f"{sorted(set(retrieval_params) - expected_retrieval_keys)}"
            ),
        )

        enable_notes = retrieval_params.get("enable_notes")
        require(
            isinstance(enable_notes, bool),
            "rag retrieval_params.enable_notes must be a boolean",
        )

        raw_state_slice_radius = int(
            index_params.get("raw_state_slice_radius", DEFAULT_RAW_STATE_SLICE_RADIUS)
        )
        raw_state_search_top_k = int(
            retrieval_params.get("raw_state_search_top_k", DEFAULT_RAW_STATE_TOP_K)
        )
        note_search_top_k_per_type = int(
            retrieval_params.get("note_search_top_k_per_type", DEFAULT_NOTE_TOP_K)
        )

        require(
            raw_state_slice_radius >= 0,
            "rag raw_state_slice_radius must be non-negative",
        )
        require(
            raw_state_search_top_k > 0,
            "rag raw_state_search_top_k must be positive",
        )
        require(
            note_search_top_k_per_type > 0,
            "rag note_search_top_k_per_type must be positive",
        )

        normalized: MemoryConfig = {
            "memory_type": cls.memory_type,
            "memory_params": {
                "trajectory_pool_root": memory_params.get("trajectory_pool_root"),
                "workspace_dir": memory_params.get("workspace_dir"),
                "trajectories_root_dir": memory_params.get("trajectories_root_dir"),
                "controller_params": {
                    "model": str(controller_params.get("model", DEFAULT_CONTROLLER_MODEL)).strip(),
                    "base_url": str(controller_params.get("base_url", DEFAULT_CONTROLLER_BASE_URL)).strip(),
                    "api_key_env": str(
                        controller_params.get("api_key_env", DEFAULT_CONTROLLER_API_KEY_ENV)
                    ).strip(),
                    "api_key_file": (
                        str(controller_params["api_key_file"]).strip()
                        if isinstance(controller_params.get("api_key_file"), str)
                        and str(controller_params.get("api_key_file")).strip()
                        else None
                    ),
                    "max_completion_tokens": int(
                        controller_params.get(
                            "max_completion_tokens",
                            DEFAULT_CONTROLLER_MAX_COMPLETION_TOKENS,
                        )
                    ),
                    "timeout_seconds": float(
                        controller_params.get("timeout_seconds", DEFAULT_CONTROLLER_TIMEOUT_SECONDS)
                    ),
                    "max_retries": int(
                        controller_params.get("max_retries", DEFAULT_CONTROLLER_MAX_RETRIES)
                    ),
                    "disable_thinking": bool(
                        controller_params.get("disable_thinking", DEFAULT_CONTROLLER_DISABLE_THINKING)
                    ),
                    "temperature": float(
                        controller_params.get("temperature", DEFAULT_CONTROLLER_TEMPERATURE)
                    ),
                    "top_p": float(controller_params.get("top_p", DEFAULT_CONTROLLER_TOP_P)),
                    "top_k": int(controller_params.get("top_k", DEFAULT_CONTROLLER_TOP_K)),
                },
                "embedding_params": {
                    "model": str(embedding_params.get("model", DEFAULT_EMBEDDING_MODEL)).strip(),
                    "base_url": str(embedding_params.get("base_url", DEFAULT_EMBEDDING_BASE_URL)).strip(),
                    "api_key_env": str(
                        embedding_params.get("api_key_env", DEFAULT_EMBEDDING_API_KEY_ENV)
                    ).strip(),
                    "api_key_file": (
                        str(embedding_params["api_key_file"]).strip()
                        if isinstance(embedding_params.get("api_key_file"), str)
                        and str(embedding_params.get("api_key_file")).strip()
                        else None
                    ),
                    "max_input_tokens": int(
                        embedding_params.get("max_input_tokens", DEFAULT_EMBEDDING_MAX_INPUT_TOKENS)
                    ),
                    "query_instruction": str(
                        embedding_params.get("query_instruction", DEFAULT_EMBEDDING_QUERY_INSTRUCTION)
                    ).strip(),
                },
                "index_params": {
                    "raw_state_slice_radius": raw_state_slice_radius,
                },
                "retrieval_params": {
                    "enable_notes": enable_notes,
                    "raw_state_search_top_k": raw_state_search_top_k,
                    "note_search_top_k_per_type": note_search_top_k_per_type,
                },
            },
        }
        return normalized

    @classmethod
    def _loaded_config_nonquery_signature(cls, memory_config: MemoryConfig) -> dict[str, Any]:
        memory_params = memory_config["memory_params"]
        retrieval_params = memory_params["retrieval_params"]
        return {
            "memory_type": memory_config["memory_type"],
            "enable_notes": retrieval_params["enable_notes"],
            "embedding_params": deepcopy(memory_params["embedding_params"]),
            "index_params": deepcopy(memory_params["index_params"]),
        }

    @classmethod
    def reconcile_loaded_memory_config(
        cls,
        saved_config: MemoryConfig,
        requested_config: MemoryConfig | None,
    ) -> MemoryConfig:
        saved_normalized = cls._normalize_reconcilable_memory_config(saved_config)
        if requested_config is None:
            return deepcopy(saved_normalized)

        require(
            requested_config["memory_type"] == cls.memory_type,
            (
                "Requested memory config type does not match saved "
                "rag artifact type: "
                f"{requested_config['memory_type']} vs {cls.memory_type}"
            ),
        )
        requested_normalized = cls._normalize_reconcilable_memory_config(requested_config)

        saved_nonquery_signature = cls._loaded_config_nonquery_signature(saved_normalized)
        requested_nonquery_signature = cls._loaded_config_nonquery_signature(requested_normalized)
        require(
            saved_nonquery_signature == requested_nonquery_signature,
            (
                "rag prebuilt-memory loading only allows "
                "retrieval-side config changes within the same variant. Non-query "
                "fields in the requested config must exactly match the saved "
                "artifact config."
            ),
        )

        effective = deepcopy(saved_normalized)
        effective["memory_params"]["controller_params"] = deepcopy(
            requested_normalized["memory_params"]["controller_params"]
        )
        effective["memory_params"]["retrieval_params"] = deepcopy(
            requested_normalized["memory_params"]["retrieval_params"]
        )
        return effective

    def __init__(self, memory_params: dict[str, object]) -> None:
        super().__init__(memory_params)
        normalized = self._normalize_reconcilable_memory_config(
            {
                "memory_type": self.memory_type,
                "memory_params": dict(memory_params),
            }
        )
        params = normalized["memory_params"]

        workspace_dir = params.get("workspace_dir")
        trajectories_root_dir = params.get("trajectories_root_dir")
        trajectory_pool_root = params.get("trajectory_pool_root")
        controller_params = dict(params["controller_params"])
        embedding_params = dict(params["embedding_params"])
        index_params = dict(params["index_params"])
        retrieval_params = dict(params["retrieval_params"])

        require(
            workspace_dir is None or (isinstance(workspace_dir, str) and workspace_dir.strip()),
            "rag workspace_dir must be null or a non-empty string",
        )
        require(
            trajectories_root_dir is None
            or (isinstance(trajectories_root_dir, str) and trajectories_root_dir.strip()),
            "rag trajectories_root_dir must be null or a non-empty string",
        )
        require(
            trajectory_pool_root is None
            or (isinstance(trajectory_pool_root, str) and trajectory_pool_root.strip()),
            "rag trajectory_pool_root must be null or a non-empty string",
        )

        self.workspace_dir = (
            Path(workspace_dir).resolve()
            if isinstance(workspace_dir, str) and workspace_dir.strip()
            else None
        )
        self.trajectories_root_dir = (
            Path(trajectories_root_dir).resolve()
            if isinstance(trajectories_root_dir, str) and trajectories_root_dir.strip()
            else None
        )
        self.trajectory_pool_root = None
        if isinstance(trajectory_pool_root, str) and trajectory_pool_root.strip():
            self.trajectory_pool_root = normalize_trajectory_pool_root(Path(trajectory_pool_root))
            require(
                self.trajectory_pool_root.exists() and self.trajectory_pool_root.is_dir(),
                (
                    "rag trajectory_pool_root must point to an existing "
                    f"directory: {self.trajectory_pool_root}"
                ),
            )

        self.controller_model = str(controller_params["model"]).strip()
        self.controller_base_url = str(controller_params["base_url"]).strip()
        self.controller_api_key_env = str(controller_params["api_key_env"]).strip()
        controller_api_key_file = controller_params.get("api_key_file")
        self.controller_api_key_file = (
            str(controller_api_key_file).strip()
            if isinstance(controller_api_key_file, str) and controller_api_key_file.strip()
            else None
        )
        self.controller_max_completion_tokens = int(controller_params["max_completion_tokens"])
        self.controller_timeout_seconds = float(controller_params["timeout_seconds"])
        self.controller_max_retries = int(controller_params["max_retries"])
        self.controller_disable_thinking = bool(controller_params["disable_thinking"])
        self.controller_temperature = float(controller_params["temperature"])
        self.controller_top_p = float(controller_params["top_p"])
        self.controller_top_k = int(controller_params["top_k"])

        self.embedding_model = str(embedding_params["model"]).strip()
        self.embedding_base_url = str(embedding_params["base_url"]).strip()
        self.embedding_api_key_env = str(embedding_params["api_key_env"]).strip()
        embedding_api_key_file = embedding_params.get("api_key_file")
        self.embedding_api_key_file = (
            str(embedding_api_key_file).strip()
            if isinstance(embedding_api_key_file, str) and embedding_api_key_file.strip()
            else None
        )
        self.embedding_max_input_tokens = int(embedding_params["max_input_tokens"])
        self.embedding_query_instruction = str(embedding_params["query_instruction"]).strip()

        self.raw_state_slice_radius = int(index_params["raw_state_slice_radius"])
        self.enable_notes = bool(retrieval_params["enable_notes"])
        self.raw_state_search_top_k = int(retrieval_params["raw_state_search_top_k"])
        self.note_search_top_k_per_type = int(retrieval_params["note_search_top_k_per_type"])

        require(
            self.controller_model,
            "rag controller model must be non-empty",
        )
        require(
            self.controller_base_url,
            "rag controller base_url must be non-empty",
        )
        require(
            self.controller_api_key_env,
            "rag controller api_key_env must be non-empty",
        )
        require(
            self.embedding_model,
            "rag embedding model must be non-empty",
        )
        require(
            self.embedding_base_url,
            "rag embedding base_url must be non-empty",
        )
        require(
            self.embedding_api_key_env,
            "rag embedding api_key_env must be non-empty",
        )
        require(
            self.embedding_query_instruction,
            "rag embedding query_instruction must be non-empty",
        )

        self._runtime_local = threading.local()
        self._controller_client_init_lock = threading.Lock()
        self._embedding_client_init_lock = threading.Lock()
        self._async_controller_init_lock = threading.Lock()
        self._embedding_tokenizer_init_lock = threading.Lock()
        self._async_controller_client = None
        self._embedding_tokenizer = None

        self.inserted_trajectory_ids: list[str] = []
        self.raw_state_entries: list[dict[str, Any]] = []
        self.procedure_note_entries: list[dict[str, Any]] = []
        self.hint_note_entries: list[dict[str, Any]] = []
        self.raw_state_embeddings = _empty_embedding_matrix()
        self.procedure_note_embeddings = _empty_embedding_matrix()
        self.hint_note_embeddings = _empty_embedding_matrix()

        if self.workspace_dir is not None:
            self._ensure_workspace_layout(self.workspace_dir)

    @property
    def memory_config(self) -> MemoryConfig:
        memory_params: dict[str, Any] = {
            "trajectory_pool_root": (
                str(self.trajectory_pool_root) if self.trajectory_pool_root is not None else None
            ),
            "workspace_dir": str(self.workspace_dir) if self.workspace_dir is not None else None,
            "trajectories_root_dir": (
                str(self.trajectories_root_dir) if self.trajectories_root_dir is not None else None
            ),
            "controller_params": {
                "model": self.controller_model,
                "base_url": self.controller_base_url,
                "api_key_env": self.controller_api_key_env,
                "api_key_file": self.controller_api_key_file,
                "max_completion_tokens": self.controller_max_completion_tokens,
                "timeout_seconds": self.controller_timeout_seconds,
                "max_retries": self.controller_max_retries,
                "disable_thinking": self.controller_disable_thinking,
                "temperature": self.controller_temperature,
                "top_p": self.controller_top_p,
                "top_k": self.controller_top_k,
            },
            "embedding_params": {
                "model": self.embedding_model,
                "base_url": self.embedding_base_url,
                "api_key_env": self.embedding_api_key_env,
                "api_key_file": self.embedding_api_key_file,
                "max_input_tokens": self.embedding_max_input_tokens,
                "query_instruction": self.embedding_query_instruction,
            },
            "index_params": {
                "raw_state_slice_radius": self.raw_state_slice_radius,
            },
            "retrieval_params": {
                "enable_notes": self.enable_notes,
                "raw_state_search_top_k": self.raw_state_search_top_k,
                "note_search_top_k_per_type": self.note_search_top_k_per_type,
            },
        }
        return {
            "memory_type": self.memory_type,
            "memory_params": memory_params,
        }

    def configure_runtime(self, **kwargs: object) -> None:
        _ = kwargs
        self._get_embedding_client()
        self._get_embedding_tokenizer()
        return None

    def query(
        self,
        query: str,
        query_image: str | None = None,
    ) -> list[MemoryContextItem]:
        _ = query_image
        require(
            isinstance(query, str) and query.strip(),
            "rag query must be non-empty",
        )

        raw_results = self._search_entries(
            entries=self.raw_state_entries,
            embeddings=self.raw_state_embeddings,
            query_text=query,
            top_k=self.raw_state_search_top_k,
        )
        note_results = self._search_note_query(query) if self.enable_notes else {
            "procedure_results": [],
            "hint_results": [],
        }

        items: list[MemoryContextItem] = []
        if self.enable_notes:
            items.extend(self._build_note_context_items(query, note_results))
        items.extend(self._build_raw_state_context_items(raw_results))
        return items

    def _search_note_query(self, note_query: str) -> dict[str, list[dict[str, Any]]]:
        if not self.enable_notes or not note_query:
            return {"procedure_results": [], "hint_results": []}
        return {
            "procedure_results": self._search_entries(
                entries=self.procedure_note_entries,
                embeddings=self.procedure_note_embeddings,
                query_text=note_query,
                top_k=self.note_search_top_k_per_type,
            ),
            "hint_results": self._search_entries(
                entries=self.hint_note_entries,
                embeddings=self.hint_note_embeddings,
                query_text=note_query,
                top_k=self.note_search_top_k_per_type,
            ),
        }

    def insert(self, trajectory: dict[str, object]) -> None:
        require(
            self.workspace_dir is not None,
            "rag insert requires workspace_dir",
        )
        require(
            self.trajectories_root_dir is not None,
            "rag insert requires trajectories_root_dir",
        )
        prepared = prepare_trajectory_insert(
            trajectory,
            trajectories_root_dir=self.trajectories_root_dir,
        )
        trajectory_id = prepared.trajectory_id
        if trajectory_id in set(self.inserted_trajectory_ids):
            raise RuntimeError(f"Duplicate trajectory insert attempted: {trajectory_id}")

        trajectory_dir = self.workspace_dir / "trajectories" / trajectory_id
        require(
            not trajectory_dir.exists(),
            f"Refusing to overwrite existing trajectory dir: {trajectory_dir}",
        )

        pooled_trajectory_dir = (
            self.trajectory_pool_root / trajectory_id if self.trajectory_pool_root is not None else None
        )
        if pooled_trajectory_dir is not None and pooled_trajectory_dir.exists():
            artifact_bundle = self._load_trajectory_artifact_bundle_from_pool(
                prepared=prepared,
                pooled_trajectory_dir=pooled_trajectory_dir,
            )
            self._materialize_pooled_trajectory_evidence(
                pooled_trajectory_dir=pooled_trajectory_dir,
                trajectory_dir=trajectory_dir,
            )
        else:
            trajectory_dir.mkdir(parents=True, exist_ok=False)
            materialize_prepared_trajectory(prepared, trajectory_dir)
            artifact_bundle = self._build_trajectory_artifact_bundle(
                trajectory_dir=trajectory_dir,
                simplified_trajectory=prepared.simplified,
            )

        self.inserted_trajectory_ids.append(trajectory_id)
        self.raw_state_entries.extend(artifact_bundle["raw_state_entries"])
        self.raw_state_embeddings = _append_embedding_rows(
            self.raw_state_embeddings,
            artifact_bundle["raw_state_embeddings"],
        )

        if self.enable_notes:
            procedure_entry = artifact_bundle["procedure_note_entry"]
            hint_entry = artifact_bundle["hint_note_entry"]
            require(
                procedure_entry is not None and hint_entry is not None,
                "rag expected note entries when enable_notes=true",
            )
            self.procedure_note_entries.append(procedure_entry)
            self.hint_note_entries.append(hint_entry)
            self.procedure_note_embeddings = _append_embedding_rows(
                self.procedure_note_embeddings,
                artifact_bundle["procedure_note_embedding"],
            )
            self.hint_note_embeddings = _append_embedding_rows(
                self.hint_note_embeddings,
                artifact_bundle["hint_note_embedding"],
            )

    def _materialize_pooled_trajectory_evidence(
        self,
        *,
        pooled_trajectory_dir: Path,
        trajectory_dir: Path,
    ) -> None:
        trajectory_dir.mkdir(parents=True, exist_ok=False)
        relative_symlink(pooled_trajectory_dir / "trajectory.json", trajectory_dir / "trajectory.json")
        relative_symlink(pooled_trajectory_dir / "screenshots", trajectory_dir / "screenshots")

    def _build_trajectory_artifact_bundle(
        self,
        *,
        trajectory_dir: Path,
        simplified_trajectory: dict[str, Any],
    ) -> dict[str, Any]:
        raw_state_entries = self._build_raw_state_entries(simplified_trajectory)
        procedure_entry: dict[str, Any] | None = None
        hint_entry: dict[str, Any] | None = None
        if self.enable_notes:
            procedure_entry, hint_entry = self._build_note_entries(
                trajectory_dir=trajectory_dir,
                simplified_trajectory=simplified_trajectory,
            )

        return {
            "raw_state_entries": raw_state_entries,
            "raw_state_embeddings": self._embed_texts(
                [entry["slice_axtree_text"] for entry in raw_state_entries],
                is_query=False,
            ),
            "procedure_note_entry": procedure_entry,
            "hint_note_entry": hint_entry,
            "procedure_note_embedding": (
                self._embed_texts([procedure_entry["note_text"]], is_query=False)
                if procedure_entry is not None
                else _empty_embedding_matrix()
            ),
            "hint_note_embedding": (
                self._embed_texts([hint_entry["note_text"]], is_query=False)
                if hint_entry is not None
                else _empty_embedding_matrix()
            ),
        }

    def _load_trajectory_artifact_bundle_from_pool(
        self,
        *,
        prepared: Any,
        pooled_trajectory_dir: Path,
    ) -> dict[str, Any]:
        validate_pooled_trajectory_dir(prepared, pooled_trajectory_dir=pooled_trajectory_dir)
        artifact_dir = pooled_trajectory_dir / TRAJECTORY_ARTIFACT_DIRNAME
        require(
            artifact_dir.exists() and artifact_dir.is_dir(),
            (
                "Missing upstream V3 pooled artifact directory for "
                f"{prepared.trajectory_id}: {artifact_dir}"
            ),
        )
        index_payload = load_json(artifact_dir / TRAJECTORY_ARTIFACT_INDEX_FILENAME)
        require(
            isinstance(index_payload, dict),
            (
                "Upstream V3 pooled artifact index must be an object: "
                f"{artifact_dir / TRAJECTORY_ARTIFACT_INDEX_FILENAME}"
            ),
        )
        require(
            index_payload.get("artifact_schema_version") == TRAJECTORY_ARTIFACT_SCHEMA_VERSION,
            (
                f"Unsupported upstream V3 pooled artifact schema for {prepared.trajectory_id}: "
                f"{index_payload.get('artifact_schema_version')}"
            ),
        )
        require(
            index_payload.get("trajectory_id") == prepared.trajectory_id,
            f"Pooled artifact trajectory_id mismatch for {prepared.trajectory_id}",
        )
        require(
            index_payload.get("trajectory_fingerprint") == prepared.fingerprint,
            f"Pooled artifact trajectory fingerprint mismatch for {prepared.trajectory_id}",
        )

        config_snapshot = index_payload.get("config_snapshot", {})
        require(
            isinstance(config_snapshot, dict),
            f"Pooled artifact config_snapshot must be an object for {prepared.trajectory_id}",
        )
        require(
            config_snapshot.get("raw_state_slice_radius") == self.raw_state_slice_radius,
            (
                "Pooled artifact raw_state_slice_radius mismatch for "
                f"{prepared.trajectory_id}: expected {self.raw_state_slice_radius}, "
                f"got {config_snapshot.get('raw_state_slice_radius')}"
            ),
        )
        require(
            config_snapshot.get("embedding_model") == self.embedding_model,
            (
                "Pooled artifact embedding_model mismatch for "
                f"{prepared.trajectory_id}: expected {self.embedding_model}, "
                f"got {config_snapshot.get('embedding_model')}"
            ),
        )
        require(
            config_snapshot.get("embedding_max_input_tokens") == self.embedding_max_input_tokens,
            (
                "Pooled artifact embedding_max_input_tokens mismatch for "
                f"{prepared.trajectory_id}: expected {self.embedding_max_input_tokens}, "
                f"got {config_snapshot.get('embedding_max_input_tokens')}"
            ),
        )
        require(
            config_snapshot.get("embedding_query_instruction") == self.embedding_query_instruction,
            (
                "Pooled artifact embedding_query_instruction mismatch for "
                f"{prepared.trajectory_id}"
            ),
        )
        if self.enable_notes:
            require(
                config_snapshot.get("note_prompt_version") == NOTE_GENERATION_PROMPT_VERSION,
                (
                    "Pooled artifact note_prompt_version mismatch for "
                    f"{prepared.trajectory_id}: expected {NOTE_GENERATION_PROMPT_VERSION}, "
                    f"got {config_snapshot.get('note_prompt_version')}"
                ),
            )

        raw_state_entries = _read_jsonl(artifact_dir / TRAJECTORY_ARTIFACT_RAW_STATE_FILENAME)
        raw_state_embeddings = self._load_embedding_matrix_file(
            artifact_dir / TRAJECTORY_ARTIFACT_RAW_STATE_EMBEDDINGS_FILENAME,
            expected_rows=len(raw_state_entries),
            field_name="raw_state_embeddings",
        )

        procedure_entry: dict[str, Any] | None = None
        hint_entry: dict[str, Any] | None = None
        procedure_note_embedding = _empty_embedding_matrix()
        hint_note_embedding = _empty_embedding_matrix()
        if self.enable_notes:
            procedure_entry = load_json(artifact_dir / TRAJECTORY_ARTIFACT_PROCEDURE_NOTE_FILENAME)
            hint_entry = load_json(artifact_dir / TRAJECTORY_ARTIFACT_HINT_NOTE_FILENAME)
            require(
                isinstance(procedure_entry, dict),
                f"Procedure note payload must be an object for {prepared.trajectory_id}",
            )
            require(
                isinstance(hint_entry, dict),
                f"Hint note payload must be an object for {prepared.trajectory_id}",
            )
            procedure_note_embedding = self._load_embedding_matrix_file(
                artifact_dir / TRAJECTORY_ARTIFACT_PROCEDURE_EMBEDDING_FILENAME,
                expected_rows=1,
                field_name="procedure_note_embedding",
            )
            hint_note_embedding = self._load_embedding_matrix_file(
                artifact_dir / TRAJECTORY_ARTIFACT_HINT_EMBEDDING_FILENAME,
                expected_rows=1,
                field_name="hint_note_embedding",
            )

        return {
            "raw_state_entries": raw_state_entries,
            "raw_state_embeddings": raw_state_embeddings,
            "procedure_note_entry": procedure_entry,
            "hint_note_entry": hint_entry,
            "procedure_note_embedding": procedure_note_embedding,
            "hint_note_embedding": hint_note_embedding,
        }

    def _build_index_payload(self) -> dict[str, Any]:
        domain = "unknown"
        if self.workspace_dir is not None and self.inserted_trajectory_ids:
            first_path = self.workspace_dir / "trajectories" / self.inserted_trajectory_ids[0] / "trajectory.json"
            if first_path.exists():
                payload = load_json(first_path)
                if isinstance(payload, dict):
                    domain = _domain_from_url(str(payload.get("start_url", "")))
        return {
            "memory_type": self.memory_type,
            "domain": domain,
            "inserted_trajectory_ids": list(self.inserted_trajectory_ids),
            "trajectory_count": len(self.inserted_trajectory_ids),
            "entry_counts": {
                "raw_state": len(self.raw_state_entries),
                "procedure_notes": len(self.procedure_note_entries),
                "hint_notes": len(self.hint_note_entries),
            },
            "embedding_dimensions": {
                "raw_state": int(self.raw_state_embeddings.shape[1]) if self.raw_state_embeddings.size else 0,
                "procedure_notes": (
                    int(self.procedure_note_embeddings.shape[1])
                    if self.procedure_note_embeddings.size
                    else 0
                ),
                "hint_notes": int(self.hint_note_embeddings.shape[1]) if self.hint_note_embeddings.size else 0,
            },
            "controller_params": {
                "model": self.controller_model,
                "base_url": self.controller_base_url,
                "max_completion_tokens": self.controller_max_completion_tokens,
                "timeout_seconds": self.controller_timeout_seconds,
                "max_retries": self.controller_max_retries,
                "disable_thinking": self.controller_disable_thinking,
                "temperature": self.controller_temperature,
                "top_p": self.controller_top_p,
                "top_k": self.controller_top_k,
            },
            "embedding_params": {
                "model": self.embedding_model,
                "base_url": self.embedding_base_url,
                "max_input_tokens": self.embedding_max_input_tokens,
                "query_instruction": self.embedding_query_instruction,
            },
            "trajectory_pool_root": (
                str(self.trajectory_pool_root) if self.trajectory_pool_root is not None else None
            ),
            "index_params": {
                "raw_state_slice_radius": self.raw_state_slice_radius,
            },
            "retrieval_params": {
                "enable_notes": self.enable_notes,
                "raw_state_search_top_k": self.raw_state_search_top_k,
                "note_search_top_k_per_type": self.note_search_top_k_per_type,
            },
        }

    def _build_preview_payload(self) -> dict[str, Any]:
        def _note_example(entry: dict[str, Any]) -> dict[str, Any]:
            return {
                "entry_id": entry["entry_id"],
                "trajectory_id": entry["trajectory_id"],
                "title": entry["title"],
                "description": entry["description"],
                "content": _truncate_middle(str(entry["content"]), PREVIEW_TEXT_MAX_CHARS),
            }

        def _raw_state_example(entry: dict[str, Any]) -> dict[str, Any]:
            return {
                "entry_id": entry["entry_id"],
                "trajectory_id": entry["trajectory_id"],
                "slice_state_indexes": list(entry["slice_state_indexes"]),
                "slice_action_sequence": _truncate_middle(
                    str(entry["slice_action_sequence"]),
                    PREVIEW_TEXT_MAX_CHARS,
                ),
                "slice_axtree_text": _truncate_middle(
                    str(entry["slice_axtree_text"]),
                    PREVIEW_TEXT_MAX_CHARS,
                ),
            }

        payload = {
            "memory_type": self.memory_type,
            "workspace_dir": str(self.workspace_dir) if self.workspace_dir is not None else None,
            "trajectory_pool_root": (
                str(self.trajectory_pool_root) if self.trajectory_pool_root is not None else None
            ),
            "controller": {
                "model": self.controller_model,
                "base_url": self.controller_base_url,
            },
            "embedding": {
                "model": self.embedding_model,
                "base_url": self.embedding_base_url,
                "query_instruction": self.embedding_query_instruction,
            },
            "retrieval": {
                "enable_notes": self.enable_notes,
                "raw_state_search_top_k": self.raw_state_search_top_k,
                "note_search_top_k_per_type": self.note_search_top_k_per_type,
            },
            "counts": {
                "trajectories": len(self.inserted_trajectory_ids),
                "raw_state_entries": len(self.raw_state_entries),
                "procedure_note_entries": len(self.procedure_note_entries),
                "hint_note_entries": len(self.hint_note_entries),
            },
            "pools": {
                "raw_state": {
                    "description": "High-fidelity raw AXTree state slices retrieved directly with the original benchmark question as the embedding query.",
                    "examples": [_raw_state_example(entry) for entry in self.raw_state_entries[:PREVIEW_EXAMPLE_COUNT]],
                },
            },
        }
        if self.enable_notes:
            payload["pools"]["procedure_notes"] = {
                "description": "Trajectory-level reusable procedure notes.",
                "examples": [
                    _note_example(entry) for entry in self.procedure_note_entries[:PREVIEW_EXAMPLE_COUNT]
                ],
            }
            payload["pools"]["hint_notes"] = {
                "description": "Trajectory-level reusable hint notes.",
                "examples": [_note_example(entry) for entry in self.hint_note_entries[:PREVIEW_EXAMPLE_COUNT]],
            }
        return payload

    def _save_backend(self, output_dir: Path) -> None:
        self._ensure_workspace_layout(output_dir)
        save_json(output_dir / "index.json", self._build_index_payload())
        _write_jsonl(output_dir / "pools" / "raw_state_pool.jsonl", self.raw_state_entries)
        np.save(output_dir / "embeddings" / "raw_state.npy", self.raw_state_embeddings)

        if self.enable_notes:
            _write_jsonl(output_dir / "pools" / "procedure_note_pool.jsonl", self.procedure_note_entries)
            _write_jsonl(output_dir / "pools" / "hint_note_pool.jsonl", self.hint_note_entries)
            np.save(output_dir / "embeddings" / "procedure_notes.npy", self.procedure_note_embeddings)
            np.save(output_dir / "embeddings" / "hint_notes.npy", self.hint_note_embeddings)

        save_json(output_dir / "previews" / "query_prompt_preview.json", self._build_preview_payload())

        if self.workspace_dir is not None and self.workspace_dir.resolve() != output_dir.resolve():
            src_trajectories_dir = self.workspace_dir / "trajectories"
            dst_trajectories_dir = output_dir / "trajectories"
            if dst_trajectories_dir.exists():
                shutil.rmtree(dst_trajectories_dir)
            shutil.copytree(src_trajectories_dir, dst_trajectories_dir)

    def _load_backend(self, input_dir: Path) -> None:
        self.workspace_dir = input_dir.resolve()
        self._ensure_workspace_layout(self.workspace_dir)
        index_payload = load_json(self.workspace_dir / "index.json")
        require(
            isinstance(index_payload, dict),
            "rag index.json must be an object",
        )
        inserted_ids = index_payload.get("inserted_trajectory_ids")
        require(
            isinstance(inserted_ids, list) and all(isinstance(item, str) and item for item in inserted_ids),
            "rag index.json must contain inserted_trajectory_ids as non-empty strings",
        )
        self.inserted_trajectory_ids = list(inserted_ids)
        self.raw_state_entries = _read_jsonl(self.workspace_dir / "pools" / "raw_state_pool.jsonl")
        self.raw_state_embeddings = np.load(self.workspace_dir / "embeddings" / "raw_state.npy")

        if self.enable_notes:
            self.procedure_note_entries = _read_jsonl(self.workspace_dir / "pools" / "procedure_note_pool.jsonl")
            self.hint_note_entries = _read_jsonl(self.workspace_dir / "pools" / "hint_note_pool.jsonl")
            self.procedure_note_embeddings = np.load(self.workspace_dir / "embeddings" / "procedure_notes.npy")
            self.hint_note_embeddings = np.load(self.workspace_dir / "embeddings" / "hint_notes.npy")
        else:
            self.procedure_note_entries = []
            self.hint_note_entries = []
            self.procedure_note_embeddings = _empty_embedding_matrix()
            self.hint_note_embeddings = _empty_embedding_matrix()
