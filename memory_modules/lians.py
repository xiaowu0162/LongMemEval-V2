from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
from typing import Any

from .memory import Memory, MemoryContextItem, register_memory, require


@register_memory
class LiansMemory(Memory):
    """LongMemEval-V2 adapter for the local Lians memory engine.

    One memory is stored per trajectory state. Dataset order is mapped to
    deterministic event times because trajectories lack wall-clock timestamps.
    """

    memory_type = "lians"

    @classmethod
    def reconcile_loaded_memory_config(cls, saved_config, requested_config):
        require(saved_config["memory_type"] == cls.memory_type, "saved memory is not lians")
        if requested_config is not None:
            require(requested_config["memory_type"] == cls.memory_type, "requested memory is not lians")
            saved = {k: v for k, v in saved_config["memory_params"].items()
                     if k not in {"workspace_dir", "trajectories_root_dir"}}
            requested = {k: v for k, v in requested_config["memory_params"].items()
                         if k not in {"workspace_dir", "trajectories_root_dir"}}
            require(saved == requested, "requested Lians parameters do not match saved memory")
        return saved_config

    def __init__(self, memory_params: dict[str, object]) -> None:
        super().__init__(memory_params)
        allowed = {"workspace_dir", "trajectories_root_dir", "agent_id", "top_k", "embedding_provider"}
        unexpected = sorted(set(memory_params) - allowed)
        require(not unexpected, f"lians config contains unexpected keys: {unexpected}")
        self.agent_id = str(memory_params.get("agent_id", "longmemeval-v2")).strip()
        self.top_k = int(memory_params.get("top_k", 8))
        self.embedding_provider = str(memory_params.get("embedding_provider", "sentence-transformers")).strip()
        require(self.agent_id, "lians agent_id must be non-empty")
        require(self.top_k > 0, "lians top_k must be positive")
        workspace = memory_params.get("workspace_dir")
        require(isinstance(workspace, str) and workspace.strip(), "lians requires workspace_dir")
        self.workspace_dir = Path(workspace).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        trajectories_root = memory_params.get("trajectories_root_dir")
        self.trajectories_root_dir = (
            Path(trajectories_root).resolve()
            if isinstance(trajectories_root, str) and trajectories_root.strip()
            else None
        )
        self.db_path = self.workspace_dir / "lians.db"
        self._client = self._build_client()
        self._next_state_offset = 0

    def _build_client(self):
        try:
            from lians import LocalLiansClient
        except ImportError as exc:
            raise RuntimeError("LiansMemory requires `pip install lians-sdk[local]`") from exc
        return LocalLiansClient(
            db_path=str(self.db_path), namespace="longmemeval-v2",
            embedding_provider=self.embedding_provider,
        )

    @staticmethod
    def _state_parts(state: dict[str, Any]) -> tuple[str, str | None]:
        observation = state.get("observation")
        observation = observation if isinstance(observation, dict) else {}
        text = state.get("accessibility_tree", state.get("text", observation.get("text", "")))
        screenshot = state.get("screenshot", observation.get("screenshot"))
        fields = [("URL", state.get("url")), ("Action", state.get("action")),
                  ("Thoughts", state.get("thought", state.get("thoughts"))), ("Page", text)]
        rendered = "\n".join(f"{label}: {value}" for label, value in fields
                             if isinstance(value, str) and value.strip())
        return rendered, screenshot if isinstance(screenshot, str) else None

    def insert(self, trajectory: dict[str, object]) -> None:
        trajectory_id = trajectory.get("id")
        require(isinstance(trajectory_id, str) and trajectory_id, "trajectory id must be non-empty")
        states = trajectory.get("states")
        if not isinstance(states, list):
            states = trajectory.get("content")
        require(isinstance(states, list), f"trajectory {trajectory_id} states/content must be a list")
        base_time = datetime(2000, 1, 1, tzinfo=timezone.utc)
        items: list[dict[str, Any]] = []
        for state_index, state in enumerate(states):
            require(isinstance(state, dict), f"trajectory {trajectory_id} state must be an object")
            content, screenshot = self._state_parts(state)
            if not content:
                continue
            items.append({
                "content": f"Trajectory {trajectory_id}, state {state_index}\n{content}",
                "event_time": base_time + timedelta(seconds=self._next_state_offset + state_index),
                "source": f"longmemeval-v2:{trajectory_id}:{state_index}",
                "metadata": {"trajectory_id": trajectory_id, "state_index": state_index,
                             "screenshot": screenshot, "synthetic_event_time": True},
                "importance": 0.5,
            })
        if items:
            self._client.add_batch(agent_id=self.agent_id, items=items)
        self._next_state_offset += max(len(states), 1)

    def query(self, query: str, query_image: str | None = None) -> list[MemoryContextItem]:
        result = self._client.recall(agent_id=self.agent_id, query=query, k=self.top_k)
        context: list[MemoryContextItem] = []
        for memory in result.get("memories", []):
            content = memory.get("content")
            if isinstance(content, str) and content.strip():
                context.append({"type": "text", "value": content})
            metadata = memory.get("metadata")
            screenshot = metadata.get("screenshot") if isinstance(metadata, dict) else None
            if isinstance(screenshot, str):
                screenshot_path = Path(screenshot)
                if not screenshot_path.is_absolute() and self.trajectories_root_dir is not None:
                    candidates = [self.trajectories_root_dir / screenshot_path,
                                  self.trajectories_root_dir / "screenshots" / screenshot_path]
                    screenshot_path = next((path for path in candidates if path.exists()), screenshot_path)
                if screenshot_path.exists():
                    context.append({"type": "image", "value": str(screenshot_path.resolve())})
        return context

    def _save_backend(self, output_dir: Path) -> None:
        self.close()
        shutil.copy2(self.db_path, output_dir / "lians.db")
        portable_config = self.memory_config
        portable_config["memory_params"]["workspace_dir"] = str((output_dir / "workspace").resolve())
        (output_dir / "memory_config.json").write_text(
            json.dumps(portable_config, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        self._client = self._build_client()

    def _load_backend(self, input_dir: Path) -> None:
        saved_db = input_dir / "lians.db"
        require(saved_db.exists(), f"Missing saved Lians database: {saved_db}")
        self.close()
        if saved_db.resolve() != self.db_path.resolve():
            shutil.copy2(saved_db, self.db_path)
        self._client = self._build_client()

    def close(self) -> None:
        client = getattr(self, "_client", None)
        if client is not None:
            client.close()
            self._client = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
