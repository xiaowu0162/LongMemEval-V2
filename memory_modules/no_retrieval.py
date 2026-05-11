from .memory import Memory, MemoryContextItem, register_memory


@register_memory
class NoRetrievalMemory(Memory):
    """Memory backend for the paper's no-retrieval baseline."""

    memory_type = "no_retrieval"

    def insert(self, trajectory: dict[str, object]) -> None:
        return None

    def query(
        self,
        query: str,
        query_image: str | None = None,
    ) -> list[MemoryContextItem]:
        return []

