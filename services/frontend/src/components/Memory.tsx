// The memory panel: view and delete what the assistant has remembered (FR-4.5).
//
// Shows saved preferences, facts, habits, FAQs, and page-range summaries, with
// an optional kind filter. Deleting removes both the stored content and its
// vector embedding server-side.

import { useCallback, useEffect, useState } from "react";

import { deleteMemory, listMemories } from "../api/client";
import { MEMORY_KIND_LABELS, type MemoryKind, type StoredMemory } from "../api/types";
import { Alert, Badge, Button, EmptyState, FieldLabel, RefreshButton, Select } from "./ui";

export function Memory(): React.JSX.Element {
  const [memories, setMemories] = useState<StoredMemory[]>([]);
  const [filter, setFilter] = useState<MemoryKind | "">("");
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const page = await listMemories(filter || undefined);
    setMemories(page.items);
  }, [filter]);

  useEffect(() => {
    refresh().catch(() => setError("Couldn't load your memories."));
  }, [refresh]);

  const remove = async (id: string) => {
    setError(null);
    try {
      await deleteMemory(id);
      await refresh();
    } catch {
      setError("Couldn't delete that memory.");
    }
  };

  return (
    <section aria-labelledby="memory-heading" className="mx-auto max-w-2xl space-y-4">
      <div className="flex items-center justify-between gap-3">
        <h2 id="memory-heading" className="text-xl font-semibold text-stone-900">
          What Recap remembers
        </h2>
        <RefreshButton onRefresh={refresh} />
      </div>

      <FieldLabel className="block max-w-xs space-y-1">
        <span>Filter by kind</span>
        <Select
          id="memory-filter"
          value={filter}
          onChange={(e) => setFilter(e.target.value as MemoryKind | "")}
        >
          <option value="">All kinds</option>
          {Object.entries(MEMORY_KIND_LABELS).map(([kind, label]) => (
            <option key={kind} value={kind}>
              {label}
            </option>
          ))}
        </Select>
      </FieldLabel>

      {error && <Alert>{error}</Alert>}

      {memories.length === 0 ? (
        <EmptyState>Nothing saved yet.</EmptyState>
      ) : (
        <ul className="divide-y divide-stone-100 rounded-xl border border-stone-200 bg-white shadow-sm">
          {memories.map((memory) => (
            <li key={memory.id} className="flex items-start justify-between gap-3 px-4 py-3">
              <div className="min-w-0 space-y-1">
                <Badge tone="info">{MEMORY_KIND_LABELS[memory.type]}</Badge>
                <p className="text-sm text-stone-800">
                  {memory.content}
                  {memory.page_start !== null && memory.page_end !== null && (
                    <span className="text-stone-500">
                      {" "}
                      (pp. {memory.page_start}-{memory.page_end})
                    </span>
                  )}
                </p>
              </div>
              <Button variant="danger" size="sm" className="shrink-0" onClick={() => void remove(memory.id)}>
                Delete
              </Button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
