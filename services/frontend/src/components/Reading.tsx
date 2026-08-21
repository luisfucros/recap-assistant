// Reading progress: track your position in each document.
//
// Lists the user's documents merged with their reading state, and for each lets
// them set the current page, cancel/reopen the book, and choose a per-book
// spoiler-safe setting (default / on / off). Reading state drives what the
// assistant retrieves, so this is where the user tells it where they are.

import { useCallback, useEffect, useState } from "react";

import { listDocuments, listProgress, updateProgress } from "../api/client";
import {
  type DocumentSummary,
  type ProgressUpdate,
  READING_STATUS_LABELS,
  type ReadingProgress,
  type ReadingStatus,
} from "../api/types";
import { Alert, Badge, type BadgeTone, Button, EmptyState, Input, RefreshButton, Select } from "./ui";

const STATUS_TONES: Record<ReadingStatus, BadgeTone> = {
  not_started: "neutral",
  reading: "info",
  completed: "success",
  cancelled: "neutral",
};

/** The per-book spoiler-safe override rendered as a 3-way select. */
type SpoilerChoice = "default" | "on" | "off";

function spoilerChoice(value: boolean | null | undefined): SpoilerChoice {
  if (value === true) return "on";
  if (value === false) return "off";
  return "default";
}

function spoilerValue(choice: SpoilerChoice): boolean | null {
  if (choice === "on") return true;
  if (choice === "off") return false;
  return null;
}

export function Reading(): React.JSX.Element {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [progress, setProgress] = useState<Record<string, ReadingProgress>>({});
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const [page, list] = await Promise.all([listDocuments(), listProgress()]);
    const byId: Record<string, ReadingProgress> = {};
    for (const group of [list.reading, list.completed, list.cancelled]) {
      for (const row of group) byId[row.document_id] = row;
    }
    setDocuments(page.items);
    setProgress(byId);
  }, []);

  useEffect(() => {
    refresh().catch(() => setError("Couldn't load your reading progress."));
  }, [refresh]);

  const apply = useCallback(
    async (documentId: string, patch: ProgressUpdate) => {
      setError(null);
      try {
        const updated = await updateProgress(documentId, patch);
        setProgress((prev) => ({ ...prev, [documentId]: updated }));
      } catch {
        setError("Couldn't update your progress.");
      }
    },
    [],
  );

  if (documents.length === 0) {
    return (
      <section aria-labelledby="reading-heading" className="mx-auto max-w-2xl space-y-4">
        <div className="flex items-center justify-between gap-3">
          <h2 id="reading-heading" className="text-xl font-semibold text-stone-900">
            Reading progress
          </h2>
          <RefreshButton onRefresh={refresh} />
        </div>
        <EmptyState>Upload a document to start tracking your progress.</EmptyState>
      </section>
    );
  }

  return (
    <section aria-labelledby="reading-heading" className="mx-auto max-w-2xl space-y-4">
      <div className="flex items-center justify-between gap-3">
        <h2 id="reading-heading" className="text-xl font-semibold text-stone-900">
          Reading progress
        </h2>
        <RefreshButton onRefresh={refresh} />
      </div>
      {error && <Alert>{error}</Alert>}
      <ul className="space-y-3">
        {documents.map((doc) => (
          <ReadingRow
            key={doc.id}
            doc={doc}
            progress={progress[doc.id]}
            onApply={(patch) => apply(doc.id, patch)}
          />
        ))}
      </ul>
    </section>
  );
}

function ReadingRow({
  doc,
  progress,
  onApply,
}: {
  doc: DocumentSummary;
  progress: ReadingProgress | undefined;
  onApply: (patch: ProgressUpdate) => void | Promise<void>;
}): React.JSX.Element {
  const status: ReadingStatus = progress?.status ?? "not_started";
  const [page, setPage] = useState<string>(String(progress?.current_page ?? 0));

  // Keep the input in sync when progress arrives/refreshes.
  useEffect(() => {
    setPage(String(progress?.current_page ?? 0));
  }, [progress?.current_page]);

  const title = doc.title || doc.filename;
  const label = `page-${doc.id}`;

  return (
    <li className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <span className="font-medium text-stone-900">{title}</span>
        <Badge tone={STATUS_TONES[status]} data-testid={`status-${doc.id}`}>
          {READING_STATUS_LABELS[status]}
        </Badge>
      </div>

      <div className="mt-3 flex flex-wrap items-end gap-3">
        <div className="space-y-1">
          <label htmlFor={label} className="block text-xs font-medium text-stone-600">
            Current page
          </label>
          <div className="flex items-center gap-1.5">
            <Input
              id={label}
              type="number"
              min={0}
              max={doc.page_count ?? undefined}
              value={page}
              onChange={(e) => setPage(e.target.value)}
              className="w-20"
            />
            {doc.page_count != null && (
              <span className="text-sm text-stone-500">of {doc.page_count}</span>
            )}
          </div>
        </div>
        <Button size="sm" onClick={() => void onApply({ current_page: Number(page) })}>
          Update
        </Button>

        <div className="space-y-1">
          <label className="block text-xs font-medium text-stone-600">
            Spoiler-safe
            <Select
              aria-label={`Spoiler-safe for ${title}`}
              value={spoilerChoice(progress?.spoiler_safe)}
              onChange={(e) =>
                void onApply({ spoiler_safe: spoilerValue(e.target.value as SpoilerChoice) })
              }
              className="mt-1 w-40 py-1.5 text-sm"
            >
              <option value="default">Use my default</option>
              <option value="on">On</option>
              <option value="off">Off</option>
            </Select>
          </label>
        </div>

        {status === "cancelled" ? (
          <Button size="sm" variant="secondary" onClick={() => void onApply({ status: "reading" })}>
            Reopen
          </Button>
        ) : (
          <Button size="sm" variant="ghost" onClick={() => void onApply({ status: "cancelled" })}>
            Cancel
          </Button>
        )}
      </div>
    </li>
  );
}
