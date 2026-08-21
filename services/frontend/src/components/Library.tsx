// The document library: upload PDFs, watch them ingest, delete them.
//
// Ingestion is asynchronous on the backend, so a freshly uploaded document is
// `pending`; this view polls the list while any document is still `pending`/
// `processing` and stops once everything is terminal (`indexed`/`failed`),
// giving live status without a websocket.

import { clsx } from "clsx";
import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, deleteDocument, listDocuments, retryDocument, uploadDocument } from "../api/client";
import { type DocumentSummary, isTerminalStatus } from "../api/types";
import { Alert, Badge, type BadgeTone, Button, EmptyState, RefreshButton, Spinner } from "./ui";

const POLL_INTERVAL_MS = 3000;

const STATUS_LABELS: Record<DocumentSummary["status"], string> = {
  pending: "Queued",
  processing: "Processing…",
  indexed: "Ready",
  failed: "Failed",
};

const STATUS_TONES: Record<DocumentSummary["status"], BadgeTone> = {
  pending: "warning",
  processing: "info",
  indexed: "success",
  failed: "danger",
};

/** Map an upload failure to a friendly message. */
function uploadErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    switch (error.code) {
      case "DUPLICATE_DOCUMENT":
        return "You've already uploaded this document.";
      case "UNSUPPORTED_MEDIA_TYPE":
        return "Only PDF files are supported.";
      case "PAYLOAD_TOO_LARGE":
        return "That file is too large to upload.";
      default:
        return error.message;
    }
  }
  return "Something went wrong uploading that file.";
}

export function Library(): React.JSX.Element {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    const page = await listDocuments();
    setDocuments(page.items);
  }, []);

  useEffect(() => {
    refresh().catch(() => setError("Couldn't load your library."));
  }, [refresh]);

  // Poll only while something is still ingesting; stop once all are terminal.
  // Transient poll errors are ignored — the next tick retries.
  const hasInFlight = documents.some((doc) => !isTerminalStatus(doc.status));
  useEffect(() => {
    if (!hasInFlight) return;
    const timer = setInterval(() => void refresh().catch(() => {}), POLL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [hasInFlight, refresh]);

  const upload = useCallback(
    async (file: File) => {
      setError(null);
      setUploading(true);
      try {
        await uploadDocument(file);
        await refresh();
      } catch (err) {
        setError(uploadErrorMessage(err));
      } finally {
        setUploading(false);
      }
    },
    [refresh],
  );

  const onFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) void upload(file);
    event.target.value = ""; // allow re-selecting the same file
  };

  const onDrop = (event: React.DragEvent) => {
    event.preventDefault();
    setDragOver(false);
    const file = event.dataTransfer.files?.[0];
    if (file) void upload(file);
  };

  const remove = async (id: string) => {
    setError(null);
    try {
      await deleteDocument(id);
      await refresh();
    } catch {
      setError("Couldn't delete that document.");
    }
  };

  const retry = async (id: string) => {
    setError(null);
    try {
      await retryDocument(id);
      await refresh();
    } catch {
      setError("Couldn't retry that document.");
    }
  };

  return (
    <section aria-labelledby="library-heading" className="mx-auto max-w-2xl space-y-4">
      <div className="flex items-center justify-between gap-3">
        <h2 id="library-heading" className="text-xl font-semibold text-stone-900">
          Your library
        </h2>
        <RefreshButton onRefresh={refresh} />
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        data-drag-over={dragOver}
        className={clsx(
          "rounded-xl border-2 border-dashed px-4 py-6 text-center transition-colors",
          dragOver ? "border-indigo-400 bg-indigo-50" : "border-stone-300 bg-white",
        )}
      >
        <label className="inline-flex cursor-pointer flex-col items-center gap-2">
          <span className="text-sm font-medium text-indigo-600">Upload a PDF</span>
          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf"
            disabled={uploading}
            onChange={onFileChange}
            className="text-sm text-stone-500"
          />
        </label>
        {uploading && (
          <p role="status" className="mt-2 flex items-center justify-center gap-2 text-sm text-stone-500">
            <Spinner /> Uploading…
          </p>
        )}
        <p className="mt-1 text-xs text-stone-400">or drop a PDF here</p>
      </div>

      {error && <Alert>{error}</Alert>}

      {documents.length === 0 ? (
        <EmptyState>No documents yet. Upload a PDF to get started.</EmptyState>
      ) : (
        <ul className="divide-y divide-stone-100 rounded-xl border border-stone-200 bg-white shadow-sm">
          {documents.map((doc) => (
            <li key={doc.id} className="flex items-center justify-between gap-3 px-4 py-3">
              <div className="min-w-0">
                <p className="truncate text-sm font-medium text-stone-900">
                  {doc.title || doc.filename}
                </p>
                {doc.title && doc.title !== doc.filename && (
                  <p className="truncate text-xs text-stone-400">{doc.filename}</p>
                )}
                {doc.status === "failed" && doc.failure_reason && (
                  <p className="truncate text-xs text-red-600">{doc.failure_reason}</p>
                )}
              </div>
              <div className="flex shrink-0 items-center gap-2">
                <Badge tone={STATUS_TONES[doc.status]} data-testid={`status-${doc.id}`}>
                  {STATUS_LABELS[doc.status]}
                </Badge>
                {doc.status === "failed" && (
                  <Button variant="secondary" size="sm" onClick={() => void retry(doc.id)}>
                    Retry
                  </Button>
                )}
                <Button variant="danger" size="sm" onClick={() => void remove(doc.id)}>
                  Delete
                </Button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
