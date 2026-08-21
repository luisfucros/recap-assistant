// Recommendations view: explainable suggestions from the reader's own library (FR-5).
//
// Reads only the internal, never-gated signal (GET /recommendations) — a web-
// enriched recommendation is a chat-turn capability (the `recommend` tool's
// include_web branch), surfaced there via the same HITL approval prompt as
// any other gated tool call, not on this page.

import { useCallback, useEffect, useState } from "react";

import { listRecommendations } from "../api/client";
import type { Recommendation } from "../api/types";
import { Alert, EmptyState, RefreshButton } from "./ui";

export function Recommendations(): React.JSX.Element {
  const [items, setItems] = useState<Recommendation[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setError(null);
    try {
      const page = await listRecommendations();
      setItems(page.items);
    } catch {
      setError("Couldn't load recommendations.");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return (
    <section aria-labelledby="recommendations-heading" className="mx-auto max-w-2xl space-y-4">
      <div className="flex items-center justify-between gap-3">
        <h2 id="recommendations-heading" className="text-xl font-semibold text-stone-900">
          Recommended for you
        </h2>
        <RefreshButton onRefresh={refresh} />
      </div>

      {error && <Alert>{error}</Alert>}

      {items.length === 0 && !error ? (
        <EmptyState>
          No recommendations yet — finish or start a book, or tell the assistant a reading
          preference, to get started.
        </EmptyState>
      ) : (
        <ul className="grid gap-3 sm:grid-cols-2">
          {items.map((item, i) => (
            <li key={i} className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm">
              <p className="font-medium text-stone-900">
                {item.title}
                {item.author && <span className="font-normal text-stone-500"> by {item.author}</span>}
              </p>
              <p className="mt-1 text-sm text-stone-600">{item.reason}</p>
              {item.url && (
                <a
                  href={item.url}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-2 inline-block break-all text-sm text-indigo-600 hover:text-indigo-700"
                >
                  {item.url}
                </a>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
