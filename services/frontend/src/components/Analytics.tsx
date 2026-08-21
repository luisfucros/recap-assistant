// Reading analytics dashboard.
//
// Shows the user's pace, streaks, status counts, and a simple pages-over-time
// bar chart over a trailing window (default 30 days). All figures come from the
// backend's cached AnalyticsSummary — this view only renders them.

import { useCallback, useEffect, useState } from "react";

import { getAnalytics } from "../api/client";
import type { AnalyticsSummary } from "../api/types";
import { Alert, Card, EmptyState, RefreshButton, Spinner } from "./ui";

function Stat({ label, value }: { label: string; value: string | number }): React.JSX.Element {
  return (
    <Card className="px-4 py-3">
      <div className="text-2xl font-semibold text-stone-900">{value}</div>
      <div className="text-sm text-stone-500">{label}</div>
    </Card>
  );
}

export function Analytics(): React.JSX.Element {
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setError(null);
    try {
      setSummary(await getAnalytics());
    } catch {
      setError("Couldn't load your analytics.");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  if (error) {
    return (
      <section aria-labelledby="analytics-heading" className="mx-auto max-w-2xl space-y-4">
        <div className="flex items-center justify-between gap-3">
          <h2 id="analytics-heading" className="text-xl font-semibold text-stone-900">
            Reading analytics
          </h2>
          <RefreshButton onRefresh={refresh} />
        </div>
        <Alert>{error}</Alert>
      </section>
    );
  }

  if (!summary) {
    return (
      <section aria-labelledby="analytics-heading" className="mx-auto max-w-2xl space-y-4">
        <div className="flex items-center justify-between gap-3">
          <h2 id="analytics-heading" className="text-xl font-semibold text-stone-900">
            Reading analytics
          </h2>
          <RefreshButton onRefresh={refresh} />
        </div>
        <p role="status" className="flex items-center gap-2 text-sm text-stone-500">
          <Spinner /> Loading…
        </p>
      </section>
    );
  }

  const maxPages = Math.max(1, ...summary.pages_over_time.map((d) => d.pages));

  return (
    <section aria-labelledby="analytics-heading" className="mx-auto max-w-2xl space-y-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h2 id="analytics-heading" className="text-xl font-semibold text-stone-900">
            Reading analytics
          </h2>
          <p className="text-sm text-stone-500">Last {summary.window_days} days</p>
        </div>
        <RefreshButton onRefresh={refresh} />
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Pages read" value={summary.pages_read} />
        <Stat label="Pace (pages/day)" value={summary.pace_pages_per_day} />
        <Stat label="Current streak" value={`${summary.current_streak_days}d`} />
        <Stat label="Longest streak" value={`${summary.longest_streak_days}d`} />
        <Stat label="Reading" value={summary.documents_started} />
        <Stat label="Completed" value={summary.documents_completed} />
        <Stat label="Cancelled" value={summary.documents_cancelled} />
      </div>

      <div>
        <h3 className="mb-2 text-sm font-semibold text-stone-700">Pages over time</h3>
        {summary.pages_over_time.length === 0 ? (
          <EmptyState>No reading recorded yet.</EmptyState>
        ) : (
          <Card className="space-y-2 p-4">
            {summary.pages_over_time.map((day) => (
              <div key={day.day} className="flex items-center gap-3 text-sm">
                <span className="w-24 shrink-0 text-stone-500">{day.day}</span>
                <span className="h-2 flex-1 overflow-hidden rounded-full bg-indigo-100">
                  <span
                    aria-hidden
                    className="block h-full rounded-full bg-indigo-500"
                    style={{ width: `${Math.max((day.pages / maxPages) * 100, 4)}%` }}
                  />
                </span>
                <span className="w-8 shrink-0 text-right font-medium text-stone-700">
                  {day.pages}
                </span>
              </div>
            ))}
          </Card>
        )}
      </div>
    </section>
  );
}
