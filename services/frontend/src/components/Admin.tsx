// Admin console (FR-21): provision accounts and run evaluation datasets.
//
// Mounted only for `user.is_admin`. Evaluation scoring is a background job
// (FR-12.5): Run enqueues a pending row; this panel polls in-flight runs
// the same way Library polls ingestion status. Completed runs show retrieval
// and answer-quality averages plus expandable per-case scores — not raw JSON.

import { useCallback, useEffect, useState } from "react";

import {
  ApiError,
  createUser,
  listEvaluationDatasets,
  listEvaluations,
  runEvaluation,
} from "../api/client";
import {
  isTerminalEvalStatus,
  parseEvaluationCases,
  parseEvaluationSummary,
  type EvaluationCaseResult,
  type EvaluationDataset,
  type EvaluationRun,
  type EvaluationRunStatus,
  type EvaluationSummary,
} from "../api/types";
import { Alert, Badge, Button, Card, Checkbox, EmptyState, FieldLabel, Input, RefreshButton, Select } from "./ui";
import type { BadgeTone } from "./ui";

const POLL_INTERVAL_MS = 3000;

const STATUS_LABELS: Record<EvaluationRunStatus, string> = {
  pending: "Queued",
  running: "Running…",
  completed: "Completed",
  failed: "Failed",
};

const STATUS_TONES: Record<EvaluationRunStatus, BadgeTone> = {
  pending: "warning",
  running: "info",
  completed: "success",
  failed: "danger",
};

function createErrorMessage(error: unknown): string {
  if (error instanceof ApiError && error.code === "USER_ALREADY_EXISTS") {
    return "An account with this email already exists.";
  }
  return "Couldn't create that account.";
}

function formatPct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function Metric({ label, value }: { label: string; value: string }): React.JSX.Element {
  return (
    <div className="rounded-lg bg-stone-50 px-3 py-2">
      <div className="text-xs text-stone-500">{label}</div>
      <div className="text-sm font-semibold tabular-nums text-stone-900">{value}</div>
    </div>
  );
}

function RetrievalMetrics({
  hit_rate,
  recall,
  mrr,
}: {
  hit_rate: number;
  recall: number;
  mrr: number;
}): React.JSX.Element {
  return (
    <div className="grid grid-cols-3 gap-2">
      <Metric label="Hit rate" value={formatPct(hit_rate)} />
      <Metric label="Recall" value={formatPct(recall)} />
      <Metric label="MRR" value={formatPct(mrr)} />
    </div>
  );
}

function RunScores({ summary }: { summary: EvaluationSummary }): React.JSX.Element {
  const judgedNote =
    summary.blocked || summary.interrupted
      ? ` · ${summary.blocked} blocked, ${summary.interrupted} paused`
      : "";
  return (
    <div className="space-y-2">
      <p className="text-xs text-stone-600">
        {summary.cases} {summary.cases === 1 ? "case" : "cases"}
        {judgedNote}
      </p>
      <p className="text-xs font-medium text-stone-500">Retrieval</p>
      <RetrievalMetrics {...summary.retrieval} />
      <p className="text-xs font-medium text-stone-500">Answer quality</p>
      <div className="grid grid-cols-3 gap-2">
        <Metric label="Faithfulness" value={formatPct(summary.answer_quality.faithfulness)} />
        <Metric label="Relevance" value={formatPct(summary.answer_quality.relevance)} />
        <Metric label="Citations OK" value={formatPct(summary.answer_quality.citation_ok_rate)} />
      </div>
    </div>
  );
}

function CaseScores({ item }: { item: EvaluationCaseResult }): React.JSX.Element {
  let outcome = "Judged";
  if (item.blocked) outcome = "Blocked by guardrail";
  else if (item.interrupted) outcome = "Paused for confirmation";
  else if (!item.answer_quality) outcome = "Not judged";

  return (
    <details className="rounded-lg border border-stone-100 bg-stone-50/80 px-3 py-2">
      <summary className="cursor-pointer text-sm font-medium text-stone-800">
        {item.case_id}
        <span className="ml-2 font-normal text-stone-500">{outcome}</span>
      </summary>
      <div className="mt-2 space-y-2">
        <RetrievalMetrics {...item.retrieval} />
        {item.answer_quality && (
          <div className="grid grid-cols-3 gap-2">
            <Metric label="Faithfulness" value={formatPct(item.answer_quality.faithfulness)} />
            <Metric label="Relevance" value={formatPct(item.answer_quality.relevance)} />
            <Metric
              label="Citations"
              value={
                item.answer_quality.citation_ok === undefined
                  ? "—"
                  : item.answer_quality.citation_ok
                    ? "OK"
                    : "Missed"
              }
            />
          </div>
        )}
        {item.answer_quality?.reasoning && (
          <p className="text-xs text-stone-600">{item.answer_quality.reasoning}</p>
        )}
        {item.answer && (
          <p className="text-xs leading-relaxed text-stone-700">{item.answer}</p>
        )}
      </div>
    </details>
  );
}

function EvaluationRunResults({ run }: { run: EvaluationRun }): React.JSX.Element | null {
  if (run.status !== "completed") return null;
  const summary = parseEvaluationSummary(run.summary);
  const cases = parseEvaluationCases(run.results);
  if (!summary) return null;
  return (
    <div className="space-y-3 pt-1">
      <RunScores summary={summary} />
      {cases.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-medium text-stone-500">Cases</p>
          {cases.map((item) => (
            <CaseScores key={item.case_id} item={item} />
          ))}
        </div>
      )}
    </div>
  );
}

export function Admin(): React.JSX.Element {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [asAdmin, setAsAdmin] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [createOk, setCreateOk] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);

  const [datasets, setDatasets] = useState<EvaluationDataset[]>([]);
  const [datasetName, setDatasetName] = useState("");
  const [runs, setRuns] = useState<EvaluationRun[]>([]);
  const [evalError, setEvalError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  const refreshRuns = useCallback(async () => {
    const page = await listEvaluations();
    setRuns(page.items);
  }, []);

  const refreshDatasetsAndRuns = useCallback(async () => {
    const [ds, page] = await Promise.all([listEvaluationDatasets(), listEvaluations()]);
    setDatasets(ds.items);
    setRuns(page.items);
    setDatasetName((current) => current || ds.items[0]?.name || "");
  }, []);

  useEffect(() => {
    refreshDatasetsAndRuns().catch(() => setEvalError("Couldn't load evaluations."));
  }, [refreshDatasetsAndRuns]);

  const hasInFlight = runs.some((run) => !isTerminalEvalStatus(run.status));
  useEffect(() => {
    if (!hasInFlight) return;
    const timer = setInterval(() => void refreshRuns().catch(() => {}), POLL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [hasInFlight, refreshRuns]);

  const onCreate = async (event: React.FormEvent) => {
    event.preventDefault();
    setCreateError(null);
    setCreateOk(null);
    setCreating(true);
    try {
      const user = await createUser({
        email,
        password,
        display_name: displayName || null,
        is_admin: asAdmin,
      });
      setCreateOk(`Created ${user.email}.`);
      setEmail("");
      setPassword("");
      setDisplayName("");
      setAsAdmin(false);
    } catch (err) {
      setCreateError(createErrorMessage(err));
    } finally {
      setCreating(false);
    }
  };

  const onRun = async () => {
    if (!datasetName) return;
    setEvalError(null);
    setRunning(true);
    try {
      await runEvaluation(datasetName);
      await refreshRuns();
    } catch {
      setEvalError("Couldn't start that evaluation.");
    } finally {
      setRunning(false);
    }
  };

  return (
    <section aria-labelledby="admin-heading" className="mx-auto max-w-3xl space-y-6">
      <div className="flex items-center justify-between gap-3">
        <h2 id="admin-heading" className="text-xl font-semibold text-stone-900">
          Admin
        </h2>
        <RefreshButton onRefresh={refreshDatasetsAndRuns} />
      </div>

      <Card className="space-y-4 p-4">
        <h3 className="text-sm font-semibold text-stone-800">Create user</h3>
        {createError && <Alert>{createError}</Alert>}
        {createOk && (
          <p role="status" className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-800">
            {createOk}
          </p>
        )}
        <form onSubmit={(e) => void onCreate(e)} className="space-y-3">
          <FieldLabel className="block space-y-1">
            <span>Email</span>
            <Input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="off"
            />
          </FieldLabel>
          <FieldLabel className="block space-y-1">
            <span>Password</span>
            <Input
              type="password"
              required
              minLength={8}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="new-password"
            />
          </FieldLabel>
          <FieldLabel className="block space-y-1">
            <span>Display name</span>
            <Input value={displayName} onChange={(e) => setDisplayName(e.target.value)} />
          </FieldLabel>
          <label className="flex items-center gap-2 text-sm text-stone-700">
            <Checkbox
              checked={asAdmin}
              onChange={(e) => setAsAdmin(e.target.checked)}
            />
            Admin
          </label>
          <Button type="submit" disabled={creating}>
            Create account
          </Button>
        </form>
      </Card>

      <Card className="space-y-4 p-4">
        <h3 className="text-sm font-semibold text-stone-800">Evaluations</h3>
        {evalError && <Alert>{evalError}</Alert>}
        <div className="flex flex-wrap items-end gap-3">
          <FieldLabel className="block min-w-48 flex-1 space-y-1">
            <span>Dataset</span>
            <Select
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              disabled={datasets.length === 0}
            >
              {datasets.map((ds) => (
                <option key={ds.name} value={ds.name}>
                  {ds.name} ({ds.version})
                </option>
              ))}
            </Select>
          </FieldLabel>
          <Button type="button" onClick={() => void onRun()} disabled={running || !datasetName}>
            Run
          </Button>
        </div>

        {runs.length === 0 ? (
          <EmptyState>No evaluation runs yet.</EmptyState>
        ) : (
          <ul className="divide-y divide-stone-100">
            {runs.map((run) => (
              <li key={run.id} className="space-y-1 py-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm font-medium text-stone-900">
                    {run.dataset_name}@{run.dataset_version}
                  </span>
                  <Badge tone={STATUS_TONES[run.status]}>{STATUS_LABELS[run.status]}</Badge>
                </div>
                <p className="text-xs text-stone-500">
                  {run.prompt_version} · {run.llm_provider}:{run.llm_model} · {run.embedding_model}
                </p>
                {run.error && <p className="text-xs text-red-700">{run.error}</p>}
                <EvaluationRunResults run={run} />
              </li>
            ))}
          </ul>
        )}
      </Card>
    </section>
  );
}
