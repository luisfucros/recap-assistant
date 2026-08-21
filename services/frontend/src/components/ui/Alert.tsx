// A role="alert" banner. Every panel's error message (and a couple of
// non-error HITL prompts that also need assertive announcement) renders
// through here so they look and sound consistent.

import { clsx } from "clsx";

export type AlertTone = "danger" | "warning" | "info";

const TONES: Record<AlertTone, string> = {
  danger: "border-red-200 bg-red-50 text-red-800",
  warning: "border-amber-200 bg-amber-50 text-amber-800",
  info: "border-indigo-200 bg-indigo-50 text-indigo-800",
};

export function Alert({
  tone = "danger",
  className,
  ...props
}: React.HTMLAttributes<HTMLParagraphElement> & { tone?: AlertTone }): React.JSX.Element {
  return (
    <p
      role="alert"
      className={clsx("rounded-lg border px-3 py-2 text-sm", TONES[tone], className)}
      {...props}
    />
  );
}
