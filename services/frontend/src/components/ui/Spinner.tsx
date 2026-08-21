// A small inline loading indicator (role="status" lives on the caller's own
// element, since each usage already carries its own accessible text).

import { clsx } from "clsx";

export function Spinner({ className }: { className?: string }): React.JSX.Element {
  return (
    <span
      aria-hidden
      className={clsx(
        "inline-block size-4 animate-spin rounded-full border-2 border-stone-300 border-t-indigo-600",
        className,
      )}
    />
  );
}
