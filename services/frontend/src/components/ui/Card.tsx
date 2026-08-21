// The one surface every panel is built on: a white card on the app's stone
// background, so sections read as distinct without needing extra dividers.

import { clsx } from "clsx";

export function Card({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>): React.JSX.Element {
  return (
    <div
      className={clsx("rounded-xl border border-stone-200 bg-white shadow-sm", className)}
      {...props}
    />
  );
}
