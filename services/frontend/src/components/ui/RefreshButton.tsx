// A small button that re-runs a panel's own data-fetch on demand.
//
// Every dashboard panel fetches its data once on mount, and panels stay
// mounted (never remounted) when the user switches sections — so nothing else
// in the panel picks up a change made elsewhere (e.g. a document uploaded in
// Library) until this is clicked or the page is reloaded.

import { useState } from "react";

import { Button, type ButtonProps } from "./Button";
import { Spinner } from "./Spinner";

export interface RefreshButtonProps extends Omit<ButtonProps, "onClick" | "children"> {
  /** Re-fetch this panel's data; errors are the caller's responsibility to surface. */
  onRefresh: () => Promise<void>;
  label?: string;
}

export function RefreshButton({
  onRefresh,
  label = "Refresh",
  variant = "secondary",
  size = "sm",
  ...props
}: RefreshButtonProps): React.JSX.Element {
  const [refreshing, setRefreshing] = useState(false);

  const handleClick = async () => {
    setRefreshing(true);
    try {
      await onRefresh();
    } finally {
      setRefreshing(false);
    }
  };

  return (
    <Button
      variant={variant}
      size={size}
      onClick={() => void handleClick()}
      disabled={refreshing}
      {...props}
    >
      {refreshing && <Spinner />}
      {refreshing ? "Refreshing…" : label}
    </Button>
  );
}
