// The "nothing here yet" placeholder every data-fetching panel shows instead
// of a bare blank area (FR-20.3).

export function EmptyState({
  children,
}: {
  children: React.ReactNode;
}): React.JSX.Element {
  return (
    <p className="rounded-lg border border-dashed border-stone-300 px-4 py-6 text-center text-sm text-stone-500">
      {children}
    </p>
  );
}
