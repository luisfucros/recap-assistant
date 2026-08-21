// Styled wrappers over the native form controls. Every prop (id, value,
// onChange, required, disabled, aria-*...) forwards untouched, so existing
// <label>/htmlFor associations and test queries (getByLabelText, etc.) are
// unaffected — only appearance is added here.

import { clsx } from "clsx";
import { forwardRef } from "react";

const CONTROL =
  "w-full rounded-lg border border-stone-300 bg-white px-3 py-2 text-sm text-stone-900 " +
  "placeholder:text-stone-400 focus-visible:border-indigo-500 disabled:cursor-not-allowed " +
  "disabled:bg-stone-100 disabled:text-stone-400";

export const Input = forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  function Input({ className, ...props }, ref) {
    return <input ref={ref} className={clsx(CONTROL, className)} {...props} />;
  },
);

export const Textarea = forwardRef<
  HTMLTextAreaElement,
  React.TextareaHTMLAttributes<HTMLTextAreaElement>
>(function Textarea({ className, ...props }, ref) {
  return <textarea ref={ref} className={clsx(CONTROL, "resize-none", className)} {...props} />;
});

export const Select = forwardRef<
  HTMLSelectElement,
  React.SelectHTMLAttributes<HTMLSelectElement>
>(function Select({ className, ...props }, ref) {
  return <select ref={ref} className={clsx(CONTROL, className)} {...props} />;
});

export function Checkbox({
  className,
  ...props
}: React.InputHTMLAttributes<HTMLInputElement>): React.JSX.Element {
  return (
    <input
      type="checkbox"
      className={clsx(
        "size-4 rounded border-stone-300 text-indigo-600 focus-visible:outline-indigo-500",
        className,
      )}
      {...props}
    />
  );
}

/** A field's text label — a plain span sized/weighted to pair with the controls above. */
export function FieldLabel({
  className,
  ...props
}: React.LabelHTMLAttributes<HTMLLabelElement>): React.JSX.Element {
  return <label className={clsx("text-sm font-medium text-stone-700", className)} {...props} />;
}
