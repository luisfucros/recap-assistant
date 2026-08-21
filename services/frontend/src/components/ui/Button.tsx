// A single styled <button>, forwarding every native prop untouched so
// existing `type`/`disabled`/`onClick`/label-adjacent usage keeps working —
// only appearance is centralized here.

import { clsx } from "clsx";
import { forwardRef } from "react";

export type ButtonVariant = "primary" | "secondary" | "danger" | "ghost";
export type ButtonSize = "sm" | "md";

const BASE =
  "inline-flex items-center justify-center gap-1.5 rounded-lg font-medium " +
  "transition-colors disabled:cursor-not-allowed disabled:opacity-50";

const VARIANTS: Record<ButtonVariant, string> = {
  primary: "bg-indigo-600 text-white hover:bg-indigo-700 disabled:hover:bg-indigo-600",
  secondary:
    "bg-white text-stone-700 border border-stone-300 hover:bg-stone-50 disabled:hover:bg-white",
  danger: "bg-white text-red-600 border border-red-200 hover:bg-red-50 disabled:hover:bg-white",
  ghost: "text-stone-600 hover:bg-stone-100 disabled:hover:bg-transparent",
};

const SIZES: Record<ButtonSize, string> = {
  sm: "px-2.5 py-1.5 text-sm",
  md: "px-4 py-2 text-sm",
};

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "secondary", size = "md", className, ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      className={clsx(BASE, VARIANTS[variant], SIZES[size], className)}
      {...props}
    />
  );
});
