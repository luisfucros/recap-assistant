// The signed-in landing view.
//
// A persistent shell (brand, section nav, account controls) around one
// focused panel at a time (FR-20.2) — library (upload/ingest/delete),
// reading progress, chat with the assistant, memory, recommendations, and
// analytics. Every panel stays mounted (just visually hidden off-section) so
// switching tabs never interrupts an in-flight poll or resets local state.

import { useState } from "react";
import { clsx } from "clsx";

import { LANGUAGES, LANGUAGE_LABELS, type Language } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { Analytics } from "./Analytics";
import { Admin } from "./Admin";
import { Chat } from "./Chat";
import { Library } from "./Library";
import { Memory } from "./Memory";
import { Reading } from "./Reading";
import { Recommendations } from "./Recommendations";
import { Checkbox, FieldLabel, Select } from "./ui";

type SectionKey =
  | "chat"
  | "library"
  | "reading"
  | "memory"
  | "recommendations"
  | "analytics"
  | "admin";

const READER_SECTIONS: { key: SectionKey; label: string }[] = [
  { key: "chat", label: "Chat" },
  { key: "library", label: "Library" },
  { key: "reading", label: "Reading" },
  { key: "memory", label: "Memory" },
  { key: "recommendations", label: "Recommendations" },
  { key: "analytics", label: "Analytics" },
];

export function Dashboard(): React.JSX.Element {
  const { user, logout, updateProfile } = useAuth();
  const [saving, setSaving] = useState(false);
  const [section, setSection] = useState<SectionKey>("chat");

  if (!user) return <></>; // guarded by App; keeps the type narrow

  const sections = user.is_admin
    ? [...READER_SECTIONS, { key: "admin" as const, label: "Admin" }]
    : READER_SECTIONS;

  const onLanguageChange = async (language: Language) => {
    setSaving(true);
    try {
      await updateProfile({ preferred_language: language });
    } finally {
      setSaving(false);
    }
  };

  const onSpoilerSafeChange = async (spoilerSafe: boolean) => {
    setSaving(true);
    try {
      await updateProfile({ spoiler_safe: spoilerSafe });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col bg-stone-50 md:flex-row">
      <aside className="flex flex-col border-b border-stone-200 bg-white md:w-56 md:border-b-0 md:border-r">
        <div className="px-5 py-4">
          <h1 className="font-serif text-xl font-semibold text-stone-900">Recap</h1>
          <p className="text-xs text-stone-500">Welcome back, {user.display_name || user.email}.</p>
        </div>

        <nav
          aria-label="Sections"
          className="flex gap-1 overflow-x-auto px-3 pb-3 md:flex-col md:overflow-visible"
        >
          {sections.map((s) => (
            <button
              key={s.key}
              type="button"
              aria-current={section === s.key ? "page" : undefined}
              onClick={() => setSection(s.key)}
              className={clsx(
                "shrink-0 rounded-lg px-3 py-2 text-left text-sm font-medium transition-colors",
                section === s.key
                  ? "bg-indigo-50 text-indigo-700"
                  : "text-stone-600 hover:bg-stone-100",
              )}
            >
              {s.label}
            </button>
          ))}
        </nav>

        <div className="mt-auto space-y-3 border-t border-stone-200 px-5 py-4">
          <FieldLabel className="block space-y-1 text-xs">
            <span>Language</span>
            <Select
              value={user.preferred_language}
              disabled={saving}
              onChange={(e) => void onLanguageChange(e.target.value as Language)}
              className="py-1.5 text-sm"
            >
              {LANGUAGES.map((lang) => (
                <option key={lang} value={lang}>
                  {LANGUAGE_LABELS[lang]}
                </option>
              ))}
            </Select>
          </FieldLabel>

          <label className="flex items-center gap-2 text-xs text-stone-600">
            <Checkbox
              checked={user.spoiler_safe}
              disabled={saving}
              onChange={(e) => void onSpoilerSafeChange(e.target.checked)}
            />
            Spoiler-safe (hide content past where I&rsquo;ve read)
          </label>

          <button
            type="button"
            onClick={() => void logout()}
            className="text-xs font-medium text-stone-500 hover:text-stone-700"
          >
            Log out
          </button>
        </div>
      </aside>

      <main className="min-w-0 flex-1 p-4 md:p-8">
        <div className={section === "chat" ? "" : "hidden"}>
          <Chat />
        </div>
        <div className={section === "library" ? "" : "hidden"}>
          <Library />
        </div>
        <div className={section === "reading" ? "" : "hidden"}>
          <Reading />
        </div>
        <div className={section === "memory" ? "" : "hidden"}>
          <Memory />
        </div>
        <div className={section === "recommendations" ? "" : "hidden"}>
          <Recommendations />
        </div>
        <div className={section === "analytics" ? "" : "hidden"}>
          <Analytics />
        </div>
        {user.is_admin && (
          <div className={section === "admin" ? "" : "hidden"}>
            <Admin />
          </div>
        )}
      </main>
    </div>
  );
}
