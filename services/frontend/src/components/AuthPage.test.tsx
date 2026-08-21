import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AuthProvider } from "../auth/AuthContext";
import { json, mockFetch, unauthorized } from "../test/apiMock";
import { AuthPage } from "./AuthPage";

function renderPage() {
  render(
    <AuthProvider>
      <AuthPage />
    </AuthProvider>,
  );
}

afterEach(() => vi.restoreAllMocks());

describe("AuthPage", () => {
  it("offers a Google sign-in link to the backend flow", async () => {
    mockFetch({ "GET /users/me": () => unauthorized() });
    renderPage();
    await screen.findByRole("heading", { name: /log in/i }); // let the session probe settle

    const link = screen.getByRole("link", { name: /continue with google/i });
    expect(link).toHaveAttribute("href", expect.stringContaining("/auth/google/login"));
  });

  it("switches to the register form and reveals the name field", async () => {
    mockFetch({ "GET /users/me": () => unauthorized() });
    renderPage();
    await screen.findByRole("heading", { name: /log in/i });

    expect(screen.queryByLabelText(/name/i)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /create an account/i }));
    expect(screen.getByLabelText(/name/i)).toBeInTheDocument();
  });

  it("shows a friendly message when credentials are rejected", async () => {
    mockFetch({
      "GET /users/me": () => unauthorized(),
      "POST /auth/login": () => json({ detail: "bad", code: "INVALID_CREDENTIALS" }, 401),
    });
    renderPage();

    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: "ada@example.com" } });
    fireEvent.change(screen.getByLabelText(/password/i), { target: { value: "wrong-one" } });
    fireEvent.click(screen.getByRole("button", { name: /^log in$/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/incorrect email or password/i);
  });

  it("submits credentials to the login endpoint", async () => {
    const routes = {
      "GET /users/me": () => unauthorized(),
      "POST /auth/login": vi.fn(() =>
        json({
          id: "u1",
          email: "ada@example.com",
          display_name: null,
          preferred_language: "en",
          spoiler_safe: true,
        }),
      ),
    };
    mockFetch(routes);
    renderPage();

    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: "ada@example.com" } });
    fireEvent.change(screen.getByLabelText(/password/i), { target: { value: "hunter2!" } });
    fireEvent.click(screen.getByRole("button", { name: /^log in$/i }));

    await waitFor(() => expect(routes["POST /auth/login"]).toHaveBeenCalled());
  });
});
