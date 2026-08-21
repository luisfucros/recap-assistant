// Extends Vitest's expect with jest-dom matchers (e.g. toBeInTheDocument).
import "@testing-library/jest-dom";

// jsdom doesn't implement the Blob URL APIs; components that preview a
// just-attached file (e.g. Chat's attachment thumbnails) need a stand-in.
if (!URL.createObjectURL) {
  URL.createObjectURL = () => "blob:mock-url";
}
if (!URL.revokeObjectURL) {
  URL.revokeObjectURL = () => undefined;
}
