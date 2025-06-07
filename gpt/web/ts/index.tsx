import { createRoot } from "react-dom/client";

import React, { useEffect, useState } from "react";
import { GptMessage } from "./worker-types";

const root = createRoot(document.getElementById("app")!);

root.render(<App />);

const worker = new Worker(new URL('worker.ts', import.meta.url), {
  type: "module"
});

function App() {
  const [output, setOutput] = useState("LOADING");

  useEffect(() => {
    const handler = (e: MessageEvent<GptMessage>) => {
      if (e.data.type === "output") {
        setOutput(e.data.text);
      }
    };
    worker.addEventListener("message", handler);

    worker.postMessage({
      type: "generate"
    } satisfies GptMessage);

    return () => {
      worker.removeEventListener("message", handler);
    }
  }, []);

  return <pre>{output}</pre>;
}
