import { createRoot } from "react-dom/client";

import React, { useEffect, useMemo, useState } from "react";
import { GptMessage } from "./worker-types";

const root = createRoot(document.getElementById("app")!);

root.render(<App />);

function App() {
  const [output, setOutput] = useState("LOADING");

  const worker = useMemo(() => {
    return new Worker(new URL("worker.ts", import.meta.url), {
      type: "module",
    });
  }, []);

  useEffect(() => {
    const handler = (e: MessageEvent<GptMessage>) => {
      if (e.data.type === "output") {
        setOutput(e.data.text);
      } else if (e.data.type === "done") {
        worker.terminate();
      }
    };
    worker.addEventListener("message", handler);

    worker.postMessage({
      type: "generate",
      chars: 500,
      model: {
        type: "transformer",
        url: getUrl("/weights/default-tiny-shakespeare.safetensors").toString(),
        n_embed: 32,
        block_size: 8,
        num_layers: 1,
        num_heads: 4,
      },
    } satisfies GptMessage);

    return () => {
      worker.removeEventListener("message", handler);
    };
  }, []);

  return <pre>{output}</pre>;
}

function getUrl(path: string): URL {
  const baseUrl = import.meta.env.BASE_URL;
  const rootUrl = new URL(baseUrl, import.meta.url);
  return new URL(path.slice(1), rootUrl);
}
