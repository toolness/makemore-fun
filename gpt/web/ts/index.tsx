import { createRoot } from "react-dom/client"

import React, { useEffect, useMemo, useState } from "react"
import { GptMessage, ModelInfo } from "./worker-types"

const root = createRoot(document.getElementById("app")!)

root.render(<App />)

interface ModelChoice {
    title: string
    description: React.ReactNode
    params: ModelInfo
}

const MODEL_CHOICES: ModelChoice[] = [
    {
        title: "Untrained bigram",
        description: <p>Cool</p>,
        params: {
            type: "bigram",
            url: absoluteUrl("/weights/garbage.safetensors"),
        },
    },
    {
        title: "Trained bigram",
        description: <p>Cool</p>,
        params: {
            type: "bigram",
            url: absoluteUrl("/weights/bigram.safetensors"),
        },
    },
    {
        title: "Tiny transformer",
        description: <p>Cool</p>,
        params: {
            type: "transformer",
            url: absoluteUrl("/weights/default-tiny-shakespeare.safetensors"),
            n_embed: 32,
            block_size: 8,
            num_layers: 1,
            num_heads: 4,
        },
    },
    {
        title: "Small transformer",
        description: <p>Cool</p>,
        params: {
            type: "transformer",
            url: absoluteUrl("/weights/massive.safetensors"),
            n_embed: 192,
            block_size: 128,
            num_layers: 6,
            num_heads: 6,
        },
    },
]

function App() {
    const [output, setOutput] = useState("LOADING")

    const worker = useMemo(() => {
        return new Worker(new URL("worker.ts", import.meta.url), {
            type: "module",
        })
    }, [])

    useEffect(() => {
        const handler = (e: MessageEvent<GptMessage>) => {
            if (e.data.type === "output") {
                setOutput(e.data.text)
            } else if (e.data.type === "done") {
                worker.terminate()
            }
        }
        worker.addEventListener("message", handler)

        worker.postMessage({
            type: "generate",
            chars: 500,
            temperature: 1.0,
            initialContext: "\n",
            model: MODEL_CHOICES[2].params,
        } satisfies GptMessage)

        return () => {
            worker.removeEventListener("message", handler)
        }
    }, [])

    return <pre>{output}</pre>
}

function absoluteUrl(path: string): string {
    const baseUrl = import.meta.env.BASE_URL
    const rootUrl = new URL(baseUrl, import.meta.url)
    return new URL(path.slice(1), rootUrl).toString()
}
