import { createRoot } from "react-dom/client"

import React, { useEffect, useState } from "react"
import { GptMessage, ModelInfo } from "./worker-types"
import { NumberSlider } from "./number-slider"

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
        description: (
            <p>
                Every neural net starts out untrained and produces utter
                garbage. This is what it looks like.
            </p>
        ),
        params: {
            type: "bigram",
            url: absoluteUrl("/weights/garbage.safetensors"),
        },
    },
    {
        title: "Trained bigram",
        description: (
            <p>
                This is a simple trained bigram model. Each character is based
                solely on the one before it, so it's not very effective. But it
                does only have a few thousand parameters that only took about
                one second to train.
            </p>
        ),
        params: {
            type: "bigram",
            url: absoluteUrl("/weights/bigram.safetensors"),
        },
    },
    {
        title: "Tiny transformer",
        description: (
            <>
                <p>
                    This is a very small transformer model. Each character is
                    based on the 8 that came before it. It only has one
                    self-attention/feed-forward layer, which makes it better
                    than the bigram model, but it's still gibberish--although
                    its output is starting to look like <em>some</em> kind of
                    language.
                </p>
                <p>
                    It has about 17,000 parameters and took around 10 seconds to
                    train.
                </p>
            </>
        ),
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
        title: "Medium transformer",
        description: (
            <>
                <p>
                    This is a larger transformer model that outputs mostly
                    English, although it's still nonsensical. It has six
                    self-attention/feed-forward layers and each character is
                    based on the 128 that preceded it.
                </p>
                <p>
                    It has about 2.7 million parameters and took 12 minutes to
                    train on a RTX 3080.
                </p>
            </>
        ),
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
    const [temperature, setTemperature] = useState(1.0)
    const [initialContext, setInitialContext] = useState("")

    return (
        <div>
            <h1>Language model fun</h1>
            <p>
                This is Atul's attempt to make a Generative Pre-trained
                Transformer (GPT) model based on Andrej Karpathy's{" "}
                <a
                    href="https://www.youtube.com/watch?v=kCc8FmEb1nY"
                    target="_blank"
                >
                    Let's build GPT: from scratch, in code, spelled out
                </a>
                , which is based on the paper{" "}
                <a href="https://arxiv.org/abs/1706.03762" target="_blank">
                    Attention Is All You Need
                </a>
                .
            </p>
            <p>
                All models were trained on{" "}
                <a
                    href="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
                    target="_blank"
                >
                    Tiny Shakespeare
                </a>{" "}
                and use individual characters for tokens.
            </p>
            <details>
                <summary
                    style={{
                        cursor: "pointer",
                    }}
                >
                    Slightly advanced stuff
                </summary>
                <div style={{ paddingTop: 10 }}>
                    <NumberSlider
                        min={0}
                        max={3}
                        value={temperature}
                        step={0.1}
                        onChange={setTemperature}
                        label="Temperature:"
                    />
                </div>
                <div style={{ paddingTop: 10 }}>
                    <label htmlFor="context">Initial context:</label>
                    <textarea
                        style={{
                            display: "block",
                            marginTop: 4,
                            width: "10em",
                            height: "4em",
                            fontFamily: "monospace",
                        }}
                        value={initialContext}
                        onChange={(e) => setInitialContext(e.target.value)}
                    ></textarea>
                </div>
            </details>
            {MODEL_CHOICES.map((choice, i) => (
                <ModelChoice
                    key={i}
                    choice={choice}
                    temperature={temperature}
                    initialContext={initialContext}
                />
            ))}
        </div>
    )
}

function ModelChoice(props: {
    choice: ModelChoice
    temperature: number
    initialContext: string
}) {
    const { choice, temperature, initialContext } = props
    const [genKey, setGenKey] = useState<number | undefined>()

    return (
        <div>
            <h2>{choice.title} model</h2>
            {choice.description}
            {genKey === undefined ? (
                <button onClick={() => setGenKey(0)}>Try it</button>
            ) : (
                <div style={{ minHeight: "30em" }}>
                    <button onClick={() => setGenKey(genKey + 1)}>
                        Try again
                    </button>{" "}
                    <button onClick={() => setGenKey(undefined)}>Close</button>
                    <Generate
                        key={genKey}
                        chars={500}
                        temperature={temperature}
                        initialContext={initialContext}
                        model={choice.params}
                    />
                </div>
            )}
        </div>
    )
}

function Generate(props: {
    chars: number
    temperature: number
    initialContext: string
    model: ModelInfo
}) {
    const { model, chars, temperature, initialContext } = props
    const [output, setOutput] = useState("Loading...")

    useEffect(() => {
        const worker = new Worker(new URL("worker.ts", import.meta.url), {
            type: "module",
        })

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
            chars,
            temperature,
            initialContext,
            model,
        } satisfies GptMessage)

        return () => {
            worker.removeEventListener("message", handler)
            worker.terminate()
        }
    }, [model, chars, temperature, initialContext])

    return <pre>{output}</pre>
}

function absoluteUrl(path: string): string {
    const baseUrl = import.meta.env.BASE_URL
    const rootUrl = new URL(baseUrl, import.meta.url)
    return new URL(path.slice(1), rootUrl).toString()
}
