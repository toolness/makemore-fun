import init, { generate } from "../pkg/web.js";
import { GptMessage } from "./worker-types.js";

async function run() {
  await init();

  const stuff = await fetch(getUrl("/weights/default-tiny-shakespeare.safetensors"));

  const uint8Array = new Uint8Array(await stuff.arrayBuffer());

  const text = generate(uint8Array, 500, 1.0, BigInt(Date.now()));

  return text;
}

onmessage = async (e: MessageEvent<GptMessage>) => {
  if (e.data.type === "generate") {
    const text = await run();
    postGptMessage({
      type: "output",
      text
    })
  }
}

function postGptMessage(message: GptMessage) {
  postMessage(message);
}

function getUrl(path: string): URL {
  const baseUrl = import.meta.env.BASE_URL
  const rootUrl = new URL(baseUrl, import.meta.url);
  return new URL(path.slice(1), rootUrl);
}
