import init, { WasmLanguageModel } from "../pkg/web.js";
import { GptMessage } from "./worker-types.js";

async function run() {
  await init();

  const safetensors = await fetch(
    getUrl("/weights/default-tiny-shakespeare.safetensors")
  );
  const safetensorsU8 = new Uint8Array(await safetensors.arrayBuffer());
  const model = WasmLanguageModel.transformer(32, 8, 1, 4, 0.0, safetensorsU8);
  const generator = model.create_generator(BigInt(Date.now()), 1.0, "\n");

  const chars: string[] = []
  for (let i = 0; i < 500; i++) {
    chars.push(generator.next_token());
  }

  return chars.join("");
}

onmessage = async (e: MessageEvent<GptMessage>) => {
  if (e.data.type === "generate") {
    const text = await run();
    postGptMessage({
      type: "output",
      text,
    });
  }
};

function postGptMessage(message: GptMessage) {
  postMessage(message);
}

function getUrl(path: string): URL {
  const baseUrl = import.meta.env.BASE_URL;
  const rootUrl = new URL(baseUrl, import.meta.url);
  return new URL(path.slice(1), rootUrl);
}
