/// embed_ai.rs : optional embedded small local model & CPU inference via Candle
/// ponytail: single global Mutex<model> + greedy decode, O(n) KV cache; upgrade to sampler + flash-attn if needed

pub fn is_embed_model(model: &str) -> bool {
    let m = model.trim().to_lowercase();
    m == "embed"
        || m == "local"
        || m.contains("smollm")
        || m.contains("smol")
        || m.contains("360m")
        || m == "huggingfacetb/smollm2-360m-instruct"
        || m == "huggingfacetb/smollm2-360m"
}

#[cfg(not(feature = "embed"))]
pub struct EmbedProvider;

#[cfg(not(feature = "embed"))]
impl EmbedProvider {
    pub fn new() -> Result<Self, String> {
        Err("bs was built without --features embed".to_string())
    }
    pub async fn complete(&self, _messages: &[crate::provider::Message]) -> Result<crate::provider::Completion, crate::provider::Error> {
        Err(crate::provider::Error::Embed("embed feature not enabled".to_string()))
    }
}

#[cfg(feature = "embed")]
mod inner {
    use std::sync::{Arc, Mutex};

    use candle_core::{DType, Device, Tensor, D};
    use candle_nn::VarBuilder;
    use candle_transformers::models::llama::{Cache, Llama, LlamaConfig, LlamaEosToks};
    use tokenizers::Tokenizer;

    use crate::provider::{Completion, Error, Message};

    const MODEL_ID: &str = "HuggingFaceTB/SmolLM2-360M-Instruct";
    const MAX_TOKENS: usize = 512;

    pub struct EmbedProvider {
        // ponytail: global lock serializes inference; SmolLM2-360M is CPU-bound, no parallelism benefit
        inner: Arc<Mutex<Option<Loaded>>>,
        device: Device,
    }

    #[allow(dead_code)]
    struct Loaded {
        model: Llama,
        tokenizer: Tokenizer,
        cache: Cache,
        config: candle_transformers::models::llama::Config,
        eos_token: u32,
        device: Device,
    }

    impl EmbedProvider {
        pub fn new() -> Result<Self, String> {
            Ok(Self {
                inner: Arc::new(Mutex::new(None)),
                device: Device::Cpu,
            })
        }

        pub async fn complete(&self, messages: &[Message]) -> Result<Completion, Error> {
            let prompt = format_prompt(messages);
            let inner = self.inner.clone();
            let device = self.device.clone();
            // spawn_blocking for CPU inference
            let content = tokio::task::spawn_blocking(move || {
                let mut guard = inner.lock().unwrap();
                if guard.is_none() {
                    let loaded = load_model(&device).map_err(|e| Error::Embed(e))?;
                    *guard = Some(loaded);
                }
                let loaded = guard.as_mut().unwrap();
                generate(loaded, &prompt).map_err(|e| Error::Embed(e))
            })
            .await
            .map_err(|e| Error::Embed(format!("join error: {e}")))??
            ;
            Ok(Completion {
                content: Some(content),
                tool_calls: Vec::new(),
            })
        }
    }

    fn format_prompt(messages: &[Message]) -> String {
        // SmolLM2 chat template: <|im_start|>role\ncontent<|im_end|>\n ; brave simple
        let mut out = String::new();
        for m in messages {
            match m {
                Message::System(s) => {
                    out.push_str("<|im_start|>system\n");
                    out.push_str(s);
                    out.push_str("<|im_end|>\n");
                }
                Message::User(s) => {
                    out.push_str("<|im_start|>user\n");
                    out.push_str(s);
                    out.push_str("<|im_end|>\n");
                }
                Message::Assistant { content, .. } => {
                    if let Some(c) = content {
                        out.push_str("<|im_start|>assistant\n");
                        out.push_str(c);
                        out.push_str("<|im_end|>\n");
                    }
                }
                Message::Tool { content, .. } => {
                    out.push_str("<|im_start|>tool\n");
                    out.push_str(content);
                    out.push_str("<|im_end|>\n");
                }
            }
        }
        out.push_str("<|im_start|>assistant\n");
        out
    }

    fn load_model(device: &Device) -> Result<Loaded, String> {
        // use hf-hub sync API (cached at HF_HOME or ~/.cache/huggingface/hub)
        let api = hf_hub::api::sync::Api::new().map_err(|e| format!("hf hub api: {e}"))?;
        let repo = api.model(MODEL_ID.to_string());
        let tokenizer_path = repo.get("tokenizer.json").map_err(|e| format!("tokenizer.json: {e}"))?;
        let config_path = repo.get("config.json").map_err(|e| format!("config.json: {e}"))?;
        // SmolLM2-360M-Instruct is single file model.safetensors
        let model_path = match repo.get("model.safetensors") {
            Ok(p) => p,
            Err(_) => {
                // try sharded
                repo.get("model-00001-of-00001.safetensors")
                    .map_err(|e| format!("model.safetensors: {e}"))?
            }
        };
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| format!("tokenizer: {e}"))?;
        let config_str = std::fs::read_to_string(&config_path).map_err(|e| format!("config read: {e}"))?;
        let llama_config: LlamaConfig =
            serde_json::from_str(&config_str).map_err(|e| format!("config parse: {e}"))?;
        let eos_token = match &llama_config.eos_token_id {
            Some(LlamaEosToks::Single(id)) => *id,
            Some(LlamaEosToks::Multiple(ids)) => ids[0],
            None => 0,
        };
        // fallback eos
        let eos_token = tokenizer.token_to_id("<|im_end|>").unwrap_or(eos_token);
        let config = llama_config.into_config(false);
        let dtype = DType::F32;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], dtype, device)
                .map_err(|e| format!("varbuilder: {e}"))?
        };
        let cache = Cache::new(true, dtype, &config, device).map_err(|e| format!("cache: {e}"))?;
        let model = Llama::load(vb, &config).map_err(|e| format!("model load: {e}"))?;
        Ok(Loaded {
            model,
            tokenizer,
            cache,
            config,
            eos_token,
            device: device.clone(),
        })
    }

    fn generate(loaded: &mut Loaded, prompt: &str) -> Result<String, String> {
        let tokens = loaded
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| format!("encode: {e}"))?
            .get_ids()
            .to_vec();
        if tokens.is_empty() {
            return Err("empty prompt tokens".to_string());
        }
        let mut all_tokens = tokens.clone();
        let mut next_tokens = tokens;
        let mut index_pos = 0;
        let mut generated = Vec::new();

        for _ in 0..MAX_TOKENS {
            let input = Tensor::new(next_tokens.as_slice(), &loaded.device)
                .map_err(|e| format!("tensor: {e}"))?
                .unsqueeze(0)
                .map_err(|e| format!("unsqueeze: {e}"))?;
            let logits = loaded
                .model
                .forward(&input, index_pos, &mut loaded.cache)
                .map_err(|e| format!("forward: {e}"))?;
            let logits = logits
                .squeeze(0)
                .map_err(|e| format!("squeeze: {e}"))?
                .to_dtype(DType::F32)
                .map_err(|e| format!("dtype: {e}"))?;
            let logits = if logits.dim(0).map_err(|e| format!("dim: {e}"))? > 1 {
                // take last token logits
                logits
                    .narrow(0, logits.dim(0).unwrap() - 1, 1)
                    .map_err(|e| format!("narrow: {e}"))?
                    .squeeze(0)
                    .map_err(|e| format!("squeeze2: {e}"))?
            } else {
                logits
            };
            let next_token = logits
                .argmax(D::Minus1)
                .map_err(|e| format!("argmax: {e}"))?
                .to_scalar::<u32>()
                .map_err(|e| format!("scalar: {e}"))?;
            if next_token == loaded.eos_token {
                break;
            }
            // also break on <|im_end|>
            generated.push(next_token);
            all_tokens.push(next_token);
            next_tokens = vec![next_token];
            index_pos = all_tokens.len() - 1;
            if generated.len() >= MAX_TOKENS {
                break;
            }
        }
        let text = loaded
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| format!("decode: {e}"))?;
        Ok(text.trim().to_string())
    }

    // helper removed - using candle's LlamaEosToks directly
}

#[cfg(feature = "embed")]
pub use inner::EmbedProvider;

// is_embed_model is always available
