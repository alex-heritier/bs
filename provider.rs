use async_openai::config::OpenAIConfig;
use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestToolMessageArgs, ChatCompletionRequestUserMessageArgs, ChatCompletionTool,
    ChatCompletionTools, CreateChatCompletionRequestArgs, FunctionCall, FunctionObject,
};
use async_openai::Client;

// --- Error: unified for remote + local ---
#[derive(Debug)]
pub enum Error {
    OpenAI(async_openai::error::OpenAIError),
    Embed(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::OpenAI(e) => write!(f, "{e}"),
            Error::Embed(s) => write!(f, "embed: {s}"),
        }
    }
}
impl std::error::Error for Error {}

impl From<async_openai::error::OpenAIError> for Error {
    fn from(e: async_openai::error::OpenAIError) -> Self {
        Error::OpenAI(e)
    }
}

#[derive(Clone)]
pub enum Message {
    System(String),
    User(String),
    Assistant {
        content: Option<String>,
        tool_calls: Vec<ToolCall>,
    },
    Tool {
        id: String,
        content: String,
    },
}

#[derive(Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

pub struct Completion {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

// helper for embed url detection
#[allow(dead_code)]
fn is_embed_url(url: &str) -> bool {
    let u = url.trim().to_lowercase();
    u == "embed" || u == "local"
}

// Remote provider (existing OpenAI-compatible)
pub struct RemoteProvider {
    client: Client<OpenAIConfig>,
    model: String,
}

impl RemoteProvider {
    pub fn new(url: &str, api_key: &str, model: &str) -> Self {
        let config = OpenAIConfig::new().with_api_key(api_key).with_api_base(url);
        Self {
            client: Client::with_config(config),
            model: model.to_owned(),
        }
    }

    pub async fn complete(&self, messages: &[Message]) -> Result<Completion, Error> {
        let messages = messages
            .iter()
            .map(to_openai_message)
            .collect::<Result<Vec<_>, Error>>()?;
        let mut request = CreateChatCompletionRequestArgs::default();
        request
            .model(&self.model)
            .messages(messages)
            .tools(tool_definitions());
        let response = self
            .client
            .chat()
            .create(request.build().map_err(|e| Error::Embed(e.to_string()))?)
            .await
            .map_err(Error::OpenAI)?;
        let message = response
            .choices
            .into_iter()
            .next()
            .map(|choice| choice.message)
            .ok_or_else(|| Error::Embed("empty response".into()))?;
        let tool_calls = message
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .filter_map(|call| match call {
                ChatCompletionMessageToolCalls::Function(call) => Some(ToolCall {
                    id: call.id,
                    name: call.function.name,
                    arguments: call.function.arguments,
                }),
                ChatCompletionMessageToolCalls::Custom(_) => None,
            })
            .collect();
        Ok(Completion {
            content: message.content,
            tool_calls,
        })
    }
}

// Unified Provider enum
pub enum Provider {
    Remote(RemoteProvider),
    #[cfg(feature = "embed")]
    Local(crate::embed_ai::EmbedProvider),
}

impl Provider {
    pub fn new(url: &str, api_key: &str, model: &str) -> Self {
        #[cfg(feature = "embed")]
        {
            if crate::embed_ai::is_embed_model(model) || is_embed_url(url) {
                match crate::embed_ai::EmbedProvider::new() {
                    Ok(local) => return Self::Local(local),
                    Err(e) => {
                        eprintln!("embed init failed ({e}), falling back to remote");
                    }
                }
            }
        }
        Self::Remote(RemoteProvider::new(url, api_key, model))
    }

    pub async fn complete(&self, messages: &[Message]) -> Result<Completion, Error> {
        match self {
            Provider::Remote(r) => r.complete(messages).await,
            #[cfg(feature = "embed")]
            Provider::Local(l) => l.complete(messages).await,
        }
    }
}

#[allow(deprecated)]
fn to_openai_message(message: &Message) -> Result<ChatCompletionRequestMessage, Error> {
    Ok(match message {
        Message::System(content) => ChatCompletionRequestMessage::System(
            ChatCompletionRequestSystemMessageArgs::default()
                .content(content.clone())
                .build()
                .map_err(|e| Error::Embed(e.to_string()))?,
        ),
        Message::User(content) => ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content(content.clone())
                .build()
                .map_err(|e| Error::Embed(e.to_string()))?,
        ),
        Message::Assistant {
            content,
            tool_calls,
        } => {
            let calls = tool_calls
                .iter()
                .map(|call| {
                    ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
                        id: call.id.clone(),
                        function: FunctionCall {
                            name: call.name.clone(),
                            arguments: call.arguments.clone(),
                        },
                    })
                })
                .collect();
            ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
                content: content
                    .clone()
                    .map(ChatCompletionRequestAssistantMessageContent::Text),
                refusal: None,
                name: None,
                audio: None,
                tool_calls: (!tool_calls.is_empty()).then_some(calls),
                function_call: None,
            })
        }
        Message::Tool { id, content } => ChatCompletionRequestMessage::Tool(
            ChatCompletionRequestToolMessageArgs::default()
                .tool_call_id(id.clone())
                .content(content.clone())
                .build()
                .map_err(|e| Error::Embed(e.to_string()))?,
        ),
    })
}

fn tool_definitions() -> Vec<ChatCompletionTools> {
    vec![
        function_tool(
            "bash",
            "Run a shell command.",
            serde_json::json!({
                "type": "object",
                "properties": { "command": { "type": "string" } },
                "required": ["command"]
            }),
        ),
        function_tool(
            "str_replace_editor",
            "Create or edit a file by replacing text.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "old": { "type": "string" },
                    "new": { "type": "string" }
                },
                "required": ["path", "old", "new"]
            }),
        ),
    ]
}

fn function_tool(
    name: &str,
    description: &str,
    parameters: serde_json::Value,
) -> ChatCompletionTools {
    ChatCompletionTools::Function(ChatCompletionTool {
        function: FunctionObject {
            name: name.to_owned(),
            description: Some(description.to_owned()),
            parameters: Some(parameters),
            strict: None,
        },
    })
}

// helper for embed_ai to avoid circular dep, re-export LlamaEosToks shape when needed
#[cfg(feature = "embed")]
#[allow(dead_code)]
pub mod llama_eos_helper {
    #[derive(Debug, Clone, serde::Deserialize)]
    #[serde(untagged)]
    pub enum LlamaEosToks {
        Single(u32),
        Multiple(Vec<u32>),
    }
}
