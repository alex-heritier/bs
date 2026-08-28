use async_openai::config::OpenAIConfig;
use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestToolMessageArgs, ChatCompletionRequestUserMessageArgs, ChatCompletionTool,
    ChatCompletionTools, CreateChatCompletionRequestArgs, FunctionCall, FunctionObject,
};
use async_openai::Client;

pub type Error = async_openai::error::OpenAIError;

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

pub struct Provider {
    client: Client<OpenAIConfig>,
    model: String,
}

impl Provider {
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
            .collect::<Result<Vec<_>, _>>()?;
        let mut request = CreateChatCompletionRequestArgs::default();
        request
            .model(&self.model)
            .messages(messages)
            .tools(tool_definitions());
        let response = self.client.chat().create(request.build()?).await?;
        let message = response
            .choices
            .into_iter()
            .next()
            .map(|choice| choice.message)
            .ok_or_else(|| Error::InvalidArgument("empty response".into()))?;
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

#[allow(deprecated)]
fn to_openai_message(message: &Message) -> Result<ChatCompletionRequestMessage, Error> {
    Ok(match message {
        Message::System(content) => ChatCompletionRequestMessage::System(
            ChatCompletionRequestSystemMessageArgs::default()
                .content(content.clone())
                .build()?,
        ),
        Message::User(content) => ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content(content.clone())
                .build()?,
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
                .build()?,
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
