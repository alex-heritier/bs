use async_openai::config::OpenAIConfig;
use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionTools, CreateChatCompletionRequestArgs,
};
use async_openai::Client;

pub struct Provider {
    client: Client<OpenAIConfig>,
    model: String,
}

impl Provider {
    pub fn new(url: &str, api_key: &str, model: &str) -> Self {
        let config = OpenAIConfig::new()
            .with_api_key(api_key)
            .with_api_base(url);

        Self {
            client: Client::with_config(config),
            model: model.to_owned(),
        }
    }

    pub async fn complete(
        &self,
        messages: Vec<ChatCompletionRequestMessage>,
        tools: Vec<ChatCompletionTools>,
    ) -> Result<String, async_openai::error::OpenAIError> {
        let mut request = CreateChatCompletionRequestArgs::default();
        request.model(&self.model).messages(messages);
        if !tools.is_empty() {
            request.tools(tools);
        }

        let response = self.client.chat().create(request.build()?).await?;
        response
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .ok_or_else(|| async_openai::error::OpenAIError::InvalidArgument("empty response".into()))
    }
}
