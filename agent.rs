use crate::config::Config;
use crate::provider::{self, Message};
use crate::tools;

const SYSTEM_PROMPT: &str = "You are a helpful software engineer assistant.";

pub struct Agent {
    provider: provider::Provider,
    messages: Vec<Message>,
}

impl Agent {
    pub fn new(config: &Config) -> Self {
        Self {
            provider: provider::Provider::new(
                &config.provider_url,
                &config.provider_api_key,
                &config.model,
            ),
            messages: vec![Message::System(SYSTEM_PROMPT.to_owned())],
        }
    }

    pub async fn respond(&mut self, input: &str) -> Result<String, provider::Error> {
        self.messages.push(Message::User(input.to_owned()));

        for _ in 0..16 {
            let completion = self.provider.complete(&self.messages).await?;
            if completion.tool_calls.is_empty() {
                let content = completion.content.unwrap_or_default();
                self.messages.push(Message::Assistant {
                    content: Some(content.clone()),
                    tool_calls: Vec::new(),
                });
                return Ok(content);
            }

            self.messages.push(Message::Assistant {
                content: completion.content,
                tool_calls: completion.tool_calls.clone(),
            });
            for call in completion.tool_calls {
                let result = tools::call(&call.name, &call.arguments);
                let result = match result {
                    Ok(result) => result,
                    Err(error) => format!("error: {error}"),
                };
                self.messages.push(Message::Tool {
                    id: call.id,
                    content: result,
                });
            }
        }

        Ok("tool call limit reached".to_owned())
    }
}
