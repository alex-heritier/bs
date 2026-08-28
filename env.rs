use std::env;

pub fn openrouter_api_key() -> Option<String> {
    env::var("OPENROUTER_API_KEY")
        .ok()
        .filter(|key| !key.trim().is_empty())
}
