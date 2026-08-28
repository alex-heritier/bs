use std::env as std_env;
use std::fs;
use std::io;
use std::path::PathBuf;

const PROVIDER_URL: &str = "https://openrouter.ai/api/v1";
const MODEL: &str = "openrouter/free";
const TEMPLATE: &str =
    "provider-url=https://openrouter.ai/api/v1\nprovider-api-key=\nmodel=openrouter/free\n";

#[derive(Debug)]
pub struct Config {
    pub provider_url: String,
    pub provider_api_key: String,
    pub model: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            provider_url: PROVIDER_URL.to_owned(),
            provider_api_key: String::new(),
            model: MODEL.to_owned(),
        }
    }
}

impl Config {
    pub fn load() -> io::Result<Self> {
        let path = config_path()?;
        let contents = match fs::read_to_string(&path) {
            Ok(contents) => contents,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                fs::write(&path, TEMPLATE)?;
                println!("Created config file at {}", path.display());
                TEMPLATE.to_owned()
            }
            Err(error) => return Err(error),
        };

        let mut config = Self::default();
        for line in contents.lines() {
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };

            match key.trim() {
                "provider-url" => config.provider_url = value.trim().to_owned(),
                "provider-api-key" => config.provider_api_key = value.trim().to_owned(),
                "model" => config.model = value.trim().to_owned(),
                _ => {}
            }
        }

        if config.provider_api_key.trim().is_empty() {
            match crate::env::openrouter_api_key() {
                Some(key) => {
                    config.provider_api_key = key;
                    println!(
                        "Config has no API key; using OPENROUTER_API_KEY from the environment"
                    );
                }
                None => println!(
                    "Config has no API key and OPENROUTER_API_KEY was not found in the environment"
                ),
            }
        }

        Ok(config)
    }
}

fn config_path() -> io::Result<PathBuf> {
    std_env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join(".bs"))
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "HOME is not set"))
}
