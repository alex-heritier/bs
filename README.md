# bs - Bootstrap w/o BS

bs is an ultra-minimal AI agent meant to help users bootstrap their computer systems with the help of AI.

bs is:
- ultra minimal
- system dependency free
- small static binary executable
- (optional) comes bundled with a small CPU inference-able local model for bootstrapping without internet access

bs is not:
- a fully featured coding tool like OpenCode
- a fully featured agent platform like Hermes

## Technical decisions

bs is written in Rust so it can be resource efficient and small in binary size. It uses a single ~/.bs config file
to configure model API url, model name, and model API key. It can optionally track chat sessions using a session .jsonl file.

As a stretch goal, bs can optionally be built with an embedded local inference stack and model. Maybe llama-cpp Rust port
if such a thing exists, and a Liquid small sub 500M model.

As another stretch goal, bs can either be built with or dynamically download additional extensions, for now the only one
being a Telegram bot chat gateway. These extensions are downloaded from the github repo. This is not meant to be a
flexible community extensions marketplace but rather a fixed set of first party extensions.

## Files

The resulting codebase must strictly adhere to the following set of files, any modifications to this must be proposed to and approved by myself.

- cli.rs : CLI entrypoint
- config.rs : manage ~/.bs config file read & write
- env.rs : manage getting AI API keys from env
- agent.rs : AI agent and loop
- tools.rs : tools made accessible to agent: bash, str_replace_editor
- session.rs : read & write session jsonl files
- provider.rs : manage inference provider connection
- telegram.rs : optional Telegram gateway entrypoint, needs to be either embeddable in the executable or downloadable separately as a binary extension
- embed_ai.rs : optional embedded-in-the-executable small model & local CPU inference stack
