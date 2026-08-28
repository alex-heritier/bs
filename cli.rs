mod agent;
mod config;
mod env;
mod provider;
mod session;
mod tools;

use std::io::{self, BufRead, Write};

#[tokio::main]
async fn main() {
    let config = config::Config::load().expect("could not load ~/.bs");
    let mut session = session_path()
        .map(|path| session::Session::open(&path).expect("could not open session file"));
    let stdin = io::stdin();
    let mut agent = agent::Agent::new(&config);

    println!("No BS boostrapping!");
    println!("0.0.1-alpha");

    print!("> ");
    io::stdout().flush().unwrap();
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }

        let response = agent.respond(&line).await.expect("provider request failed");
        if let Some(session) = &mut session {
            session
                .append("user", &line)
                .expect("could not write session");
            session
                .append("assistant", &response)
                .expect("could not write session");
        }
        println!("{}", response);
        print!("> ");
        io::stdout().flush().unwrap();
    }
}

fn session_path() -> Option<String> {
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        if argument == "--session" {
            return Some(args.next().expect("--session requires a file path"));
        }
    }
    None
}
