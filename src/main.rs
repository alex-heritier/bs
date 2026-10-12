use std::io::{self, BufRead, Write};

fn main() {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    loop {
        print!("> ");
        io::stdout().flush().unwrap();
        match lines.next() {
            Some(Ok(line)) => {
                let l = line.trim();
                if l.is_empty() { continue; }
                println!("{}", eval(l));
            }
            _ => break,
        }
    }
}

// Placeholder evaluator: echoes input until a real language lands.
fn eval(s: &str) -> String {
    match s {
        "quit" | "exit" => std::process::exit(0),
        _ => format!("=> {}", s),
    }
}
