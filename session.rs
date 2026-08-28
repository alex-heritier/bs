use std::fs::{File, OpenOptions};
use std::io::{self, Write};

pub struct Session {
    file: File,
}

impl Session {
    pub fn open(path: &str) -> io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self { file })
    }

    pub fn append(&mut self, role: &str, content: &str) -> io::Result<()> {
        writeln!(
            self.file,
            "{{\"role\":\"{}\",\"content\":\"{}\"}}",
            escape(role),
            escape(content)
        )
    }
}

fn escape(value: &str) -> String {
    value
        .chars()
        .flat_map(|character| match character {
            '"' => "\\\"".chars().collect::<Vec<_>>(),
            '\\' => "\\\\".chars().collect(),
            '\n' => "\\n".chars().collect(),
            '\r' => "\\r".chars().collect(),
            '\t' => "\\t".chars().collect(),
            character if character.is_control() => {
                format!("\\u{:04x}", character as u32).chars().collect()
            }
            character => vec![character],
        })
        .collect()
}
