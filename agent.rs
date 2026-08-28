use crate::tools;

const SOURCE: &str = r#"#include <stdio.h>

int main(void) {
    puts("hello from bs");
    return 0;
}
"#;

pub struct Agent {
    step: usize,
}

impl Agent {
    pub fn new() -> Self {
        Self { step: 0 }
    }

    pub fn respond(&mut self, _input: &str) -> String {
        let response = match self.step {
            0 => "I will inspect /tmp before changing anything.".to_owned(),
            1 => tool("bash: ls -la /tmp", tools::bash("ls -la /tmp")),
            2 => "The directory listing is complete. I will search for existing bs artifacts.".to_owned(),
            3 => tool(
                "bash: grep -R -n bs_mock /tmp",
                tools::bash("grep -R -n bs_mock /tmp || true")
                    .map(|output| if output.is_empty() { "no matches".to_owned() } else { output }),
            ),
            4 => "The search is complete. I will write a tiny C program in /tmp.".to_owned(),
            5 => tool(
                "str_replace_editor: write /tmp/bs_mock.c",
                tools::str_replace_editor("/tmp/bs_mock.c", "", SOURCE),
            ),
            6 => "The C source is ready. I will compile it with the system C compiler.".to_owned(),
            7 => tool(
                "bash: cc /tmp/bs_mock.c -o /tmp/bs_mock",
                tools::bash("cc /tmp/bs_mock.c -o /tmp/bs_mock"),
            ),
            8 => "Compilation succeeded. I will run the generated program.".to_owned(),
            9 => tool("bash: /tmp/bs_mock", tools::bash("/tmp/bs_mock")),
            10 => "The compiled program ran successfully and printed its greeting.".to_owned(),
            _ => "The mock tool workflow is complete.".to_owned(),
        };

        self.step += 1;
        response
    }
}

fn tool<T>(name: &str, result: std::io::Result<T>) -> String
where
    T: ToString,
{
    match result {
        Ok(output) if output.to_string().is_empty() => format!("[tool: {name}] ok"),
        Ok(output) => format!("[tool: {name}]\n{}", output.to_string()),
        Err(error) => format!("[tool: {name}] error: {error}"),
    }
}
