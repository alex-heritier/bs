use std::io;
use std::process::Command;

pub fn bash(command: &str) -> io::Result<String> {
    command_output(Command::new("sh").args(["-c", command]))
}

pub fn str_replace_editor(path: &str, old: &str, new: &str) -> io::Result<String> {
    let contents = match std::fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == io::ErrorKind::NotFound && old.is_empty() => String::new(),
        Err(error) => return Err(error),
    };
    let updated = if old.is_empty() {
        new.to_owned()
    } else {
        contents.replacen(old, new, 1)
    };

    if updated == contents && !old.is_empty() {
        return Err(io::Error::other("old text was not found"));
    }

    std::fs::write(path, updated)?;
    Ok(format!("wrote {path}"))
}

fn command_output(command: &mut Command) -> io::Result<String> {
    let output = command.output()?;
    if !output.status.success() {
        return Err(io::Error::other(
            String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        ));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}
