mod cli;
mod commands;
mod error;

use crate::cli::{Cli, Commands};
use anyhow::Result;
use clap::Parser;
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match &cli.command {
        Commands::Test {
            input_directory_path,
            output_directory_path,
        } => {
            let input_directory_path = Path::new(input_directory_path).canonicalize()?;
            let output_directory_path = PathBuf::from(output_directory_path);

            commands::test::run(input_directory_path, output_directory_path)?;
        }
    };

    Ok(())
}
