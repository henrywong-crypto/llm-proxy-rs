mod exchange;
mod proxy;
mod tui;

use std::sync::Arc;

use axum::Router;
use axum::routing::any;
use clap::Parser;
use crossterm::execute;
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use tokio::sync::mpsc;

use crate::proxy::AppState;
use crate::tui::App;

#[derive(Parser)]
#[command(
    name = "inspector",
    about = "TUI debug proxy for LLM request/response inspection"
)]
struct Args {
    /// Target URL to proxy requests to
    #[arg(short, long)]
    target: String,

    /// Listen address
    #[arg(short, long, default_value = "127.0.0.1:8081")]
    listen: String,

    /// Disable TLS certificate verification
    #[arg(long)]
    no_verify: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let runtime = tokio::runtime::Runtime::new()?;

    let (tx, mut rx) = mpsc::channel(256);

    let client = reqwest::Client::builder()
        .danger_accept_invalid_certs(args.no_verify)
        .no_proxy()
        .build()?;

    let state = Arc::new(AppState {
        client,
        target: args.target.clone(),
        tx,
    });

    let listen_addr = args.listen.clone();

    // Spawn proxy server in background
    runtime.spawn(async move {
        let app = Router::new()
            .route("/{*path}", any(proxy::proxy_handler))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind(&listen_addr)
            .await
            .expect("Failed to bind listener");

        axum::serve(listener, app).await.unwrap();
    });

    // Setup terminal
    terminal::enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let mut app = App::new(args.listen.clone(), args.target.clone());
    let mut needs_redraw = true;

    // Main TUI loop
    let result = loop {
        // Drain new exchanges from channel
        if app.drain_channel(&mut rx) {
            needs_redraw = true;
        }

        // Render only when state changed
        if needs_redraw {
            if let Err(e) = terminal.draw(|frame| app.render(frame)) {
                break Err(e.into());
            }
            needs_redraw = false;
        }

        // Handle input events
        match app.handle_event() {
            Ok(changed) => {
                if changed {
                    needs_redraw = true;
                }
            }
            Err(e) => break Err(e.into()),
        }

        if app.should_quit {
            break Ok(());
        }
    };

    // Restore terminal
    terminal::disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}
