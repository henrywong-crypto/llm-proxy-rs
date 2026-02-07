use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Tabs,
};
use tokio::sync::mpsc;

use crate::exchange::CapturedExchange;

const TAB_HEADERS: usize = 0;
const TAB_REQUEST: usize = 1;
const TAB_RESPONSE: usize = 2;

/// Scroll indices: 0=req headers, 1=res headers, 2=request body, 3=response SSE events, 4=response body
const SCROLL_REQ_HEADERS: usize = 0;
const SCROLL_RES_HEADERS: usize = 1;
const SCROLL_REQUEST: usize = 2;
const SCROLL_RES_EVENTS: usize = 3;
const SCROLL_RES_BODY: usize = 4;

enum View {
    List,
    Detail {
        tab: usize,
        /// For split-pane tabs (Headers, Response): 0 = top, 1 = bottom
        pane_focus: usize,
        scroll: [usize; 5],
    },
}

pub struct App {
    exchanges: Vec<CapturedExchange>,
    selected: usize,
    scroll_offset: usize,
    listen_addr: String,
    target_addr: String,
    view: View,
    pub should_quit: bool,
}

impl App {
    pub fn new(listen_addr: String, target_addr: String) -> Self {
        Self {
            exchanges: Vec::new(),
            selected: 0,
            scroll_offset: 0,
            listen_addr,
            target_addr,
            view: View::List,
            should_quit: false,
        }
    }

    pub fn drain_channel(&mut self, rx: &mut mpsc::Receiver<CapturedExchange>) -> bool {
        let was_at_bottom = self.is_at_bottom();
        let mut received = false;
        while let Ok(exchange) = rx.try_recv() {
            self.exchanges.push(exchange);
            received = true;
        }
        if received && was_at_bottom && !self.exchanges.is_empty() {
            if matches!(self.view, View::List) {
                self.selected = self.exchanges.len() - 1;
            }
        }
        received
    }

    pub fn handle_event(&mut self) -> std::io::Result<bool> {
        if event::poll(std::time::Duration::from_millis(33))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    return Ok(false);
                }
                return match &self.view {
                    View::List => Ok(self.handle_list_key(key.code)),
                    View::Detail { .. } => Ok(self.handle_detail_key(key.code)),
                };
            }
        }
        Ok(false)
    }

    fn handle_list_key(&mut self, key: KeyCode) -> bool {
        match key {
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            KeyCode::Up => self.select_prev(),
            KeyCode::Down => self.select_next(),
            KeyCode::Enter | KeyCode::Char(' ') => {
                if !self.exchanges.is_empty() {
                    self.view = View::Detail {
                        tab: TAB_HEADERS,
                        pane_focus: 0,
                        scroll: [0; 5],
                    };
                }
            }
            _ => return false,
        }
        true
    }

    fn handle_detail_key(&mut self, key: KeyCode) -> bool {
        match key {
            KeyCode::Esc | KeyCode::Char('q') => {
                self.view = View::List;
            }
            KeyCode::Tab => {
                // Toggle pane focus on split tabs (Headers, Response)
                if let View::Detail {
                    tab, pane_focus, ..
                } = &mut self.view
                {
                    if *tab == TAB_HEADERS || *tab == TAB_RESPONSE {
                        *pane_focus = if *pane_focus == 0 { 1 } else { 0 };
                    } else {
                        *tab = (*tab + 1) % 3;
                        *pane_focus = 0;
                    }
                }
            }
            KeyCode::Right => {
                if let View::Detail {
                    tab, pane_focus, ..
                } = &mut self.view
                {
                    *tab = (*tab + 1) % 3;
                    *pane_focus = 0;
                }
            }
            KeyCode::BackTab | KeyCode::Left => {
                if let View::Detail {
                    tab, pane_focus, ..
                } = &mut self.view
                {
                    *tab = (*tab + 2) % 3;
                    *pane_focus = 0;
                }
            }
            KeyCode::Up => {
                if let View::Detail {
                    tab,
                    pane_focus,
                    scroll,
                } = &mut self.view
                {
                    let idx = active_scroll_idx(*tab, *pane_focus);
                    scroll[idx] = scroll[idx].saturating_sub(1);
                }
            }
            KeyCode::Down => {
                if let View::Detail {
                    tab,
                    pane_focus,
                    scroll,
                } = &mut self.view
                {
                    let idx = active_scroll_idx(*tab, *pane_focus);
                    scroll[idx] += 1;
                }
            }
            KeyCode::PageUp => {
                if let View::Detail {
                    tab,
                    pane_focus,
                    scroll,
                } = &mut self.view
                {
                    let idx = active_scroll_idx(*tab, *pane_focus);
                    scroll[idx] = scroll[idx].saturating_sub(20);
                }
            }
            KeyCode::PageDown => {
                if let View::Detail {
                    tab,
                    pane_focus,
                    scroll,
                } = &mut self.view
                {
                    let idx = active_scroll_idx(*tab, *pane_focus);
                    scroll[idx] += 20;
                }
            }
            KeyCode::Home => {
                if let View::Detail {
                    tab,
                    pane_focus,
                    scroll,
                } = &mut self.view
                {
                    let idx = active_scroll_idx(*tab, *pane_focus);
                    scroll[idx] = 0;
                }
            }
            _ => return false,
        }
        true
    }

    fn select_prev(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        }
    }

    fn select_next(&mut self) {
        if !self.exchanges.is_empty() && self.selected < self.exchanges.len() - 1 {
            self.selected += 1;
        }
    }

    fn is_at_bottom(&self) -> bool {
        self.exchanges.is_empty() || self.selected == self.exchanges.len() - 1
    }

    pub fn render(&mut self, frame: &mut Frame) {
        match self.view {
            View::List => self.render_list(frame),
            View::Detail { .. } => self.render_detail(frame),
        }
    }

    // ── List view ──────────────────────────────────────────────

    fn render_list(&mut self, frame: &mut Frame) {
        let area = frame.area();
        let layout = Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).split(area);

        self.render_exchanges(frame, layout[0]);
        self.render_list_status_bar(frame, layout[1]);
    }

    fn render_exchanges(&mut self, frame: &mut Frame, area: Rect) {
        let title = format!(" Inspector ({} → {}) ", self.listen_addr, self.target_addr);
        let block = Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray));

        let inner = block.inner(area);
        frame.render_widget(block, area);

        if self.exchanges.is_empty() {
            frame.render_widget(
                Paragraph::new("Waiting for requests...")
                    .style(Style::default().fg(Color::DarkGray)),
                inner,
            );
            return;
        }

        let mut all_lines: Vec<(Line, Option<usize>)> = Vec::new();

        for (idx, exchange) in self.exchanges.iter().enumerate() {
            let is_selected = idx == self.selected;
            for line in build_summary_lines(exchange, is_selected) {
                all_lines.push((line, Some(idx)));
            }
            all_lines.push((Line::default(), None));
        }

        let visible_height = inner.height as usize;
        self.adjust_scroll(&all_lines, visible_height);

        let visible_lines: Vec<Line> = all_lines
            .iter()
            .skip(self.scroll_offset)
            .take(visible_height)
            .map(|(line, _)| line.clone())
            .collect();

        frame.render_widget(Paragraph::new(Text::from(visible_lines)), inner);

        if all_lines.len() > visible_height {
            let mut state = ScrollbarState::new(all_lines.len().saturating_sub(visible_height))
                .position(self.scroll_offset);
            frame.render_stateful_widget(
                Scrollbar::new(ScrollbarOrientation::VerticalRight),
                area,
                &mut state,
            );
        }
    }

    fn adjust_scroll(&mut self, all_lines: &[(Line, Option<usize>)], visible_height: usize) {
        let mut start = 0;
        let mut end = 0;
        let mut found = false;

        for (i, (_, idx)) in all_lines.iter().enumerate() {
            if *idx == Some(self.selected) {
                if !found {
                    start = i;
                    found = true;
                }
                end = i;
            }
        }
        if !found {
            return;
        }
        if start < self.scroll_offset {
            self.scroll_offset = start;
        } else if end >= self.scroll_offset + visible_height {
            self.scroll_offset = end.saturating_sub(visible_height - 1);
        }
    }

    fn render_list_status_bar(&self, frame: &mut Frame, area: Rect) {
        let count = self.exchanges.len();
        let bar = Line::from(vec![
            key_span(" ↑↓"),
            dim_span(" Navigate  "),
            key_span("Enter"),
            dim_span(" Inspect  "),
            key_span("q"),
            dim_span(" Quit  "),
            dim_span(&format!("│ {} requests", count)),
        ]);
        frame.render_widget(
            Paragraph::new(bar).style(Style::default().bg(Color::Black)),
            area,
        );
    }

    // ── Detail view ────────────────────────────────────────────

    fn render_detail(&mut self, frame: &mut Frame) {
        let area = frame.area();

        let Some(exchange) = self.exchanges.get(self.selected) else {
            self.view = View::List;
            return;
        };

        let (tab, pane_focus, scroll) = match &self.view {
            View::Detail {
                tab,
                pane_focus,
                scroll,
            } => (*tab, *pane_focus, *scroll),
            _ => return,
        };

        // Layout: header(4) + tabs(1) + content + status(1)
        let layout = Layout::vertical([
            Constraint::Length(4),
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .split(area);

        // ── Header ──
        render_detail_header(frame, layout[0], exchange);

        // ── Tab bar ──
        let tab_titles = vec!["Headers", "Request", "Response"];
        let tabs = Tabs::new(tab_titles)
            .select(tab)
            .style(Style::default().fg(Color::DarkGray))
            .highlight_style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
            .divider("│");
        frame.render_widget(tabs, layout[1]);

        // ── Tab content ──
        let content_area = layout[2];

        match tab {
            TAB_HEADERS => {
                let panes =
                    Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)])
                        .split(content_area);

                let req_text = build_header_list(&exchange.request_headers);
                let res_text = build_header_list(&exchange.response_headers);

                render_scrollable_text(
                    frame,
                    panes[0],
                    " Request Headers",
                    &req_text,
                    scroll[SCROLL_REQ_HEADERS],
                    pane_focus == 0,
                );
                render_scrollable_text(
                    frame,
                    panes[1],
                    " Response Headers",
                    &res_text,
                    scroll[SCROLL_RES_HEADERS],
                    pane_focus == 1,
                );

                clamp_scroll(
                    &mut self.view,
                    SCROLL_REQ_HEADERS,
                    req_text.lines().count(),
                    panes[0].height,
                );
                clamp_scroll(
                    &mut self.view,
                    SCROLL_RES_HEADERS,
                    res_text.lines().count(),
                    panes[1].height,
                );
            }
            TAB_REQUEST => {
                let text = exchange
                    .request_body
                    .as_ref()
                    .map(|v| serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string()))
                    .unwrap_or_default();
                render_scrollable_text(
                    frame,
                    content_area,
                    " Request Body",
                    &text,
                    scroll[SCROLL_REQUEST],
                    true,
                );
                clamp_scroll(
                    &mut self.view,
                    SCROLL_REQUEST,
                    text.lines().count(),
                    content_area.height,
                );
            }
            TAB_RESPONSE => {
                if exchange.sse_events.is_empty() {
                    let text = exchange
                        .response_body
                        .as_ref()
                        .map(|v| serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string()))
                        .unwrap_or_default();
                    render_scrollable_text(
                        frame,
                        content_area,
                        " Response Body",
                        &text,
                        scroll[SCROLL_RES_BODY],
                        true,
                    );
                    clamp_scroll(
                        &mut self.view,
                        SCROLL_RES_BODY,
                        text.lines().count(),
                        content_area.height,
                    );
                } else {
                    let panes =
                        Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)])
                            .split(content_area);

                    let events_text = build_sse_events_text(&exchange.sse_events);
                    let body_text = exchange
                        .response_body
                        .as_ref()
                        .map(|v| serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string()))
                        .unwrap_or_default();

                    render_scrollable_text(
                        frame,
                        panes[0],
                        " SSE Events",
                        &events_text,
                        scroll[SCROLL_RES_EVENTS],
                        pane_focus == 0,
                    );
                    render_scrollable_text(
                        frame,
                        panes[1],
                        " Response Body",
                        &body_text,
                        scroll[SCROLL_RES_BODY],
                        pane_focus == 1,
                    );

                    clamp_scroll(
                        &mut self.view,
                        SCROLL_RES_EVENTS,
                        events_text.lines().count(),
                        panes[0].height,
                    );
                    clamp_scroll(
                        &mut self.view,
                        SCROLL_RES_BODY,
                        body_text.lines().count(),
                        panes[1].height,
                    );
                }
            }
            _ => {}
        }

        // ── Status bar ──
        let has_split =
            tab == TAB_HEADERS || (tab == TAB_RESPONSE && !exchange.sse_events.is_empty());
        let bar = if has_split {
            Line::from(vec![
                key_span(" ↑↓"),
                dim_span(" Scroll  "),
                key_span("Tab"),
                dim_span(" Switch Pane  "),
                key_span("←→"),
                dim_span(" Switch Tab  "),
                key_span("PgUp"),
                dim_span("/"),
                key_span("PgDn"),
                dim_span(" Page  "),
                key_span("Esc"),
                dim_span(" Back"),
            ])
        } else {
            Line::from(vec![
                key_span(" ↑↓"),
                dim_span(" Scroll  "),
                key_span("Tab"),
                dim_span("/"),
                key_span("←→"),
                dim_span(" Switch Tab  "),
                key_span("PgUp"),
                dim_span("/"),
                key_span("PgDn"),
                dim_span(" Page  "),
                key_span("Esc"),
                dim_span(" Back"),
            ])
        };
        frame.render_widget(
            Paragraph::new(bar).style(Style::default().bg(Color::Black)),
            layout[3],
        );
    }
}

// ── Helpers ────────────────────────────────────────────────────

fn key_span(s: &str) -> Span<'static> {
    Span::styled(
        s.to_string(),
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    )
}

fn dim_span(s: &str) -> Span<'static> {
    Span::styled(s.to_string(), Style::default().fg(Color::DarkGray))
}

fn status_color(code: u16) -> Color {
    match code {
        200..=299 => Color::Green,
        300..=399 => Color::Cyan,
        400..=499 => Color::Yellow,
        500..=599 => Color::Red,
        _ => Color::White,
    }
}

fn build_summary_lines(exchange: &CapturedExchange, is_selected: bool) -> Vec<Line<'static>> {
    let sc = status_color(exchange.status);

    let highlight = if is_selected {
        Style::default().add_modifier(Modifier::BOLD)
    } else {
        Style::default()
    };

    let bg = if is_selected {
        Color::DarkGray
    } else {
        Color::Reset
    };

    let mut lines = Vec::new();

    lines.push(Line::from(vec![
        Span::styled(
            format!(" ▶ {} ", exchange.timestamp),
            Style::default().fg(Color::DarkGray).bg(bg),
        ),
        Span::styled(
            format!("{} ", exchange.method),
            highlight.fg(Color::Green).bg(bg),
        ),
        Span::styled(exchange.path.clone(), highlight.bg(bg)),
        Span::styled(" ", Style::default().bg(bg)),
        if exchange.error.is_some() {
            Span::styled(
                "ERR".to_string(),
                Style::default()
                    .fg(Color::Red)
                    .add_modifier(Modifier::BOLD)
                    .bg(bg),
            )
        } else {
            Span::styled(
                format!("{}", exchange.status),
                Style::default().fg(sc).add_modifier(Modifier::BOLD).bg(bg),
            )
        },
    ]));

    if !exchange.request_summary.is_empty() {
        lines.push(Line::from(vec![Span::styled(
            format!("   {}", exchange.request_summary),
            Style::default().fg(Color::White).bg(bg),
        )]));
    }

    if let Some(ref err) = exchange.error {
        lines.push(Line::from(vec![Span::styled(
            format!("   {}", err),
            Style::default().fg(Color::Red).bg(bg),
        )]));
    } else if !exchange.response_summary.is_empty() {
        lines.push(Line::from(vec![Span::styled(
            format!("   {}", exchange.response_summary),
            Style::default().fg(Color::DarkGray).bg(bg),
        )]));
    }

    lines
}

fn render_detail_header(frame: &mut Frame, area: Rect, exchange: &CapturedExchange) {
    let sc = status_color(exchange.status);
    let lines = vec![
        Line::from(vec![
            Span::styled(
                format!(" {} ", exchange.method),
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                exchange.path.clone(),
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::styled("  ", Style::default()),
            Span::styled(
                format!("{}", exchange.status),
                Style::default().fg(sc).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  {}", exchange.timestamp),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(vec![Span::styled(
            format!(" {}", exchange.request_summary),
            Style::default().fg(Color::White),
        )]),
        Line::from(vec![Span::styled(
            format!(" {}", exchange.response_summary),
            Style::default().fg(Color::DarkGray),
        )]),
    ];

    let block = Block::default()
        .borders(Borders::BOTTOM)
        .border_style(Style::default().fg(Color::DarkGray));
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn build_header_list(headers: &[(String, String)]) -> String {
    if headers.is_empty() {
        return "(none)".to_string();
    }
    headers
        .iter()
        .map(|(name, value)| format!("{}: {}", name, value))
        .collect::<Vec<_>>()
        .join("\n")
}

fn build_sse_events_text(events: &[(String, String)]) -> String {
    let mut text = String::new();
    for (i, (event_type, data)) in events.iter().enumerate() {
        let label = if event_type.is_empty() {
            format!("[{}]", i + 1)
        } else {
            format!("[{}] {}", i + 1, event_type)
        };
        text.push_str(&label);
        text.push('\n');

        if data != "[DONE]" {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                let pretty = serde_json::to_string_pretty(&json).unwrap_or_else(|_| data.clone());
                for line in pretty.lines() {
                    text.push_str("  ");
                    text.push_str(line);
                    text.push('\n');
                }
            } else {
                text.push_str("  ");
                text.push_str(data);
                text.push('\n');
            }
        } else {
            text.push_str("  [DONE]\n");
        }
        text.push('\n');
    }
    text
}

fn active_scroll_idx(tab: usize, pane_focus: usize) -> usize {
    match tab {
        TAB_HEADERS => {
            if pane_focus == 0 {
                SCROLL_REQ_HEADERS
            } else {
                SCROLL_RES_HEADERS
            }
        }
        TAB_REQUEST => SCROLL_REQUEST,
        TAB_RESPONSE => {
            if pane_focus == 0 {
                SCROLL_RES_EVENTS
            } else {
                SCROLL_RES_BODY
            }
        }
        _ => 0,
    }
}

fn clamp_scroll(view: &mut View, scroll_idx: usize, total_lines: usize, area_height: u16) {
    if let View::Detail { scroll, .. } = view {
        let visible = area_height.saturating_sub(2) as usize;
        if total_lines > visible {
            scroll[scroll_idx] = scroll[scroll_idx].min(total_lines - visible);
        } else {
            scroll[scroll_idx] = 0;
        }
    }
}

fn render_scrollable_text(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    text: &str,
    scroll: usize,
    focused: bool,
) {
    let border_color = if focused {
        Color::Cyan
    } else {
        Color::DarkGray
    };

    let block = Block::default()
        .title(title.to_string())
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    if text.is_empty() {
        frame.render_widget(
            Paragraph::new("(empty)").style(Style::default().fg(Color::DarkGray)),
            inner,
        );
        return;
    }

    let lines: Vec<Line> = text
        .lines()
        .map(|l| {
            Line::from(Span::styled(
                l.to_string(),
                Style::default().fg(Color::Gray),
            ))
        })
        .collect();

    let total = lines.len();
    frame.render_widget(Paragraph::new(lines).scroll((scroll as u16, 0)), inner);

    let visible = inner.height as usize;
    if total > visible {
        let mut sb_state = ScrollbarState::new(total.saturating_sub(visible)).position(scroll);
        frame.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight),
            area,
            &mut sb_state,
        );
    }
}
