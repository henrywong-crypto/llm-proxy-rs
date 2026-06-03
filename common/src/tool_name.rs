use std::collections::HashMap;

/// AWS Bedrock Converse rejects tool names longer than this (Anthropic allows
/// up to 128). Longer names are aliased on the way out and restored on the way
/// back via [`ToolNameMap`].
pub const BEDROCK_TOOL_NAME_MAX_LEN: usize = 64;

/// Maps a tool name to one Bedrock accepts. Names within the limit pass through
/// unchanged; longer ones are truncated and suffixed with a stable hash, so
/// distinct names stay distinct and a given name always aliases to the same
/// value — letting replayed `tool_use` history line up with the tool spec
/// without any cross-request state.
pub fn alias_tool_name(name: &str) -> String {
    if name.len() <= BEDROCK_TOOL_NAME_MAX_LEN {
        return name.to_owned();
    }
    // prefix + '_' + 8 hex digits == BEDROCK_TOOL_NAME_MAX_LEN.
    let prefix: String = name.chars().take(BEDROCK_TOOL_NAME_MAX_LEN - 9).collect();
    format!("{prefix}_{:08x}", fnv1a32(name))
}

/// Reverse lookup from a Bedrock alias back to the original tool name, built
/// once per request from its configured tools.
#[derive(Debug, Default)]
pub struct ToolNameMap(HashMap<String, String>);

impl ToolNameMap {
    /// Builds the map by aliasing each original name, keeping only entries that
    /// were actually shortened.
    pub fn from_originals<I: IntoIterator<Item = String>>(originals: I) -> Self {
        Self(
            originals
                .into_iter()
                .filter_map(|name| {
                    let alias = alias_tool_name(&name);
                    (alias != name).then_some((alias, name))
                })
                .collect(),
        )
    }

    /// Restores the original name for a Bedrock alias, or returns it unchanged
    /// when it was never aliased.
    pub fn restore<'a>(&'a self, bedrock_name: &'a str) -> &'a str {
        self.0
            .get(bedrock_name)
            .map_or(bedrock_name, String::as_str)
    }
}

/// FNV-1a (32-bit): a tiny, stable, dependency-free hash for the alias suffix.
fn fnv1a32(s: &str) -> u32 {
    s.bytes().fold(0x811c_9dc5, |hash, byte| {
        (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_within_limit_pass_through() {
        assert_eq!(alias_tool_name("get_weather"), "get_weather");
        let exactly_max = "a".repeat(BEDROCK_TOOL_NAME_MAX_LEN);
        assert_eq!(alias_tool_name(&exactly_max), exactly_max);
    }

    #[test]
    fn long_names_are_shortened_within_limit_and_stable() {
        let name = "x".repeat(100);
        let alias = alias_tool_name(&name);
        assert!(alias.len() <= BEDROCK_TOOL_NAME_MAX_LEN);
        assert_eq!(alias, alias_tool_name(&name));
    }

    #[test]
    fn long_names_sharing_a_prefix_get_distinct_aliases() {
        let a = format!("{}_A", "p".repeat(70));
        let b = format!("{}_B", "p".repeat(70));
        assert_ne!(alias_tool_name(&a), alias_tool_name(&b));
    }

    #[test]
    fn map_restores_aliased_names_and_passes_others_through() {
        let long = format!("tool_{}", "z".repeat(80));
        let map = ToolNameMap::from_originals([long.clone(), "short".to_owned()]);
        assert_eq!(map.restore(&alias_tool_name(&long)), long);
        assert_eq!(map.restore("short"), "short"); // never aliased, not stored
        assert_eq!(map.restore("unknown"), "unknown");
    }
}
