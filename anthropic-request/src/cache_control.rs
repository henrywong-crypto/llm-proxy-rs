use serde::{Deserialize, Serialize};

/// Bedrock rejects more than one cache point in a single content array, so when
/// an upstream request marks several blocks in one array with `cache_control`
/// we keep only the last cache point. The last one caches the largest prefix of
/// the array, and dropping the earlier ones leaves every non-cache-point block
/// untouched.
pub fn retain_last_cache_point<T>(blocks: Vec<T>, is_cache_point: impl Fn(&T) -> bool) -> Vec<T> {
    let last_cache_point = blocks.iter().rposition(&is_cache_point);

    blocks
        .into_iter()
        .enumerate()
        .filter(|(index, block)| !is_cache_point(block) || Some(*index) == last_cache_point)
        .map(|(_, block)| block)
        .collect()
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub cache_control_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

impl TryFrom<&CacheControl> for aws_sdk_bedrockruntime::types::CachePointBlock {
    type Error = anyhow::Error;

    fn try_from(cache_control: &CacheControl) -> Result<Self, Self::Error> {
        let ttl = cache_control
            .ttl
            .as_deref()
            .map(aws_sdk_bedrockruntime::types::CacheTtl::from);

        Ok(aws_sdk_bedrockruntime::types::CachePointBlock::builder()
            .r#type(aws_sdk_bedrockruntime::types::CachePointType::Default)
            .set_ttl(ttl)
            .build()?)
    }
}

#[cfg(test)]
mod tests {
    use aws_sdk_bedrockruntime::types::{CachePointBlock, CacheTtl};

    use super::*;

    #[test]
    fn retain_last_cache_point_keeps_only_the_final_one() {
        // Models the offending wire order: [block, cp, block, cp].
        let blocks = vec!["text", "CP", "tool", "CP"];
        let kept = retain_last_cache_point(blocks, |b| *b == "CP");
        assert_eq!(kept, vec!["text", "tool", "CP"]);
    }

    #[test]
    fn retain_last_cache_point_leaves_single_cache_point_untouched() {
        let blocks = vec!["tool", "CP"];
        let kept = retain_last_cache_point(blocks, |b| *b == "CP");
        assert_eq!(kept, vec!["tool", "CP"]);
    }

    #[test]
    fn retain_last_cache_point_noop_without_cache_points() {
        let blocks = vec!["a", "b", "c"];
        let kept = retain_last_cache_point(blocks, |b| *b == "CP");
        assert_eq!(kept, vec!["a", "b", "c"]);
    }

    #[test]
    fn cache_control_without_ttl_produces_no_bedrock_ttl() {
        let cache_control = CacheControl {
            cache_control_type: "ephemeral".to_string(),
            ttl: None,
        };
        let cache_point = CachePointBlock::try_from(&cache_control).unwrap();
        assert!(cache_point.ttl().is_none());
    }

    #[test]
    fn cache_control_with_five_minute_ttl_maps_to_bedrock() {
        let cache_control = CacheControl {
            cache_control_type: "ephemeral".to_string(),
            ttl: Some("5m".to_string()),
        };
        let cache_point = CachePointBlock::try_from(&cache_control).unwrap();
        assert_eq!(cache_point.ttl(), Some(&CacheTtl::FiveMinutes));
    }

    #[test]
    fn cache_control_with_one_hour_ttl_maps_to_bedrock() {
        let cache_control = CacheControl {
            cache_control_type: "ephemeral".to_string(),
            ttl: Some("1h".to_string()),
        };
        let cache_point = CachePointBlock::try_from(&cache_control).unwrap();
        assert_eq!(cache_point.ttl(), Some(&CacheTtl::OneHour));
    }
}
