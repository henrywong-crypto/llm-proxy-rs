use aws_config::BehaviorVersion;
use aws_sdk_bedrock::types::{InferenceProfileModelSource, Tag};
use aws_sdk_costexplorer::types::{
    DateInterval, Dimension, DimensionValues, Expression, Granularity, GroupDefinition,
    GroupDefinitionType,
};
use tracing::info;

/// Creates an application inference profile via the Bedrock `CreateInferenceProfile` API.
///
/// Equivalent to:
/// ```sh
/// aws bedrock create-inference-profile \
///   --inference-profile-name <name> \
///   --model-source copyFrom=<model-source-arn> \
///   --tags key=<key>,value=<value>
/// ```
pub async fn create_inference_profile(
    name: &str,
    model_source_arn: &str,
    tags: &[(String, String)],
) -> anyhow::Result<String> {
    let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
    let client = aws_sdk_bedrock::Client::new(&config);

    let mut builder = client
        .create_inference_profile()
        .inference_profile_name(name)
        .model_source(InferenceProfileModelSource::CopyFrom(
            model_source_arn.to_string(),
        ));

    for (key, value) in tags {
        let tag = Tag::builder()
            .key(key)
            .value(value)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build tag: {}", e))?;
        builder = builder.tags(tag);
    }

    info!(
        "Creating inference profile '{}' from source '{}'",
        name, model_source_arn
    );

    let response = builder
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("CreateInferenceProfile API error: {}", e))?;

    let arn = response.inference_profile_arn().to_string();
    info!("Created inference profile: {}", arn);

    Ok(arn)
}

/// Queries AWS Cost Explorer for Bedrock costs grouped by inference profile ARN.
///
/// `profile_arns` — the inference profile ARNs to query costs for.
/// `start` / `end` — date range in `YYYY-MM-DD` format (start inclusive, end exclusive).
///
/// Returns a vec of `(profile_arn, cost_amount, cost_unit)` for each profile that had usage.
pub async fn get_inference_profile_costs(
    profile_arns: &[String],
    start: &str,
    end: &str,
) -> anyhow::Result<Vec<(String, String, String)>> {
    let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
    let client = aws_sdk_costexplorer::Client::new(&config);

    let time_period = DateInterval::builder()
        .start(start)
        .end(end)
        .build()
        .map_err(|e| anyhow::anyhow!("Invalid date interval: {}", e))?;

    let service_filter = Expression::builder()
        .dimensions(
            DimensionValues::builder()
                .key(Dimension::Service)
                .values("Amazon Bedrock")
                .build(),
        )
        .build();

    let mut resource_values = DimensionValues::builder().key(Dimension::ResourceId);
    for arn in profile_arns {
        resource_values = resource_values.values(arn);
    }
    let resource_filter = Expression::builder()
        .dimensions(resource_values.build())
        .build();

    let filter = Expression::builder()
        .set_and(Some(vec![service_filter, resource_filter]))
        .build();

    let group_by = GroupDefinition::builder()
        .r#type(GroupDefinitionType::Dimension)
        .key("RESOURCE_ID")
        .build();

    info!(
        "Querying Cost Explorer for {} inference profiles from {} to {}",
        profile_arns.len(),
        start,
        end
    );

    let response = client
        .get_cost_and_usage()
        .time_period(time_period)
        .granularity(Granularity::Monthly)
        .filter(filter)
        .metrics("BlendedCost")
        .group_by(group_by)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("Cost Explorer API error: {}", e))?;

    let mut results = Vec::new();

    for result_by_time in response.results_by_time() {
        for group in result_by_time.groups() {
            let keys = group.keys();
            let resource_id = keys.first().map(|s| s.as_str()).unwrap_or("unknown");

            if let Some(metrics) = group.metrics() {
                if let Some(cost) = metrics.get("BlendedCost") {
                    let amount = cost.amount().unwrap_or("0");
                    let unit = cost.unit().unwrap_or("USD");
                    info!("Profile {}: {} {}", resource_id, amount, unit);
                    results.push((
                        resource_id.to_string(),
                        amount.to_string(),
                        unit.to_string(),
                    ));
                }
            }
        }
    }

    Ok(results)
}
