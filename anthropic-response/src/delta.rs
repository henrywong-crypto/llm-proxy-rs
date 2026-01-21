use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum Delta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },

    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },

    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },

    #[serde(rename = "signature_delta")]
    SignatureDelta { signature: String },

    #[serde(rename = "citations_delta")]
    CitationsDelta { citation: Citation },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum Citation {
    #[serde(rename = "page_location")]
    PageLocation {
        cited_text: String,
        document_index: i32,
        document_title: String,
        start_page_number: i32,
        end_page_number: i32,
    },
    #[serde(rename = "char_location")]
    CharLocation {
        cited_text: String,
        document_index: i32,
        document_title: String,
        start_char_index: i32,
        end_char_index: i32,
    },
    #[serde(rename = "web_search_result_location")]
    WebSearchResultLocation {
        cited_text: String,
        url: String,
        title: String,
        encrypted_index: String,
    },
}
