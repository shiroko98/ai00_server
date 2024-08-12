use std::{collections::HashMap, sync::Arc};

use ai00_core::{
    run::StateId, FinishReason, GenerateRequest, ThreadRequest, Token, TokenCounter, MAX_TOKENS,
};
use derivative::Derivative;
use futures_util::StreamExt;
use salvo::{
    oapi::{extract::JsonBody, ToResponse, ToSchema},
    prelude::*,
    sse::SseEvent,
    Depot, Writer,
};
use serde::{Deserialize, Serialize};

use super::*;
use crate::{
    api::request_info,
    types::{Array, ThreadSender},
    SLEEP,
};

#[derive(Debug, Derivative, Deserialize, ToSchema)]
#[derivative(Default)]
#[serde(default)]
#[salvo(schema(
    example = json!({
        "prompt": [
            "The Eiffel Tower is located in the city of"
        ],
        "stop": [
            "\n\n",
            "."
        ],
        "stream": false,
        "max_tokens": 1000,
        "sampler_override": {
            "type": "Nucleus",
            "top_p": 0.5,
            "top_k": 128,
            "temperature": 1,
            "presence_penalty": 0.3,
            "frequency_penalty": 0.3,
            "penalty": 400,
            "penalty_decay": 0.99654026
        },
        "state": "00000000-0000-0000-0000-000000000000"
    })
))]
pub struct CompletionRequest {
    prompt: Array<String>,
    state: StateId,
    #[derivative(Default(value = "256"))]
    max_tokens: usize,
    #[derivative(Default(value = "Array::Vec(vec![\
    \"\\n\\nUser\".to_string(), \
    \"\\n\\nQuestion\".to_string(), \
    \"\\n\\nQ\".to_string(), \
    \"\\n\\nHuman\".to_string(), \
    \"\\n\\nBob\".to_string(), \
    \"\\n\\nAssistant\".to_string(), \
    \"\\n\\nAnswer\".to_string(), \
    \"\\n\\nA\".to_string(), \
    \"\\n\\nBot\".to_string(), \
    \"\\n\\nAlice\".to_string(), \
    \"\\n\\nObservation\".to_string(), \
    \"\\n\\nSystem\".to_string(), \
    \"\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\".to_string()\
    ])"))]
    stop: Array<String>,
    stream: bool,
    #[serde(alias = "logit_bias")]
    bias: HashMap<u16, f32>,
    bnf_schema: Option<String>,
    sampler: NucleusParams,
    sampler_override: Option<SamplerParams>,
}

impl From<CompletionRequest> for GenerateRequest {
    fn from(value: CompletionRequest) -> Self {
        let CompletionRequest {
            prompt,
            state,
            max_tokens,
            stop,
            sampler,
            sampler_override,
            bias,
            bnf_schema,
            ..
        } = value;

        let prompt = Vec::from(prompt).join("");
        let max_tokens = max_tokens.min(MAX_TOKENS);
        let stop = stop.into();
        let bias = Arc::new(bias);
        let sampler = match sampler_override {
            Some(sampler) => sampler.into(),
            None => SamplerParams::Nucleus(sampler).into(),
        };

        Self {
            prompt,
            max_tokens,
            stop,
            sampler,
            bias,
            bnf_schema,
            state,
            ..Default::default()
        }
    }
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
pub struct CompletionChoice {
    text: String,
    index: usize,
    finish_reason: FinishReason,
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
#[salvo(schema(
    example = json!({
        "object": "text_completion",
        "model": "assets/models\\RWKV-x060-World-3B-v2.1-20240417-ctx4096.st",
        "choices": [
            {
                "text": " Paris, France",
                "index": 0,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt": 11,
            "completion": 4,
            "total": 15,
            "duration": {
                "secs": 0,
                "nanos": 260801800
            }
        }
    })
))]
pub struct CompletionResponse {
    object: String,
    model: String,
    choices: Vec<CompletionChoice>,
    #[serde(rename = "usage")]
    counter: TokenCounter,
}

#[derive(Debug, Derivative, Serialize, ToSchema, ToResponse)]
#[derivative(Default)]
#[serde(rename_all = "snake_case")]
pub enum PartialCompletionRecord {
    Content(String),
    #[derivative(Default)]
    #[serde(untagged)]
    None(HashMap<String, String>),
}

#[derive(Debug, Default, Serialize, ToSchema, ToResponse)]
pub struct PartialCompletionChoice {
    delta: PartialCompletionRecord,
    index: usize,
    finish_reason: FinishReason,
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
#[salvo(schema(
    example = json!({
        "object": "text_completion.chunk",
        "model": "assets/models\\RWKV-x060-World-3B-v2.1-20240417-ctx4096.st",
        "choices": [
            {
                "delta": {
                    "content": " Paris"
                },
                "index": 0,
                "finish_reason": null
            }
        ]
    })
))]
pub struct PartialCompletionResponse {
    object: String,
    model: String,
    choices: Vec<PartialCompletionChoice>,
}

async fn respond_one(depot: &mut Depot, request: CompletionRequest, res: &mut Response) {
    let sender = depot.obtain::<ThreadSender>().unwrap();
    let info = request_info(sender.clone(), SLEEP).await;
    let model_name = info.reload.model_path.to_string_lossy().into_owned();

    let (token_sender, token_receiver) = flume::unbounded();
    let request = Box::new(request.into());
    let _ = sender.send(ThreadRequest::Generate {
        request,
        tokenizer: info.tokenizer,
        sender: token_sender,
    });

    let mut token_counter = TokenCounter::default();
    let mut finish_reason = FinishReason::Null;
    let mut text = String::new();
    let mut stream = token_receiver.into_stream();

    while let Some(token) = stream.next().await {
        match token {
            Token::Start => {}
            Token::Content(token) => {
                text += &token;
            }
            Token::Stop(reason, counter) => {
                finish_reason = reason;
                token_counter = counter;
                break;
            }
            _ => unreachable!(),
        }
    }

    let json = Json(CompletionResponse {
        object: "text_completion".into(),
        model: model_name,
        choices: vec![CompletionChoice {
            text,
            index: 0,
            finish_reason,
        }],
        counter: token_counter,
    });
    res.render(json);
}

async fn respond_stream(depot: &mut Depot, request: CompletionRequest, res: &mut Response) {
    let sender = depot.obtain::<ThreadSender>().unwrap();
    let info = request_info(sender.clone(), SLEEP).await;
    let model_name = info.reload.model_path.to_string_lossy().into_owned();

    let (token_sender, token_receiver) = flume::unbounded();
    let request = Box::new(request.into());
    let _ = sender.send(ThreadRequest::Generate {
        request,
        tokenizer: info.tokenizer,
        sender: token_sender,
    });

    let stream = token_receiver.into_stream().skip(1).map(move |token| {
        let choice = match token {
            Token::Content(token) => PartialCompletionChoice {
                delta: PartialCompletionRecord::Content(token),
                ..Default::default()
            },
            Token::Stop(finish_reason, _) => PartialCompletionChoice {
                finish_reason,
                ..Default::default()
            },
            Token::Done => return Ok(SseEvent::default().text("[DONE]")),
            _ => unreachable!(),
        };

        match serde_json::to_string(&PartialCompletionResponse {
            object: "text_completion.chunk".into(),
            model: model_name.clone(),
            choices: vec![choice],
        }) {
            Ok(json_text) => Ok(SseEvent::default().text(json_text)),
            Err(err) => Err(err),
        }
    });
    salvo::sse::stream(res, stream);
}

/// Generate completions for the given text.
#[endpoint(
    responses(
        (status_code = 200, description = "Generate one response if `stream` is false.", body = CompletionResponse),
        (status_code = 201, description = "Generate SSE response if `stream` is true", body = PartialCompletionResponse)
    )
)]
pub async fn completions(depot: &mut Depot, req: JsonBody<CompletionRequest>, res: &mut Response) {
    let request = req.0;
    match request.stream {
        true => respond_stream(depot, request, res).await,
        false => respond_one(depot, request, res).await,
    }
}
