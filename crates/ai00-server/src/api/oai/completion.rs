use std::{collections::HashMap, sync::Arc};

use ai00_core::{
    FinishReason, GenerateRequest, InputState, ThreadRequest, Token, TokenCounter, MAX_TOKENS,
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
        "sampler": {
            "type": "Nucleus",
            "top_p": 0.5,
            "top_k": 128,
            "temperature": 1,
            "presence_penalty": 0.3,
            "frequency_penalty": 0.3,
            "penalty_decay": 0.99654026
        },
        "state": "00000000-0000-0000-0000-000000000000"
    })
))]
struct CompletionRequest {
    prompt: Array<String>,
    state: InputState,
    #[derivative(Default(value = "256"))]
    max_tokens: usize,
    #[derivative(Default(value = "CompletionRequest::default_stop_words()"))]
    stop: Array<String>,
    stream: bool,
    #[serde(alias = "logit_bias")]
    bias: HashMap<u16, f32>,
    bnf_schema: Option<String>,
    #[serde(alias = "sampler_override")]
    sampler: Option<SamplerParams>,
    #[derivative(Default(value = "0.5"))]
    top_p: f32,
    #[derivative(Default(value = "128"))]
    top_k: usize,
    #[derivative(Default(value = "1.0"))]
    temperature: f32,
}

impl CompletionRequest {
    // 默认的 stop words
    pub fn default_stop_words() -> Array<String> {
        Array::Vec(vec![
            "\n\nUser".to_string(),
            "\n\nQuestion".to_string(),
            "\n\nQ".to_string(),
            "\n\nHuman".to_string(),
            "\n\nBob".to_string(),
            "\n\nAssistant".to_string(),
            "\n\nAnswer".to_string(),
            "\n\nA".to_string(),
            "\n\nBot".to_string(),
            "\n\nAlice".to_string(),
            "\n\nObservation".to_string(),
            "\n\nSystem".to_string(),
            "\n\n\nSystem".to_string(),
            "\n\n\nAssistant".to_string(),
            "\n\n\nUser".to_string(),
            "\n\n\n\nSystem".to_string(),
            "\n\n\n\nAssistant".to_string(),
            "\n\n\n\nUser".to_string(),
            "\n\n\n\n".to_string(),
            "ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ".to_string(),
        ])
    }

    // 合并默认 stop words 与请求中的 stop words，并去重
    pub fn merge_stop_words(&mut self) {
        let default_stops: Vec<String> = Self::default_stop_words().into();  // 利用 From trait 实现的转换
        let mut current_stops: Vec<String> = self.stop.clone().into();        // 同样利用 From trait 进行转换

        // 合并两个 Vec<String>
        current_stops.extend(default_stops);

        // 去重，确保列表中没有重复的停止词
        current_stops.sort();
        current_stops.dedup();

        // 将合并后且去重的 Vec<String> 转换回 Array<String>
        self.stop = Array::Vec(current_stops);
    }
}

impl From<CompletionRequest> for GenerateRequest {
    fn from(value: CompletionRequest) -> Self {
        let CompletionRequest {
            prompt,
            state,
            max_tokens,
            stop,
            sampler,
            top_p,
            top_k,
            temperature,
            bias,
            bnf_schema,
            ..
        } = value;

        let prompt = Vec::from(prompt).join("");
        let max_tokens = max_tokens.min(MAX_TOKENS);
        let stop = stop.into();
        let bias = Arc::new(bias);
        let sampler = match sampler {
            Some(sampler) => sampler.into(),
            None => SamplerParams::Nucleus(NucleusParams {
                top_p,
                top_k,
                temperature,
                ..Default::default()
            })
            .into(),
        };
        let state = state.into();

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
struct CompletionChoice {
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
struct CompletionResponse {
    object: String,
    model: String,
    choices: Vec<CompletionChoice>,
    #[serde(rename = "usage")]
    counter: TokenCounter,
}

#[derive(Debug, Derivative, Serialize, ToSchema, ToResponse)]
#[derivative(Default)]
#[serde(rename_all = "snake_case")]
enum PartialCompletionRecord {
    Content(String),
    #[derivative(Default)]
    #[serde(untagged)]
    None(HashMap<String, String>),
}

#[derive(Debug, Default, Serialize, ToSchema, ToResponse)]
struct PartialCompletionChoice {
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
struct PartialCompletionResponse {
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
                // // 如果生成的 token 是 "0"，则停止生成
                // if token == "0" {
                //     finish_reason = FinishReason::Stop; // 设置停止原因
                //     break;
                // }
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
            // Token::Content(token) => {
            //     // 如果生成的 token 为 "0"，则立即停止生成
            //     if token == "0" {
            //         return Ok(SseEvent::default().text("[DONE]"));
            //     }
            //     PartialCompletionChoice {
            //         delta: PartialCompletionRecord::Content(token),
            //         ..Default::default()
            //     }
            // }
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
    let mut request = req.0;
    request.merge_stop_words();
    match request.stream {
        true => respond_stream(depot, request, res).await,
        false => respond_one(depot, request, res).await,
    }
}
