use std::{collections::HashMap, sync::Arc};

use ai00_core::{
    FinishReason, GenerateRequest, InputState, ThreadRequest, Token, TokenCounter, MAX_TOKENS,
};
use derivative::Derivative;
use futures_util::StreamExt;
use itertools::Itertools;
use regex::Regex;
use salvo::{oapi::extract::JsonBody, prelude::*, sse::SseEvent, Depot, Writer};
use serde::{Deserialize, Serialize};

use super::*;
use crate::{
    api::request_info,
    types::{Array, ThreadSender},
    SLEEP,
};

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, ToSchema)]
enum Role {
    #[default]
    #[serde(alias = "system")]
    System,
    #[serde(alias = "user")]
    User,
    #[serde(alias = "assistant")]
    Assistant,
    #[serde(alias = "observation", alias = "tool", alias = "Tool")]
    Observation,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::System => write!(f, "System"),
            Role::User => write!(f, "User"),
            Role::Assistant => write!(f, "Assistant"),
            Role::Observation => write!(f, "Observation"),
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, ToSchema)]
struct ChatRecord {
    role: Role,
    content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
struct ChatTemplate {
    record: String,
    prefix: String,
    sep: String,
}

impl Default for ChatTemplate {
    fn default() -> Self {
        Self {
            record: "{role}: {content}".into(),
            prefix: "{assistant}:".into(),
            sep: "\n\n".into(),
        }
    }
}

#[derive(Debug, Derivative, Deserialize, ToSchema)]
#[derivative(Default)]
#[serde(default)]
#[salvo(schema(
    example = json!({
        "messages": [
            {
                "role": "user",
                "content": "Hi!"
            },
            {
                "role": "assistant",
                "content": "Hello, I am your AI assistant. If you have any questions or instructions, please let me know!"
            },
            {
                "role": "user",
                "content": "Tell me about water."
            }
        ],
        "names": {
            "user": "User",
            "assistant": "Assistant"
        },
        "template": {
            "record": "{role}: {content}",
            "prefix": "{assistant}:",
            "sep": "\n\n"
        },
        "stop": [
            "\n\nUser:"
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
struct ChatRequest {
    messages: Array<ChatRecord>,
    names: HashMap<Role, String>,
    template: ChatTemplate,
    state: InputState,
    #[derivative(Default(value = "256"))]
    max_tokens: usize,
    #[derivative(Default(value = "ChatRequest::default_stop_words()"))]
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

impl ChatRequest {
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

impl From<ChatRequest> for GenerateRequest {
    fn from(value: ChatRequest) -> Self {
        let ChatRequest {
            messages,
            names,
            template,
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

        let sep = template.sep;
        let re = Regex::new(r"\n(\s*\n)+").unwrap();
        let prompt = Vec::from(messages.clone())
            .into_iter()
            .map(|ChatRecord { role, content }| {
                let role = names.get(&role).cloned().unwrap_or(role.to_string());
                let content = re.replace_all(&content, "\n");
                let content = content.trim();
                template
                    .record
                    .replace("{role}", &role)
                    .replace("{content}", content)
            })
            .join(&sep);
        let model_text = Vec::from(messages)
            .into_iter()
            .filter(|record| record.role == Role::Assistant)
            .map(|record| record.content)
            .join(&sep);

        let assistant = names
            .get(&Role::Assistant)
            .cloned()
            .unwrap_or(Role::Assistant.to_string());
        let user = names
            .get(&Role::User)
            .cloned()
            .unwrap_or(Role::User.to_string());
        let prefix = template
            .prefix
            .replace("{assistant}", &assistant)
            .replace("{user}", &user);
        let prompt = format!("{prompt}{sep}{prefix}");

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
            model_text,
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
struct ChatChoice {
    message: ChatRecord,
    index: usize,
    finish_reason: FinishReason,
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
#[salvo(schema(
    example = json!({
        "object": "chat.completion",
        "model": "assets/models\\RWKV-x060-World-3B-v2.1-20240417-ctx4096.st",
        "choices": [
            {
                "message": {
                    "role": "Assistant",
                    "content": "Water is a chemical compound made up of two hydrogen atoms and one oxygen atom, bonded together with a shared electron pair. It is the most abundant substance in the Earth's atmosphere and oceans, and is essential for life on Earth. Water can exist in many different forms, including liquid water, ice, and solid ice. It is also an important solvent for many chemical reactions and processes. Water is found in all parts of the world and is essential for life on Earth."
                },
                "index": 0,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt": 41,
            "completion": 97,
            "total": 138,
            "duration": {
                "secs": 8,
                "nanos": 381235000
            }
        }
    })
))]
struct ChatResponse {
    object: String,
    model: String,
    choices: Vec<ChatChoice>,
    #[serde(rename = "usage")]
    counter: TokenCounter,
}

#[derive(Debug, Derivative, Serialize, ToSchema, ToResponse)]
#[derivative(Default)]
#[serde(rename_all = "snake_case")]
enum PartialChatRecord {
    Role(Role),
    Content(String),
    #[derivative(Default)]
    #[serde(untagged)]
    None(HashMap<String, String>),
}

#[derive(Debug, Default, Serialize, ToSchema, ToResponse)]
struct PartialChatChoice {
    delta: PartialChatRecord,
    index: usize,
    finish_reason: FinishReason,
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
#[salvo(schema(
    example = json!({
        "object": "chat.completion.chunk",
        "model": "assets/models\\RWKV-x060-World-3B-v2.1-20240417-ctx4096.st",
        "choices": [
            {
                "delta": {
                    "role": "Assistant"
                },
                "index": 0,
                "finish_reason": null
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 186,
            "total_tokens": 201
        }
    })
))]
struct PartialChatResponse {
    object: String,
    model: String,
    choices: Vec<PartialChatChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<HashMap<String, usize>>, // 新增字段
}

async fn respond_one(depot: &mut Depot, request: ChatRequest, res: &mut Response) {
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
                // // 检查是否生成的 token 为 "0"
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

    let json = Json(ChatResponse {
        object: "chat.completion".into(),
        model: model_name,
        choices: vec![ChatChoice {
            message: ChatRecord {
                role: Role::Assistant,
                content: text.trim().into(),
            },
            index: 0,
            finish_reason,
        }],
        counter: token_counter,
    });
    res.render(json);
}

async fn respond_stream(depot: &mut Depot, request: ChatRequest, res: &mut Response) {
    let sender = depot.obtain::<ThreadSender>().unwrap();
    let info = request_info(sender.clone(), SLEEP).await;
    let model_name = info.reload.model_path.to_string_lossy().into_owned();

    let (token_sender, token_receiver) = flume::unbounded();
    let request = Box::new(request.into());
    println!("request field: {:?}", request);
    let _ = sender.send(ThreadRequest::Generate {
        request,
        tokenizer: info.tokenizer,
        sender: token_sender,
    });

    let mut token_counter = TokenCounter::default();
    let mut start_token = true;

    let stream = token_receiver.into_stream().map(move |token| {
        let choice = match token {
            Token::Start => PartialChatChoice {
                delta: PartialChatRecord::Role(Role::Assistant),
                ..Default::default()
            },
            Token::Content(token) => {
                let token = match start_token {
                    true => token.trim_start().into(),
                    false => token,
                };
                start_token = false;
                PartialChatChoice {
                    delta: PartialChatRecord::Content(token),
                    ..Default::default()
                }
            }
            Token::Stop(reason, counter) => {
                token_counter = counter; // 更新 token 统计
                PartialChatChoice {
                    finish_reason: reason,
                    ..Default::default()
                }
            }
            Token::Done => {
                // 在 [DONE] 之前发送包含 usage 的 JSON 数据
                let usage_response = PartialChatResponse {
                    object: "chat.completion.chunk".into(),
                    model: model_name.clone(),
                    choices: vec![], // 没有 choice 数据
                    usage: Some(HashMap::from([
                        ("prompt_tokens".to_string(), token_counter.prompt),
                        ("completion_tokens".to_string(), token_counter.completion),
                        ("total_tokens".to_string(), token_counter.total),
                    ])),
                };
                let usage_json = serde_json::to_string(&usage_response).unwrap();
                return Ok(SseEvent::default().text(usage_json));
            }
            _ => unreachable!(),
        };

        match serde_json::to_string(&PartialChatResponse {
            object: "chat.completion.chunk".into(),
            model: model_name.clone(),
            choices: vec![choice],
            usage: None, // 如果不需要使用数据
        }) {
            Ok(json_text) => Ok(SseEvent::default().text(json_text)),
            Err(err) => Err(err),
        }
    });

    salvo::sse::stream(res, stream);
}

/// Generate chat completions with context.
#[endpoint(
    responses(
        (status_code = 200, description = "Generate one response if `stream` is false.", body = ChatResponse),
        (status_code = 201, description = "Generate SSE response if `stream` is true.", body = PartialChatResponse)
    )
)]
pub async fn chat_completions(depot: &mut Depot, req: JsonBody<ChatRequest>, res: &mut Response) {
    let mut request = req.0;
    // println!("Stop field: {:?}", request.stop);
    request.merge_stop_words();
    // println!("Stop field: {:?}", request.stop);
    match request.stream {
        true => respond_stream(depot, request, res).await,
        false => respond_one(depot, request, res).await,
    }
}
