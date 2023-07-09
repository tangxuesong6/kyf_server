use std::error::Error;
use std::time::{Duration, Instant};

use anyhow::Result;
use async_openai::{
    Client,
    types::{ChatCompletionRequestMessageArgs, CreateChatCompletionRequestArgs, Role},
};
use async_openai::config::OpenAIConfig;
use async_openai::types::ChatCompletionRequestMessage;
use axum::{Json, Router};
use axum::routing::post;
use clap::Parser;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::signal;
use tokio::sync::OnceCell;
use tracing::{debug, error, Level, trace};

static API_KEY: OnceCell<String> = OnceCell::const_new();

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let collector = tracing_subscriber::fmt().with_max_level(Level::TRACE).with_env_filter("openai_api_server").finish();
    // let collector = tracing_subscriber::fmt().with_max_level(Level::TRACE).finish();
    tracing::subscriber::set_global_default(collector).expect("Unable to set a global collector");

    debug!("start");
    let args = Args::parse();
    let port = args.port;
    debug!("port: {}", port);

    match args.api_key {
        None => {}
        Some(key) => {
            API_KEY.get_or_init(|| async {
                debug!("api_key: {}", key);
                key
            }).await;
        }
    }

    tokio::spawn(async move {
        start_server(port).await.unwrap();
    });

    match signal::ctrl_c().await {
        Ok(()) => {
            trace!("{}", "shut down")
        }
        Err(err) => {
            error!("Unable to listen for shutdown signal: {}", err);
        }
    }
    Ok(())
}

async fn start_server(port: u16) -> Result<()> {
    let app = Router::new().route("/chat", post(openai_handler));

    axum::Server::try_bind(&format!("0.0.0.0:{}", port).parse()?)?
        .serve(app.into_make_service())
        .await?;
    Ok(())
}


// {
// "api_key":"sk-xxxxxxxxx",
// "max_tokens":1024,
// "contents":[
// {
// "role":"user",
// "content":"红花油的味道太刺鼻怎么办"
// }
// ]
// }
#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct Content {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct UserInput {
    #[serde(default = "default_api_key")]
    pub api_key: String,
    pub max_tokens: u16,
    pub contents: Vec<Content>,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct OpenAiResponse {
    message: String,
    code: u16,
}

fn default_api_key() -> String {
    String::new()
}

async fn openai_handler(Json(input): Json<UserInput>) -> Json<Value> {
    let max_tokens = input.max_tokens;
    let contents = input.contents;


    let api_key: String;
    if input.api_key.is_empty() {
        match API_KEY.get() {
            None => {
                return Json(json!({
                "message":"api_key is empty",
                "code":500u16,
            }));
            }
            Some(key) => {
                api_key = key.to_string()
            }
        }
    } else {
        api_key = input.api_key;
    }

    let mut request_messages: Vec<ChatCompletionRequestMessage> = Vec::new();
    for content in contents {
        let request_message = ChatCompletionRequestMessageArgs::default()
            .role(get_role(content.role))
            .content(content.content)
            .build().unwrap();
        request_messages.push(request_message);
    }

    let cfg = OpenAIConfig::default().with_api_key(api_key);
    let client = Client::with_config(cfg);

    let request = match CreateChatCompletionRequestArgs::default()
        .max_tokens(max_tokens)
        .model("gpt-3.5-turbo")
        .messages(request_messages)
        .build() {
        Ok(s) => { s }
        Err(err) => {
            return Json(json!({
                "message":format!("{}",err),
                "code":500u16,
            }));
        }
    };
    let start_time = Instant::now();
    let response = match client.chat().create(request).await {
        Ok(s) => { s }
        Err(err) => {
            return Json(json!({
                "message":format!("{}",err),
                "code":500u16,
            }));
        }
    };
    if response.choices.is_empty() {
        return Json(json!({
            "message":"no choices",
            "code":500u16,
        }));
    }
    let resp = match &response.choices[0].message.content {
        None => {
            return Json(json!({
                "message":"no content",
                "code":500u16,
            }));
        }
        Some(s) => { s }
    };
    let end_time = Instant::now();
    let duration = time_diff(start_time, end_time);
    debug!("duration: {:?}", duration.as_millis());
    Json(json!({
       "message":resp,
       "code":200u16,
   }))
}

fn get_role(role: String) -> Role {
    match role.as_str() {
        "user" => { Role::User }
        "assistant" => { Role::Assistant }
        "system" => { Role::System }
        _ => { Role::User }
    }
}

fn time_diff(start_time: Instant, end_time: Instant) -> Duration {
    end_time.duration_since(start_time)
}

#[derive(Parser, Debug)]
#[command(long_about = None)]
#[command(name = "clap_demo")]
#[command(author = "song")]
#[command(version = "0.1.0")]
#[command(about = "openai_demo about")]
struct Args {
    #[arg(short, long, default_value = None)]
    api_key: Option<String>,
    #[arg(short, long, default_value_t = 10802)]
    port: u16,

}