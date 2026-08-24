# Model backends

The agent is not tied to OpenAI. Everything downstream of the model, meaning the
planner, the worker and the verifier, holds only a LangChain `BaseChatModel`, so
which backend sits behind it is a configuration question.

There are two ways to configure one. The first covers most cases.

---

## Tier 1: an OpenAI-dialect endpoint

Configuration only. No extra package.

```bash
LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:8080/v1
LLM_MODEL=whatever-the-server-calls-it
LLM_API_KEY=not-needed
```

`LLM_PROVIDER=openai` names the wire format rather than the vendor. Any endpoint
serving `/v1/chat/completions` works, which covers most local runtimes and
gateways:

| Backend | `LLM_BASE_URL` |
|---|---|
| llama.cpp server | `http://localhost:8080/v1` |
| vLLM | `http://localhost:8000/v1` |
| LM Studio | `http://localhost:1234/v1` |
| Ollama (compatibility port) | `http://localhost:11434/v1` |
| LocalAI | `http://localhost:8080/v1` |
| LiteLLM proxy | your proxy URL |
| A company gateway | whatever they gave you |

The model string is passed through untouched, so a name this client has never
heard of is fine.

### The dialect switch

Setting `LLM_BASE_URL` also selects the wire dialect. With no override the client
uses OpenAI's Responses API. This matters because a reasoning model on
`/v1/chat/completions` rejects `reasoning_effort` alongside function tools unless
it is set to `'none'`, and the agent always registers tools, so on
chat-completions a reasoning model does not reason.

Few endpoints other than OpenAI implement `/v1/responses`. An endpoint override
is therefore treated as an indication that the backend is not OpenAI, and the
client uses chat-completions instead. Set `LLM_API=responses` or `LLM_API=chat`
to override that choice.

---

## Tier 2: a native integration

Configuration plus one install.

```bash
pip install '.[anthropic]'      # or .[google] / .[ollama] / .[bedrock]
```

```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-5
LLM_API_KEY=sk-ant-...
```

Recognised names include `anthropic`, `google_genai`, `ollama`, `azure_openai`,
`bedrock`, `litellm`, `mistralai`, `groq`, `deepseek`, `xai` and `openrouter`,
27 in total, plus any other LangChain integration referenced by its own name.
The spellings `claude`, `gemini`, `google` and `azure` are also accepted.

If the integration package is not installed, the client reports the `pip install`
command required.

Tier 1 can also reach Claude and Gemini through their compatibility endpoints.
The difference is that a native integration exposes provider-specific behaviour
that a compatibility layer does not carry, such as Anthropic's thinking blocks
and prompt caching, Gemini's thinking budget, and Ollama's `num_predict`.

---

## Settings

Every setting has an `LLM_*` name. The older `OPENAI_*` names still work as
fallbacks, so an existing `.env` needs no edit; `LLM_*` wins when both are set.

| Variable | What it does | Default |
|---|---|---|
| `LLM_PROVIDER` | Backend, or the OpenAI wire format | `openai` |
| `LLM_MODEL` | Model id, passed through untouched | `gpt-5.6-sol` |
| `LLM_API_KEY` | API key | -- |
| `LLM_BASE_URL` | Endpoint override | -- |
| `LLM_EFFORT` | `minimal` / `low` / `medium` / `high` | provider default |
| `LLM_API` | OpenAI dialect: `responses` / `chat` | auto |
| `LLM_TEMPERATURE` | Sampling temperature, non-reasoning models | `0` |
| `LLM_EXTRA` | Extra kwargs as JSON, merged last | `{}` |

`LLM_EFFORT` is one setting for every backend. LangChain treats reasoning effort
as a standard parameter and each provider translates it into its own shape, so
the same value works for an OpenAI reasoning model and an Anthropic thinking
budget.

`LLM_EXTRA` accepts any keyword argument the backend supports that this client
does not model. It is merged last and takes precedence over the defaults:

```bash
LLM_EXTRA={"num_predict": 512, "max_retries": 2}
```

A backend option this client does not know about is therefore a configuration
change rather than a code change.

The client prints the resolved backend at startup, for example
`openai:some-local-model @ http://localhost:8080/v1 [chat]`. Since the endpoint
is configurable, this line confirms which backend is actually in use.

---

## Model requirements

Two capabilities are required, and configuration cannot substitute for either.

**Tool calling.** Every build, verification and render goes through a tool, so a
model that cannot call tools reliably cannot drive the agent.

**Vision**, for the image-to-CAD workflow only. Reading a dimensioned drawing
requires a model that accepts images. Work driven from a written description
does not.

Neither can be confirmed in advance for an arbitrary endpoint. LangChain
publishes a capability profile for models it recognises, but for an unfamiliar
model id it reports the capability as unknown rather than true or false, so the
first real failure is the first reliable signal.

### How this tends to fail

The failures observed in testing were not malformed tool calls. In both cases the
calls were well formed and the model reached the tools correctly. The difficulty
was in how the tools were used.

One model repeatedly issued a skill name as an MGED command. Skill names describe
work to be done with the tools; they are not commands, and the command never
succeeded. The error returned for an unrecognised command now says so directly
and asks the model not to repeat the same name, rather than suggesting it
diagnose the failure, which previously encouraged a retry loop.

A second model built the part correctly and verified it against the raytracer on
the first attempt, then continued calling `declare_assumption` several hundred
times instead of reporting the result. The work was already complete and correct.

The second case is worth understanding, because a capable model avoiding it is
not the same as the system preventing it.

### Loop limits

The worker's tool loop is bounded, which it previously was not. `create_agent`
has no maximum-iteration setting, and the graph's `recursion_limit` counts outer
supersteps, so a loop inside the worker never reached it.

| Limit | Value | On reaching it |
|---|---|---|
| Model calls per worker run | 50 | The loop ends and the run finishes normally |
| `declare_assumption` calls per run | 30 | Further calls are refused with a message asking the model to stop; the run continues |

Both are well above normal use. A successful run of the same case completes in
under ten model calls. In the case above the limit changed the outcome from an
unbounded run to a pass, because the geometry had been correct from the start.

If a model cannot meet the requirements above, the deterministic tool path
remains available and requires no model at all:

```bash
./evals/run.sh          # builds and raytrace-checks every case, no API key
```

---

## Notes

The `--v1` client (`brlcad-mcp chat --v1`) predates this work and is retained for
comparison against the current client. It connects to OpenAI directly, although
it does read `OPENAI_API_BASE` from the environment. Use the default client for
any other backend.
