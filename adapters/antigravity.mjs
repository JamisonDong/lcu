// Antigravity relay: a stdio MCP server for Google Antigravity (CLI, App, IDE) in front of the original runtime,
// through the shared client. Antigravity writes MCP initialization instructions to disk instead of injecting them
// into the model context, rejects eager tools whose input schema has a top-level `oneOf` (which conflicts with
// Antigravity's injected `toolSummary` field), and Antigravity CLI 1.5.11 cancels `elicitation/create` requests
// immediately even though it advertises elicitation support. Antigravity's own command hooks (`antigravity.mjs
// --hook`, installed by `lcu setup --agent antigravity`) reach this relay over a private Unix socket named for the
// Antigravity process that runs both: they deliver the original instructions once per conversation through
// `injectSteps` and end turns on `Stop` (while preserving concurrent conversations and subagents using Antigravity's
// per-call `_meta["antigravity.google/conversation_id"]`). Native-app approvals are forwarded as a form when the
// client advertises form elicitation; when the client cancels or cannot show the prompt, the call result tells the
// model to run `lcu apps allow <App>` itself (Touch ID or password, lasting until revoked) and to offer `lcu apps
// revoke <App>` when its task is done.
import { execFileSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { chmodSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync } from 'node:fs';
import { createConnection, createServer } from 'node:net';
import { homedir, tmpdir } from 'node:os';
import { basename, isAbsolute, join } from 'node:path';
import { StringDecoder } from 'node:string_decoder';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import { persistAudioContent } from './audio-files.mjs';
import {
  APPROVAL_TIMEOUT_MS,
  createCuaClient,
  isMainModule,
  MODEL_TOOLS,
  nativeAppApprovalOptions,
  nativeAppApprovalResponse,
} from './client.mjs';

/** The Antigravity hook events LCU registers, with the default LCU turn end each one means. */
export const HOOK_EVENTS = {
  SessionStart: null,
  PreInvocation: null,
  Stop: 'Stop',
};
/** Seconds Antigravity gives each hook (`timeout` in hooks.json); the hook gives up on the relay a little earlier. */
export const HOOK_TIMEOUTS = { SessionStart: 10, PreInvocation: 30, Stop: 60 };
/** Once Antigravity is gone, open turns get this long to end upstream before the original server is closed anyway. */
export const SHUTDOWN_DRAIN_TIMEOUT_MS = 2_000;
const REQUEST_LIMIT = 64 * 1024;
/** The line above the original instructions when a call result carries them (hooks could not reach the relay). */
export const INSTRUCTIONS_HEADER = 'LCU: Computer Use instructions for this session:';
/** Per-call metadata key Antigravity attaches to `tools/call` naming the active conversation or subagent. */
export const CONVERSATION_ID_META = 'antigravity.google/conversation_id';
/** Ancestors of the hook checked for a relay: `sh -c`, then Antigravity (plus wrappers). */
const HOOK_ANCESTORS = 4;
const TOP_LEVEL_COMBINATORS = ['oneOf', 'anyOf'];

function asError(error) {
  return error instanceof Error ? error.message : String(error);
}

/** Longest socket directory: a socket path must fit in 104 bytes on macOS, and its name takes up to 24. */
const MAX_DIRECTORY_BYTES = 78;

/**
 * The private directory of relay sockets: `LCU_ANTIGRAVITY_CONTROL_DIR` (tests), else beside LCU's other state,
 * or in the temporary directory when that path is too long for a socket. The relay and its hooks share
 * Antigravity's environment, so both choose the same one.
 */
export function controlDirectory({
  env = process.env,
  home = env.HOME || homedir(),
  temporary = env.TMPDIR || tmpdir(),
  uid = process.getuid?.(),
} = {}) {
  if (env.LCU_ANTIGRAVITY_CONTROL_DIR) return env.LCU_ANTIGRAVITY_CONTROL_DIR;
  const state = join(env.XDG_STATE_HOME || join(home, '.local', 'state'), 'lcu', 'antigravity');
  return Buffer.byteLength(state) <= MAX_DIRECTORY_BYTES ? state : join(temporary, `lcu-antigravity-${uid}`);
}

// `lcu apps allow` waits this long for Touch ID or the password (AUTH_TIMEOUT in lcu/apps.mjs).
const AUTH_WAIT_SECONDS = 300;
const shellWord = (word) => (/^[A-Za-z0-9._\/-]+$/.test(word) ? word : `"${word.replace(/["\\$`]/g, '\\$&')}"`);

/** The absolute `lcu` launcher among the relay's original command (setup passes it), else plain `lcu`. */
export function launcherOf(command = []) {
  return [...command].reverse().find(word => typeof word === 'string' && isAbsolute(word) && basename(word) === 'lcu') ?? 'lcu';
}

/** The tool-result text that replaces an approval prompt Antigravity did not show. `lcu` is the launcher to name. */
export function approvalHint(app, lcu = 'lcu') {
  const run = (verb) => `\`${shellWord(lcu)} apps ${verb} ${shellWord(app)}\``;
  return `LCU: Computer Use needs the person's approval to control ${app}, and Antigravity could not show the ` +
    `approval prompt. Run ${run('allow')} yourself in a terminal: the person approves with Touch ID or their ` +
    `password, which allows ${app} until it is revoked. It waits up to 5 minutes for them, so give the command a ` +
    `timeout of at least ${AUTH_WAIT_SECONDS} seconds. Then retry the call. If the command fails, is refused or ` +
    'times out (for example, nobody is there to approve), ask the person to run it themselves and stop. In your ' +
    `final answer, say that ${app} stays allowed until revoked, show ${run('revoke')}, and offer to run it for them.`;
}

/**
 * The descriptor Antigravity gets for an original public tool. Antigravity injects a required `toolSummary` field
 * into top-level eager MCP tool schemas; when the original `js` lists its argument forms (`code`, or
 * `completion_summary` alone) under a top-level `oneOf` with `additionalProperties: false`, validation rejects
 * every call with or without `toolSummary`. Merging top-level `oneOf`/`anyOf` into one object with the union of
 * branch properties and intersection of required keys matches the Claude relay's adaptation; calls still reach the
 * original server unchanged, and it validates them against its own schema.
 */
export function antigravityToolDescriptor(tool) {
  const schema = tool.inputSchema;
  const combinators = TOP_LEVEL_COMBINATORS.filter(key => Array.isArray(schema?.[key]));
  if (combinators.length === 0) return tool;
  const inputSchema = { ...schema, type: 'object', properties: { ...schema.properties } };
  const required = new Set(Array.isArray(schema.required) ? schema.required : []);
  for (const key of combinators) {
    delete inputSchema[key];
    const forms = schema[key].filter(form => form && typeof form === 'object');
    for (const form of forms) {
      for (const [name, property] of Object.entries(form.properties ?? {})) inputSchema.properties[name] ??= property;
    }
    const formRequired = forms.map(form => (Array.isArray(form.required) ? form.required : []));
    for (const name of formRequired[0] ?? []) {
      if (formRequired.every(names => names.includes(name))) required.add(name);
    }
  }
  if (required.size > 0) inputSchema.required = [...required];
  else delete inputSchema.required;
  return { ...tool, inputSchema };
}

function nativeForm(params, approval) {
  const choices = approval.choices;
  return {
    ...params,
    mode: 'form',
    message: approval.message,
    requestedSchema: {
      type: 'object',
      properties: {
        choice: {
          type: 'string',
          enum: choices.map(choice => choice.value),
          enumNames: choices.map(choice => choice.label),
        },
      },
      required: ['choice'],
    },
  };
}

/** Map Antigravity's Stop `terminationReason` to LCU's upstream turn-end event. */
export function stopEventOf(terminationReason) {
  return terminationReason === undefined || terminationReason === '' || terminationReason === 'NO_TOOL_CALL'
    ? 'Stop' : 'Interrupt';
}

/**
 * Run the relay on stdio. `hostPid` is the Antigravity process that started it, which also runs the hooks.
 * `createClient` (tests) replaces createCuaClient.
 */
export async function runAntigravityRelay({
  command, cwd, env = process.env, hostPid = process.ppid, directory = controlDirectory({ env }),
  shutdownDrainMs = SHUTDOWN_DRAIN_TIMEOUT_MS, createClient = createCuaClient,
} = {}) {
  // Conversation id -> { active: {turnId, calls}, pending: {turnId, event}, cleaning, told }.
  const sessions = new Map();
  // Most recently observed conversation id from hooks (used only when `_meta` omits `conversation_id`).
  let current;
  const fallback = `antigravity-${randomUUID()}`;
  // The launcher named in approval hints, so they work when `lcu` is not on PATH.
  const lcu = launcherOf(command);
  // Calls in flight -> the apps whose approval Antigravity could not show during them.
  const inFlight = new Set();
  let server;

  const client = createClient({
    command, cwd, env, adapter: 'antigravity',
    allowedOrigins: env.LCU_APPROVED_ORIGINS ? JSON.parse(env.LCU_APPROVED_ORIGINS) : [],
    onElicitation: async (params, { signal }) => {
      const approval = nativeAppApprovalOptions(params);
      const clientCaps = server?.getClientCapabilities?.();
      if (approval) {
        if (clientCaps?.elicitation?.form) {
          try {
            const answer = await server.elicitInput(nativeForm(params, approval), {
              signal, timeout: APPROVAL_TIMEOUT_MS,
            });
            if (answer?.action === 'accept' && typeof answer.content?.choice === 'string') {
              return nativeAppApprovalResponse(params, answer.content.choice);
            }
            if (answer?.action === 'decline') return { action: 'decline' };
          } catch {
            // Fall through to cancel + approvalHint when the surface cannot answer elicitation.
          }
        }
        // An approval request does not name the call it came from, so only a lone call can get the hint;
        // with calls overlapping, the model still sees the runtime's own "not approved" error.
        if (inFlight.size === 1) for (const call of inFlight) call.add(approval.resource);
        // Cancel, not decline: nothing is decided, so a retry after `lcu apps allow` is not refused.
        return { action: 'cancel' };
      }
      if (clientCaps?.elicitation) {
        try {
          const answer = await server.elicitInput(params, { signal, timeout: APPROVAL_TIMEOUT_MS });
          if (answer?.action === 'accept' || answer?.action === 'decline' || answer?.action === 'cancel') {
            return answer;
          }
        } catch {
          // Fall through to cancel.
        }
      }
      return { action: 'cancel' };
    },
  });

  const stateOf = (sessionId) => {
    let state = sessions.get(sessionId);
    if (!state) sessions.set(sessionId, state = {});
    return state;
  };

  /** End the conversation's open LCU turn upstream; a failure keeps it pending for the next call. */
  async function finish(sessionId, event) {
    const state = sessions.get(sessionId);
    if (!state) return;
    if (state.active) {
      state.pending = { turnId: state.active.turnId, event };
      state.active = undefined;
    }
    if (state.cleaning) return state.cleaning;
    const cleanup = state.pending;
    if (!cleanup) return;
    const attempt = (async () => {
      await client.turnEnded({ sessionId, turnId: cleanup.turnId, event: cleanup.event });
      if (state.pending === cleanup) state.pending = undefined;
    })();
    state.cleaning = attempt;
    try {
      await attempt;
    } finally {
      if (state.cleaning === attempt) state.cleaning = undefined;
    }
  }

  /**
   * The open LCU turn of `sessionId`, retrying any failed earlier turn end first (without interrupting other
   * concurrently active conversations or subagents).
   */
  async function turnOf(sessionId) {
    const state = stateOf(sessionId);
    const unfinished = [...sessions].filter(([id, other]) => id !== sessionId && (other.pending || other.cleaning));
    if (state.active && !unfinished.length) return state.active;
    try {
      await Promise.all([
        ...unfinished.map(([id]) => finish(id, 'Interrupt')),
        ...(state.active ? [] : [finish(sessionId, 'Interrupt')]),
      ]);
    } catch (error) {
      throw new Error('LCU is still finishing the previous turn, so Computer Use actions are paused. ' +
        `Try again in a few seconds. ${asError(error)}`);
    }
    state.active ??= { turnId: randomUUID(), calls: 0 };
    return state.active;
  }

  function instructionsFor(state) {
    if (state.told) return {};
    const instructions = client.instructions;
    if (!instructions) return {};
    state.told = true;
    return { systemMessage: instructions };
  }

  /** Answer one Antigravity hook: `{systemMessage?}`. */
  async function onHook(event, conversationId, { invocationNum, terminationReason } = {}) {
    if (!Object.hasOwn(HOOK_EVENTS, event) || typeof conversationId !== 'string' || !conversationId) {
      throw new Error('Expected an Antigravity hook event and conversationId');
    }
    if (event === 'SessionStart') {
      current = conversationId;
      return instructionsFor(stateOf(conversationId));
    }
    if (event === 'PreInvocation') {
      const early = current === undefined ? sessions.get(fallback) : undefined;
      if (early) {
        // Mid-turn continuation of a call that ran under the fallback session: keep `fallback` adoptable by Stop.
        if (invocationNum > 0) return early.told ? {} : instructionsFor(early);
        current = conversationId;
        if (early.told) stateOf(conversationId).told = true;
        if (early.active) await finish(fallback, early.active.calls > 0 ? 'Interrupt' : 'Stop').catch(() => {});
      }
      current = conversationId;
      const state = stateOf(conversationId);
      // `invocationNum === 0` marks the start of a new user turn; if a previous turn ended without firing Stop,
      // close it here. Mid-turn continuations (`invocationNum > 0`) stay inside the same open turn.
      if ((invocationNum === undefined || invocationNum === 0) && state.active) {
        await finish(conversationId, state.active.calls > 0 ? 'Interrupt' : 'Stop').catch(() => {});
      }
      return instructionsFor(state);
    }
    // Stop
    const stopEvent = stopEventOf(terminationReason);
    if (current === undefined && sessions.has(fallback) && !sessions.has(conversationId)) {
      current = conversationId;
      if (sessions.get(fallback).told) stateOf(conversationId).told = true;
      await finish(fallback, stopEvent);
      return {};
    }
    if (!sessions.has(conversationId)) return {};
    await finish(conversationId, stopEvent);
    return {};
  }

  // Hook requests: one JSON line `{hookEventName, conversationId, invocationNum?, terminationReason?}` in,
  // one `{ok, result|error}` line out.
  mkdirSync(directory, { recursive: true, mode: 0o700 });
  if (statSync(directory).uid !== process.getuid()) {
    throw new Error(`LCU socket directory belongs to another account: ${directory}`);
  }
  chmodSync(directory, 0o700);
  const socketPath = join(directory, `${hostPid}-${process.pid}.sock`);
  rmSync(socketPath, { force: true });
  const control = createServer(socket => {
    let input = '';
    const decoder = new StringDecoder('utf8');
    const reply = (value) => { if (!socket.destroyed) socket.end(`${JSON.stringify(value)}\n`); };
    socket.on('error', () => {});
    socket.on('data', async chunk => {
      if (input === null) return;
      input += decoder.write(chunk);
      if (input.length > REQUEST_LIMIT) {
        input = null;
        reply({ ok: false, error: 'request too large' });
        return;
      }
      const newline = input.indexOf('\n');
      if (newline < 0) return;
      const line = input.slice(0, newline);
      input = null;
      try {
        const request = JSON.parse(line);
        reply({
          ok: true,
          result: await onHook(request?.hookEventName, request?.conversationId, {
            invocationNum: request?.invocationNum,
            terminationReason: request?.terminationReason,
          }),
        });
      } catch (error) {
        reply({ ok: false, error: asError(error) });
      }
    });
  });

  let shutdown;
  const close = () => shutdown ??= (async () => {
    control.close();
    rmSync(socketPath, { force: true });
    let timer;
    const drained = Promise.allSettled([...sessions.keys()].map(id => finish(id, 'Interrupt')));
    await Promise.race([drained, new Promise(resolve => { timer = setTimeout(resolve, shutdownDrainMs); })]);
    clearTimeout(timer);
    await client.close().catch(error => console.error('Antigravity MCP relay upstream close failed:', asError(error)));
    await server?.close().catch(() => {});
  })();

  try {
    await client.connect();
    await new Promise((resolve, reject) => {
      control.once('error', reject);
      control.listen(socketPath, resolve);
    });
    chmodSync(socketPath, 0o600);
    server = new Server({ name: 'lcu-antigravity-relay', version: '0.1.0' }, {
      capabilities: { tools: {} },
      instructions: client.instructions,
    });
    server.onerror = error => console.error('Antigravity MCP relay server error:', asError(error));
    server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: client.publicTools().map(antigravityToolDescriptor),
    }));
    server.setRequestHandler(CallToolRequestSchema, async (request, extra) => {
      const { name, arguments: args = {}, _meta } = request.params;
      if (!MODEL_TOOLS.has(name)) return { isError: true, content: [{ type: 'text', text: `Unknown LCU tool: ${name}` }] };
      const metaSession = typeof _meta?.[CONVERSATION_ID_META] === 'string' && _meta[CONVERSATION_ID_META]
        ? _meta[CONVERSATION_ID_META] : undefined;
      const sessionId = metaSession ?? current ?? fallback;
      const turn = await turnOf(sessionId);
      const approvals = new Set();
      turn.calls++;
      inFlight.add(approvals);
      try {
        const result = await persistAudioContent(await client.call(name, args, {
          sessionId, turnId: turn.turnId, toolCallId: randomUUID(), signal: extra.signal,
        }));
        const added = [...approvals].map(app => ({ type: 'text', text: approvalHint(app, lcu) }));
        // If hooks could not reach the relay before this conversation's first call (e.g. lazy MCP startup),
        // carry the unchanged original instructions once before the result.
        const state = stateOf(sessionId);
        const leading = [];
        if (!state.told && client.instructions) {
          state.told = true;
          leading.push({ type: 'text', text: `${INSTRUCTIONS_HEADER}\n\n${client.instructions}` });
        }
        if (!added.length && !leading.length) return result;
        return { ...result, content: [...leading, ...(result.content ?? []), ...added] };
      } finally {
        inFlight.delete(approvals);
        turn.calls--;
      }
    });
    server.onclose = () => { void close(); };
    await server.connect(new StdioServerTransport());
  } catch (error) {
    await close();
    throw error;
  }
  // The SDK's stdio transport does not surface stdin EOF, which is how Antigravity closes the server.
  process.stdin.once('end', () => { void close(); });
  return { server, socketPath, sessions, close };
}

/** The parent pid of `pid`, or undefined. */
function parentOf(pid) {
  try {
    const parent = Number(execFileSync('/bin/ps', ['-o', 'ppid=', '-p', String(pid)],
      { encoding: 'utf8', timeout: 5_000, stdio: ['ignore', 'pipe', 'ignore'] }).trim());
    return Number.isInteger(parent) && parent > 1 ? parent : undefined;
  } catch {
    return undefined;
  }
}

/** Send one request to a relay socket; resolves to its result, or rejects. */
function ask(socketPath, request, timeoutMs) {
  return new Promise((resolve, reject) => {
    let buffer = '';
    const decoder = new StringDecoder('utf8');
    const socket = createConnection(socketPath);
    const timer = setTimeout(() => { socket.destroy(); reject(new Error('timed out')); }, timeoutMs);
    const done = (error, value) => {
      clearTimeout(timer);
      socket.destroy();
      if (error) reject(error); else resolve(value);
    };
    socket.on('error', error => done(error));
    socket.on('connect', () => socket.write(`${JSON.stringify(request)}\n`));
    socket.on('data', chunk => {
      buffer += decoder.write(chunk);
      const newline = buffer.indexOf('\n');
      if (newline < 0) return;
      try {
        const response = JSON.parse(buffer.slice(0, newline));
        if (response?.ok) done(undefined, response.result ?? {});
        else done(new Error(typeof response?.error === 'string' ? response.error : 'relay refused the hook'));
      } catch (error) {
        done(error);
      }
    });
    socket.on('end', () => done(new Error('relay closed the connection')));
  });
}

/**
 * Antigravity command hook: tell the relays of the Antigravity process that runs this hook about `hookEventName`,
 * and print `injectSteps` for `SessionStart`/`PreInvocation` when the relay returns original instructions.
 * Always writes valid JSON (`{}` when there is nothing to inject) and exits 0 so Antigravity is never blocked.
 */
export async function runAntigravityHook({
  stdin = readFileSync(0, 'utf8'), env = process.env, directory = controlDirectory({ env }),
  parent = process.ppid, write = text => process.stdout.write(text),
} = {}) {
  const finishEmpty = () => write('{}');
  let input;
  try {
    input = JSON.parse(stdin);
  } catch {
    finishEmpty();
    return;
  }
  const event = input?.hookEventName;
  const conversationId = input?.conversationId;
  if (!Object.hasOwn(HOOK_EVENTS, event) || typeof conversationId !== 'string' || !conversationId) {
    finishEmpty();
    return;
  }
  let names;
  try {
    if (!statSync(directory).isDirectory()) {
      finishEmpty();
      return;
    }
    names = readdirSync(directory);
  } catch {
    finishEmpty();
    return;
  }
  const timeoutMs = (HOOK_TIMEOUTS[event] - 2) * 1000;
  const payload = {
    hookEventName: event,
    conversationId,
    ...(typeof input.invocationNum === 'number' ? { invocationNum: input.invocationNum } : {}),
    ...(typeof input.terminationReason === 'string' ? { terminationReason: input.terminationReason } : {}),
  };
  const messages = [];
  for (let pid = parent, depth = 0; pid && depth < HOOK_ANCESTORS; pid = parentOf(pid), depth++) {
    const sockets = names.filter(name => name.startsWith(`${pid}-`) && name.endsWith('.sock'));
    if (!sockets.length) continue;
    await Promise.all(sockets.map(async name => {
      const socketPath = join(directory, name);
      try {
        const result = await ask(socketPath, payload, timeoutMs);
        if (typeof result.systemMessage === 'string' && result.systemMessage) {
          messages.push(result.systemMessage);
        }
      } catch (error) {
        // A relay that exited without removing its socket.
        if (['ECONNREFUSED', 'ENOENT'].includes(error?.code)) rmSync(socketPath, { force: true });
      }
    }));
    break;
  }
  const unique = [...new Set(messages)];
  if ((event === 'SessionStart' || event === 'PreInvocation') && unique.length) {
    write(JSON.stringify({
      injectSteps: unique.map(text => ({ systemMessage: { systemMessage: text } })),
    }));
    return;
  }
  finishEmpty();
}

async function main() {
  const argv = process.argv.slice(2);
  if (argv.length === 1 && argv[0] === '--hook') {
    await runAntigravityHook().catch(() => process.stdout.write('{}'));
    return;
  }
  if (!argv.length) throw new Error('Usage: antigravity.mjs ORIGINAL_MCP_COMMAND [ARG ...] | antigravity.mjs --hook');
  await runAntigravityRelay({ command: argv });
}

if (isMainModule(import.meta.url)) {
  main().catch(error => {
    console.error('Antigravity MCP relay failed:', asError(error));
    process.exit(1);
  });
}
