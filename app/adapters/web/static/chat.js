const elements = Object.fromEntries(
  Array.from(document.querySelectorAll("[id]"), (element) => [element.id, element]),
);
const state = { active: null, sessions: [], pending: false, loading: false, historyError: "" };
const contextUsage = { key: null, controller: null, failed: false };
const mobile = window.matchMedia("(max-width: 700px)");
let selection = 0;

const providerNames = {
  openai: "OpenAI",
  gemini: "Google Gemini",
  qwen: "通义千问",
  deepseek: "DeepSeek",
  glm: "智谱 GLM",
};
const knownProviders = {
  dashscope: { label: "阿里云百炼", icon: "qwen" },
  deepseek: { label: "DeepSeek", icon: "deepseek" },
  glm: { label: "智谱 GLM", icon: "glm" },
  gemini: { label: "Google Gemini", icon: "gemini" },
  openai: { label: "OpenAI", icon: "openai" },
  default: { label: "默认配置", icon: "local" },
  offline: { label: "离线模式", icon: "local" },
};
const modelStorageKey = "myagent.model-preferences";
const modelPreferences = readModelPreferences();
const lastModelByProvider = new Map();
let catalog = [];
let modelGroups = new Map();
let modelOffline = false;
let catalogReady = false;
let catalogError = "";
let openPanel = "";
let modelMutationPending = false;
let modelToDelete = null;

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function readModelPreferences() {
  try {
    const saved = JSON.parse(localStorage.getItem(modelStorageKey));
    if (typeof saved?.selected === "string") {
      return { provider: saved.provider || "", selected: saved.selected };
    }
  } catch {
    // 本地存储不可用时仍可在当前页面选择模型。
  }
  return { provider: "", selected: "" };
}

function rememberModel(provider, id = "") {
  modelPreferences.provider = provider;
  modelPreferences.selected = id;
  if (id) lastModelByProvider.set(provider, id);
  try {
    localStorage.setItem(modelStorageKey, JSON.stringify(modelPreferences));
  } catch {
    // 浏览器可能禁用本地存储。
  }
}

function selectedModel() {
  return catalog.find((model) => model.id === modelPreferences.selected && model.available);
}

function firstAvailable(models) {
  return models.find((model) => model.available && model.is_default) || models.find((model) => model.available);
}

function setCatalog(models, preferredId = "") {
  if (!Array.isArray(models)) throw new Error("服务返回的模型列表格式有误。");
  catalogReady = true;
  catalogError = "";
  catalog = models;
  modelGroups = new Map();
  for (const model of catalog) {
    if (!modelGroups.has(model.provider)) {
      const meta = Object.hasOwn(knownProviders, model.provider) ? knownProviders[model.provider] : null;
      modelGroups.set(model.provider, {
        id: model.provider,
        label: meta?.label || model.vendor_label || model.provider,
        icon: model.provider === "default" ? model.vendor : meta?.icon || model.vendor,
        models: [],
      });
    }
    modelGroups.get(model.provider).models.push(model);
  }

  const preferred = catalog.find((model) => model.id === preferredId);
  const saved = catalog.find((model) => model.id === modelPreferences.selected);
  const group = modelGroups.get(preferred?.provider || saved?.provider || modelPreferences.provider);
  const candidate = (preferred?.available && preferred)
    || (!preferred && saved?.available && saved)
    || firstAvailable(group?.models || []);
  const keepGroup = preferred || (group && !modelPreferences.selected);
  const fallback = keepGroup ? null : firstAvailable(catalog);
  const active = candidate || fallback;
  rememberModel(active?.provider || group?.id || catalog[0]?.provider || "", active?.id);
  renderModelPicker();
}

function renderProviderIcon(icon, provider) {
  const hasMark = Object.hasOwn(providerNames, provider);
  icon.classList.toggle("local-icon", !hasMark);
  icon.setAttribute("aria-hidden", "true");
  icon.replaceChildren();
  if (!hasMark) return;
  const namespace = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(namespace, "svg");
  const use = document.createElementNS(namespace, "use");
  use.setAttribute("href", "#provider-" + provider);
  svg.append(use);
  icon.append(svg);
}

function pickerOption({ title, subtitle, icon, pressed, disabled, choose }) {
  const button = node("button", "model-option");
  button.type = "button";
  button.disabled = Boolean(disabled);
  button.setAttribute("aria-pressed", String(pressed));
  if (icon) {
    const mark = node("span", "provider-icon");
    renderProviderIcon(mark, icon);
    button.append(mark);
  }
  const text = node("span", "model-option-text");
  text.append(node("span", "model-option-name", title));
  if (subtitle) text.append(node("span", "model-option-provider", subtitle));
  button.append(text);
  button.addEventListener("click", choose);
  return button;
}

function setNotice(id, message = "") {
  const box = elements[id];
  box.textContent = message;
  box.hidden = !message;
}

function renderModelPicker() {
  const group = modelGroups.get(modelPreferences.provider);
  const model = selectedModel();
  const models = group?.models || [];
  const vendorLabel = group?.label || "厂商";
  const modelLabel = model?.name || (catalogReady ? "选择模型" : "正在加载");
  elements["provider-trigger-label"].textContent = vendorLabel;
  elements["provider-trigger"].setAttribute("aria-label", "选择厂商，当前：" + vendorLabel);
  elements["provider-trigger"].title = vendorLabel;
  renderProviderIcon(elements["provider-trigger-icon"], group?.icon || "local");
  elements["model-trigger-label"].textContent = modelLabel;
  elements["model-trigger"].setAttribute("aria-label", "选择模型，当前：" + modelLabel);
  elements["model-trigger"].title = modelLabel;
  elements["model-panel-note"].textContent = modelOffline ? "离线模式" : "";

  const providers = [...modelGroups.values()].map((item) => pickerOption({
    title: item.label,
    subtitle: item.models.some((entry) => entry.available) ? "" : "暂无可用模型",
    icon: item.icon,
    pressed: item.id === group?.id,
    choose: () => chooseProvider(item),
  }));
  const rows = models.map((item) => {
    const row = node("div", "model-option-row");
    row.append(pickerOption({
      title: item.name,
      subtitle: item.available ? "" : modelOffline ? "离线模式下不可用" : "连接未就绪",
      icon: item.vendor,
      pressed: item.id === model?.id,
      disabled: !item.available,
      choose: () => chooseModel(item),
    }));
    if (item.custom) {
      const action = node("button", "delete-model");
      action.type = "button";
      action.title = "删除 " + item.name;
      action.setAttribute("aria-label", action.title);
      action.setAttribute("aria-haspopup", "dialog");
      const mark = node("span", "trash-icon");
      mark.setAttribute("aria-hidden", "true");
      action.append(mark);
      action.addEventListener("click", () => confirmDeleteModel(item));
      row.append(action);
    }
    return row;
  });
  elements["provider-options"].replaceChildren(...providers);
  elements["model-options"].replaceChildren(...rows);
  const loadingMessage = catalogReady ? "暂无模型" : "正在加载…";
  setNotice("provider-empty", providers.length ? "" : catalogError || loadingMessage);
  setNotice("model-empty", rows.length ? "" : catalogError || loadingMessage);
  if (catalogReady) {
    const status = elements["model-status"];
    const label = elements["model-label"];
    status.dataset.state = catalogError ? "error" : modelOffline ? "offline" : model ? "live" : "error";
    label.textContent = catalogError || model?.name || "暂无可用模型";
    label.title = model?.id || label.textContent;
  }
  updateComposer();
  void refreshContextUsage();
}

function setPickerPanel(name = "", returnFocus = "") {
  if (name) setContextPopover(false);
  openPanel = name;
  for (const panel of ["provider", "model"]) {
    elements[panel + "-panel"].hidden = panel !== name;
    elements[panel + "-trigger"].setAttribute("aria-expanded", String(panel === name));
  }
  if (name) {
    const list = elements[name + "-options"];
    const focus = list.querySelector('[aria-pressed="true"]') || list.querySelector("button:not(:disabled)");
    (focus || (name === "model" ? elements["open-add-model"] : elements["provider-trigger"])).focus();
  }
  if (returnFocus) elements[returnFocus + "-trigger"].focus();
}

function chooseProvider(group) {
  const previous = lastModelByProvider.get(group.id);
  const model = group.models.find((item) => item.id === previous && item.available) || firstAvailable(group.models);
  rememberModel(group.id, model?.id);
  setNotice("model-feedback");
  renderModelPicker();
  setPickerPanel("model");
}

function chooseModel(model) {
  rememberModel(model.provider, model.id);
  setNotice("model-feedback");
  setPickerPanel("", "model");
  renderModelPicker();
}

function updateModelForm() {
  const custom = elements["model-provider"].value === "custom";
  elements["custom-provider-field"].hidden = !custom;
  elements["custom-provider"].required = custom;
  if (custom) elements["model-connection"].open = true;
  setNotice("model-error");
}

function openAddModel() {
  const form = elements["model-form"];
  const providerField = elements["model-provider"];
  const group = modelGroups.get(modelPreferences.provider);
  form.reset();
  elements["model-connection"].open = false;
  if (group && !["default", "offline"].includes(group.id)) {
    providerField.value = Array.from(providerField.options).some((option) => option.value === group.id) ? group.id : "custom";
    if (providerField.value === "custom") elements["custom-provider"].value = group.id;
  }
  updateModelForm();
  if (group && !group.models.some((item) => item.available) && group.id !== "offline") {
    elements["model-connection"].open = true;
  }
  setPickerPanel();
  elements["model-dialog"].showModal();
}

function setModelMutation(dialog, pending) {
  modelMutationPending = pending;
  dialog.setAttribute("aria-busy", String(pending));
  for (const field of dialog.querySelectorAll("button, select, input")) field.disabled = pending;
}

async function refreshAfterMutation(preferredId, notice) {
  try {
    const data = await request("/api/v1/models");
    setCatalog(data.models, preferredId);
  } catch {
    notice += " 暂未同步完整列表，刷新页面可重新加载。";
  }
  setNotice("model-feedback", notice);
}

async function addModel(event) {
  event.preventDefault();
  if (modelMutationPending) return;
  const dialog = elements["model-dialog"];
  const picker = elements["model-provider"].value;
  const custom = picker === "custom";
  const provider = custom ? elements["custom-provider"].value.trim() : picker;
  const model = elements["model-id"].value.trim();
  if (!provider || !model) {
    setNotice("model-error", "请填写厂商和模型 ID。");
    return;
  }
  const group = modelGroups.get(provider);
  const payload = {
    provider,
    model,
    name: elements["model-name"].value.trim(),
    vendor_label: custom ? group?.label || provider : "",
    base_url: elements["custom-base-url"].value.trim(),
    api_key: elements["custom-api-key"].value.trim(),
  };
  setNotice("model-error");
  setModelMutation(dialog, true);
  try {
    const created = await request("/api/v1/models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    setCatalog([...catalog.filter((item) => item.id !== created.id), created], created.id);
    const notice = created.available ? "已保存模型。" : "已保存模型，当前连接未就绪。";
    await refreshAfterMutation(created.id, notice);
  } catch (error) {
    if (error.status === 422) elements["model-connection"].open = true;
    setNotice("model-error", error.status ? error.message : "保存失败，连接已中断，请重试。");
    return;
  } finally {
    setModelMutation(dialog, false);
  }
  dialog.close();
  setPickerPanel("model");
}

function confirmDeleteModel(model) {
  modelToDelete = model;
  elements["delete-description"].textContent = "将从当前模型列表移除「" + model.name + "」。";
  setNotice("delete-error");
  setPickerPanel();
  elements["delete-dialog"].showModal();
}

async function deleteModel(event) {
  event.preventDefault();
  if (modelMutationPending || !modelToDelete) return;
  const model = modelToDelete;
  const dialog = elements["delete-dialog"];
  setNotice("delete-error");
  setModelMutation(dialog, true);
  try {
    await request("/api/v1/models/" + encodeURIComponent(model.id), { method: "DELETE" }).catch((error) => {
      if (error.status !== 404) throw error;
    });
    setCatalog(catalog.filter((item) => item.id !== model.id));
    await refreshAfterMutation("", "已移除模型。");
  } catch (error) {
    setNotice("delete-error", error.status ? error.message : "删除失败，连接已中断，请重试。");
    return;
  } finally {
    setModelMutation(dialog, false);
  }
  modelToDelete = null;
  dialog.close();
  setPickerPanel("model");
}

async function request(path, options) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const error = new Error(data?.error || "请求失败。");
    error.status = response.status;
    throw error;
  }
  if (data === null) throw new Error("服务返回了无法读取的响应。");
  return data;
}

function sessionPath(id) {
  return `/api/sessions/${encodeURIComponent(id)}`;
}

function setContextPopover(open, restoreFocus = false) {
  const trigger = elements["context-trigger"];
  elements["context-popover"].hidden = !open;
  trigger.setAttribute("aria-expanded", String(open));
  if (restoreFocus) trigger.focus();
}

function renderContextUsage(message, percent = 0, percentLabel = "") {
  elements["context-popover"].textContent = message;
  elements["context-ring"].style.setProperty("--context-fill", `${Math.max(0, Math.min(100, percent))}%`);
  elements["context-trigger"].setAttribute("aria-label", percentLabel ? `上下文使用量，已使用 ${percentLabel}` : "上下文使用量");
}

function compactTokens(tokens) {
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}M`;
  if (tokens >= 1_000) return `${(tokens / 1_000).toFixed(1)}K`;
  return String(tokens);
}

function displayContextReport(report, session, model) {
  const useReported = report.usage_source === "reported"
    && Number.isFinite(report.last_prompt_tokens)
    && (!model || model.id === session.last_model);
  const used = useReported ? report.last_prompt_tokens : report.estimated_context_tokens;
  const limit = report.context_window_tokens;
  const qualifier = useReported ? "" : "约 ";
  if (!Number.isFinite(limit) || limit <= 0) {
    renderContextUsage(`已使用 ${qualifier}${compactTokens(used)} tokens · 窗口未知`);
    return;
  }
  const percent = (used / limit) * 100;
  const percentLabel = percent > 0 && percent < 0.1 ? "<0.1%" : `${percent.toFixed(1)}%`;
  renderContextUsage(`${qualifier}${percentLabel} · ${compactTokens(used)} / ${compactTokens(limit)} 上下文已使用`, percent, percentLabel);
}

async function refreshContextUsage(force = false) {
  const session = state.loading ? null : state.active;
  const model = selectedModel();
  const key = state.loading ? null : session ? `${session.id}\0${model?.id || ""}` : "";
  if (!force && key === contextUsage.key && !state.loading) return;

  contextUsage.controller?.abort();
  contextUsage.controller = null;
  contextUsage.key = key;
  contextUsage.failed = false;
  if (!session || state.pending) {
    renderContextUsage(state.loading || state.pending ? "正在更新上下文用量…" : "发送消息后显示上下文用量。");
    return;
  }

  const controller = new AbortController();
  contextUsage.controller = controller;
  const query = model ? `?${new URLSearchParams({ model: model.id })}` : "";
  renderContextUsage("正在更新上下文用量…");
  try {
    const report = await request(`${sessionPath(session.id)}/context-usage${query}`, { signal: controller.signal });
    if (!controller.signal.aborted) displayContextReport(report, session, model);
  } catch {
    if (!controller.signal.aborted) {
      contextUsage.failed = true;
      renderContextUsage("暂时无法读取上下文用量。");
    }
  } finally {
    if (contextUsage.controller === controller) contextUsage.controller = null;
  }
}

function showError(text = "") {
  elements.error.textContent = text;
  elements.error.hidden = !text;
}

function rememberSession(id) {
  const url = new URL(window.location.href);
  url.hash = id ? encodeURIComponent(id) : "";
  history.replaceState(null, "", url);
}

function setSidebar(open, restoreFocus = false) {
  if (open) {
    setPickerPanel();
    setContextPopover(false);
  }
  elements.sidebar.dataset.open = String(open);
  elements["open-sidebar"].setAttribute("aria-expanded", String(open));
  elements.backdrop.hidden = !open;
  elements.main.inert = open;
  if (open) elements["close-sidebar"].focus();
  if (restoreFocus) elements["open-sidebar"].focus();
}

function renderHistory() {
  const empty = elements["empty-history"];
  empty.hidden = state.sessions.length > 0;
  empty.textContent = state.historyError || "暂无会话";
  const items = state.sessions.map(({ id, title }) => {
    const item = node("li");
    const button = node("button", "session-button");
    const icon = node("span", "bubble-icon");
    icon.setAttribute("aria-hidden", "true");
    button.type = "button";
    button.title = title;
    button.disabled = state.pending;
    if (id === state.active?.id) button.setAttribute("aria-current", "true");
    button.append(icon, node("span", "session-title", title));
    button.addEventListener("click", () => selectSession(id));
    item.append(button);
    return item;
  });
  elements.sessions.replaceChildren(...items);
}

function storeSummary({ id, title }) {
  state.sessions = [{ id, title }, ...state.sessions.filter((session) => session.id !== id)];
}

function appendInline(target, text) {
  const tokens = /\*\*([^*\r\n]+)\*\*|`([^`\r\n]+)`|\[([^\]\r\n]+)\]\((https?:\/\/[^\s)]+)\)/g;
  let offset = 0;
  for (const token of text.matchAll(tokens)) {
    target.append(text.slice(offset, token.index));
    if (token[1]) {
      target.append(node("strong", "", token[1]));
    } else if (token[2]) {
      target.append(node("code", "", token[2]));
    } else {
      const link = node("a", "", token[3]);
      link.href = token[4];
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      target.append(link);
    }
    offset = token.index + token[0].length;
  }
  target.append(text.slice(offset));
}

function renderReply(content) {
  const reply = node("div", "reply");
  const parts = content.split(/(?:^|\r?\n)```[^\r\n]*\r?\n([\s\S]*?)\r?\n```(?=\r?\n|$)/g);
  parts.forEach((part, index) => {
    if (index % 2) {
      const code = node("pre");
      code.append(node("code", "", part));
      reply.append(code);
    } else {
      appendInline(reply, part);
    }
  });
  return reply;
}

function assistantMessage(content, pending = false, modelName = "") {
  const message = node("article", "message assistant");
  const mark = node("span", "agent-mark");
  mark.setAttribute("aria-hidden", "true");
  message.setAttribute("aria-label", "MyAgent");
  const reply = pending ? node("div", "pending", content) : renderReply(content);
  if (pending) reply.setAttribute("role", "status");
  if (!pending && modelName) {
    reply.append(node("div", "reply-model", `模型：${modelName}`));
  }
  message.append(mark, reply);
  return message;
}

function toolDetails(messages) {
  const details = node("details", "tools");
  details.append(node("summary", "", `工具记录（${messages.length}）`));
  const list = node("ul", "tool-list");
  for (const message of messages) {
    const item = node("li");
    const content = message.content;
    let status = "已完成";
    let kind = "success";
    if (content.startsWith("[REJECTED]")) {
      status = "已拦截";
      kind = "rejected";
    } else if (content.startsWith("[ERROR]")) {
      status = "执行失败";
      kind = "failed";
    }
    item.append(
      node("code", "", message.tool_name || "工具"),
      node("span", `tool-state ${kind}`, status),
      node("p", "", content.replace(/^\[(REJECTED|ERROR)\]\s*/, "")),
    );
    list.append(item);
  }
  details.append(list);
  return details;
}

function renderConversation() {
  const messages = state.active?.messages || [];
  const conversation = elements.conversation;
  elements.welcome.hidden = messages.length > 0;
  conversation.hidden = messages.length === 0;
  elements["chat-title"].textContent = state.active?.title || "新对话";

  const rows = [];
  let tools = [];
  for (const message of messages) {
    if (message.role === "tool") {
      tools.push(message);
      continue;
    }
    if (message.intermediate) continue;
    if (tools.length) {
      rows.push(toolDetails(tools));
      tools = [];
    }
    if (message.role === "user") {
      const row = node("article", "message user");
      row.setAttribute("aria-label", "你");
      const lines = String(message.content || "").split(/\r?\n/);
      const quoteLines = [];
      let idx = 0;
      while (idx < lines.length && lines[idx].startsWith(">")) {
        quoteLines.push(lines[idx].replace(/^>\s?/, ""));
        idx += 1;
      }
      if (quoteLines.length && idx < lines.length && lines[idx].trim() === "") {
        idx += 1;
      }
      if (quoteLines.length && idx < lines.length) {
        row.append(node("blockquote", "user-quote", quoteLines.join("\n")));
        row.append(document.createTextNode(lines.slice(idx).join("\n")));
      } else {
        row.textContent = message.content;
      }
      rows.push(row);
    } else if (message.role === "assistant" && message.content) {
      rows.push(assistantMessage(message.content, false, message.model_name || ""));
    }
  }
  if (tools.length) rows.push(toolDetails(tools));
  if (state.pending) {
    if (state.streamTools?.length) rows.push(toolDetails(state.streamTools));
    if (state.streamText) {
      const live = assistantMessage(state.streamText, false, "");
      live.dataset.streaming = "true";
      rows.push(live);
    } else {
      rows.push(assistantMessage("正在处理…", true));
    }
  }
  conversation.replaceChildren(...rows);
  elements.viewport.scrollTop = elements.viewport.scrollHeight;
}

function updateComposer() {
  const busy = state.pending || state.loading;
  elements.prompt.readOnly = busy;
  elements.send.disabled = busy || !selectedModel() || !elements.prompt.value.trim();
  elements.send.textContent = state.pending ? "处理中" : "发送";
  elements["new-chat"].disabled = state.pending;
  elements["open-add-model"].disabled = state.pending || !catalogReady;
  for (const button of elements["model-options"].querySelectorAll(".delete-model")) {
    button.disabled = state.pending;
  }
}

function newChat() {
  selection += 1;
  state.loading = false;
  state.active = null;
  setContextPopover(false);
  void refreshContextUsage();
  void loadSideBranches("standalone");
  elements.prompt.value = "";
  rememberSession();
  setSidebar(false);
  showError();
  renderHistory();
  renderConversation();
  updateComposer();
  elements.prompt.focus();
}

async function selectSession(id, focus = true) {
  const current = ++selection;
  state.loading = true;
  setContextPopover(false);
  void refreshContextUsage();
  setSidebar(false);
  showError();
  updateComposer();
  try {
    const session = await request(sessionPath(id));
    if (current !== selection) return;
    if (state.active?.id !== id) elements.prompt.value = "";
    state.active = session;
    rememberSession(id);
    renderHistory();
    renderConversation();
    void loadSideBranches(id);
  } catch (error) {
    if (current !== selection) return;
    if (!state.active && error.status === 404) rememberSession();
    showError(error.status ? error.message : "暂时无法加载会话，请重试。");
  } finally {
    if (current === selection) {
      state.loading = false;
      void refreshContextUsage();
      updateComposer();
      if (focus) elements.prompt.focus();
    }
  }
}

function formatStreamToolContent(ev) {
  if (!ev.approved) return `[REJECTED] ${ev.summary || "审核未通过"}`;
  if (!ev.success) return `[ERROR] ${ev.summary || "执行失败"}`;
  return ev.summary || "";
}

async function consumeSSE(path, payload, handlers) {
  const response = await fetch(path, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    let data = {};
    try {
      data = await response.json();
    } catch {
      // 非 JSON 错误体
    }
    const error = new Error(data.error || "请求失败。");
    error.status = response.status;
    throw error;
  }
  if (!response.body) {
    throw new Error("浏览器不支持流式响应读取。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  const dispatchFrame = (rawFrame) => {
    let eventName = "message";
    const dataLines = [];
    for (const line of rawFrame.split(/\r?\n/)) {
      if (!line || line.startsWith(":")) continue;
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trimStart());
      }
    }
    if (!dataLines.length) return;
    const dataText = dataLines.join("\n");
    let parsed = dataText;
    try {
      parsed = JSON.parse(dataText);
    } catch {
      // 保留纯文本
    }
    handlers[eventName]?.(parsed);
  };

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const frames = buffer.split(/\r?\n\r?\n/);
    buffer = frames.pop() || "";
    for (const frame of frames) {
      if (frame.trim()) dispatchFrame(frame);
    }
    if (done) {
      if (buffer.trim()) dispatchFrame(buffer);
      break;
    }
  }
}

async function sendMessage(event) {
  event.preventDefault();
  const rawPrompt = elements.prompt.value.trim();
  const quoteText = (state.quote || "").trim();
  const activeModel = selectedModel();
  if (!rawPrompt || !activeModel || state.pending || state.loading) return;

  // 支持在主输入框直接输入 "/btw 问题" 快速打开侧边分支追问
  if (/^\/btw(?:\s+|$)/i.test(rawPrompt)) {
    const btwQuestion = rawPrompt.replace(/^\/btw\s*/i, "").trim();
    elements.prompt.value = "";
    if (quoteText) {
      setSideQuote(quoteText);
      setMainQuote("");
    }
    setSidePanel(true);
    updateComposer();
    if (btwQuestion) {
      elements["side-prompt"].value = btwQuestion;
      updateSideComposer();
      elements["side-form"].requestSubmit();
    }
    return;
  }

  setPickerPanel();
  setContextPopover(false);
  hideSelectionToolbar();

  const content = quoteText
    ? `${quoteText.split(/\r?\n/).map((line) => `> ${line}`).join("\n")}\n\n${rawPrompt}`
    : rawPrompt;

  const previous = state.active;
  const id = previous?.id || `sess_${crypto.randomUUID()}`;
  const messages = previous?.messages || [];
  state.pending = true;
  state.streamText = "";
  state.streamTools = [];
  setMainQuote("");
  state.active = {
    id,
    title: previous?.title || Array.from(rawPrompt).slice(0, 40).join(""),
    messages: [...messages, { role: "user", content }],
  };
  elements.prompt.value = "";
  showError();
  renderHistory();
  renderConversation();
  updateComposer();

  let streamCycle = 0;
  let streamError = null;
  let finished = false;
  try {
    await consumeSSE(
      `${sessionPath(id)}/messages/stream`,
      { content, provider: activeModel.provider, model: activeModel.model },
      {
        delta(data) {
          if (typeof data?.cycle === "number" && data.cycle !== streamCycle) {
            streamCycle = data.cycle;
            state.streamText = "";
          }
          if (typeof data?.content === "string" && data.content) {
            state.streamText += data.content;
            renderConversation();
          }
        },
        tool(data) {
          state.streamTools.push({
            role: "tool",
            tool_name: data?.tool_name || "工具",
            content: formatStreamToolContent(data || {}),
          });
          state.streamText = "";
          renderConversation();
        },
        done(session) {
          state.active = session;
          storeSummary(state.active);
          rememberSession(id);
          finished = true;
        },
        error(data) {
          const err = new Error(data?.error || "本轮处理未完成，请重试。");
          err.status = data?.status || 502;
          streamError = err;
        },
      },
    );
    if (streamError) throw streamError;
    if (!finished) {
      const err = new Error("流式响应提前中断。");
      err.status = 0;
      throw err;
    }
  } catch (error) {
    state.active = previous;
    showError(error.status ? error.message : "连接中断，正在检查消息是否已保存…");
    let saved = false;
    if (!error.status || error.status === 502) {
      const session = await request(sessionPath(id)).catch(() => null);
      const newMessages = session?.messages.slice(messages.length) || [];
      saved = newMessages.some((message) => message.role === "user" && message.content === content);
      if (saved) {
        state.active = session;
        storeSummary(session);
        rememberSession(id);
        if (!error.status) showError("连接中断，已重新加载会话。");
      }
    }
    if (!saved) {
      elements.prompt.value = rawPrompt;
      if (quoteText) setMainQuote(quoteText);
      if (!error.status) showError("连接中断，请确认本地服务正在运行后重试。");
    }
  } finally {
    state.pending = false;
    state.streamText = "";
    state.streamTools = [];
    void refreshContextUsage(true);
    renderHistory();
    renderConversation();
    updateComposer();
    if (elements.sidebar.dataset.open !== "true") elements.prompt.focus();
  }
}

/* ---- 划词浮窗（Quote Ctrl+L / Side Question Ctrl+E）与侧边独立分支聊天 ---- */
const sideState = {
  open: false,
  parentId: "standalone",
  sideId: null,
  branches: [],
  quote: "",
  messages: [],
  pending: false,
  streamText: "",
  streamTools: [],
};
let activeSelectionText = "";
let sideBranchReqSeq = 0;

function currentParentSessionId() {
  return state.active?.id || "standalone";
}

function setMainQuote(text) {
  state.quote = (text || "").trim();
  const hasQuote = Boolean(state.quote);
  elements["quote-bar"].hidden = !hasQuote;
  elements["quote-bar-text"].textContent = state.quote;
}

function setSideQuote(text) {
  sideState.quote = (text || "").trim();
  const hasQuote = Boolean(sideState.quote);
  elements["side-quote-card"].hidden = !hasQuote;
  elements["side-quote-text"].textContent = sideState.quote;
}

function showSideError(text = "") {
  elements["side-error"].textContent = text;
  elements["side-error"].hidden = !text;
}

function renderSideBranches() {
  const select = elements["side-branch-select"];
  const delBtn = elements["delete-side-branch"];
  const options = [];
  const blank = document.createElement("option");
  blank.value = "";
  blank.textContent = sideState.branches.length
    ? `新分支（已有 ${sideState.branches.length} 个独立分支）`
    : "新分支（自动独立落盘）";
  options.push(blank);
  for (const b of sideState.branches) {
    const opt = document.createElement("option");
    opt.value = b.id;
    opt.textContent = `${b.title || b.id} (${b.message_count || 0})`;
    options.push(opt);
  }
  select.replaceChildren(...options);
  select.value = sideState.sideId || "";
  select.disabled = sideState.pending;
  delBtn.hidden = !sideState.sideId;
  delBtn.disabled = sideState.pending;
}

function resetSideDraft(keepQuote = false) {
  sideState.sideId = null;
  sideState.messages = [];
  sideState.streamText = "";
  sideState.streamTools = [];
  if (!keepQuote) setSideQuote("");
  showSideError();
  renderSideBranches();
  renderSideConversation();
  updateSideComposer();
}

async function loadSideBranches(parentId = currentParentSessionId(), selectSideId = "") {
  const targetParent = parentId || "standalone";
  const seq = ++sideBranchReqSeq;
  const switchedParent = sideState.parentId !== targetParent;
  sideState.parentId = targetParent;
  if (switchedParent && !selectSideId) {
    resetSideDraft(false);
  }
  try {
    const branches = await request(`${sessionPath(targetParent)}/side-chats`);
    if (seq !== sideBranchReqSeq) return;
    sideState.branches = Array.isArray(branches) ? branches : [];
    if (selectSideId) {
      sideState.sideId = selectSideId;
    } else if (sideState.sideId && !sideState.branches.some((b) => b.id === sideState.sideId)) {
      sideState.sideId = null;
      sideState.messages = [];
    }
    renderSideBranches();
  } catch {
    if (seq !== sideBranchReqSeq) return;
    sideState.branches = [];
    renderSideBranches();
  }
}

async function selectSideBranch(sideId) {
  if (!sideId) {
    resetSideDraft(false);
    elements["side-prompt"].focus();
    return;
  }
  showSideError();
  try {
    const detail = await request(`/api/side-chats/${encodeURIComponent(sideId)}`);
    sideState.sideId = detail.id;
    sideState.messages = Array.isArray(detail.messages) ? detail.messages : [];
    setSideQuote(detail.quote || "");
    renderSideBranches();
    renderSideConversation();
    updateSideComposer();
    elements["side-prompt"].focus();
  } catch (error) {
    showSideError(error.message || "无法加载该侧边分支。");
  }
}

async function deleteActiveSideBranch() {
  if (!sideState.sideId || sideState.pending) return;
  const targetId = sideState.sideId;
  showSideError();
  try {
    await request(`/api/side-chats/${encodeURIComponent(targetId)}`, { method: "DELETE" });
    sideState.branches = sideState.branches.filter((b) => b.id !== targetId);
    resetSideDraft(false);
  } catch (error) {
    showSideError(error.message || "删除侧边分支失败。");
  }
}

function setSidePanel(open) {
  sideState.open = Boolean(open);
  elements["side-panel"].hidden = !sideState.open;
  elements.workspace.dataset.sideOpen = String(sideState.open);
  elements["toggle-side"].setAttribute("aria-expanded", String(sideState.open));
  if (sideState.open) {
    void loadSideBranches(currentParentSessionId(), sideState.sideId || "");
    renderSideConversation();
    updateSideComposer();
    elements["side-prompt"].focus();
  }
}

function updateSideComposer() {
  const busy = sideState.pending;
  elements["side-prompt"].readOnly = busy;
  elements["side-send"].disabled = busy || !selectedModel() || !elements["side-prompt"].value.trim();
  elements["side-send"].textContent = busy ? "思考中" : "发送";
  elements["clear-side"].disabled = busy;
  elements["side-branch-select"].disabled = busy;
  elements["delete-side-branch"].disabled = busy;
}

function renderSideConversation() {
  const list = elements["side-conversation"];
  const hasAny = sideState.messages.length > 0 || sideState.pending;
  elements["side-empty"].hidden = hasAny;
  list.hidden = !hasAny;

  const rows = [];
  let tools = [];
  for (const msg of sideState.messages) {
    if (msg.role === "tool") {
      tools.push(msg);
      continue;
    }
    if (msg.intermediate) continue;
    if (tools.length) {
      rows.push(toolDetails(tools));
      tools = [];
    }
    if (msg.role === "user") {
      rows.push(node("article", "side-msg user", msg.content));
    } else if (msg.role === "assistant" && msg.content) {
      const box = node("article", "side-msg assistant");
      const reply = renderReply(msg.content);
      if (msg.model_name) reply.append(node("div", "reply-model", `模型：${msg.model_name}`));
      box.append(reply);
      rows.push(box);
    }
  }
  if (tools.length) rows.push(toolDetails(tools));
  if (sideState.pending) {
    if (sideState.streamTools.length) rows.push(toolDetails(sideState.streamTools));
    const box = node("article", "side-msg assistant");
    box.append(sideState.streamText ? renderReply(sideState.streamText) : node("div", "pending", "正在思考…"));
    rows.push(box);
  }
  list.replaceChildren(...rows);
  elements["side-viewport"].scrollTop = elements["side-viewport"].scrollHeight;
}

async function sendSideMessage(event) {
  event.preventDefault();
  const question = elements["side-prompt"].value.trim();
  const activeModel = selectedModel();
  if (!question || !activeModel || sideState.pending) return;

  const previousMessages = [...sideState.messages];
  const parentId = currentParentSessionId();
  sideState.parentId = parentId;
  sideState.messages.push({ role: "user", content: question });
  sideState.pending = true;
  sideState.streamText = "";
  sideState.streamTools = [];
  elements["side-prompt"].value = "";
  showSideError();
  renderSideBranches();
  renderSideConversation();
  updateSideComposer();

  let streamCycle = 0;
  let streamError = null;
  let doneData = null;
  try {
    await consumeSSE(
      "/api/side-chat/stream",
      {
        question,
        quote: sideState.quote,
        session_id: state.active?.id || null,
        side_id: sideState.sideId,
        provider: activeModel.provider,
        model: activeModel.model,
      },
      {
        session(data) {
          if (data?.side_id) sideState.sideId = data.side_id;
        },
        delta(data) {
          if (typeof data?.cycle === "number" && data.cycle !== streamCycle) {
            streamCycle = data.cycle;
            sideState.streamText = "";
          }
          if (typeof data?.content === "string" && data.content) {
            sideState.streamText += data.content;
            renderSideConversation();
          }
        },
        tool(data) {
          sideState.streamTools.push({
            role: "tool",
            tool_name: data?.tool_name || "工具",
            content: formatStreamToolContent(data || {}),
          });
          sideState.streamText = "";
          renderSideConversation();
        },
        done(data) {
          doneData = data;
        },
        error(data) {
          streamError = new Error(data?.error || "侧边追问失败，请重试。");
        },
      },
    );
    if (streamError) throw streamError;
    if (doneData) {
      if (doneData.side_id) sideState.sideId = doneData.side_id;
      if (Array.isArray(doneData.messages)) {
        sideState.messages = doneData.messages;
      } else {
        sideState.messages.push({
          role: "assistant",
          content: doneData.reply || sideState.streamText,
          model_name: doneData.model_name || "",
        });
      }
      await loadSideBranches(parentId, sideState.sideId || "");
    } else if (sideState.streamText) {
      sideState.messages.push({ role: "assistant", content: sideState.streamText, model_name: "" });
    }
  } catch (error) {
    sideState.messages = previousMessages;
    elements["side-prompt"].value = question;
    showSideError(error.message || "侧边追问连接中断，请重试。");
  } finally {
    sideState.pending = false;
    sideState.streamText = "";
    sideState.streamTools = [];
    renderSideBranches();
    renderSideConversation();
    updateSideComposer();
    elements["side-prompt"].focus();
  }
}

function getConversationSelection() {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed || sel.rangeCount === 0) return null;
  const text = sel.toString().trim();
  if (!text) return null;
  const range = sel.getRangeAt(0);
  const container = range.commonAncestorContainer;
  const host = container.nodeType === Node.TEXT_NODE ? container.parentElement : container;
  if (!host || !elements.conversation.contains(host)) return null;
  const rect = range.getBoundingClientRect();
  if (!rect || (rect.width === 0 && rect.height === 0)) return null;
  return { text, rect };
}

function hideSelectionToolbar() {
  activeSelectionText = "";
  elements["selection-toolbar"].hidden = true;
}

function updateSelectionToolbar() {
  const found = getConversationSelection();
  if (!found) {
    hideSelectionToolbar();
    return;
  }
  activeSelectionText = found.text;
  const bar = elements["selection-toolbar"];
  bar.hidden = false;
  const barRect = bar.getBoundingClientRect();
  const centerX = found.rect.left + found.rect.width / 2;
  const left = Math.max(12, Math.min(window.innerWidth - barRect.width - 12, centerX - barRect.width / 2));
  let top = found.rect.bottom + 8;
  if (top + barRect.height + 12 > window.innerHeight) {
    top = Math.max(12, found.rect.top - barRect.height - 8);
  }
  bar.style.left = `${Math.round(left)}px`;
  bar.style.top = `${Math.round(top)}px`;
}

function triggerQuoteFromSelection(explicitText = "") {
  const text = (explicitText || activeSelectionText || getConversationSelection()?.text || "").trim();
  if (!text) return false;
  setMainQuote(text);
  hideSelectionToolbar();
  window.getSelection()?.removeAllRanges();
  elements.prompt.focus();
  return true;
}

function triggerSideFromSelection(explicitText = "") {
  const text = (explicitText || activeSelectionText || getConversationSelection()?.text || "").trim();
  hideSelectionToolbar();
  window.getSelection()?.removeAllRanges();
  if (text) {
    if (sideState.quote !== text && !sideState.pending) {
      resetSideDraft(false);
    }
    setSideQuote(text);
    setSidePanel(true);
    return true;
  }
  setSidePanel(!sideState.open);
  return true;
}

elements.composer.addEventListener("submit", sendMessage);
elements.prompt.addEventListener("input", updateComposer);
elements.prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
    event.preventDefault();
    if (!elements.send.disabled) elements.composer.requestSubmit();
  }
});
for (const name of ["provider", "model"]) {
  elements[name + "-trigger"].addEventListener("click", () => setPickerPanel(openPanel === name ? "" : name));
}
elements["model-picker"].addEventListener("focusout", (event) => {
  if (event.relatedTarget && !elements["model-picker"].contains(event.relatedTarget)) setPickerPanel();
});
elements["model-picker"].addEventListener("keydown", (event) => {
  if (event.key === "Escape" && openPanel) {
    event.preventDefault();
    setPickerPanel("", openPanel);
  }
});
document.addEventListener("pointerdown", (event) => {
  if (openPanel && !elements["model-picker"].contains(event.target)) setPickerPanel();
  if (!elements["context-usage"].contains(event.target)) setContextPopover(false);
  if (!elements["selection-toolbar"].contains(event.target) && !elements.conversation.contains(event.target)) {
    hideSelectionToolbar();
  }
});
elements["selection-toolbar"].addEventListener("mousedown", (event) => event.preventDefault());
elements["selection-quote"].addEventListener("click", () => triggerQuoteFromSelection());
elements["selection-side"].addEventListener("click", () => triggerSideFromSelection());
elements["clear-quote"].addEventListener("click", () => {
  setMainQuote("");
  elements.prompt.focus();
});
elements["toggle-side"].addEventListener("click", () => setSidePanel(!sideState.open));
elements["close-side"].addEventListener("click", () => setSidePanel(false));
elements["clear-side-quote"].addEventListener("click", () => setSideQuote(""));
elements["clear-side"].addEventListener("click", () => {
  resetSideDraft(false);
  elements["side-prompt"].focus();
});
elements["side-branch-select"].addEventListener("change", (event) => {
  void selectSideBranch(event.target.value);
});
elements["delete-side-branch"].addEventListener("click", () => {
  void deleteActiveSideBranch();
});
elements["side-form"].addEventListener("submit", sendSideMessage);
elements["side-prompt"].addEventListener("input", updateSideComposer);
elements["side-prompt"].addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
    event.preventDefault();
    if (!elements["side-send"].disabled) elements["side-form"].requestSubmit();
  }
});
elements.conversation.addEventListener("mouseup", () => queueMicrotask(updateSelectionToolbar));
elements.conversation.addEventListener("keyup", () => queueMicrotask(updateSelectionToolbar));
elements.viewport.addEventListener("scroll", hideSelectionToolbar, { passive: true });
document.addEventListener("selectionchange", () => {
  if (!getConversationSelection()) hideSelectionToolbar();
});
elements["context-trigger"].addEventListener("click", () => {
  const open = elements["context-popover"].hidden;
  if (open) setPickerPanel();
  setContextPopover(open);
  if (open && contextUsage.failed) void refreshContextUsage(true);
});
elements["open-add-model"].addEventListener("click", openAddModel);

/* ---- Customizations Token 预算监控与明细面板 ---- */
const customizationCategoryColors = {
  memory: "#7c5ce6",
  rules: "#f59e0b",
  skills: "#3b82f6",
  builtin_tools: "#10b981",
  mcp_tools: "#22c55e",
};
const customizationStatusLabels = {
  included: "已注入",
  demoted_to_pointer: "路径指针",
  excluded: "已排除",
};
const expandedCustomizationCategories = new Set();

function formatPercent(ratio) {
  const pct = (Number(ratio) || 0) * 100;
  if (pct > 0 && pct < 0.1) return "<0.1%";
  return `${pct.toFixed(1)}%`;
}

function renderCustomizationsReport(report) {
  const used = Number(report.used_tokens || 0);
  const budget = Math.max(1, Number(report.budget_tokens || 20000));
  const pctText = formatPercent(report.usage_ratio ?? used / budget);
  elements["customizations-total"].textContent = `${used.toLocaleString()} / ${budget.toLocaleString()} tokens (${pctText})`;

  const bar = elements["customizations-bar"];
  bar.setAttribute("aria-valuenow", String(Math.min(100, Math.round((used / budget) * 100))));
  const segments = [];
  for (const cat of report.categories || []) {
    if (!cat.tokens) continue;
    const seg = node("span", "custom-progress-seg");
    seg.style.width = `${Math.min(100, Math.max(0.6, (cat.tokens / budget) * 100))}%`;
    seg.style.background = customizationCategoryColors[cat.category] || "#5d43b6";
    seg.title = `${cat.label}: ${cat.tokens.toLocaleString()} tokens (${formatPercent(cat.ratio)})`;
    segments.push(seg);
  }
  bar.replaceChildren(...segments);

  const alerts = elements["customizations-alerts"];
  const alertPills = [];
  if (report.demoted_count > 0) {
    alertPills.push(node("span", "custom-alert-pill demoted", `已降级为路径指针：${report.demoted_count} 项`));
  }
  if (report.excluded_count > 0) {
    alertPills.push(node("span", "custom-alert-pill excluded", `超预算自动排除：${report.excluded_count} 项`));
  }
  alerts.hidden = alertPills.length === 0;
  alerts.replaceChildren(...alertPills);

  const container = elements["customizations-categories"];
  const blocks = (report.categories || []).map((cat) => {
    const block = node("div", "custom-cat-block");
    const row = node("div", "custom-cat-row");
    const meta = node("div", "custom-cat-meta");
    const dot = node("span", "custom-cat-dot");
    dot.style.background = customizationCategoryColors[cat.category] || "#5d43b6";
    meta.append(
      dot,
      node("span", "custom-cat-name", cat.label),
      node("span", "custom-cat-stats", `(${Number(cat.tokens || 0).toLocaleString()} tokens) ${formatPercent(cat.ratio)}`),
    );

    const toggle = node("button", "custom-toggle-btn");
    toggle.type = "button";
    const isExpanded = expandedCustomizationCategories.has(cat.category);
    if (!cat.items || cat.items.length === 0) {
      toggle.textContent = "暂无项目";
      toggle.disabled = true;
    } else {
      const noun = cat.items.length === 1 ? "1 项" : `${cat.items.length} 项`;
      toggle.textContent = isExpanded ? `收起明细（${noun}）` : `展开明细（${noun}）`;
      toggle.setAttribute("aria-expanded", String(isExpanded));
      toggle.addEventListener("click", () => {
        if (expandedCustomizationCategories.has(cat.category)) {
          expandedCustomizationCategories.delete(cat.category);
        } else {
          expandedCustomizationCategories.add(cat.category);
        }
        renderCustomizationsReport(report);
      });
    }
    row.append(meta, toggle);
    block.append(row);

    if (isExpanded && cat.items && cat.items.length > 0) {
      const list = node("div", "custom-breakdown-list");
      for (const item of cat.items) {
        const itemEl = node("div", "custom-item-row");
        const top = node("div", "custom-item-top");
        const right = node("div", "custom-item-right");
        const badge = node(
          "span",
          `custom-status-badge ${item.status}`,
          customizationStatusLabels[item.status] || item.status,
        );
        const tokenLabel = item.status === "included"
          ? `${item.tokens.toLocaleString()} tokens (${formatPercent(item.ratio)})`
          : `原 ${item.original_tokens.toLocaleString()} → ${item.tokens.toLocaleString()} tokens`;
        right.append(badge, node("span", "", tokenLabel));
        top.append(node("span", "custom-item-name", item.name), right);
        itemEl.append(top);

        const subParts = [];
        if (item.source_path) subParts.push(item.source_path);
        if (item.reason) subParts.push(item.reason);
        else if (item.description) subParts.push(item.description);
        if (subParts.length) {
          itemEl.append(node("div", "custom-item-sub", subParts.join(" · ")));
        }
        list.append(itemEl);
      }
      block.append(list);
    }
    return block;
  });
  container.replaceChildren(...blocks);
}

async function loadCustomizationsUsage() {
  elements["customizations-error"].hidden = true;
  elements["customizations-total"].textContent = "正在计算…";
  try {
    const report = await request("/api/customizations/usage");
    // 若有降级或排除项，自动展开对应分类，避免静默忽略
    for (const cat of report.categories || []) {
      if (cat.demoted_count > 0 || cat.excluded_count > 0) {
        expandedCustomizationCategories.add(cat.category);
      }
    }
    renderCustomizationsReport(report);
  } catch (error) {
    elements["customizations-total"].textContent = "读取失败";
    elements["customizations-error"].textContent = error.message || "无法获取 Customizations 用量。";
    elements["customizations-error"].hidden = false;
  }
}

elements["open-customizations"].addEventListener("click", () => {
  setPickerPanel();
  setContextPopover(false);
  elements["customizations-dialog"].showModal();
  void loadCustomizationsUsage();
});
elements["refresh-customizations"].addEventListener("click", () => void loadCustomizationsUsage());
elements["close-customizations"].addEventListener("click", () => elements["customizations-dialog"].close());
elements["model-provider"].addEventListener("change", updateModelForm);
elements["model-form"].addEventListener("submit", addModel);
elements["cancel-model"].addEventListener("click", () => elements["model-dialog"].close());
elements["delete-form"].addEventListener("submit", deleteModel);
elements["cancel-delete"].addEventListener("click", () => {
  modelToDelete = null;
  elements["delete-dialog"].close();
});
for (const id of ["model-dialog", "delete-dialog"]) {
  const dialog = elements[id];
  dialog.addEventListener("cancel", (event) => {
    if (modelMutationPending) event.preventDefault();
  });
  dialog.addEventListener("close", () => {
    if (dialog.open) return;
    if (id === "model-dialog") elements["custom-api-key"].value = "";
    if (!openPanel && !elements["model-dialog"].open && !elements["delete-dialog"].open) {
      elements["model-trigger"].focus();
    }
  });
}
elements["new-chat"].addEventListener("click", newChat);
elements["open-sidebar"].addEventListener("click", () => setSidebar(true));
elements["close-sidebar"].addEventListener("click", () => setSidebar(false, true));
elements.backdrop.addEventListener("click", () => setSidebar(false, true));
document.addEventListener("keydown", (event) => {
  const mod = event.ctrlKey || event.metaKey;
  if (mod && !event.shiftKey && !event.altKey) {
    const key = event.key.toLowerCase();
    if (key === "l") {
      const sel = activeSelectionText || getConversationSelection()?.text || "";
      if (sel) {
        event.preventDefault();
        triggerQuoteFromSelection(sel);
      }
      return;
    }
    if (key === "e") {
      event.preventDefault();
      triggerSideFromSelection();
      return;
    }
  }
  if (event.key !== "Escape") return;
  if (!elements["selection-toolbar"].hidden) hideSelectionToolbar();
  if (!elements["context-popover"].hidden) setContextPopover(false, true);
  if (elements.sidebar.dataset.open === "true") setSidebar(false, true);
});
mobile.addEventListener("change", () => setSidebar(false));

async function loadHistory() {
  try {
    const sessions = await request("/api/sessions");
    const known = new Set(state.sessions.map(({ id }) => id));
    state.sessions.push(...sessions.filter(({ id }) => !known.has(id)));
  } catch {
    state.historyError = "无法加载会话，刷新后重试。";
  }
  renderHistory();
}

async function loadStatus() {
  try {
    const data = await request("/api/status");
    modelOffline = data.models.some((item) => item.provider === "offline");
    setCatalog(data.models);
  } catch {
    catalogReady = true;
    catalogError = "模型列表加载失败";
    renderModelPicker();
  }
}

let availableId = "";
try {
  availableId = decodeURIComponent(window.location.hash.slice(1));
} catch {
  rememberSession();
}
if (availableId) selectSession(availableId, false);
renderModelPicker();
loadHistory();
loadStatus();
