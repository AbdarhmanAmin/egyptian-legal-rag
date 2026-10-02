const form = document.querySelector("#ask-form");
const input = document.querySelector("#question");
const sendButton = document.querySelector("#send-button");
const messages = document.querySelector("#messages");
const welcome = document.querySelector("#welcome");
const charCount = document.querySelector("#char-count");
const statusDot = document.querySelector("#status-dot");
const statusText = document.querySelector("#status-text");

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
  charCount.textContent = `${input.value.length} / 2000`;
}

function addMessage(text, role) {
  const element = document.createElement("div");
  element.className = `message ${role}`;
  element.textContent = text;
  messages.append(element);
  element.scrollIntoView({ behavior: "smooth", block: "nearest" });
  return element;
}

function addAnswer(result) {
  const wrapper = document.createElement("div");
  wrapper.className = "message assistant";

  const label = document.createElement("div");
  label.className = "assistant-label";
  const mark = document.createElement("span");
  mark.className = "brand-mark";
  mark.textContent = "م";
  label.append(mark, document.createTextNode(" MIZAN · ANSWER"));

  const answer = document.createElement("div");
  answer.textContent = result.answer;
  wrapper.append(label, answer);

  if (result.sources?.length) {
    const sourceList = document.createElement("div");
    sourceList.className = "sources";
    const title = document.createElement("strong");
    title.textContent = "REFERENCES";
    sourceList.append(title);
    result.sources.forEach((source) => {
      const chip = document.createElement("span");
      chip.className = "source-chip";
      chip.textContent = source;
      sourceList.append(chip);
    });
    wrapper.append(sourceList);
  }

  messages.append(wrapper);
  wrapper.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function updateHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("API unavailable");
    const health = await response.json();
    statusDot.className = "status-dot online";
    statusText.textContent = `${health.documents_indexed.toLocaleString()} articles ready`;
  } catch {
    statusDot.className = "status-dot offline";
    statusText.textContent = "Local library unavailable";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question || sendButton.disabled) return;

  welcome.hidden = true;
  addMessage(question, "user");
  input.value = "";
  resizeInput();
  sendButton.disabled = true;

  const loading = document.createElement("div");
  loading.className = "loading";
  loading.setAttribute("aria-label", "Preparing answer");
  for (let i = 0; i < 3; i += 1) loading.append(document.createElement("span"));
  messages.append(loading);

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "The question could not be answered.");
    addAnswer(payload);
  } catch (error) {
    addMessage(`Sorry, I could not get an answer. ${error.message} Please check that the API and local index are running.`, "assistant");
  } finally {
    loading.remove();
    sendButton.disabled = false;
    input.focus();
    updateHealth();
  }
});

input.addEventListener("input", resizeInput);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.querySelector("span:nth-child(2)").textContent;
    resizeInput();
    form.requestSubmit();
  });
});

document.querySelector("#new-chat").addEventListener("click", () => {
  messages.replaceChildren();
  welcome.hidden = false;
  input.value = "";
  resizeInput();
  input.focus();
});

resizeInput();
updateHealth();
