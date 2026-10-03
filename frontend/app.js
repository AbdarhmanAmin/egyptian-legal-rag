const form = document.querySelector("#ask-form");
const input = document.querySelector("#question");
const sendButton = document.querySelector("#send-button");
const messages = document.querySelector("#messages");
const welcome = document.querySelector("#welcome");
const charCount = document.querySelector("#char-count");
const statusDot = document.querySelector("#status-dot");
const statusText = document.querySelector("#status-text");
const languageToggle = document.querySelector("#language-toggle");

const copy = {
  en: {
    newChat: "New conversation",
    sidebarNote: "Grounded in the Egyptian Civil Code, with article references to help you verify each answer.",
    privacy: "Your questions are sent to the configured answer provider to generate a response.",
    scope: "EGYPTIAN CIVIL CODE",
    apiDocs: "API docs",
    eyebrow: "CITED ANSWERS FROM THE CIVIL CODE",
    headline: 'Clarity begins<br />with the <em>right question.</em>',
    intro: "Explore the Egyptian Civil Code in plain language. Ask in Arabic or English and get an answer grounded in cited articles.",
    suggestionLabel: "A FEW PLACES TO START",
    suggestions: ["What makes a contract legally binding?", "ما أثر العقد الصحيح بين الطرفين؟", "When can an obligation be terminated?"],
    questionLabel: "Ask a question about the Egyptian Civil Code",
    placeholder: "Ask about the Egyptian Civil Code…",
    inputHint: "Arabic or English <span>·</span> Enter to send",
    send: "Send question",
    disclaimer: "Mizan provides legal information, not legal advice. Verify the cited articles for your situation.",
    connecting: "Connecting to your local library…",
    localAvailable: "Local library unavailable",
    articlesReady: (count) => `${count.toLocaleString("en-US")} articles ready`,
    answer: "ANSWER",
    references: "REFERENCES",
    preparing: "Preparing answer",
    error: "Sorry, I could not get an answer. Please check that the API and local index are running.",
    errorDetail: (detail) => ` ${detail}`,
  },
  ar: {
    newChat: "محادثة جديدة",
    sidebarNote: "إجابات مستندة إلى القانون المدني المصري، مع أرقام المواد لتسهيل التحقق.",
    privacy: "تُرسل أسئلتك إلى مزود الإجابات المُعدّ لإنشاء الرد.",
    scope: "القانون المدني المصري",
    apiDocs: "توثيق API",
    eyebrow: "إجابات موثّقة من مواد القانون المدني",
    headline: 'الوضوح يبدأ<br />من <em>السؤال الصحيح.</em>',
    intro: "استكشف القانون المدني المصري بلغة واضحة. اسأل بالعربية أو الإنجليزية واحصل على إجابة مدعومة بأرقام المواد.",
    suggestionLabel: "أسئلة للبدء",
    suggestions: ["ما شروط صحة العقد؟", "ما أثر العقد الصحيح بين الطرفين؟", "متى ينقضي الالتزام؟"],
    questionLabel: "اسأل عن القانون المدني المصري",
    placeholder: "اكتب سؤالك عن القانون المدني المصري…",
    inputHint: "العربية أو الإنجليزية <span>·</span> اضغط Enter للإرسال",
    send: "إرسال السؤال",
    disclaimer: "ميزان يقدم معلومات قانونية ولا يقدم استشارة قانونية. تحقق من المواد المذكورة بما يناسب حالتك.",
    connecting: "جارٍ الاتصال بالمكتبة المحلية…",
    localAvailable: "المكتبة المحلية غير متاحة",
    articlesReady: (count) => `${count.toLocaleString("ar-EG")} مادة جاهزة`,
    answer: "الإجابة",
    references: "المراجع",
    preparing: "جارٍ إعداد الإجابة",
    error: "عذرًا، تعذر الحصول على إجابة. تحقق من تشغيل الواجهة البرمجية وفهرس المستندات المحلي.",
    errorDetail: (detail) => ` ${detail}`,
  },
};

function getInitialLanguage() {
  const saved = localStorage.getItem("mizan-language");
  if (saved === "ar" || saved === "en") return saved;
  return navigator.language?.toLowerCase().startsWith("ar") ? "ar" : "en";
}

let language = getInitialLanguage();

function setLanguage(nextLanguage) {
  language = nextLanguage;
  localStorage.setItem("mizan-language", language);
  const isArabic = language === "ar";
  const text = copy[language];
  document.documentElement.lang = language;
  document.documentElement.dir = isArabic ? "rtl" : "ltr";
  document.querySelectorAll("[data-i18n]").forEach((element) => {
    const key = element.dataset.i18n;
    if (text[key] !== undefined) element.innerHTML = text[key];
  });
  document.querySelectorAll("[data-suggestion]").forEach((element) => {
    element.textContent = text.suggestions[Number(element.dataset.suggestion)];
  });
  input.placeholder = text.placeholder;
  sendButton.setAttribute("aria-label", text.send);
  languageToggle.textContent = isArabic ? "English" : "العربية";
  languageToggle.setAttribute("aria-label", isArabic ? "Switch to English" : "التبديل إلى العربية");
  statusText.textContent = text.connecting;
  updateHealth();
}

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
  charCount.textContent = `${input.value.length} / 2000`;
}

function addMessage(text, role) {
  const element = document.createElement("div");
  element.className = `message ${role}`;
  element.textContent = text;
  element.dir = "auto";
  messages.append(element);
  element.scrollIntoView({ behavior: "smooth", block: "nearest" });
  return element;
}

function addAnswer(result) {
  const wrapper = document.createElement("div");
  wrapper.className = "message assistant";
  wrapper.dir = "auto";

  const label = document.createElement("div");
  label.className = "assistant-label";
  const mark = document.createElement("span");
  mark.className = "brand-mark";
  mark.textContent = "م";
  label.append(mark, document.createTextNode(` MIZAN · ${copy[language].answer}`));

  const answer = document.createElement("div");
  answer.textContent = result.answer;
  wrapper.append(label, answer);

  if (result.sources?.length) {
    const sourceList = document.createElement("div");
    sourceList.className = "sources";
    const title = document.createElement("strong");
    title.textContent = copy[language].references;
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
    statusText.textContent = copy[language].articlesReady(health.documents_indexed);
  } catch {
    statusDot.className = "status-dot offline";
    statusText.textContent = copy[language].localAvailable;
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
  loading.setAttribute("aria-label", copy[language].preparing);
  for (let i = 0; i < 3; i += 1) loading.append(document.createElement("span"));
  messages.append(loading);

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "");
    addAnswer(payload);
  } catch (error) {
    addMessage(`${copy[language].error}${error.message ? copy[language].errorDetail(error.message) : ""}`, "assistant");
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
    input.value = button.querySelector("[data-suggestion]").textContent;
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

languageToggle.addEventListener("click", () => setLanguage(language === "ar" ? "en" : "ar"));
resizeInput();
setLanguage(language);
