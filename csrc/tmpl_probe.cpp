// E41b leg 0 — the deciding tokenization, WITHOUT loading a model.
//
// Question: does the template render an assistant turn the SAME way when it is the
// last message (add_generation_prompt=false — what canon writes into KV) as when it
// is a past message followed by a user turn (what the next turn renders)? If not,
// the canonical KV can never be a prefix of the next render and canon's benefit is
// structurally zero, independent of any client-echo bug.
//
// Why a separate tool rather than reasoning about the Jinja: llama.cpp does NOT
// interpret the GGUF Jinja for known families — it dispatches to its own built-in
// handler. So the only ground truth is llama_chat_apply_template itself. That
// function takes the template STRING, not a model, so this probe reads the template
// from a file (extracted from GGUF metadata) and never touches the weights. It can
// therefore run while a measurement window is owned by another experiment.
//
//   tmpl_probe <template-file>
//
// Prints both renderings with special tokens made visible, plus a byte-level verdict.
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::string render(const char * tmpl,
                          const std::vector<llama_chat_message> & msgs,
                          bool add_gen) {
    std::vector<char> buf(1 << 16);
    int32_t r = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), add_gen,
                                          buf.data(), (int32_t) buf.size());
    if (r <= 0 || r > (int32_t) buf.size()) return std::string();
    return std::string(buf.data(), r);
}

// make control/special markers visible without altering byte content
static std::string vis(const std::string & s) {
    std::string o;
    for (char c : s) {
        if (c == '\n') o += "\\n";
        else o += c;
    }
    return o;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <template-file>\n", argv[0]); return 1; }
    FILE * f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    std::string tmpl;
    char chunk[4096]; size_t n;
    while ((n = fread(chunk, 1, sizeof chunk, f)) > 0) tmpl.append(chunk, n);
    fclose(f);
    printf("template: %zu chars\n\n", tmpl.size());

    const char * SYS  = "You are a helpful assistant.";
    const char * USER = "Explain how the Earth formed.";
    const char * ANS  = "ANSWERBODY";
    const char * USER2 = "Summarize that.";

    // (i) what canon writes: assistant LAST, no generation prompt
    std::vector<llama_chat_message> last = {
        {"system", SYS}, {"user", USER}, {"assistant", ANS}};
    std::string A = render(tmpl.c_str(), last, false);

    // (ii) what the next turn renders: the same assistant turn is now PAST
    std::vector<llama_chat_message> past = {
        {"system", SYS}, {"user", USER}, {"assistant", ANS}, {"user", USER2}};
    std::string B = render(tmpl.c_str(), past, true);

    if (A.empty() || B.empty()) {
        fprintf(stderr, "render failed (A=%zu B=%zu) — template not handled\n", A.size(), B.size());
        return 1;
    }

    printf("=== (i) assistant LAST, add_generation_prompt=false  [what canon writes] ===\n%s\n\n",
           vis(A).c_str());
    printf("=== (ii) assistant PAST + user turn, add_generation_prompt=true  [next render] ===\n%s\n\n",
           vis(B).c_str());

    // The decisive question: is (i) a byte-prefix of (ii)? That is exactly the
    // property canon relies on — the KV it builds must be a prefix of the next render.
    const bool is_prefix = B.compare(0, A.size(), A) == 0 && A.size() <= B.size();
    printf("=== VERDICT ===\n");
    printf("(i) is a byte-prefix of (ii): %s\n", is_prefix ? "YES" : "NO");
    if (!is_prefix) {
        size_t i = 0;
        while (i < A.size() && i < B.size() && A[i] == B[i]) i++;
        printf("first divergence at byte %zu\n", i);
        printf("  (i)  continues: |%s|\n", vis(A.substr(i, 60)).c_str());
        printf("  (ii) continues: |%s|\n", vis(B.substr(i, 60)).c_str());
        printf("\nOUTCOME (a): STRUCTURAL MISMATCH CONFIRMED — the canonical form is not a\n"
               "prefix of the next turn's render, so the KV match must break there no matter\n"
               "how correct the client echo is.\n");
    } else {
        printf("\nOUTCOME (b): MISMATCH CLEARED — the canonical form IS a prefix of the next\n"
               "render, so the harness echo bug was the whole story.\n");
    }
    // locate the assistant span in each, to show the marker/terminator shapes plainly
    auto span = [](const std::string & s, const char * body) {
        size_t p = s.find(body);
        if (p == std::string::npos) return std::string("<body not found>");
        size_t a = s.rfind("<|start|>", p);
        size_t e = s.find("<|", p + strlen(body));
        size_t end = s.find(">", e);
        return s.substr(a, (end == std::string::npos ? s.size() : end + 1) - a);
    };
    printf("\nassistant turn as rendered:\n  (i)  |%s|\n  (ii) |%s|\n",
           vis(span(A, ANS)).c_str(), vis(span(B, ANS)).c_str());
    return 0;
}
