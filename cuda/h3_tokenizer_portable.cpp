// Portable form of the upstream tokenizer: NFC, Qwen pre-tokenization, byte BPE.
#include "vendor/json.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <unicode/normalizer2.h>
#include <unicode/uchar.h>
#include <unicode/utf8.h>
#include <unordered_map>
#include <vector>
extern "C" {
#include "h3_tokenizer.h"
}
using std::string;
using std::vector;
using Map = std::unordered_map<string, uint32_t>;
struct h3_tokenizer {
    Map vocab, ranks, added;
    vector<string> inverse, inverse_added;
    string bytes[256];
    int16_t decoder[324];
};
static void error_text(char *e, size_t n, const char *s) {
    if (e && n)
        snprintf(e, n, "%s", s);
}
static string utf8(char32_t c) {
    icu::UnicodeString u;
    u.append(UChar32(c));
    string s;
    u.toUTF8String(s);
    return s;
}
static std::u32string points(const string &s) {
    std::u32string out;
    int32_t i = 0, n = s.size();
    while (i < n) {
        UChar32 c;
        U8_NEXT(s.data(), i, n, c);
        if (c < 0)
            throw std::runtime_error("invalid UTF-8");
        out.push_back(c);
    }
    return out;
}
static bool letter(char32_t c) {
    auto t = u_charType(c);
    return t >= U_UPPERCASE_LETTER && t <= U_OTHER_LETTER;
}
static bool number(char32_t c) {
    auto t = u_charType(c);
    return t == U_DECIMAL_DIGIT_NUMBER || t == U_LETTER_NUMBER || t == U_OTHER_NUMBER;
}
static bool space(char32_t c) { return u_isUWhiteSpace(c) || (c >= 0x1c && c <= 0x1f); }
static string slice(const std::u32string &s, size_t a, size_t b) {
    string out;
    for (; a < b; a++)
        out += utf8(s[a]);
    return out;
}
static vector<string> pretokenize(const string &input) {
    points(input);
    UErrorCode status = U_ZERO_ERROR;
    auto *nfc = icu::Normalizer2::getNFCInstance(status);
    icu::UnicodeString norm;
    nfc->normalize(icu::UnicodeString::fromUTF8(input), norm, status);
    if (U_FAILURE(status))
        throw std::runtime_error("NFC normalization failed");
    string normalized;
    norm.toUTF8String(normalized);
    auto s = points(normalized);
    vector<string> out;
    size_t i = 0, n = s.size();
    while (i < n) {
        size_t contraction = 0;
        if (s[i] == '\'')
            for (const string word : {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"}) {
                if (i + word.size() > n)
                    continue;
                bool match = true;
                for (size_t j = 1; j < word.size(); j++) {
                    char32_t c = s[i + j];
                    if (c >= 'A' && c <= 'Z')
                        c += 'a' - 'A';
                    if (c != char32_t(word[j]))
                        match = false;
                }
                if (match) {
                    contraction = word.size();
                    break;
                }
            }
        if (contraction) {
            out.push_back(slice(s, i, i + contraction));
            i += contraction;
            continue;
        }
        size_t begin = i;
        if (!letter(s[i]) && s[i] != '\r' && s[i] != '\n' && !number(s[i]) && i + 1 < n &&
            letter(s[i + 1]))
            begin++;
        if (letter(s[begin])) {
            size_t end = begin;
            while (end < n && letter(s[end]))
                end++;
            out.push_back(slice(s, i, end));
            i = end;
            continue;
        }
        if (number(s[i])) {
            out.push_back(slice(s, i, i + 1));
            i++;
            continue;
        }
        begin = i + (s[i] == ' ' && i + 1 < n && !space(s[i + 1]) && !letter(s[i + 1]) &&
                     !number(s[i + 1]));
        size_t end = begin;
        while (end < n && !space(s[end]) && !letter(s[end]) && !number(s[end]))
            end++;
        if (end > begin) {
            while (end < n && (s[end] == '\r' || s[end] == '\n'))
                end++;
            out.push_back(slice(s, i, end));
            i = end;
            continue;
        }
        if (space(s[i])) {
            end = i + 1;
            while (end < n && space(s[end]))
                end++;
            size_t newline = 0;
            for (size_t j = i; j < end; j++)
                if (s[j] == '\r' || s[j] == '\n')
                    newline = j + 1;
            size_t stop = newline ? newline : end == n ? end : end - i > 1 ? end - 1 : i + 1;
            out.push_back(slice(s, i, stop));
            i = stop;
            continue;
        }
        throw std::runtime_error("unable to pre-tokenize input");
    }
    return out;
}
static string pairkey(const string &a, const string &b) { return a + "\xef\xbf\xbf" + b; }
static void encode_plain(const h3_tokenizer &t, const string &text, vector<uint32_t> &out) {
    for (const auto &piece : pretokenize(text)) {
        vector<string> s;
        for (unsigned char c : piece)
            s.push_back(t.bytes[c]);
        while (s.size() > 1) {
            size_t best = s.size();
            uint32_t rank = UINT32_MAX;
            for (size_t i = 0; i + 1 < s.size(); i++) {
                auto p = t.ranks.find(pairkey(s[i], s[i + 1]));
                if (p != t.ranks.end() && p->second < rank) {
                    best = i;
                    rank = p->second;
                }
            }
            if (best == s.size())
                break;
            string a = s[best], b = s[best + 1];
            vector<string> m;
            for (size_t i = 0; i < s.size();) {
                if (i + 1 < s.size() && s[i] == a && s[i + 1] == b) {
                    m.push_back(a + b);
                    i += 2;
                } else
                    m.push_back(s[i++]);
            }
            s = std::move(m);
        }
        for (auto &v : s) {
            auto p = t.vocab.find(v);
            if (p == t.vocab.end())
                throw std::runtime_error("BPE token absent from vocabulary");
            out.push_back(p->second);
        }
    }
}
extern "C" h3_tokenizer *h3_tokenizer_load(const char *path, char *err, size_t size) {
    try {
        if (!path)
            throw std::runtime_error("tokenizer path required");
        std::ifstream in(path);
        auto j = nlohmann::json::parse(in);
        auto &m = j.at("model");
        if (m.at("type") != "BPE" || !m.at("unk_token").is_null() ||
            j.at("normalizer").at("type") != "NFC")
            throw std::runtime_error("unsupported tokenizer specification");
        auto t = std::make_unique<h3_tokenizer>();
        uint32_t maximum = 0;
        for (auto it = m.at("vocab").begin(); it != m.at("vocab").end(); ++it) {
            uint32_t id = it.value();
            t->vocab[it.key()] = id;
            maximum = std::max(maximum, id);
        }
        for (auto &v : j.at("added_tokens")) {
            for (const char *k : {"single_word", "lstrip", "rstrip", "normalized"})
                if (v.value(k, false))
                    throw std::runtime_error("unsupported added-token policy");
            uint32_t id = v.at("id");
            t->added[v.at("content")] = id;
            maximum = std::max(maximum, id);
        }
        t->inverse.resize(size_t(maximum) + 1);
        t->inverse_added.resize(size_t(maximum) + 1);
        for (auto &v : t->vocab)
            t->inverse[v.second] = v.first;
        for (auto &v : t->added)
            t->inverse_added[v.second] = v.first;
        uint32_t rank = 0;
        for (auto &v : m.at("merges")) {
            string a, b;
            if (v.is_array() && v.size() == 2) {
                a = v[0];
                b = v[1];
            } else if (v.is_string()) {
                string p = v;
                auto pos = p.find(' ');
                if (pos == string::npos)
                    throw std::runtime_error("invalid BPE merge");
                a = p.substr(0, pos);
                b = p.substr(pos + 1);
            } else
                throw std::runtime_error("invalid BPE merge");
            t->ranks[pairkey(a, b)] = rank++;
        }
        std::fill(std::begin(t->decoder), std::end(t->decoder), -1);
        unsigned extra = 0;
        for (unsigned b = 0; b < 256; b++) {
            bool visible = (b >= 33 && b <= 126) || (b >= 0xa1 && b <= 0xac) || (b >= 0xae);
            unsigned cp = visible ? b : 256 + extra++;
            t->bytes[b] = utf8(cp);
            t->decoder[cp] = b;
        }
        if (err && size)
            err[0] = 0;
        return t.release();
    } catch (const std::exception &e) {
        error_text(err, size, e.what());
        return nullptr;
    }
}
extern "C" void h3_tokenizer_free(h3_tokenizer *t) { delete t; }
extern "C" void h3_tokenizer_ids_free(uint32_t *p) { free(p); }
extern "C" int h3_tokenizer_encode(const h3_tokenizer *t, const char *text, int pad, uint32_t **ids,
                                   size_t *count, char *err, size_t size) {
    try {
        if (!t || !text || !ids || !count)
            throw std::runtime_error("invalid tokenizer arguments");
        *ids = nullptr;
        *count = 0;
        string s = text;
        points(s);
        vector<uint32_t> out;
        size_t start = 0;
        while (start < s.size()) {
            size_t best = string::npos;
            const string *token = nullptr;
            uint32_t id = 0;
            for (auto &v : t->added) {
                size_t p = s.find(v.first, start);
                if (p != string::npos &&
                    (p < best || (p == best && v.first.size() > token->size()))) {
                    best = p;
                    token = &v.first;
                    id = v.second;
                }
            }
            if (!token)
                break;
            if (best > start)
                encode_plain(*t, s.substr(start, best - start), out);
            out.push_back(id);
            start = best + token->size();
        }
        if (start < s.size())
            encode_plain(*t, s.substr(start), out);
        if (out.empty() && pad)
            out.push_back(H3_PAD_TOKEN_ID);
        if (!out.empty()) {
            *ids = (uint32_t *)malloc(out.size() * sizeof(uint32_t));
            if (!*ids)
                throw std::bad_alloc();
            memcpy(*ids, out.data(), out.size() * sizeof(uint32_t));
        }
        *count = out.size();
        if (err && size)
            err[0] = 0;
        return 1;
    } catch (const std::exception &e) {
        error_text(err, size, e.what());
        return 0;
    }
}
extern "C" char *h3_tokenizer_decode(const h3_tokenizer *t, const uint32_t *ids, size_t count,
                                     char *err, size_t size) {
    try {
        if (!t || (!ids && count))
            throw std::runtime_error("invalid tokenizer arguments");
        string result, bytes;
        auto flush = [&]() {
            if (bytes.empty())
                return;
            try {
                points(bytes);
                result += bytes;
            } catch (...) {
                result += "\xef\xbf\xbd";
            }
            bytes.clear();
        };
        for (size_t i = 0; i < count; i++) {
            auto id = ids[i];
            if (id >= t->inverse.size())
                throw std::runtime_error("token ID out of range");
            if (!t->inverse_added[id].empty()) {
                flush();
                result += t->inverse_added[id];
                continue;
            }
            if (t->inverse[id].empty())
                throw std::runtime_error("unknown token ID");
            for (auto cp : points(t->inverse[id])) {
                if (cp >= 324 || t->decoder[cp] < 0)
                    throw std::runtime_error("invalid byte token");
                bytes += char(t->decoder[cp]);
            }
        }
        flush();
        if (err && size)
            err[0] = 0;
        return strdup(result.c_str());
    } catch (const std::exception &e) {
        error_text(err, size, e.what());
        return nullptr;
    }
}
