#include "json.h"

namespace {
struct Parser {
    const char* p;
    const char* end;

    void ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    }
    [[noreturn]] void fail(const char* msg) {
        fprintf(stderr, "json parse error: %s near offset %zd\n", msg, p - (end - (end - p)));
        exit(1);
    }

    JVal value() {
        ws();
        if (p >= end) fail("eof");
        char c = *p;
        if (c == '{') return object();
        if (c == '[') return array();
        if (c == '"') {
            JVal v;
            v.kind = JVal::STR;
            v.str = string();
            return v;
        }
        if (c == 't') {
            expect("true");
            JVal v;
            v.kind = JVal::BOOL;
            v.b = true;
            return v;
        }
        if (c == 'f') {
            expect("false");
            JVal v;
            v.kind = JVal::BOOL;
            v.b = false;
            return v;
        }
        if (c == 'n') {
            expect("null");
            return JVal{};
        }
        return number();
    }
    void expect(const char* lit) {
        for (const char* q = lit; *q; ++q, ++p)
            if (p >= end || *p != *q) fail("literal");
    }
    std::string string() {
        if (*p != '"') fail("string");
        p++;
        std::string out;
        while (p < end && *p != '"') {
            if (*p == '\\') {
                p++;
                if (p >= end) fail("escape");
                switch (*p) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'u': {
                        if (end - p < 4) fail("u-escape");
                        unsigned cp = 0;
                        for (int i = 1; i <= 4; i++) {
                            char h = p[i];
                            cp <<= 4;
                            if (h >= '0' && h <= '9') cp |= h - '0';
                            else if (h >= 'a' && h <= 'f') cp |= h - 'a' + 10;
                            else if (h >= 'A' && h <= 'F') cp |= h - 'A' + 10;
                            else fail("hex");
                        }
                        p += 4;
                        // surrogate pair
                        if (cp >= 0xD800 && cp <= 0xDBFF && end - p > 6 && p[1] == '\\' && p[2] == 'u') {
                            unsigned lo = 0;
                            for (int i = 3; i <= 6; i++) {
                                char h = p[i];
                                lo <<= 4;
                                if (h >= '0' && h <= '9') lo |= h - '0';
                                else if (h >= 'a' && h <= 'f') lo |= h - 'a' + 10;
                                else fail("hex");
                            }
                            if (lo >= 0xDC00 && lo <= 0xDFFF) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                p += 6;
                            }
                        }
                        // utf8 encode
                        if (cp < 0x80) out += (char)cp;
                        else if (cp < 0x800) {
                            out += (char)(0xC0 | (cp >> 6));
                            out += (char)(0x80 | (cp & 63));
                        } else if (cp < 0x10000) {
                            out += (char)(0xE0 | (cp >> 12));
                            out += (char)(0x80 | ((cp >> 6) & 63));
                            out += (char)(0x80 | (cp & 63));
                        } else {
                            out += (char)(0xF0 | (cp >> 18));
                            out += (char)(0x80 | ((cp >> 12) & 63));
                            out += (char)(0x80 | ((cp >> 6) & 63));
                            out += (char)(0x80 | (cp & 63));
                        }
                        break;
                    }
                    default: fail("escape");
                }
                p++;
            } else {
                out += *p++;
            }
        }
        if (p >= end) fail("unterminated string");
        p++;  // closing quote
        return out;
    }
    JVal number() {
        char* endp = nullptr;
        double d = strtod(p, &endp);
        if (endp == p) fail("number");
        p = endp;
        JVal v;
        v.kind = JVal::NUM;
        v.num = d;
        return v;
    }
    JVal object() {
        p++;  // {
        JVal v;
        v.kind = JVal::OBJ;
        ws();
        if (p < end && *p == '}') { p++; return v; }
        while (true) {
            ws();
            std::string key = string();
            ws();
            if (*p != ':') fail("colon");
            p++;
            v.obj[key] = value();
            ws();
            if (*p == ',') { p++; continue; }
            if (*p == '}') { p++; return v; }
            fail("object");
        }
    }
    JVal array() {
        p++;  // [
        JVal v;
        v.kind = JVal::ARR;
        ws();
        if (p < end && *p == ']') { p++; return v; }
        while (true) {
            v.arr.push_back(value());
            ws();
            if (*p == ',') { p++; continue; }
            if (*p == ']') { p++; return v; }
            fail("array");
        }
    }
};
}  // namespace

JVal json_parse(const char* text, size_t len) {
    Parser ps{text, text + len};
    JVal v = ps.value();
    return v;
}
