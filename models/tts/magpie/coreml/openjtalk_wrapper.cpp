/**
 * OpenJTalk G2P Wrapper Implementation
 *
 * Runs the OpenJTalk frontend pipeline (text2mecab → MeCab → NJD processing)
 * and returns word features as JSON for the Swift-side pitch accent calculator.
 *
 * Based on pyopenjtalk's openjtalk.pyx implementation.
 */

#include "openjtalk_wrapper.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>

// OpenJTalk headers — mecab.h contains C++ code (templates), so it must
// be included outside extern "C". The other headers are pure C.
#include "mecab.h"

extern "C" {
#include "njd.h"
#include "jpcommon.h"
#include "text2mecab.h"
#include "mecab2njd.h"
#include "njd2jpcommon.h"
#include "njd_set_pronunciation.h"
#include "njd_set_digit.h"
#include "njd_set_accent_phrase.h"
#include "njd_set_accent_type.h"
#include "njd_set_unvoiced_vowel.h"
#include "njd_set_long_vowel.h"
}

// Buffer size for text2mecab (matches pyopenjtalk)
#define TEXT2MECAB_BUFFER_SIZE 8192

// Global state (matches pyopenjtalk's singleton pattern)
static Mecab g_mecab;
static NJD g_njd;
static JPCommon g_jpcommon;
static int g_initialized = 0;

// JSON string builder helpers
static void json_append(char** buf, size_t* len, size_t* cap, const char* str) {
    size_t slen = strlen(str);
    while (*len + slen + 1 > *cap) {
        *cap *= 2;
        *buf = (char*)realloc(*buf, *cap);
    }
    memcpy(*buf + *len, str, slen);
    *len += slen;
    (*buf)[*len] = '\0';
}

// Escape a string for JSON (handles ", \, control chars)
static void json_append_escaped(char** buf, size_t* len, size_t* cap, const char* str) {
    json_append(buf, len, cap, "\"");
    if (str) {
        for (const char* p = str; *p; p++) {
            switch (*p) {
                case '"':  json_append(buf, len, cap, "\\\""); break;
                case '\\': json_append(buf, len, cap, "\\\\"); break;
                case '\n': json_append(buf, len, cap, "\\n"); break;
                case '\r': json_append(buf, len, cap, "\\r"); break;
                case '\t': json_append(buf, len, cap, "\\t"); break;
                default:
                    if ((unsigned char)*p < 0x20) {
                        char esc[8];
                        snprintf(esc, sizeof(esc), "\\u%04x", (unsigned char)*p);
                        json_append(buf, len, cap, esc);
                    } else {
                        char c[2] = {*p, '\0'};
                        json_append(buf, len, cap, c);
                    }
                    break;
            }
        }
    }
    json_append(buf, len, cap, "\"");
}

int openjtalk_init(const char* dict_path) {
    if (g_initialized) {
        openjtalk_destroy();
    }

    Mecab_initialize(&g_mecab);
    NJD_initialize(&g_njd);
    JPCommon_initialize(&g_jpcommon);

    if (Mecab_load(&g_mecab, dict_path) != 1) {
        Mecab_clear(&g_mecab);
        NJD_clear(&g_njd);
        JPCommon_clear(&g_jpcommon);
        return 0;
    }

    g_initialized = 1;
    return 1;
}

char* openjtalk_g2p(const char* text) {
    if (!g_initialized || !text) {
        return NULL;
    }

    char mecab_buf[TEXT2MECAB_BUFFER_SIZE];

    // Step 1: Preprocess text for MeCab
    text2mecab(mecab_buf, text);

    // Step 2: Run MeCab morphological analysis
    if (Mecab_analysis(&g_mecab, mecab_buf) != 1) {
        return NULL;
    }

    // Step 3: Convert MeCab output to NJD
    mecab2njd(&g_njd, Mecab_get_feature(&g_mecab), Mecab_get_size(&g_mecab));

    // Step 4: Run NJD processing pipeline
    njd_set_pronunciation(&g_njd);
    njd_set_digit(&g_njd);
    njd_set_accent_phrase(&g_njd);
    njd_set_accent_type(&g_njd);
    njd_set_unvoiced_vowel(&g_njd);
    njd_set_long_vowel(&g_njd);

    // Step 5: Build JSON array from NJD nodes
    size_t cap = 1024;
    size_t len = 0;
    char* json = (char*)malloc(cap);
    json[0] = '\0';

    json_append(&json, &len, &cap, "[");

    int first = 1;
    NJDNode* node = g_njd.head;
    while (node != NULL) {
        if (!first) {
            json_append(&json, &len, &cap, ",");
        }
        first = 0;

        json_append(&json, &len, &cap, "{");

        // "string" field
        json_append(&json, &len, &cap, "\"string\":");
        json_append_escaped(&json, &len, &cap, NJDNode_get_string(node));

        // "pos" field
        json_append(&json, &len, &cap, ",\"pos\":");
        json_append_escaped(&json, &len, &cap, NJDNode_get_pos(node));

        // "pron" field
        json_append(&json, &len, &cap, ",\"pron\":");
        json_append_escaped(&json, &len, &cap, NJDNode_get_pron(node));

        // "acc" field
        char num_buf[32];
        json_append(&json, &len, &cap, ",\"acc\":");
        snprintf(num_buf, sizeof(num_buf), "%d", NJDNode_get_acc(node));
        json_append(&json, &len, &cap, num_buf);

        // "mora_size" field
        json_append(&json, &len, &cap, ",\"mora_size\":");
        snprintf(num_buf, sizeof(num_buf), "%d", NJDNode_get_mora_size(node));
        json_append(&json, &len, &cap, num_buf);

        // "chain_flag" field
        json_append(&json, &len, &cap, ",\"chain_flag\":");
        snprintf(num_buf, sizeof(num_buf), "%d", NJDNode_get_chain_flag(node));
        json_append(&json, &len, &cap, num_buf);

        json_append(&json, &len, &cap, "}");

        node = node->next;
    }

    json_append(&json, &len, &cap, "]");

    // Reset NJD and JPCommon for next call
    NJD_refresh(&g_njd);
    JPCommon_refresh(&g_jpcommon);
    Mecab_refresh(&g_mecab);

    return json;
}

void openjtalk_free_string(char* s) {
    if (s) {
        free(s);
    }
}

void openjtalk_destroy(void) {
    if (g_initialized) {
        Mecab_clear(&g_mecab);
        NJD_clear(&g_njd);
        JPCommon_clear(&g_jpcommon);
        g_initialized = 0;
    }
}
